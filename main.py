#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import asyncio
import signal
from exchange import connect_exchange, fetch_fees, send_telegram_message
from data import get_historical_data, prepare_lstm_data, prepare_gru_data, add_features
from model import train_lstm_model, train_gru_model
from strategy import trade_pair, select_profitable_pairs, finalize_report
from order_management import shutdown, check_and_cancel_orders
from limits import check_drawdown, check_daily_loss_limit, check_volatility
from config import TRADING_PAIRS, INITIAL_BALANCE, MIN_ORDER_SIZE
from globals import MAX_OPEN_ORDERS, daily_losses
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ],
    force=True
)

running = True


async def log_balances(balances):
    """Вывод текущего баланса в лог."""
    total_binance = max(sum(balance['quote_binance'] for balance in balances.values()), 0)
    balance_summary = {pair: {
        'base': balances[pair]['base'],
        'quote_binance': balances[pair]['quote_binance']
    } for pair in balances}
    logging.info(f"Текущий баланс: Total USDT: {total_binance:.2f}, Детали по парам: {balance_summary}")


async def final_report(exchanges, balances):
    """Итоговый отчёт о прибыли/убытках с учётом остатков активов."""
    total_initial = 7537.93
    total_final_usdt = sum(balance['quote_binance'] for balance in balances.values())
    total_fees = sum(balance['total_fees'] for balance in balances.values())
    total_assets_value = 0

    for pair in balances:
        if balances[pair]['base'] > 0:
            order_book = await exchanges['binance'].fetch_order_book(pair)
            ask_price = order_book['asks'][0][0]
            asset_value = balances[pair]['base'] * ask_price
            total_assets_value += asset_value
            logging.info(
                f"{pair}: Остатки {balances[pair]['base']:.4f} по цене {ask_price:.2f}, стоимость: {asset_value:.2f} USDT")

    total_final = total_final_usdt + total_assets_value
    profit_loss = total_final - total_initial - total_fees
    logging.info(
        f"Итоговый отчёт: Начальный баланс: {total_initial:.2f} USDT, Конечный баланс (USDT + активы): {total_final:.2f} USDT, Комиссии: {total_fees:.2f} USDT, Прибыль/Убыток: {profit_loss:.2f} USDT")
    for pair in balances:
        pl = balances[pair]['quote_binance'] + (
            balances[pair]['base'] * ask_price if balances[pair]['base'] > 0 else 0) - (
                         total_initial / len(TRADING_PAIRS)) - balances[pair]['total_fees']
        logging.info(f"{pair}: Прибыль/Убыток: {pl:.2f} USDT, Комиссии: {balances[pair]['total_fees']:.2f} USDT")


async def main():
    global running
    logging.info("Запуск скрипта")
    exchanges = {
        'binance': await connect_exchange('binance'),
    }

    try:
        fees = {
            'binance': {pair: {'maker': 0.001, 'taker': 0.001} for pair in TRADING_PAIRS} if 'testnet.binance.vision' in
                                                                                             exchanges['binance'].urls[
                                                                                                 'api'][
                                                                                                 'public'] else await fetch_fees(
                exchanges['binance']),
            'bingx': {pair: {'maker': 0.001, 'taker': 0.001} for pair in TRADING_PAIRS}
        }

        data = await get_historical_data(exchanges['binance'], 'ETH/USDT', limit=2000)
        data = add_features(data)
        X, y, scaler = prepare_lstm_data(data)
        logging.info(f"Данные для обучения LSTM: X.shape={X.shape}, y.mean={y.mean():.4f}")
        pred_model = train_lstm_model(X, y)

        loss_data = await get_historical_data(exchanges['binance'], 'ETH/USDT', limit=2000)
        loss_data = add_features(loss_data)
        X_loss, y_loss, loss_scaler = prepare_gru_data(loss_data)
        logging.info(f"Обучение GRU-модели с X_loss.shape={X_loss.shape}, y_loss.shape={y_loss.shape}")
        loss_model = train_gru_model(X_loss, y_loss)

        balances = {}
        open_orders = defaultdict(list)
        total_initial_balance = 7537.93
        initial_allocation = total_initial_balance / len(TRADING_PAIRS)
        for pair in TRADING_PAIRS:
            balances[pair] = {
                'base': 0.0,
                'quote_binance': initial_allocation,
                'quote_bingx': 100.0,
                'entry_price': None,
                'total_fees': 0.0
            }

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(exchanges, balances, open_orders)))

        iteration = 0
        total_iterations = 10
        while running and iteration < total_iterations:
            try:
                logging.info(f"Начало итерации {iteration + 1}")
                await log_balances(balances)
                total_binance = sum(balance['quote_binance'] for balance in balances.values())
                if total_binance < MIN_ORDER_SIZE:
                    logging.warning(
                        f"Общий баланс USDT ({total_binance:.2f}) ниже минимального ({MIN_ORDER_SIZE}). Остановка торговли.")
                    break

                can_trade_drawdown, reason_drawdown = check_drawdown(balances)
                if not can_trade_drawdown:
                    print(f"Торговля остановлена: {reason_drawdown}")
                    await send_telegram_message(f"Торговля остановлена: {reason_drawdown}")
                    break

                # Синхронизация открытых ордеров перед итерацией
                for pair in TRADING_PAIRS:
                    await check_and_cancel_orders(exchanges['binance'], pair, balances, 0, open_orders)

                # Динамический процент: от 5% до 1%
                trade_fraction = 0.05 - (0.04 * iteration / (total_iterations - 1))

                logging.info("Вызов select_profitable_pairs")
                profitable_pairs = await select_profitable_pairs(exchanges, fees, pred_model, scaler, balances)
                logging.info(f"Выбраны пары: {[pair[0] for pair in profitable_pairs]} с лимитом {MAX_OPEN_ORDERS}")
                await send_telegram_message(
                    f"Выбраны пары для торговли: {[pair[0] for pair in profitable_pairs]} с лимитом {MAX_OPEN_ORDERS}")

                tasks = []
                for pair_data in profitable_pairs:
                    historical_data = await get_historical_data(exchanges['binance'], pair_data[0], limit=100)
                    historical_data = add_features(historical_data)
                    if historical_data.empty:
                        logging.warning(f"{pair_data[0]}: Нет достаточно данных для расчёта индикаторов, пропускаем")
                        continue
                    atr = historical_data.iloc[-1]['ATR']
                    can_trade_loss, reason_loss = await check_daily_loss_limit(exchanges['binance'], pair_data[0],
                                                                               balances, atr, loss_model, loss_scaler)
                    can_trade_vol, reason_vol = await check_volatility(exchanges['binance'], pair_data[0], atr)
                    if not can_trade_loss:
                        print(f"{pair_data[0]}: Торговля остановлена до следующего дня: {reason_loss}")
                        await send_telegram_message(
                            f"{pair_data[0]}: Торговля остановлена до следующего дня: {reason_loss}")
                        continue
                    if not can_trade_vol:
                        print(f"{pair_data[0]}: Торговля приостановлена: {reason_vol}")
                        await send_telegram_message(f"{pair_data[0]}: Торговля приостановлена: {reason_vol}")
                        continue
                    if len(open_orders[pair_data[0]]) >= MAX_OPEN_ORDERS:
                        print(f"{pair_data[0]}: Достигнут лимит открытых ордеров ({MAX_OPEN_ORDERS})")
                        await send_telegram_message(
                            f"{pair_data[0]}: Достигнут лимит открытых ордеров ({MAX_OPEN_ORDERS})")
                        continue
                    task = trade_pair(exchanges, pair_data, balances, pred_model, scaler, fees, atr, loss_model,
                                      loss_scaler, open_orders, trade_fraction)
                    tasks.append(asyncio.ensure_future(task))

                if tasks:
                    await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

                iteration += 1
                print(f"Итерация {iteration} завершена")
                logging.info(f"Конец итерации {iteration}")

                await asyncio.sleep(10)

            except Exception as e:
                logging.error(f"Ошибка в цикле итерации: {str(e)}")
                await asyncio.sleep(10)

        logging.info("Достигнута последняя итерация, завершаем работу")
        await final_report(exchanges, balances)
        await shutdown(exchanges, balances, open_orders)
        await finalize_report(exchanges, balances)

    except Exception as e:
        logging.error(f"Критическая ошибка в main: {str(e)}")
        raise
    finally:
        logging.info("Закрытие соединения с Binance")
        await exchanges['binance'].close()


if __name__ == "__main__":
    asyncio.run(main())
