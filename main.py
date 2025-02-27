#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import asyncio
import signal
from exchange import connect_exchange, fetch_fees, send_telegram_message
from data import get_historical_data, prepare_lstm_data, prepare_gru_data, add_features
from model import train_lstm_model, train_gru_model
from strategy import trade_pair, select_profitable_pairs
from order_management import shutdown
from limits import check_drawdown, check_daily_loss_limit, check_volatility
from config import TRADING_PAIRS, INITIAL_BALANCE
from globals import MAX_OPEN_ORDERS
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
        for pair in TRADING_PAIRS:
            balances[pair] = {
                'base': 0.0,
                'quote_binance': 128.0,
                'quote_bingx': 100.0,
                'entry_price': None,
                'total_fees': 0.0
            }

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(exchanges, balances, open_orders)))

        iteration = 0
        while running and iteration < 10:
            try:
                logging.info(f"Начало итерации {iteration + 1}")
                can_trade_drawdown, reason_drawdown = check_drawdown(balances)
                if not can_trade_drawdown:
                    print(f"Торговля остановлена: {reason_drawdown}")
                    await send_telegram_message(f"Торговля остановлена: {reason_drawdown}")
                    break

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
                    tasks.append(trade_pair(exchanges, pair_data, balances, pred_model, scaler, fees, atr, loss_model,
                                            loss_scaler, open_orders))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                iteration += 1
                print(f"Итерация {iteration} завершена")
                logging.info(f"Конец итерации {iteration}")

            except Exception as e:
                logging.error(f"Ошибка в цикле итерации: {str(e)}")
                await asyncio.sleep(10)

            await asyncio.sleep(300)

        if not running:
            await shutdown(exchanges, balances, open_orders)

    except Exception as e:
        logging.error(f"Критическая ошибка в main: {str(e)}")
        raise
    finally:
        logging.info("Закрытие соединения с Binance")
        await exchanges['binance'].close()


if __name__ == "__main__":
    asyncio.run(main())
