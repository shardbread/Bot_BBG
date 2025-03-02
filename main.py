#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import logging
from config import BINANCE_API_KEY, BINANCE_SECRET, BINGX_API_KEY, BINGX_SECRET_KEY, TRADING_PAIRS, ITERATIONS, \
    TRADE_FRACTION
from exchange import setup_exchange
from model import train_lstm_model, train_gru_model
from data import get_historical_data, prepare_lstm_data, add_features
from strategy import select_profitable_pairs, trade_pair, finalize_report
from order_management import check_and_cancel_orders
from limits import calculate_optimal_limit
from globals import MAX_OPEN_ORDERS, daily_losses

# Настройка логирования
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    logging.info("Запуск скрипта")

    # Инициализация биржи (только Binance, BingX временно отключён)
    binance = await setup_exchange('binance', BINANCE_API_KEY, BINANCE_SECRET, testnet=True)
    exchanges = {'binance': binance}
    fees = {'binance': 0.001}

    # Подготовка данных и обучение моделей
    historical_data = await get_historical_data(binance, 'BTC/USDT', limit=1000)
    if historical_data.empty:
        logging.error("Не удалось получить исторические данные, завершаем работу")
        return
    historical_data = await add_features(historical_data)
    if historical_data.empty:
        logging.error("Не удалось добавить признаки к данным, завершаем работу")
        return
    X, y, scaler = prepare_lstm_data(historical_data)
    if X.size == 0:
        logging.error("Данные для LSTM пусты, завершаем работу")
        return
    pred_model = train_lstm_model(X, y)
    X_loss, y_loss, loss_scaler = prepare_lstm_data(historical_data)
    if X_loss.size == 0:
        logging.error("Данные для GRU пусты, завершаем работу")
        return
    loss_model = train_gru_model(X_loss, y_loss)

    # Инициализация балансов
    initial_balance = 7537.93  # Начальный баланс в USDT
    bingx_balance = 600.00    # Оставляем для будущего включения BingX
    balances = {
        pair: {
            'base': 0.0,
            'quote_binance': initial_balance / len(TRADING_PAIRS),
            'quote_bingx': 0.0,
            'entry_price': 0.0,
            'total_fees': 0.0,
            'cost': 0.0,
            'revenue': 0.0
        } for pair in TRADING_PAIRS
    }

    # Инициализация глобальных переменных
    for pair in TRADING_PAIRS:
        daily_losses[pair] = 0.0

    open_orders = {pair: [] for pair in TRADING_PAIRS}

    # Основной цикл торговли
    for iteration in range(ITERATIONS):
        logging.info(f"Начало итерации {iteration + 1}")
        total_balance = sum(b['quote_binance'] for b in balances.values())
        logging.info(f"Текущий баланс: Total USDT: {total_balance:.2f}, Детали по парам: {balances}")

        logging.info("Вызов select_profitable_pairs")
        profitable_pairs = await select_profitable_pairs(exchanges, fees, pred_model, scaler, balances)

        if not profitable_pairs:
            logging.info("Нет прибыльных пар для торговли в этой итерации")
        else:
            atr = 0.02  # Фиксированное значение ATR (2%) как заглушка

            tasks = [
                trade_pair(
                    exchanges, pair_data, balances, pred_model, scaler, fees, atr,
                    loss_model, loss_scaler, open_orders, trade_fraction=TRADE_FRACTION,
                    iteration=iteration
                ) for pair_data in profitable_pairs
            ]
            await asyncio.gather(*tasks)

        # В последней итерации продаём остатки для всех пар с ненулевым balance_base
        if iteration == ITERATIONS - 1:
            for pair in TRADING_PAIRS:
                if balances[pair]['base'] > 0:
                    # Создаём фиктивный pair_data для вызова trade_pair
                    pair_data = (pair, 0.0)  # prediction не важен для продажи остатков
                    await trade_pair(
                        exchanges, pair_data, balances, pred_model, scaler, fees, atr,
                        loss_model, loss_scaler, open_orders, trade_fraction=TRADE_FRACTION,
                        iteration=iteration
                    )

        logging.info(f"Конец итерации {iteration + 1}")

    logging.info("Достигнута последняя итерация, завершаем работу")
    await finalize_report(exchanges, balances)

    logging.info("Инициирована остановка бота. Завершаем открытые ордера...")
    for pair in TRADING_PAIRS:
        await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)

    logging.info("Все ордера завершены. Бот остановлен.")
    await finalize_report(exchanges, balances)

    # Закрытие соединения
    logging.info("Закрытие соединения с Binance")
    await binance.close()

if __name__ == "__main__":
    asyncio.run(main())
