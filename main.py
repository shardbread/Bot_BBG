#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import logging
import ccxt.async_support as ccxt
from config import API_KEY, SECRET_KEY, BINGX_API_KEY, BINGX_SECRET_KEY, TRADING_PAIRS, ITERATIONS
from exchange import setup_exchange
from model import train_lstm_model, train_gru_model
from data import get_historical_data, prepare_lstm_data, add_features
from order_management import check_and_cancel_orders
from strategy import select_profitable_pairs, trade_pair, finalize_report
from limits import calculate_atr
from globals import MAX_OPEN_ORDERS, daily_losses

# Настройка логирования
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    logging.info("Запуск скрипта")

    # Инициализация бирж
    binance = await setup_exchange('binance', API_KEY, SECRET_KEY, testnet=True)
    bingx = await setup_exchange('bingx', BINGX_API_KEY, BINGX_SECRET_KEY)

    exchanges = {'binance': binance, 'bingx': bingx}
    fees = {'binance': 0.001, 'bingx': 0.001}  # Предполагаемые комиссии

    # Подготовка данных и обучение моделей
    historical_data = await get_historical_data(binance, 'BTC/USDT', limit=1000)
    X, y, scaler = prepare_lstm_data(historical_data)
    pred_model = train_lstm_model(X, y)
    X_loss, y_loss, loss_scaler = prepare_lstm_data(add_features(historical_data))
    loss_model = train_gru_model(X_loss, y_loss)

    # Инициализация балансов
    initial_balance = 7537.93  # Начальный баланс в USDT
    bingx_balance = 600.00    # Баланс BingX в USDT
    balances = {
        pair: {
            'base': 0.0,
            'quote_binance': initial_balance / len(TRADING_PAIRS),
            'quote_bingx': bingx_balance / len(TRADING_PAIRS),
            'entry_price': 0.0,
            'total_fees': 0.0,
            'cost': 0.0,    # Затраты на покупку
            'revenue': 0.0  # Доходы от продажи
        } for pair in TRADING_PAIRS
    }

    # Инициализация глобальных переменных
    for pair in TRADING_PAIRS:
        daily_losses[pair] = 0.0

    open_orders = {pair: [] for pair in TRADING_PAIRS}

    # Основной цикл торговли
    for iteration in range(1, ITERATIONS + 1):
        logging.info(f"Начало итерации {iteration}")
        total_balance = sum(b['quote_binance'] for b in balances.values())
        logging.info(f"Текущий баланс: Total USDT: {total_balance:.2f}, Детали по парам: {balances}")

        logging.info("Вызов select_profitable_pairs")
        profitable_pairs = await select_profitable_pairs(exchanges, fees, pred_model, scaler, balances)

        if not profitable_pairs:
            logging.info("Нет прибыльных пар для торговли в этой итерации")
            continue

        atr = calculate_atr(historical_data)

        tasks = [
            trade_pair(
                exchanges, pair_data, balances, pred_model, scaler, fees, atr,
                loss_model, loss_scaler, open_orders, trade_fraction=0.1
            ) for pair_data in profitable_pairs
        ]
        await asyncio.gather(*tasks)

        logging.info(f"Конец итерации {iteration}")

    logging.info("Достигнута последняя итерация, завершаем работу")
    await finalize_report(exchanges, balances)

    # Закрытие оставшихся открытых ордеров
    logging.info("Инициирована остановка бота. Завершаем открытые ордера...")
    for pair in TRADING_PAIRS:
        await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)

    logging.info("Все ордера завершены. Бот остановлен.")
    await finalize_report(exchanges, balances)

    # Закрытие соединений
    logging.info("Закрытие соединения с Binance")
    await binance.close()
    await bingx.close()

if __name__ == "__main__":
    asyncio.run(main())
