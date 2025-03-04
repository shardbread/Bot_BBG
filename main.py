#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import logging
import json
import os
from config import TRADING_PAIRS, ITERATIONS, LOOKBACK
from exchange import get_ticker, Exchange

from data import get_historical_data, prepare_lstm_data, add_features
from model import train_models
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

running = True

def signal_handler(signum, frame):
    global running
    logging.info("Получен сигнал остановки (Ctrl+C). Завершаем работу...")
    running = False

async def load_initial_balance():
    """Загрузка начального баланса из файла"""
    if os.path.exists('initial_balance.json'):
        with open('initial_balance.json', 'r') as f:
            data = json.load(f)
            return data.get('initial_balance', 0.0)
    return None

async def save_initial_balance(initial_balance):
    """Сохранение начального баланса в файл"""
    with open('initial_balance.json', 'w') as f:
        json.dump({'initial_balance': initial_balance}, f)

async def get_initial_balance(exchange):
    """Получение начального баланса из API Binance"""
    try:
        balance_info = await exchange.fetch_balance()
        total_usdt = balance_info.get('USDT', {}).get('free', 0)
        total_assets_value = 0.0
        for pair in TRADING_PAIRS:
            base = pair.split('/')[0]
            base_balance = balance_info.get(base, {}).get('free', 0)
            if base_balance > 0:
                ticker = await get_ticker(exchange, pair)
                asset_value = base_balance * ticker['ask']
                total_assets_value += asset_value
        initial_balance = total_usdt + total_assets_value
        logging.info(f"Начальный баланс из API Binance: {initial_balance:.2f} USDT")
        return initial_balance
    except Exception as e:
        logging.error(f"Ошибка получения начального баланса из API: {str(e)}")
        return 0.0

async def load_balances():
    """Загрузка сохранённых балансов из файла"""
    if os.path.exists('balances.json'):
        with open('balances.json', 'r') as f:
            return json.load(f)
    return None

async def save_balances(balances):
    """Сохранение текущих балансов в файл"""
    with open('balances.json', 'w') as f:
        json.dump(balances, f)


async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Запуск скрипта")

    binance = Exchange('binance', testnet=True)
    bingx = Exchange('bingx', testnet=True)
    exchanges = {'binance': binance, 'bingx': bingx}
    fees = {'binance': 0.001, 'bingx': 0.001}

    initial_balance = await binance.fetch_balance()
    balances = {
        pair: {'base': 0.0, 'quote_binance': initial_balance.get('USDT', {}).get('free', 0) / len(TRADING_PAIRS),
               'quote_bingx': 0.0, 'entry_price': 0.0, 'total_fees': 0.0, 'cost': 0.0, 'revenue': 0.0}
        for pair in TRADING_PAIRS}

    logging.info(f"Начальный баланс из API Binance: {initial_balance.get('USDT', {}).get('free', 0):.2f} USDT")
    with open('initial_balance.json', 'w') as f:
        json.dump({'USDT': initial_balance.get('USDT', {}).get('free', 0)}, f)

    historical_data = await get_historical_data(binance, 'BTC/USDT')
    historical_data = await add_features(historical_data)
    X, y, scaler = prepare_lstm_data(historical_data)
    lstm_model, gru_model = train_models(X, y)

    open_orders = {pair: [] for pair in TRADING_PAIRS}
    trade_fraction = 0.3

    try:
        for iteration in range(ITERATIONS):
            logging.info(f"Начало итерации {iteration + 1}")
            total_usdt = sum(pair_data['quote_binance'] for pair_data in balances.values())
            logging.info(f"Текущий баланс: Total USDT: {total_usdt:.2f}, Детали по парам: {balances}")

            profitable_pairs = await select_profitable_pairs(exchanges, fees, lstm_model, scaler, balances)

            tasks = []
            for pair_data in profitable_pairs:
                # Ансамбль: усредняем предсказания LSTM и GRU
                prediction_data = await get_historical_data(exchanges['binance'], pair_data[0], limit=LOOKBACK + 100)
                prediction_data = await add_features(prediction_data)
                X_pred, _, _ = prepare_lstm_data(prediction_data)
                lstm_pred = lstm_model.predict(X_pred[-1:], verbose=0)[0][0]
                gru_pred = gru_model.predict(X_pred[-1:], verbose=0)[0][0]
                ensemble_pred = (lstm_pred + gru_pred) / 2
                pair_data = (pair_data[0], ensemble_pred, pair_data[2], pair_data[3])
                tasks.append(trade_pair(exchanges, pair_data, balances, lstm_model, scaler, fees,
                                        pair_data[3], None, None, open_orders, trade_fraction, iteration))
            if tasks:
                await asyncio.gather(*tasks)
            logging.info(f"Конец итерации {iteration + 1}")
            await asyncio.sleep(5)

        await finalize_report(exchanges, balances, initial_balance.get('USDT', {}).get('free', 0))
        logging.info("Цикл торговли завершён. Ожидание 5 минут перед следующим циклом...")
        await asyncio.sleep(300)

    except KeyboardInterrupt:
        logging.info("Получен сигнал остановки (Ctrl+C). Завершаем работу...")
        logging.info("Инициирована остановка бота. Завершаем открытые ордера...")
        for pair in TRADING_PAIRS:
            await check_and_cancel_orders(exchanges['binance'], pair, balances, balances[pair].get('atr', 0.04),
                                          open_orders)
            await trade_pair(exchanges, (pair, 0, 0, balances[pair].get('atr', 0.04)), balances, lstm_model, scaler,
                             fees,
                             balances[pair].get('atr', 0.04), None, None, open_orders, trade_fraction, ITERATIONS - 1)
        await finalize_report(exchanges, balances, initial_balance.get('USDT', {}).get('free', 0))
        logging.info("Все ордера завершены. Бот остановлен.")

    finally:
        await binance.close()
        await bingx.close()
        logging.info("Закрытие соединения с Binance")


if __name__ == "__main__":
    asyncio.run(main())
