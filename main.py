#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import logging
import signal
import sys
import time
import json
import os
from config import BINANCE_API_KEY, BINANCE_SECRET, BINGX_API_KEY, BINGX_SECRET_KEY, TRADING_PAIRS, ITERATIONS, TRADE_FRACTION
from exchange import setup_exchange, get_ticker
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
    signal.signal(signal.SIGINT, signal_handler)
    logging.info("Запуск скрипта")

    binance = await setup_exchange('binance', BINANCE_API_KEY, BINANCE_SECRET, testnet=True)  # Тестовый режим
    exchanges = {'binance': binance}
    fees = {'binance': 0.001}

    initial_balance = await load_initial_balance()
    if initial_balance is None:
        initial_balance = await get_initial_balance(exchanges['binance'])
        await save_initial_balance(initial_balance)

    historical_data = await get_historical_data(binance, 'BTC/USDT', limit=1000)
    if historical_data.empty:
        logging.error("Не удалось получить исторические данные, завершаем работу")
        await binance.close()
        return
    historical_data = await add_features(historical_data)
    if historical_data.empty:
        logging.error("Не удалось добавить признаки к данным, завершаем работу")
        await binance.close()
        return
    X, y, scaler = prepare_lstm_data(historical_data)
    if X.size == 0:
        logging.error("Данные для LSTM пусты, завершаем работу")
        await binance.close()
        return
    pred_model = train_lstm_model(X, y)
    X_loss, y_loss, loss_scaler = prepare_lstm_data(historical_data)
    if X_loss.size == 0:
        logging.error("Данные для GRU пусты, завершаем работу")
        await binance.close()
        return
    loss_model = train_gru_model(X_loss, y_loss)

    balance_info = await exchanges['binance'].fetch_balance()
    total_usdt = balance_info.get('USDT', {}).get('free', 0)
    saved_balances = await load_balances()

    balances = {}
    for pair in TRADING_PAIRS:
        base = pair.split('/')[0]
        if saved_balances and pair in saved_balances:
            balances[pair] = saved_balances[pair]
            balances[pair]['base'] = balance_info.get(base, {}).get('free', 0)
            balances[pair]['quote_binance'] = total_usdt / len(TRADING_PAIRS)
        else:
            balances[pair] = {
                'base': balance_info.get(base, {}).get('free', 0),
                'quote_binance': total_usdt / len(TRADING_PAIRS),
                'quote_bingx': 0.0,
                'entry_price': 0.0,
                'total_fees': 0.0,
                'cost': 0.0,
                'revenue': 0.0
            }

    for pair in TRADING_PAIRS:
        daily_losses[pair] = 0.0

    open_orders = {pair: [] for pair in TRADING_PAIRS}

    while running:
        try:
            for iteration in range(ITERATIONS):
                if not running:
                    break
                logging.info(f"Начало итерации {iteration + 1}")
                total_balance = sum(b['quote_binance'] for b in balances.values())
                logging.info(f"Текущий баланс: Total USDT: {total_balance:.2f}, Детали по парам: {balances}")

                logging.info("Вызов select_profitable_pairs")
                profitable_pairs = await select_profitable_pairs(exchanges, fees, pred_model, scaler, balances)

                if not profitable_pairs:
                    logging.info("Нет прибыльных пар для торговли в этой итерации")
                else:
                    tasks = [
                        trade_pair(
                            exchanges, pair_data, balances, pred_model, scaler, fees, pair_data[3],
                            loss_model, loss_scaler, open_orders, trade_fraction=TRADE_FRACTION,
                            iteration=iteration
                        ) for pair_data in profitable_pairs
                    ]
                    await asyncio.gather(*tasks)

                if iteration == ITERATIONS - 1:
                    for pair in TRADING_PAIRS:
                        if balances[pair]['base'] > 0:
                            historical_data = await get_historical_data(exchanges['binance'], pair, limit=100)
                            historical_data = await add_features(historical_data)
                            atr = historical_data['ATR'].iloc[-1] if not historical_data.empty else 0.02
                            pair_data = (pair, 0.0, 0.0, atr)
                            await trade_pair(
                                exchanges, pair_data, balances, pred_model, scaler, fees, atr,
                                loss_model, loss_scaler, open_orders, trade_fraction=TRADE_FRACTION,
                                iteration=iteration
                            )

                logging.info(f"Конец итерации {iteration + 1}")
                await asyncio.sleep(5)

            if running:
                await finalize_report(exchanges, balances, initial_balance)
                await save_balances(balances)
                logging.info("Цикл торговли завершён. Ожидание 5 минут перед следующим циклом...")
                await asyncio.sleep(300)

        except Exception as e:
            logging.error(f"Произошла ошибка в цикле торговли: {str(e)}")
            await asyncio.sleep(60)

    logging.info("Инициирована остановка бота. Завершаем открытые ордера...")
    for pair in TRADING_PAIRS:
        historical_data = await get_historical_data(exchanges['binance'], pair, limit=100)
        historical_data = await add_features(historical_data)
        atr = historical_data['ATR'].iloc[-1] if not historical_data.empty else 0.02
        await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)

    await finalize_report(exchanges, balances, initial_balance)
    await save_balances(balances)
    logging.info("Все ордера завершены. Бот остановлен.")
    logging.info("Закрытие соединения с Binance")
    await binance.close()

if __name__ == "__main__":
    asyncio.run(main())
