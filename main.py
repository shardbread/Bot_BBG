#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import logging
from config import TRADING_PAIRS, ITERATIONS
from exchange import Exchange

from model import train_models
from strategy import select_profitable_pairs, trade_pair, finalize_report

# Настройка логирования
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def main():
    global INITIAL_TOTAL_USDT
    logging.info("Запуск скрипта")

    exchanges = {
        'binance': Exchange('binance', testnet=True),
        'bingx': Exchange('binance', testnet=True)  # bingx
    }

    binance_balance = await exchanges['binance'].fetch_balance()
    logging.info(f"Начальный баланс из API Binance: {binance_balance}")
    usdt_data = binance_balance.get('USDT', 0)
    INITIAL_TOTAL_USDT = float(usdt_data['free']) if isinstance(usdt_data, dict) else float(usdt_data)

    # Инициализация и синхронизация баланса
    balances = {pair: {
        'base': 0.0,
        'quote_binance': INITIAL_TOTAL_USDT / len(TRADING_PAIRS),
        'quote_bingx': 0.0,
        'entry_price': 0.0,
        'total_fees': 0.0,
        'cost': 0.0,
        'revenue': 0.0
    } for pair in TRADING_PAIRS}
    logging.info(f"Распределённый баланс: {balances}")

    pred_model, scaler = await train_models(exchanges['binance'], TRADING_PAIRS[0])
    if pred_model is None or scaler is None:
        logging.error("Не удалось обучить модель, завершение работы")
        for exchange in exchanges.values():
            await exchange.close()
        return

    fees = {'binance': 0.001, 'bingx': 0.001}

    for iteration in range(ITERATIONS):
        # Синхронизация баланса перед каждой итерацией
        binance_balance = await exchanges['binance'].fetch_balance()
        total_usdt_actual = binance_balance['free'].get('USDT', 0)
        logging.info(f"Итерация {iteration + 1}: Реальный баланс USDT на Binance: {total_usdt_actual}")

        logging.info(f"Начало итерации {iteration + 1}")
        logging.info(
            f"Текущий баланс: Total USDT: {sum(b['quote_binance'] for b in balances.values()):.2f}, Детали по парам: {balances}")
        profitable_pairs = await select_profitable_pairs(exchanges, fees, pred_model, scaler, balances)
        tasks = [trade_pair(exchanges, pair_data[0], pred_model, scaler, balances, iteration + 1) for pair_data in
                 profitable_pairs]
        await asyncio.gather(*tasks)
        logging.info(f"Конец итерации {iteration + 1}")
        await asyncio.sleep(5)

    await finalize_report(exchanges, balances, INITIAL_TOTAL_USDT)

    for exchange in exchanges.values():
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())

