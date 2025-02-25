#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py
import asyncio
import signal
from exchange import connect_exchange, fetch_fees, send_telegram_message
from data import get_historical_data, prepare_lstm_data, prepare_gru_data
from model import train_lstm_model, train_gru_model
from strategy import trade_pair, select_profitable_pairs
from order_management import shutdown
from limits import check_drawdown, check_daily_loss_limit, check_volatility
from config import TRADING_PAIRS, INITIAL_BALANCE
from globals import MAX_OPEN_ORDERS
import logging
from collections import defaultdict

running = True


async def main():
    global running
    exchanges = {
        'binance': await connect_exchange('binance'),
        'bingx': await connect_exchange('bingx')
    }

    fees = {
        'binance': await fetch_fees(exchanges['binance']),
        'bingx': await fetch_fees(exchanges['bingx'])
    }

    data = await get_historical_data(exchanges['binance'], 'ETH/USDT', limit=2000)
    X, y, scaler = prepare_lstm_data(data)
    pred_model = train_lstm_model(X, y)

    loss_data = await get_historical_data(exchanges['binance'], 'ETH/USDT', limit=2000)
    X_loss, y_loss, loss_scaler = prepare_gru_data(loss_data, [0] * 38)
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
            can_trade_drawdown, reason_drawdown = check_drawdown(balances)
            if not can_trade_drawdown:
                print(f"Торговля остановлена: {reason_drawdown}")
                await send_telegram_message(f"Торговля остановлена: {reason_drawdown}")
                break

            profitable_pairs = await select_profitable_pairs(exchanges, fees, pred_model, scaler, balances)
            logging.info(f"Выбраны пары: {profitable_pairs} с лимитом {MAX_OPEN_ORDERS}")
            await send_telegram_message(f"Выбраны пары для торговли: {profitable_pairs} с лимитом {MAX_OPEN_ORDERS}")

            tasks = []
            for pair in profitable_pairs:
                atr = (await get_historical_data(exchanges['binance'], pair)).iloc[-1]['ATR']
                can_trade_loss, reason_loss = await check_daily_loss_limit(exchanges['binance'], pair, balances, atr,
                                                                           loss_model, loss_scaler)
                can_trade_vol, reason_vol = await check_volatility(exchanges['binance'], pair, atr)
                if not can_trade_loss:
                    print(f"{pair}: Торговля остановлена до следующего дня: {reason_loss}")
                    await send_telegram_message(f"{pair}: Торговля остановлена до следующего дня: {reason_loss}")
                    continue
                if not can_trade_vol:
                    print(f"{pair}: Торговля приостановлена: {reason_vol}")
                    await send_telegram_message(f"{pair}: Торговля приостановлена: {reason_vol}")
                    continue
                if len(open_orders[pair]) >= MAX_OPEN_ORDERS:
                    print(f"{pair}: Достигнут лимит открытых ордеров ({MAX_OPEN_ORDERS})")
                    await send_telegram_message(f"{pair}: Достигнут лимит открытых ордеров ({MAX_OPEN_ORDERS})")
                    continue
                tasks.append(
                    trade_pair(exchanges, pair, balances, pred_model, scaler, fees, atr, loss_model, loss_scaler,
                               open_orders))
            await asyncio.gather(*tasks)

            iteration += 1
            print(f"Итерация {iteration} завершена")

        except Exception as e:
            logging.error(f"Ошибка: {str(e)}")
            await asyncio.sleep(10)

        await asyncio.sleep(300)

    if not running:
        await shutdown(exchanges, balances, open_orders)

    await exchanges['binance'].close()
    await exchanges['bingx'].close()


if __name__ == "__main__":
    asyncio.run(main())
