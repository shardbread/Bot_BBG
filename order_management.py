#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# order_management.py
from exchange import manage_request, send_telegram_message
from globals import open_orders, running, daily_losses, historical_losses
import logging
import asyncio


async def check_and_cancel_orders(exchange, symbol, balances, atr, open_orders):
    if symbol in open_orders:
        timeout = max(120, atr * 60)
        for order in open_orders[symbol][:]:
            order_info = await manage_request(exchange, 'fetch_order', order['id'], symbol)
            if order_info['status'] == 'closed':
                balance_key = 'quote_binance' if exchange.id == 'binance' else 'quote_bingx'
                if order_info['side'] == 'buy':
                    balances[symbol]['base'] += order_info['filled']
                    balances[symbol][balance_key] -= order_info['filled'] * order_info['price'] + order_info['fee'][
                        'cost']
                elif order_info['side'] == 'sell':
                    balances[symbol][balance_key] += order_info['filled'] * order_info['price'] - order_info['fee'][
                        'cost']
                    balances[symbol]['base'] -= order_info['filled']
                    if order_info['price'] < balances[symbol]['entry_price']:
                        loss = (balances[symbol]['entry_price'] - order_info['price']) * order_info['filled']
                        daily_losses[symbol] += loss
                        historical_losses[symbol].append(loss)
                balances[symbol]['total_fees'] += order_info['fee']['cost']
                open_orders[symbol].remove(order)
                msg = f"{symbol}: Ордер {order['id']} исполнен, {order_info['side']} {order_info['filled']} по {order_info['price']}, Комиссия ${order_info['fee']['cost']:.2f}"
                logging.info(msg)
                await send_telegram_message(msg)
            elif time.time() - order['timestamp'] > timeout:
                await manage_request(exchange, 'cancel_order', order['id'], symbol)
                open_orders[symbol].remove(order)
                msg = f"{symbol}: Ордер {order['id']} отменен (не исполнен за {timeout / 60:.1f} мин, ATR: {atr:.2f})"
                logging.info(msg)
                await send_telegram_message(msg)


async def shutdown(exchanges, balances, open_orders):
    global running
    running = False
    logging.info("Инициирована остановка бота. Завершаем открытые ордера...")
    await send_telegram_message("Бот останавливается. Завершаем все открытые ордера.")

    for pair in open_orders:
        for exchange in exchanges.values():
            while open_orders[pair]:
                await check_and_cancel_orders(exchange, pair, balances, 0, open_orders)
                await asyncio.sleep(1)

    logging.info("Все ордера завершены. Бот остановлен.")
    await send_telegram_message("Все ордера завершены. Бот остановлен.")
