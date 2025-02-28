#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import asyncio
from exchange import send_telegram_message


async def check_and_cancel_orders(exchange, symbol, balances, atr, open_orders):
    base, quote = symbol.split('/')
    balance_key = 'quote_' + exchange.name.lower()
    if not open_orders[symbol]:
        return

    orders_to_remove = []
    for i, order in enumerate(open_orders[symbol]):
        try:
            order_info = await exchange.fetch_order(order['id'], symbol)
            if order_info is not None and order_info['status'] == 'closed':
                fee_cost = order_info.get('fee', {}).get('cost', 0.0) if order_info.get('fee') is not None else 0.0
                if order['side'] == 'buy':
                    balances[symbol][balance_key] -= order_info['filled'] * order_info['price'] + fee_cost
                    balances[symbol]['base'] += order_info['filled']
                else:
                    balances[symbol][balance_key] += order_info['filled'] * order_info['price'] - fee_cost
                    balances[symbol]['base'] -= order_info['filled']
                    balances[symbol]['total_fees'] += fee_cost
                from globals import daily_losses
                daily_losses[symbol] += fee_cost
                orders_to_remove.append(i)
                logging.info(f"{symbol}: Ордер {order['id']} закрыт, статус: {order_info['status']}")
            elif order_info is None:
                logging.warning(f"{symbol}: Ордер {order['id']} не найден на бирже")
                orders_to_remove.append(i)
        except Exception as e:
            logging.error(f"Ошибка проверки ордера {order['id']}: {str(e)}")

    for index in sorted(orders_to_remove, reverse=True):
        open_orders[symbol].pop(index)


async def shutdown(exchanges, balances, open_orders):
    logging.info("Инициирована остановка бота. Завершаем открытые ордера...")
    for exchange_name, exchange in exchanges.items():
        for symbol in balances:
            await check_and_cancel_orders(exchange, symbol, balances, 0, open_orders)
            base, quote = symbol.split('/')
            balance_key = 'quote_' + exchange_name
            if balances[symbol]['base'] > 0:
                try:
                    order_book = await exchange.fetch_order_book(symbol)
                    ask_price = order_book['asks'][0][0]
                    order = await exchange.create_limit_sell_order(symbol, balances[symbol]['base'], ask_price)
                    balances[symbol][balance_key] += balances[symbol]['base'] * ask_price
                    balances[symbol]['base'] = 0
                    await send_telegram_message(
                        f"{symbol}: Проданы остатки {balances[symbol]['base']} {base} на {exchange_name} по {ask_price}")
                except Exception as e:
                    logging.error(f"Ошибка продажи остатков {symbol} на {exchange_name}: {str(e)}")
                    await send_telegram_message(f"Ошибка продажи остатков {symbol} на {exchange_name}: {str(e)}")
    logging.info("Все ордера завершены. Бот остановлен.")
