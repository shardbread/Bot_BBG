#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import asyncio
from exchange import manage_request, send_telegram_message


async def check_and_cancel_orders(exchange, pair, balances, atr, open_orders):
    try:
        for order in open_orders[pair][:]:  # Копируем список, чтобы избежать изменения во время итерации
            order_id = order['id']
            order_status = await manage_request(exchange, 'fetch_order', order_id, pair)
            if order_status['status'] == 'closed':
                logging.info(f"{pair}: Ордер {order_id} закрыт, статус: {order_status['status']}")
                if order['side'] == 'buy':
                    filled_amount = order_status.get('filled', 0)
                    balances[pair]['base'] += filled_amount
                    balances[pair]['quote_binance'] -= filled_amount * order_status['price']
                elif order['side'] == 'sell':
                    filled_amount = order_status.get('filled', 0)
                    balances[pair]['base'] -= filled_amount
                    balances[pair]['quote_binance'] += filled_amount * order_status['price']
                open_orders[pair].remove(order)
            elif order_status['status'] == 'open':
                current_price = (await exchange.fetch_ticker(pair))['last']
                if order['side'] == 'buy' and current_price < order_status['price'] * (1 - atr):
                    await manage_request(exchange, 'cancel_order', order_id, pair)
                    logging.info(f"{pair}: Ордер {order_id} отменён из-за ATR")
                    await send_telegram_message(f"{pair}: Ордер {order_id} отменён из-за ATR")
                    open_orders[pair].remove(order)
                elif order['side'] == 'sell' and current_price > order_status['price'] * (1 + atr):
                    await manage_request(exchange, 'cancel_order', order_id, pair)
                    logging.info(f"{pair}: Ордер {order_id} отменён из-за ATR")
                    await send_telegram_message(f"{pair}: Ордер {order_id} отменён из-за ATR")
                    open_orders[pair].remove(order)
    except Exception as e:
        logging.error(f"Ошибка при проверке ордеров для {pair}: {str(e)}")
