#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt.async_support as ccxt
import logging
import asyncio

async def setup_exchange(exchange_name, api_key, secret_key, testnet=False):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange

async def get_ticker(exchange, pair):
    try:
        ticker = await exchange.fetch_ticker(pair)
        return ticker
    except Exception as e:
        logging.error(f"Ошибка при получении тикера для {pair}: {str(e)}")
        return {'bid': 0, 'ask': 0}

async def manage_request(exchange, method, *args, **kwargs):
    try:
        if method == 'create_limit_buy_order':
            order = await exchange.create_limit_buy_order(*args, **kwargs)
        elif method == 'create_market_sell_order':
            order = await exchange.create_market_sell_order(*args, **kwargs)
        elif method == 'fetch_order':
            order = await exchange.fetch_order(*args, **kwargs)
        elif method == 'cancel_order':
            order = await exchange.cancel_order(*args, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        return order
    except Exception as e:
        logging.error(f"Ошибка в {method}: {str(e)}")
        raise e

async def send_telegram_message(bot_message):
    # Заглушка для отправки в Telegram (пока без реальной интеграции)
    logging.info(f"Telegram сообщение: {bot_message}")
    # Для реальной отправки раскомментируйте и настройте:
    # import telegram
    # bot = telegram.Bot(token='YOUR_TELEGRAM_BOT_TOKEN')
    # await bot.send_message(chat_id='YOUR_CHAT_ID', text=bot_message)
