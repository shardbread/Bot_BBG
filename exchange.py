#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exchange.py
import ccxt.async_support as ccxt_async
import aiohttp
import logging
from collections import defaultdict
import time
from config import BINANCE_API_KEY, BINANCE_SECRET, BINGX_API_KEY, BINGX_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

request_timestamps = []
cache = defaultdict(dict)

async def connect_exchange(exchange_name):
    if exchange_name == 'binance':
        if not BINANCE_API_KEY or not BINANCE_SECRET:
            raise ValueError("Binance API Key or Secret not set in environment variables")
        return ccxt_async.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    elif exchange_name == 'bingx':
        if not BINGX_API_KEY or not BINGX_SECRET:
            raise ValueError("BingX API Key or Secret not set in environment variables")
        return ccxt_async.bingx({
            'apiKey': BINGX_API_KEY,
            'secret': BINGX_SECRET,
            'enableRateLimit': True
        })
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

async def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Telegram Bot Token or Chat ID not set in environment variables")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
    async with aiohttp.ClientSession() as session:
        try:
            await session.get(url)
        except Exception as e:
            logging.error(f"Ошибка отправки Telegram сообщения: {str(e)}")

async def manage_request(exchange, method, *args, **kwargs):
    global request_timestamps
    current_time = time.time()
    request_timestamps = [t for t in request_timestamps if current_time - t < 60]
    limit = 20
    if len(request_timestamps) >= limit:
        await asyncio.sleep(1 / limit)
    request_timestamps.append(current_time)
    return await getattr(exchange, method)(*args, **kwargs)

async def fetch_fees(exchange):
    fees = await manage_request(exchange, 'fetch_trading_fees')
    return {pair: {'maker': fee['maker'], 'taker': fee['taker']} for pair, fee in fees.items()}
