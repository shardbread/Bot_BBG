#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ccxt.async_support as ccxt
import logging
import asyncio
from config import BINANCE_API_KEY, BINANCE_SECRET, BINGX_API_KEY, BINGX_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import httpx

async def connect_exchange(exchange_name):
    if exchange_name == 'binance':
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        exchange.set_sandbox_mode(True)
    elif exchange_name == 'bingx':
        exchange = ccxt.bingx({
            'apiKey': BINGX_API_KEY,
            'secret': BINGX_SECRET,
            'enableRateLimit': True,
        })
    await exchange.load_markets()
    return exchange

async def fetch_fees(exchange):
    fees = {}
    markets = await exchange.load_markets()
    for symbol in markets:
        fees[symbol] = {'maker': markets[symbol]['maker'], 'taker': markets[symbol]['taker']}
    return fees

async def get_ticker(exchange, symbol):
    ticker = await exchange.fetch_ticker(symbol)
    return {
        'bid': ticker['bid'],
        'ask': ticker['ask'],
        'last': ticker['last']
    }

async def manage_request(exchange, method, *args, **kwargs):
    try:
        result = await getattr(exchange, method)(*args, **kwargs)
        return result
    except Exception as e:
        logging.error(f"Ошибка в {method}: {str(e)}")
        raise e

async def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram токен или чат ID не настроены")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        except Exception as e:
            logging.error(f"Ошибка отправки Telegram сообщения: {str(e)}")
