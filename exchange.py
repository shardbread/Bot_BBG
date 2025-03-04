#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exchange.py
import ccxt.async_support as ccxt
import asyncio
import logging
import telegram
import os
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_SECRET, BINGX_API_KEY, \
    BINGX_SECRET_KEY


class Exchange:
    def __init__(self, exchange_name, testnet=False):
        if exchange_name == 'binance':
            self.exchange = ccxt.binance({
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET,
                'enableRateLimit': True,
            })
            if testnet:
                self.exchange.set_sandbox_mode(True)
                logging.info("binance настроен в тестовом режиме")
        elif exchange_name == 'bingx':
            self.exchange = ccxt.bingx({
                'apiKey': BINGX_API_KEY,
                'secret': BINGX_SECRET_KEY,
                'enableRateLimit': True,
            })
            if testnet:
                logging.info("bingx настроен в тестовом режиме")
        else:
            raise ValueError(f"Неизвестная биржа: {exchange_name}")
        self.name = exchange_name

    async def fetch_balance(self):
        return await self.exchange.fetch_balance()

    async def fetch_ticker(self, pair):
        return await self.exchange.fetch_ticker(pair)

    async def fetch_order_book(self, pair):
        return await self.exchange.fetch_order_book(pair)

    async def fetch_order(self, order_id, pair):
        return await self.exchange.fetch_order(order_id, pair)

    async def create_limit_buy_order(self, pair, amount, price):
        return await self.exchange.create_order(pair, 'limit', 'buy', amount, price)

    async def create_limit_sell_order(self, pair, amount, price):
        return await self.exchange.create_order(pair, 'limit', 'sell', amount, price)

    async def fetch_ohlcv(self, pair, timeframe='1h', limit=100):
        return await self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)

    async def close(self):
        await self.exchange.close()
        logging.info(f"Соединение с {self.name} закрыто")

async def get_ticker(exchange, symbol):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        return {'bid': ticker['bid'], 'ask': ticker['ask']}
    except Exception as e:
        logging.error(f"Ошибка получения тикера для {symbol} на {exchange.name}: {str(e)}")
        return {'bid': 0, 'ask': 0}

async def manage_request(exchange, method, *args, **kwargs):
    try:
        func = getattr(exchange, method)
        result = await func(*args, **kwargs)
        return result
    except Exception as e:
        logging.error(f"Ошибка в manage_request для {method} на {exchange.name}: {str(e)}")
        raise e

async def send_telegram_message(message):
    try:
        from telegram import Bot  # Исправлено: импортируем Bot из модуля telegram
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info(f"Telegram сообщение: {message}")
    except Exception as e:
        logging.error(f"Ошибка отправки сообщения в Telegram: {str(e)}")
