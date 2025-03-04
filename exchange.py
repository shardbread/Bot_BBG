#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# exchange.py
import ccxt.async_support as ccxt
import asyncio
import logging
import telegram
import os
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class Exchange:
    def __init__(self, name, testnet=False):
        self.name = name
        self.testnet = testnet
        exchange_class = getattr(ccxt, name)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'asyncio_loop': asyncio.get_event_loop(),
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logging.info(f"{name} настроен в тестовом режиме")
        else:
            # Здесь можно добавить API ключи для реальной торговли
            self.exchange.apiKey = os.getenv(f'{name.upper()}_API_KEY', '')
            self.exchange.secret = os.getenv(f'{name.upper()}_API_SECRET', '')
            logging.info(f"{name} настроен для реальной торговли")

    async def fetch_balance(self):
        try:
            balance = await self.exchange.fetch_balance()
            return balance['free']
        except Exception as e:
            logging.error(f"Ошибка получения баланса на {self.name}: {str(e)}")
            return {}

    async def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logging.error(f"Ошибка получения OHLCV на {self.name} для {symbol}: {str(e)}")
            return []

    async def fetch_order_book(self, symbol):
        try:
            order_book = await self.exchange.fetch_order_book(symbol)
            return order_book
        except Exception as e:
            logging.error(f"Ошибка получения книги ордеров на {self.name} для {symbol}: {str(e)}")
            return {'bids': [], 'asks': []}

    async def create_limit_buy_order(self, symbol, amount, price):
        try:
            order = await self.exchange.create_limit_buy_order(symbol, amount, price)
            logging.info(f"Создан лимитный ордер на покупку {amount} {symbol} по {price} на {self.name}")
            return order
        except Exception as e:
            logging.error(f"Ошибка создания лимитного ордера на покупку {symbol} на {self.name}: {str(e)}")
            raise e

    async def create_market_sell_order(self, symbol, amount):
        try:
            order = await self.exchange.create_market_sell_order(symbol, amount)
            logging.info(f"Создан рыночный ордер на продажу {amount} {symbol} на {self.name}")
            return order
        except Exception as e:
            logging.error(f"Ошибка создания рыночного ордера на продажу {symbol} на {self.name}: {str(e)}")
            raise e

    async def fetch_order(self, order_id, symbol):
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logging.error(f"Ошибка получения статуса ордера {order_id} на {self.name}: {str(e)}")
            return None

    async def cancel_order(self, order_id, symbol):
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logging.info(f"Ордер {order_id} для {symbol} отменён на {self.name}")
        except Exception as e:
            logging.error(f"Ошибка отмены ордера {order_id} на {self.name}: {str(e)}")

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
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info(f"Telegram сообщение: {message}")
    except Exception as e:
        logging.error(f"Ошибка отправки сообщения в Telegram: {str(e)}")
