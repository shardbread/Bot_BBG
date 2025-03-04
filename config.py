#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

MODE = "test"

CACHE_TIMEOUT = 60
BASE_PRICE_ADJUSTMENT = 0.002
DEPTH_LEVELS = 5
BASE_MAX_POSITION_SIZE = 0.2
INITIAL_MAX_OPEN_ORDERS = 2
LOOKBACK = 120
INITIAL_TOTAL_USDT = 228.0
MIN_ORDER_SIZE = 10.0
MIN_SELL_SIZE = 0.1  # Увеличенный порог для продажи остатков
MAX_PREDICTION = 0.25
MAX_PROB = 0.25  # Снижено с 0.3 до 0.25 для охвата всех пар
TRADE_FRACTION = 0.3

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET')
BINGX_API_KEY = os.getenv('BINGX_API_KEY')
BINGX_SECRET_KEY = os.getenv('BINGX_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

TRADING_PAIRS = [
    'ETH/USDT', 'BTC/USDT', 'DOGE/USDT', 'XRP/USDT', 'BNB/USDT', 'ADA/USDT'
]

ITERATIONS = 10
