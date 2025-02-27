#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ccxt.async_support as ccxt_async
import asyncio
from config import BINANCE_API_KEY, BINANCE_SECRET

async def test_binance():
    exchange = ccxt_async.binance({
        'apiKey': "2fKlDhMtjeAUCFMYZobv5WWJAqSSy20kfCiE5Z3tTVjB5vLumBi19BLoPauTtNYw",
        'secret': "JjjvOSyxMNmrZTkmMk8I7RYdrF6lN9rKvDa1Sp6xDUfvcoUKad4OWrNKpBBNyljz",
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'urls': {
            'api': {
                'public': 'https://testnet.binance.vision/api',
                'private': 'https://testnet.binance.vision/api',
            }
        }
    })
    exchange.set_sandbox_mode(True)
    try:
        balance = await exchange.fetch_balance()
        print("Баланс:", balance['total'])
    except Exception as e:
        print("Ошибка:", e)
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(test_binance())
