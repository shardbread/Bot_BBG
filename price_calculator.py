#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import asyncio


async def get_order_book(exchange, symbol):
    order_book = await exchange.fetch_order_book(symbol)
    return {
        'bids': order_book['bids'],
        'asks': order_book['asks']
    }


async def get_best_price_and_amount(exchange, symbol, order_book, side, max_position_size, balances, atr, loss_model,
                                    loss_scaler, exchange_name):
    forecast_loss = 0
    if exchange_name == 'binance':
        quote_balance = balances[symbol]['quote_binance']
    else:
        quote_balance = balances[symbol]['quote_bingx']

    if side == 'buy':
        orders = order_book['bids']
        price_key, amount_key = 0, 1
    else:
        orders = order_book['asks']
        price_key, amount_key = 0, 1

    total_amount = 0
    total_cost = 0
    best_price = 0
    for price, amount in orders:
        available_amount = min(amount, max_position_size - total_amount)
        if available_amount <= 0:
            break
        total_amount += available_amount
        total_cost += available_amount * price
        best_price = price

    if total_amount == 0:
        return None, 0

    return best_price, total_amount
