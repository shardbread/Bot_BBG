#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# limits.py
import numpy as np

from config import MAX_DRAWDOWN, BASE_DAILY_LOSS_LIMIT, VOLATILITY_THRESHOLD, LOOKBACK, SEQUENCE_LENGTH, \
    INITIAL_BALANCE, MIN_ORDER_SIZE
from data import get_historical_data, prepare_gru_data
from exchange import get_ticker
from globals import daily_losses, historical_losses, last_day, MAX_OPEN_ORDERS
import logging
import time


def check_drawdown(balances):
    total_value = sum(
        balance['base'] * (balance['entry_price'] if balance['entry_price'] else 0) + balance['quote_binance'] +
        balance['quote_bingx'] for balance in balances.values())
    drawdown = (INITIAL_BALANCE - total_value) / INITIAL_BALANCE
    if drawdown > MAX_DRAWDOWN:
        return False, f"Превышена максимальная просадка {drawdown * 100:.1f}% (> {MAX_DRAWDOWN * 100}%)"
    return True, ""


async def check_daily_loss_limit(exchange, pair, balances, atr, model, scaler):
    global daily_losses, last_day
    current_day = time.strftime("%Y-%m-%d")
    if current_day != last_day:
        daily_losses.clear()
        last_day = current_day

    volatility_factor = min(1.0, 1 - atr / (INITIAL_BALANCE * VOLATILITY_THRESHOLD))
    ticker = await get_ticker(exchange, pair)
    current_spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
    current_volume = (await get_historical_data(exchange, pair, limit=1))['volume'].iloc[-1]
    forecasted_loss = await forecast_loss(exchange, pair, atr, current_spread, current_volume, model, scaler)
    balance_key = 'quote_binance' if exchange.id == 'binance' else 'quote_bingx'
    dynamic_daily_loss_limit = max(BASE_DAILY_LOSS_LIMIT * volatility_factor,
                                   forecasted_loss / balances[pair][balance_key] * 1.5)

    if daily_losses[pair] > balances[pair][balance_key] * dynamic_daily_loss_limit:
        return False, f"{pair}: Превышен адаптивный дневной лимит убытков {daily_losses[pair]:.2f} (> {balances[pair][balance_key] * dynamic_daily_loss_limit:.2f}, Forecasted: {forecasted_loss:.2f})"
    return True, ""


async def check_volatility(exchange, symbol, atr):
    data = await get_historical_data(exchange, symbol)
    avg_price = data['close'].mean()
    volatility = atr / avg_price
    if volatility > VOLATILITY_THRESHOLD:
        return False, f"{symbol}: Высокая волатильность {volatility * 100:.1f}% (> {VOLATILITY_THRESHOLD * 100}%)"
    return True, ""


async def forecast_loss(exchange, pair, atr, current_spread, current_volume, model, scaler):
    if len(historical_losses[pair]) < LOOKBACK + SEQUENCE_LENGTH:
        return 0.0

    data = await get_historical_data(exchange, pair, limit=LOOKBACK + SEQUENCE_LENGTH)
    X_gru, _, scaler = prepare_gru_data(data, historical_losses[pair])
    atr_values = data['ATR'].values[-10:]
    rsi_values = data['RSI'].values[-10:]
    macd_values = data['MACD'].values[-10:]
    volume_values = data['volume'].values[-10:]
    bb_upper_values = data['BB_upper'].values[-10:]
    bb_lower_values = data['BB_lower'].values[-10:]
    bb_width_values = (data['BB_upper'] - data['BB_lower']).values[-10:]
    spreads = [current_spread] * 10

    X = np.column_stack([atr_values, spreads, volume_values, rsi_values, macd_values, bb_upper_values, bb_lower_values,
                         bb_width_values])
    X_scaled = scaler.transform(X)
    X_gru = np.array([X_scaled])

    forecasted_loss = model.predict(X_gru, verbose=0)[0][0]
    return max(0, min(forecasted_loss, INITIAL_BALANCE * BASE_DAILY_LOSS_LIMIT))


async def calculate_optimal_limit(balances):
    total_binance = sum(balance['quote_binance'] for balance in balances.values())
    total_bingx = sum(balance['quote_bingx'] for balance in balances.values())
    total_balance = total_binance + total_bingx

    min_balance_per_pair = MIN_ORDER_SIZE * 2
    max_possible_pairs = int(total_balance / min_balance_per_pair)

    optimal_limit = max(1, min(max_possible_pairs, 4))
    logging.info(
        f"Рассчитан оптимальный лимит пар: {optimal_limit} (Binance: {total_binance:.2f}, BingX: {total_bingx:.2f})")
    return optimal_limit
