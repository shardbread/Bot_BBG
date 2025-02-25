#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# price_calculator.py
from config import BASE_PRICE_ADJUSTMENT, DEPTH_LEVELS, BASE_MAX_POSITION_SIZE, BASE_DAILY_LOSS_LIMIT, \
    VOLATILITY_THRESHOLD
from data import get_historical_data, get_ticker
from limits import forecast_loss


async def get_best_price_and_amount(exchange, symbol, order_book, side, desired_amount, balances, atr, model, scaler,
                                    exchange_name):
    levels = order_book['bids'] if side == 'buy' else order_book['asks']
    total_amount = 0
    total_cost = 0
    depth_amount = 0

    for i, (price, amount) in enumerate(levels[:DEPTH_LEVELS]):
        depth_amount += amount
        amount_to_take = min(desired_amount - total_amount, amount)
        if amount_to_take > 0:
            total_amount += amount_to_take
            total_cost += amount_to_take * price
            if total_amount >= desired_amount:
                break

    avg_price = total_cost / total_amount if total_amount > 0 else levels[0][0]
    available_amount = min(total_amount, desired_amount)

    data = await get_historical_data(exchange, symbol)
    avg_price_hist = data['close'].mean()
    volatility = atr / avg_price_hist
    depth_factor = min(depth_amount / desired_amount, 1.0) if desired_amount > 0 else 1.0
    position_size_factor = max(0.5, 1 - volatility / VOLATILITY_THRESHOLD) * depth_factor

    ticker = await get_ticker(exchange, symbol)
    current_spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
    current_volume = data['volume'].iloc[-1]
    forecasted_loss = await forecast_loss(exchange, symbol, atr, current_spread, current_volume, model, scaler)
    balance_key = 'quote_binance' if exchange_name == 'binance' else 'quote_bingx'
    loss_risk_factor = max(0.5, 1 - forecasted_loss / (BASE_DAILY_LOSS_LIMIT * balances[symbol][balance_key]))
    dynamic_max_position_size = BASE_MAX_POSITION_SIZE * position_size_factor * loss_risk_factor

    total_value = balances[symbol]['base'] * avg_price + balances[symbol][balance_key]
    max_allowed_amount = (total_value * dynamic_max_position_size) / avg_price
    final_amount = min(available_amount, max_allowed_amount)

    spread = (ticker['ask'] - ticker['bid']) / ticker['bid']
    adjustment_factor = min(max(spread * 10, 0.5), 2.0)
    price_adjustment = BASE_PRICE_ADJUSTMENT * adjustment_factor

    depth_factor = min(depth_amount / desired_amount, 1.0) if desired_amount > 0 else 1.0
    final_adjustment = price_adjustment / depth_factor if depth_factor > 0 else price_adjustment
    adjusted_price = avg_price * (1 + final_adjustment) if side == 'buy' else avg_price * (1 - final_adjustment)

    return adjusted_price, final_amount
