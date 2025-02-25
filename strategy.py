#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# strategy.py
from config import FIXED_STOP_LOSS, MIN_ORDER_SIZE, TRADING_PAIRS, LOOKBACK
from data import get_historical_data, get_order_book, get_ticker, prepare_lstm_data
from exchange import manage_request, send_telegram_message
from order_management import check_and_cancel_orders
from price_calculator import get_best_price_and_amount
from limits import calculate_optimal_limit
from globals import MAX_OPEN_ORDERS
import logging
import time


async def select_profitable_pairs(exchanges, fees, pred_model, scaler, balances):
    global MAX_OPEN_ORDERS
    MAX_OPEN_ORDERS = await calculate_optimal_limit(balances)

    profitable_pairs = []
    for pair in TRADING_PAIRS:
        try:
            binance_ticker = await get_ticker(exchanges['binance'], pair)
            bingx_ticker = await get_ticker(exchanges['bingx'], pair)

            binance_bid = binance_ticker['bid']
            binance_ask = binance_ticker['ask']
            bingx_bid = bingx_ticker['bid']
            bingx_ask = bingx_ticker['ask']

            spread_buy_binance_sell_bingx = (bingx_ask - binance_bid) / min(binance_bid,
                                                                            bingx_ask) if binance_bid < bingx_ask else 0
            spread_buy_bingx_sell_binance = (binance_ask - bingx_bid) / min(bingx_bid,
                                                                            binance_ask) if bingx_bid < binance_ask else 0
            max_spread = max(spread_buy_binance_sell_bingx, spread_buy_bingx_sell_binance)

            min_spread = fees['binance'].get(pair, {'maker': 0.001})['maker'] + \
                         fees['bingx'].get(pair, {'maker': 0.001})['maker'] + 0.005

            prediction_data = await get_historical_data(exchanges['binance'], pair, limit=LOOKBACK + 100)
            X, _, pred_scaler = prepare_lstm_data(prediction_data)
            prediction = pred_model.predict(X[-1:], verbose=0)[0][0]

            score = max_spread * 100 + prediction

            if max_spread > min_spread or prediction > 0.7:
                profitable_pairs.append((pair, score, max_spread, prediction))
        except Exception as e:
            logging.warning(f"Ошибка при анализе пары {pair}: {str(e)}")
            continue

    profitable_pairs.sort(key=lambda x: x[1], reverse=True)
    selected_pairs = [pair[0] for pair in profitable_pairs[:MAX_OPEN_ORDERS]]

    total_binance = sum(balance['quote_binance'] for balance in balances.values())
    total_bingx = sum(balance['quote_bingx'] for balance in balances.values())

    for pair in TRADING_PAIRS:
        if pair in selected_pairs:
            rank = selected_pairs.index(pair) + 1
            weight = 1 / rank
            total_weight = sum(1 / (i + 1) for i in range(len(selected_pairs)))
            balances[pair]['quote_binance'] = total_binance * (weight / total_weight)
            balances[pair]['quote_bingx'] = total_bingx * (weight / total_weight)
        else:
            balances[pair]['quote_binance'] = 0.0
            balances[pair]['quote_bingx'] = 0.0

    return selected_pairs


async def trade_pair(exchanges, pair, balances, model, scaler, fees, atr, loss_model, loss_scaler, open_orders):
    await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)
    await check_and_cancel_orders(exchanges['bingx'], pair, balances, atr, open_orders)

    binance_order_book = await get_order_book(exchanges['binance'], pair)
    bingx_order_book = await get_order_book(exchanges['bingx'], pair)
    binance_bid, binance_bid_amount = await get_best_price_and_amount(exchanges['binance'], pair, binance_order_book,
                                                                      'buy', 0.1, balances, atr, loss_model,
                                                                      loss_scaler, 'binance')
    binance_ask, binance_ask_amount = await get_best_price_and_amount(exchanges['binance'], pair, binance_order_book,
                                                                      'sell', 0.1, balances, atr, loss_model,
                                                                      loss_scaler, 'binance')
    bingx_bid, bingx_bid_amount = await get_best_price_and_amount(exchanges['bingx'], pair, bingx_order_book, 'buy',
                                                                  0.1, balances, atr, loss_model, loss_scaler, 'bingx')
    bingx_ask, bingx_ask_amount = await get_best_price_and_amount(exchanges['bingx'], pair, bingx_order_book, 'sell',
                                                                  0.1, balances, atr, loss_model, loss_scaler, 'bingx')
    base, quote = pair.split('/')
    balance_base = balances[pair]['base']
    balance_quote_binance = balances[pair]['quote_binance']
    balance_quote_bingx = balances[pair]['quote_bingx']
    entry_price = balances[pair]['entry_price']
    binance_maker_fee = fees['binance'][pair]['maker']
    bingx_maker_fee = fees['bingx'][pair]['maker']

    spread = abs(binance_bid - bingx_ask) / min(binance_bid, bingx_ask) if binance_bid < bingx_ask else abs(
        bingx_bid - binance_ask) / min(bingx_bid, binance_ask)
    min_spread = binance_maker_fee + bingx_maker_fee + 0.005
    if spread > min_spread:
        if binance_bid < bingx_ask and balance_quote_binance > MIN_ORDER_SIZE and balance_quote_bingx > MIN_ORDER_SIZE:
            amount = min(balance_quote_binance / binance_bid, balance_quote_bingx / bingx_ask, 0.1, binance_bid_amount,
                         bingx_ask_amount)
            buy_order = await manage_request(exchanges['binance'], 'create_limit_buy_order', pair, amount, binance_bid)
            sell_order = await manage_request(exchanges['bingx'], 'create_limit_sell_order', pair, amount, bingx_ask)
            open_orders[pair].append({'id': buy_order['id'], 'timestamp': time.time(), 'side': 'buy', 'amount': amount})
            open_orders[pair].append(
                {'id': sell_order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
            msg = f"Арбитраж {pair}: Выставлен ордер на покупку {amount:.4f} на Binance по {binance_bid}, продажу на BingX по {bingx_ask}"
            logging.info(msg)
            await send_telegram_message(msg)
        elif bingx_bid < binance_ask and balance_quote_bingx > MIN_ORDER_SIZE and balance_quote_binance > MIN_ORDER_SIZE:
            amount = min(balance_quote_bingx / bingx_bid, balance_quote_binance / binance_ask, 0.1, bingx_bid_amount,
                         binance_ask_amount)
            buy_order = await manage_request(exchanges['bingx'], 'create_limit_buy_order', pair, amount, bingx_bid)
            sell_order = await manage_request(exchanges['binance'], 'create_limit_sell_order', pair, amount,
                                              binance_ask)
            open_orders[pair].append({'id': buy_order['id'], 'timestamp': time.time(), 'side': 'buy', 'amount': amount})
            open_orders[pair].append(
                {'id': sell_order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
            msg = f"Арбитраж {pair}: Выставлен ордер на покупку {amount:.4f} на BingX по {bingx_bid}, продажу на Binance по {binance_ask}"
            logging.info(msg)
            await send_telegram_message(msg)

    prediction_data = await get_historical_data(exchanges['binance'], pair, limit=LOOKBACK + 100)
    X, _, pred_scaler = prepare_lstm_data(prediction_data)
    prediction = model.predict(X[-1:], verbose=0)[0][0]
    prob = prediction
    prediction = 1 if prediction > 0.5 else 0

    atr_stop_loss = atr * 2
    fixed_stop_loss = entry_price * (1 - FIXED_STOP_LOSS) if entry_price else 0

    if prediction == 1 and prob > 0.7 and balance_quote_binance > MIN_ORDER_SIZE:
        amount = min(balance_quote_binance * 0.1 / binance_bid, balance_quote_binance / binance_bid, binance_bid_amount)
        order = await manage_request(exchanges['binance'], 'create_limit_buy_order', pair, amount, binance_bid)
        open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'buy', 'amount': amount})
        balances[pair]['entry_price'] = binance_bid
        msg = f"{pair}: Выставлен ордер на покупку {amount:.4f} {base} на Binance по {binance_bid}, Уверенность: {prob:.2f}"
        logging.info(msg)
        await send_telegram_message(msg)

    elif balance_base > 0:
        if prediction == 0 and prob > 0.7:
            amount = min(balance_base * 0.1, balance_base, binance_ask_amount)
            order = await manage_request(exchanges['binance'], 'create_limit_sell_order', pair, amount, binance_ask)
            open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
            msg = f"{pair}: Выставлен ордер на продажу {amount:.4f} {base} на Binance по {binance_ask}, Уверенность: {prob:.2f}"
            logging.info(msg)
            await send_telegram_message(msg)
        elif entry_price and (binance_ask < entry_price - atr_stop_loss or binance_ask < fixed_stop_loss):
            stop_reason = "ATR" if binance_ask < entry_price - atr_stop_loss else "Fixed"
            amount = min(balance_base, binance_ask_amount)
            order = await manage_request(exchanges['binance'], 'create_limit_sell_order', pair, amount, binance_ask)
            open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
            msg = f"{pair}: Стоп-лосс ({stop_reason}): Выставлен ордер на продажу {amount:.4f} {base} на Binance по {binance_ask} (Entry: {entry_price:.2f}, ATR: {atr_stop_loss:.2f}, Fixed: {fixed_stop_loss:.2f})"
            logging.info(msg)
            await send_telegram_message(msg)

    from .globals import daily_losses
    print(
        f"{pair}: {balances[pair]['base']:.4f} {base}, Binance: {balances[pair]['quote_binance']:.2f} USDT, BingX: {balances[pair]['quote_bingx']:.2f} USDT, Общие комиссии ${balances[pair]['total_fees']:.2f}, Дневные убытки ${daily_losses[pair]:.2f}")
