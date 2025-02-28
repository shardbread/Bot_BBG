#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import FIXED_STOP_LOSS, MIN_ORDER_SIZE, TRADING_PAIRS, LOOKBACK, MAX_PREDICTION, MAX_PROB
from data import get_historical_data, prepare_lstm_data, add_features
from exchange import get_ticker, manage_request, send_telegram_message
from order_management import check_and_cancel_orders
from price_calculator import get_best_price_and_amount, get_order_book
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
            bingx_ticker = binance_ticker  # Заглушка для теста

            binance_bid = binance_ticker['bid']
            binance_ask = binance_ticker['ask']
            bingx_bid = bingx_ticker['bid']
            bingx_ask = bingx_ticker['ask']

            spread_buy_binance_sell_bingx = (bingx_ask - binance_bid) / min(binance_bid,
                                                                            bingx_ask) if binance_bid < bingx_ask else 0
            spread_buy_bingx_sell_binance = (binance_ask - bingx_bid) / min(bingx_bid,
                                                                            binance_ask) if bingx_bid < binance_ask else 0
            max_spread = max(spread_buy_binance_sell_bingx, spread_buy_bingx_sell_binance)

            min_spread = 0.0001

            prediction_data = await get_historical_data(exchanges['binance'], pair, limit=LOOKBACK + 100)
            prediction_data = add_features(prediction_data)
            X, _, pred_scaler = prepare_lstm_data(prediction_data)
            prediction = pred_model.predict(X[-1:], verbose=0)[0][0]

            score = max_spread * 100 + prediction

            logging.info(
                f"{pair}: max_spread={max_spread:.6f}, min_spread={min_spread:.6f}, prediction={prediction:.6f}, score={score:.6f}")

            if max_spread > min_spread or prediction > MAX_PREDICTION:
                profitable_pairs.append((pair, score, max_spread, prediction))
                logging.info(f"{pair} выбрана как прибыльная")
            else:
                logging.info(f"{pair} не выбрана: max_spread <= min_spread и prediction <= {MAX_PREDICTION}")
        except Exception as e:
            logging.error(f"Ошибка при анализе пары {pair}: {str(e)}")
            continue

    profitable_pairs.sort(key=lambda x: x[1], reverse=True)
    selected_pairs = [(pair[0], pair[3]) for pair in profitable_pairs[:MAX_OPEN_ORDERS]]

    if not selected_pairs:
        logging.info("Нет прибыльных пар, баланс остаётся неизменным")
    else:
        total_binance = max(sum(balance['quote_binance'] for balance in balances.values()), 0)
        total_bingx = sum(balance['quote_bingx'] for balance in balances.values())
        allocation_per_pair = total_binance / len(selected_pairs) if total_binance > 0 else 0
        for pair in TRADING_PAIRS:
            if pair in [p[0] for p in selected_pairs]:
                balances[pair]['quote_binance'] = min(allocation_per_pair, total_binance)
                balances[pair]['quote_bingx'] = total_bingx / len(selected_pairs)
            else:
                balances[pair]['quote_binance'] = 0.0
                balances[pair]['quote_bingx'] = 0.0

    logging.info(f"Выбраны пары: {selected_pairs} с лимитом {MAX_OPEN_ORDERS}")
    return selected_pairs


async def trade_pair(exchanges, pair_data, balances, model, scaler, fees, atr, loss_model, loss_scaler, open_orders):
    pair, prediction = pair_data
    await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)

    binance_order_book = await get_order_book(exchanges['binance'], pair)
    binance_bid, binance_bid_amount = await get_best_price_and_amount(exchanges['binance'], pair, binance_order_book,
                                                                      'buy', 0.1, balances, atr, loss_model,
                                                                      loss_scaler, 'binance')
    binance_ask, binance_ask_amount = await get_best_price_and_amount(exchanges['binance'], pair, binance_order_book,
                                                                      'sell', 0.1, balances, atr, loss_model,
                                                                      loss_scaler, 'binance')
    base, quote = pair.split('/')
    balance_base = balances[pair]['base']
    balance_quote_binance = balances[pair]['quote_binance']
    entry_price = balances[pair]['entry_price']

    prob = prediction

    logging.info(
        f"{pair}: Проверка условий торговли: prob={prob:.6f}, MAX_PROB={MAX_PROB}, balance_quote_binance={balance_quote_binance:.2f}, MIN_ORDER_SIZE={MIN_ORDER_SIZE}")

    atr_stop_loss = atr * 2
    fixed_stop_loss = entry_price * (1 - FIXED_STOP_LOSS) if entry_price else 0

    if balance_quote_binance > MIN_ORDER_SIZE:
        min_notional = 10.0
        amount = max(min_notional / binance_bid, balance_quote_binance * 0.1 / binance_bid,
                     binance_bid_amount)  # Уменьшено до 10%
        if pair == 'XRP/USDT':
            amount = max(amount, 5.0)
        elif pair == 'ETH/USDT':
            amount = max(amount, 0.01)
        elif pair == 'BNB/USDT':
            amount = max(amount, 0.1)
        elif pair == 'ADA/USDT':
            amount = max(amount, 10.0)
        elif pair == 'DOGE/USDT':
            amount = max(amount, 100.0)
        elif pair == 'BTC/USDT':
            amount = max(amount, 0.001)

        required_balance = amount * binance_bid
        if balance_quote_binance > required_balance:
            logging.info(
                f"{pair}: Рассчитан amount={amount:.6f} для покупки, bid={binance_bid}, balance_quote_binance={balance_quote_binance}")
            order = await manage_request(exchanges['binance'], 'create_limit_buy_order', pair, amount, binance_bid)
            open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'buy', 'amount': amount})
            balances[pair]['entry_price'] = binance_bid
            balances[pair]['quote_binance'] -= amount * binance_bid
            balances[pair]['base'] += amount
            msg = f"{pair}: Выставлен ордер на покупку {amount:.4f} {base} на Binance по {binance_bid}, Уверенность: {prob:.2f}"
            logging.info(msg)
            await send_telegram_message(msg)
        else:
            logging.warning(
                f"{pair}: Недостаточно баланса для покупки: требуется {required_balance:.2f}, доступно {balance_quote_binance:.2f}")

    elif balance_base > 0:
        if prob < MAX_PROB:
            amount = min(balance_base * 1.0, balance_base, binance_ask_amount)
            logging.info(
                f"{pair}: Рассчитан amount={amount:.6f} для продажи, ask={binance_ask}, balance_base={balance_base}")
            order = await manage_request(exchanges['binance'], 'create_limit_sell_order', pair, amount, binance_ask)
            open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
            balances[pair]['quote_binance'] += amount * binance_ask
            balances[pair]['base'] -= amount
            msg = f"{pair}: Выставлен ордер на продажу {amount:.4f} {base} на Binance по {binance_ask}, Уверенность: {prob:.2f}"
            logging.info(msg)
            await send_telegram_message(msg)
        elif entry_price and (binance_ask < entry_price - atr_stop_loss or binance_ask < fixed_stop_loss):
            stop_reason = "ATR" if binance_ask < entry_price - atr_stop_loss else "Fixed"
            amount = min(balance_base, binance_ask_amount)
            order = await manage_request(exchanges['binance'], 'create_limit_sell_order', pair, amount, binance_ask)
            open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
            balances[pair]['quote_binance'] += amount * binance_ask
            balances[pair]['base'] -= amount
            msg = f"{pair}: Стоп-лосс ({stop_reason}): Выставлен ордер на продажу {amount:.4f} {base} на Binance по {binance_ask} (Entry: {entry_price:.2f}, ATR: {atr_stop_loss:.2f}, Fixed: {fixed_stop_loss:.2f})"
            logging.info(msg)
            await send_telegram_message(msg)

    from globals import daily_losses
    print(
        f"{pair}: {balances[pair]['base']:.4f} {base}, Binance: {balances[pair]['quote_binance']:.2f} USDT, BingX: {balances[pair]['quote_bingx']:.2f} USDT, Общие комиссии ${balances[pair]['total_fees']:.2f}, Дневные убытки ${daily_losses[pair]:.2f}")
