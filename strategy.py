#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import FIXED_STOP_LOSS, MIN_ORDER_SIZE, TRADING_PAIRS, LOOKBACK, MAX_PREDICTION, MAX_PROB, MIN_SELL_SIZE, ITERATIONS
from data import get_historical_data, prepare_lstm_data, add_features
from exchange import get_ticker, manage_request, send_telegram_message
from price_calculator import get_best_price_and_amount, get_order_book
from order_management import check_and_cancel_orders
from limits import calculate_optimal_limit
from globals import MAX_OPEN_ORDERS
import logging
import time
import asyncio

async def select_profitable_pairs(exchanges, fees, pred_model, scaler, balances):
    global MAX_OPEN_ORDERS
    MAX_OPEN_ORDERS = await calculate_optimal_limit(balances)

    profitable_pairs = []
    for pair in TRADING_PAIRS:
        try:
            binance_ticker = await get_ticker(exchanges['binance'], pair)
            bingx_ticker = binance_ticker

            binance_bid = binance_ticker['bid']
            binance_ask = binance_ticker['ask']
            bingx_bid = bingx_ticker['bid']
            bingx_ask = bingx_ticker['ask']

            spread_buy_binance_sell_bingx = (bingx_ask - binance_bid) / min(binance_bid, bingx_ask) if binance_bid < bingx_ask else 0
            spread_buy_bingx_sell_binance = (binance_ask - bingx_bid) / min(bingx_bid, binance_ask) if bingx_bid < binance_ask else 0
            max_spread = max(spread_buy_binance_sell_bingx, spread_buy_bingx_sell_binance)

            min_spread = 0.0001

            prediction_data = await get_historical_data(exchanges['binance'], pair, limit=LOOKBACK + 100)
            prediction_data = await add_features(prediction_data)
            if prediction_data.empty:
                logging.error(f"Данные для {pair} пусты после add_features, пропускаем пару")
                continue
            X, _, pred_scaler = prepare_lstm_data(prediction_data)
            if X.size == 0:
                logging.error(f"Подготовленные данные для {pair} пусты, пропускаем пару")
                continue
            prediction = pred_model.predict(X[-1:], verbose=0)[0][0]
            atr = prediction_data['ATR'].iloc[-1]

            score = max_spread * 100 + prediction

            logging.info(
                f"{pair}: max_spread={max_spread:.6f}, min_spread={min_spread:.6f}, prediction={prediction:.6f}, atr={atr:.6f}, score={score:.6f}")

            if max_spread > min_spread or prediction > MAX_PREDICTION:
                profitable_pairs.append((pair, score, max_spread, atr, prediction))
                logging.info(f"{pair} выбрана как прибыльная")
            else:
                logging.info(f"{pair} не выбрана: max_spread <= min_spread и prediction <= {MAX_PREDICTION}")
        except Exception as e:
            logging.error(f"Ошибка при анализе пары {pair}: {str(e)}")
            continue

    profitable_pairs.sort(key=lambda x: x[1], reverse=True)
    selected_pairs = [(pair[0], pair[4], pair[2], pair[3]) for pair in profitable_pairs[:MAX_OPEN_ORDERS]]

    if not selected_pairs:
        logging.info("Нет прибыльных пар, баланс остаётся неизменным")
    else:
        total_binance = max(sum(balance['quote_binance'] for balance in balances.values()), 0)
        allocation_per_pair = total_binance / len(TRADING_PAIRS)
        for pair in TRADING_PAIRS:
            if pair in [p[0] for p in selected_pairs]:
                balances[pair]['quote_binance'] = min(allocation_per_pair, balances[pair]['quote_binance'] + allocation_per_pair)

    logging.info(f"Выбраны пары: {selected_pairs} с лимитом {MAX_OPEN_ORDERS}")
    return selected_pairs

async def trade_pair(exchanges, pair_data, balances, model, scaler, fees, atr, loss_model, loss_scaler, open_orders, trade_fraction, iteration):
    pair, prediction, max_spread, atr = pair_data
    await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)

    binance_order_book = await get_order_book(exchanges['binance'], pair)
    binance_bid, binance_bid_amount = await get_best_price_and_amount(exchanges['binance'], pair, binance_order_book, 'buy', 0.1, balances, atr, loss_model, loss_scaler, 'binance')
    binance_ask, binance_ask_amount = await get_best_price_and_amount(exchanges['binance'], pair, binance_order_book, 'sell', 0.1, balances, atr, loss_model, loss_scaler, 'binance')
    base, quote = pair.split('/')
    balance_quote_binance = balances[pair]['quote_binance']
    entry_price = balances[pair]['entry_price']

    prob = prediction

    logging.info(
        f"{pair}: Проверка условий торговли: prob={prob:.6f}, MAX_PROB={MAX_PROB}, balance_quote_binance={balance_quote_binance:.2f}, MIN_ORDER_SIZE={MIN_ORDER_SIZE}, atr={atr:.6f}")

    atr_stop_loss = atr * 2 if atr else 0.04
    fixed_stop_loss = entry_price * (1 - FIXED_STOP_LOSS) if entry_price else 0

    balance_info = await exchanges['binance'].fetch_balance()
    for p in TRADING_PAIRS:
        base_p = p.split('/')[0]
        available_base = balance_info.get(base_p, {}).get('free', 0)
        balances[p]['base'] = available_base

    total_usdt = balance_info.get('USDT', {}).get('free', 0)
    if iteration != ITERATIONS - 1 and balance_quote_binance > MIN_ORDER_SIZE and total_usdt > MIN_ORDER_SIZE:
        min_notional = 10.0
        amount = max(min_notional / binance_bid, balance_quote_binance * trade_fraction / binance_bid, binance_bid_amount)
        if pair == 'XRP/USDT':
            amount = max(amount, 5.0)
            amount = round(amount, 1)
        elif pair == 'ETH/USDT':
            amount = max(amount, 0.01)
            amount = round(amount, 3)
        elif pair == 'BNB/USDT':
            amount = max(amount, 0.1)
            amount = round(amount, 2)
        elif pair == 'ADA/USDT':
            amount = max(amount, 15.0)
            amount = round(amount, 1)
        elif pair == 'DOGE/USDT':
            amount = max(amount, 100.0)
            amount = round(amount, 0)
        elif pair == 'BTC/USDT':
            amount = max(amount, 0.0005)
            amount = round(amount, 4)
        else:
            amount = round(amount, 2)

        required_balance = amount * binance_bid
        available_quote = balance_info.get('USDT', {}).get('free', 0)
        if required_balance <= available_quote and balance_quote_binance >= required_balance and total_usdt >= required_balance:
            logging.info(
                f"{pair}: Рассчитан amount={amount:.6f} для покупки, bid={binance_bid}, balance_quote_binance={balance_quote_binance}")
            order = await manage_request(exchanges['binance'], 'create_limit_buy_order', pair, amount, binance_bid)
            open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'buy', 'amount': amount})
            balances[pair]['entry_price'] = binance_bid
            balances[pair]['quote_binance'] -= required_balance
            balances[pair]['base'] += amount
            balances[pair]['cost'] = balances[pair].get('cost', 0) + required_balance
            balances[pair]['total_fees'] += required_balance * fees['binance']
            msg = f"{pair}: Выставлен ордер на покупку {amount:.4f} {base} на Binance по {binance_bid}, Уверенность: {prob:.2f}"
            logging.info(msg)
            await send_telegram_message(msg)
        else:
            logging.warning(
                f"{pair}: Недостаточно баланса для покупки: требуется {required_balance:.2f}, доступно {min(balance_quote_binance, available_quote, total_usdt):.2f}")

    await asyncio.sleep(3)
    await check_and_cancel_orders(exchanges['binance'], pair, balances, atr, open_orders)

    balance_info = await exchanges['binance'].fetch_balance()
    available_base = balance_info.get(base, {}).get('free', 0)
    balances[pair]['base'] = available_base
    balance_base = available_base

    if balance_base > 0:
        min_notional_sell = 10.0
        min_amount = min_notional_sell / binance_ask

        min_amounts = {
            'XRP/USDT': 5.0,
            'ADA/USDT': 15.0,
            'BNB/USDT': 0.1,
            'ETH/USDT': 0.01,
            'BTC/USDT': 0.0001,
            'DOGE/USDT': 100.0,
        }
        precisions = {
            'XRP/USDT': 1,
            'ADA/USDT': 1,
            'BNB/USDT': 2,
            'ETH/USDT': 4,
            'BTC/USDT': 4,
            'DOGE/USDT': 0,
        }

        pair_min_amount = min_amounts.get(pair, 0.1)
        pair_precision = precisions.get(pair, 2)

        if iteration == ITERATIONS - 1:
            amount = balance_base
            # Убедимся, что amount не меньше минимальной точности и порога notional
            if amount < pair_min_amount or amount * binance_ask < min_notional_sell:
                amount = max(pair_min_amount, min_notional_sell / binance_ask)
        else:
            calculated_amount = balance_base * trade_fraction
            amount = max(calculated_amount, pair_min_amount)

        amount = round(amount, pair_precision)
        amount = min(amount, balance_base)

        if amount * binance_ask >= min_notional_sell or iteration == ITERATIONS - 1:
            logging.info(
                f"{pair}: Рассчитан amount={amount:.6f} для продажи остатков, ask={binance_ask}, balance_base={balance_base}, available_base={available_base}")
            try:
                order = await manage_request(exchanges['binance'], 'create_market_sell_order', pair, amount)
                open_orders[pair].append({'id': order['id'], 'timestamp': time.time(), 'side': 'sell', 'amount': amount})
                filled_amount = order.get('filled', amount)
                filled_price = order.get('price', binance_ask) or binance_ask
                sold_value = filled_amount * filled_price * (1 - fees['binance'])
                fee = filled_amount * filled_price * fees['binance']
                balances[pair]['quote_binance'] += sold_value
                balances[pair]['total_fees'] += fee
                if iteration == ITERATIONS - 1:
                    balances[pair]['base'] = 0.0
                else:
                    balances[pair]['base'] -= filled_amount
                    balances[pair]['base'] = max(balances[pair]['base'], 0.0)
                balances[pair]['revenue'] = balances[pair].get('revenue', 0) + sold_value
                msg = f"{pair}: Выставлен рыночный ордер на продажу остатков {filled_amount:.4f} {base} по {filled_price:.2f}, получено {sold_value:.2f} USDT, комиссия: {fee:.2f}"
                logging.info(msg)
                await send_telegram_message(msg)
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"{pair}: Ошибка продажи остатков: {str(e)}")
        else:
            logging.info(f"{pair}: Остаток {amount:.6f} слишком мал для продажи (ниже минимального порога {min_notional_sell} USDT)")

    from globals import daily_losses
    print(
        f"{pair}: {balances[pair]['base']:.4f} {base}, Binance: {balances[pair]['quote_binance']:.2f} USDT, BingX: {balances[pair]['quote_bingx']:.2f} USDT")

async def finalize_report(exchanges, balances, initial_balance=None):
    total_usdt = sum(balance['quote_binance'] for balance in balances.values())
    remaining_assets_value = 0

    balance_info = await exchanges['binance'].fetch_balance()
    for pair in TRADING_PAIRS:
        base = pair.split('/')[0]
        available_base = balance_info.get(base, {}).get('free', 0)
        balances[pair]['base'] = available_base
        balance_base = available_base

        if balance_base > 0:
            ticker = await get_ticker(exchanges['binance'], pair)
            price = ticker['ask']
            value = balance_base * price
            remaining_assets_value += value
            logging.info(f"{pair}: Остатки {balance_base:.4f} по цене {price:.2f}, стоимость: {value:.2f} USDT")

    final_balance = total_usdt + remaining_assets_value
    total_fees = sum(balance['total_fees'] for balance in balances.values())
    if initial_balance is None:
        initial_balance = final_balance
    profit_loss = final_balance - initial_balance - total_fees
    logging.info(
        f"Итоговый отчёт: Начальный баланс: {initial_balance:.2f} USDT, Конечный баланс (USDT + активы): {final_balance:.2f} USDT, Комиссии: {total_fees:.2f} USDT, Прибыль/Убыток: {profit_loss:.2f} USDT")
    for pair in TRADING_PAIRS:
        cost = balances[pair].get('cost', 0)
        revenue = balances[pair].get('revenue', 0)
        pair_profit_loss = revenue - cost + (
            balances[pair]['base'] * (await get_ticker(exchanges['binance'], pair))['ask'] if balances[pair]['base'] != 0 else 0) - balances[pair]['total_fees']
        logging.info(
            f"{pair}: Остатки {balances[pair]['base']:.4f}, USDT: {balances[pair]['quote_binance']:.2f}, Прибыль/Убыток: {pair_profit_loss:.2f} USDT")
