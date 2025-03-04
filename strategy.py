#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import MIN_ORDER_SIZE, TRADING_PAIRS, LOOKBACK, MAX_PREDICTION, MAX_PROB, MIN_SELL_SIZE
from data import get_historical_data, prepare_lstm_data, add_features
from exchange import send_telegram_message
from limits import calculate_optimal_limit
import logging
import asyncio


async def select_profitable_pairs(exchanges, fees, pred_model, scaler, balances):
    global MAX_OPEN_ORDERS
    MAX_OPEN_ORDERS = await calculate_optimal_limit(balances)

    profitable_pairs = []
    min_atr = 0.0005  # Добавлен фильтр по ATR
    for pair in TRADING_PAIRS:
        try:
            binance_ticker = await exchanges['binance'].fetch_ticker(pair)
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

            logging.info(f"{pair}: max_spread={max_spread:.6f}, min_spread={min_spread:.6f}, prediction={prediction:.6f}, atr={atr:.6f}, score={score:.6f}")

            if (max_spread > min_spread or prediction > MAX_PREDICTION) and atr > min_atr:
                profitable_pairs.append((pair, score, max_spread, atr, prediction))
                logging.info(f"{pair} выбрана как прибыльная")
            else:
                logging.info(f"{pair} не выбрана: max_spread <= min_spread, prediction <= {MAX_PREDICTION}, или atr <= {min_atr}")
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


async def trade_pair(exchanges, pair, pred_model, scaler, balances, iteration):
    try:
        exchange_binance = exchanges['binance']
        ticker = await exchange_binance.fetch_ticker(pair)
        bid, ask = ticker['bid'], ticker['ask']

        # Получаем реальный баланс с биржи и синхронизируем
        binance_balance = await exchange_binance.fetch_balance()
        usdt_free = float(binance_balance['free'].get('USDT', 0))
        # Обновляем доступный баланс для пары, если он больше реального
        if balances[pair]['quote_binance'] > usdt_free:
            balances[pair]['quote_binance'] = usdt_free

        # Предсказание
        historical_data = await get_historical_data(exchange_binance, pair, limit=LOOKBACK + 100)
        data_with_features = await add_features(historical_data)
        X, _, scaler = prepare_lstm_data(data_with_features)
        prediction = pred_model.predict(X[-1:], verbose=0)[0][0]
        logging.info(f"Итерация {iteration}: Предсказание для {pair}: {prediction}")

        # Логика покупки
        if prediction > 0.5 and balances[pair]['quote_binance'] > 0:
            amount = balances[pair]['quote_binance'] / bid
            cost = amount * bid
            fee = cost * 0.001  # Комиссия Binance
            total_cost = cost + fee
            logging.info(
                f"Попытка покупки {pair}: amount={amount}, cost={cost}, fee={fee}, total_cost={total_cost}, usdt_free={usdt_free}")
            if usdt_free >= total_cost:
                order = await exchange_binance.create_limit_buy_order(pair, amount, bid)
                balances[pair]['quote_binance'] -= total_cost
                balances[pair]['base'] += amount
                balances[pair]['cost'] += cost
                balances[pair]['total_fees'] += fee
                balances[pair]['entry_price'] = bid
                logging.info(f"Куплено {amount} {pair} по {bid}, стоимость: {cost}, комиссия: {fee}")
            else:
                logging.warning(
                    f"Недостаточно средств для покупки {pair}: требуется {total_cost}, доступно {usdt_free}")
        else:
            logging.info(
                f"Покупка {pair} не выполнена: prediction={prediction} <= 0.5 или quote_binance={balances[pair]['quote_binance']} <= 0")

        # Логика продажи
        if prediction < 0.4 and balances[pair]['base'] > 0:
            amount = balances[pair]['base']
            order = await exchange_binance.create_limit_sell_order(pair, amount, ask)
            revenue = amount * ask
            fee = revenue * 0.001  # Комиссия Binance
            balances[pair]['quote_binance'] += revenue - fee
            balances[pair]['base'] = 0
            balances[pair]['revenue'] += revenue
            balances[pair]['total_fees'] += fee
            balances[pair]['entry_price'] = 0
            logging.info(f"Продано {amount} {pair} по {ask}, выручка: {revenue}, комиссия: {fee}")

    except Exception as e:
        logging.error(f"Ошибка в trade_pair для {pair}: {str(e)}")


async def finalize_report(exchanges, balances, initial_total_usdt):
    exchange_binance = exchanges['binance']
    total_usdt = 0
    total_fees = 0

    logging.info("Финализация остатков и создание отчёта")
    for pair in balances:
        amount = balances[pair]['base']
        if amount > 0:  # Проверяем, что есть что продавать
            ticker = await exchange_binance.fetch_ticker(pair)
            ask = ticker['ask']
            if ask:
                order = await exchange_binance.create_limit_sell_order(pair, amount, ask)
                balances[pair]['revenue'] += amount * ask
                balances[pair]['base'] = 0
                logging.info(f"Проданы все остатки {amount} {pair} по {ask}")
        total_usdt += balances[pair]['quote_binance'] + balances[pair]['revenue'] - balances[pair]['cost']
        total_fees += balances[pair]['total_fees']

    profit_loss = total_usdt - initial_total_usdt
    roi = (profit_loss / initial_total_usdt) * 100 if initial_total_usdt > 0 else 0

    logging.info(f"Итоговый отчёт: Начальный баланс: {initial_total_usdt:.2f} USDT, Конечный баланс: {total_usdt:.2f} USDT, Комиссии: {total_fees:.2f} USDT, Прибыль/Убыток: {profit_loss:.2f} USDT (ROI: {roi:.2f}%)")
    for pair in balances:
        pl = balances[pair]['revenue'] - balances[pair]['cost']
        logging.info(f"{pair}: Остатки {balances[pair]['base']:.4f}, USDT: {balances[pair]['quote_binance']:.2f}, Прибыль/Убыток: {pl:.2f} USDT")
