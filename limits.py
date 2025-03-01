#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging


async def calculate_optimal_limit(balances):
    try:
        total_binance = sum(balance['quote_binance'] for balance in balances.values())
        total_bingx = sum(balance['quote_bingx'] for balance in balances.values())
        limit = max(1, int(min(total_binance, total_bingx) / 1000))
        logging.info(f"Рассчитан оптимальный лимит пар: {limit} (Binance: {total_binance:.2f}, BingX: {total_bingx:.2f})")
        return limit
    except Exception as e:
        logging.error(f"Ошибка при расчёте лимита: {str(e)}")
        return 1  # Минимальный лимит по умолчанию
