#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from binance.client import Client
import logging
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Ваши API-ключ и секрет для Testnet
API_KEY = "CGohejYAHO9aXXi8LwSkr4gv9PvGmnOjOZMolMo9ZWsIP76qx9BUbrkSWlIRtxvI"
API_SECRET = "8xvh9f7cG4WOQSmnNkmBHz9aaV9Zu1U3ZPbYYLe0f1oin3I5sxYE2kQg6LBD9Mmm"

# Настройка клиента для Binance Testnet
client = Client(API_KEY, API_SECRET, testnet=True)

def check_balance():
    """Проверка баланса на Testnet."""
    try:
        account = client.get_account()
        balances = {asset['asset']: float(asset['free']) for asset in account['balances'] if float(asset['free']) > 0}
        logging.info(f"Текущий баланс: {balances}")
        return balances
    except Exception as e:
        logging.error(f"Ошибка при проверке баланса: {str(e)}")
        return None

def replenish_balance():
    """Сообщение о необходимости пополнения баланса."""
    logging.warning("Баланс USDT недостаточен для торговли!")
    logging.info("На Binance Spot Testnet нет прямого API для пополнения.")
    logging.info("1. Создайте новый API-ключ на https://testnet.binance.vision/")
    logging.info("2. Новый ключ автоматически получит начальный баланс (например, 10000 BUSD).")
    logging.info("3. Обновите API_KEY и API_SECRET в скрипте.")

def main():
    balances = check_balance()
    if balances:
        usdt_balance = balances.get('USDT', 0.0)
        if usdt_balance < 10.0:  # MIN_ORDER_SIZE из config.py
            replenish_balance()
        else:
            logging.info(f"Баланс USDT достаточен: {usdt_balance:.2f}")
    else:
        replenish_balance()

if __name__ == "__main__":
    main()
