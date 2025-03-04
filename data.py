#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from config import LOOKBACK


async def get_historical_data(exchange, symbol, timeframe='1m', limit=LOOKBACK + 100):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Получены исторические данные для {symbol}: {df.shape}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при получении данных для {symbol}: {str(e)}")
        return pd.DataFrame()

async def add_features(df):
    try:
        df = df.copy()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['TR'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        # Новые признаки
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean()
        df['Volatility'] = df['close'].rolling(window=14).std()
        df.dropna(inplace=True)
        logging.info(f"Добавлены признаки: {df.shape}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при добавлении признаков: {str(e)}")
        return df

def prepare_lstm_data(df):
    try:
        features = ['close', 'volume', 'MA10', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'ATR', 'Volume_MA10', 'Volatility']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])
        X, y = [], []
        for i in range(LOOKBACK, len(scaled_data)):
            X.append(scaled_data[i-LOOKBACK:i])
            y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)
        X = np.array(X)
        y = np.array(y)
        logging.info(f"Подготовлены данные для LSTM: X.shape={X.shape}, y.mean={y.mean():.4f}")
        return X, y, scaler
    except Exception as e:
        logging.error(f"Ошибка при подготовке данных для LSTM: {str(e)}")
        return np.array([]), np.array([]), None
