#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import MinMaxScaler
import logging

async def get_historical_data(exchange, symbol, timeframe='5m', limit=100):
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def compute_rsi(close, window=14):
    rsi = RSIIndicator(close=close, window=window)
    return rsi.rsi()

def compute_macd(close, window_slow=26, window_fast=12, window_sign=9):
    macd = MACD(close=close, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()

def compute_bollinger_bands(close, window=20, window_dev=2):
    bb = BollingerBands(close=close, window=window, window_dev=window_dev)
    return bb.bollinger_hband(), bb.bollinger_lband()

def compute_atr(df, window=14):
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
    return atr.average_true_range()

def add_features(df):
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['close'])
    macd, macd_signal, _ = compute_macd(df['close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    bb_upper, bb_lower = compute_bollinger_bands(df['close'])
    df['BB_upper'] = bb_upper
    df['BB_lower'] = bb_lower
    df['ATR'] = compute_atr(df)
    df['Target'] = (df['close'].shift(-1) > df['close'] * 1.005).astype(int)  # Рост > 0.5%
    return df.dropna()

def prepare_lstm_data(df, lookback=60):
    scaler = MinMaxScaler()
    features = ['MA10', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'close', 'ATR']
    X_data = scaler.fit_transform(df[features])
    y_data = df['Target'].values
    X, y = [], []
    for i in range(lookback, len(X_data)):
        X.append(X_data[i-lookback:i])
        y.append(y_data[i])
    X, y = np.array(X), np.array(y)
    logging.info(f"Подготовлены данные для LSTM: X.shape={X.shape}, y.mean={y.mean():.4f}")
    return X, y, scaler

def prepare_gru_data(df, lookback=38):
    scaler = MinMaxScaler()
    features = ['MA10', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'close', 'ATR']
    X_data = scaler.fit_transform(df[features])
    y_data = df['Target'].values
    X, y = [], []
    for i in range(lookback, len(X_data)):
        X.append(X_data[i-lookback:i])
        y.append(y_data[i])
    X, y = np.array(X), np.array(y)
    logging.info(f"Подготовлены данные для GRU: X.shape={X.shape}, y.mean={y.mean():.4f}")
    return X, y, scaler
