#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import LOOKBACK, SEQUENCE_LENGTH


async def get_historical_data(exchange, symbol, timeframe='5m', limit=LOOKBACK + SEQUENCE_LENGTH):
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


async def get_ticker(exchange, symbol):
    return await exchange.fetch_ticker(symbol)


async def get_order_book(exchange, symbol, limit=10):
    return await exchange.fetch_order_book(symbol, limit=limit)


def add_features(df):
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
    df['TR'] = pd.concat([df['high'] - df['low'],
                          (df['high'] - df['close'].shift()).abs(),
                          (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['Target'] = (df['close'].shift(-1) > df['close'] * 1.005).astype(int)
    return df.dropna()


def prepare_lstm_data(df, lookback=60):
    scaler = MinMaxScaler()
    features = ['MA10', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'close', 'ATR']
    X_data = scaler.fit_transform(df[features])
    y_data = df['Target'].values
    X, y = [], []
    for i in range(lookback, len(X_data)):
        X.append(X_data[i - lookback:i])
        y.append(y_data[i])
    return np.array(X), np.array(y), scaler


def prepare_gru_data(df, historical_losses, lookback=LOOKBACK):
    df = add_features(df)
    atr_values = df['ATR'].values[-lookback - 10:]
    rsi_values = df['RSI'].values[-lookback - 10:]
    macd_values = df['MACD'].values[-lookback - 10:]
    volume_values = df['volume'].values[-lookback - 10:]
    bb_upper_values = df['BB_upper'].values[-lookback - 10:]
    bb_lower_values = df['BB_lower'].values[-lookback - 10:]
    bb_width_values = (df['BB_upper'] - df['BB_lower']).values[-lookback - 10:]
    spreads = [df['close'].iloc[-1] * 0.001] * (lookback + 10)

    X = np.column_stack([atr_values, spreads, volume_values, rsi_values, macd_values, bb_upper_values, bb_lower_values,
                         bb_width_values])
    y = np.array(historical_losses[-lookback - 10:])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_gru, y_gru = [], []
    for i in range(10, len(X_scaled)):
        X_gru.append(X_scaled[i - 10:i])
        y_gru.append(y[i])
    return np.array(X_gru), np.array(y_gru), scaler
