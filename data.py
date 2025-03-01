#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import asyncio


async def get_historical_data(exchange, pair, timeframe='1h', limit=1000):
    try:
        ohlcv = await exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Получены исторические данные для {pair}: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при получении данных для {pair}: {str(e)}")
        return pd.DataFrame()


async def add_features(df):
    try:
        # Средние скользящие
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        # ATR (Average True Range)
        df['TR'] = df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)),
                          abs(x['low'] - x['close'].shift(1))), axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        # Удаляем NaN
        df = df.dropna()
        logging.info(f"Добавлены функции к данным: {df.columns.tolist()}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при добавлении функций: {str(e)}")
        return df


def prepare_lstm_data(df, lookback=60):
    try:
        features = ['MA10', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR']
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_data = scaler.fit_transform(df[features])

        X, y = [], []
        for i in range(lookback, len(X_data)):
            X.append(X_data[i - lookback:i])
            y.append(1 if df['close'].iloc[i] > df['close'].iloc[i - 1] else 0)

        X = np.array(X)
        y = np.array(y)
        logging.info(f"Подготовлены данные для LSTM: X.shape={X.shape}, y.mean={y.mean():.4f}")
        return X, y, scaler
    except Exception as e:
        logging.error(f"Ошибка при подготовке данных для LSTM: {str(e)}")
        return np.array([]), np.array([]), None
