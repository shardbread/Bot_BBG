#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from sklearn.preprocessing import MinMaxScaler
import logging
from data import get_historical_data, prepare_lstm_data, add_features
from config import LOOKBACK

async def train_models(exchange, pair):
    try:
        # Получение исторических данных с биржи
        historical_data = await get_historical_data(exchange, pair, limit=LOOKBACK + 100)
        if historical_data is None or historical_data.empty:
            logging.error(f"Не удалось получить исторические данные для {pair}")
            return None, None

        # Добавление признаков
        data_with_features = await add_features(historical_data)
        if data_with_features is None or data_with_features.empty:
            logging.error(f"Не удалось добавить признаки для {pair}")
            return None, None

        # Подготовка данных для LSTM
        X, y, scaler = prepare_lstm_data(data_with_features)
        if X.size == 0 or y.size == 0:
            logging.error(f"Подготовленные данные для {pair} пусты: X={X.shape}, y={y.shape}")
            return None, None

        logging.info(f"Подготовлены данные для LSTM: X.shape={X.shape}, y.mean={np.mean(y):.4f}")

        # Построение LSTM модели
        lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
        lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        lstm_accuracy = evaluate_model(lstm_model, X, y)
        logging.info(f"Модель LSTM обучена, средняя точность: {lstm_accuracy:.4f}")

        # Построение GRU модели
        gru_model = build_gru_model((X.shape[1], X.shape[2]))
        gru_model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        gru_accuracy = evaluate_model(gru_model, X, y)
        logging.info(f"Модель GRU обучена, средняя точность: {gru_accuracy:.4f}")

        # Выбор модели с лучшей точностью
        pred_model = lstm_model if lstm_accuracy > gru_accuracy else gru_model
        return pred_model, scaler

    except Exception as e:
        logging.error(f"Ошибка в train_models: {str(e)}")
        return None, None

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(GRU(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X, verbose=0)
    accuracy = np.mean((predictions.flatten() > 0.5) == y)
    return accuracy
