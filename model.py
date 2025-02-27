#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from config import SEQUENCE_LENGTH
import logging

def build_gru_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(30, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(30, kernel_regularizer=l2(0.01))))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_gru_model(X, y):
    model = build_gru_model(input_shape=(X.shape[1], X.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    logging.info(f"Модель GRU обучена, средняя точность: {model.evaluate(X, y, verbose=0)[1]:.4f}")
    return model

def train_lstm_model(X, y):
    model = Sequential()
    model.add(Bidirectional(GRU(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(50)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    logging.info(f"Модель LSTM обучена, средняя точность: {model.evaluate(X, y, verbose=0)[1]:.4f}")
    return model
