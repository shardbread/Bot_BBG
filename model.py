#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from sklearn.model_selection import KFold
import numpy as np
import logging

def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(input_shape):
    model = Sequential([
        Bidirectional(GRU(100, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(GRU(50)),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X, y, model_name):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                           callbacks=[early_stopping], verbose=0)
        val_accuracy = max(history.history['val_accuracy'])
        accuracies.append(val_accuracy)
    mean_accuracy = np.mean(accuracies)
    logging.info(f"Модель {model_name} обучена, средняя точность: {mean_accuracy:.4f}")
    return model

def train_models(X, y):
    lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
    gru_model = build_gru_model((X.shape[1], X.shape[2]))
    lstm_model = train_model(lstm_model, X, y, "LSTM")
    gru_model = train_model(gru_model, X, y, "GRU")
    return lstm_model, gru_model
