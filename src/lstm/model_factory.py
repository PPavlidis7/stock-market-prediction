#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import (
    MeanSquaredError,
    Huber
)
from tensorflow.random import set_seed


_RANDOM_SEED = 23
_ACTIVATION_FUNCTION = 'relu'
_METRICS = ['mse', 'mae', 'mape']
_OPTIMIZER = Adam()
_LOSSES = {
    'mse': MeanSquaredError(),
    'huber': Huber()
}

set_seed(_RANDOM_SEED)


def build_model(
    layers: int,
    neurons: int,
    loss: str,
    dropout: float,
    n_timesteps: int,
    n_features: int
) -> Sequential:
    return _build_lstm_plain(layers, neurons, loss, dropout, n_timesteps, n_features)


def _build_lstm_plain(
    layers: int,
    neurons: int,
    loss: str,
    dropout: float,
    n_timesteps: int,
    n_features: int
) -> Sequential:
    model = Sequential()

    model.add(LSTM(
        units = neurons,
        activation = _ACTIVATION_FUNCTION,
        dropout = dropout,
        return_sequences = layers > 1,
        input_shape = (n_timesteps, n_features)))

    for i in range(layers - 1):
        model.add(LSTM(
            units = neurons,
            activation = _ACTIVATION_FUNCTION,
            dropout = dropout,
            return_sequences = True if i != layers - 2 else False))

    model.add(Dense(units = 1, activation = 'linear'))

    model.compile(optimizer = _OPTIMIZER, loss = _LOSSES[loss], metrics = _METRICS)

    return model
