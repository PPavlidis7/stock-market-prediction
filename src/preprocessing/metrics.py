#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from tensorflow.keras.losses import huber


def calc_error_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame
) -> pd.DataFrame:
    metrics = {ticker: {
        'mse': mean_squared_error(y_true[ticker], y_pred[ticker]),
        'mae': mean_absolute_error(y_true[ticker], y_pred[ticker]),
        'mape': mean_absolute_percentage_error(y_true[ticker], y_pred[ticker]),
        'huber': huber(y_true[ticker], y_pred[ticker]).numpy()
    } for ticker in y_pred}

    return pd.DataFrame(metrics).transpose()


def aggregate_metrics(metrics: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    concatenated_metrics = pd.concat([period_metrics.mean() for period_metrics in metrics.values()])

    return {
        'aggregate_means': concatenated_metrics.groupby(concatenated_metrics.index).mean(),
        'aggregate_stds': concatenated_metrics.groupby(concatenated_metrics.index).std()
    }
