#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed

from preprocessing.metrics import calc_error_metrics

_RANDOM_SEED = 23
_WORKDAYS_PER_WEEK = 5
_BATCH_SIZE = 32
_EPOCHS = 100
_VERBOSE = 2

set_seed(_RANDOM_SEED)


class GridSearchBacktester:

    def __init__(
            self,
            retrain_window: int,
            prediction_window: int
    ):
        self._rertain_window = retrain_window
        self._prediction_window = prediction_window

    def run(
            self,
            retrain_subsets: dict[str, dict],
            model: Sequential
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        weights = model.get_weights()

        predicted_returns = {}
        metrics = {}
        previous_tickers = None

        for date, retrain_subset in retrain_subsets.items():
            test = retrain_subset['test']
            y_true = retrain_subset['y_true']

            if not y_true.columns.equals(previous_tickers):
                model.set_weights(weights)

            previous_tickers = y_true.columns

            model.fit(
                retrain_subset['X_train'],
                retrain_subset['y_train'],
                batch_size=_BATCH_SIZE,
                epochs=_EPOCHS,
                verbose=_VERBOSE)

            subset_predictions = pd.DataFrame(
                data=np.concatenate([retrain_subset['labels_scaler'].inverse_transform(
                    model.predict(test[testing_date]['X_test'], batch_size=_BATCH_SIZE).reshape(1, -1))
                    for testing_date in test.keys()], axis=0),
                index=list(y_true.index)[_WORKDAYS_PER_WEEK * self._prediction_window:],
                columns=y_true.columns)

            subset_predictions[subset_predictions < 0] = 0

            index = y_true.iloc[:_WORKDAYS_PER_WEEK * self._rertain_window].index

            predicted_returns[date] = subset_predictions.set_index(index) / y_true.loc[index] - 1
            metrics[date] = calc_error_metrics(
                y_true=y_true.iloc[_WORKDAYS_PER_WEEK * self._prediction_window:],
                y_pred=subset_predictions)

        return metrics, predicted_returns
