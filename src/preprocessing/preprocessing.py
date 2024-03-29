#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa

import datetime as dt
import multiprocessing as mp
import pickle
from itertools import repeat

import pandas as pd

from data_fetching import fetch_all_assets
from feature_building import build_features_datasets
from transformations import calc_retrain_dates, fit_features_datasets_scalers, fit_labels_scaler, \
    scale_features_datasets, scale_labels, calc_retrain_subset, reshape_retrain_subset

sample_days = 5
retrain_window = 8
prediction_window = 4
n_timesteps = 8
timesteps_step = 1

assets = fetch_all_assets()

dates = list(pd.bdate_range(start='1991-01-01', end='2002-12-30').strftime('%Y-%m-%d'))
retrain_dates = calc_retrain_dates(dates=dates, start='1991-06-28', end='2001-06-28', retrain_window=retrain_window)


def main(retrain_date, retrain_subsets):
    start = (dt.datetime.strptime(retrain_date, '%Y-%m-%d') - dt.timedelta(days=90)).strftime('%Y-%m-%d')
    end = retrain_date

    retrain_tickers = assets[(assets['ProductType'] == 'S') & (assets['Currency'] == 'USD') & (
            assets['StartDate'] <= start) & ((pd.isnull(assets['EndDate'])) | (assets['EndDate'] >= end))][
        'Ticker'].to_list()

    retrain_date_index = dates.index(retrain_date)

    start = dates[retrain_date_index - n_timesteps - 5 * prediction_window - sample_days]
    start_2 = dates[retrain_date_index - n_timesteps - 5 * prediction_window - sample_days - 39]

    end = dates[retrain_date_index + 5 * retrain_window + 5 * prediction_window]
    end_2 = dates[retrain_date_index + 5 * retrain_window + 5 * prediction_window + 10]

    datasets = {}
    for ticker in retrain_tickers:
        try:
            datasets[ticker] = pd.read_csv(f'../resources/datasets_2/{ticker}', index_col=0).loc[start_2:end_2]
        except:
            print(ticker)


    features_datasets, labels = build_features_datasets(datasets=datasets)
    features_datasets = {ticker: features_dataset.loc[start:end] for ticker, features_dataset in
                         features_datasets.items()}
    labels = labels.loc[start:end]

    known_features_datasets = {ticker: features_datasets[ticker].loc[:retrain_date] for ticker in
                               features_datasets.keys()}
    unknown_features_datasets = {ticker: features_datasets[ticker].loc[retrain_date:].iloc[1:] for ticker in
                                 features_datasets.keys()}

    known_labels = labels.loc[:retrain_date]
    unknown_labels = labels.loc[retrain_date:].iloc[1:]

    features_datasets_scalers = fit_features_datasets_scalers(features_datasets=known_features_datasets)
    labels_scaler = fit_labels_scaler(labels=known_labels)

    scaled_known_features_datasets = scale_features_datasets(features_datasets=known_features_datasets,
                                                             fitted_scalers=features_datasets_scalers)
    scaled_known_labels = scale_labels(labels=known_labels, fitted_scaler=labels_scaler)

    scaled_unknown_features_datasets = scale_features_datasets(features_datasets=unknown_features_datasets,
                                                               fitted_scalers=features_datasets_scalers)
    scaled_unknown_labels = scale_labels(labels=unknown_labels, fitted_scaler=labels_scaler)

    merged_features_datasets = {
        ticker: pd.concat((scaled_known_features_datasets[ticker], scaled_unknown_features_datasets[ticker])) for ticker
        in features_datasets.keys()}
    merged_labels = pd.concat((scaled_known_labels, scaled_unknown_labels))

    retrain_subset = calc_retrain_subset(
        features_datasets=merged_features_datasets,
        labels=merged_labels,
        retrain_date=retrain_date,
        sample_days=sample_days,
        retrain_window=retrain_window,
        prediction_window=prediction_window,
        n_timesteps=n_timesteps,
        timesteps_step=timesteps_step)

    X_train, y_train, test = reshape_retrain_subset(retrain_subset)

    true_start = labels.index.get_loc(list(test.keys())[0])
    true_end = true_start + 5 * retrain_window * 2

    y_true = labels.iloc[true_start:true_end]
    y_true.columns = retrain_tickers

    retrain_subsets[retrain_date] = {
        'X_train': X_train,
        'y_train': y_train,
        'y_true': y_true,
        'test': test,
        'labels_scaler': labels_scaler
    }


def run_parallel():
    with mp.Manager() as manager:
        d = manager.dict()
        with manager.Pool() as pool:
            pool.starmap(main, zip(retrain_dates, repeat(d)))
        # `d` is a DictProxy object that can be converted to dict
        retrain_subsets = dict(d)
        print(retrain_subsets)

    sum = []
    for key, value in retrain_subsets.items(): sum += (value['size'])
    print(len(set(sum)))
    x = 1
    # with open(f'../resources/S{sample_days}-T{n_timesteps}', 'wb') as f:
    #     pickle.dump(retrain_subsets, f)


if __name__ == '__main__':
    run_parallel()
