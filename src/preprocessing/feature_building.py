#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import ta

_POLYNOMIALS_DEGREE = 4


def build_features_datasets(datasets: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start, end = _get_dates_range(datasets)
    dates = list(pd.bdate_range(start=start, end=end).strftime('%Y-%m-%d'))
    filled_datasets = {ticker: dataset.reindex(dates, method='ffill').fillna(method='bfill')
                       for ticker, dataset in datasets.items()}

    features_datasets = {ticker: pd.concat(
        objs=[
            _raw_OHLCV(dataset, ticker).iloc[33:],
            _polynomials(dataset, ticker, degree=_POLYNOMIALS_DEGREE).iloc[33:],
            _technical_indicators(dataset, ticker).iloc[33:]],
        axis=1) for ticker, dataset in filled_datasets.items()}

    labels = pd.DataFrame({f'{ticker}_Close': dataset['ClosePrice']
                           for ticker, dataset in filled_datasets.items()}).iloc[33:]

    return features_datasets, labels


def _get_dates_range(datasets: dict[str, pd.DataFrame]) -> tuple[str, str]:
    start = min([str(dataset.index[0]) for dataset in datasets.values()])
    end = max([str(dataset.index[-1]) for dataset in datasets.values()])

    return start, end


def _raw_OHLCV(
        dataset: pd.DataFrame,
        ticker: str
) -> pd.DataFrame:
    return pd.DataFrame({
        f'{ticker}_Open': dataset['Open'],
        f'{ticker}_High': dataset['MaxPrice'],
        f'{ticker}_Low': dataset['MinPrice'],
        f'{ticker}_Close': dataset['ClosePrice'],
        f'{ticker}_Volume': dataset['Volume']})


def _polynomials(
        dataset: pd.DataFrame,
        ticker: str,
        degree: int
) -> pd.DataFrame:
    return pd.DataFrame({f'{ticker}_poly_{d}': dataset['ClosePrice'] ** d for d in range(2, degree + 1)})


def _technical_indicators(
        dataset: pd.DataFrame,
        ticker: str
) -> pd.DataFrame:
    close = dataset['ClosePrice']
    high = dataset['MaxPrice']
    low = dataset['MinPrice']

    return pd.DataFrame({
        f'{ticker}_RSI': ta.momentum.RSIIndicator(close).rsi(),
        f'{ticker}_MACD': ta.trend.MACD(close).macd_diff(),
        f'{ticker}_Williams_%R': ta.momentum.WilliamsRIndicator(high, low, close).williams_r(),
        f'{ticker}_Stoc_Osc': ta.momentum.StochasticOscillator(high, low, close).stoch(),
        f'{ticker}_ROC': ta.momentum.ROCIndicator(close, window=1).roc(),
        f'{ticker}_Bol_Bands': ta.volatility.BollingerBands(close).bollinger_mavg(),
        f'{ticker}_Par_SAR': ta.trend.PSARIndicator(high, low, close).psar(),
        f'{ticker}_ADX': ta.trend.ADXIndicator(high, low, close).adx()})
