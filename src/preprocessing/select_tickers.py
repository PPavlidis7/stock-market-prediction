#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt
import pandas as pd

from data_fetching import fetch_all_assets, fetch_assets_data
from transformations import calc_retrain_dates, calc_assets_returns


def main():
    retrain_window = 8
    dates = list(pd.bdate_range(start='1991-04-01', end='2002-12-30').strftime('%Y-%m-%d'))
    retrain_dates = calc_retrain_dates(dates=dates, start='1991-06-28', end='2001-06-28', retrain_window=retrain_window)
    tickers = set()
    for _date in retrain_dates:
        date = dt.datetime.strptime(_date, '%Y-%m-%d')
        start = (date - dt.timedelta(days=90)).strftime('%Y-%m-%d')
        end = date.strftime('%Y-%m-%d')

        assets = fetch_all_assets()
        available_stocks = assets[(assets['ProductType'] == 'S') & (assets['StartDate'] <= start) & (
                    (pd.isnull(assets['EndDate'])) | (assets['EndDate'] >= end))]

        us_datasets = fetch_assets_data(
            assets=available_stocks[available_stocks['Currency'] == 'USD']['Ticker'].to_list(),
            start=start,
            end=end)

        us_close_prices = pd.DataFrame(
            data={ticker: us_datasets[ticker]['ClosePrice'] for ticker in us_datasets.keys()},
            dtype=float
        ).dropna(how='all', axis=1)

        returns = calc_assets_returns(close_prices=us_close_prices)
        tickers.update(returns.columns)
        print("Finished date: {}.   Done: {} ". format(date, (1/len(retrain_dates)) * 100, ))

    with open('tickers.txt', 'w') as f:
        f.write("\n".join(tickers))


if __name__ == '__main__':
    main()
