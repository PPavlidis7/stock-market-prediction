#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from data_fetching import fetch_all_assets, fetch_assets_data
from transformations import calc_assets_returns


def write_tickers_date(us_close_prices):
    for ticker, data in us_close_prices.items():
        if len(data) > 90:
            data.to_csv(f'../resources/datasets_2/{ticker}')


def main():
    start = '1991-01-01'
    end = '2002-12-30'

    assets = fetch_all_assets()
    available_stocks = assets[(assets['ProductType'] == 'S')]

    us_datasets = fetch_assets_data(
        assets=available_stocks[available_stocks['Currency'] == 'USD']['Ticker'].to_list(),
        start=start,
        end=end)
    write_tickers_date(us_datasets)

    us_close_prices = pd.DataFrame(
        data={ticker: us_datasets[ticker]['ClosePrice'] for ticker in us_datasets.keys() if
              len(us_datasets[ticker]) > 90},
        dtype=float
    ).dropna(how='all', axis=1)

    with open('../resources/tickers.txt', 'w') as f:
        f.write("\n".join(list(us_close_prices.columns)))


if __name__ == '__main__':
    main()
