import time
import logging

import pyodbc
import pandas as pd


_logger = logging.getLogger(__name__)


def _in_clause(tickers: list[str]) -> str:
    return ''.join(f'\'{ticker}\',' for ticker in tickers)[:-1]


def _open_connection() -> pyodbc.Connection:

    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};\
                    SERVER=10.0.0.7\\sql2014,52332;\
                    DATABASE=deepinvest;\
                    UID=deepinvest;\
                    PWD=deep1nv3$t!;"
    )


def _fetch_market_data(
        query: str,
        tickers: list[str]
) -> dict[str, pd.DataFrame]:
    datasets = {}

    if tickers:
        columns = ['Ticker', 'PriceDate', 'ClosePrice', 'Open', 'MinPrice', 'MaxPrice']

        if 'Volume' in query:
            columns.append('Volume')

        connection = _open_connection()
        cursor = connection.cursor()

        st = time.perf_counter()
        rows = cursor.execute(query).fetchall()

        _logger.info(f'Fetched {len(rows)} rows in {round(time.perf_counter() - st, 4)} sec.')

        if rows:
            market_data = pd.DataFrame([tuple(row) for row in rows])
            market_data.columns = columns
            market_data.index = market_data['PriceDate']
            market_data = market_data.drop(['PriceDate'], axis=1)

            datasets = {ticker: market_data[market_data['Ticker'] == ticker].iloc[:, 1:] for ticker in tickers}

        cursor.close()
        connection.close()

    return datasets


def execute_custom_query(query: str) -> pd.DataFrame:
    connection = _open_connection()
    cursor = connection.cursor()

    st = time.perf_counter()

    result = pd.DataFrame([tuple(row) for row in cursor.execute(query).fetchall()])

    _logger.info(f'Fetched {len(result)} rows in {round(time.perf_counter() - st, 4)} sec.')

    cursor.close()
    connection.close()

    return result


def fetch_all_assets() -> pd.DataFrame:
    query = 'select * from DeepInvest.dbo.V_Stock;'

    assets = execute_custom_query(query)
    assets.index = assets[0]
    assets = assets.drop([0], axis=1)
    assets.columns = ['Ticker', 'TickerEN', 'Isin', 'TitleName', 'Country', 'Currency', 'XCode',
                      'StockExchangeName', 'StartDate', 'EndDate', 'ProductType']

    return assets


def fetch_assets_data(
        assets: list[str],
        start: str,
        end: str
) -> dict[str, pd.DataFrame]:
    query = f'''select Ticker, PriceDate, ClosePrice, [Open], MinPrice, MaxPrice, Volume
                from DeepInvest.dbo.V_Prices
                where Ticker in ({_in_clause(assets)})
                and PriceDate between '{start}' and '{end}';'''

    return _fetch_market_data(query=query, tickers=assets)
