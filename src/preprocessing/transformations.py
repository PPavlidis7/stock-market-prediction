import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_DROP_NA_THRESHOLD_PERCENTAGE = 0.98
_WORKDAYS_PER_WEEK = 5


def calc_assets_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    returns = close_prices.pct_change().iloc[1:].dropna(
        axis=1, thresh=(close_prices.shape[0] - 1) * _DROP_NA_THRESHOLD_PERCENTAGE)

    returns[pd.isna(returns)] = 0
    returns = returns.loc[:, (returns != 0).any(axis=0)]

    return returns


def calc_retrain_dates(
        dates: list[str],
        start: str,
        end: str,
        retrain_window: int,
) -> list[str]:
    return [dates[i] for i in range(dates.index(start), dates.index(end), _WORKDAYS_PER_WEEK * retrain_window)]


def fit_features_datasets_scalers(features_datasets: dict[str, pd.DataFrame]) -> dict[str, StandardScaler]:
    fitted_scalers = {}

    for ticker, features_dataset in features_datasets.items():
        numerical_features = features_dataset.iloc[:, :16]

        fitted_scalers[ticker] = StandardScaler().fit(numerical_features)

    return fitted_scalers


def fit_labels_scaler(labels: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(labels)


def scale_features_datasets(
        features_datasets: dict[str, pd.DataFrame],
        fitted_scalers: dict[str, StandardScaler]
) -> dict[str, pd.DataFrame]:
    scaled_features_datasets = {}

    for ticker, features_dataset in features_datasets.items():
        numerical_features = features_dataset.iloc[:, :16]
        weekday_features = features_dataset.iloc[:, 16:]

        scaled_features_datasets[ticker] = pd.DataFrame(
            data=np.concatenate((fitted_scalers[ticker].transform(numerical_features), weekday_features), axis=1),
            index=features_dataset.index,
            columns=features_dataset.columns)

    return scaled_features_datasets


def scale_labels(
        labels: pd.DataFrame,
        fitted_scaler: StandardScaler
) -> pd.DataFrame:
    return pd.DataFrame(
        data=fitted_scaler.transform(labels),
        index=labels.index,
        columns=labels.columns)


def calc_retrain_subset(
        features_datasets: dict[str, pd.DataFrame],
        labels: pd.DataFrame,
        retrain_date: str,
        sample_days: int,
        retrain_window: int,
        prediction_window: int,
        n_timesteps: int,
        timesteps_step: int
) -> dict[str, dict]:
    retrain_subset = {}

    dates = list(labels.index)
    retrain_date_index = dates.index(retrain_date)

    training_samples = {}
    training_samples_dates = dates[retrain_date_index - _WORKDAYS_PER_WEEK * prediction_window - sample_days + 1:
                                   retrain_date_index - _WORKDAYS_PER_WEEK * prediction_window + 1]

    for training_sample_date in training_samples_dates:
        training_samples[training_sample_date] = {
            'X_train': _calc_features_timesteps(features_datasets, training_sample_date, n_timesteps, timesteps_step),
            'y_train': _calc_labels(labels, training_sample_date, prediction_window)
        }

    retrain_subset['training_samples'] = training_samples

    testing_samples = {}
    testing_dates = dates[retrain_date_index + 1:retrain_date_index + _WORKDAYS_PER_WEEK * retrain_window + 1]

    for testing_date in testing_dates:
        testing_samples[testing_date] = {
            'X_test': _calc_features_timesteps(features_datasets, testing_date, n_timesteps, timesteps_step),
            'y_test': _calc_labels(labels, testing_date, prediction_window)
        }

    retrain_subset['testing_samples'] = testing_samples

    return retrain_subset


def reshape_retrain_subset(retrain_subset: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, dict[str, dict]]:
    X_train = []
    y_train = []

    for sample in retrain_subset['training_samples'].values():
        for features_timesteps in sample['X_train'].values():
            X_train.append(features_timesteps.to_numpy())

        y_train.extend(sample['y_train'].to_numpy().reshape(-1))

    test = {}

    for date, sample in retrain_subset['testing_samples'].items():
        X_test = []
        y_test = []

        for features_timesteps in sample['X_test'].values():
            X_test.append(features_timesteps)

        y_test.extend(sample['y_test'].to_numpy().reshape(-1))

        test[date] = {
            'X_test': np.asarray(X_test),
            'y_test': np.asarray(y_test)
        }

    return np.asarray(X_train), np.asarray(y_train), test


def _calc_features_timesteps(
        features_datasets: dict[str, pd.DataFrame],
        date: str,
        n_timesteps: int,
        step: int
) -> dict[str, pd.DataFrame]:
    features_timesteps = {}

    for ticker, features in features_datasets.items():
        first_timestep_index = features.index.get_loc(date)

        features_timesteps[ticker] = features.iloc[
            np.arange(
                start=first_timestep_index - step * n_timesteps + step,
                stop=first_timestep_index + step,
                step=step)]

    return features_timesteps


def _calc_labels(
        labels: pd.DataFrame,
        date: str,
        prediction_window: int
) -> pd.DataFrame:
    return pd.DataFrame(
        data={ticker: [close.iloc[close.index.get_loc(date) + _WORKDAYS_PER_WEEK * prediction_window]]
              for ticker, close in labels.items()},
        index=[labels.index[labels.index.get_loc(date) + _WORKDAYS_PER_WEEK * prediction_window]])
