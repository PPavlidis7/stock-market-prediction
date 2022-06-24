#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import pickle
import argparse
import multiprocessing as mp
from itertools import product, repeat

from dal.training_metrics_dal import TrainingMetricsDAL
from dal.db_session import get_session
from lstm.backtester import GridSearchBacktester
from preprocessing.metrics import aggregate_metrics
from lstm.model_factory import build_model
from dal.training_metrics_schema import build_training_metrics_schema

_WORKDAYS_PER_WEEK = 5
_CONFIGS_DIRECTORY_PATH = 'resources/backtesting/configs'
_COVARIANCE_PATH = 'resources/backtesting/covariance'
_DATASETS_PATH = 'resources/datasets'
_BENCHMARK_TO_COUNTRY = {
    'GSPC': 'us',
    'ΓΔ': 'gr'
}

training_metrics_dal = TrainingMetricsDAL(db_session=get_session())


def generate_grid_combinations(grid: dict[str, list]) -> list[dict]:
    hyperparameter, values = zip(*grid.items())

    return [dict(zip(hyperparameter, v)) for v in product(*values)]


def _validate_config_name(config_name: str) -> str:
    if not os.path.exists(f'{_CONFIGS_DIRECTORY_PATH}/{config_name}'):
        raise argparse.ArgumentTypeError('Config file not found.')

    return config_name


def train_model(model_grid_point, _features, _sample_days, backtester, training_metrics_schemas_list):
    features = _features[0]
    sample_days = _sample_days[0]
    model = build_model(
        layers=model_grid_point['l'],
        neurons=model_grid_point['n'],
        loss=model_grid_point['loss'],
        dropout=model_grid_point['d'],
        n_timesteps=model_grid_point['t'],
        n_features=features)

    with open(f'resources/S{sample_days}-T8', 'rb') as f:
        retrain_subsets = pickle.load(f)

    model_grid_point_str = ''.join(f'{key}={value},' for key, value in model_grid_point.items())[:-1]

    # print(model_grid_point_str)

    training_start = time.perf_counter()
    metrics, predicted_returns = backtester.run(
        retrain_subsets=retrain_subsets,
        model=model)

    with open('logging.txt', 'a') as f:
        f.write(
            f'Model grid point {model_grid_point_str} completed in {round(time.perf_counter() - training_start, 4)} secs.\n')

    training_metrics_schema = build_training_metrics_schema(
        model_grid_point=model_grid_point,
        training_metrics=aggregate_metrics(metrics=metrics))

    training_metrics_schemas_list.append(training_metrics_schema)


def main():
    parser = argparse.ArgumentParser(description='Runs grid search backtesting and stores metrics in database.')
    parser.add_argument('--config_name', action='store', type=_validate_config_name, required=True)
    args = vars(parser.parse_args())

    with open(f"lstm/{args['config_name']}", 'r') as f:
        backtesting_configuration = json.load(f)

    retrain_window = backtesting_configuration['retrain_window']
    prediction_window = backtesting_configuration['prediction_window']
    sample_days = backtesting_configuration['sample_days']
    features = backtesting_configuration['features']

    model_grids = backtesting_configuration['model_grids']

    backtester = GridSearchBacktester(
        prediction_window=prediction_window,
        retrain_window=retrain_window)

    models_grid_points = generate_grid_combinations(grid=model_grids)

    with mp.Manager() as manager:
        training_metrics_schemas_list = manager.list()

        with manager.Pool() as pool:
            pool.starmap(
                train_model,
                zip(
                    models_grid_points,
                    repeat([features]),
                    repeat([sample_days]),
                    repeat(backtester),
                    repeat(training_metrics_schemas_list)
                )
            )

        training_metrics_schemas_list = list(training_metrics_schemas_list)

    for training_metrics_schema in training_metrics_schemas_list:
        training_metrics_dal.create(new_training_metrics=training_metrics_schema)


if __name__ == '__main__':
    main()
