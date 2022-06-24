from enum import Enum

from pydantic import BaseModel


class LossFunction(str, Enum):
    mse = 'mse'
    huber = 'huber'


class TrainingMetricsBase(BaseModel):
    layers: int
    neurons: int
    timesteps: int
    dropout: float
    loss_function: LossFunction

    mse: float
    mae: float
    huber: float
    mape: float

    mse_std: float
    mae_std: float
    huber_std: float
    mape_std: float


class TrainingMetricsCreate(TrainingMetricsBase):
    pass


class TrainingMetrics(TrainingMetricsBase):
    id: int

    class Config:
        orm_mode = True
        use_enum_values = True
        arbitrary_types_allowed = True


def build_training_metrics_schema(
    model_grid_point: dict,
    training_metrics: dict
) -> TrainingMetricsCreate:
    loss_functions = {
        'mse': LossFunction.mse,
        'huber': LossFunction.huber
    }

    return TrainingMetricsCreate(
        layers = model_grid_point['l'],
        neurons = model_grid_point['n'],
        timesteps = model_grid_point['t'],
        dropout = model_grid_point['d'],
        loss_function = loss_functions[model_grid_point['loss']],
        mse = training_metrics['aggregate_means']['mse'],
        mae = training_metrics['aggregate_means']['mae'],
        huber = training_metrics['aggregate_means']['huber'],
        mape = training_metrics['aggregate_means']['mape'],
        mse_std = training_metrics['aggregate_stds']['mse'],
        mae_std = training_metrics['aggregate_stds']['mae'],
        huber_std = training_metrics['aggregate_stds']['huber'],
        mape_std = training_metrics['aggregate_stds']['mape'])