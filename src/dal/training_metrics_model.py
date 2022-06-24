import enum

from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.types import Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class LossFunction(enum.Enum):
    mse = 'mse'
    huber = 'huber'


class TrainingMetrics(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    layers = Column(Integer)
    neurons = Column(Integer)
    timesteps = Column(Integer)
    dropout = Column(Float)
    loss_function = Column(Enum(LossFunction))

    mse = Column(Float)
    mae = Column(Float)
    huber = Column(Float)
    mape = Column(Float)

    mse_std = Column(Float)
    mae_std = Column(Float)
    huber_std = Column(Float)
    mape_std = Column(Float)
