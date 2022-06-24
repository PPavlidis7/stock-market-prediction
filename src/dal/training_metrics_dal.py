from sqlalchemy.future import select
from sqlalchemy.orm.session import Session

from .training_metrics_model import TrainingMetrics
from .training_metrics_schema import TrainingMetricsCreate


class TrainingMetricsDAL:

    def __init__(
        self,
        db_session: Session
    ):
        self._db_session = db_session

    def get_all(self) -> list[TrainingMetrics]:
        with self._db_session as session:
            q = session.execute(select(TrainingMetrics).order_by(TrainingMetrics.id))

            return q.scalars().all()

    def create(
        self,
        new_training_metrics: TrainingMetricsCreate
    ) -> TrainingMetrics:
        with self._db_session as session:
            new_record = TrainingMetrics(**new_training_metrics.dict())

            session.add(new_record)

            session.commit()
            session.refresh(new_record)

            return new_record
