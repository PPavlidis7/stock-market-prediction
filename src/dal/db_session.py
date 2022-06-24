from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

_engine = create_engine("postgresql+psycopg2://admin:admin@164.30.0.8:5432/thesis", echo=False, future=True)


def get_session() -> Session:
    Session = sessionmaker(_engine)

    return Session()
