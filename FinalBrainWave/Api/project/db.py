import psycopg2
from project.config import Config


import peewee
from contextvars import ContextVar
from fastapi import Depends

settings = Config()

DB_NAME = settings.DB_NAME
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_HOST = settings.DB_HOST
DB_PORT = settings.DB_PORT
DB_SCHEMA = settings.DB_SCHEMA

db_state_default = {"closed": None, "conn": None, "ctx": None, "transactions": None}
db_state = ContextVar("db_state", default=db_state_default.copy())

class PeeweeConnectionState(peewee._ConnectionState):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        self._state.get()[name] = value

    def __getattr__(self, name):
        return self._state.get()[name]


db = peewee.PostgresqlDatabase(
    DB_NAME, 
    user=DB_USER, 
    password=DB_PASS, 
    host=DB_HOST, 
    port=DB_PORT,
    options=f"-c search_path={settings.DB_SCHEMA}"
    )

db._state = PeeweeConnectionState()

async def reset_db_state():
    db._state._state.set(db_state_default.copy())
    db._state.reset()


def get_db(db_state=Depends(reset_db_state)):
    try:
        db.connect()
        yield
    finally:
        if not db.is_closed():
            db.close()


