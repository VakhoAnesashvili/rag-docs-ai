import psycopg
from pathlib import Path
from pgvector.psycopg import register_vector

from app.config import settings


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema() -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(sql)
        conn.commit()