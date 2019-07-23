"""
Unit tests for the backend connection arguments.
"""
from contextlib import suppress
import pytest

from databases import Database
from databases.backends.mysql import MySQLBackend
from databases.backends.postgres import PostgresBackend
from tests.test_databases import POSTGRES_URLS, async_adapter


def test_postgres_pool_size():
    backend = PostgresBackend("postgres://localhost/database?min_size=1&max_size=20")
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"min_size": 1, "max_size": 20}


def test_postgres_explicit_pool_size():
    backend = PostgresBackend("postgres://localhost/database", min_size=1, max_size=20)
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"min_size": 1, "max_size": 20}


def test_postgres_ssl():
    backend = PostgresBackend("postgres://localhost/database?ssl=true")
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"ssl": True}


def test_postgres_explicit_ssl():
    backend = PostgresBackend("postgres://localhost/database", ssl=True)
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"ssl": True}


urls_with_options = [
    (f"{POSTGRES_URLS[0]}?min_size=1&max_size=20", suppress()),
    (f"{POSTGRES_URLS[0]}?min_size=0&max_size=0", pytest.raises(ValueError)),
    (f"{POSTGRES_URLS[0]}?min_size=10&max_size=0", pytest.raises(ValueError)),
]
@pytest.mark.parametrize("database_url, expectation", urls_with_options)
@async_adapter
async def test_postgres_pool_size_connect(database_url, expectation):
    with expectation:
        async with Database(database_url) as db:
            db.connect()


def test_mysql_pool_size():
    backend = MySQLBackend("mysql://localhost/database?min_size=1&max_size=20")
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"minsize": 1, "maxsize": 20}


def test_mysql_explicit_pool_size():
    backend = MySQLBackend("mysql://localhost/database", min_size=1, max_size=20)
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"minsize": 1, "maxsize": 20}


def test_mysql_ssl():
    backend = MySQLBackend("mysql://localhost/database?ssl=true")
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"ssl": True}


def test_mysql_explicit_ssl():
    backend = MySQLBackend("mysql://localhost/database", ssl=True)
    kwargs = backend._get_connection_kwargs()
    assert kwargs == {"ssl": True}
