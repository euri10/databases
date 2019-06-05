import asyncio
import functools
import logging
import os

import pytest
import sqlalchemy
from sqlalchemy import select, Integer
from sqlalchemy.dialects.postgresql import JSONB

from databases import Database, DatabaseURL

assert "TEST_DATABASE_URLS" in os.environ, "TEST_DATABASE_URLS is not set."

DATABASE_URLS = [url.strip() for url in os.environ["TEST_DATABASE_URLS"].split(",")]

POSTGRES_ONLY = [url for url in DATABASE_URLS if "postgres" in url]

logger = logging.getLogger("databases")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

metadata = sqlalchemy.MetaData()

jsonitems = sqlalchemy.Table(
    "jsonitems", metadata, sqlalchemy.Column("metadata", JSONB())
)


@pytest.fixture(autouse=True, scope="module")
def create_test_database():
    # Create test databases
    for url in DATABASE_URLS:
        database_url = DatabaseURL(url)
        if database_url.dialect == "mysql":
            url = str(database_url.replace(driver="pymysql"))
        engine = sqlalchemy.create_engine(url)
        metadata.create_all(engine)

    # Run the test suite
    yield

    # Drop test databases
    for url in DATABASE_URLS:
        database_url = DatabaseURL(url)
        if database_url.dialect == "mysql":
            url = str(database_url.replace(driver="pymysql"))
        engine = sqlalchemy.create_engine(url)
        metadata.drop_all(engine)


def async_adapter(wrapped_func):
    """
    Decorator used to run async test cases.
    """

    @functools.wraps(wrapped_func)
    def run_sync(*args, **kwargs):
        loop = asyncio.get_event_loop()
        task = wrapped_func(*args, **kwargs)
        return loop.run_until_complete(task)

    return run_sync


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_json_index_operations_operand_int(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSON-OP-TABLE

        ->	int
        Get JSON array element (indexed from zero, negative integers count from the end)
        '[{"a":"foo"},{"b":"bar"},{"c":"baz"}]'::json->2
        {"c":"baz"}

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON

        Index operations (the -> operator):
            data_table.c.data['some key']
            data_table.c.data[5]
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                [{"a": "foo"}, {"b": "bar"}, {"c": "baz"}]
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata[2]])
            r = await database.fetch_one(query)
            assert r["anon_1"] == {"c": "baz"}


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_json_index_operations_operand_text(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSON-OP-TABLE

        ->	text
        Get JSON object field by key
        '{"a": {"b":"foo"}}'::json->'a'
        {"b":"foo"}

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON

        Index operations (the -> operator):
            data_table.c.data['some key']
            data_table.c.data[5]
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"a": {"b": "foo"}}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata['a']])
            r = await database.fetch_one(query)
            assert r["anon_1"] == {"b": "foo"}


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_json_index_operations_operand_int_returning_text(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSON-OP-TABLE

        ->>	int
        Get JSON array element as text
        '[1,2,3]'::json->>2
        3

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON

        Index operations with CAST (equivalent to CAST(col ->> ['some key'] AS <type>)):
        data_table.c.data['some key'].astext.cast(Integer) == 5

    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"somekey": 12345}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata["somekey"].cast(Integer)])
            r = await database.fetch_one(query)
            assert len(r) == 1
            # should return text but ayncpg casts
            assert r["anon_1"] == 12345


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_json_index_operations_operand_text_returning_text(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSON-OP-TABLE

        ->>	text
        Get JSON object field as text
        '{"a":1,"b":2}'::json->>'b'
        2

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON

        Index operations with CAST (equivalent to CAST(col ->> ['some key'] AS <type>)):
        data_table.c.data['some key'].astext.cast(Integer) == 5

    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"a": 1, "b": 2}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata["b"].astext])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == "2"


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_json_pathindex_operations_operand_textarray(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSON-OP-TABLE

        #>	text[]
        Get JSON object at specified path
        '{"a": {"b":{"c": "foo"}}}'::json#>'{a,b}'
        {"c": "foo"}


    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON

        Path index operations (the #> operator):
        data_table.c.data[('key_1', 'key_2', 5, ..., 'key_n')]

    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"a": {"b": {"c": "foo"}}}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata[('a', 'b')]])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == {"c": "foo"}


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_json_pathindex_operations_operand_textarray_returning_text(
        database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSON-OP-TABLE

        #>>	text[]
        Get JSON object at specified path as text
        '{"a":[1,2,3],"b":[4,5,6]}'::json#>>'{a,2}'
        3

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSON

        data_table.c.data[('key_1', 'key_2', 5, ..., 'key_n')].astext == 'some value'

    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"a": [1, 2, 3], "b": [4, 5, 6]}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata[('a', '2')].astext])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == "3"


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_contains(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        @>
        jsonb
        Does the left JSON value contain the right JSON path/value entries at the top level?
        '{"a":1, "b":2}'::jsonb @> '{"b":2}'::jsonb

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSONB.Comparator.contains

    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"a": 1, "b": 2}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata.contains({"b": 2})])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == True
            query = select([jsonitems.c.metadata.contains({"a": 4})])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == False


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_contained_by(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        <@
        jsonb
        Are the left JSON path/value entries contained at the top level within the right JSON value?
        '{"b":2}'::jsonb <@ '{"a":1, "b":2}'::jsonb

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSONB.Comparator.contained_by


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"b": 2}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata.contained_by({"a":1, "b":2})])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == True
            query = select([jsonitems.c.metadata.contained_by({"a": 1, "b": 3})])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == False


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_has_key(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        ?
        text
        Does the string exist as a top-level key within the JSON value?
        '{"a":1, "b":2}'::jsonb ? 'b'

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSONB.Comparator.has_key


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            inserts = [
                {"a":1, "b":2}
            ]
            query = jsonitems.insert()
            r = await database.execute_many(
                query, values=[{"metadata": insert} for insert in inserts]
            )
            query = select([jsonitems])
            r = await database.fetch_all(query)
            assert len(r) == 1
            query = select([jsonitems.c.metadata.has_key("b")])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == True
            query = select([jsonitems.c.metadata.has_key("c")])
            r = await database.fetch_one(query)
            assert len(r) == 1
            assert r["anon_1"] == False


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_has_any(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        ?|
        text[]
        Do any of these array strings exist as top-level keys?
        '{"a":1, "b":2, "c":3}'::jsonb ?| array['b', 'c']

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSONB.Comparator.has_any


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_has_all(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        ?&
        text[]
        Do all of these array strings exist as top-level keys?
        '["a", "b"]'::jsonb ?& array['a', 'b']

    From https://docs.sqlalchemy.org/en/13/dialects/postgresql.html#sqlalchemy.dialects.postgresql.JSONB.Comparator.has_all


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_concatenate(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        ||
        jsonb
        Concatenate two jsonb values into a new jsonb value
        '["a", "b"]'::jsonb || '["c", "d"]'::jsonb

    From ??


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_delete_key_operand_text(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        -
        text
        Delete key/value pair or string element from left operand. Key/value pairs are matched based on their key value.
        '{"a": "b"}'::jsonb - 'a'

    From ??


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_delete_key_operand_text_array(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        -
        text[]
        Delete multiple key/value pairs or string elements from left operand. Key/value pairs are matched based on their key value.
        '{"a": "b", "c": "d"}'::jsonb - '{a,c}'::text[]

    From ??


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_delete_key_operand_integer(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        -
        integer
        Delete the array element with specified index (Negative integers count from the end). Throws an error if top level container is not an array.
        '["a", "b"]'::jsonb - 1

    From ??


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass


@pytest.mark.parametrize("database_url", POSTGRES_ONLY)
@async_adapter
async def test_database_jsonb_delete_path_operand_text_array(database_url):
    """
    From https://www.postgresql.org/docs/11/functions-json.html#FUNCTIONS-JSONB-OP-TABLE

        #-
        text[]
        Delete the field or element with specified path (for JSON arrays, negative integers count from the end)
        '["a", {"b":1}]'::jsonb #- '{1,b}'

    From ??


    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            pass
