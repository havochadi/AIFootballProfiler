import sqlite3

import pytest

from football_profiler import storage as S


def test_database_context_commits_and_closes_connection():
    with S.db() as connection:
        connection.execute("INSERT INTO manual_events(dataset_id,player_id,kind) VALUES('test','p1','pass')")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    assert len(S.manual_events("test", "p1")) == 1


def test_failed_database_context_rolls_back_and_closes_connection():
    with pytest.raises(ValueError, match="test rollback"):
        with S.db() as connection:
            connection.execute("INSERT INTO manual_events(dataset_id,player_id,kind) VALUES('test','p1','pass')")
            raise ValueError("test rollback")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
    assert S.manual_events("test", "p1") == []
