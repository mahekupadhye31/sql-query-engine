import pytest

from sqlengine.ast import WhereClause
from sqlengine.parser import parse


def test_select_star_no_where():
    stmt = parse("SELECT * FROM users")
    assert stmt.columns == ["*"]
    assert stmt.table == "users"
    assert stmt.where is None


def test_select_columns():
    stmt = parse("SELECT id, name FROM users")
    assert stmt.columns == ["id", "name"]


def test_where_equals_int():
    stmt = parse("SELECT * FROM users WHERE id = 42")
    assert stmt.where == WhereClause("id", "=", 42)


def test_where_equals_string():
    stmt = parse("SELECT * FROM users WHERE name = 'alice'")
    assert stmt.where == WhereClause("name", "=", "alice")


@pytest.mark.parametrize("op,expected", [
    ("!=", "!="), ("<", "<"), ("<=", "<="), (">", ">"), (">=", ">="),
])
def test_where_operators(op, expected):
    stmt = parse(f"SELECT * FROM t WHERE x {op} 5")
    assert stmt.where.op == expected


def test_negative_number():
    stmt = parse("SELECT * FROM t WHERE x = -5")
    assert stmt.where.value == -5


def test_syntax_error_missing_from():
    with pytest.raises(SyntaxError):
        parse("SELECT * users")


def test_syntax_error_bad_operator():
    with pytest.raises(SyntaxError):
        parse("SELECT * FROM t WHERE x ~ 5")
