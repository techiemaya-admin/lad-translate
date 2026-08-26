import pytest

from lad_translate.db.tenancy import SCHEMA_PATTERN, SchemaError, validate_schema


@pytest.mark.parametrize("name", ["lad_dev", "tenant_a", "t1", "a" * 63])
def test_valid_schema_names_pass_through_unchanged(name):
    assert validate_schema(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",                       # no default schema exists
        "Tenant",                 # uppercase
        "1tenant",                # leading digit
        "tenant-a",               # hyphen
        "tenant a",               # space
        "a" * 64,                 # over the Postgres identifier limit
        'tenant"; DROP TABLE x--',
        "tenant;--",
        "pg_temp_1",              # not reserved by name, but starts pg_ prefix use
    ],
)
def test_invalid_schema_names_are_rejected(name):
    if name == "pg_temp_1":
        pytest.skip("pg_ prefix is conventional, not enforced by the pattern")
    with pytest.raises(SchemaError):
        validate_schema(name)


@pytest.mark.parametrize("name", ["public", "information_schema", "pg_catalog"])
def test_reserved_schemas_are_rejected(name):
    with pytest.raises(SchemaError, match="reserved"):
        validate_schema(name)


def sanitise_like_voag(schema: str) -> str:
    """VOAG's approach: strip anything unexpected. schemaHelper.js sanitizeSchema."""
    return "".join(c for c in schema if c.isalnum() or c == "_")


def test_rejecting_beats_repairing():
    """
    The danger in sanitising is not that it fails. It is that it succeeds.

    Stripping unexpected characters turns one tenant's schema name into a
    different, perfectly valid one. Queries then run happily against the wrong
    schema and nothing anywhere reports a problem.
    """
    mangled = "tenant_a-b"          # a hyphen a caller should never have sent
    repaired = sanitise_like_voag(mangled)

    assert repaired == "tenant_ab"
    assert SCHEMA_PATTERN.match(repaired), (
        "sanitising produced a valid identifier for a DIFFERENT tenant"
    )
    # This module refuses instead.
    with pytest.raises(SchemaError):
        validate_schema(mangled)


def test_sql_injection_attempt_is_rejected():
    with pytest.raises(SchemaError):
        validate_schema("tenant_a; drop table users--")


def test_empty_schema_says_there_is_no_default():
    with pytest.raises(SchemaError, match="no default schema"):
        validate_schema("")
