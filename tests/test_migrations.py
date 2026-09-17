"""
The migration manifests and the migrations directory have to agree.

db/migrate.py applies an explicit TUPLE, not whatever is on disk, which is the
right call - a directory listing decides execution order by filename and picks
up half-written files. The cost is that adding a .sql and forgetting to
register it produces no error anywhere: the table simply never exists, and the
first anyone knows is a 42P01 from a console panel at a venue.
"""

from __future__ import annotations

from lad_translate.db.migrate import (
    CONTROL_MIGRATIONS,
    MIGRATIONS_DIR,
    TENANT_MIGRATIONS,
)


def test_every_migration_on_disk_is_registered():
    on_disk = {p.name for p in MIGRATIONS_DIR.glob("*.sql")}
    registered = set(CONTROL_MIGRATIONS) | set(TENANT_MIGRATIONS)
    missing = sorted(on_disk - registered)
    assert not missing, (
        f"{missing} exist but no manifest applies them, so those tables will "
        "never be created. Add them to CONTROL_MIGRATIONS or TENANT_MIGRATIONS."
    )


def test_every_registered_migration_exists_on_disk():
    on_disk = {p.name for p in MIGRATIONS_DIR.glob("*.sql")}
    registered = set(CONTROL_MIGRATIONS) | set(TENANT_MIGRATIONS)
    missing = sorted(registered - on_disk)
    assert not missing, f"{missing} are applied but the files are gone"


def test_a_migration_is_registered_exactly_once():
    both = list(CONTROL_MIGRATIONS) + list(TENANT_MIGRATIONS)
    assert len(both) == len(set(both)), "a migration is in both manifests"


def test_the_manifests_are_in_filename_order():
    """
    The ledger records what ran, not what order it ran in, so a manifest out
    of order applies 004 before 003 on a fresh schema and the failure is a
    missing column rather than anything that names the cause.
    """
    assert list(TENANT_MIGRATIONS) == sorted(TENANT_MIGRATIONS)
    assert list(CONTROL_MIGRATIONS) == sorted(CONTROL_MIGRATIONS)
