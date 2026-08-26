-- Control plane. Shared across tenants, holds the tenant directory only.
--
-- In VOAG this schema also holds every tenant's operational data, with
-- tenant_id as the only separation. Session data here goes in per-tenant
-- schemas instead; this schema stays small and holds the mapping.

CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.tenants (
    id           uuid PRIMARY KEY,
    slug         text NOT NULL UNIQUE,
    schema_name  text NOT NULL UNIQUE,
    display_name text,
    is_active    boolean NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),

    -- Enforced in the database as well as in Python. A schema name is
    -- interpolated into SQL and cannot be parameterised, so it is the one
    -- value that must never be able to hold anything unexpected.
    CONSTRAINT tenants_schema_name_shape
        CHECK (schema_name ~ '^[a-z][a-z0-9_]{0,62}$')
);
