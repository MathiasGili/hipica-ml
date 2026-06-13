-- =============================================================================
-- Postgres init script — runs once on first start of the postgres container.
-- Creates an additional `racing` database next to the MLflow backing store.
--
-- The application schema (races / runners / results) is provisioned by
-- SQLAlchemy / Alembic from the API container, not here.
-- =============================================================================

SELECT 'CREATE DATABASE racing'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'racing')\gexec
