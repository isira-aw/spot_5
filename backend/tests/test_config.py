"""Connection resolution — the part everyone gets wrong on the first deploy."""
import pytest


def _settings(monkeypatch, **env):
    for key in ("DATABASE_URL", "DATABASE_PUBLIC_URL", "RAILWAY_ENVIRONMENT",
                "RAILWAY_SERVICE_ID", "RAILWAY_PROJECT_ID", "PGHOST", "PGPORT",
                "PGUSER", "PGPASSWORD", "PGDATABASE", "POSTGRES_USER",
                "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from core.config import get_settings
    return get_settings(refresh=True)


INTERNAL = "postgresql://postgres:pw@postgres.railway.internal:5432/railway"
PUBLIC = "postgresql://postgres:pw@shortline.proxy.rlwy.net:41234/railway"


def test_off_platform_prefers_the_public_url_over_the_internal_one(monkeypatch):
    s = _settings(monkeypatch, DATABASE_URL=INTERNAL, DATABASE_PUBLIC_URL=PUBLIC)
    assert "proxy.rlwy.net" in s.db.url


def test_inside_railway_the_internal_url_is_used(monkeypatch):
    s = _settings(monkeypatch, DATABASE_URL=INTERNAL, DATABASE_PUBLIC_URL=PUBLIC,
                  RAILWAY_ENVIRONMENT="production")
    assert "railway.internal" in s.db.url


def test_an_internal_url_with_no_public_alternative_is_left_alone(monkeypatch):
    s = _settings(monkeypatch, DATABASE_URL=INTERNAL)
    assert "railway.internal" in s.db.url        # nothing better to offer


def test_the_public_url_alone_is_enough(monkeypatch):
    s = _settings(monkeypatch, DATABASE_PUBLIC_URL=PUBLIC)
    assert "proxy.rlwy.net" in s.db.url


def test_a_pgdata_path_pasted_into_pghost_does_not_become_the_host(monkeypatch):
    s = _settings(monkeypatch, PGHOST="/var/lib/postgresql/data/pgdata",
                  PGUSER="postgres", PGPASSWORD="pw", PGDATABASE="railway")
    assert "@localhost:5432/railway" in s.db.url
    assert "pgdata" not in s.db.url


def test_the_bare_postgres_scheme_is_upgraded_for_sqlalchemy(monkeypatch):
    s = _settings(monkeypatch, DATABASE_URL="postgres://u:p@host:5432/db")
    assert s.db.url.startswith("postgresql+psycopg2://")


def test_the_password_never_appears_in_the_safe_url(monkeypatch):
    s = _settings(monkeypatch, DATABASE_URL=PUBLIC)
    assert "pw" not in s.db.safe_url() and "***" in s.db.safe_url()


@pytest.mark.parametrize("url,expected", [
    (INTERNAL, "PRIVATE hostname"),
    ("postgresql://u:p@localhost:5432/db", "nothing is listening"),
    (PUBLIC, "did not answer"),
])
def test_the_connection_hint_names_the_actual_problem(monkeypatch, url, expected):
    _settings(monkeypatch, DATABASE_URL=url)
    from core.db import connection_hint
    assert expected in connection_hint()
