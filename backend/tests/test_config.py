"""Connection resolution — the part everyone gets wrong on the first deploy."""


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


def test_an_internal_host_is_diagnosed_without_touching_the_network(monkeypatch):
    """The private-hostname case is decided from the URL alone — no DNS needed."""
    import core.db as db
    _settings(monkeypatch, DATABASE_URL=INTERNAL)

    def explode(*args, **kwargs):
        raise AssertionError("probe_host must not be called for an internal host")

    monkeypatch.setattr(db, "probe_host", explode)
    assert "PRIVATE hostname" in db.connection_hint()


def test_a_sqlite_url_is_not_diagnosed_as_a_network_problem(monkeypatch):
    import core.db as db
    _settings(monkeypatch, DATABASE_URL="sqlite:///./spot5.db")
    assert "sqlite" in db.connection_hint().lower()


def test_a_url_with_no_host_names_the_pgdata_mistake(monkeypatch):
    import core.db as db
    _settings(monkeypatch, DATABASE_URL="postgresql:///railway")
    assert "PGDATA" in db.connection_hint()


def test_the_hint_distinguishes_a_bad_hostname_from_a_blocked_port(monkeypatch):
    """DNS failure and connection refused look identical in a pool, not here."""
    import core.db as db
    _settings(monkeypatch, DATABASE_URL="postgresql://u:p@nope.proxy.rlwy.net:41234/railway")

    monkeypatch.setattr(db, "probe_host", lambda h, p, timeout=5.0: ("dns_failed", "no such host"))
    hint = db.connection_hint()
    assert "does not resolve" in hint and "randomly assigned railway words" in hint

    monkeypatch.setattr(db, "probe_host", lambda h, p, timeout=5.0: ("refused", "1.2.3.4: refused"))
    assert "nothing accepted a connection on port 41234" in db.connection_hint()

    monkeypatch.setattr(db, "probe_host", lambda h, p, timeout=5.0: ("timeout", "1.2.3.4"))
    assert "firewall" in db.connection_hint()

    monkeypatch.setattr(db, "probe_host", lambda h, p, timeout=5.0: ("open", "1.2.3.4"))
    hint = db.connection_hint(driver_error="FATAL: password authentication failed")
    assert "network is fine" in hint and "password authentication failed" in hint


def test_a_dead_localhost_suggests_sqlite_rather_than_railway_advice(monkeypatch):
    import core.db as db
    _settings(monkeypatch, DATABASE_URL="postgresql://u:p@localhost:5432/db")
    monkeypatch.setattr(db, "probe_host", lambda h, p, timeout=5.0: ("refused", "127.0.0.1"))
    hint = db.connection_hint()
    assert "sqlite:///./spot5.db" in hint and "Railway" not in hint


def test_probe_host_reports_dns_failure_without_raising(monkeypatch):
    from core.db import probe_host
    verdict, detail = probe_host("this-host-does-not-exist.invalid", 5432, timeout=1.0)
    assert verdict == "dns_failed" and detail
