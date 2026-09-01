"""Every tunable in one place, all of it environment-driven.

Nothing in this system reads ``os.environ`` directly except this module. That is
what makes moving the deployment to another machine a matter of copying one
``.env`` file: the code carries no host-specific knowledge at all, and every
piece of *state* lives in Postgres rather than on the local disk.

Precedence for the database URL, highest first:

1. ``DATABASE_URL`` (Railway sets this for you),
2. the discrete ``PGHOST``/``PGPORT``/``PGUSER``/``PGPASSWORD``/``PGDATABASE`` vars,
3. ``POSTGRES_USER``/``POSTGRES_PASSWORD``/``POSTGRES_DB`` with a localhost host.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any
from urllib.parse import quote_plus, urlparse, urlunparse

try:                                          # optional convenience, never required
    from dotenv import load_dotenv

    for _candidate in (".env", "backend/.env",
                       os.path.join(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))), ".env")):
        if os.path.exists(_candidate):
            load_dotenv(_candidate, override=False)
            break
except Exception:                             # pragma: no cover - dotenv is optional
    pass


# ── helpers ──────────────────────────────────────────────────────────────────
def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _flag(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _csv(name: str, default: str = "") -> list[str]:
    return [p.strip() for p in _env(name, default).split(",") if p.strip()]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # backend/
REPO_DIR = os.path.dirname(BASE_DIR)

PAPER, REAL = "PAPER", "REAL"
MODES = (PAPER, REAL)


def normalize_db_url(url: str) -> str:
    """SQLAlchemy 2 refuses the bare ``postgres://`` scheme Railway hands out."""
    if not url:
        return ""
    u = urlparse(url)
    scheme = u.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+psycopg2"
    return urlunparse(u._replace(scheme=scheme))


INTERNAL_SUFFIX = ".railway.internal"


def _on_railway() -> bool:
    """True when this process is running *inside* the Railway network.

    Railway injects these into every deployment and into nothing else, so their
    presence is what decides whether the private hostname is usable.
    """
    return bool(_env("RAILWAY_ENVIRONMENT") or _env("RAILWAY_ENVIRONMENT_NAME")
                or _env("RAILWAY_SERVICE_ID") or _env("RAILWAY_PROJECT_ID")
                or _env("RAILWAY_PRIVATE_DOMAIN"))


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_internal(host: str) -> bool:
    return host.endswith(INTERNAL_SUFFIX) or host == "postgres"


def _resolve_database() -> tuple[str, str]:
    """Pick the right URL for *where this process actually runs*.

    Railway hands out two URLs for one database and they are not
    interchangeable:

    * ``DATABASE_URL`` → ``postgres.railway.internal``, resolvable only from
      inside the Railway network. Free, fast, no proxy in the path.
    * ``DATABASE_PUBLIC_URL`` → ``<name>.proxy.rlwy.net:<port>``, the TCP proxy,
      which is the only one a laptop can reach.

    Preferring ``DATABASE_URL`` everywhere — as this used to — means a local run
    tries to resolve a hostname that does not exist off-platform, which fails
    slowly and confusingly rather than immediately. So: private first on
    Railway, public first everywhere else, and an internal hostname is *never*
    used off-platform even if it is the only thing configured.

    Returns ``(url, source)`` — the source names which variable won, so the boot
    log can say which of the two it dialled.
    """
    on_rail = _on_railway()
    order = ("DATABASE_URL", "DATABASE_PUBLIC_URL") if on_rail else \
            ("DATABASE_PUBLIC_URL", "DATABASE_URL")

    skipped_internal = False
    for name in order:
        raw = _env(name)
        if not raw:
            continue
        if not on_rail and _is_internal(_host_of(raw)):
            skipped_internal = True         # unreachable from here; try the next one
            continue
        return normalize_db_url(raw), name

    # Discrete vars. Railway sets PGHOST to the internal host too, so it gets
    # the same treatment; POSTGRES_* carry no host and default to localhost.
    user = _env("PGUSER") or _env("POSTGRES_USER", "postgres")
    password = _env("PGPASSWORD") or _env("POSTGRES_PASSWORD")
    host = _env("PGHOST", "localhost")
    port = _env("PGPORT", "5432")
    database = _env("PGDATABASE") or _env("POSTGRES_DB", "railway")
    if not on_rail and _is_internal(host):
        return "", "unusable"
    # A Railway URL was configured but is unreachable from here. Defaulting to
    # localhost would quietly connect to *a different database* and look like a
    # success, so refuse instead and let database_hint() say why.
    if skipped_internal and not _env("PGHOST"):
        return "", "unusable"
    # Same reasoning for a blank config: PGHOST defaults to localhost, so an
    # empty .env would otherwise dial whatever Postgres happens to run on this
    # machine. Off Railway, require something deliberate before assuming that.
    if not on_rail and not _env("PGHOST") and not password:
        return "", "unusable"
    auth = f"{quote_plus(user)}:{quote_plus(password)}@" if password else f"{quote_plus(user)}@"
    return f"postgresql+psycopg2://{auth}{host}:{port}/{database}", "PG* variables"


def _database_url() -> str:
    return _resolve_database()[0]


def _database_source() -> str:
    return _resolve_database()[1]


def database_hint() -> str:
    """Why there is no usable URL — shown instead of a bare 'not configured'."""
    if _database_url():
        return ""
    if _is_internal(_host_of(_env("DATABASE_URL"))) or _is_internal(_env("PGHOST")):
        return ("the only database configured is Railway's private hostname "
                f"({_env('PGHOST') or _host_of(_env('DATABASE_URL'))}), which resolves only "
                "inside the Railway network — set DATABASE_PUBLIC_URL for local runs")
    return ("nothing is filled in — paste DATABASE_PUBLIC_URL from Railway → Postgres → "
            "Variables into backend/.env (DATABASE_URL is the private hostname and is used "
            "only when deployed)")


@dataclass(frozen=True)
class DatabaseSettings:
    url: str = field(default_factory=_database_url)
    source: str = field(default_factory=_database_source)
    sslmode: str = field(default_factory=lambda: _env("PGSSLMODE", "prefer"))
    pool_size: int = field(default_factory=lambda: _int("DB_POOL_SIZE", 5))
    max_overflow: int = field(default_factory=lambda: _int("DB_MAX_OVERFLOW", 5))
    pool_recycle_s: int = field(default_factory=lambda: _int("DB_POOL_RECYCLE_S", 900))
    connect_timeout_s: int = field(default_factory=lambda: _int("DB_CONNECT_TIMEOUT_S", 15))
    statement_timeout_ms: int = field(default_factory=lambda: _int("DB_STATEMENT_TIMEOUT_MS", 30_000))
    connect_retries: int = field(default_factory=lambda: _int("DB_CONNECT_RETRIES", 5))
    echo: bool = field(default_factory=lambda: _flag("DB_ECHO"))

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def safe_url(self) -> str:
        """The URL with the password blanked — safe for logs and API responses."""
        if not self.url:
            return ""
        u = urlparse(self.url)
        if u.password:
            netloc = u.netloc.replace(f":{u.password}@", ":***@")
            u = u._replace(netloc=netloc)
        return urlunparse(u)


@dataclass(frozen=True)
class LLMSettings:
    """The Agent's brain. Groq first, local Ollama second, deterministic third."""
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_base_url: str = field(default_factory=lambda: _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_fallback_models: list[str] = field(default_factory=lambda: _csv(
        "GROQ_FALLBACK_MODELS",
        "llama-3.3-70b-versatile,qwen/qwen3-32b,openai/gpt-oss-20b,llama-3.1-8b-instant"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:0.5b"))
    timeout_s: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 90))
    temperature: float = field(default_factory=lambda: _num("LLM_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1600))
    enabled: bool = field(default_factory=lambda: _flag("LLM_ENABLED", True))


@dataclass(frozen=True)
class KnowledgeBaseSettings:
    path: str = field(default_factory=lambda: _env(
        "KB_PATH", os.path.join(BASE_DIR, "llm_agent", "trading_theories_knowledge_base.md")))
    refresh_seconds: int = field(default_factory=lambda: _int("KB_REFRESH_SECONDS", 60))
    max_chars_in_prompt: int = field(default_factory=lambda: _int("KB_MAX_PROMPT_CHARS", 9000))
    max_sections_in_prompt: int = field(default_factory=lambda: _int("KB_MAX_PROMPT_SECTIONS", 7))
    persist_to_db: bool = field(default_factory=lambda: _flag("KB_PERSIST_TO_DB", True))


@dataclass(frozen=True)
class RiskCaps:
    """Hard ceilings. Admin rules in the database may tighten these, never loosen."""
    max_position_pct: float = field(default_factory=lambda: _num("MAX_POSITION_PCT", 25.0))
    max_capital_at_risk_pct: float = field(default_factory=lambda: _num("MAX_CAPITAL_AT_RISK_PCT", 2.0))
    max_trades_per_day: int = field(default_factory=lambda: _int("MAX_TRADES_PER_DAY", 12))
    max_daily_loss_pct: float = field(default_factory=lambda: _num("MAX_DAILY_LOSS_PCT", 4.0))
    max_open_positions: int = field(default_factory=lambda: _int("MAX_OPEN_POSITIONS", 1))
    min_order_quote: float = field(default_factory=lambda: _num("MIN_ORDER_QUOTE", 10.0))
    min_confidence: float = field(default_factory=lambda: _num("MIN_CONFIDENCE", 0.55))
    stop_loss_pct: float = field(default_factory=lambda: _num("DEFAULT_STOP_LOSS_PCT", 1.5))
    take_profit_pct: float = field(default_factory=lambda: _num("DEFAULT_TAKE_PROFIT_PCT", 3.0))


@dataclass(frozen=True)
class ExecutionSettings:
    mode: str = field(default_factory=lambda: _env("TRADING_MODE", PAPER).strip().upper())
    symbol: str = field(default_factory=lambda: _env("SYMBOL", "BTC/USDT"))
    quote_currency: str = field(default_factory=lambda: _env("QUOTE_CURRENCY", "USDT"))
    base_currency: str = field(default_factory=lambda: _env("BASE_CURRENCY", "BTC"))
    paper_starting_cash: float = field(default_factory=lambda: _num("PAPER_STARTING_CASH", 10_000.0))
    fee_rate: float = field(default_factory=lambda: _num("FEE_RATE", 0.00075))
    slippage_pct: float = field(default_factory=lambda: _num("SLIPPAGE_PCT", 0.0005))
    exchange_id: str = field(default_factory=lambda: _env("EXCHANGE_ID", "binance"))
    exchange_api_key: str = field(default_factory=lambda: _env("EXCHANGE_API_KEY"))
    exchange_api_secret: str = field(default_factory=lambda: _env("EXCHANGE_API_SECRET"))
    exchange_password: str = field(default_factory=lambda: _env("EXCHANGE_API_PASSWORD"))
    live_confirmed: bool = field(default_factory=lambda: _flag("LIVE_TRADING_CONFIRMED"))
    market_offline: bool = field(default_factory=lambda: _flag("MARKET_OFFLINE"))

    def __post_init__(self):
        if self.mode not in MODES:
            object.__setattr__(self, "mode", PAPER)


@dataclass(frozen=True)
class EngineSettings:
    engine_1_enabled: bool = field(default_factory=lambda: _flag("ENGINE_1_ENABLED", True))
    engine_2_enabled: bool = field(default_factory=lambda: _flag("ENGINE_2_ENABLED", True))
    engine_3_enabled: bool = field(default_factory=lambda: _flag("ENGINE_3_ENABLED", True))
    engine_1_timeout_s: int = field(default_factory=lambda: _int("ENGINE_1_TIMEOUT_S", 180))
    engine_2_timeout_s: int = field(default_factory=lambda: _int("ENGINE_2_TIMEOUT_S", 120))
    engine_1_pair: str = field(default_factory=lambda: _env("ENGINE_1_PAIR", "XBTUSDT"))
    engine_1_cache_s: int = field(default_factory=lambda: _int("ENGINE_1_CACHE_S", 240))
    engine_2_models_dir: str = field(default_factory=lambda: _env(
        "ENGINE_2_MODELS_DIR", os.path.join(BASE_DIR, "engine_2", "models")))
    engine_2_cache_s: int = field(default_factory=lambda: _int("ENGINE_2_CACHE_S", 60))
    max_signal_age_s: int = field(default_factory=lambda: _int("MAX_SIGNAL_AGE_S", 1800))
    engine_3_min_samples: int = field(default_factory=lambda: _int("ENGINE_3_MIN_SAMPLES", 40))
    engine_3_retention: int = field(default_factory=lambda: _int("ENGINE_3_MODEL_RETENTION", 10))
    engine_3_train_interval_s: int = field(default_factory=lambda: _int("ENGINE_3_TRAIN_INTERVAL_S", 21_600))


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("APP_ENV", "production"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    instance_id: str = field(default_factory=lambda: _env("INSTANCE_ID", ""))
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _int("PORT", _int("API_PORT", 8000)))
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN"))
    cycle_seconds: int = field(default_factory=lambda: _int("CYCLE_SECONDS", 900))
    autostart_scheduler: bool = field(default_factory=lambda: _flag("AUTOSTART_SCHEDULER", True))
    autotrade: bool = field(default_factory=lambda: _flag("AUTOTRADE", True))
    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    kb: KnowledgeBaseSettings = field(default_factory=KnowledgeBaseSettings)
    caps: RiskCaps = field(default_factory=RiskCaps)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    engines: EngineSettings = field(default_factory=EngineSettings)

    def public_dict(self) -> dict[str, Any]:
        """Configuration with every secret stripped — used by /health and /config."""
        d = asdict(self)
        d["db"] = {**d["db"], "url": self.db.safe_url()}
        d["llm"] = {**d["llm"], "groq_api_key": bool(self.llm.groq_api_key)}
        d["execution"] = {**d["execution"],
                          "exchange_api_key": bool(self.execution.exchange_api_key),
                          "exchange_api_secret": bool(self.execution.exchange_api_secret),
                          "exchange_password": bool(self.execution.exchange_password)}
        d["admin_token"] = bool(self.admin_token)
        return d


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide singleton; ``refresh=True`` re-reads the environment."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings


settings = get_settings()
