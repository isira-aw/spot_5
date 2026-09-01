"""Every tunable in one place, all of it environment-driven.

Nothing in this system reads ``os.environ`` directly except this module. That is
what makes moving the deployment to another machine a matter of copying two
things: this ``.env`` file and ``backend/var/spot5.db``.

The database needs no configuration. It is a SQLite file, created on first run;
``DB_PATH`` moves it and ``DATABASE_URL`` overrides it with any SQLAlchemy URL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

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


def db_path() -> str:
    """Where the database file lives. ``DB_PATH`` moves it."""
    return os.path.abspath(_env("DB_PATH", os.path.join(BASE_DIR, "var", "spot5.db")))


def _resolve_database() -> tuple[str, str]:
    """The database URL and a short label saying where it came from.

    There is nothing to configure: a SQLite file under ``backend/var/`` is
    created on first run. ``DATABASE_URL`` overrides it with any SQLAlchemy URL
    for the cases that need one (the test suite points it at a temp file).
    """
    override = _env("DATABASE_URL")
    if override:
        return override, "DATABASE_URL"
    path = db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Forward slashes: a backslash in a SQLAlchemy URL is not a path separator.
    return "sqlite:///" + path.replace("\\", "/"), "SQLite (default)"


def _database_url() -> str:
    return _resolve_database()[0]


def _database_source() -> str:
    return _resolve_database()[1]


@dataclass(frozen=True)
class DatabaseSettings:
    url: str = field(default_factory=_database_url)
    source: str = field(default_factory=_database_source)
    busy_timeout_ms: int = field(default_factory=lambda: _int("DB_BUSY_TIMEOUT_MS", 10_000))
    connect_retries: int = field(default_factory=lambda: _int("DB_CONNECT_RETRIES", 5))
    echo: bool = field(default_factory=lambda: _flag("DB_ECHO"))

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def safe_url(self) -> str:
        """The URL as shown in logs and API responses. A local file path holds no
        credentials, so there is nothing to redact."""
        return self.url


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
