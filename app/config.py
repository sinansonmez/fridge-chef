import os
from dataclasses import dataclass

DEFAULT_MAIN_MODEL = "gemini-2.5-flash-lite"
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash"
DEFAULT_HISTORY_DAYS = 5
DEFAULT_HISTORY_DB = "data/history.db"


@dataclass(frozen=True)
class Config:
    telegram_token: str
    gemini_api_key: str
    main_model: str
    fallback_model: str
    allowed_user_ids: frozenset[int]
    history_days: int
    history_db: str


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def load_config() -> Config:
    raw_ids = _require("ALLOWED_USER_IDS")
    try:
        allowed_ids = frozenset(
            int(part) for part in raw_ids.replace(" ", "").split(",") if part
        )
    except ValueError:
        raise SystemExit(
            "ALLOWED_USER_IDS must be comma-separated numeric Telegram user IDs"
        ) from None
    if not allowed_ids:
        raise SystemExit("ALLOWED_USER_IDS must contain at least one user ID")

    raw_days = os.environ.get("HISTORY_DAYS", str(DEFAULT_HISTORY_DAYS)).strip()
    try:
        history_days = int(raw_days)
    except ValueError:
        raise SystemExit("HISTORY_DAYS must be an integer number of days") from None
    if history_days < 1:
        raise SystemExit("HISTORY_DAYS must be at least 1")

    return Config(
        telegram_token=_require("TELEGRAM_BOT_TOKEN"),
        gemini_api_key=_require("GEMINI_API_KEY"),
        main_model=os.environ.get("MAIN_MODEL", DEFAULT_MAIN_MODEL).strip(),
        fallback_model=os.environ.get("FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip(),
        allowed_user_ids=allowed_ids,
        history_days=history_days,
        history_db=os.environ.get("HISTORY_DB", DEFAULT_HISTORY_DB).strip(),
    )
