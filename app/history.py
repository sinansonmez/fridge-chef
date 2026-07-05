import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS suggestions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    suggested_at TEXT NOT NULL
)
"""


class History:
    """Global rolling window of recently suggested recipe names.

    Timestamps are stored as UTC ISO-8601 strings, which compare correctly
    as text, so the window is a plain WHERE clause.
    """

    def __init__(self, path: str, window_days: int) -> None:
        self._path = Path(path)
        self._window = timedelta(days=window_days)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as conn, conn:
            conn.execute(SCHEMA)

    def _cutoff(self) -> str:
        return (datetime.now(timezone.utc) - self._window).isoformat()

    def add(self, names: list[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self._path)) as conn, conn:
            conn.executemany(
                "INSERT INTO suggestions (name, suggested_at) VALUES (?, ?)",
                [(name, now) for name in names],
            )
            conn.execute(
                "DELETE FROM suggestions WHERE suggested_at < ?", (self._cutoff(),)
            )

    def recent(self) -> list[tuple[str, datetime]]:
        with closing(sqlite3.connect(self._path)) as conn:
            rows = conn.execute(
                "SELECT name, suggested_at FROM suggestions"
                " WHERE suggested_at >= ? ORDER BY suggested_at DESC",
                (self._cutoff(),),
            ).fetchall()
        return [(name, datetime.fromisoformat(ts)) for name, ts in rows]

    def recent_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for name, _ in self.recent():
            seen.setdefault(name)
        return list(seen)

    def clear(self) -> int:
        with closing(sqlite3.connect(self._path)) as conn, conn:
            return conn.execute("DELETE FROM suggestions").rowcount
