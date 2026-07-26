"""Rate-limited, disk-cached wrapper around nba_api.

Why this exists
---------------
stats.nba.com is undocumented, unversioned, and actively hostile to
scripted access. It will hang, return HTTP 200 with an empty result set,
or start dropping you entirely if you hit it in a tight loop. A full
12-season play-by-play backfill is ~15,000 requests; at 0.75s spacing
that is roughly three hours, and you do NOT want to redo it because a
parser had a bug.

So: every response is cached to disk keyed by (endpoint, params). Reruns
are free and offline. Parsing changes never trigger a refetch. When the
backfill dies at request 11,000 you restart and it resumes.

The cache is the actual deliverable of Stage 0. Back it up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nbare.config import (
    CACHE_DIR,
    NBA_STATS_MAX_RETRIES,
    NBA_STATS_MIN_INTERVAL_S,
    NBA_STATS_TIMEOUT_S,
)

log = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe minimum-interval limiter.

    Deliberately not a token bucket: bursting is exactly what gets you
    blocked. A flat floor between requests is what survives.
    """

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval_s:
                time.sleep(self.min_interval_s - delta)
            self._last = time.monotonic()


_limiter = RateLimiter(NBA_STATS_MIN_INTERVAL_S)


def request_hash(endpoint: str, params: dict[str, Any]) -> str:
    blob = json.dumps(
        {"endpoint": endpoint, "params": params}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass(frozen=True, slots=True)
class CachedResponse:
    endpoint: str
    params: dict[str, Any]
    fetched_at: float
    payload: dict[str, Any]
    from_cache: bool


class NBAStatsCache:
    """Content-addressed JSON cache on disk."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or CACHE_DIR / "nba_stats")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by prefix so directories stay under ~5k entries.
        sub = self.root / key[:2]
        sub.mkdir(exist_ok=True)
        return sub / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("corrupt cache entry %s, discarding", key)
            p.unlink(missing_ok=True)
            return None

    def put(self, key: str, record: dict[str, Any]) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(p)  # atomic; a killed process never leaves half a file

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))

    def __bool__(self) -> bool:
        # Without this, an *empty* cache is falsy (because __len__ == 0)
        # and `cache or default` silently swaps out a perfectly good
        # cache object. A cache always exists; emptiness is not absence.
        return True


class EmptyResultError(RuntimeError):
    """stats.nba.com returned 200 with no rows -- usually soft throttling."""


def fetch(
    endpoint_cls: Callable[..., Any],
    params: dict[str, Any],
    *,
    cache: NBAStatsCache | None = None,
    force: bool = False,
    allow_empty: bool = False,
) -> CachedResponse:
    """Call an nba_api endpoint class with caching and retries.

    Parameters
    ----------
    endpoint_cls
        e.g. ``nba_api.stats.endpoints.playbyplayv3.PlayByPlayV3``
    params
        kwargs passed to the endpoint constructor.
    allow_empty
        Some endpoints legitimately return zero rows (a player with no
        playoff games). Set True to suppress the throttle heuristic.
    """
    # NB: `cache or NBAStatsCache()` is WRONG here -- NBAStatsCache defines
    # __len__, so an empty cache is falsy and the caller's object would be
    # silently discarded on the very first (always-empty) call. Identity
    # check only.
    if cache is None:
        cache = NBAStatsCache()
    name = getattr(endpoint_cls, "__name__", str(endpoint_cls))
    key = request_hash(name, params)

    if not force:
        hit = cache.get(key)
        if hit is not None:
            return CachedResponse(
                endpoint=name,
                params=params,
                fetched_at=hit["fetched_at"],
                payload=hit["payload"],
                from_cache=True,
            )

    last_err: Exception | None = None
    for attempt in range(1, NBA_STATS_MAX_RETRIES + 1):
        _limiter.wait()
        try:
            ep = endpoint_cls(timeout=NBA_STATS_TIMEOUT_S, **params)
            payload = ep.get_dict()
            if not allow_empty and _looks_empty(payload):
                raise EmptyResultError(f"{name} returned no rows for {params}")
            record = {
                "endpoint": name,
                "params": params,
                "fetched_at": time.time(),
                "payload": payload,
            }
            cache.put(key, record)
            return CachedResponse(
                endpoint=name,
                params=params,
                fetched_at=record["fetched_at"],
                payload=payload,
                from_cache=False,
            )
        except Exception as exc:  # noqa: BLE001 - we retry everything
            last_err = exc
            backoff = min(2 ** attempt, 60)
            log.warning(
                "%s attempt %d/%d failed (%s); sleeping %ds",
                name, attempt, NBA_STATS_MAX_RETRIES, exc, backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"{name} failed after {NBA_STATS_MAX_RETRIES} attempts: {last_err}"
    )


def _looks_empty(payload: dict[str, Any]) -> bool:
    """True if every result set in the payload has zero rows."""
    sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(sets, dict):
        sets = [sets]
    if not sets:
        return True
    return all(not (rs.get("rowSet") or []) for rs in sets)
