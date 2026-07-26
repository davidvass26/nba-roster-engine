"""Ingestion from stats.nba.com into the staging layer.

Each function is (a) idempotent, (b) resumable, and (c) reads through
the disk cache, so a killed backfill can be restarted with no loss.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

import polars as pl

from nbare.ingest.client import NBAStatsCache, fetch

log = logging.getLogger(__name__)


# --- payload -> DataFrame ------------------------------------------------

def result_set_to_df(payload: dict[str, Any], name: str | None = None) -> pl.DataFrame:
    """Convert an nba.com resultSets payload into a Polars DataFrame.

    nba.com returns column-headers-plus-row-arrays rather than records,
    and the header names are inconsistently cased across endpoints, so
    we normalize to snake_case here and only here.
    """
    sets = payload.get("resultSets") or payload.get("resultSet") or []
    if isinstance(sets, dict):
        sets = [sets]
    if not sets:
        return pl.DataFrame()

    chosen = None
    if name is not None:
        for rs in sets:
            if rs.get("name") == name:
                chosen = rs
                break
        if chosen is None:
            raise KeyError(f"result set {name!r} not in {[s.get('name') for s in sets]}")
    else:
        chosen = sets[0]

    headers = [_snake(h) for h in chosen.get("headers", [])]
    rows = chosen.get("rowSet", [])
    if not headers:
        return pl.DataFrame()
    return pl.DataFrame(
        rows, schema=headers, orient="row", infer_schema_length=None
    )


def _snake(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.replace(" ", "_").replace("-", "_").lower()


def parse_clock(clock: str | None, period: int) -> float | None:
    """nba.com v3 clock is ISO-8601 duration: 'PT11M23.00S'.

    Returns seconds remaining in the period. Overtime periods are 5:00,
    regulation 12:00 -- callers converting to game-elapsed time must
    account for that, which is why we return period-relative here.
    """
    if not clock:
        return None
    m = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", clock)
    if m:
        mins = int(m.group(1) or 0)
        secs = float(m.group(2) or 0.0)
        return mins * 60 + secs
    # Legacy format 'MM:SS'
    if ":" in clock:
        mm, ss = clock.split(":")
        return int(mm) * 60 + float(ss)
    return None


# --- static reference data ----------------------------------------------

def ingest_players(con, cache: NBAStatsCache | None = None) -> int:
    """All players, active and historical."""
    from nba_api.stats.endpoints import commonallplayers

    resp = fetch(
        commonallplayers.CommonAllPlayers,
        {"is_only_current_season": 0, "league_id": "00", "season": "2025-26"},
        cache=cache,
    )
    df = result_set_to_df(resp.payload)
    if df.is_empty():
        return 0

    out = df.select(
        pl.col("person_id").cast(pl.Int64).alias("nba_player_id"),
        pl.col("display_first_last").alias("full_name"),
        pl.col("display_first_last").str.split(" ").list.first().alias("first_name"),
        pl.col("display_first_last").str.split(" ").list.last().alias("last_name"),
        pl.lit(None, dtype=pl.Date).alias("birthdate"),
        pl.lit(None, dtype=pl.Utf8).alias("position"),
        pl.lit(None, dtype=pl.Int16).alias("height_in"),
        pl.lit(None, dtype=pl.Int16).alias("weight_lb"),
        pl.lit(None, dtype=pl.Int16).alias("draft_year"),
        pl.lit(None, dtype=pl.Int16).alias("draft_round"),
        pl.lit(None, dtype=pl.Int16).alias("draft_number"),
        pl.col("from_year").cast(pl.Int16, strict=False).alias("from_year"),
        pl.col("to_year").cast(pl.Int16, strict=False).alias("to_year"),
        (pl.col("rosterstatus").cast(pl.Int32, strict=False) == 1).alias("is_active"),
    ).unique(subset=["nba_player_id"])

    _upsert(con, "stg.player", out, key="nba_player_id")
    return out.height


def ingest_teams(con, cache: NBAStatsCache | None = None) -> int:
    from nba_api.stats.static import teams as static_teams

    rows = static_teams.get_teams()
    df = pl.DataFrame(rows).select(
        pl.col("id").cast(pl.Int64).alias("nba_team_id"),
        pl.col("abbreviation"),
        pl.col("nickname"),
        pl.col("city"),
        pl.col("full_name"),
    )
    _upsert(con, "stg.team", df, key="nba_team_id")
    return df.height


def ingest_season_games(
    con, season: str, season_type: str = "Regular Season",
    cache: NBAStatsCache | None = None,
) -> int:
    """Game index for a season. Source of game_ids for the PBP backfill."""
    from nba_api.stats.endpoints import leaguegamefinder

    resp = fetch(
        leaguegamefinder.LeagueGameFinder,
        {
            "season_nullable": season,
            "season_type_nullable": season_type,
            "league_id_nullable": "00",
        },
        cache=cache,
    )
    df = result_set_to_df(resp.payload)
    if df.is_empty():
        return 0

    # LeagueGameFinder returns one row per team per game; pivot to one
    # row per game using the '@' vs 'vs.' matchup convention.
    df = df.with_columns(
        pl.col("matchup").str.contains("@").alias("is_away")
    )
    home = df.filter(~pl.col("is_away"))
    away = df.filter(pl.col("is_away"))
    games = home.join(away, on="game_id", how="inner", suffix="_away").select(
        pl.col("game_id"),
        pl.lit(season).alias("season"),
        pl.lit(season_type).alias("season_type"),
        pl.col("game_date").str.to_date("%Y-%m-%d", strict=False).alias("game_date"),
        pl.col("team_id").cast(pl.Int64).alias("home_team_id"),
        pl.col("team_id_away").cast(pl.Int64).alias("away_team_id"),
        pl.col("pts").cast(pl.Int16, strict=False).alias("home_pts"),
        pl.col("pts_away").cast(pl.Int16, strict=False).alias("away_pts"),
    )
    _upsert(con, "stg.game", games, key="game_id")
    return games.height


def ingest_pbp(con, game_ids: Iterable[str], cache: NBAStatsCache | None = None) -> int:
    """Play-by-play for a list of games. The long pole of Stage 0."""
    from nba_api.stats.endpoints import playbyplayv3

    total = 0
    for i, gid in enumerate(game_ids, 1):
        try:
            resp = fetch(
                playbyplayv3.PlayByPlayV3,
                {"game_id": gid, "start_period": 0, "end_period": 14},
                cache=cache,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("PBP failed for %s: %s -- skipping, rerun later", gid, exc)
            continue

        df = result_set_to_df(resp.payload)
        if df.is_empty():
            continue

        cols = df.columns
        def col(name: str, dtype, alias: str):
            if name in cols:
                return pl.col(name).cast(dtype, strict=False).alias(alias)
            return pl.lit(None, dtype=dtype).alias(alias)

        out = df.select(
            pl.lit(gid).alias("game_id"),
            col("action_number", pl.Int32, "event_num"),
            col("period", pl.Int16, "period"),
            pl.col("clock").map_elements(
                lambda c: parse_clock(c, 0), return_dtype=pl.Float64
            ).alias("clock_seconds_left") if "clock" in cols
            else pl.lit(None, dtype=pl.Float64).alias("clock_seconds_left"),
            col("action_type", pl.Utf8, "event_type"),
            col("sub_type", pl.Utf8, "event_action_type"),
            col("description", pl.Utf8, "description"),
            col("team_id", pl.Int64, "team_id"),
            col("person_id", pl.Int64, "player1_id"),
            pl.lit(None, dtype=pl.Int64).alias("player2_id"),
            pl.lit(None, dtype=pl.Int64).alias("player3_id"),
            col("score_home", pl.Int16, "home_score"),
            col("score_away", pl.Int16, "away_score"),
        ).unique(subset=["game_id", "event_num"])

        _upsert(con, "stg.pbp_event", out, key=["game_id", "event_num"])
        total += out.height
        if i % 100 == 0:
            log.info("PBP progress: %d games, %d events", i, total)
    return total


# --- upsert helper -------------------------------------------------------

def _upsert(con, table: str, df: pl.DataFrame, key: str | list[str]) -> None:
    """Delete-then-insert on primary key. DuckDB has no MERGE we can rely
    on across versions, and this is fast enough for our volumes."""
    if df.is_empty():
        return
    keys = [key] if isinstance(key, str) else list(key)
    con.register("_incoming", df.to_arrow())
    pred = " AND ".join(f't."{k}" = i."{k}"' for k in keys)
    con.execute(
        f"DELETE FROM {table} t WHERE EXISTS "
        f"(SELECT 1 FROM _incoming i WHERE {pred})"
    )
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    con.unregister("_incoming")
