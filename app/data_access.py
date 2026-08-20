"""Cached data access for the Streamlit app.

Data contract (docs/app_spec.md): the app reads ONLY the live contract CSV
(for cap sheets and trade legality, computed live so editing the CSV
updates the app) and the precomputed files in app/data/ (for impact and
projections). It must NEVER touch the DuckDB warehouse or the nba.com
cache -- neither exists in the deployed environment, and importing
anything from `nbare.warehouse`, `nbare.rapm`, or `nbare.projection`
would either fail there or silently pull in deps (duckdb usage aside,
numpyro/jax) the deploy is meant to stay lean without.

Auto-refresh on CSV edits
--------------------------
`load_contract_years` is cached on the contract CSV's mtime, not a static
key. Editing the CSV changes its mtime, which changes the cache key, which
forces a recompute on the next Streamlit rerun (e.g. a browser refresh).
A static cache key would silently serve stale contract data forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from nbare.domain.models import CapSheet, PlayerSalary
from nbare.domain.money import Money
from nbare.ingest.contracts import load_raw, to_contract_years

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
CONTRACT_CSV = REPO_ROOT / "data" / "raw" / "bbref_contracts_2026-27.csv"
APP_DATA_DIR = APP_DIR / "data"


@st.cache_data(show_spinner=False)
def _load_contract_years_cached(csv_path: str, _mtime: float) -> pl.DataFrame:
    """`_mtime` is the real cache key (Streamlit hashes all args); it is
    prefixed with an underscore only to signal "don't use this value for
    anything but invalidation" to a reader, not to hide it from caching --
    Streamlit does not skip underscore-prefixed *value* args, only
    underscore-prefixed *unhashable* args, and a float is hashable."""
    return to_contract_years(load_raw(csv_path))


def load_contract_years() -> pl.DataFrame:
    if not CONTRACT_CSV.exists():
        st.error(f"Contract CSV not found at {CONTRACT_CSV}.")
        st.stop()
    return _load_contract_years_cached(
        str(CONTRACT_CSV), CONTRACT_CSV.stat().st_mtime
    )


def teams_in(years: pl.DataFrame, season: str) -> list[str]:
    return sorted(
        years.filter(pl.col("season") == season)["team_abbrev"].unique().to_list()
    )


def players_for_team(years: pl.DataFrame, team: str, season: str) -> pl.DataFrame:
    return (
        years.filter((pl.col("team_abbrev") == team) & (pl.col("season") == season))
        .sort("cap_hit", descending=True)
    )


def cap_sheet_for_team(years: pl.DataFrame, team: str, season: str) -> CapSheet:
    """Same construction `nbare.cli.check_trade_cmd` uses for real-data
    trade checks -- kept identical on purpose so the app and the CLI can
    never quietly disagree about what a cap sheet contains."""
    rs = players_for_team(years, team, season).to_dicts()
    sal = tuple(
        PlayerSalary(
            player_id=r["bbref_slug"], name=r["player"],
            cap_hit=Money(r["cap_hit"]), guaranteed=Money(r["guaranteed"]),
            is_dead_money=bool(r["needs_review"]),
        )
        for r in rs
    )
    certain = not any(s.is_dead_money for s in sal)
    return CapSheet(
        team=team, season=season, salaries=sal, certain=certain,
        uncertainty_notes=(
            (
                "this cap sheet includes row(s) flagged needs_review "
                "(partial guarantee or multi-team dead money) -- see "
                "CLAUDE.md / docs/export_results_spec.md's honesty rules; "
                "apron figures here are a lower bound",
            )
            if not certain
            else ()
        ),
    )


def player_row(years: pl.DataFrame, slug: str, season: str) -> dict | None:
    rows = years.filter(
        (pl.col("bbref_slug") == slug) & (pl.col("season") == season)
    ).to_dicts()
    return rows[0] if rows else None


# --- precomputed app/data/*  (RAPM leaderboard, projections, meta) --------
# Static cache key is correct here, unlike the contract CSV: these files
# are baked into the deploy by `nbare export-app-data` ahead of time and
# are not meant to change within a running app instance.

@st.cache_data(show_spinner=False)
def load_rapm_leaderboard() -> pl.DataFrame:
    path = APP_DATA_DIR / "rapm_leaderboard.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_projections() -> pl.DataFrame:
    path = APP_DATA_DIR / "projections.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    path = APP_DATA_DIR / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def rapm_lookup_by_name(leaderboard: pl.DataFrame) -> dict[str, dict]:
    """name -> row dict. Exact match only, deliberately -- this project
    does not fuzzy-match player names (see ingest/contracts.py's docstring
    on why: it silently merges distinct people). A contract-CSV name that
    doesn't match the RAPM leaderboard's name exactly shows as 'no RAPM
    data', not a guess."""
    if leaderboard.is_empty():
        return {}
    return {r["name"]: r for r in leaderboard.iter_rows(named=True)}
