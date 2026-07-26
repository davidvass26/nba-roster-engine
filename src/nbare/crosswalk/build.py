"""Resolve NBA.com player ids against Basketball-Reference and Spotrac slugs.

This module is small and boring and will consume more of your debugging
time than the RAPM solver. Name matching fails on:

  - suffixes           Jaren Jackson Jr. / Jaren Jackson
  - diacritics         Nikola Jokic / Jokić / Jokic
  - transliteration    Alperen Sengun / Şengün
  - nicknames          Cam vs Cameron, Nic vs Nicolas
  - genuine duplicates two players named Marcus Williams
  - mid-career changes Ron Artest -> Metta World Peace

Strategy: normalize aggressively, match exactly on (normalized name,
birth year) where available, fall back to fuzzy on normalized name
within a draft-year window, and route everything the fuzzy pass is not
confident about to a version-controlled overrides file. Manual overrides
always win and are never overwritten by a rebuild.
"""

from __future__ import annotations

import difflib
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl
import yaml

OVERRIDES_PATH = Path(__file__).with_name("overrides.yaml")

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
FUZZY_ACCEPT = 0.93   # auto-accept above this
FUZZY_REVIEW = 0.80   # below this we do not even suggest


def normalize_name(name: str) -> str:
    """Strip diacritics, punctuation, suffixes; lowercase; collapse space.

    >>> normalize_name("Nikola Jokić")
    'nikola jokic'
    >>> normalize_name("Jaren Jackson Jr.")
    'jaren jackson'
    >>> normalize_name("Alperen Şengün")
    'alperen sengun'
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    parts = [p for p in s.split() if p not in SUFFIXES]
    return " ".join(parts)


def load_overrides() -> dict[str, dict[str, str]]:
    if not OVERRIDES_PATH.exists():
        return {}
    data = yaml.safe_load(OVERRIDES_PATH.read_text()) or {}
    return {str(k): v for k, v in data.items()}


def match_one(
    target: str,
    candidates: dict[str, str],
) -> tuple[str | None, float]:
    """Best fuzzy match of `target` against {normalized_name: slug}."""
    if not candidates:
        return None, 0.0
    if target in candidates:
        return candidates[target], 1.0
    best = difflib.get_close_matches(
        target, list(candidates), n=1, cutoff=FUZZY_REVIEW
    )
    if not best:
        return None, 0.0
    score = difflib.SequenceMatcher(None, target, best[0]).ratio()
    return candidates[best[0]], score


def build(
    con,
    bbref: Iterable[tuple[str, str]] = (),
    spotrac: Iterable[tuple[str, str]] = (),
) -> pl.DataFrame:
    """Build stg.player_xwalk.

    Parameters
    ----------
    bbref, spotrac
        Iterables of (display_name, slug) scraped from each source.
    """
    players = con.execute(
        "SELECT nba_player_id, full_name FROM stg.player"
    ).pl()
    if players.is_empty():
        return pl.DataFrame()

    bbref_map = {normalize_name(n): s for n, s in bbref}
    spotrac_map = {normalize_name(n): s for n, s in spotrac}
    overrides = load_overrides()

    rows = []
    for pid, name in players.iter_rows():
        key = str(pid)
        if key in overrides:
            o = overrides[key]
            rows.append(
                (pid, o.get("bbref_slug"), o.get("spotrac_slug"), "manual", 1.0)
            )
            continue

        norm = normalize_name(name)
        b_slug, b_score = match_one(norm, bbref_map)
        s_slug, s_score = match_one(norm, spotrac_map)

        method = "exact" if max(b_score, s_score) == 1.0 else "fuzzy"
        if b_score < FUZZY_ACCEPT:
            b_slug = None
        if s_score < FUZZY_ACCEPT:
            s_slug = None

        rows.append((pid, b_slug, s_slug, method, min(b_score, s_score)))

    df = pl.DataFrame(
        rows,
        schema=[
            ("nba_player_id", pl.Int64),
            ("bbref_slug", pl.Utf8),
            ("spotrac_slug", pl.Utf8),
            ("match_method", pl.Utf8),
            ("match_score", pl.Float64),
        ],
        orient="row",
    ).with_columns(
        pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias("resolved_at")
    )

    con.execute("DELETE FROM stg.player_xwalk")
    con.register("_xw", df.to_arrow())
    con.execute("INSERT INTO stg.player_xwalk SELECT * FROM _xw")
    con.unregister("_xw")
    return df


def unresolved(con) -> pl.DataFrame:
    """Players still missing a slug. Work this list down by hand into
    overrides.yaml. Expect on the order of 100-300 rows; most are
    long-retired players who never appear in a modern cap sheet, so
    prioritize anyone active since 2015."""
    return con.execute(
        """
        SELECT p.nba_player_id, p.full_name, x.match_score, p.to_year
        FROM stg.player p
        LEFT JOIN stg.player_xwalk x USING (nba_player_id)
        WHERE x.bbref_slug IS NULL OR x.spotrac_slug IS NULL
        ORDER BY p.to_year DESC NULLS LAST, p.full_name
        """
    ).pl()
