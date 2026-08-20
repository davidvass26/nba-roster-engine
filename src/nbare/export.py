"""Serialize precomputed results into small, git-committable files the
deployed app reads directly.

Why this exists
----------------
The deployed Streamlit app (Streamlit Community Cloud) has no DuckDB
warehouse and no nba.com cache -- both are git-ignored and local-only. So
RAPM (and, eventually, projections) must be computed locally, ahead of
time, and exported to small files the app ships with in the repo. See
docs/export_results_spec.md and docs/app_spec.md.

This module computes NOTHING new
---------------------------------
`rapm_leaderboard_rows` is a straight serialization of an already-fitted
`RAPMResult` (produced by the same `rapm.blocks` -> `rapm.design` ->
`rapm.fit` path `fit-rapm` uses) -- it does not touch the ridge, the
possession weights, or the value formula. If a number here looks wrong,
the bug is upstream in `rapm.fit`, not here.

`projections.csv` is exported as a header-only stub (see
`write_projections_stub`). No real-data projections pipeline has been
built yet -- `nbare.projection`'s delta/hierarchy/baseline layers are only
validated on synthetic data so far (feeding them real data needs
per-season RAPM panels, ages, and position groups that nothing currently
assembles). Faking rows here to look complete would violate this
project's non-negotiable honesty rule; an honestly-empty file with a
`projections_status` note in meta.json is the correct interim state.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAPM_LEADERBOARD_COLUMNS = [
    "player_id", "name", "offense", "defense", "total", "value",
    "wins_added", "possessions",
]

PROJECTIONS_COLUMNS = [
    "player_id", "name", "season", "proj_offense", "proj_defense",
    "proj_total", "lower_80", "upper_80",
]

PROJECTIONS_STATUS_NOT_BUILT = (
    "not yet built: no real-data projections pipeline exists (Stage 3's "
    "delta/hierarchy/baseline layers are validated only on synthetic data "
    "so far). This file has headers only, zero rows."
)


def rapm_leaderboard_rows(result: Any, names: dict[int, str]) -> list[dict]:
    """One row per player from a fitted RAPMResult, in the SAME order as
    `result.ranking()` (sorted by total RAPM, descending). Preserving that
    order (rather than re-sorting here) is what makes the round-trip check
    in docs/export_results_spec.md meaningful -- the export must match
    `fit-rapm`'s own leaderboard order exactly, not just its numbers.

    A player with no entry in `names` is DROPPED, not exported under a
    bare id -- see docs/app_spec.md's "Do not show bare player ids" rule
    and this spec's own validation requirement.
    """
    rows = []
    for pid, total, off, deff in result.ranking():
        name = names.get(pid)
        if not name:
            continue
        rows.append({
            "player_id": pid,
            "name": name,
            "offense": off,
            "defense": deff,
            "total": total,
            "value": result.value.get(pid, 0.0),
            "wins_added": result.wins_added.get(pid, 0.0),
            "possessions": result.possessions.get(pid, 0.0),
        })
    return rows


def write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_projections_stub(path: Path) -> None:
    """Header-only projections.csv -- see module docstring."""
    write_csv([], PROJECTIONS_COLUMNS, path)


def build_meta(
    *,
    seasons_used: list[str],
    n_games: int,
    n_stints: int,
    lambda_: float,
    replacement_level: float,
    points_per_win: float,
    fit_type: str,
) -> dict:
    """Provenance for the app to show honestly, per docs/export_results_spec.md.

    `fit_type` is passed in, not hardcoded, so this can never silently
    drift from what was actually run (the spec's own example text,
    "pooled multi-season ridge with box-score prior", describes the
    Bayesian prior variant, `fit_rapm_bayesian` -- the export currently
    runs plain zero-mean `fit_rapm`, so the caller must pass the string
    that matches reality).
    """
    return {
        "seasons_used": seasons_used,
        "n_games": n_games,
        "n_stints": n_stints,
        "lambda": lambda_,
        "replacement_level": replacement_level,
        "points_per_win": points_per_win,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fit_type": fit_type,
        "projections_status": PROJECTIONS_STATUS_NOT_BUILT,
    }


def write_meta_json(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
