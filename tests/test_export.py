"""Tests for app-data export (docs/export_results_spec.md).

Strategy: fit a tiny, real RAPMResult from hand-built OffenseBlocks (same
approach as test_rapm.py's value-metric tests) rather than mocking the
result type, so these tests exercise the real dataclass contract
(`ranking()`, `.value`, `.wins_added`, `.possessions`). The export layer
itself computes nothing -- these tests check that serialization preserves
order, drops unnamed players, and round-trips through real files.
"""

from __future__ import annotations

import csv
import json

import pytest

from nbare.export import (
    LAST_SEASON_COLUMNS,
    PROJECTIONS_COLUMNS,
    RAPM_LEADERBOARD_COLUMNS,
    RAPM_LEADERBOARD_WITH_LAST_COLUMNS,
    add_last_season,
    build_meta,
    rapm_leaderboard_rows,
    write_csv,
    write_meta_json,
    write_projections_stub,
)
from nbare.rapm.design import OffenseBlock, build_design
from nbare.rapm.fit import fit_rapm


@pytest.fixture()
def result():
    # Three players with clearly different possession counts, one team
    # dominant enough that ranking order is deterministic and known.
    P1, P2, P3, OPP = 1, 2, 3, 99
    blocks = [
        OffenseBlock("g1", frozenset({P1}), frozenset({OPP}), 12.0, 10.0),
        OffenseBlock("g1", frozenset({OPP}), frozenset({P1}), 8.0, 10.0),
        OffenseBlock("g1", frozenset({P2}), frozenset({OPP}), 10.0, 10.0),
        OffenseBlock("g1", frozenset({OPP}), frozenset({P2}), 9.0, 10.0),
        OffenseBlock("g1", frozenset({P3}), frozenset({OPP}), 9.0, 10.0),
        OffenseBlock("g1", frozenset({OPP}), frozenset({P3}), 10.0, 10.0),
    ]
    design = build_design(blocks)
    return fit_rapm(design, fixed_lambda=100.0)


def test_rapm_leaderboard_rows_matches_ranking_order(result):
    names = {1: "Player One", 2: "Player Two", 3: "Player Three", 99: "Opponent"}
    rows = rapm_leaderboard_rows(result, names)

    expected_order = [pid for pid, *_ in result.ranking()]
    assert [r["player_id"] for r in rows] == expected_order

    for r in rows:
        pid = r["player_id"]
        assert r["name"] == names[pid]
        assert r["total"] == pytest.approx(result.total(pid))
        assert r["value"] == pytest.approx(result.value[pid])
        assert r["wins_added"] == pytest.approx(result.wins_added[pid])
        assert r["possessions"] == pytest.approx(result.possessions[pid])


def test_rapm_leaderboard_rows_drops_players_without_a_name(result):
    """No bare ids in the export -- a player missing from stg.player's
    name join must be dropped, not exported with a fallback like str(id)."""
    names = {1: "Player One", 2: "Player Two"}  # 3 and 99 are unnamed
    rows = rapm_leaderboard_rows(result, names)
    ids = {r["player_id"] for r in rows}
    assert ids == {1, 2}
    assert all(isinstance(r["name"], str) and r["name"] for r in rows)


def test_write_csv_round_trips(tmp_path):
    rows = [
        {"player_id": 1, "name": "A", "offense": 1.5, "defense": -0.5,
         "total": 1.0, "value": 10.0, "wins_added": 0.33, "possessions": 100.0},
    ]
    path = tmp_path / "app" / "data" / "rapm_leaderboard.csv"
    write_csv(rows, RAPM_LEADERBOARD_COLUMNS, path)

    with path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == RAPM_LEADERBOARD_COLUMNS
        got = list(reader)
    assert len(got) == 1
    assert got[0]["name"] == "A"
    assert float(got[0]["total"]) == pytest.approx(1.0)


def test_write_projections_stub_has_headers_and_zero_rows(tmp_path):
    path = tmp_path / "projections.csv"
    write_projections_stub(path)
    with path.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == PROJECTIONS_COLUMNS
        assert list(reader) == []


def test_build_meta_reflects_actual_fit_type_not_the_spec_example():
    """meta.json's fit_type must describe what was ACTUALLY run: it is
    whatever the caller passes (the label the fit returned), never a
    hardcoded spec example string."""
    meta = build_meta(
        seasons_used=["2023-24", "2024-25", "2025-26"],
        n_games=3169,
        n_stints=97449,
        lambda_=5000.0,
        replacement_level=-2.5,
        points_per_win=30.0,
        fit_type="pooled multi-season ridge RAPM (zero-mean prior)",
    )
    assert meta["fit_type"] == "pooled multi-season ridge RAPM (zero-mean prior)"
    assert "box-score prior" not in meta["fit_type"]
    assert meta["seasons_used"] == ["2023-24", "2024-25", "2025-26"]
    assert meta["n_games"] == 3169
    assert meta["lambda"] == 5000.0
    assert "projections_status" in meta
    assert "generated_at" in meta


def test_write_meta_json_round_trips(tmp_path):
    meta = build_meta(
        seasons_used=["2025-26"], n_games=1, n_stints=1, lambda_=1.0,
        replacement_level=-2.5, points_per_win=30.0, fit_type="test",
    )
    path = tmp_path / "meta.json"
    write_meta_json(meta, path)
    assert json.loads(path.read_text()) == meta


# --- last-season columns ---------------------------------------------------

def test_add_last_season_attaches_single_season_numbers_by_player(result):
    """Planted: the 'last season' fit only contains players 1 and 2 with
    known ratings. Pooled order is preserved, last_* values come from the
    last-season result for the right player, and player 3 -- absent from
    the last season -- gets None, not 0.0 (0.0 would claim 'league
    average')."""
    P1, P2, OPP = 1, 2, 99
    last = fit_rapm(build_design([
        OffenseBlock("g9", frozenset({P1}), frozenset({OPP}), 14.0, 10.0),
        OffenseBlock("g9", frozenset({OPP}), frozenset({P1}), 7.0, 10.0),
        OffenseBlock("g9", frozenset({P2}), frozenset({OPP}), 8.0, 10.0),
        OffenseBlock("g9", frozenset({OPP}), frozenset({P2}), 11.0, 10.0),
    ]), fixed_lambda=100.0)
    names = {1: "Player One", 2: "Player Two", 3: "Player Three"}
    pooled = rapm_leaderboard_rows(result, names)
    rows = add_last_season(pooled, last)

    assert [r["player_id"] for r in rows] == [r["player_id"] for r in pooled]
    by = {r["player_id"]: r for r in rows}
    for pid in (P1, P2):
        assert by[pid]["last_offense"] == pytest.approx(last.offense[pid])
        assert by[pid]["last_defense"] == pytest.approx(last.defense[pid])
        assert by[pid]["last_total"] == pytest.approx(last.total(pid))
        assert by[pid]["last_wins_added"] == pytest.approx(last.wins_added[pid])
        assert by[pid]["last_possessions"] == pytest.approx(last.possessions[pid])
        # pooled numbers untouched
        assert by[pid]["total"] == pytest.approx(result.total(pid))
    assert all(by[3][c] is None for c in LAST_SEASON_COLUMNS)
    # The planted last-season edge: P1 outscored, P2 was outscored.
    assert by[P1]["last_total"] > 0 > by[P2]["last_total"]


def test_last_season_columns_round_trip_with_blank_cells(tmp_path, result):
    rows = add_last_season(
        rapm_leaderboard_rows(result, {1: "A", 2: "B", 3: "C"}),
        fit_rapm(build_design([
            OffenseBlock("g9", frozenset({1}), frozenset({99}), 12.0, 10.0),
            OffenseBlock("g9", frozenset({99}), frozenset({1}), 9.0, 10.0),
        ]), fixed_lambda=100.0),
    )
    path = tmp_path / "rapm_leaderboard.csv"
    write_csv(rows, RAPM_LEADERBOARD_WITH_LAST_COLUMNS, path)
    with path.open() as f:
        got = {r["name"]: r for r in csv.DictReader(f)}
    assert got["A"]["last_total"] != ""
    assert got["B"]["last_total"] == ""   # not in last season -> blank, not 0


def test_build_meta_records_last_season_provenance():
    last = {"season": "2025-26", "n_games": 1000, "n_stints": 30000,
            "lambda": 3000.0, "fit_type": "single-season ridge RAPM"}
    meta = build_meta(
        seasons_used=["2024-25", "2025-26"], n_games=2000, n_stints=60000,
        lambda_=5000.0, replacement_level=-2.5, points_per_win=30.0,
        fit_type="pooled", last_season=last,
    )
    assert meta["last_season"] == last
    assert "last_season" not in build_meta(
        seasons_used=["2025-26"], n_games=1, n_stints=1, lambda_=1.0,
        replacement_level=-2.5, points_per_win=30.0, fit_type="x",
    )
