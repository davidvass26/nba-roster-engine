"""Tests for the Stage 3 real-data connector: bio parsing, panel assembly,
and the posterior -> projections.csv summary.

Synthetic-first per CLAUDE.md: each test plants the answer (ratings, ages,
positions, posterior draws with known quantiles) and asserts the connector
recovers it exactly. The statistics themselves (layers 1-3) are tested in
test_projection_{delta,hierarchy,baseline,project}.py; these tests guard
the joins that feed them, where a silent misalignment would corrupt every
projection without any model-level test noticing.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.export import projection_rows, update_meta_projections
from nbare.ingest.client import NBAStatsCache, request_hash
from nbare.ingest.nba_stats import (
    parse_player_ages,
    parse_player_positions,
    player_bio_from_cache,
    position_group,
)
from nbare.projection.panel import (
    UNKNOWN_POSITION,
    SeasonRatings,
    build_panel,
    next_season,
)
from nbare.projection.project import PlayerProjection


# --- bio parsing -----------------------------------------------------------

def _rs(headers, rows):
    return {"resultSets": [{"name": "X", "headers": headers, "rowSet": rows}]}


def _ages_payload(rows):
    return _rs(["PLAYER_ID", "PLAYER_NAME", "AGE"], rows)


def _index_payload(rows):
    return _rs(["PERSON_ID", "PLAYER_LAST_NAME", "POSITION"], rows)


def test_position_group_uses_primary_listed_position():
    assert [position_group(p) for p in ("G", "G-F", "F-G", "F-C", "C-F", "C")] == \
        ["G", "G", "F", "F", "C", "C"]
    assert position_group("") is None and position_group(None) is None
    assert position_group("X") is None


def test_parse_player_ages_skips_null_ages():
    ages = parse_player_ages(_ages_payload([[1, "A", 25.0], [2, "B", None]]))
    assert ages == {1: 25}


def test_parse_player_positions_skips_unknown():
    pos = parse_player_positions(_index_payload([[1, "A", "G-F"], [2, "B", ""]]))
    assert pos == {1: "G"}


def test_player_bio_from_cache_latest_position_wins_and_reports_missing(tmp_path):
    cache = NBAStatsCache(tmp_path)

    def put(cls_name, params, payload):
        cache.put(request_hash(cls_name, params), {"payload": payload, "fetched_at": 0.0})

    for season, age, pos in (("2023-24", 24, "F"), ("2024-25", 25, "C")):
        put("LeagueDashPlayerStats",
            {"season": season, "season_type_all_star": "Regular Season"},
            _ages_payload([[7, "P", float(age)]]))
        put("PlayerIndex", {"season": season, "league_id": "00"},
            _index_payload([[7, "P", pos]]))

    ages, positions, missing = player_bio_from_cache(
        ["2024-25", "2023-24", "2025-26"], cache=cache
    )
    assert ages == {(7, "2023-24"): 24, (7, "2024-25"): 25}
    assert positions == {7: "C"}          # 2024-25 listing beats 2023-24
    assert missing == ["2025-26"]


# --- panel assembly --------------------------------------------------------

def _sr(season, ratings, poss=1000.0):
    """ratings: {pid: (off, def)}"""
    return SeasonRatings(
        season=season,
        offense={p: o for p, (o, _) in ratings.items()},
        defense={p: d for p, (_, d) in ratings.items()},
        off_possessions={p: poss for p in ratings},
        def_possessions={p: poss for p in ratings},
    )


def test_build_panel_recovers_planted_rows():
    srs = [
        _sr("2023-24", {1: (1.0, -0.5), 2: (0.2, 0.3)}),
        _sr("2024-25", {1: (1.5, -0.2)}),
    ]
    ages = {(1, "2023-24"): 25, (1, "2024-25"): 26, (2, "2023-24"): 30}
    res = build_panel(srs, ages, {1: "G", 2: "C"})

    got = [(r.player_id, r.season, r.age, r.position_group, r.off_rating, r.def_rating)
           for r in res.seasons]
    assert got == [
        (1, "2023-24", 25, "G", 1.0, -0.5),
        (1, "2024-25", 26, "G", 1.5, -0.2),
        (2, "2023-24", 30, "C", 0.2, 0.3),
    ]
    assert res.seasons[0].off_possessions == 1000.0
    assert res.seasons[0].off_se is None  # no SEs supplied -> unknown, not 0
    assert res.dropped_no_age == res.dropped_low_possessions == 0


def test_build_panel_carries_standard_errors_to_the_right_rows():
    sr = SeasonRatings(
        season="2023-24", offense={1: 1.0, 2: 2.0}, defense={1: 0.0, 2: 0.0},
        off_possessions={1: 1000.0, 2: 1000.0}, def_possessions={1: 1000.0, 2: 1000.0},
        off_se={1: 0.4, 2: 0.9}, def_se={1: 0.5, 2: 1.1},
    )
    res = build_panel([sr], {(1, "2023-24"): 25, (2, "2023-24"): 30}, {})
    by = {r.player_id: r for r in res.seasons}
    assert (by[1].off_se, by[1].def_se) == (0.4, 0.5)
    assert (by[2].off_se, by[2].def_se) == (0.9, 1.1)


def test_build_panel_drops_rows_without_age_and_low_possessions():
    srs = [_sr("2023-24", {1: (1.0, 0.0), 2: (0.0, 0.0)}),
           _sr("2023-24b", {3: (0.0, 0.0)}, poss=50.0)]
    res = build_panel(srs, {(1, "2023-24"): 25, (3, "2023-24b"): 22}, {})
    assert [r.player_id for r in res.seasons] == [1]
    assert res.dropped_no_age == 1           # player 2: no age, never imputed
    assert res.dropped_low_possessions == 1  # player 3: 50 possessions


def test_build_panel_drops_players_whose_ages_do_not_step_by_one():
    """A +2 age jump across consecutive seasons is a data error; it would
    otherwise create a fake age transition in the delta method."""
    srs = [_sr("2023-24", {1: (0, 0), 2: (0, 0)}), _sr("2024-25", {1: (0, 0), 2: (0, 0)})]
    ages = {(1, "2023-24"): 25, (1, "2024-25"): 27,   # bad
            (2, "2023-24"): 25, (2, "2024-25"): 26}   # good
    res = build_panel(srs, ages, {})
    assert {r.player_id for r in res.seasons} == {2}
    assert res.dropped_age_inconsistent_players == 1


def test_build_panel_allows_a_gap_season_with_matching_age_gap():
    srs = [_sr("2023-24", {1: (0, 0)}), _sr("2024-25", {}), _sr("2025-26", {1: (0, 0)})]
    res = build_panel(srs, {(1, "2023-24"): 25, (1, "2025-26"): 27}, {})
    assert len(res.seasons) == 2 and res.dropped_age_inconsistent_players == 0


def test_build_panel_unknown_position_is_its_own_group_not_a_guess():
    res = build_panel([_sr("2023-24", {1: (0, 0)})], {(1, "2023-24"): 25}, {})
    assert res.seasons[0].position_group == UNKNOWN_POSITION
    assert res.unknown_position_players == 1


def test_next_season():
    assert next_season("2025-26") == "2026-27"
    assert next_season("2025-26", 4) == "2029-30"
    assert next_season("2099-00") == "2100-01"


# --- posterior -> projections.csv summary ------------------------------------

def test_projection_rows_recovers_planted_quantiles_and_sums_per_draw():
    """Planted draws: offense = 0..999/100 (uniform-ish grid), defense =
    -offense + 1 for every draw, so total is EXACTLY 1.0 in every draw.
    Summing medians/quantiles separately would give a wide interval; per-
    draw summing must give a zero-width interval at 1.0."""
    off = np.linspace(-5, 5, 1001)[:, None] * np.ones((1, 2))
    deff = 1.0 - off
    proj = {7: PlayerProjection(player_id=7, future_ages=(27, 28),
                                offense_samples=off, defense_samples=deff)}
    rows = projection_rows(proj, {7: "Planted Player"}, {7: 26}, "2025-26")

    assert [(r["season"], r["age"]) for r in rows] == [("2026-27", 27), ("2027-28", 28)]
    r = rows[0]
    assert r["proj_offense"] == pytest.approx(0.0)
    assert r["proj_defense"] == pytest.approx(1.0)
    assert r["proj_total"] == pytest.approx(1.0)
    assert r["lower_80"] == pytest.approx(1.0)
    assert r["upper_80"] == pytest.approx(1.0)


def test_projection_rows_interval_is_10th_to_90th_percentile():
    off = np.linspace(0, 10, 1001)[:, None]
    proj = {1: PlayerProjection(1, (30,), off, np.zeros_like(off))}
    r = projection_rows(proj, {1: "A"}, {1: 29}, "2025-26")[0]
    assert r["lower_80"] == pytest.approx(1.0)
    assert r["upper_80"] == pytest.approx(9.0)
    assert r["proj_total"] == pytest.approx(5.0)


def test_projection_rows_drops_unnamed_players():
    off = np.zeros((10, 1))
    proj = {1: PlayerProjection(1, (30,), off, off), 2: PlayerProjection(2, (30,), off, off)}
    rows = projection_rows(proj, {1: "A"}, {1: 29, 2: 29}, "2025-26")
    assert {r["player_id"] for r in rows} == {1}


def test_update_meta_projections_leaves_rapm_keys_alone(tmp_path):
    import json

    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"lambda": 8000.0, "projections_status": "not yet built"}))
    update_meta_projections(path, {"status": "built", "n_players": 3})
    meta = json.loads(path.read_text())
    assert meta["lambda"] == 8000.0
    assert meta["projections_status"] == "built"
    assert meta["projections"]["n_players"] == 3


def test_build_panel_drops_rows_missing_one_side_measurement():
    """likelihood_measurements omits a side with no data beyond the prior;
    such a player-season cannot form a row (both sides share it)."""
    sr = SeasonRatings(
        season="2023-24", offense={1: 1.0, 2: 0.5}, defense={1: 0.2},
        off_possessions={1: 1000.0, 2: 1000.0}, def_possessions={1: 1000.0, 2: 1000.0},
    )
    res = build_panel([sr], {(1, "2023-24"): 25, (2, "2023-24"): 26}, {})
    assert [r.player_id for r in res.seasons] == [1]
    assert res.dropped_no_measurement == 1
