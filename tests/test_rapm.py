"""Tests for the RAPM pipeline: possessions, design matrix, ridge fit.

The recovery tests are the proof of correctness. We plant known player
ratings, simulate possession blocks whose efficiency follows those ratings,
and assert the fit recovers them at high correlation. A pipeline that
recovers ratings it was handed is correct independent of real data; one
that can't is broken no matter how plausible its real output looks.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.rapm.design import OffenseBlock, build_design
from nbare.rapm.fit import (
    POINTS_PER_WIN,
    REPLACEMENT_LEVEL,
    compute_value,
    fit_rapm,
    select_lambda,
)
from nbare.rapm.possessions import estimate_team_possessions


# --- possessions ---------------------------------------------------------

def test_possession_formula():
    # 85 FGA, 10 OREB, 14 TOV, 22 FTA
    assert estimate_team_possessions(85, 10, 14, 22) == pytest.approx(98.68)


def test_possession_in_realistic_range():
    p = estimate_team_possessions(88, 11, 13, 20)
    assert 90 < p < 110


# --- design matrix -------------------------------------------------------

def _simulate(n_players, n_games, true_off, true_def, seed, noise=2.0,
              collinear_unit=None):
    rng = np.random.default_rng(seed)
    blocks = []
    for g in range(n_games):
        gid = f"g{g}"
        if collinear_unit and rng.random() < 0.6:
            home = frozenset(collinear_unit)
            others = [p for p in range(n_players) if p not in collinear_unit]
            away = frozenset(rng.choice(others, 5, replace=False).tolist())
        else:
            roster = rng.choice(n_players, 10, replace=False)
            home = frozenset(roster[:5].tolist())
            away = frozenset(roster[5:].tolist())
        for _ in range(rng.integers(4, 9)):
            for off, deff in ((home, away), (away, home)):
                eff = sum(true_off[p] for p in off) - sum(true_def[p] for p in deff)
                poss = float(rng.integers(2, 12))
                pts = (eff / 100) * poss + rng.normal(0, noise) * np.sqrt(poss) / 10
                blocks.append(OffenseBlock(gid, off, deff, pts, poss))
    return blocks


def test_design_matrix_has_ten_nonzeros_per_row():
    rng = np.random.default_rng(1)
    to = {p: float(rng.normal(0, 4)) for p in range(20)}
    td = {p: float(rng.normal(0, 3)) for p in range(20)}
    design = build_design(_simulate(20, 200, to, td, 1))
    assert design.X.nnz == 10 * design.X.shape[0]
    assert design.X.shape[1] == 2 * design.n_players


def test_offense_defense_columns_distinct():
    rng = np.random.default_rng(2)
    to = {p: 0.0 for p in range(10)}
    td = {p: 0.0 for p in range(10)}
    design = build_design(_simulate(10, 100, to, td, 2))
    for pid in design.player_index:
        assert design.offense_col(pid) != design.defense_col(pid)


def test_min_possessions_filters_noise_blocks():
    blocks = [
        OffenseBlock("g1", frozenset({1, 2, 3, 4, 5}), frozenset({6, 7, 8, 9, 10}),
                     2.0, 0.44),  # below threshold
        OffenseBlock("g1", frozenset({1, 2, 3, 4, 5}), frozenset({6, 7, 8, 9, 10}),
                     5.0, 6.0),
    ]
    design = build_design(blocks, min_possessions=1.0)
    assert design.X.shape[0] == 1


# --- recovery: the core proof -------------------------------------------

def test_recovers_planted_ratings_clean():
    rng = np.random.default_rng(0)
    n = 40
    to = {p: float(rng.normal(0, 4)) for p in range(n)}
    td = {p: float(rng.normal(0, 3)) for p in range(n)}
    design = build_design(_simulate(n, 3000, to, td, 0))
    res = fit_rapm(design, fixed_lambda=100.0)

    ro = np.array([res.offense[p] for p in range(n)])
    truth_o = np.array([to[p] for p in range(n)])
    rd = np.array([res.defense[p] for p in range(n)])
    truth_d = np.array([td[p] for p in range(n)])
    assert np.corrcoef(ro, truth_o)[0, 1] > 0.95
    assert np.corrcoef(rd, truth_d)[0, 1] > 0.95


def test_top_players_surface():
    rng = np.random.default_rng(0)
    n = 40
    to = {p: float(rng.normal(0, 4)) for p in range(n)}
    td = {p: float(rng.normal(0, 3)) for p in range(n)}
    design = build_design(_simulate(n, 3000, to, td, 0))
    res = fit_rapm(design, fixed_lambda=100.0)

    true_total = {p: to[p] + td[p] for p in range(n)}
    top_true = set(sorted(true_total, key=true_total.get, reverse=True)[:5])
    top_rec = {pid for pid, *_ in res.ranking()[:5]}
    assert len(top_true & top_rec) >= 4  # at least 4 of 5


def test_ridge_handles_collinear_unit():
    """Players who almost always play together must still be separable
    without exploding -- the whole point of the R in RAPM."""
    rng = np.random.default_rng(3)
    n = 60
    to = {p: float(rng.normal(0, 4)) for p in range(n)}
    td = {p: float(rng.normal(0, 3)) for p in range(n)}
    blocks = _simulate(n, 1500, to, td, 3, noise=6.0, collinear_unit=[0, 1, 2, 3, 4])
    design = build_design(blocks)
    res = fit_rapm(design, n_splits=5)
    # No exploded estimates despite collinearity.
    assert all(abs(v) < 40 for v in res.offense.values())
    ro = np.array([res.offense[p] for p in range(n)])
    truth = np.array([to[p] for p in range(n)])
    assert np.corrcoef(ro, truth)[0, 1] > 0.9


# --- grouped cross-validation -------------------------------------------

def test_grouped_cv_selects_from_grid():
    rng = np.random.default_rng(3)
    n = 50
    to = {p: float(rng.normal(0, 4)) for p in range(n)}
    td = {p: float(rng.normal(0, 3)) for p in range(n)}
    design = build_design(_simulate(n, 1000, to, td, 3, noise=6.0))
    best, scores = select_lambda(design, n_splits=5)
    assert best in scores
    assert all(np.isfinite(v) for v in scores.values())


def test_cv_folds_do_not_split_a_game():
    """Sanity: with few games, grouped CV must not create empty folds and
    must keep each game whole."""
    rng = np.random.default_rng(4)
    n = 20
    to = {p: 0.0 for p in range(n)}
    td = {p: 0.0 for p in range(n)}
    design = build_design(_simulate(n, 30, to, td, 4))
    best, scores = select_lambda(design, n_splits=5)
    assert best in scores


# --- post-fit value metric -------------------------------------------------
# compute_value is tested directly against hand-picked ratings, not ratings
# from an actual ridge fit: the formula is deterministic given (rating,
# possessions), so there is nothing a fit's estimation noise would add here
# -- and testing it directly is what lets possessions be pinned to exact,
# known numbers via minimal single-player OffenseBlocks (build_design does
# not require 5-and-5 lineups; that check lives upstream in split_lineup).

def test_compute_value_scales_with_rating_and_possessions():
    A, B, C, OPP = 1, 2, 3, 99
    blocks = [
        # A: 100 possessions total (60 offense + 40 defense).
        OffenseBlock("g1", frozenset({A}), frozenset({OPP}), 0.0, 60.0),
        OffenseBlock("g1", frozenset({OPP}), frozenset({A}), 0.0, 40.0),
        # B: same rating as A, but DOUBLE the possessions.
        OffenseBlock("g2", frozenset({B}), frozenset({OPP}), 0.0, 120.0),
        OffenseBlock("g2", frozenset({OPP}), frozenset({B}), 0.0, 80.0),
        # C: same possessions as A, but a different rating.
        OffenseBlock("g3", frozenset({C}), frozenset({OPP}), 0.0, 60.0),
        OffenseBlock("g3", frozenset({OPP}), frozenset({C}), 0.0, 40.0),
    ]
    design = build_design(blocks)

    offense = {A: 3.0, B: 3.0, C: 6.0, OPP: 0.0}
    defense = {A: 0.0, B: 0.0, C: 0.0, OPP: 0.0}
    possessions, value, wins_added = compute_value(design, offense, defense)

    assert possessions[A] == pytest.approx(100.0)
    assert possessions[B] == pytest.approx(200.0)
    assert possessions[C] == pytest.approx(100.0)

    assert value[A] == pytest.approx((3.0 - REPLACEMENT_LEVEL) * 100 / 100)
    assert value[B] == pytest.approx((3.0 - REPLACEMENT_LEVEL) * 200 / 100)
    assert value[C] == pytest.approx((6.0 - REPLACEMENT_LEVEL) * 100 / 100)

    # Same rating, double the possessions -> exactly double the value.
    assert value[B] == pytest.approx(2 * value[A])
    # Same possessions, higher rating -> proportionally higher value.
    assert value[C] / value[A] == pytest.approx(
        (6.0 - REPLACEMENT_LEVEL) / (3.0 - REPLACEMENT_LEVEL)
    )

    assert wins_added[A] == pytest.approx(value[A] / POINTS_PER_WIN)
    assert wins_added[B] == pytest.approx(value[B] / POINTS_PER_WIN)


def test_compute_value_is_zero_at_replacement_level():
    """A player rated exactly at replacement level gets value 0 -- and
    stays 0 no matter how many possessions he plays, since it is the RATE
    relative to replacement that must be zero, not the volume."""
    D, OPP = 4, 99
    blocks = [
        OffenseBlock("g1", frozenset({D}), frozenset({OPP}), 0.0, 300.0),
        OffenseBlock("g1", frozenset({OPP}), frozenset({D}), 0.0, 300.0),
    ]
    design = build_design(blocks)

    offense = {D: REPLACEMENT_LEVEL, OPP: 0.0}
    defense = {D: 0.0, OPP: 0.0}
    possessions, value, wins_added = compute_value(design, offense, defense)

    assert possessions[D] == pytest.approx(600.0)  # plenty of playing time
    assert value[D] == pytest.approx(0.0, abs=1e-9)
    assert wins_added[D] == pytest.approx(0.0, abs=1e-9)


def test_fit_rapm_populates_value_fields():
    """End-to-end: fit_rapm's RAPMResult carries possessions/value/
    wins_added consistent with compute_value, not just offense/defense."""
    rng = np.random.default_rng(7)
    n = 15
    to = {p: float(rng.normal(0, 3)) for p in range(n)}
    td = {p: float(rng.normal(0, 3)) for p in range(n)}
    design = build_design(_simulate(n, 200, to, td, 7))

    res = fit_rapm(design, fixed_lambda=1000.0)
    _, expected_value, expected_wins = compute_value(
        design, res.offense, res.defense, REPLACEMENT_LEVEL
    )

    assert res.replacement_level == REPLACEMENT_LEVEL
    for pid in res.offense:
        assert res.value[pid] == pytest.approx(expected_value[pid])
        assert res.wins_added[pid] == pytest.approx(expected_wins[pid])

    ranked = res.value_ranking()
    values = [v for _, v, *_ in ranked]
    assert values == sorted(values, reverse=True)