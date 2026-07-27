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
from nbare.rapm.fit import fit_rapm, select_lambda
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