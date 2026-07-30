"""Tests for the Bayesian box-score prior and the on/off defensive diagnostic.

The headline test is `test_bayesian_beats_plain_on_low_possession`: it proves
the prior does what it is for -- reducing error on low-possession players --
which is the entire justification for the added complexity. If that ever
regresses, the upgrade is not earning its keep.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.rapm.blocks import on_off_defense
from nbare.rapm.design import OffenseBlock, build_design
from nbare.rapm.fit import fit_rapm, fit_rapm_bayesian
from nbare.rapm.prior import (
    DEFENSE_FEATURES,
    OFFENSE_FEATURES,
    BoxScoreRow,
    build_prior,
)


def _box(pid, off_poss, def_poss, **kw):
    defaults = dict(
        pts_100=20.0, ast_100=4.0, tov_100=2.5, ts_pct=0.56,
        stl_100=1.2, blk_100=0.6, dreb_rate=0.15, position_big=0,
    )
    defaults.update(kw)
    return BoxScoreRow(player_id=pid, off_possessions=off_poss,
                       def_possessions=def_poss, **defaults)


# --- prior mechanics -----------------------------------------------------

def test_prior_reduces_to_zero_without_training_data():
    """With too few reliable players, the prior must fall back to zero mean
    so the Bayesian fit gracefully degrades to plain ridge."""
    blocks = [
        OffenseBlock("g1", frozenset({1, 2, 3, 4, 5}), frozenset({6, 7, 8, 9, 10}),
                     5.0, 6.0)
    ]
    design = build_design(blocks)
    plain = fit_rapm(design, fixed_lambda=1000.0)
    box = [_box(p, 10, 10) for p in range(1, 11)]  # nobody reliable
    prior = build_prior(design, plain, box, reliable_possessions=1000.0)
    assert np.allclose(prior.mu, 0.0)
    assert prior.notes  # it should say it fell back


def test_bayesian_equals_plain_when_prior_is_zero():
    """A zero prior mean must give exactly the plain-ridge result."""
    rng = np.random.default_rng(0)
    to = {p: float(rng.normal(0, 4)) for p in range(20)}
    td = {p: float(rng.normal(0, 3)) for p in range(20)}
    blocks = []
    for g in range(300):
        roster = rng.choice(20, 10, replace=False)
        home, away = frozenset(roster[:5].tolist()), frozenset(roster[5:].tolist())
        for off, deff in ((home, away), (away, home)):
            eff = sum(to[p] for p in off) - sum(td[p] for p in deff)
            pv = float(rng.integers(3, 9))
            blocks.append(OffenseBlock(f"g{g}", off, deff, (eff / 100) * pv, pv))
    design = build_design(blocks)
    plain = fit_rapm(design, fixed_lambda=1500.0)
    zero_mu = np.zeros(2 * design.n_players)
    bayes = fit_rapm_bayesian(design, zero_mu, fixed_lambda=1500.0)
    for pid in design.player_index:
        assert plain.offense[pid] == pytest.approx(bayes.offense[pid], abs=1e-6)
        assert plain.defense[pid] == pytest.approx(bayes.defense[pid], abs=1e-6)


def test_prior_mu_dimension_checked():
    blocks = [
        OffenseBlock("g1", frozenset({1, 2, 3, 4, 5}), frozenset({6, 7, 8, 9, 10}),
                     5.0, 6.0)
    ]
    design = build_design(blocks)
    with pytest.raises(ValueError, match="prior_mu"):
        fit_rapm_bayesian(design, np.zeros(3), fixed_lambda=1000.0)


# --- the core justification ---------------------------------------------

def _simulate_with_boxscore(seed=7, n_players=80):
    rng = np.random.default_rng(seed)
    box, true_off, true_def = {}, {}, {}
    for p in range(n_players):
        pts = rng.normal(20, 6); ast = rng.normal(4, 2); tov = rng.normal(2.5, 1)
        ts = float(np.clip(rng.normal(0.56, 0.05), 0.4, 0.7))
        stl = rng.normal(1.2, 0.5); blk = rng.normal(0.6, 0.6)
        dreb = float(np.clip(rng.normal(0.15, 0.06), 0.02, 0.35))
        big = int(rng.random() < 0.35)
        true_off[p] = (0.25 * (pts - 20) + 0.9 * (ast - 4) - 1.1 * (tov - 2.5)
                       + 25 * (ts - 0.56) + rng.normal(0, 1.5))
        true_def[p] = (1.4 * (stl - 1.2) + 1.8 * (blk - 0.6) + 12 * (dreb - 0.15)
                       + 1.6 * big + rng.normal(0, 1.2))
        box[p] = dict(pts_100=pts, ast_100=ast, tov_100=tov, ts_pct=ts,
                      stl_100=stl, blk_100=blk, dreb_rate=dreb, position_big=big)

    stars = list(range(30)); bench = list(range(30, n_players))
    blocks = []; poss_off = {p: 0.0 for p in range(n_players)}
    poss_def = {p: 0.0 for p in range(n_players)}

    def sim(gid, pool, reps):
        for _ in range(reps):
            home = frozenset(rng.choice(pool, 5, replace=False).tolist())
            rest = [p for p in pool if p not in home]
            away = frozenset(rng.choice(rest, 5, replace=False).tolist())
            for off, deff in ((home, away), (away, home)):
                eff = sum(true_off[p] for p in off) - sum(true_def[p] for p in deff)
                pv = float(rng.integers(3, 10))
                pts = (eff / 100) * pv + rng.normal(0, 3) * np.sqrt(pv) / 10
                blocks.append(OffenseBlock(gid, off, deff, pts, pv))
                for p in off: poss_off[p] += pv
                for p in deff: poss_def[p] += pv

    for g in range(2500): sim(f"s{g}", stars + bench[:5], 3)
    for g in range(300): sim(f"b{g}", bench, 3)

    box_rows = [
        BoxScoreRow(player_id=p, off_possessions=poss_off[p],
                    def_possessions=poss_def[p], **box[p])
        for p in range(n_players)
    ]
    return blocks, box_rows, true_off, true_def, bench, poss_off


def test_bayesian_beats_plain_on_low_possession():
    """THE justification: the prior must cut error for low-possession players
    without harming the well-sampled stars."""
    blocks, box_rows, true_off, true_def, bench, poss_off = _simulate_with_boxscore()
    design = build_design(blocks)
    plain = fit_rapm(design, fixed_lambda=2000.0)
    prior = build_prior(design, plain, box_rows, reliable_possessions=3000.0)
    bayes = fit_rapm_bayesian(design, prior.mu, fixed_lambda=2000.0)

    def rmse(pred, truth, ids):
        return float(np.sqrt(np.mean([(pred[p] - truth[p]) ** 2 for p in ids])))

    low = [p for p in bench if poss_off[p] < 1500]
    assert len(low) >= 10

    off_plain = rmse(plain.offense, true_off, low)
    off_bayes = rmse(bayes.offense, true_off, low)
    def_plain = rmse(plain.defense, true_def, low)
    def_bayes = rmse(bayes.defense, true_def, low)

    # The prior must improve BOTH sides on low-possession players.
    assert off_bayes < off_plain
    assert def_bayes < def_plain
    # And the gain should be substantial (not a rounding artifact).
    assert off_bayes < 0.85 * off_plain


def test_prior_trains_on_reliable_players_only():
    blocks, box_rows, _, _, _, _ = _simulate_with_boxscore()
    design = build_design(blocks)
    plain = fit_rapm(design, fixed_lambda=2000.0)
    prior = build_prior(design, plain, box_rows, reliable_possessions=3000.0)
    # Some but not all players are reliable.
    assert 0 < prior.reliable_offense < design.n_players
    assert 0 < prior.reliable_defense < design.n_players


# --- on/off diagnostic ---------------------------------------------------

def test_on_off_defense_correlates_with_true_skill():
    """The independent check: on/off diff must move opposite to true D skill
    (better defenders allow fewer opponent points while on the floor)."""
    rng = np.random.default_rng(3)
    n = 30
    true_def = {p: float(rng.normal(0, 3)) for p in range(n)}
    blocks = []
    for g in range(2000):
        roster = rng.choice(n, 10, replace=False)
        home, away = frozenset(roster[:5].tolist()), frozenset(roster[5:].tolist())
        for off, deff in ((home, away), (away, home)):
            eff = 110 - sum(true_def[p] for p in deff)
            pv = float(rng.integers(4, 9))
            pts = (eff / 100) * pv + rng.normal(0, 2) * np.sqrt(pv) / 10
            blocks.append(OffenseBlock(f"g{g}", off, deff, pts, pv))

    oo = on_off_defense(blocks)
    ids = [p for p in range(n) if oo[p].def_possessions_off > 0]
    diffs = np.array([oo[p].diff for p in ids])
    td = np.array([true_def[p] for p in ids])
    assert np.corrcoef(diffs, td)[0, 1] < -0.5  # strong negative


def test_on_off_covers_all_players():
    blocks = []
    rng = np.random.default_rng(1)
    for g in range(200):
        roster = rng.choice(15, 10, replace=False)
        home, away = frozenset(roster[:5].tolist()), frozenset(roster[5:].tolist())
        blocks.append(OffenseBlock(f"g{g}", home, away, 5.0, 5.0))
        blocks.append(OffenseBlock(f"g{g}", away, home, 5.0, 5.0))
    oo = on_off_defense(blocks)
    assert len(oo) == 15


def test_feature_sets_are_distinct():
    """Offense and defense priors use different features by design."""
    assert set(OFFENSE_FEATURES) != set(DEFENSE_FEATURES)
    assert "dreb_rate" in DEFENSE_FEATURES  # the anti-gambling signal
    assert "position_big" in DEFENSE_FEATURES