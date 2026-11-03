"""Tests for the box-prior vs RAPM disagreement report.

The headline test plants one player the box score underrates and one it
overrates, and requires the report to find both, in the right direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.rapm.design import OffenseBlock, build_design
from nbare.rapm.fit import fit_rapm, fit_rapm_bayesian, rapm_standard_errors
from nbare.rapm.prior_gap import PlayerGap, PriorGaps, SideGap, prior_gaps

N_PLAYERS = 40
UNDERRATED, OVERRATED = 0, 1
PLANTED_GAP = 6.0


def _planted_fit(*, no_box: frozenset[int] = frozenset()):
    """Everyone's true offense equals their box-implied rating, except the
    two planted players, who differ by +/- PLANTED_GAP. Defense agrees for
    everyone. `no_box` players get the 0.0 "no box row" prior mean."""
    rng = np.random.default_rng(11)
    box_off = {p: float(rng.normal(0, 2)) for p in range(N_PLAYERS)}
    box_def = {p: float(rng.normal(0, 1.5)) for p in range(N_PLAYERS)}
    true_off = dict(box_off)
    true_off[UNDERRATED] += PLANTED_GAP
    true_off[OVERRATED] -= PLANTED_GAP

    blocks = []
    for g in range(3000):
        roster = rng.choice(N_PLAYERS, 10, replace=False)
        home, away = frozenset(roster[:5].tolist()), frozenset(roster[5:].tolist())
        for off, deff in ((home, away), (away, home)):
            eff = sum(true_off[p] for p in off) - sum(box_def[p] for p in deff)
            pv = float(rng.integers(3, 10))
            pts = (eff / 100) * pv + rng.normal(0, 3) * np.sqrt(pv) / 10
            blocks.append(OffenseBlock(f"g{g}", off, deff, pts, pv))
    design = build_design(blocks)

    mu = np.zeros(2 * design.n_players)
    for pid, i in design.player_index.items():
        if pid not in no_box:
            mu[2 * i], mu[2 * i + 1] = box_off[pid], box_def[pid]
    return design, fit_rapm_bayesian(design, mu, fixed_lambda=500.0)


def test_recovers_planted_disagreement_in_both_directions():
    design, fit = _planted_fit()
    ranked = prior_gaps(design, fit).ranked(side="offense")

    assert ranked[0].player_id == UNDERRATED
    assert ranked[-1].player_id == OVERRATED
    assert ranked[0].offense.gap > 0.5 * PLANTED_GAP
    assert ranked[-1].offense.gap < -0.5 * PLANTED_GAP
    # Nobody else's gap comes close: the planted pair stand alone.
    others = [abs(p.offense.gap) for p in ranked[1:-1]]
    assert max(others) < 0.5 * abs(ranked[0].offense.gap)


def test_z_separates_planted_players_from_the_field():
    design, fit = _planted_fit()
    by_z = prior_gaps(design, fit).ranked(side="offense", by="z")
    assert by_z[0].player_id == UNDERRATED and by_z[0].offense.z > 3
    assert by_z[-1].player_id == OVERRATED and by_z[-1].offense.z < -3


def test_total_sums_sides_and_combines_se_in_quadrature():
    p = PlayerGap(1, 100.0, SideGap(3.0, 1.0, 0.3), SideGap(1.0, 2.0, 0.4))
    assert p.total.rapm == pytest.approx(4.0)
    assert p.total.prior == pytest.approx(3.0)
    assert p.total.gap == pytest.approx(1.0)
    assert p.total.se == pytest.approx(0.5)
    assert p.total.z == pytest.approx(2.0)


def test_players_without_a_box_prior_are_left_out_not_compared_to_zero():
    """mu = 0.0 is a default, not a prediction: a planted star with no box
    row must not surface as a huge gap."""
    design, fit = _planted_fit(no_box=frozenset({UNDERRATED, 5, 6}))
    gaps = prior_gaps(design, fit)
    assert gaps.n_no_box_prior == 3
    listed = {p.player_id for p in gaps.players}
    assert listed.isdisjoint({UNDERRATED, 5, 6})
    assert len(listed) == N_PLAYERS - 3


def test_se_is_the_posterior_sd_per_side():
    design, fit = _planted_fit()
    off_se, def_se = rapm_standard_errors(design, fit)
    for p in prior_gaps(design, fit).players:
        assert p.offense.se == off_se[p.player_id]
        assert p.defense.se == def_se[p.player_id]


def test_min_possessions_filters_and_bad_options_raise():
    low = PlayerGap(1, 10.0, SideGap(1.0, 0.0, 1.0), SideGap(1.0, 0.0, 1.0))
    high = PlayerGap(2, 5000.0, SideGap(0.5, 0.0, 1.0), SideGap(0.0, 0.0, 1.0))
    gaps = PriorGaps(players=[low, high], n_no_box_prior=0)
    assert [p.player_id for p in gaps.ranked(min_possessions=100.0)] == [2]
    with pytest.raises(ValueError, match="side"):
        gaps.ranked(side="both")
    with pytest.raises(ValueError, match="by"):
        gaps.ranked(by="size")


def test_plain_ridge_result_is_rejected():
    """A zero-prior fit has no box score to compare against."""
    design, _ = _planted_fit()
    with pytest.raises(ValueError, match="prior_mean"):
        prior_gaps(design, fit_rapm(design, fixed_lambda=500.0))
