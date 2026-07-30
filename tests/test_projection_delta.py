"""Layer 1 recovery test: the delta method's aging curve.

The headline test, `test_delta_method_immune_to_survivorship`, is the whole
justification for this module. It proves the delta method recovers the true
decline curve even though the synthetic data is built with the exact
contamination (skill-coupled career length) that breaks a naive
age-vs-rating regression -- and it checks the naive regression actually
fails on this same data, so the comparison is not a strawman.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.projection.delta import (
    compute_player_deltas,
    cumulative_curve,
    league_delta_curve,
)
from nbare.projection.synthetic import make_league_curve_players, true_level


def test_recovers_planted_delta_curve():
    ds = make_league_curve_players(n_players=400, seed=1)
    deltas = compute_player_deltas(ds.seasons, "off_rating", "off_possessions")
    recovered = league_delta_curve(deltas)

    common_ages = [a for a in ds.delta_curve if a in recovered]
    assert len(common_ages) >= (ds.age_max - ds.age_min) - 2

    errs = [abs(recovered[a] - ds.delta_curve[a]) for a in common_ages]
    assert np.mean(errs) < 0.35
    assert max(errs) < 1.0


def test_cumulative_curve_matches_planted_shape():
    ds = make_league_curve_players(n_players=400, seed=2)
    deltas = compute_player_deltas(ds.seasons, "off_rating", "off_possessions")
    recovered_deltas = league_delta_curve(deltas)
    curve = cumulative_curve(recovered_deltas, ds.anchor_age)

    for age in range(ds.age_min, ds.age_max + 1):
        planted = true_level(age, ds.anchor_age, curvature=0.08)
        assert curve[age] == pytest.approx(planted, abs=1.2)


def test_delta_method_immune_to_survivorship():
    """The core claim: despite career length depending on skill (bad players
    vanish young, good players persist old), the delta-method curve tracks
    the true decline at older ages -- while a naive cross-sectional mean
    rating by age does NOT (it stays flat/rising because only good players
    remain in the sample), demonstrating the bias this method avoids.
    """
    ds = make_league_curve_players(n_players=500, seed=3)
    deltas = compute_player_deltas(ds.seasons, "off_rating", "off_possessions")
    recovered = league_delta_curve(deltas)

    old_ages = [a for a in recovered if a >= 33]
    assert old_ages
    # True deltas at old ages are strongly negative (decline).
    true_old = [ds.delta_curve[a] for a in old_ages]
    assert np.mean(true_old) < -1.0
    # The delta method recovers that decline, not a flat/positive trend.
    recovered_old = [recovered[a] for a in old_ages]
    assert np.mean(recovered_old) < -0.5

    # Contrast: the naive cross-sectional mean rating by age, on this same
    # survivorship-laden data, is biased toward flat/rising at old ages.
    by_age: dict[int, list[float]] = {}
    for s in ds.seasons:
        by_age.setdefault(s.age, []).append(s.off_rating)
    naive_mean_by_age = {a: float(np.mean(v)) for a, v in by_age.items()}

    young_mean = np.mean([naive_mean_by_age[a] for a in range(22, 26)])
    old_mean = np.mean([naive_mean_by_age[a] for a in range(33, 38)])
    # The naive cross-section does NOT show the true decline -- it is flat
    # or still rising, because only the best players are left at old ages.
    assert old_mean > young_mean - 2.0


def test_skips_non_consecutive_age_gaps():
    ds = make_league_curve_players(n_players=50, seed=4)
    # Duplicate one player's rows with a manufactured 2-year gap to confirm
    # it is skipped rather than silently divided/guessed at.
    from dataclasses import replace
    p0_rows = [s for s in ds.seasons if s.player_id == 0]
    if len(p0_rows) < 2:
        pytest.skip("need at least 2 seasons for player 0 in this seed")
    gapped = [p0_rows[0], replace(p0_rows[1], age=p0_rows[0].age + 2)]
    deltas = compute_player_deltas(gapped, "off_rating", "off_possessions")
    assert deltas == []
