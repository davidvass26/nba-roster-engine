"""Layer 2 recovery tests: hierarchical partial pooling of aging curves.

Three claims to prove, matching the spec's layer-2 acceptance criteria:
1. The model recovers each position's planted curve shape.
2. Sparse players (1-2 seasons) get pulled toward their position curve and
   carry visibly more posterior uncertainty than rich players.
3. Rich players' (8+ seasons) own recovered curve tracks their own true
   (idiosyncratic) trajectory better than the flat position curve does --
   i.e. their own data is allowed to dominate, not just get averaged away.

MCMC is fit once per module (expensive) and reused across assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.projection.hierarchy import fit_hierarchical_curve
from nbare.projection.synthetic import (
    POSITION_SHAPE_PARAMS,
    make_position_curve_players,
)


def _true_level(pos: str, age: int, anchor_age: int) -> float:
    p = POSITION_SHAPE_PARAMS[pos]
    return p["linear"] * (age - anchor_age) - p["curvature"] * (age - anchor_age) ** 2


@pytest.fixture(scope="module")
def fitted():
    ds = make_position_curve_players(n_players_per_position=18, seed=5)
    result = fit_hierarchical_curve(
        ds.seasons, "off_rating", "off_possessions",
        age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
        num_warmup=600, num_samples=600, num_chains=2, seed=0,
    )
    return ds, result


def test_recovers_position_curve_shapes(fitted):
    ds, result = fitted
    for pos in ("G", "F", "C"):
        samples = result.position_curve_samples(pos)  # (S, K)
        mean_curve = samples.mean(axis=0)
        errs = []
        for k, age in enumerate(result.age_grid):
            planted = _true_level(pos, age, ds.anchor_age)
            errs.append(abs(mean_curve[k] - planted))
        assert np.mean(errs) < 1.2, f"position {pos} mean curve error too high"


def test_position_curves_are_distinguishable(fitted):
    """Guards (slow decline) and centers (fast decline) must recover as
    visibly different shapes -- if the model over-pools, they'd collapse to
    the same league-average curve."""
    _, result = fitted
    g_curve = result.position_curve_samples("G").mean(axis=0)
    c_curve = result.position_curve_samples("C").mean(axis=0)
    old_idx = [k for k, a in enumerate(result.age_grid) if a >= 34]
    assert old_idx
    # Guards should be well above centers at old ages (slower decline).
    assert np.mean(g_curve[old_idx]) > np.mean(c_curve[old_idx]) + 1.5


def test_sparse_players_shrink_toward_position_curve(fitted):
    ds, result = fitted
    sparse = [p for p in ds.sparse_player_ids if p in result.player_index]
    assert len(sparse) >= 5

    errs = []
    for pid in sparse:
        pos = ds.player_position[pid]
        own_shift = ds.player_own_shift[pid]
        # player_relative_curve_samples excludes the player's baseline
        # (quality) -- it is a pure curve shape, pinned to 0 at the anchor
        # age -- so the comparison target must be too.
        player_mean = result.player_relative_curve_samples(pid).mean(axis=0)
        for k, age in enumerate(result.age_grid):
            true_val = _true_level(pos, age, ds.anchor_age) + own_shift * (age - ds.anchor_age)
            errs.append(player_mean[k] - true_val)
    # No systematic bias, and shrinkage keeps sparse players' curves
    # reasonably close to their (small-deviation) true curve.
    assert abs(np.mean(errs)) < 1.5
    assert np.std(errs) < 3.0


def test_sparse_players_have_wider_posterior_than_rich(fitted):
    ds, result = fitted
    sparse = [p for p in ds.sparse_player_ids if p in result.player_index]
    rich = [p for p in ds.rich_player_ids if p in result.player_index]
    assert sparse and rich

    def avg_posterior_sd(ids):
        sds = []
        for pid in ids:
            curve_samples = result.player_curve_samples(pid)  # (S, K)
            sds.append(curve_samples.std(axis=0).mean())
        return float(np.mean(sds))

    sparse_sd = avg_posterior_sd(sparse)
    rich_sd = avg_posterior_sd(rich)
    assert sparse_sd > rich_sd


def test_rich_players_own_curve_beats_flat_position_curve(fitted):
    """A rich player's recovered trajectory should track his own true
    (idiosyncratic) curve better than the plain position curve does --
    own data must be allowed to dominate when there's enough of it."""
    ds, result = fitted
    rich = [p for p in ds.rich_player_ids if p in result.player_index]
    assert len(rich) >= 5

    own_err, position_err = [], []
    for pid in rich:
        pos = ds.player_position[pid]
        own_shift = ds.player_own_shift[pid]
        if abs(own_shift) < 0.05:
            continue  # too small a deviation to be a meaningful check
        player_mean = result.player_relative_curve_samples(pid).mean(axis=0)
        position_mean = result.position_curve_samples(pos).mean(axis=0)
        for k, age in enumerate(result.age_grid):
            true_val = _true_level(pos, age, ds.anchor_age) + own_shift * (age - ds.anchor_age)
            own_err.append((player_mean[k] - true_val) ** 2)
            position_err.append((position_mean[k] - true_val) ** 2)

    assert np.mean(own_err) < np.mean(position_err)


def test_r_hat_convergence(fitted):
    _, result = fitted
    r_hats = result.r_hat_summary()
    assert r_hats  # non-empty
    for site, r_hat in r_hats.items():
        assert r_hat < 1.05, f"{site} failed to converge: r_hat={r_hat}"
