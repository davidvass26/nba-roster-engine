"""Known-measurement-error mode of the trajectory hierarchy (layer 2 +
`se_attr`).

Why this layer exists: on the real 2023-26 panel (<=3 seasons per player)
the learned-noise model could not separate true season-to-season talent
change (sigma_player) from rating noise (obs_sigma) -- R-hat ~1.2 on both,
worse with longer chains. Feeding the single-season RAPM fit's own
posterior SDs as known errors should make sigma_player identifiable.
These tests plant exactly that regime (`make_known_se_panel`) and require:
convergence, recovery of the planted sigma_player, recovery of the
position curves, and calibrated OUT-OF-SAMPLE predictive coverage (each
player's last season held out, then predicted).
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.projection.hierarchy import fit_hierarchical_curve
from nbare.projection.synthetic import make_known_se_panel


def _split_last_season(seasons):
    last = {}
    for r in seasons:
        if r.player_id not in last or r.age > last[r.player_id].age:
            last[r.player_id] = r
    held = set(id(r) for r in last.values())
    return [r for r in seasons if id(r) not in held], list(last.values())


@pytest.fixture(scope="module")
def fitted():
    ds = make_known_se_panel(seed=3)
    train, holdout = _split_last_season(ds.seasons)
    res = fit_hierarchical_curve(
        train, "off_rating", "off_possessions",
        age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
        num_warmup=1500, num_samples=1500, num_chains=4, seed=0,
        se_attr="off_se",
    )
    return ds, res, holdout


def test_known_se_mode_converges(fitted):
    """sigma_player mixes slowly with ~1.5 training seasons per player
    (ESS ~25 at 2x800 draws); 4 chains x 1500 is what reliably clears the
    bar, and is the CLI's default for the same reason."""
    _, res, _ = fitted
    rhat = res.r_hat_summary()
    assert "obs_sigma" not in rhat  # noise is known, not sampled
    assert max(v for v in rhat.values() if v == v) < 1.05, rhat


def test_recovers_planted_sigma_player(fitted):
    ds, res, _ = fitted
    s = res.samples["sigma_player"]
    lo, hi = np.quantile(s, [0.05, 0.95])
    assert lo <= ds.sigma_player <= hi, (lo, hi)
    assert abs(float(np.mean(s)) - ds.sigma_player) < 0.15


def test_recovers_position_curve_shapes(fitted):
    ds, res, _ = fitted
    for pos, curve in ds.delta_curve_by_position.items():
        est = res.samples["position_delta"][:, res.position_index[pos], :].mean(axis=0)
        truth = np.array([curve[a] for a in res.age_grid[:-1]])
        # Mid-grid ages are well covered; edges have few players.
        mid = slice(2, len(truth) - 2)
        assert np.corrcoef(est[mid], truth[mid])[0, 1] > 0.8, pos


def test_holdout_predictive_coverage_is_calibrated(fitted):
    """Each player's LAST season was held out; predicting it from the
    earlier ones must land inside the 80% interval ~80% of the time. This
    is the property the app's 80% bands promise."""
    _, res, holdout = fitted
    cov, n = res.holdout_coverage(holdout, "off_rating", "off_se", prob=0.8)
    assert n > 300
    assert 0.74 <= cov <= 0.86, cov


def test_in_sample_coverage_overstates_calibration(fitted):
    """Documents WHY the holdout check is the one used: on the same
    correctly-specified model, in-sample coverage runs well above nominal."""
    _, res, holdout = fitted
    in_sample = res.posterior_predictive_coverage(prob=0.8)
    out_sample, _ = res.holdout_coverage(holdout, "off_rating", "off_se", prob=0.8)
    assert in_sample > out_sample


def test_missing_se_raises_rather_than_guessing():
    ds = make_known_se_panel(n_players_per_position=3, seed=1)
    from dataclasses import replace

    bad = [replace(ds.seasons[0], off_se=None)] + ds.seasons[1:]
    with pytest.raises(ValueError):
        fit_hierarchical_curve(
            bad, "off_rating", "off_possessions",
            age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
            num_warmup=5, num_samples=5, num_chains=1, se_attr="off_se",
        )


# --- learned baseline spread + smoothed league curve (real-data settings) -----

def _league_curve_error(res, ds):
    """RMSE of the posterior-mean league delta curve vs the planted mean of
    the three position curves (the league level of the hierarchy)."""
    est = res.samples["league_delta"].mean(axis=0)
    truth = np.array([np.mean([ds.delta_curve_by_position[p][a] for p in ds.delta_curve_by_position])
                      for a in res.age_grid[:-1]])
    return float(np.sqrt(np.mean((est - truth) ** 2)))


@pytest.fixture(scope="module")
def thin_edge():
    ds = make_known_se_panel(seed=5, edge_thin=True, baseline_sd=1.5, se_range=(1.0, 3.0))
    kw = dict(age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
              num_warmup=1000, num_samples=1000, num_chains=4, seed=0, se_attr="off_se")
    learned = fit_hierarchical_curve(ds.seasons, "off_rating", "off_possessions",
                                     baseline_prior_sd=None, smooth_league_curve=True, **kw)
    fixed = fit_hierarchical_curve(ds.seasons, "off_rating", "off_possessions",
                                   baseline_prior_sd=6.0, smooth_league_curve=False, **kw)
    return ds, learned, fixed


def test_learned_baseline_spread_recovers_planted_sd_and_converges(thin_edge):
    ds, learned, _ = thin_edge
    rhat = learned.r_hat_summary()
    assert max(v for v in rhat.values() if v == v) < 1.05, rhat
    s = learned.samples["sigma_baseline"]
    lo, hi = np.quantile(s, [0.05, 0.95])
    assert lo <= ds.baseline_sd <= hi, (lo, hi)


def test_learned_spread_shrinks_noisy_levels_toward_truth(thin_edge):
    """With noisy measurements (SE 1-3, true spread 1.5) a fixed wide
    baseline prior (sd 6) barely shrinks, so estimated levels spread far
    wider than the truth -- the real-data symptom. The learned spread must
    get both the spread and the per-player error closer to the planted
    truth."""
    ds, learned, fixed = thin_edge

    def levels(res):
        age_pos = {a: k for k, a in enumerate(res.age_grid)}
        keys = sorted(ds.true_level)
        est = np.array([res.player_curve_samples(p)[:, age_pos[a]].mean() for p, a in keys])
        return est, np.array([ds.true_level[k] for k in keys])

    est_l, truth = levels(learned)
    est_f, _ = levels(fixed)
    rmse = lambda e: float(np.sqrt(np.mean((e - truth) ** 2)))  # noqa: E731
    assert rmse(est_l) < rmse(est_f)
    assert abs(est_l.std() - truth.std()) < abs(est_f.std() - truth.std())


def test_smoothed_league_curve_is_closer_to_truth_at_thin_edges(thin_edge):
    ds, learned, fixed = thin_edge
    assert _league_curve_error(learned, ds) < _league_curve_error(fixed, ds)
