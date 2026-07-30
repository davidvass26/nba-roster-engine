"""Layer 3 recovery tests: the box-score-informed baseline.

Mirrors `test_prior.py`'s headline test (`test_bayesian_beats_plain_on_low_
possession`) one level up the stack: the box-score baseline must recover
the planted box-score -> baseline relationship AND cut error for
low-data (sparse) players specifically, without needing a third redundant
MCMC fit -- `BoxScoreProjectionModel.prelim` is the zero-mean fit already
computed internally, reused here as the "plain" comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.projection.baseline import (
    DEFAULT_RELIABLE_SEASONS,
    fit_box_score_projection,
)
from nbare.projection.synthetic import make_boxscore_baseline_players
from nbare.rapm.prior import DEFENSE_FEATURES, OFFENSE_FEATURES


@pytest.fixture(scope="module")
def offense_fit():
    ds = make_boxscore_baseline_players(n_players_per_position=15, seed=9)
    model = fit_box_score_projection(
        ds.seasons, ds.box_rows, "off_rating", "off_possessions",
        age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
        prelim_num_warmup=300, prelim_num_samples=300,
        num_warmup=500, num_samples=500, num_chains=2, seed=0,
    )
    return ds, model


def test_uses_offense_features(offense_fit):
    _, model = offense_fit
    assert model.box_model.features == OFFENSE_FEATURES


def test_uses_defense_features():
    ds = make_boxscore_baseline_players(n_players_per_position=8, seed=10)
    model = fit_box_score_projection(
        ds.seasons, ds.box_rows, "def_rating", "def_possessions",
        age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
        prelim_num_warmup=200, prelim_num_samples=200,
        num_warmup=200, num_samples=200, num_chains=2, seed=0,
    )
    assert model.box_model.features == DEFENSE_FEATURES


def test_some_but_not_all_players_reliable(offense_fit):
    ds, model = offense_fit
    n_players = len(ds.true_off_baseline)
    assert 0 < model.reliable_n < n_players
    assert not model.notes  # should NOT have fallen back to zero mean


def test_recovers_box_score_baseline_relationship(offense_fit):
    """The box-score regression's predictions must correlate strongly with
    the PLANTED per-player offense baseline (not the noisy per-season
    ratings) -- this is the actual relationship layer 3 is supposed to
    learn."""
    ds, model = offense_fit
    ids = list(ds.true_off_baseline.keys())
    rows = [ds.box_rows[pid] for pid in ids]
    predicted = model.box_model.predict(rows)
    true = np.array([ds.true_off_baseline[pid] for pid in ids])

    assert np.corrcoef(predicted, true)[0, 1] > 0.7
    # Predictions should not be systematically biased.
    assert abs(np.mean(predicted - true)) < 1.5


def test_sparse_players_benefit_from_box_score_baseline(offense_fit):
    """THE justification, mirrored from rapm/prior.py: box-score-informed
    baselines must beat the zero-mean (plain hierarchical) baseline for
    sparse (low-data) players."""
    ds, model = offense_fit
    sparse = [p for p in ds.sparse_player_ids if p in model.result.player_index]
    assert len(sparse) >= 5

    def baseline_error(result):
        errs = []
        for pid in sparse:
            i = result.player_index[pid]
            pred = float(result.samples["baseline"][:, i].mean())
            errs.append((pred - ds.true_off_baseline[pid]) ** 2)
        return float(np.mean(errs))

    plain_mse = baseline_error(model.prelim)
    boxscore_mse = baseline_error(model.result)
    assert boxscore_mse < plain_mse


def test_r_hat_convergence(offense_fit):
    _, model = offense_fit
    r_hats = model.result.r_hat_summary()
    assert r_hats
    for site, r_hat in r_hats.items():
        assert r_hat < 1.05, f"{site} failed to converge: r_hat={r_hat}"


def test_posterior_predictive_coverage(offense_fit):
    _, model = offense_fit
    coverage = model.result.posterior_predictive_coverage(prob=0.9)
    assert 0.75 < coverage <= 1.0


def test_reliable_seasons_constant_is_sane():
    assert DEFAULT_RELIABLE_SEASONS >= 2
