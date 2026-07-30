"""Tests for the top-level projection output (project.py).

This is plumbing over already-validated statistics (layers 1-3), so the
fixture here uses a small, cheap fit -- these tests check SHAPE and the
honesty/truncation contract, not recovery accuracy (that is layers 1-3's
job).
"""

from __future__ import annotations

import numpy as np
import pytest

from nbare.projection.baseline import fit_box_score_projection
from nbare.projection.project import project_players
from nbare.projection.synthetic import make_boxscore_baseline_players


@pytest.fixture(scope="module")
def models():
    ds = make_boxscore_baseline_players(n_players_per_position=8, seed=11)
    offense = fit_box_score_projection(
        ds.seasons, ds.box_rows, "off_rating", "off_possessions",
        age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
        prelim_num_warmup=150, prelim_num_samples=150,
        num_warmup=150, num_samples=150, num_chains=2, seed=0,
    )
    defense = fit_box_score_projection(
        ds.seasons, ds.box_rows, "def_rating", "def_possessions",
        age_min=ds.age_min, age_max=ds.age_max, anchor_age=ds.anchor_age,
        prelim_num_warmup=150, prelim_num_samples=150,
        num_warmup=150, num_samples=150, num_chains=2, seed=0,
    )
    return ds, offense, defense


def test_projects_every_common_player_within_range(models):
    ds, offense, defense = models
    # Pick current ages well inside the fitted grid so nothing truncates.
    current_age = {pid: ds.age_min for pid in ds.true_off_baseline
                   if pid in offense.result.player_index and pid in defense.result.player_index}
    projections = project_players(offense, defense, current_age, years_ahead=4)

    assert projections
    for pid, proj in projections.items():
        assert proj.future_ages == tuple(range(ds.age_min + 1, ds.age_min + 5))
        n_samples_off = offense.result.samples["baseline"].shape[0]
        assert proj.offense_samples.shape == (n_samples_off, 4)
        assert proj.defense_samples.shape == (n_samples_off, 4)


def test_truncates_rather_than_extrapolates_past_age_max(models):
    """The honesty contract: a player near age_max must get FEWER future
    years, never a fabricated curve value beyond the fitted grid."""
    ds, offense, defense = models
    pid = next(iter(set(offense.result.player_index) & set(defense.result.player_index)))
    current_age = {pid: ds.age_max - 1}  # only age_max itself is coverable
    projections = project_players(offense, defense, current_age, years_ahead=4)

    assert pid in projections
    assert projections[pid].future_ages == (ds.age_max,)
    assert projections[pid].offense_samples.shape[1] == 1


def test_drops_players_entirely_past_the_fitted_grid(models):
    ds, offense, defense = models
    pid = next(iter(set(offense.result.player_index) & set(defense.result.player_index)))
    current_age = {pid: ds.age_max}  # age_max + 1 is out of range entirely
    projections = project_players(offense, defense, current_age, years_ahead=4)
    assert pid not in projections


def test_drops_players_missing_from_either_model(models):
    ds, offense, defense = models
    current_age = {999999: ds.age_min}  # not a real player in either fit
    projections = project_players(offense, defense, current_age, years_ahead=4)
    assert not projections


def test_offense_and_defense_columns_are_coherent_draws(models):
    """Column k of offense_samples and column k of defense_samples must
    come from the SAME posterior draw index, not independent shuffles."""
    ds, offense, defense = models
    pid = next(iter(set(offense.result.player_index) & set(defense.result.player_index)))
    current_age = {pid: ds.age_min}
    proj = project_players(offense, defense, current_age, years_ahead=1)[pid]

    off_full = offense.result.player_curve_samples(pid)
    def_full = defense.result.player_curve_samples(pid)
    off_age_pos = {a: k for k, a in enumerate(offense.result.age_grid)}
    def_age_pos = {a: k for k, a in enumerate(defense.result.age_grid)}
    target_age = ds.age_min + 1

    np.testing.assert_array_equal(
        proj.offense_samples[:, 0], off_full[:, off_age_pos[target_age]]
    )
    np.testing.assert_array_equal(
        proj.defense_samples[:, 0], def_full[:, def_age_pos[target_age]]
    )
