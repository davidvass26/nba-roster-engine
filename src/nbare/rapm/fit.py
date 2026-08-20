"""Fit offense/defense-split RAPM by ridge regression with grouped CV.

Ridge, not least squares
------------------------
Plain adjusted plus-minus (unregularized least squares) is famously
unstable: collinear lineups let the solver hand one player +40 and his
teammate -35 to fit noise, and low-minute players get wild estimates from
tiny samples. Ridge adds a penalty lambda * ||beta||^2, which shrinks
ratings toward zero (league average) unless the data demands otherwise.

The Bayesian reading, worth stating: ridge IS a Gaussian prior that every
player is league-average until proven otherwise. A player with thousands of
possessions overwhelms the prior; one with fifty stays near zero. That is
the mathematically correct treatment of "I barely have data on this guy,"
not a hack. (Stage 2b will replace the zero-mean prior with a box-score-
informed one; this module is the zero-mean baseline.)

Why grouped cross-validation for lambda
---------------------------------------
lambda controls shrinkage and must be chosen, not guessed. We choose it by
cross-validation: try a grid, keep the lambda that best predicts held-out
efficiency. The critical subtlety: split by GAME, never by random row.
Stints from the same game share lineups, pace, and officiating; if stints
from one game land in both train and test, the model "cheats" and CV picks
too small a lambda. GroupKFold on game_id prevents that leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from nbare.rapm.design import DesignMatrix

# A geometric grid spanning "barely any shrinkage" to "heavy shrinkage".
# RAPM's useful lambda is large because the per-row signal is weak; the grid
# is centered accordingly and can be overridden.
DEFAULT_LAMBDA_GRID: tuple[float, ...] = (
    100.0, 300.0, 1000.0, 2000.0, 3000.0, 5000.0, 8000.0, 15000.0, 30000.0,
)

# Replacement level: the RAPM rating of a readily-replaceable player (a
# min-salary free agent / end-of-bench call-up) -- the baseline "value"
# measures ABOVE. -2.0 is BPM's published replacement-level convention; we
# default half a point lower as a starting point, since this is a
# possession-weighted ridge fit on lineup data, not a box-score regression,
# and its shrinkage/variance behavior is not the same as BPM's. This has
# NOT been empirically calibrated against this project's own fits (e.g. by
# checking where confirmed replacement-level players actually land in a
# real fit's rating distribution) -- it is a configurable placeholder, not
# a derived constant. Override via `compute_value`'s `replacement_level`
# argument once real calibration is done.
REPLACEMENT_LEVEL: float = -2.5

# Points-per-win: the standard sabermetrics-style conversion (also used by
# box-score value metrics like VORP) treating ~30 points of value over a
# season as worth one added win. Same caveat as REPLACEMENT_LEVEL: a
# borrowed convention, not calibrated against this RAPM's own scale.
POINTS_PER_WIN: float = 30.0


def _player_possessions(design: DesignMatrix) -> dict[int, float]:
    """Total possessions each player was on the floor for: the sum of
    block possession weights (`design.w`) over every design row where
    they appear, offense or defense combined. Reads directly off the same
    design matrix the ridge fit on -- not a fresh pass over blocks -- so
    this can never disagree with what the model actually saw.
    """
    Xc = design.X.tocsc()
    poss: dict[int, float] = {}
    for pid, i in design.player_index.items():
        off_rows = Xc.getcol(2 * i).indices
        def_rows = Xc.getcol(2 * i + 1).indices
        poss[pid] = float(design.w[off_rows].sum() + design.w[def_rows].sum())
    return poss


def compute_value(
    design: DesignMatrix,
    offense: dict[int, float],
    defense: dict[int, float],
    replacement_level: float = REPLACEMENT_LEVEL,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Post-fit value metric. Returns (possessions, value, wins_added).

        value = (total_rapm - replacement_level) * possessions / 100
        wins_added = value / POINTS_PER_WIN

    This turns the RATE stat (RAPM, points per 100 possessions) into a
    VOLUME stat by multiplying by playing time -- the same rate-times-
    exposure idea as win shares or BPM*minutes. It is computed strictly
    from already-fitted ratings and the design matrix's own possession
    weights; nothing here feeds back into or changes the ridge
    coefficients.
    """
    possessions = _player_possessions(design)
    value: dict[int, float] = {}
    wins_added: dict[int, float] = {}
    for pid in offense:
        total = offense[pid] + defense.get(pid, 0.0)
        poss = possessions.get(pid, 0.0)
        v = (total - replacement_level) * poss / 100.0
        value[pid] = v
        wins_added[pid] = v / POINTS_PER_WIN
    return possessions, value, wins_added


@dataclass
class RAPMResult:
    offense: dict[int, float]      # player_id -> offensive rating (pts/100)
    defense: dict[int, float]      # player_id -> defensive rating (pts/100)
    intercept: float
    best_lambda: float
    cv_scores: dict[float, float]  # lambda -> mean held-out weighted MSE
    n_rows: int
    n_players: int
    possessions: dict[int, float]  # player_id -> total possessions (off + def)
    value: dict[int, float]        # player_id -> (total - replacement) * poss/100
    wins_added: dict[int, float]   # player_id -> value / POINTS_PER_WIN
    replacement_level: float       # the replacement_level value used above

    def total(self, player_id: int) -> float:
        """Combined RAPM: offense plus defense (both in 'good is positive')."""
        return self.offense.get(player_id, 0.0) + self.defense.get(player_id, 0.0)

    def ranking(self) -> list[tuple[int, float, float, float]]:
        """(player_id, total, offense, defense) sorted by total, best first."""
        rows = [
            (pid, self.total(pid), self.offense.get(pid, 0.0),
             self.defense.get(pid, 0.0))
            for pid in self.offense
        ]
        rows.sort(key=lambda t: t[1], reverse=True)
        return rows

    def value_ranking(self) -> list[tuple[int, float, float, float, float]]:
        """(player_id, value, wins_added, total_rapm, possessions) sorted
        by value, best first. A rate leaderboard (`ranking`) and this
        volume leaderboard answer different questions -- a high-rate
        low-minutes player can rank above a compiler-type on `ranking`
        while ranking well below him here, and that is the point of
        having both."""
        rows = [
            (pid, self.value.get(pid, 0.0), self.wins_added.get(pid, 0.0),
             self.total(pid), self.possessions.get(pid, 0.0))
            for pid in self.offense
        ]
        rows.sort(key=lambda t: t[1], reverse=True)
        return rows


def _weighted_mse(y_true, y_pred, w) -> float:
    err = y_true - y_pred
    return float(np.sum(w * err * err) / np.sum(w))


def select_lambda(
    design: DesignMatrix,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    n_splits: int = 5,
) -> tuple[float, dict[float, float]]:
    """Grouped cross-validation over lambda. Returns (best_lambda, scores).

    Folds are split on game_id so no game appears in both train and test.
    Each fold's held-out error is possession-weighted, matching the fit.
    """
    groups = design.group
    unique_games = np.unique(groups)
    k = min(n_splits, len(unique_games))
    if k < 2:
        # Not enough distinct games to cross-validate; fall back to the grid
        # midpoint rather than pretending to have validated.
        mid = lambda_grid[len(lambda_grid) // 2]
        return mid, {mid: float("nan")}

    gkf = GroupKFold(n_splits=k)
    scores: dict[float, list[float]] = {lam: [] for lam in lambda_grid}

    for train_idx, test_idx in gkf.split(design.X, design.y, groups):
        Xtr, Xte = design.X[train_idx], design.X[test_idx]
        ytr, yte = design.y[train_idx], design.y[test_idx]
        wtr, wte = design.w[train_idx], design.w[test_idx]
        for lam in lambda_grid:
            model = Ridge(alpha=lam, fit_intercept=True, solver="sparse_cg")
            model.fit(Xtr, ytr, sample_weight=wtr)
            pred = model.predict(Xte)
            scores[lam].append(_weighted_mse(yte, pred, wte))

    mean_scores = {lam: float(np.mean(v)) for lam, v in scores.items()}
    best = min(mean_scores, key=mean_scores.get)
    return best, mean_scores


def fit_rapm(
    design: DesignMatrix,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    n_splits: int = 5,
    fixed_lambda: float | None = None,
    replacement_level: float = REPLACEMENT_LEVEL,
) -> RAPMResult:
    """Fit ridge RAPM, choosing lambda by grouped CV unless one is fixed.

    `fixed_lambda` skips CV -- useful for teaching/debugging the mechanics
    before trusting the CV wrapper, and for fast synthetic recovery checks.

    The ridge fit itself only ever touches offense/defense; `value` and
    `wins_added` are computed strictly AFTER, from the fitted ratings plus
    possession weights already in `design` -- they cannot influence the
    coefficients above.
    """
    if fixed_lambda is not None:
        best_lambda, cv_scores = fixed_lambda, {fixed_lambda: float("nan")}
    else:
        best_lambda, cv_scores = select_lambda(design, lambda_grid, n_splits)

    model = Ridge(alpha=best_lambda, fit_intercept=True, solver="sparse_cg")
    model.fit(design.X, design.y, sample_weight=design.w)

    coef = model.coef_
    offense: dict[int, float] = {}
    defense: dict[int, float] = {}
    for pid, i in design.player_index.items():
        offense[pid] = float(coef[2 * i])
        # Stored as "good is positive": the design used -1 in defense
        # columns, so a defender who lowers opponent efficiency gets a
        # positive coefficient already. Keep the raw sign.
        defense[pid] = float(coef[2 * i + 1])

    possessions, value, wins_added = compute_value(
        design, offense, defense, replacement_level
    )
    return RAPMResult(
        offense=offense,
        defense=defense,
        intercept=float(model.intercept_),
        best_lambda=best_lambda,
        cv_scores=cv_scores,
        n_rows=design.X.shape[0],
        n_players=design.n_players,
        possessions=possessions,
        value=value,
        wins_added=wins_added,
        replacement_level=replacement_level,
    )


def fit_rapm_bayesian(
    design: DesignMatrix,
    prior_mu: np.ndarray,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    n_splits: int = 5,
    fixed_lambda: float | None = None,
    replacement_level: float = REPLACEMENT_LEVEL,
) -> RAPMResult:
    """Fit RAPM shrinking toward a box-score prior mean instead of zero.

    Ridge with prior mean mu minimizes ||X b - y||^2 + lambda ||b - mu||^2.
    Closed form: substitute gamma = b - mu, giving target y' = y - X mu;
    solve ordinary weighted ridge for gamma; then b = gamma + mu. This
    reuses the exact same solver as plain ridge, so the only change is a
    transformed target and adding mu back at the end.

    A player with few possessions is pulled toward his box-score-predicted
    rating (his slice of mu); a player with many possessions still has his
    rating driven by the data. That is the whole point.
    """
    if prior_mu.shape[0] != design.X.shape[1]:
        raise ValueError(
            f"prior_mu has {prior_mu.shape[0]} entries but design has "
            f"{design.X.shape[1]} columns"
        )

    # Transform the target: y' = y - X mu. Everything downstream is ordinary
    # ridge on (X, y', w); we add mu back to the coefficients at the end.
    offset = design.X @ prior_mu
    y_shifted = design.y - offset

    shifted = DesignMatrix(
        X=design.X, y=y_shifted, w=design.w,
        player_index=design.player_index, n_players=design.n_players,
        group=design.group,
    )
    if fixed_lambda is not None:
        best_lambda, cv_scores = fixed_lambda, {fixed_lambda: float("nan")}
    else:
        best_lambda, cv_scores = select_lambda(shifted, lambda_grid, n_splits)

    model = Ridge(alpha=best_lambda, fit_intercept=True, solver="sparse_cg")
    model.fit(shifted.X, shifted.y, sample_weight=shifted.w)

    coef = model.coef_ + prior_mu  # add the prior mean back
    offense: dict[int, float] = {}
    defense: dict[int, float] = {}
    for pid, i in design.player_index.items():
        offense[pid] = float(coef[2 * i])
        defense[pid] = float(coef[2 * i + 1])

    possessions, value, wins_added = compute_value(
        design, offense, defense, replacement_level
    )
    return RAPMResult(
        offense=offense,
        defense=defense,
        intercept=float(model.intercept_),
        best_lambda=best_lambda,
        cv_scores=cv_scores,
        n_rows=design.X.shape[0],
        n_players=design.n_players,
        possessions=possessions,
        value=value,
        wins_added=wins_added,
        replacement_level=replacement_level,
    )


@dataclass
class OnOffDefense:
    """On/off defensive rating for one player: opponent points per 100
    possessions with the player ON the floor vs OFF it. Negative `diff`
    means the team defends BETTER (allows fewer points) with them on.

    This is a crude, contaminated measure -- it does not adjust for
    teammates the way RAPM does -- which is exactly why it is a useful
    INDEPENDENT check on defensive RAPM rather than a competing metric. If a
    player's defensive RAPM and on/off agree, that is real corroboration
    from a differently-computed signal.
    """

    player_id: int
    opp_pts_100_on: float
    opp_pts_100_off: float
    def_possessions_on: float
    def_possessions_off: float

    @property
    def diff(self) -> float:
        """on minus off; negative == defends better on the floor."""
        return self.opp_pts_100_on - self.opp_pts_100_off