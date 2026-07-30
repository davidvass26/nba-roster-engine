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


@dataclass
class RAPMResult:
    offense: dict[int, float]      # player_id -> offensive rating (pts/100)
    defense: dict[int, float]      # player_id -> defensive rating (pts/100)
    intercept: float
    best_lambda: float
    cv_scores: dict[float, float]  # lambda -> mean held-out weighted MSE
    n_rows: int
    n_players: int

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
) -> RAPMResult:
    """Fit ridge RAPM, choosing lambda by grouped CV unless one is fixed.

    `fixed_lambda` skips CV -- useful for teaching/debugging the mechanics
    before trusting the CV wrapper, and for fast synthetic recovery checks.
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

    return RAPMResult(
        offense=offense,
        defense=defense,
        intercept=float(model.intercept_),
        best_lambda=best_lambda,
        cv_scores=cv_scores,
        n_rows=design.X.shape[0],
        n_players=design.n_players,
    )


def fit_rapm_bayesian(
    design: DesignMatrix,
    prior_mu: np.ndarray,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    n_splits: int = 5,
    fixed_lambda: float | None = None,
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

    return RAPMResult(
        offense=offense,
        defense=defense,
        intercept=float(model.intercept_),
        best_lambda=best_lambda,
        cv_scores=cv_scores,
        n_rows=design.X.shape[0],
        n_players=design.n_players,
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