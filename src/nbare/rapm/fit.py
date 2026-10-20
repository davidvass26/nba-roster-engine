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


def player_side_possessions(
    design: DesignMatrix,
) -> tuple[dict[int, float], dict[int, float]]:
    """(offensive possessions, defensive possessions) per player: the sum of
    block possession weights (`design.w`) over the design rows where they
    appear on that side. Reads directly off the same design matrix the
    ridge fit on -- not a fresh pass over blocks -- so this can never
    disagree with what the model actually saw.
    """
    Xc = design.X.tocsc()
    off_poss: dict[int, float] = {}
    def_poss: dict[int, float] = {}
    for pid, i in design.player_index.items():
        off_poss[pid] = float(design.w[Xc.getcol(2 * i).indices].sum())
        def_poss[pid] = float(design.w[Xc.getcol(2 * i + 1).indices].sum())
    return off_poss, def_poss


def compute_value(
    design: DesignMatrix,
    offense: dict[int, float],
    defense: dict[int, float],
    replacement_level: float = REPLACEMENT_LEVEL,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Post-fit value metric. Returns (possessions, value, wins_added).

        value = ((off - R/2) * off_poss + (def - R/2) * def_poss) / 100
        wins_added = value / POINTS_PER_WIN

    where R is `replacement_level` and `possessions` (returned, and shown
    in the app) is off_poss + def_poss.

    This turns the RATE stat (RAPM, points per 100 possessions) into a
    VOLUME stat by multiplying by playing time -- the same rate-times-
    exposure idea as win shares or BPM*minutes. It is computed strictly
    from already-fitted ratings and the design matrix's own possession
    weights; nothing here feeds back into or changes the ridge
    coefficients.

    Why each side is weighted by ITS OWN possessions: the offensive rating
    is points per 100 *offensive* possessions and the defensive rating is
    points per 100 *defensive* possessions, so total RAPM (off + def) is a
    rate per 100 possessions OF EACH KIND -- i.e. per ~200 combined. An
    earlier version multiplied total RAPM by combined possessions, which
    counted every possession twice and doubled value and wins added (a
    ~32-win/season SGA). Since off_poss ~= def_poss for any real player,
    the formula above is ~= (total - R) * combined_poss / 200.

    Replacement level R is a TOTAL (off + def) rating, so it is split
    evenly across the two sides (R/2 per side). That split is an
    assumption -- the placeholder R itself is not calibrated by side --
    but it is exactly consistent with R as a total whenever a player's
    offensive and defensive possessions are equal.
    """
    off_poss, def_poss = player_side_possessions(design)
    half_r = replacement_level / 2.0
    possessions: dict[int, float] = {}
    value: dict[int, float] = {}
    wins_added: dict[int, float] = {}
    for pid in offense:
        po, pd = off_poss.get(pid, 0.0), def_poss.get(pid, 0.0)
        v = ((offense[pid] - half_r) * po + (defense.get(pid, 0.0) - half_r) * pd) / 100.0
        possessions[pid] = po + pd
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
    value: dict[int, float]        # player_id -> points above replacement (see compute_value)
    wins_added: dict[int, float]   # player_id -> value / POINTS_PER_WIN
    replacement_level: float       # the replacement_level value used above
    # Ridge prior mean over the design's 2P columns: None = zero (plain
    # ridge); set by fit_rapm_bayesian. Needed to divide the prior back out
    # of the posterior (see `likelihood_measurements`).
    prior_mean: np.ndarray | None = None

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
        prior_mean=prior_mu,
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

def _posterior_variance(design: DesignMatrix, result: RAPMResult) -> tuple[np.ndarray, float]:
    """(diag of the ridge posterior covariance over all 2P columns, sigma^2).
    See `rapm_standard_errors` for the derivation."""
    X = design.X.tocsr()
    w = design.w
    n_cols = X.shape[1]
    coef = np.zeros(n_cols)
    for pid, i in design.player_index.items():
        coef[2 * i] = result.offense[pid]
        coef[2 * i + 1] = result.defense[pid]
    resid = design.y - (X @ coef + result.intercept)
    sigma2 = float(np.sum(w * resid**2) / len(resid))

    xtw = X.T.multiply(w).tocsr()                      # X' W, sparse
    xtwx = (xtw @ X).toarray()                          # X' W X, dense
    s = np.asarray(xtw.sum(axis=1)).ravel()             # X' w
    xtwx -= np.outer(s, s) / w.sum()                    # weighted centering
    xtwx[np.diag_indices(n_cols)] += result.best_lambda
    return sigma2 * np.diag(np.linalg.inv(xtwx)), sigma2


def rapm_standard_errors(
    design: DesignMatrix, result: RAPMResult
) -> tuple[dict[int, float], dict[int, float]]:
    """Posterior standard deviation of every player's offensive and
    defensive rating: (offense_se, defense_se), pts/100.

    The math (exact, not a heuristic): weighted ridge with penalty lambda
    is the posterior mode of a Gaussian model with y_i ~ N(x_i b + c,
    sigma^2 / w_i) and prior b ~ N(mu, (sigma^2 / lambda) I). Its posterior
    covariance is

        Cov(b) = sigma^2 * (Xc' W Xc + lambda I)^-1

    where Xc is X centered by its weighted column means (the intercept is
    fit unpenalized, which is what centering accounts for). The prior MEAN
    mu (zero, or a box-score prior) does not enter, so this serves both
    `fit_rapm` and `fit_rapm_bayesian`. sigma^2 is estimated from the
    weighted residuals: mean(w * r^2). Dense inverse of a (2P x 2P) matrix
    with P ~ 500-800 players -- a fraction of a second.

    NOTE: these are POSTERIOR SDs around SHRUNK estimates. Do not feed
    (rating, se) to a second model as an unbiased measurement -- that
    counts the prior twice. Use `likelihood_measurements` for that.
    """
    var, _ = _posterior_variance(design, result)
    off_se: dict[int, float] = {}
    def_se: dict[int, float] = {}
    for pid, i in design.player_index.items():
        off_se[pid] = float(np.sqrt(var[2 * i]))
        def_se[pid] = float(np.sqrt(var[2 * i + 1]))
    return off_se, def_se


@dataclass(frozen=True)
class LikelihoodMeasurements:
    """What the possession DATA alone says about each rating, with the
    ridge prior divided back out: estimate and standard error per player,
    per side. Players whose posterior carries (numerically) no information
    beyond the prior are ABSENT -- they have no measurement, not a zero."""

    offense: dict[int, float]
    offense_se: dict[int, float]
    defense: dict[int, float]
    defense_se: dict[int, float]


def likelihood_measurements(
    design: DesignMatrix, result: RAPMResult, *, min_info_ratio: float = 0.05
) -> LikelihoodMeasurements:
    """Divide the ridge prior out of each rating's (marginal) posterior.

    Why: Stage 3 treats each single-season rating as a noisy MEASUREMENT of
    true talent. A ridge estimate is not one -- it has already been pulled
    toward its prior (zero, or the box-score prior), and its posterior SD
    (`rapm_standard_errors`) already includes the prior's information.
    Feeding (shrunk estimate, posterior SD) to the aging model counts the
    prior twice and flattens real season-to-season change: on the real
    2023-26 panel the posterior SDs (~1.2) EXCEEDED the spread of the
    ratings themselves (~1.05), and the hierarchy concluded talent never
    changes (sigma_player ~0.05).

    Gaussian division, per coordinate: posterior N(m, v), prior N(mu0,
    tau^2) with tau^2 = sigma^2 / lambda, so the likelihood is N(l, s^2):

        1/s^2 = 1/v - 1/tau^2
        l     = s^2 * (m/v - mu0/tau^2)

    Exact when the design is orthogonal (l is then the OLS estimate); with
    correlated teammates the marginal v already folds in that ambiguity,
    so s is if anything conservative. A coordinate whose data precision is
    below `min_info_ratio` of the prior precision has essentially no data
    behind it and is left out rather than given an enormous, numerically
    fragile SE.
    """
    var, sigma2 = _posterior_variance(design, result)
    tau2 = sigma2 / result.best_lambda
    mu0 = result.prior_mean if result.prior_mean is not None else np.zeros(len(var))
    out: dict[str, dict[int, float]] = {k: {} for k in ("o", "os", "d", "ds")}
    for pid, i in design.player_index.items():
        for col, m, ek, sk in ((2 * i, result.offense[pid], "o", "os"),
                               (2 * i + 1, result.defense[pid], "d", "ds")):
            data_prec = 1.0 / var[col] - 1.0 / tau2
            if data_prec <= min_info_ratio / tau2:
                continue
            s2 = 1.0 / data_prec
            out[ek][pid] = float(s2 * (m / var[col] - mu0[col] / tau2))
            out[sk][pid] = float(np.sqrt(s2))
    return LikelihoodMeasurements(out["o"], out["os"], out["d"], out["ds"])
