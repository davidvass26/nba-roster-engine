"""Box-score-informed prior for Bayesian RAPM.

The upgrade over plain ridge
----------------------------
Plain ridge RAPM shrinks every player toward ZERO (league average) when the
possession data is thin. That is safe but wasteful: a rookie who posted
strong box-score numbers and one who barely played both get pulled to zero
equally, even though the box score already distinguishes them.

This module replaces "shrink toward zero" with "shrink toward what the box
score predicts." A player with 50 possessions gets pulled toward his
box-score-implied rating; a player with thousands of possessions still has
his RAPM dominated by the possession data. This is where the biggest
accuracy gain lives -- low-minute players, the exact population plain RAPM
handles worst.

The mechanism (ridge with a non-zero prior mean)
------------------------------------------------
Ordinary ridge minimizes ||X beta - y||^2 + lambda ||beta||^2, i.e. it
shrinks beta toward 0. Ridge with prior mean mu minimizes

    ||X beta - y||^2 + lambda ||beta - mu||^2

which shrinks beta toward mu. This has a clean closed form: substitute
gamma = beta - mu, so the target becomes y' = y - X mu; solve ordinary
(weighted) ridge for gamma; then beta = gamma + mu. No new solver needed --
we reuse sklearn Ridge on the transformed target. See fit.fit_rapm_bayesian.

Two-stage estimation
--------------------
1. Fit plain ridge RAPM to get ratings for everyone.
2. On players with ENOUGH possessions (reliable RAPM), regress box-score
   features -> rating. This learns "what does a rating look like for a
   player with this box score." Separate models for offense and defense,
   because defense needs different, richer features (see below).
3. Predict a prior mean mu for EVERY player from their box score.

Why the defensive prior is richer
----------------------------------
The box score barely captures defense. Steals and blocks reward gambling
and shot-blocking while missing positional defenders (a rim protector's
value is in shots never taken, which records no stat). So a steals+blocks
prior systematically misranks the best defenders. We add defensive rebound
rate (ending a possession is real defensive value, and unlike steals it
does not reward gambling) and a position signal (bigs anchor defense
differently than guards). This is a deliberately multi-signal prior for the
noisier half -- it will not be perfect, but it beats zero and it beats
steals+blocks alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nbare.rapm.design import DesignMatrix
from nbare.rapm.fit import RAPMResult

# A player needs at least this many offensive/defensive possessions before
# his plain-ridge rating is trusted as a training label for the prior model.
# Below this, the rating is too noisy to teach the box-score regression.
DEFAULT_RELIABLE_POSSESSIONS = 1000.0


@dataclass(frozen=True, slots=True)
class BoxScoreRow:
    """Per-player box-score aggregates, already normalized to per-100
    possessions where noted. This is the feature source for the prior.

    Defensive features (stl, blk, dreb_rate) are deliberately more numerous
    than offensive ones because defense is harder to read from a box score.
    `position_big` is a 0/1 flag: rim-protecting bigs anchor defense in a
    way guards do not, and it materially improves the defensive prior.
    """

    player_id: int
    off_possessions: float
    def_possessions: float
    # offensive features (per 100 possessions)
    pts_100: float
    ast_100: float
    tov_100: float
    ts_pct: float          # true shooting %, 0..1
    # defensive features
    stl_100: float
    blk_100: float
    dreb_rate: float       # defensive rebounds / available, 0..1
    position_big: int = 0  # 1 for C/PF-ish, 0 otherwise


OFFENSE_FEATURES = ("pts_100", "ast_100", "tov_100", "ts_pct")
DEFENSE_FEATURES = ("stl_100", "blk_100", "dreb_rate", "position_big")


@dataclass
class PriorModel:
    """A fitted box-score -> rating linear model for one side (off or def)."""

    features: tuple[str, ...]
    coef: np.ndarray        # shape (n_features,)
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    n_train: int

    def predict(self, rows: list[BoxScoreRow]) -> np.ndarray:
        X = _feature_matrix(rows, self.features)
        Xs = (X - self.feature_mean) / self.feature_std
        return Xs @ self.coef + self.intercept


@dataclass
class PriorMeans:
    """The prior-mean vector mu over the design's 2*n_players columns, plus
    the models that produced it (for inspection/diagnostics)."""

    mu: np.ndarray
    offense_model: PriorModel
    defense_model: PriorModel
    reliable_offense: int
    reliable_defense: int
    notes: list[str] = field(default_factory=list)


def _feature_matrix(rows: list[BoxScoreRow], features: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[getattr(r, f) for f in features] for r in rows], dtype=np.float64
    )


def _fit_linear(
    X: np.ndarray, y: np.ndarray, ridge: float = 1.0
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Small standardized ridge regression for the prior model.

    Standardize features (so the mild ridge penalty is even-handed), solve
    the normal equations with a tiny ridge for numerical stability, and
    return coef in standardized space plus the standardization constants.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    n_features = Xs.shape[1]
    A = Xs.T @ Xs + ridge * np.eye(n_features)
    b = Xs.T @ (y - y.mean())
    coef = np.linalg.solve(A, b)
    intercept = float(y.mean())
    return coef, intercept, mean, std


def build_prior(
    design: DesignMatrix,
    plain: RAPMResult,
    box_rows: list[BoxScoreRow],
    *,
    reliable_possessions: float = DEFAULT_RELIABLE_POSSESSIONS,
) -> PriorMeans:
    """Build the prior-mean vector mu from box scores + a plain-ridge fit.

    Trains separate offense and defense box-score models on players whose
    plain-ridge rating is reliable (enough possessions), then predicts a
    prior mean for every player in the design.
    """
    box_by_id = {r.player_id: r for r in box_rows}
    notes: list[str] = []

    # Training set: players with enough possessions AND a box-score row.
    train_off, y_off = [], []
    train_def, y_def = [], []
    for pid in design.player_index:
        row = box_by_id.get(pid)
        if row is None:
            continue
        if row.off_possessions >= reliable_possessions and pid in plain.offense:
            train_off.append(row)
            y_off.append(plain.offense[pid])
        if row.def_possessions >= reliable_possessions and pid in plain.defense:
            train_def.append(row)
            y_def.append(plain.defense[pid])

    if len(train_off) < len(OFFENSE_FEATURES) + 2:
        notes.append(
            f"only {len(train_off)} reliable offensive players; offense prior "
            "falls back to zero mean"
        )
    if len(train_def) < len(DEFENSE_FEATURES) + 2:
        notes.append(
            f"only {len(train_def)} reliable defensive players; defense prior "
            "falls back to zero mean"
        )

    off_model = _make_model(train_off, np.array(y_off), OFFENSE_FEATURES)
    def_model = _make_model(train_def, np.array(y_def), DEFENSE_FEATURES)

    # Predict mu for every player. Players with no box-score row keep mu=0
    # (pure league-average prior), which is the safe default.
    mu = np.zeros(2 * design.n_players, dtype=np.float64)
    present = [design.player_index[pid] for pid in design.player_index
               if pid in box_by_id]
    present_rows = [box_by_id[pid] for pid in design.player_index
                    if pid in box_by_id]
    if present_rows:
        off_pred = off_model.predict(present_rows)
        def_pred = def_model.predict(present_rows)
        for k, col_base in enumerate(present):
            mu[2 * col_base] = off_pred[k]
            mu[2 * col_base + 1] = def_pred[k]

    return PriorMeans(
        mu=mu,
        offense_model=off_model,
        defense_model=def_model,
        reliable_offense=len(train_off),
        reliable_defense=len(train_def),
        notes=notes,
    )


def _make_model(
    rows: list[BoxScoreRow], y: np.ndarray, features: tuple[str, ...]
) -> PriorModel:
    """Fit a PriorModel, or a zero model if there is not enough training data."""
    if len(rows) < len(features) + 2:
        # Degenerate: not enough to fit. Return a zero model so the prior
        # gracefully reduces to plain ridge (shrink toward zero).
        return PriorModel(
            features=features,
            coef=np.zeros(len(features)),
            intercept=0.0,
            feature_mean=np.zeros(len(features)),
            feature_std=np.ones(len(features)),
            n_train=len(rows),
        )
    X = _feature_matrix(rows, features)
    coef, intercept, mean, std = _fit_linear(X, y)
    return PriorModel(
        features=features, coef=coef, intercept=intercept,
        feature_mean=mean, feature_std=std, n_train=len(rows),
    )