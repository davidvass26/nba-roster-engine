"""Layer 3: box-score-informed baseline for the trajectory hierarchy.

This is the two-stage estimation from `rapm/prior.py`, applied to
`hierarchy.py`'s `baseline` parameter instead of ridge RAPM's coefficients:

1. Fit the layer-2 hierarchy with an uninformative (mean 0, wide sd)
   baseline prior. This gives every player a preliminary baseline point
   estimate (posterior mean).
2. On players with ENOUGH seasons (a reliable estimate), regress box-score
   features -> that preliminary baseline. Offense and defense get separate
   models with separate feature sets, reusing `OFFENSE_FEATURES` /
   `DEFENSE_FEATURES` / `BoxScoreRow` / the standardized-ridge fit helper
   from `rapm/prior.py` verbatim -- this is deliberately the same
   regression machinery, not a reimplementation, so a bug fixed there does
   not need fixing twice.
3. Refit the hierarchy with `baseline_prior_mean` set to the box-score
   prediction for every player (a tighter `baseline_prior_sd`, since this
   prior mean now carries real information). A player with little data is
   pulled toward what his box score predicts; a player with a long career
   still has his baseline set mostly by his own data.

Box score is v1-baseline-only: it shifts WHERE the curve sits, never its
shape. The curve (player_delta/position_delta/league_delta) hierarchy from
`hierarchy.py` is untouched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbare.projection.hierarchy import HierarchicalCurveResult, fit_hierarchical_curve
from nbare.projection.types import PlayerSeason
from nbare.rapm.prior import (
    DEFENSE_FEATURES,
    OFFENSE_FEATURES,
    BoxScoreRow,
    PriorModel,
    _make_model,
)

# A player needs at least this many seasons before his preliminary
# hierarchical baseline is trusted as a training label for the box-score
# regression -- the trajectory-hierarchy analogue of
# `rapm.prior.DEFAULT_RELIABLE_POSSESSIONS`.
DEFAULT_RELIABLE_SEASONS = 4


@dataclass
class BoxScoreProjectionModel:
    """The final (box-score-informed) hierarchical fit, plus the box-score
    regression that produced its baseline prior (kept for inspection)."""

    result: HierarchicalCurveResult
    prelim: HierarchicalCurveResult  # the uninformative-prior fit used to train box_model
    box_model: PriorModel
    reliable_n: int
    notes: list[str]


def fit_box_score_projection(
    seasons: list[PlayerSeason],
    box_rows: dict[int, BoxScoreRow],
    rating_attr: str,
    weight_attr: str,
    *,
    age_min: int,
    age_max: int,
    anchor_age: int,
    reliable_seasons: float = DEFAULT_RELIABLE_SEASONS,
    baseline_prior_sd: float = 6.0,
    prelim_num_warmup: int = 400,
    prelim_num_samples: int = 400,
    num_warmup: int = 600,
    num_samples: int = 600,
    num_chains: int = 2,
    seed: int = 0,
) -> BoxScoreProjectionModel:
    """Fit the box-score-informed hierarchy for one side (offense or
    defense -- pick `rating_attr`/`weight_attr`/`features` accordingly)."""
    features = OFFENSE_FEATURES if rating_attr == "off_rating" else DEFENSE_FEATURES

    prelim = fit_hierarchical_curve(
        seasons, rating_attr, weight_attr, age_min=age_min, age_max=age_max,
        anchor_age=anchor_age, baseline_prior_mean=None, baseline_prior_sd=20.0,
        num_warmup=prelim_num_warmup, num_samples=prelim_num_samples,
        num_chains=num_chains, seed=seed,
    )
    prelim_baseline_mean = {
        pid: float(prelim.samples["baseline"][:, i].mean())
        for pid, i in prelim.player_index.items()
    }

    season_count: dict[int, int] = {}
    for s in seasons:
        season_count[s.player_id] = season_count.get(s.player_id, 0) + 1

    notes: list[str] = []
    train_rows: list[BoxScoreRow] = []
    train_y: list[float] = []
    for pid in prelim.player_index:
        if season_count.get(pid, 0) < reliable_seasons:
            continue
        row = box_rows.get(pid)
        if row is None:
            continue
        train_rows.append(row)
        train_y.append(prelim_baseline_mean[pid])

    if len(train_rows) < len(features) + 2:
        notes.append(
            f"only {len(train_rows)} reliable players; box-score baseline "
            "falls back to zero mean"
        )

    box_model = _make_model(train_rows, np.array(train_y), features)

    predicted_mean: dict[int, float] = {}
    for pid in prelim.player_index:
        row = box_rows.get(pid)
        predicted_mean[pid] = float(box_model.predict([row])[0]) if row is not None else 0.0

    final = fit_hierarchical_curve(
        seasons, rating_attr, weight_attr, age_min=age_min, age_max=age_max,
        anchor_age=anchor_age, baseline_prior_mean=predicted_mean,
        baseline_prior_sd=baseline_prior_sd, num_warmup=num_warmup,
        num_samples=num_samples, num_chains=num_chains, seed=seed,
    )

    return BoxScoreProjectionModel(
        result=final, prelim=prelim, box_model=box_model,
        reliable_n=len(train_rows), notes=notes,
    )
