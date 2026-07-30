"""Top-level Stage 3 output: posterior samples per player per future year.

This module does no new statistics -- the delta method, the hierarchy, and
the box-score baseline are already validated in `delta.py`, `hierarchy.py`,
and `baseline.py`. This is orchestration: slice each fitted model's
posterior at the specific future ages Stage 4 asked about, for both sides,
and hand back samples (never a point estimate -- the optimizer needs the
uncertainty, per the spec's Target section).

The honesty rule this module enforces
----------------------------------------
A hierarchical curve is only fit over the age grid it was trained on
(`age_min..age_max`). Projecting a player past `age_max` would require
extrapolating a quadratic-ish curve into ages the model has never seen --
exactly the kind of confident-guess the project's non-negotiable #1
principle forbids. So `project_players` TRUNCATES `future_ages` to
whatever falls inside the fitted grid, and drops a player entirely (rather
than returning empty/fabricated arrays) if none of his requested years are
covered. Callers must check `PlayerProjection.future_ages` against what
they asked for, not assume `years_ahead` was honored in full.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbare.projection.baseline import BoxScoreProjectionModel


@dataclass(frozen=True, slots=True)
class PlayerProjection:
    """Posterior samples of RAPM at each covered future age.

    `offense_samples`/`defense_samples` have shape (S, len(future_ages)) --
    S posterior draws, one column per age in `future_ages`. Draw from the
    same column index across both arrays to get a coherent (offense,
    defense) pair from one posterior sample, not independent marginals.
    """

    player_id: int
    future_ages: tuple[int, ...]
    offense_samples: np.ndarray
    defense_samples: np.ndarray


def project_players(
    offense_model: BoxScoreProjectionModel,
    defense_model: BoxScoreProjectionModel,
    current_age_by_player: dict[int, int],
    years_ahead: int = 4,
) -> dict[int, PlayerProjection]:
    """Project offense and defense RAPM forward for every player present in
    BOTH fitted models and `current_age_by_player`.

    A player absent from either fitted model (no fitted trajectory to draw
    from) is silently excluded, not defaulted to a league-average guess --
    callers who need a projection for such a player must fit them into the
    hierarchy first (even a single season is enough to enter the fit and
    be pooled toward his position curve).
    """
    off_result = offense_model.result
    def_result = defense_model.result
    off_age_pos = {a: k for k, a in enumerate(off_result.age_grid)}
    def_age_pos = {a: k for k, a in enumerate(def_result.age_grid)}

    common_ids = (
        set(off_result.player_index) & set(def_result.player_index)
        & set(current_age_by_player)
    )

    projections: dict[int, PlayerProjection] = {}
    for pid in sorted(common_ids):
        current_age = current_age_by_player[pid]
        requested_ages = [current_age + y for y in range(1, years_ahead + 1)]
        covered_ages = [a for a in requested_ages if a in off_age_pos and a in def_age_pos]
        if not covered_ages:
            continue

        off_curve = off_result.player_curve_samples(pid)  # (S, K)
        def_curve = def_result.player_curve_samples(pid)  # (S, K)
        off_idx = [off_age_pos[a] for a in covered_ages]
        def_idx = [def_age_pos[a] for a in covered_ages]

        projections[pid] = PlayerProjection(
            player_id=pid,
            future_ages=tuple(covered_ages),
            offense_samples=off_curve[:, off_idx],
            defense_samples=def_curve[:, def_idx],
        )
    return projections
