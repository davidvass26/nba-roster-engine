"""Where do the box score and the possession data disagree about a player?

The comparison
--------------
The box-score prior mean mu says what a player's stat line usually implies;
the RAPM rating b says where he landed after the possession data had its
say. The gap b - mu is positive when RAPM likes him more than his box score
does (the classic "doesn't show up in the box score" player) and negative
when his box score flatters him.

The traps handled here
----------------------
- A gap is not a difference between two independent numbers. b was SHRUNK
  TOWARD mu, so the gap is pulled to zero by exactly the shrinkage a
  low-possession player gets: he has a small gap because there is no data,
  not because the two agree. That is why every gap carries `se`. The
  posterior of (true rating - mu) is N(gap, se^2) -- mu is a fixed number
  here, so this is exact -- which makes `z` = gap / se the answer to "how
  sure are we that this player really differs from his box score?" Rank by
  gap for size, by z for confidence.
- No box score is not a box score of zero. `build_prior` leaves mu at
  exactly 0.0 for players with no feature row (too few box possessions).
  That 0 is a default, not a prediction, and comparing against it would
  report a phantom gap equal to the player's whole rating. Those players
  are left out and counted, never listed. A real prediction is a linear
  model's output with a nonzero intercept, so it does not land on exactly
  0.0 on BOTH sides.
- The total's SE is approximate. `rapm_standard_errors` gives marginal SDs
  per side, so `total` adds them in quadrature, ignoring the covariance
  between one player's offense and defense columns. On the pooled 2023-26
  fit the exact SE was within 0.5% of this for every 2,000+ possession
  player.
- The prior is narrower than the ratings. It is a regression fit to
  plain-ridge labels, so it regresses toward the middle: on the same fit
  the prior's SD is about half the ratings'. Good players therefore lean
  positive and bad players negative regardless of anything specific to
  them, so read the extremes of either list with that in mind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbare.rapm.design import DesignMatrix
from nbare.rapm.fit import RAPMResult, rapm_standard_errors

SIDES = ("total", "offense", "defense")
RANK_KEYS = ("gap", "z")


@dataclass(frozen=True, slots=True)
class SideGap:
    """One rating vs its box-score prior mean, pts/100. Positive `gap`
    means RAPM rates the player above his box score."""

    rapm: float
    prior: float
    se: float

    @property
    def gap(self) -> float:
        return self.rapm - self.prior

    @property
    def z(self) -> float:
        return self.gap / self.se


@dataclass(frozen=True, slots=True)
class PlayerGap:
    player_id: int
    possessions: float      # off + def, as on the leaderboard
    offense: SideGap
    defense: SideGap

    @property
    def total(self) -> SideGap:
        return SideGap(
            rapm=self.offense.rapm + self.defense.rapm,
            prior=self.offense.prior + self.defense.prior,
            se=float(np.hypot(self.offense.se, self.defense.se)),
        )


@dataclass
class PriorGaps:
    players: list[PlayerGap]
    n_no_box_prior: int     # left out: mu is the 0.0 default, not a prediction

    def ranked(
        self, *, side: str = "total", by: str = "gap", min_possessions: float = 0.0
    ) -> list[PlayerGap]:
        """Players with at least `min_possessions`, highest `by` first: the
        head is who RAPM likes more than the box score, the tail who it
        likes less."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if by not in RANK_KEYS:
            raise ValueError(f"by must be one of {RANK_KEYS}, got {by!r}")
        rows = [p for p in self.players if p.possessions >= min_possessions]
        rows.sort(key=lambda p: getattr(getattr(p, side), by), reverse=True)
        return rows


def prior_gaps(design: DesignMatrix, result: RAPMResult) -> PriorGaps:
    """Compare every player's rating to his box-score prior mean.

    Requires a `fit_rapm_bayesian` result: a plain ridge fit has no box
    prior to compare against, and treating its implicit zero as one would
    just re-rank players by rating.
    """
    if result.prior_mean is None:
        raise ValueError("result has no prior_mean: fit with the box prior")

    off_se, def_se = rapm_standard_errors(design, result)
    players: list[PlayerGap] = []
    n_no_box = 0
    for pid, i in design.player_index.items():
        mu_off = float(result.prior_mean[2 * i])
        mu_def = float(result.prior_mean[2 * i + 1])
        if mu_off == 0.0 and mu_def == 0.0:
            n_no_box += 1
            continue
        players.append(PlayerGap(
            player_id=pid,
            possessions=result.possessions[pid],
            offense=SideGap(result.offense[pid], mu_off, off_se[pid]),
            defense=SideGap(result.defense[pid], mu_def, def_se[pid]),
        ))
    return PriorGaps(players=players, n_no_box_prior=n_no_box)
