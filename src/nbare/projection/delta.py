"""Layer 1: the delta method for estimating an aging curve.

The trap this avoids
---------------------
The naive approach -- regress rating against age across the whole league --
is contaminated by survivorship bias. Bad players get cut in their
mid-20s; only good players are still on a roster at 35. A cross-sectional
age-vs-performance regression therefore reads "players at 35" as "great",
when really it is "the ones who weren't good enough already left the
sample." The bias grows with age, which is exactly backwards from the true
decline curve.

The delta method (Tango et al.) sidesteps this by never comparing across
players. For each player with ratings in two *consecutive* seasons, take
the delta: rating(age+1) - rating(age). Average those deltas across all
players observed at that age transition. Because every delta compares a
player only to himself, a departing player contributes no delta for the
age he never played, rather than a fabricated decline -- survivorship
selects which deltas exist, not what they say.

This module is deliberately non-hierarchical and non-parametric: one
league-wide weighted average per age transition, nothing pooled by player
or position yet. That comes in `hierarchy.py` (layer 2). Validate this
layer alone first so a bug here cannot hide inside the hierarchy's extra
machinery.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from nbare.projection.types import PlayerSeason


@dataclass(frozen=True, slots=True)
class PlayerDelta:
    """One player's rating change across one age transition."""

    player_id: int
    age_from: int          # delta covers age_from -> age_from + 1
    delta: float
    weight: float           # reliability weight, see compute_player_deltas


def compute_player_deltas(
    seasons: list[PlayerSeason], rating_attr: str, weight_attr: str
) -> list[PlayerDelta]:
    """Extract within-player, consecutive-age deltas.

    Only age_from -> age_from + 1 transitions are used -- a gap season
    (age_from -> age_from + 2, e.g. an injury year with no data) is
    skipped rather than divided by two and guessed at, per the project's
    "never guess an unknown fact" rule.

    The weight is the MIN of the two seasons' possession counts: a delta
    is only as reliable as its less-reliable half.
    """
    by_player: dict[int, list[PlayerSeason]] = defaultdict(list)
    for s in seasons:
        by_player[s.player_id].append(s)

    deltas: list[PlayerDelta] = []
    for pid, rows in by_player.items():
        rows.sort(key=lambda r: r.age)
        for a, b in zip(rows, rows[1:]):
            if b.age != a.age + 1:
                continue
            w = min(getattr(a, weight_attr), getattr(b, weight_attr))
            if w <= 0:
                continue
            delta = getattr(b, rating_attr) - getattr(a, rating_attr)
            deltas.append(PlayerDelta(player_id=pid, age_from=a.age,
                                       delta=delta, weight=w))
    return deltas


def league_delta_curve(deltas: list[PlayerDelta]) -> dict[int, float]:
    """Possession-weighted mean delta at each age transition, pooling every
    player who has one. This is the plain (non-hierarchical) league curve.
    """
    num: dict[int, float] = defaultdict(float)
    den: dict[int, float] = defaultdict(float)
    for d in deltas:
        num[d.age_from] += d.weight * d.delta
        den[d.age_from] += d.weight
    return {age: num[age] / den[age] for age in num if den[age] > 0}


def cumulative_curve(
    delta_curve: dict[int, float], anchor_age: int
) -> dict[int, float]:
    """Integrate age-transition deltas into a relative rating curve, pinned
    to 0 at `anchor_age`. Only the SHAPE is identified by deltas -- there is
    no data here that fixes the overall level, so callers must add their
    own baseline (a per-player free parameter in later layers).
    """
    if not delta_curve:
        return {anchor_age: 0.0}

    ages = sorted(delta_curve)
    min_age, max_age = ages[0], ages[-1] + 1

    curve: dict[int, float] = {anchor_age: 0.0}

    level = 0.0
    for age in range(anchor_age, max_age):
        if age not in delta_curve:
            break
        level += delta_curve[age]
        curve[age + 1] = level

    level = 0.0
    for age in range(anchor_age - 1, min_age - 1, -1):
        if age not in delta_curve:
            break
        level -= delta_curve[age]
        curve[age] = level

    return curve
