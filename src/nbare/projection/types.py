"""Shared data types for Stage 3 (player projections).

`PlayerSeason` is the one input record every layer of the projection model
reads. Keeping it in its own module (rather than duplicating fields inside
`delta.py`/`hierarchy.py`/`baseline.py`) means the three layers can share a
single synthetic generator and a single set of test fixtures, which matters
because the whole point of the layered build is that each layer is tested
against the *same kind* of data the next layer will see.

`position_group` is deliberately a free-form string, not the `position_big`
0/1 flag from `rapm/prior.py`. That flag is a feature used to predict a
defensive *baseline* from a box score; the hierarchy in this module pools
aging *trajectories*, which is a coarser, position-group-shaped effect
(e.g. "G"/"F"/"C") and need not use the same grouping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerSeason:
    """One player's rating in one season.

    `off_rating`/`def_rating` are RAPM-scale ratings (points per 100
    possessions, "good is positive" -- same convention as `RAPMResult`).
    `off_possessions`/`def_possessions` size the observation's reliability;
    they are used as regression weights, never dropped silently.
    """

    player_id: int
    season: str        # e.g. "2024-25", sortable string
    age: int
    position_group: str  # e.g. "G", "F", "C"
    off_rating: float
    def_rating: float
    off_possessions: float
    def_possessions: float
