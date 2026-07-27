"""Build the sparse design matrix for offense/defense-split RAPM.

The model
---------
Each player gets TWO latent ratings: an offensive rating (points produced
per 100 possessions above average when on offense) and a defensive rating
(points prevented per 100 possessions when on defense). Stack them into one
parameter vector beta of length 2 * n_players.

Each stint contributes two rows (see possessions.py): one where the home
lineup is on offense, one where the away lineup is. For a single offensive
possession-block with offensive lineup O and defensive lineup D, the
expected efficiency (points per 100 possessions) is:

    y  =  sum_{p in O} off_rating[p]  -  sum_{p in D} def_rating[p]  +  intercept

So the design row has +1 in the OFFENSE column of each offensive player and
-1 in the DEFENSE column of each defensive player. The sign convention makes
a positive defensive rating mean "good defender" (subtracts from opponent
efficiency), which is the intuitive reading.

Why sparse
----------
n_rows ~ 2 * n_stints (hundreds of thousands over multiple seasons),
n_cols = 2 * n_players (~1000-1400). Every row has exactly 10 nonzeros.
Dense this is hundreds of GB; as a CSR matrix it is a few hundred MB and
ridge solves in seconds. scipy.sparse is not an optimization here, it is
what makes the problem exist at all.

Weighting
---------
Each row is weighted by its possession count: a block of 8 possessions is 8x
the evidence of 1. sklearn's Ridge takes sample_weight directly, so we carry
a weight vector alongside X and y rather than pre-multiplying (which would
distort the regularization).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# Efficiency is expressed per this many possessions so ratings read on the
# familiar "points per 100" scale.
PER_POSSESSIONS = 100.0

# Minimum possessions for a block to be included. Sub-possession noise (a
# stint with 0.44 estimated possessions from a lone FTA) adds nothing but
# variance.
MIN_BLOCK_POSSESSIONS = 1.0


@dataclass
class DesignMatrix:
    """The assembled RAPM system.

    X : CSR matrix, shape (n_rows, 2 * n_players)
    y : efficiency (points per 100 poss) per row
    w : possession weight per row
    player_index : maps player_id -> base column; offense col is 2*i,
                   defense col is 2*i + 1
    """

    X: sp.csr_matrix
    y: np.ndarray
    w: np.ndarray
    player_index: dict[int, int]
    n_players: int
    group: np.ndarray  # game_id per row, for grouped cross-validation

    def offense_col(self, player_id: int) -> int:
        return 2 * self.player_index[player_id]

    def defense_col(self, player_id: int) -> int:
        return 2 * self.player_index[player_id] + 1


@dataclass(frozen=True, slots=True)
class OffenseBlock:
    """One possession-block: an offensive lineup vs a defensive lineup, with
    points scored and possessions used. This is the atomic design-matrix
    row. possessions.py turns each stint into two of these."""

    game_id: str
    offense: frozenset[int]
    defense: frozenset[int]
    points: float
    possessions: float


def build_design(
    blocks: list[OffenseBlock],
    *,
    min_possessions: float = MIN_BLOCK_POSSESSIONS,
) -> DesignMatrix:
    """Assemble the sparse design matrix from offense blocks.

    Players are indexed in first-seen order. Both an offensive and a
    defensive column are allocated for every player, even if (rarely) they
    only ever appear on one side, so the parameter vector is uniform.
    """
    usable = [b for b in blocks if b.possessions >= min_possessions]
    if not usable:
        raise ValueError("no blocks meet the minimum possession threshold")

    # Assign a stable column index per player.
    player_index: dict[int, int] = {}
    for b in usable:
        for pid in (*b.offense, *b.defense):
            if pid not in player_index:
                player_index[pid] = len(player_index)
    n_players = len(player_index)
    n_cols = 2 * n_players

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    y = np.empty(len(usable), dtype=np.float64)
    w = np.empty(len(usable), dtype=np.float64)
    group = np.empty(len(usable), dtype=object)

    for r, b in enumerate(usable):
        for pid in b.offense:
            rows.append(r)
            cols.append(2 * player_index[pid])       # offense column
            vals.append(1.0)
        for pid in b.defense:
            rows.append(r)
            cols.append(2 * player_index[pid] + 1)   # defense column
            vals.append(-1.0)
        y[r] = PER_POSSESSIONS * b.points / b.possessions
        w[r] = b.possessions
        group[r] = b.game_id

    X = sp.csr_matrix(
        (vals, (rows, cols)), shape=(len(usable), n_cols), dtype=np.float64
    )
    return DesignMatrix(
        X=X, y=y, w=w, player_index=player_index, n_players=n_players,
        group=group,
    )