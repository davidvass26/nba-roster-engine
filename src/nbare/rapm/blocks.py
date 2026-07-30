"""Turn reconstructed stints + play-by-play into RAPM offense blocks.

The gap this bridges
--------------------
`stints.reconstruct_game` gives Stints whose `lineup` is all 10 players on
the floor, both teams MIXED. The design matrix (`design.build_design`)
needs each possession-block split by team: which five are on offense,
which five on defense. So this connector must:

  1. split each stint's 10-player lineup into its two 5-player teams, and
  2. attribute the events inside each stint's time window to the team on
     offense, tallying possessions and points per side.

Step 1 needs a fact the Stint does not carry: which team each player is on.
The reconstruction dropped team membership because the minutes check did
not need it. Rather than re-plumb the stint builder, we derive the
player->team map from the box score (which has nba_team_id per player) and
pass it in. This is the cleanest seam and keeps the stint code focused.

Output
------
A flat list of OffenseBlock, two per stint (home-offense and away-offense),
ready for build_design. Stints that cannot be cleanly split (a lineup that
is not 5-and-5 by team, usually a symptom of an upstream reconstruction
error) are skipped and counted, never silently forced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from nbare.rapm.design import OffenseBlock
from nbare.rapm.possessions import (
    _is_fga,
    _is_fta,
    _is_oreb,
    _is_turnover,
    _points_of,
    estimate_team_possessions,
)
from nbare.rapm.stints import Stint, period_length_s


@dataclass
class BlockResult:
    blocks: list[OffenseBlock]
    skipped_stints: int = 0
    warnings: list[str] = field(default_factory=list)


def player_team_map(box: pl.DataFrame, *, player_col: str = "nba_player_id",
                    team_col: str = "nba_team_id") -> dict[int, int]:
    """player_id -> team_id, from the box score."""
    return {r[player_col]: r[team_col] for r in box.iter_rows(named=True)}


def split_lineup(
    lineup: frozenset[int], team_of: dict[int, int]
) -> tuple[int, frozenset[int], int, frozenset[int]] | None:
    """Split a 10-player lineup into (team_a, five_a, team_b, five_b).

    Returns None if the lineup does not partition into exactly two teams of
    five -- which means either a player is missing from the team map or the
    reconstruction produced a bad lineup. Either way the caller skips it
    rather than guessing.
    """
    by_team: dict[int, set[int]] = {}
    for pid in lineup:
        team = team_of.get(pid)
        if team is None:
            return None
        by_team.setdefault(team, set()).add(pid)
    if len(by_team) != 2:
        return None
    (ta, a), (tb, b) = by_team.items()
    if len(a) != 5 or len(b) != 5:
        return None
    return ta, frozenset(a), tb, frozenset(b)


def _elapsed(period: int, clock_left: float | None) -> float:
    if clock_left is None:
        return 0.0
    return period_length_s(period) - clock_left


def blocks_for_game(
    pbp: pl.DataFrame,
    stints: list[Stint],
    team_of: dict[int, int],
) -> BlockResult:
    """Build offense blocks for one game.

    For each stint we take the events whose elapsed time falls in the
    stint's [start_s, end_s) window within its period, tally each team's
    possessions and points, and emit two OffenseBlocks (one per team on
    offense). Events are attributed to the acting `team_id`.
    """
    if pbp.is_empty() or not stints:
        return BlockResult([], 0, ["empty pbp or no stints"])

    game_id = stints[0].game_id
    # Precompute elapsed time per event once.
    pev = pbp.with_columns(
        pl.struct(["period", "clock_seconds_left"])
        .map_elements(
            lambda s: _elapsed(s["period"], s["clock_seconds_left"]),
            return_dtype=pl.Float64,
        )
        .alias("_elapsed")
    )

    blocks: list[OffenseBlock] = []
    skipped = 0
    warnings: list[str] = []

    for st in stints:
        split = split_lineup(st.lineup, team_of)
        if split is None:
            skipped += 1
            continue
        team_a, five_a, team_b, five_b = split

        window = pev.filter(
            (pl.col("period") == st.period)
            & (pl.col("_elapsed") >= st.start_s)
            & (pl.col("_elapsed") < st.end_s)
        )

        tallies = {
            team_a: {"fga": 0, "oreb": 0, "tov": 0, "fta": 0, "pts": 0},
            team_b: {"fga": 0, "oreb": 0, "tov": 0, "fta": 0, "pts": 0},
        }
        for row in window.iter_rows(named=True):
            tid = row.get("team_id")
            if tid not in tallies:
                continue
            t = tallies[tid]
            if _is_fga(row):
                t["fga"] += 1
            if _is_oreb(row):
                t["oreb"] += 1
            if _is_turnover(row):
                t["tov"] += 1
            if _is_fta(row):
                t["fta"] += 1
            t["pts"] += _points_of(row)

        a, b = tallies[team_a], tallies[team_b]
        a_poss = estimate_team_possessions(a["fga"], a["oreb"], a["tov"], a["fta"])
        b_poss = estimate_team_possessions(b["fga"], b["oreb"], b["tov"], b["fta"])

        # team A on offense vs team B on defense
        if a_poss > 0:
            blocks.append(OffenseBlock(game_id, five_a, five_b, float(a["pts"]), a_poss))
        # team B on offense vs team A on defense
        if b_poss > 0:
            blocks.append(OffenseBlock(game_id, five_b, five_a, float(b["pts"]), b_poss))

    return BlockResult(blocks=blocks, skipped_stints=skipped, warnings=warnings)


def blocks_from_warehouse(con, game_ids: list[str]) -> BlockResult:
    """End-to-end: read PBP + box from the warehouse, reconstruct stints,
    and emit blocks for a set of games. This is the real-data entry point
    the RAPM fit consumes.
    """
    from nbare.rapm.stints import reconstruct_game

    all_blocks: list[OffenseBlock] = []
    total_skipped = 0
    warnings: list[str] = []

    for gid in game_ids:
        pbp = con.execute(
            "SELECT * FROM stg.pbp_event WHERE game_id = ?", [gid]
        ).pl()
        box = con.execute(
            "SELECT * FROM stg.box_player WHERE game_id = ?", [gid]
        ).pl()
        if pbp.is_empty() or box.is_empty():
            warnings.append(f"{gid}: missing pbp or box, skipped")
            continue
        recon = reconstruct_game(pbp)
        team_of = player_team_map(box)
        res = blocks_for_game(pbp, recon.stints, team_of)
        all_blocks.extend(res.blocks)
        total_skipped += res.skipped_stints
        warnings.extend(f"{gid}: {w}" for w in res.warnings)

    return BlockResult(
        blocks=all_blocks, skipped_stints=total_skipped, warnings=warnings
    )


def on_off_defense(blocks: list[OffenseBlock]) -> dict[int, "object"]:
    """Compute on/off defensive rating for every player from offense blocks.

    For each player we accumulate, over all defensive blocks (blocks where
    they are on the DEFENSE side), opponent points and possessions while
    they are ON the floor; and separately, over every block where they are
    NOT on the floor at all, the opponent points/possessions while OFF.

    "Off" here means blocks in which the player appears on neither side --
    i.e. genuinely resting. Returns OnOffDefense per player. Players who are
    never off (impossible in real data, possible in tiny synthetic sets) get
    off-possessions of zero and are effectively skipped by diagnostics.

    This is an INDEPENDENT check on defensive RAPM, not a competing metric.
    """
    from nbare.rapm.fit import OnOffDefense

    # opponent efficiency of a defensive block = points the OFFENSE scored.
    # For a player on defense in a block, the block's points are what they
    # allowed. For "off", we need blocks where the player is absent entirely.
    on_pts: dict[int, float] = {}
    on_poss: dict[int, float] = {}
    off_pts: dict[int, float] = {}
    off_poss: dict[int, float] = {}

    all_players: set[int] = set()
    for b in blocks:
        all_players |= b.offense | b.defense

    for pid in all_players:
        on_pts[pid] = on_poss[pid] = off_pts[pid] = off_poss[pid] = 0.0

    for b in blocks:
        on_court = b.offense | b.defense
        for pid in b.defense:
            on_pts[pid] += b.points
            on_poss[pid] += b.possessions
        # "off" = players not on the court at all for this defensive block
        for pid in all_players - on_court:
            off_pts[pid] += b.points
            off_poss[pid] += b.possessions

    out: dict[int, object] = {}
    for pid in all_players:
        on_eff = (100.0 * on_pts[pid] / on_poss[pid]) if on_poss[pid] > 0 else 0.0
        off_eff = (100.0 * off_pts[pid] / off_poss[pid]) if off_poss[pid] > 0 else 0.0
        out[pid] = OnOffDefense(
            player_id=pid,
            opp_pts_100_on=on_eff,
            opp_pts_100_off=off_eff,
            def_possessions_on=on_poss[pid],
            def_possessions_off=off_poss[pid],
        )
    return out