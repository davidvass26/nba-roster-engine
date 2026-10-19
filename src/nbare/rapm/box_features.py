"""Per-game box scores -> per-player `BoxScoreRow`s for the RAPM prior.

Why this module exists
----------------------
`rapm/prior.py` and `fit.fit_rapm_bayesian` were built and validated on
synthetic `BoxScoreRow`s, but nothing produced real ones, so every real fit
shrank toward ZERO rather than toward the box score. This is the missing
connector: it aggregates per-player-game counting stats (from
`ingest.nba_stats.box_detail_from_cache`) into the exact feature set the
prior consumes. It deliberately does NOT redefine `BoxScoreRow` or the
feature lists -- those stay owned by `prior.py`.

Per-100 rates without on-court possession counts
------------------------------------------------
The box score gives a player's counting stats and seconds, not the number
of possessions he was on the floor for. We estimate it the standard way
(the same idea as Basketball-Reference's per-100 and rebound-rate
formulas): a player's share of his team's floor time,

    share = seconds_played / (team_seconds / 5)

times the team's possessions in that game (POSS = FGA - OREB + TOV +
0.44*FTA, the same estimate `rapm/possessions.py` uses). His defensive
possessions use the OPPONENT's possessions; his defensive-rebound
opportunities are share * (team DREB + opponent OREB). Each is summed over
games first and divided once, so a 4-minute cameo does not get the same
weight as a 40-minute start.

These per-100 rates are box-internal (both numerator and denominator come
from the box score, over the same games), so they are self-consistent even
though the RAPM design only includes games that pass the minutes gate. The
`off_possessions` / `def_possessions` fields are different: `build_prior`
uses them to decide whose plain-ridge RAPM is reliable enough to TRAIN the
prior, so when the design's own per-side possessions are supplied we use
those -- reliability should be measured on the data the rating came from.

The traps handled here
----------------------
- Tiny samples. A player with 30 possessions can post a 150 pts/100 or a
  1.2 TS%. Feed that to the prior and the prior mean is extreme for
  exactly the players whose ratings the prior dominates. Players below
  `min_box_possessions` get NO row, so `build_prior` leaves them at a zero
  (league-average) prior mean -- the safe default, not a guess.
- Position is only partly observed. nba.com labels only starters, and
  `stg.player.position` is empty. `position_big` is 1 only when a majority
  of a player's labeled starts are at "C"; a player with no labeled starts
  gets 0. That 0 means "not known to be a big", which is the best the data
  supports; the count of such players is returned so the caller can report
  it rather than hide it.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nbare.rapm.possessions import FTA_POSSESSION_WEIGHT
from nbare.rapm.prior import BoxScoreRow

# Minimum ESTIMATED offensive possessions (from the box score) before a
# player's box-score rates are trusted as prior features. ~250 possessions
# is roughly 5 full games; below that, per-100 rates are mostly noise.
# A judgment call, not a derived constant -- see the module docstring.
DEFAULT_MIN_BOX_POSSESSIONS = 250.0

_COUNT_COLS = ("pts", "ast", "tov", "fga", "fta", "oreb", "dreb", "stl", "blk")


@dataclass
class BoxFeatureResult:
    rows: list[BoxScoreRow]
    n_players_seen: int
    n_below_min: int          # players dropped for too few box possessions
    n_position_unknown: int   # rows whose position_big is 0 for lack of starts


def _team_poss(fga, oreb, tov, fta):
    return fga - oreb + tov + FTA_POSSESSION_WEIGHT * fta


def box_score_rows(
    detail: pl.DataFrame,
    side_possessions: tuple[dict[int, float], dict[int, float]] | None = None,
    *,
    min_box_possessions: float = DEFAULT_MIN_BOX_POSSESSIONS,
) -> BoxFeatureResult:
    """Aggregate per-player-game box detail (BOX_DETAIL_SCHEMA) into one
    `BoxScoreRow` per player. See the module docstring for the formulas.

    `side_possessions`, if given, is (offensive, defensive) possessions per
    player from the RAPM design (`fit.player_side_possessions`); it fills
    the rows' `off_possessions`/`def_possessions`. Otherwise the box-score
    estimates are used.
    """
    if detail.is_empty():
        return BoxFeatureResult([], 0, 0, 0)

    d = detail.with_columns(
        pl.col("seconds_played").fill_null(0).cast(pl.Float64),
        *[pl.col(c).fill_null(0).cast(pl.Float64) for c in _COUNT_COLS],
    )

    team = d.group_by(["game_id", "nba_team_id"]).agg(
        pl.col("seconds_played").sum().alias("team_seconds"),
        *[pl.col(c).sum().alias(f"team_{c}") for c in ("fga", "fta", "oreb", "dreb", "tov")],
    ).with_columns(
        _team_poss(pl.col("team_fga"), pl.col("team_oreb"),
                   pl.col("team_tov"), pl.col("team_fta")).alias("team_poss"),
    )
    opp = team.select(
        "game_id",
        pl.col("nba_team_id").alias("opp_team_id"),
        pl.col("team_poss").alias("opp_poss"),
        pl.col("team_oreb").alias("opp_oreb"),
    )
    team = team.join(opp, on="game_id").filter(
        pl.col("nba_team_id") != pl.col("opp_team_id")
    )

    pg = d.join(team, on=["game_id", "nba_team_id"]).with_columns(
        pl.when(pl.col("team_seconds") > 0)
        .then(pl.col("seconds_played") / (pl.col("team_seconds") / 5.0))
        .otherwise(0.0)
        .alias("share")
    ).with_columns(
        (pl.col("share") * pl.col("team_poss")).alias("est_off"),
        (pl.col("share") * pl.col("opp_poss")).alias("est_def"),
        (pl.col("share") * (pl.col("team_dreb") + pl.col("opp_oreb"))).alias("dreb_avail"),
        (pl.col("position") != "").alias("labeled"),
        (pl.col("position") == "C").alias("at_c"),
    )

    agg = pg.group_by("nba_player_id").agg(
        *[pl.col(c).sum() for c in _COUNT_COLS],
        pl.col("est_off").sum(), pl.col("est_def").sum(),
        pl.col("dreb_avail").sum(),
        pl.col("labeled").sum().alias("n_labeled"),
        pl.col("at_c").sum().alias("n_at_c"),
    )

    off_side, def_side = side_possessions if side_possessions else ({}, {})
    rows: list[BoxScoreRow] = []
    n_below = n_pos_unknown = 0
    for r in agg.iter_rows(named=True):
        if r["est_off"] < min_box_possessions or r["est_def"] <= 0:
            n_below += 1
            continue
        pid = int(r["nba_player_id"])
        off100 = 100.0 / r["est_off"]
        def100 = 100.0 / r["est_def"]
        shot_att = 2.0 * (r["fga"] + FTA_POSSESSION_WEIGHT * r["fta"])
        if r["n_labeled"] == 0:
            n_pos_unknown += 1
        rows.append(BoxScoreRow(
            player_id=pid,
            off_possessions=off_side.get(pid, r["est_off"]),
            def_possessions=def_side.get(pid, r["est_def"]),
            pts_100=r["pts"] * off100,
            ast_100=r["ast"] * off100,
            tov_100=r["tov"] * off100,
            ts_pct=r["pts"] / shot_att if shot_att > 0 else 0.0,
            stl_100=r["stl"] * def100,
            blk_100=r["blk"] * def100,
            dreb_rate=r["dreb"] / r["dreb_avail"] if r["dreb_avail"] > 0 else 0.0,
            position_big=int(r["n_labeled"] > 0 and r["n_at_c"] * 2 > r["n_labeled"]),
        ))

    rows.sort(key=lambda b: b.player_id)
    return BoxFeatureResult(
        rows=rows,
        n_players_seen=agg.height,
        n_below_min=n_below,
        n_position_unknown=n_pos_unknown,
    )
