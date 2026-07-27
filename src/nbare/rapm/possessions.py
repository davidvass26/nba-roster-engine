"""Estimate possessions per stint, split by which team was on offense.

Why possessions, not seconds
----------------------------
Basketball value is per-possession. A stint with 8 possessions in 90
seconds is worth more evidence than one with 2 possessions in the same
time. RAPM therefore weights each observation by possession count, not
clock time. Using time would systematically misweight fast- and slow-pace
lineups.

Why split by offensive team
---------------------------
For an offense/defense split we cannot treat a stint as one-directional.
During a single stint both teams have the ball, alternating. So each stint
contributes TWO observations to the design matrix:

  - home on offense vs away on defense, over the home team's possessions
  - away on offense vs home on defense, over the away team's possessions

Each observation carries its own possession count and its own points. This
module produces those two halves per stint.

The possession formula
----------------------
The standard estimate for one team's possessions is:

    POSS = FGA - OREB + TOV + 0.44 * FTA

The 0.44 approximates that not every free-throw trip ends a possession
(and-1s, technicals, away-from-play fouls). It is an estimate; the whole
metric is downstream of it, so it is worth getting the event classification
right, but no possession estimate is exact and that is fine -- RAPM only
needs relative pace to be sensible, not perfect.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

FTA_POSSESSION_WEIGHT = 0.44


@dataclass(frozen=True, slots=True)
class PossessionSplit:
    """One stint's two half-observations for the design matrix.

    `home_off_*` is the home lineup on offense against the away lineup on
    defense; `away_off_*` is the reverse. Points and possessions are
    per-half so each becomes an independent, possession-weighted row.
    """

    game_id: str
    period: int
    home_lineup: frozenset[int]
    away_lineup: frozenset[int]
    home_off_poss: float
    home_off_pts: int
    away_off_poss: float
    away_off_pts: int


# --- event classification ------------------------------------------------
# nba.com action_type labels vary; match loosely on lowercased substrings.

def _is_fga(row: dict) -> bool:
    et = (row.get("event_type") or "").lower()
    return "shot" in et  # "Made Shot" / "Missed Shot"


def _is_made_shot(row: dict) -> bool:
    et = (row.get("event_type") or "").lower()
    return "made" in et and "shot" in et


def _is_turnover(row: dict) -> bool:
    return "turnover" in (row.get("event_type") or "").lower()


def _is_oreb(row: dict) -> bool:
    et = (row.get("event_type") or "").lower()
    at = (row.get("event_action_type") or "").lower()
    return "rebound" in et and "offensive" in at


def _is_fta(row: dict) -> bool:
    return "free throw" in (row.get("event_type") or "").lower()


def _points_of(row: dict) -> int:
    """Points scored on this event, from the running score if present."""
    et = (row.get("event_type") or "").lower()
    if "free throw" in et and "made" in et:
        return 1
    if "made" in et and "shot" in et:
        # 3 if the description flags a three, else 2.
        desc = (row.get("description") or "").lower()
        return 3 if "3pt" in desc or "three" in desc else 2
    return 0


def estimate_team_possessions(
    fga: int, oreb: int, tov: int, fta: int
) -> float:
    """Standard single-team possession estimate."""
    return fga - oreb + tov + FTA_POSSESSION_WEIGHT * fta


def possessions_for_stint_events(
    events: pl.DataFrame,
    home_team_id: int,
    away_team_id: int,
) -> tuple[float, int, float, int]:
    """Aggregate one stint's events into (home_off_poss, home_off_pts,
    away_off_poss, away_off_pts).

    Offense is attributed by the acting team_id on each event. A shot by a
    home player counts toward home offense; the defensive lineup is simply
    the other five, so no per-event defensive attribution is needed.
    """
    tallies = {
        home_team_id: {"fga": 0, "oreb": 0, "tov": 0, "fta": 0, "pts": 0},
        away_team_id: {"fga": 0, "oreb": 0, "tov": 0, "fta": 0, "pts": 0},
    }
    for row in events.iter_rows(named=True):
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

    h, a = tallies[home_team_id], tallies[away_team_id]
    home_poss = estimate_team_possessions(h["fga"], h["oreb"], h["tov"], h["fta"])
    away_poss = estimate_team_possessions(a["fga"], a["oreb"], a["tov"], a["fta"])
    return home_poss, h["pts"], away_poss, a["pts"]