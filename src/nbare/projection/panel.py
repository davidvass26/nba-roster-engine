"""Real-data connector for Stage 3: per-season RAPM fits -> `PlayerSeason` panel.

Why this exists
---------------
Layers 1-3 (`delta.py`, `hierarchy.py`, `baseline.py`) were validated on
synthetic `PlayerSeason` panels only. Nothing assembled a REAL one, which
needs three things no single source had: a rating per player per SEASON
(the exported RAPM is a pooled multi-season fit -- one number per player,
useless for aging), that player's age in that season, and a position group.
This module joins them. Like `rapm/blocks.py` (the stints -> design
connector), it computes nothing statistical; a bug here would silently
corrupt every projection, so it is tested against planted panels.

Why separate single-season fits, not the pooled fit
----------------------------------------------------
Aging is a within-player change over time. The pooled fit assumes a player
is the SAME across every season in the window -- exactly the assumption an
aging curve exists to relax -- so its rating carries no age signal at all.
Each season is therefore fit on its own (same minutes gate, same box-score
prior), and those per-season ratings are the observations -- specifically
their PRIOR-DIVIDED measurements (`rapm.fit.likelihood_measurements`): what
that season's possession data alone says, with a standard error. The
shrunk ridge estimates themselves would count the prior twice (see that
function's docstring for what that did to the real fit).

What is dropped, and why it is reported rather than hidden
----------------------------------------------------------
- No age for that season (not in nba.com's LeagueDashPlayerStats): the row
  cannot be placed on the age axis, so it is dropped. Never imputed.
- Ages that do not step by exactly +1 between a player's consecutive
  seasons: an id/age data error. The player's rows are dropped entirely
  rather than letting a bad age create a fake age transition.
- Below `min_possessions` on either side: the season's rating is almost
  pure prior, not evidence. (The hierarchy also weights by possessions;
  this just removes rows with essentially no data.)
- No measurement on one side (`likelihood_measurements` omits a rating
  whose posterior is ~all prior): the row is dropped -- a player-season
  needs both sides because offense and defense share one row.
- Unknown position: kept, in its own group `UNKNOWN_POSITION`. Forcing such
  a player into "G"/"F"/"C" would be a guess; a separate group is pooled
  toward the league curve like every other group, which is the honest
  default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nbare.projection.types import PlayerSeason

UNKNOWN_POSITION = "U"

# Per-side possessions a player-season needs to enter the panel. ~300 is
# about six full games; below it a single-season rating is mostly prior.
DEFAULT_MIN_POSSESSIONS = 300.0


@dataclass(frozen=True, slots=True)
class SeasonRatings:
    """One season's single-season fit, reduced to what the panel needs:
    per-side measurements (pts/100, good is positive), their standard
    errors, and per-side possessions. In the real pipeline offense/defense
    and off_se/def_se come from `rapm.fit.likelihood_measurements`."""

    season: str
    offense: dict[int, float]
    defense: dict[int, float]
    off_possessions: dict[int, float]
    def_possessions: dict[int, float]
    # Measurement SEs; empty = unknown (rows then carry se=None and the
    # hierarchy must learn the noise itself).
    off_se: dict[int, float] = field(default_factory=dict)
    def_se: dict[int, float] = field(default_factory=dict)


@dataclass
class PanelResult:
    seasons: list[PlayerSeason]
    dropped_no_age: int
    dropped_low_possessions: int
    dropped_age_inconsistent_players: int
    dropped_no_measurement: int
    unknown_position_players: int


def build_panel(
    season_ratings: list[SeasonRatings],
    ages: dict[tuple[int, str], int],
    positions: dict[int, str],
    *,
    min_possessions: float = DEFAULT_MIN_POSSESSIONS,
) -> PanelResult:
    """Join per-season ratings with ages and positions into a panel sorted
    by (player_id, season). See the module docstring for the drop rules."""
    rows: list[PlayerSeason] = []
    no_age = low_poss = no_meas = 0
    for sr in sorted(season_ratings, key=lambda s: s.season):
        for pid, off in sr.offense.items():
            po = sr.off_possessions.get(pid, 0.0)
            pd = sr.def_possessions.get(pid, 0.0)
            if po < min_possessions or pd < min_possessions:
                low_poss += 1
                continue
            if pid not in sr.defense:
                no_meas += 1
                continue
            age = ages.get((pid, sr.season))
            if age is None:
                no_age += 1
                continue
            rows.append(PlayerSeason(
                player_id=pid,
                season=sr.season,
                age=age,
                position_group=positions.get(pid, UNKNOWN_POSITION),
                off_rating=off,
                def_rating=sr.defense[pid],
                off_possessions=po,
                def_possessions=pd,
                off_se=sr.off_se.get(pid),
                def_se=sr.def_se.get(pid),
            ))

    season_order = {s: k for k, s in enumerate(sorted({sr.season for sr in season_ratings}))}
    by_player: dict[int, list[PlayerSeason]] = {}
    for r in rows:
        by_player.setdefault(r.player_id, []).append(r)
    bad: set[int] = set()
    for pid, prs in by_player.items():
        prs.sort(key=lambda r: season_order[r.season])
        for a, b in zip(prs, prs[1:]):
            gap = season_order[b.season] - season_order[a.season]
            if b.age - a.age != gap:
                bad.add(pid)
                break

    kept = [r for r in rows if r.player_id not in bad]
    kept.sort(key=lambda r: (r.player_id, r.season))
    unknown = len({r.player_id for r in kept if r.position_group == UNKNOWN_POSITION})
    return PanelResult(
        seasons=kept,
        dropped_no_age=no_age,
        dropped_no_measurement=no_meas,
        dropped_low_possessions=low_poss,
        dropped_age_inconsistent_players=len(bad),
        unknown_position_players=unknown,
    )


def next_season(season: str, k: int = 1) -> str:
    """'2025-26' -> '2026-27' (k=1), '2029-30' (k=4).

    >>> next_season("2025-26"), next_season("2025-26", 4), next_season("2099-00")
    ('2026-27', '2029-30', '2100-01')
    """
    start = int(season[:4]) + k
    return f"{start}-{(start + 1) % 100:02d}"
