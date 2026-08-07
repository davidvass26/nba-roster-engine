"""Reconstruct on-court lineups (stints) from play-by-play.

What a stint is
---------------
A stint is a maximal span of game time during which the 10 players on the
floor do not change. Every substitution ends one stint and starts another.
Stage 2's RAPM design matrix has one row per stint, so this reconstruction
is the foundation the entire impact metric sits on.

The trap that makes this hard
-----------------------------
Play-by-play gives you substitution EVENTS, not lineups. To know who is on
the floor you would like to "start with the five starters and apply subs."
But nba.com play-by-play does not reliably label period starters, and a
player can be on the floor for minutes before appearing in any event. So
the robust method is:

  1. For each period, find every player who appears in an event BEFORE
     their first substitution-out. Those players were on the floor at the
     start of the period (you cannot sub out someone who was not already
     in). This recovers the period-opening lineup without trusting a
     "starter" flag.
  2. Walk events in order, applying each SUB (out, in) to the current
     lineup, cutting a stint at every sub.

Why this is validated, not trusted
-----------------------------------
Step 1 is an inference and it can be wrong -- a starter who neither scores,
rebounds, fouls, nor gets subbed in the first few minutes leaves no trace.
So the reconstruction is checked against the official box score: summed
stint seconds per player MUST equal box-score minutes. `validate_minutes`
is the gate. Do not build RAPM on a reconstruction that has not passed it;
a lineup that is quietly wrong produces a design matrix that is quietly
wrong, and the regression will not tell you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

# Regulation and overtime period lengths, in seconds.
REGULATION_PERIOD_S = 12 * 60
OVERTIME_PERIOD_S = 5 * 60

# Tolerance for the minutes check. Box-score minutes are rounded to whole
# minutes by some sources; PBP clocks are exact. A few seconds of drift per
# player is acceptable, a whole minute is a bug.
MINUTES_TOLERANCE_S = 90


def period_length_s(period: int) -> int:
    return REGULATION_PERIOD_S if period <= 4 else OVERTIME_PERIOD_S


@dataclass(frozen=True, slots=True)
class Stint:
    game_id: str
    period: int
    start_s: float          # seconds ELAPSED in the period at stint start
    end_s: float            # seconds elapsed at stint end
    lineup: frozenset[int]  # all 10 players on the floor (both teams)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ReconstructionResult:
    stints: list[Stint]
    # player_id -> reconstructed seconds on floor
    player_seconds: dict[int, float]
    warnings: list[str] = field(default_factory=list)


# --- event-type detection ------------------------------------------------
# nba.com uses different action_type strings across API versions. We match
# loosely on lowercased substrings so the code survives minor label churn.

def _is_sub(event_type: str | None) -> bool:
    return bool(event_type) and "substitution" in event_type.lower()


def _elapsed_in_period(period: int, clock_seconds_left: float | None) -> float:
    """Convert 'seconds remaining in period' to 'seconds elapsed'."""
    if clock_seconds_left is None:
        return 0.0
    return period_length_s(period) - clock_seconds_left


def reconstruct_game(
    pbp: pl.DataFrame,
    *,
    sub_out_col: str = "player1_id",
    sub_in_col: str = "player2_id",
) -> ReconstructionResult:
    """Reconstruct stints for one game from its play-by-play frame.

    Expects the columns produced by ingest.nba_stats: game_id, period,
    clock_seconds_left, event_type, player1_id (subbed OUT on a sub),
    player2_id (subbed IN on a sub), and any player id columns used to
    detect floor presence.

    NB on sub direction: nba.com's convention places the outgoing player in
    the first person slot and the incoming in the second on a substitution
    row. If a particular endpoint reverses this, the minutes check will
    fail loudly -- which is exactly what the gate is for.

    NB on event ordering: events are sorted by CLOCK, not by event_num.
    event_num (nba.com's own action counter) is not always chronological --
    confirmed on a real game where one substitution's recorded clock put it
    ~100s earlier than its event_num position implied, causing the stint
    cutter to "rewind" and double-count a 100-second window for every
    player on the floor during it (the exact-100s-per-player pattern is
    what gave this away; it was not garbage-time noise, it was one bad
    ordering assumption applied uniformly). clock_seconds_left is the
    actual chronological signal; event_num is only a reliable TIEBREAKER
    for genuinely simultaneous events (identical clock, e.g. a mass
    substitution wave), never the primary sort key.
    """
    warnings: list[str] = []
    if pbp.is_empty():
        return ReconstructionResult([], {}, ["empty play-by-play"])

    game_id = pbp["game_id"][0]
    stints: list[Stint] = []
    player_seconds: dict[int, float] = {}

    for period in sorted(pbp["period"].unique().to_list()):
        pev = (
            pbp.filter(pl.col("period") == period)
            .sort(["clock_seconds_left", "event_num"], descending=[True, False])
        )
        opening = _infer_period_openers(pev, sub_out_col, sub_in_col)
        if len(opening) != 10:
            warnings.append(
                f"period {period}: inferred {len(opening)} opening players, "
                "expected 10 (sparse period or missing events)"
            )

        current = set(opening)
        stint_start = 0.0
        plen = period_length_s(period)

        for row in pev.iter_rows(named=True):
            if not _is_sub(row.get("event_type")):
                continue
            t = _elapsed_in_period(period, row.get("clock_seconds_left"))
            out_id = row.get(sub_out_col)
            in_id = row.get(sub_in_col)

            # Close the stint at the substitution moment.
            if t > stint_start:
                _record(stints, player_seconds, game_id, period,
                        stint_start, t, current)
            stint_start = t

            if out_id in current:
                current.discard(out_id)
            elif out_id is not None:
                warnings.append(
                    f"period {period} @ {t:.0f}s: subbed out {out_id} who was "
                    "not on the floor (opening-lineup inference likely missed them)"
                )
            if in_id is not None:
                current.add(in_id)

        # Close the final stint of the period at the buzzer.
        if plen > stint_start:
            _record(stints, player_seconds, game_id, period,
                    stint_start, float(plen), current)

    return ReconstructionResult(stints, player_seconds, warnings)


def _infer_period_openers(
    pev: pl.DataFrame, sub_out_col: str, sub_in_col: str
) -> set[int]:
    """Players on the floor at the START of a period.

    A player was on the floor at tip-off of the period iff they do
    something (appear in any player slot of any event) BEFORE they are first
    subbed IN. Equivalently: the set of players subbed OUT before ever being
    subbed IN were openers, plus anyone active before their first sub-in.

    `pev` must already be sorted chronologically (by clock, not event_num --
    see `reconstruct_game`'s docstring). The "before/after" comparison here
    uses each row's POSITION in that order, not its raw event_num: a real
    game showed event_num is not always chronological, so comparing stored
    event_num values numerically would silently reintroduce the same bug
    even after the caller's sort was fixed.

    A Technical foul is EXCLUDED from "appearance" evidence. Confirmed on
    real data: a bench player can be charged a technical (for arguing from
    the bench, e.g. Matisse Thybulle, personId 1629680, box-score minutes
    0) without ever setting foot on the court -- unlike every other action
    type here (shots, rebounds, live fouls, assists), a technical does not
    require floor presence. Without this exclusion that player gets
    wrongly inferred as a period opener and credited a full period/game of
    phantom minutes. A player who genuinely IS on the floor when charged a
    technical (confirmed: real player technicals show a nonzero personal-
    foul count, "(P3.T4)", vs. a bench technical's "(P0.T1)") is unaffected
    -- he is still correctly identified as present via his other actions
    in the period.
    """
    subbed_in_at: dict[int, int] = {}   # player -> position of first sub-in
    appears_at: dict[int, int] = {}     # player -> position of first appearance

    id_cols = [c for c in ("player1_id", "player2_id", "player3_id") if c in pev.columns]

    for position, row in enumerate(pev.iter_rows(named=True)):
        is_sub = _is_sub(row.get("event_type"))
        in_id = row.get(sub_in_col) if is_sub else None
        for c in id_cols:
            pid = row.get(c)
            if pid is None:
                continue
            # On a sub row, the incoming player's "appearance" is the sub-in,
            # which does NOT prove they were already on the floor. Skip it.
            if is_sub and c == sub_in_col:
                continue
            # A Technical foul does not require floor presence -- see
            # docstring. Skip it as appearance evidence too.
            if row.get("event_action_type") == "Technical":
                continue
            appears_at.setdefault(pid, position)
        if in_id is not None:
            subbed_in_at.setdefault(in_id, position)

    openers: set[int] = set()
    for pid, first_seen in appears_at.items():
        first_in = subbed_in_at.get(pid)
        # On the floor at start if they appeared before their first sub-in
        # (or were never subbed in at all).
        if first_in is None or first_seen < first_in:
            openers.add(pid)
    return openers


def _record(stints, player_seconds, game_id, period, start_s, end_s, lineup):
    stint = Stint(
        game_id=game_id, period=period, start_s=start_s, end_s=end_s,
        lineup=frozenset(lineup),
    )
    stints.append(stint)
    dur = stint.duration_s
    for pid in lineup:
        player_seconds[pid] = player_seconds.get(pid, 0.0) + dur


# --- the validation gate -------------------------------------------------

@dataclass
class MinutesCheck:
    passed: bool
    max_error_s: float
    offenders: list[tuple[int, float, float]]  # (player, recon_s, box_s)

    def summary(self) -> str:
        if self.passed:
            return f"minutes check PASSED (max error {self.max_error_s:.0f}s)"
        lines = [f"minutes check FAILED (max error {self.max_error_s:.0f}s):"]
        for pid, r, b in self.offenders[:10]:
            lines.append(f"  player {pid}: reconstructed {r:.0f}s vs box {b:.0f}s")
        return "\n".join(lines)


def validate_minutes(
    result: ReconstructionResult,
    box: pl.DataFrame,
    *,
    player_col: str = "nba_player_id",
    seconds_col: str = "seconds_played",
    tolerance_s: float = MINUTES_TOLERANCE_S,
) -> MinutesCheck:
    """Compare reconstructed on-floor seconds to the official box score.

    This is the Stage 2 gate. Every player's reconstructed seconds must
    match their box-score seconds within tolerance. A failure means the
    lineup reconstruction is wrong -- usually a reversed sub direction or a
    missed period opener -- and RAPM built on it would be silently corrupt.
    """
    box_seconds = {
        r[player_col]: float(r[seconds_col] or 0)
        for r in box.iter_rows(named=True)
    }
    offenders: list[tuple[int, float, float]] = []
    max_err = 0.0

    all_players = set(box_seconds) | set(result.player_seconds)
    for pid in all_players:
        recon = result.player_seconds.get(pid, 0.0)
        official = box_seconds.get(pid, 0.0)
        err = abs(recon - official)
        max_err = max(max_err, err)
        if err > tolerance_s:
            offenders.append((pid, recon, official))

    offenders.sort(key=lambda t: abs(t[1] - t[2]), reverse=True)
    return MinutesCheck(
        passed=not offenders, max_error_s=max_err, offenders=offenders
    )