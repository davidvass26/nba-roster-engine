"""Generate synthetic play-by-play with KNOWN ground truth.

Because stats.nba.com is unreachable from CI (and to make the tests
deterministic), we generate fake games whose true lineup timeline we
control, emit play-by-play events consistent with that timeline, and then
assert that reconstruction recovers the truth exactly. If the reconstructor
can invert a game we built forwards, the logic is right independent of any
real data.

This is a test/dev utility, not part of the ingestion path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from nbare.rapm.stints import period_length_s

PBP_SCHEMA = {
    "game_id": pl.Utf8, "event_num": pl.Int32, "period": pl.Int16,
    "clock_seconds_left": pl.Float64, "event_type": pl.Utf8,
    "event_action_type": pl.Utf8, "description": pl.Utf8, "team_id": pl.Int64,
    "player1_id": pl.Int64, "player2_id": pl.Int64, "player3_id": pl.Int64,
    "home_score": pl.Int16, "away_score": pl.Int16,
}


@dataclass
class SyntheticGame:
    game_id: str
    pbp: pl.DataFrame
    box_seconds: dict[int, float]   # ground-truth seconds per player


def make_game(
    game_id: str = "0029999001",
    n_periods: int = 4,
    subs_per_period: int = 3,
    seed: int = 0,
) -> SyntheticGame:
    """Build a synthetic game with a controlled substitution pattern.

    Home team uses player ids 1..N, away 101..1NN. Five starters per side;
    a bench of a few players per side rotates in via substitutions. We track
    exact on-floor seconds as we go, so box_seconds is exact ground truth.
    """
    rng = random.Random(seed)
    home_starters = [1, 2, 3, 4, 5]
    home_bench = [6, 7, 8, 9, 10]
    away_starters = [101, 102, 103, 104, 105]
    away_bench = [106, 107, 108, 109, 110]

    events: list[dict] = []
    box: dict[int, float] = {}
    event_num = 0

    def add_event(period, clock_left, etype, p1=None, p2=None, filler=False):
        nonlocal event_num
        event_num += 1
        events.append({
            "game_id": game_id, "event_num": event_num, "period": period,
            "clock_seconds_left": float(clock_left), "event_type": etype,
            "event_action_type": None, "description": "synthetic",
            "team_id": None, "player1_id": p1, "player2_id": p2,
            "player3_id": None, "home_score": 0, "away_score": 0,
        })

    for period in range(1, n_periods + 1):
        plen = period_length_s(period)
        home_on = list(home_starters)
        away_on = list(away_starters)

        # A non-sub "action" event early in the period for each starter, so
        # the opener-inference has something to latch onto (mirrors real PBP
        # where starters do things before the first sub).
        for i, pid in enumerate(home_on + away_on):
            add_event(period, plen - 1 - i, "Made Shot", p1=pid)

        # Schedule substitutions at descending clock times.
        sub_times = sorted(
            rng.sample(range(60, plen - 30), subs_per_period), reverse=True
        )
        last_t = 0.0  # elapsed seconds of previous sub
        import itertools
        home_bench_iter = itertools.cycle(home_bench)
        away_bench_iter = itertools.cycle(away_bench)

        for clock_left in sub_times:
            elapsed = plen - clock_left
            # credit everyone currently on the floor for [last_t, elapsed]
            for pid in home_on + away_on:
                box[pid] = box.get(pid, 0.0) + (elapsed - last_t)
            last_t = elapsed

            # Alternate subbing a home then away player. Pick a bench player
            # not already on the floor (cycle is unbounded so this always
            # terminates as long as the bench has someone off the floor).
            if rng.random() < 0.5:
                out_id = rng.choice(home_on)
                in_id = next(home_bench_iter)
                guard = 0
                while in_id in home_on and guard < 100:
                    in_id = next(home_bench_iter)
                    guard += 1
                home_on[home_on.index(out_id)] = in_id
            else:
                out_id = rng.choice(away_on)
                in_id = next(away_bench_iter)
                guard = 0
                while in_id in away_on and guard < 100:
                    in_id = next(away_bench_iter)
                    guard += 1
                away_on[away_on.index(out_id)] = in_id
            add_event(period, clock_left, "Substitution", p1=out_id, p2=in_id)

        # credit the final segment [last_t, plen] to whoever is on the floor
        for pid in home_on + away_on:
            box[pid] = box.get(pid, 0.0) + (plen - last_t)

    pbp = pl.DataFrame(events, schema=PBP_SCHEMA)
    return SyntheticGame(game_id=game_id, pbp=pbp, box_seconds=box)


def box_frame(game: SyntheticGame) -> pl.DataFrame:
    """Ground-truth box score in the shape validate_minutes expects."""
    return pl.DataFrame(
        [
            {"game_id": game.game_id, "nba_player_id": pid,
             "nba_team_id": 1 if pid < 100 else 2,
             "seconds_played": int(round(secs)), "pts": 0, "reb": 0,
             "ast": 0, "started": False}
            for pid, secs in game.box_seconds.items()
        ]
    )


HOME_TEAM_ID = 1
AWAY_TEAM_ID = 2


@dataclass
class ScoringGame:
    game_id: str
    pbp: pl.DataFrame
    box_seconds: dict[int, float]
    # ground-truth per-stint: (home_lineup, away_lineup, home_pts, home_poss,
    #                          away_pts, away_poss)
    truth: list[tuple[frozenset[int], frozenset[int], int, int, int, int]]


def make_scoring_game(
    game_id: str = "0029999002",
    n_periods: int = 4,
    subs_per_period: int = 3,
    seed: int = 0,
) -> ScoringGame:
    """Synthetic game that emits real scoring events with team_id, so the
    stints->blocks connector can be validated against known truth.

    Within each stint we plant a fixed number of made 2PT field goals for
    each team (varying by stint), each tagged with the scoring team's
    team_id. Possessions therefore equal made FGA per team (no misses,
    rebounds, turnovers, or FTs), which keeps the ground-truth possession
    count exact: POSS = FGA. This is a deliberately simple scoring model --
    its only job is to give the connector a known answer to recover.
    """
    rng = random.Random(seed)
    home_starters = [1, 2, 3, 4, 5]
    home_bench = [6, 7, 8, 9, 10]
    away_starters = [101, 102, 103, 104, 105]
    away_bench = [106, 107, 108, 109, 110]

    events: list[dict] = []
    box: dict[int, float] = {}
    truth: list = []
    event_num = 0

    def add(period, clock_left, etype, team_id=None, p1=None, p2=None, desc="synthetic"):
        nonlocal event_num
        event_num += 1
        events.append({
            "game_id": game_id, "event_num": event_num, "period": period,
            "clock_seconds_left": float(clock_left), "event_type": etype,
            "event_action_type": None, "description": desc,
            "team_id": team_id, "player1_id": p1, "player2_id": p2,
            "player3_id": None, "home_score": 0, "away_score": 0,
        })

    import itertools

    for period in range(1, n_periods + 1):
        plen = period_length_s(period)
        home_on = list(home_starters)
        away_on = list(away_starters)

        # Opener-detection needs each starter to act before any sub. Emit a
        # made FG for each starter near the top of the period. These are real
        # scoring events, so they must be counted in the first segment's
        # ground truth (below), not left unrecorded.
        for i, pid in enumerate(home_on):
            add(period, plen - 1 - i, "Made Shot", team_id=HOME_TEAM_ID, p1=pid)
        for i, pid in enumerate(away_on):
            add(period, plen - 6 - i, "Made Shot", team_id=AWAY_TEAM_ID, p1=pid)
        opener_home_fg = len(home_on)  # 5
        opener_away_fg = len(away_on)  # 5

        sub_times = sorted(
            rng.sample(range(60, plen - 30), subs_per_period), reverse=True
        )
        # Stint boundaries in elapsed seconds: 0, each sub, plen.
        boundaries = [0.0] + [plen - t for t in sub_times] + [float(plen)]
        home_bench_iter = itertools.cycle(home_bench)
        away_bench_iter = itertools.cycle(away_bench)

        # We must know the lineup during each inter-sub segment to plant
        # scoring for it. Walk boundaries, plant scoring mid-segment, then
        # apply the sub at the segment end.
        for si in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[si], boundaries[si + 1]
            # credit floor time
            for pid in home_on + away_on:
                box[pid] = box.get(pid, 0.0) + (seg_end - seg_start)

            # Plant scoring in the MIDDLE of the segment so it lands strictly
            # inside [seg_start, seg_end).
            mid_elapsed = (seg_start + seg_end) / 2.0
            mid_clock = plen - mid_elapsed
            home_fg = rng.randint(1, 4)
            away_fg = rng.randint(1, 4)
            for _ in range(home_fg):
                shooter = rng.choice(home_on)
                add(period, mid_clock, "Made Shot", team_id=HOME_TEAM_ID, p1=shooter)
            for _ in range(away_fg):
                shooter = rng.choice(away_on)
                add(period, mid_clock, "Made Shot", team_id=AWAY_TEAM_ID, p1=shooter)

            # The period-opener shots live in the FIRST segment's window, so
            # fold them into segment 0's truth to keep it exact.
            seg_home_fg = home_fg + (opener_home_fg if si == 0 else 0)
            seg_away_fg = away_fg + (opener_away_fg if si == 0 else 0)
            truth.append((
                frozenset(home_on), frozenset(away_on),
                seg_home_fg * 2, seg_home_fg,   # pts, poss (poss == made FGA)
                seg_away_fg * 2, seg_away_fg,
            ))

            # Apply the substitution at seg_end (except after the last seg).
            if si < len(boundaries) - 2:
                clock_left = sub_times[si]
                if rng.random() < 0.5:
                    out_id = rng.choice(home_on)
                    in_id = next(home_bench_iter)
                    guard = 0
                    while in_id in home_on and guard < 100:
                        in_id = next(home_bench_iter); guard += 1
                    home_on[home_on.index(out_id)] = in_id
                    add(period, clock_left, "Substitution", team_id=HOME_TEAM_ID,
                        p1=out_id, p2=in_id)
                else:
                    out_id = rng.choice(away_on)
                    in_id = next(away_bench_iter)
                    guard = 0
                    while in_id in away_on and guard < 100:
                        in_id = next(away_bench_iter); guard += 1
                    away_on[away_on.index(out_id)] = in_id
                    add(period, clock_left, "Substitution", team_id=AWAY_TEAM_ID,
                        p1=out_id, p2=in_id)

    pbp = pl.DataFrame(events, schema=PBP_SCHEMA)
    return ScoringGame(game_id=game_id, pbp=pbp, box_seconds=box, truth=truth)


def scoring_box_frame(game: ScoringGame) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"game_id": game.game_id, "nba_player_id": pid,
             "nba_team_id": HOME_TEAM_ID if pid < 100 else AWAY_TEAM_ID,
             "seconds_played": int(round(secs)), "pts": 0, "reb": 0,
             "ast": 0, "started": False}
            for pid, secs in game.box_seconds.items()
        ]
    )