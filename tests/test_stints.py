"""Tests for Stage 2 stint reconstruction.

The strategy: generate synthetic games whose true lineup timeline we
control, then assert reconstruction recovers it exactly. A reconstructor
that can invert a forward-built game is correct independent of real data.
The reversed-sub test proves the validation GATE catches the classic bug,
so the gate is trustworthy on real data where we have no ground truth.
"""

from __future__ import annotations

import pytest

from nbare.rapm.stints import (
    REGULATION_PERIOD_S,
    period_length_s,
    reconstruct_game,
    validate_minutes,
)
from nbare.rapm.synthetic import box_frame, make_game


def test_period_lengths():
    assert period_length_s(1) == REGULATION_PERIOD_S
    assert period_length_s(4) == 12 * 60
    assert period_length_s(5) == 5 * 60  # overtime


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 42])
def test_reconstruction_recovers_ground_truth(seed):
    """Across many random substitution patterns, reconstructed seconds must
    equal the synthetic ground truth exactly."""
    game = make_game(seed=seed)
    res = reconstruct_game(game.pbp)
    check = validate_minutes(res, box_frame(game))
    assert check.passed, check.summary()
    assert check.max_error_s == 0


def test_total_floor_seconds_invariant():
    """Exactly 10 players are on the floor at all times, so total
    floor-seconds == 10 * game length regardless of substitutions."""
    game = make_game(seed=5)
    res = reconstruct_game(game.pbp)
    total_game_s = sum(period_length_s(p) for p in range(1, 5))
    assert abs(sum(res.player_seconds.values()) - 10 * total_game_s) < 1e-6


def test_every_stint_has_ten_players():
    game = make_game(seed=9)
    res = reconstruct_game(game.pbp)
    # Openers are inferred; on a clean synthetic game every stint is 10.
    for st in res.stints:
        assert len(st.lineup) == 10, f"stint {st} has {len(st.lineup)} players"


def test_gate_catches_reversed_sub_direction():
    """The canonical real-world bug: in/out swapped. The gate MUST fail."""
    game = make_game(seed=1)
    bad = reconstruct_game(
        game.pbp, sub_out_col="player2_id", sub_in_col="player1_id"
    )
    check = validate_minutes(bad, box_frame(game))
    assert not check.passed
    assert check.max_error_s > 100


def test_overtime_game_reconstructs():
    game = make_game(seed=3, n_periods=5)  # includes one OT period
    res = reconstruct_game(game.pbp)
    check = validate_minutes(res, box_frame(game))
    assert check.passed, check.summary()


def test_empty_pbp_is_handled():
    import polars as pl

    from nbare.rapm.synthetic import PBP_SCHEMA

    empty = pl.DataFrame(schema=PBP_SCHEMA)
    res = reconstruct_game(empty)
    assert res.stints == []
    assert "empty" in res.warnings[0].lower()


def test_more_substitutions_still_balances():
    game = make_game(seed=11, subs_per_period=6)
    res = reconstruct_game(game.pbp)
    check = validate_minutes(res, box_frame(game))
    assert check.passed, check.summary()


def test_event_num_out_of_sync_with_clock_does_not_double_count():
    """Real bug found on real 2025-26 data: nba.com's event_num (its own
    action counter) is NOT always chronological -- one substitution's
    recorded clock put it ~100s earlier than its event_num position
    implied. Sorting by event_num (the old behavior) processed that sub
    too late, causing `stint_start` to jump BACKWARDS in time and
    double-count the resulting overlap window for whoever was on the floor
    during it. The giveaway on the real game was every affected player
    being off by exactly the same ~100s -- not noisy, a clean uniform
    offset, which is what a double-counted shared window looks like.

    Minimal repro: two substitutions whose clocks disagree with their
    event_num order. Sorting by event_num would process the elapsed=120
    sub BEFORE the elapsed=70 one, jump stint_start back to 70, and
    double-count [70, 120) for whichever players were on the floor then.
    """
    import polars as pl

    from nbare.rapm.synthetic import PBP_SCHEMA

    def ev(event_num, clock_s, event_type, p1=None, p2=None):
        return {
            "game_id": "g1", "event_num": event_num, "period": 1,
            "clock_seconds_left": 720.0 - clock_s, "event_type": event_type,
            "event_action_type": None, "description": "test",
            "team_id": None, "player1_id": p1, "player2_id": p2,
            "player3_id": None, "home_score": 0, "away_score": 0,
        }

    events = [
        # Opening lineup {A=1, B=2}, established by acting early.
        ev(1, 5, "Made Shot", p1=1),
        ev(2, 6, "Made Shot", p1=2),
        # event_num 3 (elapsed 120s): A out, C in -- recorded LOWER
        # event_num than the sub below, but LATER in game-clock time.
        ev(3, 120, "Substitution", p1=1, p2=3),
        # event_num 4 (elapsed 70s): B out, D in -- HIGHER event_num, but
        # chronologically EARLIER (this is the out-of-order one).
        ev(4, 70, "Substitution", p1=2, p2=4),
    ]
    pbp = pl.DataFrame(events, schema=PBP_SCHEMA)

    res = reconstruct_game(pbp)

    # True chronological order: B->D at t=70, then A->C at t=120.
    # [0,70): {1,2}  [70,120): {1,4}  [120,720): {3,4}
    assert res.player_seconds[1] == pytest.approx(120.0)   # A: 0-120
    assert res.player_seconds[2] == pytest.approx(70.0)    # B: 0-70
    assert res.player_seconds[4] == pytest.approx(650.0)   # D: 70-720
    assert res.player_seconds[3] == pytest.approx(600.0)   # C: 120-720
    # No double-counting: total floor-seconds == 2 slots * period length.
    assert sum(res.player_seconds.values()) == pytest.approx(1440.0)
    # No stint should ever start before an earlier stint's own end.
    ends = [s.end_s for s in res.stints]
    starts = [s.start_s for s in res.stints]
    assert starts == sorted(starts)
    for prev_end, next_start in zip(ends, starts[1:]):
        assert next_start >= prev_end


def test_bench_technical_foul_is_not_floor_presence_evidence():
    """Real bug found by auditing a 400-game backfill: a bench player can
    be charged a technical foul (for arguing from the bench -- confirmed
    on a real game, Matisse Thybulle, box-score minutes 0) without ever
    playing. Unlike every other action type (shots, rebounds, live fouls),
    a Technical foul does not prove floor presence -- without excluding
    it, that player gets wrongly inferred as a period opener and credited
    a full period of phantom minutes. A player who genuinely IS on the
    floor when hit with a technical must still be correctly identified as
    present via his OTHER actions in the period."""
    import polars as pl

    from nbare.rapm.synthetic import PBP_SCHEMA

    def ev(event_num, clock_s, event_type, action_type=None, p1=None, p2=None):
        return {
            "game_id": "g1", "event_num": event_num, "period": 1,
            "clock_seconds_left": 720.0 - clock_s, "event_type": event_type,
            "event_action_type": action_type, "description": "test",
            "team_id": None, "player1_id": p1, "player2_id": p2,
            "player3_id": None, "home_score": 0, "away_score": 0,
        }

    events = [
        # Opener A=1, established by a genuine live-play action.
        ev(1, 5, "Made Shot", p1=1),
        # Bench player 99 is charged a technical -- NOT floor presence.
        ev(2, 10, "Foul", action_type="Technical", p1=99),
        # A genuinely-playing player (2) picks up a technical too, but is
        # ALSO established on the floor via a live-play action.
        ev(3, 15, "Made Shot", p1=2),
        ev(4, 20, "Foul", action_type="Technical", p1=2),
        # Sub near the end so player 99 (never subbed in) has zero seconds
        # if correctly excluded, rather than a phantom near-full period.
        ev(5, 700, "Substitution", p1=1, p2=3),
    ]
    pbp = pl.DataFrame(events, schema=PBP_SCHEMA)

    res = reconstruct_game(pbp)

    assert 99 not in res.player_seconds
    # Player 2 (genuinely playing, technical is incidental) keeps his full
    # floor time -- the exclusion must not remove real presence evidence.
    assert res.player_seconds[2] == pytest.approx(720.0)