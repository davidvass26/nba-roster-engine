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