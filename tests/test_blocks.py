"""Tests for the stints -> offense-blocks connector.

Strategy mirrors the rest of Stage 2: generate a game with KNOWN planted
scoring (make_scoring_game records exact per-segment points/possessions),
run the connector, and assert it recovers the plant exactly. Because the
connector is what turns real reconstructed stints into design-matrix rows,
recovering planted scoring proves the offense/defense attribution and the
team-split are correct before any real data is involved.
"""

from __future__ import annotations

import pytest

from nbare.rapm.blocks import (
    blocks_for_game,
    player_team_map,
    split_lineup,
)
from nbare.rapm.design import build_design
from nbare.rapm.fit import fit_rapm
from nbare.rapm.stints import reconstruct_game, validate_minutes
from nbare.rapm.synthetic import make_scoring_game, scoring_box_frame


# --- lineup splitting ----------------------------------------------------

def test_split_lineup_into_two_fives():
    team_of = {**{p: 1 for p in range(1, 6)}, **{p: 2 for p in range(101, 106)}}
    lineup = frozenset({1, 2, 3, 4, 5, 101, 102, 103, 104, 105})
    ta, a, tb, b = split_lineup(lineup, team_of)
    assert {ta, tb} == {1, 2}
    assert len(a) == 5 and len(b) == 5


def test_split_rejects_unknown_player():
    team_of = {p: 1 for p in range(1, 6)}  # away players missing
    lineup = frozenset({1, 2, 3, 4, 5, 101, 102, 103, 104, 105})
    assert split_lineup(lineup, team_of) is None


def test_split_rejects_non_five_five():
    team_of = {**{p: 1 for p in range(1, 7)}, **{p: 2 for p in range(101, 105)}}
    lineup = frozenset({1, 2, 3, 4, 5, 6, 101, 102, 103, 104})  # 6-4
    assert split_lineup(lineup, team_of) is None


# --- connector recovers planted scoring ---------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 5, 9, 13])
def test_connector_recovers_planted_scoring(seed):
    game = make_scoring_game(seed=seed)
    recon = reconstruct_game(game.pbp)
    box = scoring_box_frame(game)
    res = blocks_for_game(game.pbp, recon.stints, player_team_map(box))

    assert res.skipped_stints == 0

    truth_home_pts = sum(t[2] for t in game.truth)
    truth_away_pts = sum(t[4] for t in game.truth)
    truth_home_poss = sum(t[3] for t in game.truth)
    truth_away_poss = sum(t[5] for t in game.truth)

    home_blocks = [b for b in res.blocks if all(p < 100 for p in b.offense)]
    away_blocks = [b for b in res.blocks if all(p >= 100 for p in b.offense)]

    assert sum(b.points for b in home_blocks) == truth_home_pts
    assert sum(b.points for b in away_blocks) == truth_away_pts
    assert sum(b.possessions for b in home_blocks) == truth_home_poss
    assert sum(b.possessions for b in away_blocks) == truth_away_poss


def test_connector_output_is_correct_shape():
    """Every block is a 5-on-5 with disjoint lineups."""
    game = make_scoring_game(seed=4)
    recon = reconstruct_game(game.pbp)
    box = scoring_box_frame(game)
    res = blocks_for_game(game.pbp, recon.stints, player_team_map(box))
    for b in res.blocks:
        assert len(b.offense) == 5
        assert len(b.defense) == 5
        assert b.offense.isdisjoint(b.defense)


def test_minutes_still_pass_on_scoring_game():
    game = make_scoring_game(seed=7)
    recon = reconstruct_game(game.pbp)
    chk = validate_minutes(recon, scoring_box_frame(game))
    assert chk.passed, chk.summary()


# --- full chain ----------------------------------------------------------

def test_full_chain_connector_to_fit():
    """Connector output must flow straight into the design matrix and fit."""
    all_blocks = []
    for g in range(50):
        game = make_scoring_game(game_id=f"sg{g}", seed=g)
        recon = reconstruct_game(game.pbp)
        box = scoring_box_frame(game)
        res = blocks_for_game(game.pbp, recon.stints, player_team_map(box))
        all_blocks.extend(res.blocks)

    design = build_design(all_blocks)
    assert design.X.nnz == 10 * design.X.shape[0]  # 5 offense + 5 defense
    result = fit_rapm(design, fixed_lambda=1000.0)
    # Every player who appears in a block gets both an offense and defense
    # rating; the exact count depends on how many bench players the random
    # substitutions actually used, so assert the structural invariant rather
    # than a magic number.
    assert result.n_players == design.n_players
    assert set(result.offense) == set(result.defense)
    assert result.n_players >= 10  # at least the ten starters


def test_player_team_map_from_box():
    game = make_scoring_game(seed=1)
    box = scoring_box_frame(game)
    team_of = player_team_map(box)
    assert team_of[1] == 1      # home player
    assert team_of[101] == 2    # away player