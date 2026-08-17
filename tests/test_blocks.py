"""Tests for the stints -> offense-blocks connector.

Strategy mirrors the rest of Stage 2: generate a game with KNOWN planted
scoring (make_scoring_game records exact per-segment points/possessions),
run the connector, and assert it recovers the plant exactly. Because the
connector is what turns real reconstructed stints into design-matrix rows,
recovering planted scoring proves the offense/defense attribution and the
team-split are correct before any real data is involved.
"""

from __future__ import annotations

import polars as pl
import pytest

from nbare.rapm.blocks import (
    blocks_for_game,
    blocks_from_warehouse,
    classify_gate_failure,
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


# --- gate-failure classification -----------------------------------------

def _box(rows):
    return pl.DataFrame(
        {
            "game_id": ["g1"] * len(rows),
            "nba_player_id": [r[0] for r in rows],
            "nba_team_id": [r[1] for r in rows],
            "seconds_played": [100] * len(rows),
            "pts": [0] * len(rows), "reb": [0] * len(rows), "ast": [0] * len(rows),
            "started": [False] * len(rows),
        }
    )


def test_classify_gate_failure_detects_surname_collision():
    """GG Jackson / Jaren Jackson Jr. is the real case found on 2025-26
    data (see the check-minutes 400-game audit): both appear in the box
    score for the same team, and the offender DOES have nonzero
    reconstructed seconds (so it is not a pure data gap). The Jr. suffix
    must not defeat the last-name match."""
    box = _box([(1, 10), (2, 10), (3, 20)])
    names = {1: "GG Jackson", 2: "Jaren Jackson Jr.", 3: "Someone Else"}
    offenders = [(1, 500.0, 300.0)]
    assert classify_gate_failure(offenders, box, names) == "surname-collision"


def test_classify_gate_failure_detects_data_gap():
    """A player with box minutes but reconstructed seconds == 0 never
    appeared in a single pbp event -- an nba.com data gap, not a bug."""
    box = _box([(1, 10)])
    names = {1: "Solo Player"}
    offenders = [(1, 0.0, 100.0)]
    assert classify_gate_failure(offenders, box, names) == "data-gap"


def test_classify_gate_failure_falls_back_to_isolated():
    """A partial mismatch with no teammate surname collision and nonzero
    reconstructed seconds is not yet diagnosed as systematic."""
    box = _box([(1, 10), (2, 20)])
    names = {1: "Unique Name", 2: "Different Person"}
    offenders = [(1, 50.0, 100.0)]
    assert classify_gate_failure(offenders, box, names) == "isolated"


# --- gate filtering in blocks_from_warehouse ------------------------------

def test_blocks_from_warehouse_excludes_gate_failing_games(tmp_path):
    """RAPM must only be fit on trustworthy stints (CLAUDE.md: only fit on
    what you can verify). A game that fails the minutes gate -- here, a
    data gap: box minutes with zero pbp events -- must be excluded from
    blocks_from_warehouse's output entirely, not partially trusted, and
    the exclusion must be recorded with its category."""
    from nbare.ingest.nba_stats import _upsert
    from nbare.warehouse.db import connect

    con = connect(tmp_path / "test.duckdb")

    good = make_scoring_game(game_id="g_good", seed=1)
    good_box = scoring_box_frame(good)

    bad = make_scoring_game(game_id="g_bad", seed=2)
    # Player 999 has box minutes but will never appear in g_bad's pbp --
    # the data-gap case.
    bad_box = scoring_box_frame(bad).vstack(
        pl.DataFrame(
            {
                "game_id": ["g_bad"], "nba_player_id": [999], "nba_team_id": [1],
                "seconds_played": [200], "pts": [0], "reb": [0], "ast": [0],
                "started": [False],
            }
        )
    )

    all_ids = sorted(set(good.box_seconds) | set(bad.box_seconds) | {999})
    player_df = pl.DataFrame(
        {"nba_player_id": all_ids, "full_name": [f"Player {i}" for i in all_ids]}
    )
    _upsert(con, "stg.player", player_df, key="nba_player_id")

    for pbp, box in [(good.pbp, good_box), (bad.pbp, bad_box)]:
        _upsert(con, "stg.pbp_event", pbp, key=["game_id", "event_num"])
        _upsert(con, "stg.box_player", box, key=["game_id", "nba_player_id"])

    result = blocks_from_warehouse(con, ["g_good", "g_bad"])
    con.close()

    assert result.games_total == 2
    assert result.games_included == 1
    assert result.games_excluded == 1
    assert result.exclusions[0].game_id == "g_bad"
    assert result.exclusions[0].category == "data-gap"
    assert result.exclusion_summary() == {
        "data-gap": 1, "surname-collision": 0, "isolated": 0,
    }
    # None of the excluded game's blocks leak into the trusted output.
    assert result.blocks
    assert all(b.game_id == "g_good" for b in result.blocks)