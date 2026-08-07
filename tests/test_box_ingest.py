"""Tests for box-score ingestion (docs/box_ingest_spec.md).

The seconds parser is the critical field: `validate_minutes` compares
reconstructed stint-time against it, so a silent misparse here makes the
whole minutes gate meaningless. Per CLAUDE.md's synthetic-first rule, this
is validated against the synthetic scoring-game generator's KNOWN seconds
before any of this code touches nba.com.
"""

from __future__ import annotations

import polars as pl
import pytest

from nbare.ingest.nba_stats import (
    _parse_box_score_payload,
    _upsert,
    parse_minutes_to_seconds,
)
from nbare.rapm.synthetic import (
    AWAY_TEAM_ID,
    HOME_TEAM_ID,
    make_scoring_game,
    scoring_box_frame,
)
from nbare.warehouse.db import connect


# --- seconds parser --------------------------------------------------------

@pytest.mark.parametrize(
    "minutes,expected",
    [
        ("34:12", 2052),
        ("0:00", 0),
        ("48:00", 2880),
    ],
)
def test_parse_minutes_matches_spec_examples(minutes, expected):
    assert parse_minutes_to_seconds(minutes) == expected


@pytest.mark.parametrize(
    "minutes,expected",
    [
        ("PT34M12.00S", 2052),
        ("PT0M0.00S", 0),
        ("PT48M00.00S", 2880),
        ("PT9M4.30S", 544),  # 544.3 rounds to 544
    ],
)
def test_parse_minutes_handles_iso8601_duration(minutes, expected):
    assert parse_minutes_to_seconds(minutes) == expected


def test_parse_minutes_none_is_unknown_not_zero():
    """A missing field is honestly unknown -- distinct from an explicit
    empty string, which is nba.com's own convention for a DNP player."""
    assert parse_minutes_to_seconds(None) is None


def test_parse_minutes_empty_string_is_dnp_zero():
    assert parse_minutes_to_seconds("") == 0
    assert parse_minutes_to_seconds("   ") == 0


def test_parse_minutes_rejects_unrecognized_format():
    # Honest failure, not a guess -- per CLAUDE.md principle #1.
    assert parse_minutes_to_seconds("garbage") is None


def test_parse_minutes_round_trips_synthetic_ground_truth():
    """The synthetic scoring-game generator produces KNOWN seconds_played
    per player. Format them the way nba.com does (MM:SS) and confirm the
    parser recovers the exact original integer."""
    game = make_scoring_game(seed=5)
    box = scoring_box_frame(game)
    assert box.height > 0
    for row in box.iter_rows(named=True):
        secs = row["seconds_played"]
        mm, ss = divmod(secs, 60)
        formatted = f"{mm}:{ss:02d}"
        assert parse_minutes_to_seconds(formatted) == secs


# --- box score payload parser ----------------------------------------------
#
# Shape confirmed by reading nba_api's own
# NBAStatsBoxscoreTraditionalParserV3 (nba_api/stats/endpoints/_parsers/
# boxscoretraditionalv3.py) -- the raw payload our fetch() wrapper receives
# is {"boxScoreTraditional": {"homeTeam": {...}, "awayTeam": {...}}}, NOT
# the classic resultSets/rowSet shape result_set_to_df expects.

FAKE_BOX_PAYLOAD = {
    "boxScoreTraditional": {
        "gameId": "0022500001",
        "homeTeamId": 1610612738,
        "awayTeamId": 1610612747,
        "homeTeam": {
            "teamId": 1610612738,
            "players": [
                {
                    "personId": 2544, "firstName": "LeBron", "familyName": "James",
                    "position": "F", "comment": "",
                    "statistics": {"minutes": "34:12", "points": 28,
                                   "reboundsTotal": 8, "assists": 9},
                },
                {
                    "personId": 201939, "firstName": "Steph", "familyName": "Curry",
                    "position": "", "comment": "",
                    "statistics": {"minutes": "18:45", "points": 10,
                                   "reboundsTotal": 2, "assists": 3},
                },
                {
                    "personId": 999999, "firstName": "Bench", "familyName": "Warmer",
                    "position": "", "comment": "DNP - Coach's Decision",
                    "statistics": {"minutes": "", "points": 0,
                                   "reboundsTotal": 0, "assists": 0},
                },
            ],
        },
        "awayTeam": {
            "teamId": 1610612747,
            "players": [
                {
                    "personId": 101108, "firstName": "Klay", "familyName": "Thompson",
                    "position": "G", "comment": "",
                    "statistics": {"minutes": "PT30M15.00S", "points": 20,
                                   "reboundsTotal": 3, "assists": 2},
                },
            ],
        },
    },
}


def test_parse_box_score_payload_shape():
    df = _parse_box_score_payload(FAKE_BOX_PAYLOAD, "0022500001")
    assert df.columns == [
        "game_id", "nba_player_id", "nba_team_id", "seconds_played",
        "pts", "reb", "ast", "started",
    ]
    assert df.height == 4


def test_parse_box_score_payload_seconds_and_team():
    df = _parse_box_score_payload(FAKE_BOX_PAYLOAD, "0022500001")
    lebron = df.filter(pl.col("nba_player_id") == 2544).to_dicts()[0]
    assert lebron["seconds_played"] == 2052
    assert lebron["nba_team_id"] == 1610612738

    klay = df.filter(pl.col("nba_player_id") == 101108).to_dicts()[0]
    assert klay["seconds_played"] == 1815  # PT30M15.00S
    assert klay["nba_team_id"] == 1610612747


def test_parse_box_score_payload_dnp_is_zero_seconds():
    df = _parse_box_score_payload(FAKE_BOX_PAYLOAD, "0022500001")
    bench = df.filter(pl.col("nba_player_id") == 999999).to_dicts()[0]
    assert bench["seconds_played"] == 0


def test_parse_box_score_payload_started_from_position():
    df = _parse_box_score_payload(FAKE_BOX_PAYLOAD, "0022500001")
    lebron = df.filter(pl.col("nba_player_id") == 2544).to_dicts()[0]
    curry = df.filter(pl.col("nba_player_id") == 201939).to_dicts()[0]
    assert lebron["started"] is True   # position "F"
    assert curry["started"] is False   # position ""


def test_parse_box_score_payload_missing_key_returns_empty():
    assert _parse_box_score_payload({}, "0022500001").is_empty()
    assert _parse_box_score_payload({"unrelated": 1}, "0022500001").is_empty()


# --- parse-and-upsert round trip (spec's explicit validation ask) ---------

def _synthetic_box_score_payload(game) -> dict:
    """Build a fake nba.com-shaped payload from a synthetic ScoringGame,
    so the real parser can be exercised against KNOWN seconds_played."""
    home_players, away_players = [], []
    for pid, secs in game.box_seconds.items():
        mm, ss = divmod(int(round(secs)), 60)
        entry = {
            "personId": pid,
            "position": "F",
            "statistics": {
                "minutes": f"{mm}:{ss:02d}",
                "points": 0, "reboundsTotal": 0, "assists": 0,
            },
        }
        (home_players if pid < 100 else away_players).append(entry)
    return {
        "boxScoreTraditional": {
            "gameId": game.game_id,
            "homeTeam": {"teamId": HOME_TEAM_ID, "players": home_players},
            "awayTeam": {"teamId": AWAY_TEAM_ID, "players": away_players},
        },
    }


@pytest.fixture()
def con(tmp_path):
    c = connect(tmp_path / "test.duckdb")
    yield c
    c.close()


def test_parse_and_upsert_round_trips_seconds(con):
    """The spec's explicit ask: the synthetic scoring-game generator's KNOWN
    seconds_played must survive payload -> parse -> upsert -> warehouse
    exactly."""
    game = make_scoring_game(seed=6)
    payload = _synthetic_box_score_payload(game)
    df = _parse_box_score_payload(payload, game.game_id)
    assert df.height == len(game.box_seconds)

    _upsert(con, "stg.box_player", df, key=["game_id", "nba_player_id"])

    rows = con.execute(
        "SELECT nba_player_id, seconds_played FROM stg.box_player WHERE game_id = ?",
        [game.game_id],
    ).fetchall()
    got = {pid: secs for pid, secs in rows}
    for pid, secs in game.box_seconds.items():
        assert got[pid] == int(round(secs))


def test_parse_and_upsert_is_idempotent(con):
    game = make_scoring_game(seed=7)
    payload = _synthetic_box_score_payload(game)
    df = _parse_box_score_payload(payload, game.game_id)

    _upsert(con, "stg.box_player", df, key=["game_id", "nba_player_id"])
    _upsert(con, "stg.box_player", df, key=["game_id", "nba_player_id"])

    n = con.execute(
        "SELECT count(*) FROM stg.box_player WHERE game_id = ?", [game.game_id]
    ).fetchone()[0]
    assert n == len(game.box_seconds)
