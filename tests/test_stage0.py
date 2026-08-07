from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from nbare.config import league_year
from nbare.crosswalk.build import match_one, normalize_name
from nbare.domain.money import Money, parse_dollars, pct_of
from nbare.ingest.client import NBAStatsCache, _looks_empty, request_hash
from nbare.ingest.nba_stats import _upsert, parse_clock, result_set_to_df
from nbare.warehouse.db import connect, table_counts


# --- money ---------------------------------------------------------------

def test_money_rejects_float():
    with pytest.raises(TypeError):
        Money(12_345_678.0)


def test_money_parses_formatted_strings():
    assert Money("$12,345,678") == 12_345_678


def test_pct_of_is_exact_and_rounds_half_up():
    # 125% + $100k matching on a $10M outgoing salary
    assert pct_of(10_000_000, "125") + 100_000 == 12_600_000
    # half-up, not banker's rounding
    assert pct_of(1, "250") == 3          # 2.5 -> 3, not 2
    assert pct_of(3, "150") == 5          # 4.5 -> 5, not 4


def test_pct_of_never_uses_float():
    # 110% of a value chosen to expose binary float error
    assert pct_of(20_000_003, "110") == 22_000_003


def test_parse_dollars_variants():
    assert parse_dollars("$1.35M") == 1_350_000
    assert parse_dollars("221,686,000") == 221_686_000


# --- league year ---------------------------------------------------------

def test_apron_bands_2026_27():
    ly = league_year("2026-27")
    assert ly.apron_status(150_000_000) == "under_cap"
    assert ly.apron_status(170_000_000) == "over_cap"
    assert ly.apron_status(205_000_000) == "taxpayer"
    assert ly.apron_status(215_000_000) == "between_aprons"
    assert ly.apron_status(230_000_000) == "over_second_apron"


def test_apron_thresholds_are_boundary_inclusive():
    ly = league_year("2026-27")
    # Being exactly AT the apron counts as over it for restriction purposes.
    assert ly.apron_status(ly.second_apron) == "over_second_apron"
    assert ly.apron_status(ly.second_apron - 1) == "between_aprons"


def test_rule_code_is_league_year_relative():
    """Backtesting requires prior years to be swappable."""
    assert league_year("2025-26").first_apron != league_year("2026-27").first_apron


# --- warehouse -----------------------------------------------------------

@pytest.fixture()
def con(tmp_path):
    c = connect(tmp_path / "test.duckdb")
    yield c
    c.close()


def test_schema_applies_and_is_idempotent(con, tmp_path):
    from nbare.warehouse.db import apply_schema

    apply_schema(con)  # second application must not raise
    counts = table_counts(con)
    assert "stg.pbp_event" in counts
    assert "mart.stint" in counts
    assert "stg.contract_year" in counts
    assert all(v == 0 for v in counts.values())


def test_upsert_is_idempotent(con):
    df = pl.DataFrame(
        {
            "nba_team_id": [1, 2],
            "abbreviation": ["BOS", "LAL"],
            "nickname": ["Celtics", "Lakers"],
            "city": ["Boston", "Los Angeles"],
            "full_name": ["Boston Celtics", "Los Angeles Lakers"],
        }
    )
    _upsert(con, "stg.team", df, key="nba_team_id")
    _upsert(con, "stg.team", df, key="nba_team_id")
    assert con.execute("SELECT count(*) FROM stg.team").fetchone()[0] == 2


def test_upsert_updates_existing_row(con):
    base = pl.DataFrame(
        {
            "nba_team_id": [1],
            "abbreviation": ["BOS"],
            "nickname": ["Celtics"],
            "city": ["Boston"],
            "full_name": ["Boston Celtics"],
        }
    )
    _upsert(con, "stg.team", base, key="nba_team_id")
    updated = base.with_columns(pl.lit("Bahstan").alias("city"))
    _upsert(con, "stg.team", updated, key="nba_team_id")
    row = con.execute("SELECT city FROM stg.team WHERE nba_team_id = 1").fetchone()
    assert row[0] == "Bahstan"


def test_composite_key_upsert(con):
    df = pl.DataFrame(
        {
            "game_id": ["0022500001", "0022500001"],
            "event_num": [1, 2],
            "period": [1, 1],
            "clock_seconds_left": [720.0, 715.0],
            "event_type": ["period", "shot"],
            "event_action_type": [None, "Jump Shot"],
            "description": ["Start", "Made 2"],
            "team_id": [None, 1610612738],
            "player1_id": [None, 2544],
            "player2_id": [None, None],
            "player3_id": [None, None],
            "home_score": [0, 2],
            "away_score": [0, 0],
        }
    )
    _upsert(con, "stg.pbp_event", df, key=["game_id", "event_num"])
    _upsert(con, "stg.pbp_event", df, key=["game_id", "event_num"])
    assert con.execute("SELECT count(*) FROM stg.pbp_event").fetchone()[0] == 2


# --- client --------------------------------------------------------------

def test_request_hash_is_order_insensitive():
    a = request_hash("PlayByPlayV3", {"game_id": "1", "start_period": 0})
    b = request_hash("PlayByPlayV3", {"start_period": 0, "game_id": "1"})
    assert a == b


def test_request_hash_distinguishes_params():
    assert request_hash("X", {"g": "1"}) != request_hash("X", {"g": "2"})


def test_cache_roundtrip(tmp_path):
    cache = NBAStatsCache(tmp_path / "c")
    rec = {"endpoint": "X", "params": {}, "fetched_at": 1.0, "payload": {"a": 1}}
    cache.put("abc123", rec)
    assert cache.get("abc123") == rec
    assert len(cache) == 1


def test_cache_discards_corrupt_entry(tmp_path):
    cache = NBAStatsCache(tmp_path / "c")
    cache.put("deadbeef", {"payload": {}})
    p = cache._path("deadbeef")
    p.write_text("{not json")
    assert cache.get("deadbeef") is None
    assert not p.exists()


def test_empty_cache_is_truthy():
    """Regression: NBAStatsCache defines __len__, so without an explicit
    __bool__ an empty cache is falsy and `cache or default` silently
    discards the caller's cache -- meaning the FIRST call of every run
    wrote to the wrong directory. Caught in the Stage 0 dry run."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cache = NBAStatsCache(d)
        assert len(cache) == 0
        assert bool(cache) is True


def test_fetch_honours_caller_supplied_cache(tmp_path):
    """The caller's cache must be used even when it starts empty."""
    calls = {"n": 0}

    class FakeEndpoint:
        def __init__(self, **kwargs):
            calls["n"] += 1

        def get_dict(self):
            return {"resultSets": [{"headers": ["A"], "rowSet": [[1]]}]}

    cache = NBAStatsCache(tmp_path / "mine")
    from nbare.ingest.client import fetch

    first = fetch(FakeEndpoint, {"game_id": "1"}, cache=cache)
    second = fetch(FakeEndpoint, {"game_id": "1"}, cache=cache)

    assert first.from_cache is False
    assert second.from_cache is True
    assert calls["n"] == 1
    assert len(cache) == 1  # written to the caller's dir, not the default


def test_empty_detection():
    assert _looks_empty({"resultSets": [{"headers": ["A"], "rowSet": []}]})
    assert not _looks_empty({"resultSets": [{"headers": ["A"], "rowSet": [[1]]}]})
    assert _looks_empty({})


def test_empty_detection_v3_nested_shape_not_misflagged():
    """Regression: v3 endpoints (playbyplayv3, boxscoretraditionalv3, ...)
    have no 'resultSets' key at all. The old heuristic treated that as
    'empty' unconditionally, so fetch() retried and raised on every real
    v3 response even though the request succeeded -- caught building box
    score ingestion (docs/box_ingest_spec.md)."""
    assert not _looks_empty({"game": {"gameId": "1", "actions": [{"actionNumber": 1}]}})
    assert not _looks_empty({"boxScoreTraditional": {"gameId": "1", "homeTeam": {}}})
    assert _looks_empty({})
    assert _looks_empty({"game": {}})


# --- parsing -------------------------------------------------------------

def test_result_set_to_df_snake_cases_headers():
    payload = {
        "resultSets": [
            {
                "name": "CommonAllPlayers",
                "headers": ["PERSON_ID", "DISPLAY_FIRST_LAST", "fromYear"],
                "rowSet": [[2544, "LeBron James", "2003"]],
            }
        ]
    }
    df = result_set_to_df(payload)
    assert df.columns == ["person_id", "display_first_last", "from_year"]
    assert df.height == 1


def test_result_set_selects_by_name():
    payload = {
        "resultSets": [
            {"name": "A", "headers": ["X"], "rowSet": [[1]]},
            {"name": "B", "headers": ["Y"], "rowSet": [[2]]},
        ]
    }
    assert result_set_to_df(payload, "B").columns == ["y"]
    with pytest.raises(KeyError):
        result_set_to_df(payload, "C")


@pytest.mark.parametrize(
    "clock,expected",
    [
        ("PT11M23.00S", 683.0),
        ("PT00M04.30S", 4.3),
        ("PT12M00.00S", 720.0),
        ("11:23", 683.0),
        (None, None),
    ],
)
def test_parse_clock(clock, expected):
    assert parse_clock(clock, 1) == expected


# --- crosswalk -----------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Nikola Jokić", "nikola jokic"),
        ("Jaren Jackson Jr.", "jaren jackson"),
        ("Alperen Şengün", "alperen sengun"),
        ("Karl-Anthony Towns", "karl anthony towns"),
        ("P.J. Tucker", "p j tucker"),
        ("Marvin Bagley III", "marvin bagley"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_match_one_exact_beats_fuzzy():
    cands = {"lebron james": "jamesle01", "lebron jamez": "jamezle01"}
    slug, score = match_one("lebron james", cands)
    assert slug == "jamesle01"
    assert score == 1.0


def test_match_one_rejects_distant_names():
    slug, score = match_one("stephen curry", {"nikola jokic": "jokicni01"})
    assert slug is None


def test_diacritic_names_match_after_normalization():
    cands = {normalize_name("Nikola Jokić"): "jokicni01"}
    slug, _ = match_one(normalize_name("Nikola Jokic"), cands)
    assert slug == "jokicni01"
