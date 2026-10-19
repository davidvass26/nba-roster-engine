"""Tests for box-score feature aggregation (rapm/box_features.py) and the
cache-only box detail reader that feeds it.

Synthetic-first per CLAUDE.md: every test plants box scores whose per-100
rates, TS%, rebound rate, and position flag are computed by hand below, and
asserts the aggregation recovers them exactly. These are what the real-data
box-score prior is built from, so a silent formula slip here would bias
every prior mean.
"""

from __future__ import annotations

import polars as pl
import pytest

from nbare.ingest.client import NBAStatsCache, request_hash
from nbare.ingest.nba_stats import (
    BOX_DETAIL_SCHEMA,
    box_detail_from_cache,
    parse_box_score_detail,
)
from nbare.rapm.box_features import box_score_rows

HOME, AWAY = 1610612001, 1610612002
FULL = 2880


def _line(gid, pid, team, secs, position="", **stats):
    base = dict(pts=0, ast=0, tov=0, fga=0, fta=0, oreb=0, dreb=0, stl=0, blk=0)
    base.update(stats)
    return {"game_id": gid, "nba_player_id": pid, "nba_team_id": team,
            "seconds_played": secs, "position": position, **base}


def _game(gid, home_stats, away_stats, home_secs=None, home_pos=None):
    """Five identical full-game players per side unless overridden."""
    rows = []
    for k in range(5):
        rows.append(_line(gid, 100 + k, HOME,
                          (home_secs or [FULL] * 5)[k],
                          (home_pos or [""] * 5)[k], **home_stats))
        rows.append(_line(gid, 200 + k, AWAY, FULL, **away_stats))
    return rows


HOME_STATS = dict(pts=20, ast=4, tov=3, fga=16, fta=5, oreb=2, dreb=6, stl=1, blk=1)
AWAY_STATS = dict(pts=18, ast=3, tov=2, fga=18, fta=5, oreb=4, dreb=7, stl=2, blk=0)
# Home team: FGA 80, OREB 10, TOV 15, FTA 25 -> POSS = 80-10+15+11 = 96.
# Away team: FGA 90, OREB 20, TOV 10, FTA 25 -> POSS = 90-20+10+11 = 91.
HOME_POSS, AWAY_POSS = 96.0, 91.0


def _frame(rows):
    return pl.DataFrame(rows, schema=BOX_DETAIL_SCHEMA)


def test_recovers_planted_rates_exactly():
    res = box_score_rows(_frame(_game("g1", HOME_STATS, AWAY_STATS)),
                         min_box_possessions=0.0)
    row = {r.player_id: r for r in res.rows}[100]

    # Full game = share 1, so on-court possessions are the team totals.
    assert row.off_possessions == pytest.approx(HOME_POSS)
    assert row.def_possessions == pytest.approx(AWAY_POSS)
    assert row.pts_100 == pytest.approx(20 / HOME_POSS * 100)
    assert row.ast_100 == pytest.approx(4 / HOME_POSS * 100)
    assert row.tov_100 == pytest.approx(3 / HOME_POSS * 100)
    assert row.ts_pct == pytest.approx(20 / (2 * (16 + 0.44 * 5)))
    assert row.stl_100 == pytest.approx(1 / AWAY_POSS * 100)
    assert row.blk_100 == pytest.approx(1 / AWAY_POSS * 100)
    # DREB chances = team DREB (30) + opponent OREB (20) = 50.
    assert row.dreb_rate == pytest.approx(6 / 50)
    assert res.n_players_seen == 10


def test_partial_minutes_scale_possessions_by_floor_share():
    """A player on the floor for half the game gets half the team's
    possessions as his denominator (share = secs / (team_secs / 5))."""
    secs = [FULL, FULL, FULL, FULL // 2, FULL // 2]
    rows = _game("g1", HOME_STATS, AWAY_STATS, home_secs=secs)
    rows.append(_line("g1", 105, HOME, FULL, **HOME_STATS))  # 6th man
    res = box_score_rows(_frame(rows), min_box_possessions=0.0)
    by = {r.player_id: r for r in res.rows}

    # Team floor time is 5 full games; the half-game players have share 0.5.
    # Team totals now include the 6th man: FGA 96, OREB 12, TOV 18, FTA 30.
    team_poss = 96 - 12 + 18 + 0.44 * 30
    assert by[103].off_possessions == pytest.approx(0.5 * team_poss)
    assert by[103].pts_100 == pytest.approx(20 / (0.5 * team_poss) * 100)
    assert by[100].off_possessions == pytest.approx(team_poss)


def test_sums_over_games_before_dividing():
    """Two games, different paces: rates are total stats over total
    possessions, not an average of per-game rates."""
    g1 = _game("g1", HOME_STATS, AWAY_STATS)
    fast_home = dict(HOME_STATS, fga=20)  # team FGA 100 -> POSS 116
    g2 = _game("g2", fast_home, AWAY_STATS)
    res = box_score_rows(_frame(g1 + g2), min_box_possessions=0.0)
    row = {r.player_id: r for r in res.rows}[100]

    assert row.off_possessions == pytest.approx(HOME_POSS + 116.0)
    assert row.pts_100 == pytest.approx(40 / (HOME_POSS + 116.0) * 100)


def test_small_samples_get_no_row():
    """Below min_box_possessions a player gets no row, so build_prior leaves
    him at a zero prior mean instead of an extreme noisy one."""
    res = box_score_rows(_frame(_game("g1", HOME_STATS, AWAY_STATS)),
                         min_box_possessions=HOME_POSS + 1)
    # Home players have 96 off possessions (< 97) and are dropped; away
    # players have only 91 and are dropped too.
    assert res.rows == []
    assert res.n_below_min == 10


def test_position_big_needs_majority_of_labeled_starts_at_c():
    g1 = _game("g1", HOME_STATS, AWAY_STATS, home_pos=["C", "C", "F", "", ""])
    g2 = _game("g2", HOME_STATS, AWAY_STATS, home_pos=["C", "F", "F", "", ""])
    g3 = _game("g3", HOME_STATS, AWAY_STATS, home_pos=["C", "F", "F", "", ""])
    res = box_score_rows(_frame(g1 + g2 + g3), min_box_possessions=0.0)
    by = {r.player_id: r for r in res.rows}

    assert by[100].position_big == 1   # C in 3/3 labeled starts
    assert by[101].position_big == 0   # C in 1/3
    assert by[102].position_big == 0   # never C
    assert by[103].position_big == 0   # no labeled starts: unknown, reported
    # Unknown position is counted, not hidden: 2 home bench + 5 away.
    assert res.n_position_unknown == 7


def test_design_side_possessions_override_box_estimates():
    """Reliability (who trains the prior) uses the RAPM design's own
    possessions when supplied; rates stay box-internal."""
    res = box_score_rows(
        _frame(_game("g1", HOME_STATS, AWAY_STATS)),
        side_possessions=({100: 5000.0}, {100: 4900.0}),
        min_box_possessions=0.0,
    )
    row = {r.player_id: r for r in res.rows}[100]
    assert row.off_possessions == 5000.0
    assert row.def_possessions == 4900.0
    assert row.pts_100 == pytest.approx(20 / HOME_POSS * 100)


def test_empty_detail_is_empty_result():
    res = box_score_rows(pl.DataFrame(schema=BOX_DETAIL_SCHEMA))
    assert res.rows == [] and res.n_players_seen == 0


# --- payload parsing + cache-only reader ----------------------------------

def _payload():
    def p(pid, pos, **s):
        return {"personId": pid, "position": pos, "statistics": {
            "minutes": "PT30M00.00S", "points": s.get("pts", 10),
            "assists": 2, "turnovers": 1, "fieldGoalsAttempted": 9,
            "freeThrowsAttempted": 4, "reboundsOffensive": 1,
            "reboundsDefensive": 5, "steals": 1, "blocks": 2,
        }}
    return {"boxScoreTraditional": {
        "homeTeam": {"teamId": HOME, "players": [p(1, "C"), p(2, "")]},
        "awayTeam": {"teamId": AWAY, "players": [p(3, "G", pts=25)]},
    }}


def test_parse_box_score_detail_reads_every_feature_field():
    df = parse_box_score_detail(_payload(), "g1")
    assert df.height == 3
    r = df.filter(pl.col("nba_player_id") == 1).row(0, named=True)
    assert r == {
        "game_id": "g1", "nba_player_id": 1, "nba_team_id": HOME,
        "seconds_played": 1800, "position": "C", "pts": 10, "ast": 2,
        "tov": 1, "fga": 9, "fta": 4, "oreb": 1, "dreb": 5, "stl": 1, "blk": 2,
    }
    assert df.filter(pl.col("nba_player_id") == 2)["position"][0] == ""


def test_box_detail_from_cache_reports_missing_games(tmp_path):
    cache = NBAStatsCache(tmp_path)
    cache.put(request_hash("BoxScoreTraditionalV3", {"game_id": "g1"}),
              {"payload": _payload(), "fetched_at": 0.0})
    detail, missing = box_detail_from_cache(["g1", "g2"], cache=cache)
    assert detail.height == 3
    assert missing == ["g2"]


def test_fit_with_prior_rejects_unknown_prior_and_labels_zero_fit():
    """The fit-type label is returned by the fit that ran, so meta.json can
    never claim a prior the fit did not use."""
    import typer

    from nbare.cli import _fit_with_prior
    from nbare.rapm.design import OffenseBlock, build_design

    design = build_design([
        OffenseBlock("g1", frozenset({1}), frozenset({2}), 110.0, 100.0),
        OffenseBlock("g1", frozenset({2}), frozenset({1}), 100.0, 100.0),
        OffenseBlock("g2", frozenset({1}), frozenset({2}), 105.0, 100.0),
        OffenseBlock("g2", frozenset({2}), frozenset({1}), 108.0, 100.0),
    ])
    with pytest.raises(typer.BadParameter):
        _fit_with_prior(design, "guess", [], -2.5)
    _, label = _fit_with_prior(design, "zero", [], -2.5)
    assert "zero-mean" in label
