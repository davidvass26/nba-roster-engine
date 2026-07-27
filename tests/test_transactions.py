from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from nbare.ingest.transactions import (
    Amend,
    Sign,
    Trade,
    TradeLeg,
    Waive,
    apply_transactions,
    load_transactions,
)

SCHEMA = {
    "contract_id": pl.Utf8, "season": pl.Utf8, "bbref_slug": pl.Utf8,
    "player": pl.Utf8, "team_abbrev": pl.Utf8, "season_index": pl.Int16,
    "cap_hit": pl.Int64, "guaranteed": pl.Int64, "is_guaranteed": pl.Boolean,
    "option_type": pl.Utf8, "needs_review": pl.Boolean, "review_reason": pl.Utf8,
}


def _base():
    rows = [
        dict(contract_id="c1", season="2026-27", bbref_slug="aaa", player="Player A",
             team_abbrev="LAL", season_index=1, cap_hit=30_000_000, guaranteed=30_000_000,
             is_guaranteed=True, option_type=None, needs_review=False, review_reason=None),
        dict(contract_id="c2", season="2026-27", bbref_slug="bbb", player="Player B",
             team_abbrev="MEM", season_index=1, cap_hit=5_000_000, guaranteed=0,
             is_guaranteed=False, option_type=None, needs_review=False, review_reason=None),
        dict(contract_id="c3", season="2026-27", bbref_slug="ccc", player="Player C",
             team_abbrev="MEM", season_index=1, cap_hit=8_000_000, guaranteed=8_000_000,
             is_guaranteed=True, option_type=None, needs_review=False, review_reason=None),
    ]
    return pl.DataFrame(rows, schema=SCHEMA)


def test_sign_adds_player():
    res = apply_transactions(
        _base(),
        [Sign(player="New Guy", slug="new", team="LAL", cap_hit=2_400_000, guaranteed=2_400_000)],
    )
    lal = res.frame.filter(pl.col("team_abbrev") == "LAL")
    assert lal.height == 2
    assert "New Guy" in lal["player"].to_list()


def test_waive_removes_nonguaranteed_cleanly():
    res = apply_transactions(_base(), [Waive(slug="bbb", team="MEM")])
    assert res.frame.filter(pl.col("bbref_slug") == "bbb").height == 0
    # no dead money left behind for a non-guaranteed deal we didn't stretch
    assert res.frame.filter(pl.col("review_reason") == "waived_dead_money").height == 0


def test_waive_guaranteed_with_stretch_leaves_dead_money():
    res = apply_transactions(
        _base(), [Waive(slug="ccc", team="MEM", stretch_dead_money=True)]
    )
    dead = res.frame.filter(pl.col("review_reason") == "waived_dead_money")
    assert dead.height == 1
    assert dead["cap_hit"][0] == 8_000_000  # guaranteed amount stays


def test_waive_wrong_team_warns_but_applies():
    res = apply_transactions(_base(), [Waive(slug="bbb", team="LAL")])  # bbb is on MEM
    assert any("not LAL" in w or "on MEM" in w for w in res.warnings)
    assert res.frame.filter(pl.col("bbref_slug") == "bbb").height == 0


def test_trade_moves_team():
    res = apply_transactions(
        _base(),
        [Trade(legs=(TradeLeg(slug="aaa", to_team="MEM"), TradeLeg(slug="ccc", to_team="LAL")))],
    )
    assert res.frame.filter(pl.col("bbref_slug") == "aaa")["team_abbrev"][0] == "MEM"
    assert res.frame.filter(pl.col("bbref_slug") == "ccc")["team_abbrev"][0] == "LAL"


def test_amend_changes_one_field():
    res = apply_transactions(
        _base(), [Amend(slug="aaa", field="cap_hit", value=99)]
    )
    assert res.frame.filter(pl.col("bbref_slug") == "aaa")["cap_hit"][0] == 99


def test_amend_missing_player_warns():
    res = apply_transactions(_base(), [Amend(slug="zzz", field="cap_hit", value=1)])
    assert any("not found" in w for w in res.warnings)


def test_base_frame_is_not_mutated():
    base = _base()
    apply_transactions(base, [Waive(slug="aaa", team="LAL")])
    assert base.filter(pl.col("bbref_slug") == "aaa").height == 1  # still there


def test_transactions_apply_in_order():
    """Waive then re-sign the same slug: order matters, end state is the
    re-signed deal."""
    res = apply_transactions(
        _base(),
        [
            Waive(slug="bbb", team="MEM"),
            Sign(player="Player B", slug="bbb", team="LAL", cap_hit=1_000_000, guaranteed=1_000_000),
        ],
    )
    b = res.frame.filter(pl.col("bbref_slug") == "bbb")
    assert b.height == 1
    assert b["team_abbrev"][0] == "LAL"
    assert b["cap_hit"][0] == 1_000_000


def test_sign_existing_player_replaces_and_warns():
    res = apply_transactions(
        _base(),
        [Sign(player="Player A", slug="aaa", team="MEM", cap_hit=1, guaranteed=1)],
    )
    a = res.frame.filter(pl.col("bbref_slug") == "aaa")
    assert a.height == 1  # not duplicated
    assert a["team_abbrev"][0] == "MEM"
    assert any("already present" in w for w in res.warnings)


# --- YAML loading --------------------------------------------------------

def test_load_transactions_from_yaml(tmp_path):
    p = tmp_path / "ov.yaml"
    p.write_text(
        """
- sign:
    player: LeBron James
    slug: jamesle01
    team: PHI
    cap_hit: 54126380
    guaranteed: 54126380
    as_of: 2026-07-25
- waive:
    slug: caldwke01
    team: MEM
- trade:
    legs:
      - {slug: aaa, to_team: NYK}
      - {slug: bbb, to_team: LAL}
"""
    )
    txns = load_transactions(p)
    assert len(txns) == 3
    assert isinstance(txns[0], Sign)
    assert txns[0].as_of == date(2026, 7, 25)
    assert isinstance(txns[1], Waive)
    assert isinstance(txns[2], Trade)
    assert txns[2].legs[0].to_team == "NYK"


def test_malformed_transaction_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- sign: {player: X}\n  waive: {slug: y}\n")  # two keys
    with pytest.raises(ValueError, match="single-key"):
        load_transactions(p)


def test_unknown_kind_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- teleport:\n    slug: aaa\n")
    with pytest.raises(ValueError, match="unknown kind"):
        load_transactions(p)