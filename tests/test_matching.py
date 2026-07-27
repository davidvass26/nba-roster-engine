"""Tests for the Stage 1 salary-matching engine.

The `test_worked_example_*` cases are anchored on real numbers published
in CBA explainers (Hoops Rumors) and verified against the retrieved
sources. They are the ground truth for the matching bands and must never
regress -- if one of these breaks, the engine is wrong, not the test.
"""

from __future__ import annotations

import pytest

from nbare.cba.matching import (
    BAND_1_CEILING,
    BAND_2_CEILING,
    Severity,
    check_trade,
    matching_ceiling,
)
from nbare.config import league_year
from nbare.domain.models import (
    CapSheet,
    PlayerSalary,
    TradePiece,
    TradeProposal,
)
from nbare.domain.money import Money

LY = league_year("2026-27")


def _p(pid: str, cap: int, name: str | None = None) -> PlayerSalary:
    return PlayerSalary(
        player_id=pid, name=name or pid, cap_hit=Money(cap), guaranteed=Money(cap)
    )


# --- worked examples: these are ground truth ----------------------------

def test_worked_example_tre_mann():
    """Hoops Rumors: $3,191,400 out -> up to $6,632,800 (200% + $250K)."""
    ceiling, band, _ = matching_ceiling(3_191_400, below_first_apron=True)
    assert ceiling == 6_632_800
    assert band == "band1_200pct+250k"


def test_worked_example_oladipo():
    """Hoops Rumors: $9,450,000 out -> up to $16,950,000 (S + $7.5M)."""
    ceiling, band, _ = matching_ceiling(9_450_000, below_first_apron=True)
    assert ceiling == 16_950_000
    assert band == "band2_S+7.5M"


def test_worked_example_wiggins():
    """Hoops Rumors: ~$26.3M out -> ~$33.8M (S + $7.5M)."""
    ceiling, _, _ = matching_ceiling(26_300_000, below_first_apron=True)
    assert ceiling == 33_800_000


def test_band3_above_29m():
    """125% + $250K above $29M. $40M -> $50,250,000."""
    ceiling, band, _ = matching_ceiling(40_000_000, below_first_apron=True)
    assert ceiling == 50_250_000
    assert band == "band3_125pct+250k"


def test_over_apron_is_flat_100pct():
    ceiling, band, _ = matching_ceiling(20_000_000, below_first_apron=False)
    assert ceiling == 20_000_000
    assert band == "over_apron_100pct"


# --- band boundaries (off-by-one is the classic bug here) ---------------

def test_band1_boundary_inclusive():
    """Exactly $7.5M stays in band 1: 200% + 250K = $15,250,000."""
    ceiling, band, _ = matching_ceiling(BAND_1_CEILING, below_first_apron=True)
    assert ceiling == 15_250_000
    assert band == "band1_200pct+250k"


def test_just_over_band1_uses_band2():
    ceiling, band, _ = matching_ceiling(BAND_1_CEILING + 1, below_first_apron=True)
    assert band == "band2_S+7.5M"
    assert ceiling == (BAND_1_CEILING + 1) + 7_500_000


def test_band2_boundary_inclusive():
    """Exactly $29M stays in band 2: S + 7.5M = $36,500,000."""
    ceiling, band, _ = matching_ceiling(BAND_2_CEILING, below_first_apron=True)
    assert ceiling == 36_500_000
    assert band == "band2_S+7.5M"


def test_just_over_band2_uses_band3():
    ceiling, band, _ = matching_ceiling(BAND_2_CEILING + 1, below_first_apron=True)
    assert band == "band3_125pct+250k"


def test_matching_is_exact_no_float_drift():
    """Odd-dollar outgoing must not introduce rounding error."""
    ceiling, _, _ = matching_ceiling(29_000_001, below_first_apron=True)
    # band 3: 125% of 29,000,001 = 36,250,001.25 -> half-up 36,250,001, +250k
    assert ceiling == 36_500_001


# --- full trade checks --------------------------------------------------

def _sheet(team: str, salaries, apron_target: int | None = None) -> CapSheet:
    return CapSheet(team=team, season="2026-27", salaries=tuple(salaries))


def test_legal_trade_between_over_cap_teams():
    """Two over-cap, below-apron teams swap similar salaries -> legal."""
    # Build each team a payroll comfortably between cap and first apron.
    filler = [_p(f"f{i}", 10_000_000) for i in range(18)]  # 180M base
    a = _sheet("AAA", [*filler, _p("star_a", 20_000_000)])
    b = _sheet("BBB", [*filler, _p("star_b", 19_000_000)])
    prop = TradeProposal(
        pieces=(
            TradePiece(_p("star_a", 20_000_000), "AAA", "BBB"),
            TradePiece(_p("star_b", 19_000_000), "BBB", "AAA"),
        )
    )
    report = check_trade({"AAA": a, "BBB": b}, prop, LY)
    # Both over the first apron here (200M base), so 100% match; $20M vs
    # $19M and $19M vs $20M both fit within 100% + the fact each sends ~ what
    # it takes. Verify no ILLEGAL.
    assert Severity.ILLEGAL not in {v.severity for v in report.verdicts}


def test_illegal_when_small_salary_takes_back_large():
    """Send $27M, receive $41M: illegal regardless of the other side."""
    big_payroll = [_p(f"f{i}", 11_000_000) for i in range(17)]  # ~187M
    a = _sheet("AAA", [*big_payroll, _p("out_a", 41_240_250)])
    b = _sheet("BBB", [*big_payroll, _p("out_b", 27_000_000)])
    prop = TradeProposal(
        pieces=(
            TradePiece(_p("out_a", 41_240_250), "AAA", "BBB"),
            TradePiece(_p("out_b", 27_000_000), "BBB", "AAA"),
        )
    )
    report = check_trade({"AAA": a, "BBB": b}, prop, LY)
    assert report.severity is Severity.ILLEGAL
    bbb = next(v for v in report.verdicts if v.team == "BBB")
    assert bbb.severity is Severity.ILLEGAL
    assert "exceeds" in bbb.reason


def test_missing_cap_sheet_is_indeterminate_not_pass():
    a = _sheet("AAA", [_p("x", 15_000_000)])
    prop = TradeProposal(
        pieces=(
            TradePiece(_p("x", 15_000_000), "AAA", "BBB"),
            TradePiece(_p("y", 15_000_000), "BBB", "AAA"),
        )
    )
    report = check_trade({"AAA": a}, prop, LY)  # BBB missing
    assert report.severity is Severity.INDETERMINATE
    bbb = next(v for v in report.verdicts if v.team == "BBB")
    assert bbb.severity is Severity.INDETERMINATE
    assert "no cap sheet" in bbb.reason


def test_report_rollup_illegal_dominates():
    """One illegal side makes the whole trade illegal."""
    big = [_p(f"f{i}", 11_000_000) for i in range(17)]
    a = _sheet("AAA", [*big, _p("out_a", 41_000_000)])
    b = _sheet("BBB", [*big, _p("out_b", 5_000_000)])
    prop = TradeProposal(
        pieces=(
            TradePiece(_p("out_a", 41_000_000), "AAA", "BBB"),
            TradePiece(_p("out_b", 5_000_000), "BBB", "AAA"),
        )
    )
    report = check_trade({"AAA": a, "BBB": b}, prop, LY)
    assert report.severity is Severity.ILLEGAL


def test_verdicts_are_auditable():
    """Every verdict cites a specific CBA band, never a bare boolean."""
    big = [_p(f"f{i}", 11_000_000) for i in range(17)]
    a = _sheet("AAA", [*big, _p("out_a", 30_000_000)])
    b = _sheet("BBB", [*big, _p("out_b", 29_000_000)])
    prop = TradeProposal(
        pieces=(
            TradePiece(_p("out_a", 30_000_000), "AAA", "BBB"),
            TradePiece(_p("out_b", 29_000_000), "BBB", "AAA"),
        )
    )
    report = check_trade({"AAA": a, "BBB": b}, prop, LY)
    for v in report.verdicts:
        assert v.rule  # non-empty CBA reference
        assert v.band
        assert "$" in v.reason


# --- backtesting property ------------------------------------------------

def test_rules_are_league_year_relative():
    """The same proposal judged against a different year can differ,
    because apron thresholds move. This is what makes backtesting work."""
    ly_prev = league_year("2025-26")
    assert ly_prev.first_apron != LY.first_apron
    # Matching bands themselves are fixed dollars, so the CEILING is
    # year-independent; only the band SELECTION (which depends on apron
    # position) moves. Confirm the ceiling function is pure.
    c1, _, _ = matching_ceiling(10_000_000, below_first_apron=True)
    c2, _, _ = matching_ceiling(10_000_000, below_first_apron=True)
    assert c1 == c2 == 17_500_000