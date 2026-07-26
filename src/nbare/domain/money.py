"""Exact money arithmetic for cap calculations.

NBA salaries are whole dollars. Cap rules involve exact thresholds
(e.g. "125% of outgoing salary plus $100,000") where being off by a
dollar changes legality. Floating point is therefore banned in this
codebase: every dollar amount is an int, and every percentage
calculation goes through the helpers below with an explicit rounding
rule.

The CBA rounds to the nearest dollar. Python's round() uses banker's
rounding (round-half-to-even), which is NOT what we want, so
`pct_of` uses Decimal with ROUND_HALF_UP.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Final

# --- 2026-27 league-wide figures (official, NBA press release) -----------
# Source: nba.com/news/nba-salary-cap-2026-27-season
SALARY_CAP_2026_27: Final[int] = 164_961_000
MIN_TEAM_SALARY_2026_27: Final[int] = 148_465_000
TAX_LEVEL_2026_27: Final[int] = 200_428_000
FIRST_APRON_2026_27: Final[int] = 209_015_000
SECOND_APRON_2026_27: Final[int] = 221_686_000

NON_TAXPAYER_MLE_2026_27: Final[int] = 15_044_000
TAXPAYER_MLE_2026_27: Final[int] = 6_064_000
ROOM_MLE_2026_27: Final[int] = 9_366_000

# Amount of excess salary a team over both aprons may take back in a
# simultaneous trade (the "expanded traded player exception" figure).
EXPANDED_TPE_2026_27: Final[int] = 9_096_000

ROOKIE_MIN_2026_27: Final[int] = 1_357_763
TWO_WAY_SALARY_2026_27: Final[int] = 678_882


class Money(int):
    """A dollar amount. Subclasses int so it composes with normal math,
    but formats readably and refuses float construction."""

    __slots__ = ()

    def __new__(cls, value: int | str) -> "Money":
        if isinstance(value, float):
            raise TypeError(
                "Refusing to build Money from float; cap math must be exact. "
                "Pass an int number of dollars."
            )
        if isinstance(value, str):
            value = int(value.replace("$", "").replace(",", "").strip())
        return super().__new__(cls, value)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"${int(self):,}"

    __str__ = __repr__

    @property
    def millions(self) -> str:
        return f"${int(self) / 1_000_000:.3f}M"


def pct_of(amount: int, pct: str | Decimal) -> int:
    """Exact percentage of a dollar amount, rounded half-up to the dollar.

    `pct` is given as a string like "125" or "110" to avoid any float
    ever entering the calculation.

    >>> pct_of(10_000_000, "125")
    12500000
    >>> pct_of(9_999_999, "110")
    11000000
    """
    result = (Decimal(amount) * Decimal(pct) / Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return int(result)


def parse_dollars(raw: str) -> int:
    """Parse '$12,345,678' / '12345678' / '$1.35M' into int dollars."""
    s = raw.strip().replace("$", "").replace(",", "")
    if not s:
        raise ValueError("empty dollar string")
    if s[-1].upper() == "M":
        # Only used for human-entered config; convert via Decimal, not float.
        return int(
            (Decimal(s[:-1]) * Decimal(1_000_000)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
    return int(Decimal(s).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
