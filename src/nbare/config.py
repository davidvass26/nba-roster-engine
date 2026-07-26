"""Project configuration and league-year cap figures.

Cap numbers are *data*, not constants: every rule in the CBA engine is
expressed relative to a `LeagueYear`, so backtesting against the 2024-25
or 2025-26 offseason is a matter of swapping the league year rather than
editing rule code. This is the single most important structural decision
in Stage 1 -- hardcoding 2026-27 numbers into the rule engine would make
the historical validation suite impossible to write.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = Path(os.environ.get("NBARE_DATA", PROJECT_ROOT / "data"))
RAW_DIR: Final[Path] = DATA_DIR / "raw"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
DB_PATH: Final[Path] = Path(
    os.environ.get("NBARE_DB", DATA_DIR / "nbare.duckdb")
)

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class LeagueYear:
    """All cap-relevant thresholds for one league year."""

    season: str  # "2026-27"
    salary_cap: int
    min_team_salary: int
    tax_level: int
    first_apron: int
    second_apron: int
    non_taxpayer_mle: int
    taxpayer_mle: int
    room_mle: int
    bae: int
    expanded_tpe: int
    rookie_min: int
    two_way_salary: int
    max_roster: int = 15
    min_roster: int = 14

    def apron_status(self, apron_payroll: int) -> str:
        """Which band a payroll falls into. Drives salary-matching rules."""
        if apron_payroll >= self.second_apron:
            return "over_second_apron"
        if apron_payroll >= self.first_apron:
            return "between_aprons"
        if apron_payroll >= self.tax_level:
            return "taxpayer"
        if apron_payroll >= self.salary_cap:
            return "over_cap"
        return "under_cap"


LEAGUE_YEARS: Final[dict[str, LeagueYear]] = {
    # Official figures announced by the NBA for 2026-27.
    "2026-27": LeagueYear(
        season="2026-27",
        salary_cap=164_961_000,
        min_team_salary=148_465_000,
        tax_level=200_428_000,
        first_apron=209_015_000,
        second_apron=221_686_000,
        non_taxpayer_mle=15_044_000,
        taxpayer_mle=6_064_000,
        room_mle=9_366_000,
        bae=5_285_000,  # TODO: confirm against league release
        expanded_tpe=9_096_000,
        rookie_min=1_357_763,
        two_way_salary=678_882,
    ),
    # Prior year, used for backtesting the rule engine against real trades.
    "2025-26": LeagueYear(
        season="2025-26",
        salary_cap=154_647_000,
        min_team_salary=139_182_000,
        tax_level=187_895_000,
        first_apron=195_945_000,
        second_apron=207_824_000,
        non_taxpayer_mle=14_104_000,
        taxpayer_mle=5_685_000,
        room_mle=8_781_000,
        bae=5_134_000,
        expanded_tpe=8_527_000,
        rookie_min=1_272_870,
        two_way_salary=636_435,
    ),
}

CURRENT_SEASON: Final[str] = "2026-27"


def league_year(season: str | None = None) -> LeagueYear:
    return LEAGUE_YEARS[season or CURRENT_SEASON]


# --- Ingestion tuning ---------------------------------------------------
# stats.nba.com will silently start returning empty payloads or hang if
# you hit it too fast. These are conservative; a full historical backfill
# is an overnight job, not a coffee break.
NBA_STATS_MIN_INTERVAL_S: Final[float] = float(
    os.environ.get("NBARE_NBA_INTERVAL", "0.75")
)
NBA_STATS_TIMEOUT_S: Final[int] = 60
NBA_STATS_MAX_RETRIES: Final[int] = 5
