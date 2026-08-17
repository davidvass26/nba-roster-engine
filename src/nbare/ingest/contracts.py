"""Parse Basketball-Reference contract exports into the staging layer.

What this source gives you
--------------------------
Base salary by season, a "Guaranteed" total, and -- critically -- the
BBRef player slug in the last column. That slug is worth more than the
salaries: it seeds the crosswalk directly and removes any need to fuzzy
match ~440 names, a process that would silently merge Nikola Jokic with
Nikola Jovic (0.92 similarity, different people).

What this source does NOT give you, and why it matters
------------------------------------------------------
1. OPTION TYPE. We can infer *how many* years are guaranteed by walking
   the salary columns until the running total hits `Guaranteed`, but not
   whether year k+1 is a player option, team option, ETO, or simply
   non-guaranteed. Those have different cap-hold and trade consequences.
2. INCENTIVES. Apron payroll includes *likely* incentives. Missing here,
   so every apron number computed from this file is a lower bound.
3. DEAD MONEY ATTRIBUTION. Waived-and-stretched players appear on two
   teams with the SAME salary figures on both rows and different
   `Guaranteed` values. Summing naively double counts them.
4. CAP HOLDS. Free agents, empty roster charges, and unsigned draft
   picks are absent. Eight teams in this file are below the 14-man
   roster minimum, which is not legal -- it means the file is a list of
   contracts, not a roster.
5. TWO-WAY FLAGS, trade bonuses, no-trade clauses, Bird rights.

Bottom line: this is a good skeleton for base salary and an excellent
crosswalk seed. It is NOT sufficient to compute a legal apron payroll,
and Stage 1 must not pretend otherwise. Rows that cannot be resolved are
flagged, not guessed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import polars as pl

# BBRef uses three abbreviations NBA.com does not.
BBREF_TO_NBA_ABBREV: dict[str, str] = {
    "BRK": "BKN",
    "CHO": "CHA",
    "PHO": "PHX",
}

SALARY_COLUMNS: tuple[str, ...] = (
    "2026-27", "2027-28", "2028-29", "2029-30", "2030-31", "2031-32",
)

# Guarantee decomposition tolerance, in dollars. BBRef totals are exact
# sums in the clean case; anything that misses by more than a dollar is a
# partial guarantee or dead money, not a rounding artifact.
GUARANTEE_TOLERANCE = 1

# Columns that identify a contract row. `Rk` is a presentation index that
# differs between otherwise-identical rows (Jonathan Isaac appears at both
# Rk=204 and Rk=370 with identical data), so deduplication MUST exclude it
# or true duplicates survive into a primary-key violation downstream.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "player", "bbref_slug", "team_abbrev", *SALARY_COLUMNS, "Guaranteed",
)


def dedupe(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows identical on content, ignoring the Rk display index."""
    return df.unique(subset=list(IDENTITY_COLUMNS), keep="first")


@dataclass(slots=True)
class QualityReport:
    """Everything wrong with the file, enumerated rather than swallowed."""

    total_rows: int = 0
    exact_duplicate_rows: int = 0
    multi_team_players: list[str] = field(default_factory=list)
    partial_guarantee_players: list[str] = field(default_factory=list)
    fully_non_guaranteed: int = 0
    teams_below_roster_min: dict[str, int] = field(default_factory=dict)
    teams_above_roster_max: dict[str, int] = field(default_factory=dict)
    unknown_team_abbrevs: list[str] = field(default_factory=list)

    def blocking_issues(self) -> list[str]:
        """Issues that make apron payroll un-computable from this file alone."""
        out: list[str] = []
        if self.multi_team_players:
            out.append(
                f"{len(self.multi_team_players)} player(s) appear on multiple "
                "teams with identical salary rows (waive-and-stretch dead "
                "money). Cap hits cannot be attributed without a source that "
                "separates original salary from stretched dead money."
            )
        if self.teams_below_roster_min:
            out.append(
                f"{len(self.teams_below_roster_min)} team(s) below the 14-man "
                "minimum -- cap holds and unsigned free agents are missing, so "
                "team payroll is understated."
            )
        if self.teams_above_roster_max:
            out.append(
                f"{len(self.teams_above_roster_max)} team(s) above 15 standard "
                "contracts -- dead money and/or two-way deals are mixed in and "
                "not distinguishable."
            )
        out.append(
            "Likely incentives are absent from this source; every apron figure "
            "derived from it is a LOWER BOUND."
        )
        return out


def _parse_dollars(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    return int(s.replace("$", "").replace(",", "").strip())


def load_raw(path: Path | str) -> pl.DataFrame:
    """Read the two-header-row BBRef export into typed columns."""
    raw = pl.read_csv(path, skip_rows=1, infer_schema_length=0)
    df = raw.rename(
        {"-9999": "bbref_slug", "Tm": "bbref_team", "Player": "player"}
    )
    return df.with_columns(
        [
            pl.col(c).map_elements(_parse_dollars, return_dtype=pl.Int64)
            for c in (*SALARY_COLUMNS, "Guaranteed")
        ]
    ).with_columns(
        pl.col("bbref_team")
        .replace(BBREF_TO_NBA_ABBREV)
        .alias("team_abbrev")
    )


def guarantee_cliff(row: dict) -> tuple[int, bool]:
    """How many leading seasons are fully guaranteed.

    Walks the salary columns accumulating until the running total equals
    `Guaranteed`. Returns (n_guaranteed_seasons, decomposed_cleanly).

    A clean decomposition means the guarantee falls exactly on a season
    boundary, which is the normal case (402 of 413 rows in the 2026-27
    export). A failure means a partial-year guarantee or stretched dead
    money -- flagged, never guessed at.
    """
    gtd = row.get("Guaranteed")
    if gtd is None:
        return 0, True  # blank Guaranteed == nothing guaranteed
    running = 0
    for i, season in enumerate(SALARY_COLUMNS, start=1):
        val = row.get(season)
        if val is None:
            break
        running += val
        if abs(running - gtd) <= GUARANTEE_TOLERANCE:
            return i, True
    return 0, False


def contract_id_for(slug: str, team_abbrev: str) -> str:
    return hashlib.sha1(f"{slug}|{team_abbrev}".encode()).hexdigest()[:16]


def analyze(df: pl.DataFrame) -> QualityReport:
    """Enumerate data-quality problems before anything downstream trusts them."""
    rep = QualityReport(total_rows=df.height)

    # Ignore Rk: it is a display index, not part of row identity.
    rep.exact_duplicate_rows = df.height - dedupe(df).height

    by_slug = df.group_by("bbref_slug").agg(
        pl.col("player").first(), pl.col("team_abbrev").unique().alias("teams")
    )
    rep.multi_team_players = [
        r["player"]
        for r in by_slug.iter_rows(named=True)
        if len(r["teams"]) > 1
    ]

    for row in df.iter_rows(named=True):
        _, clean = guarantee_cliff(row)
        if not clean:
            rep.partial_guarantee_players.append(row["player"])
        if row["Guaranteed"] is None:
            rep.fully_non_guaranteed += 1

    counts = df.group_by("team_abbrev").agg(pl.len().alias("n"))
    for r in counts.iter_rows(named=True):
        if r["n"] < 14:
            rep.teams_below_roster_min[r["team_abbrev"]] = r["n"]
        elif r["n"] > 15:
            rep.teams_above_roster_max[r["team_abbrev"]] = r["n"]

    known = set(BBREF_TO_NBA_ABBREV.values()) | {
        "ATL", "BOS", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND",
        "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL",
        "PHI", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
    }
    rep.unknown_team_abbrevs = sorted(
        set(df["team_abbrev"].to_list()) - known
    )
    return rep


def to_contract_years(df: pl.DataFrame) -> pl.DataFrame:
    """Explode into one row per (contract, season) with guarantee flags.

    `needs_review` marks rows whose guarantee structure did not decompose
    cleanly OR whose player appears on multiple teams. Stage 1 must
    refuse to compute a legal payroll for any team containing such a row
    rather than silently producing a wrong number.
    """
    multi = {
        r["bbref_slug"]
        for r in df.group_by("bbref_slug")
        .agg(pl.col("team_abbrev").unique().alias("t"))
        .iter_rows(named=True)
        if len(r["t"]) > 1
    }

    records: list[dict] = []
    for row in dedupe(df).iter_rows(named=True):
        n_gtd, clean = guarantee_cliff(row)
        cid = contract_id_for(row["bbref_slug"], row["team_abbrev"])
        for i, season in enumerate(SALARY_COLUMNS, start=1):
            cap_hit = row.get(season)
            if cap_hit is None:
                continue
            records.append(
                {
                    "contract_id": cid,
                    "season": season,
                    "bbref_slug": row["bbref_slug"],
                    "player": row["player"],
                    "team_abbrev": row["team_abbrev"],
                    "season_index": i,
                    "cap_hit": cap_hit,
                    "guaranteed": cap_hit if i <= n_gtd else 0,
                    "is_guaranteed": i <= n_gtd,
                    # Unknown from this source. Explicitly NULL, not False --
                    # "we don't know" and "it isn't an option" are different
                    # facts and Stage 1 must be able to tell them apart.
                    "option_type": None,
                    "needs_review": (not clean) or row["bbref_slug"] in multi,
                    "review_reason": (
                        "multi_team_dead_money"
                        if row["bbref_slug"] in multi
                        else ("partial_guarantee" if not clean else None)
                    ),
                }
            )

    return pl.DataFrame(
        records,
        schema={
            "contract_id": pl.Utf8,
            "season": pl.Utf8,
            "bbref_slug": pl.Utf8,
            "player": pl.Utf8,
            "team_abbrev": pl.Utf8,
            "season_index": pl.Int16,
            "cap_hit": pl.Int64,
            "guaranteed": pl.Int64,
            "is_guaranteed": pl.Boolean,
            "option_type": pl.Utf8,
            "needs_review": pl.Boolean,
            "review_reason": pl.Utf8,
        },
    )


def crosswalk_seed(df: pl.DataFrame) -> list[tuple[str, str]]:
    """(display_name, bbref_slug) pairs -- feeds crosswalk.build directly.

    This is the highest-value column in the file. It converts crosswalk
    construction from a fuzzy-matching problem into a lookup.
    """
    return list(
        {
            (r["player"], r["bbref_slug"])
            for r in df.iter_rows(named=True)
            if r["bbref_slug"]
        }
    )


def team_payroll(
    years: pl.DataFrame, season: str = "2026-27"
) -> pl.DataFrame:
    """Base-salary payroll by team, with review contamination surfaced.

    Deliberately NOT called 'apron payroll'. Apron payroll includes likely
    incentives, cap holds, and correctly attributed dead money -- none of
    which this source provides. Naming it accurately is the difference
    between a number you can defend and one you cannot.
    """
    s = years.filter(pl.col("season") == season)
    return (
        s.group_by("team_abbrev")
        .agg(
            pl.col("cap_hit").sum().alias("base_salary_total"),
            pl.len().alias("contract_count"),
            pl.col("needs_review").sum().alias("rows_needing_review"),
            pl.col("cap_hit").filter(pl.col("needs_review")).sum().alias(
                "unattributed_salary"
            ),
        )
        .with_columns(
            (pl.col("unattributed_salary").fill_null(0)).alias(
                "unattributed_salary"
            )
        )
        .sort("base_salary_total", descending=True)
    )
