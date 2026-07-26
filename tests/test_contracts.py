from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nbare.ingest.contracts import (
    BBREF_TO_NBA_ABBREV,
    analyze,
    contract_id_for,
    crosswalk_seed,
    dedupe,
    guarantee_cliff,
    load_raw,
    team_payroll,
    to_contract_years,
)

CSV = Path(__file__).parents[1] / "data" / "raw" / "bbref_contracts_2026-27.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="contract CSV absent")


@pytest.fixture(scope="module")
def df():
    return load_raw(CSV)


@pytest.fixture(scope="module")
def years(df):
    return to_contract_years(df)


# --- parsing -------------------------------------------------------------

def test_dollar_columns_are_integers(df):
    assert df["2026-27"].dtype == pl.Int64
    assert df.filter(pl.col("bbref_slug") == "curryst01")["2026-27"][0] == 62_587_158


def test_team_abbrevs_translated_to_nba_convention(df):
    present = set(df["team_abbrev"].to_list())
    for bbref, nba in BBREF_TO_NBA_ABBREV.items():
        assert bbref not in present, f"{bbref} should have been mapped to {nba}"
        assert nba in present
    assert len(present) == 30


def test_blank_guaranteed_means_zero_guaranteed(df):
    harden = df.filter(pl.col("bbref_slug") == "hardeja01").to_dicts()[0]
    assert harden["Guaranteed"] is None
    assert guarantee_cliff(harden) == (0, True)


# --- guarantee inference -------------------------------------------------

def test_guarantee_cliff_falls_on_season_boundary():
    row = {
        "2026-27": 100, "2027-28": 200, "2028-29": 300,
        "2029-30": None, "2030-31": None, "2031-32": None,
        "Guaranteed": 300,
    }
    assert guarantee_cliff(row) == (2, True)


def test_guarantee_cliff_flags_partial_guarantee():
    """A guarantee that does not land on a season boundary is dead money
    or a partial guarantee -- must be flagged, never rounded to a guess."""
    row = {
        "2026-27": 100, "2027-28": 200, "2028-29": None,
        "2029-30": None, "2030-31": None, "2031-32": None,
        "Guaranteed": 150,
    }
    n, clean = guarantee_cliff(row)
    assert clean is False and n == 0


def test_real_option_years_detected(years):
    """Jokic's 2027-28 is not guaranteed; his 2026-27 is."""
    j = years.filter(pl.col("bbref_slug") == "jokicni01").sort("season_index")
    assert j["is_guaranteed"].to_list() == [True, False]


def test_tatum_last_year_non_guaranteed(years):
    t = years.filter(pl.col("bbref_slug") == "tatumja01").sort("season_index")
    assert t["is_guaranteed"].to_list() == [True, True, True, False]


# --- deduplication -------------------------------------------------------

def test_rk_column_excluded_from_row_identity(df):
    """Regression: Jonathan Isaac appears at Rk=204 and Rk=370 with
    otherwise identical data. Deduping on all columns (including Rk)
    misses it and produces a duplicate (contract_id, season) key."""
    isaac = df.filter(pl.col("bbref_slug") == "isaacjo01")
    assert isaac.height == 2
    assert dedupe(isaac).height == 1


def test_contract_year_primary_key_is_unique(years):
    dupes = (
        years.group_by(["contract_id", "season"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    assert dupes.height == 0, "duplicate (contract_id, season) would break the PK"


def test_contract_id_is_team_specific():
    """A stretched player on two teams must not collapse into one contract."""
    assert contract_id_for("lillada01", "MIL") != contract_id_for("lillada01", "POR")


# --- data quality report -------------------------------------------------

def test_multi_team_players_are_detected(df):
    rep = analyze(df)
    assert "Damian Lillard" in rep.multi_team_players
    assert "Bradley Beal" in rep.multi_team_players


def test_multi_team_rows_flagged_for_review(years):
    lil = years.filter(pl.col("bbref_slug") == "lillada01")
    assert lil["needs_review"].all()
    assert set(lil["review_reason"].to_list()) == {"multi_team_dead_money"}


def test_roster_count_violations_surface(df):
    """The file is a contract list, not a roster: some teams fall below
    the 14-man minimum and others exceed 15 standard deals."""
    rep = analyze(df)
    assert rep.teams_below_roster_min, "expected teams missing cap holds"
    assert rep.teams_above_roster_max, "expected teams with dead money mixed in"


def test_blocking_issues_are_reported(df):
    rep = analyze(df)
    issues = rep.blocking_issues()
    assert any("incentives" in i.lower() for i in issues)
    assert any("dead money" in i.lower() for i in issues)


def test_option_type_is_null_not_false(years):
    """We cannot distinguish player/team option from plain non-guaranteed
    with this source. Unknown must stay unknown."""
    assert years["option_type"].null_count() == years.height


# --- payroll -------------------------------------------------------------

def test_payroll_surfaces_unattributed_salary(years):
    pay = team_payroll(years, "2026-27")
    assert pay.height == 30
    contaminated = pay.filter(pl.col("rows_needing_review") > 0)
    assert contaminated.height > 0
    assert (contaminated["unattributed_salary"] > 0).all()


def test_payroll_is_not_called_apron_payroll(years):
    """Naming discipline: this is base salary, not a legal apron figure."""
    assert "base_salary_total" in team_payroll(years).columns
    assert "apron_payroll" not in team_payroll(years).columns


# --- crosswalk seed ------------------------------------------------------

def test_crosswalk_seed_covers_every_player(df, years):
    seed = crosswalk_seed(df)
    assert len(seed) == df["bbref_slug"].n_unique()
    assert ("Stephen Curry", "curryst01") in seed


def test_similar_names_have_distinct_slugs(df):
    """Jokic/Jovic and the two Gueyes are different people. The slug
    column is what makes this safe; fuzzy name matching is not."""
    slugs = dict(zip(df["player"].to_list(), df["bbref_slug"].to_list()))
    assert slugs["Nikola Jokić"] != slugs["Nikola Jović"]
    assert slugs["Mouhamadou Gueye"] != slugs["Mouhamed Gueye"]
