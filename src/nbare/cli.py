"""Command line entry points for Stage 0 ingestion."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from nbare.config import CURRENT_SEASON, DB_PATH
from nbare.warehouse.db import session, table_counts

app = typer.Typer(add_completion=False, help="NBA roster construction engine")
console = Console()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
)


@app.command()
def status() -> None:
    """Row counts for every warehouse table."""
    from nbare.ingest.client import NBAStatsCache

    with session(read_only=False) as con:
        counts = table_counts(con)
    t = Table(title=f"nbare warehouse @ {DB_PATH}")
    t.add_column("table")
    t.add_column("rows", justify="right")
    for name, n in counts.items():
        t.add_row(name, f"{n:,}")
    console.print(t)
    console.print(f"cached nba.com responses: {len(NBAStatsCache()):,}")


@app.command("ingest-teams")
def ingest_teams_cmd() -> None:
    from nbare.ingest.nba_stats import ingest_teams

    with session() as con:
        n = ingest_teams(con)
    console.print(f"[green]teams: {n}")


@app.command("ingest-players")
def ingest_players_cmd() -> None:
    from nbare.ingest.nba_stats import ingest_players

    with session() as con:
        n = ingest_players(con)
    console.print(f"[green]players: {n:,}")


@app.command("ingest-games")
def ingest_games_cmd(
    season: str = typer.Option(CURRENT_SEASON, help="e.g. 2025-26"),
    playoffs: bool = typer.Option(True, help="also ingest playoff games"),
) -> None:
    from nbare.ingest.nba_stats import ingest_season_games

    with session() as con:
        n = ingest_season_games(con, season, "Regular Season")
        if playoffs:
            n += ingest_season_games(con, season, "Playoffs")
    console.print(f"[green]games for {season}: {n:,}")


@app.command("ingest-pbp")
def ingest_pbp_cmd(
    season: str = typer.Option(CURRENT_SEASON),
    limit: int = typer.Option(0, help="0 = all games in the season"),
) -> None:
    """Resumable. Cached games are skipped for free, so rerun freely."""
    from nbare.ingest.nba_stats import ingest_pbp

    with session() as con:
        rows = con.execute(
            "SELECT game_id FROM stg.game WHERE season = ? ORDER BY game_date",
            [season],
        ).fetchall()
        game_ids = [r[0] for r in rows]
        if limit:
            game_ids = game_ids[:limit]
        console.print(f"backfilling PBP for {len(game_ids):,} games")
        n = ingest_pbp(con, game_ids)
    console.print(f"[green]pbp events: {n:,}")


@app.command("ingest-contracts")
def ingest_contracts_cmd(
    path: str = typer.Argument(..., help="Basketball-Reference contract CSV"),
    season: str = typer.Option(CURRENT_SEASON),
) -> None:
    """Load a BBRef contract export and print the data-quality report.

    The report is not decoration. Every blocking issue it lists is a
    reason Stage 1 cannot compute a legal apron payroll from this source
    alone, and Stage 1 is expected to refuse rather than guess.
    """
    from nbare.ingest.contracts import (
        analyze, load_raw, team_payroll, to_contract_years,
    )

    df = load_raw(path)
    rep = analyze(df)
    years = to_contract_years(df)

    t = Table(title="contract data quality")
    t.add_column("check")
    t.add_column("value", justify="right")
    t.add_row("rows", f"{rep.total_rows:,}")
    t.add_row("exact duplicate rows", str(rep.exact_duplicate_rows))
    t.add_row("multi-team (dead money)", str(len(rep.multi_team_players)))
    t.add_row("partial guarantees", str(len(rep.partial_guarantee_players)))
    t.add_row("fully non-guaranteed", str(rep.fully_non_guaranteed))
    t.add_row("teams below 14-man min", str(len(rep.teams_below_roster_min)))
    t.add_row("teams above 15 standard", str(len(rep.teams_above_roster_max)))
    console.print(t)

    for issue in rep.blocking_issues():
        console.print(f"[yellow]! {issue}")

    pay = team_payroll(years, season)
    bad = pay.filter(pl_col_gt(pay, "rows_needing_review"))
    console.print(
        f"\n[green]{years.height:,} contract-years parsed; "
        f"{years['needs_review'].sum()} flagged for review across "
        f"{bad.height} team(s)."
    )


def pl_col_gt(df, col: str, n: int = 0):
    import polars as pl

    return pl.col(col) > n


@app.command("build-xwalk")
def build_xwalk_cmd(
    contracts: str = typer.Option(
        None, help="BBRef contract CSV -- seeds bbref slugs directly"
    ),
) -> None:
    """Build the id crosswalk.

    Passing --contracts turns crosswalk construction from a fuzzy-matching
    problem into a lookup: the BBRef export carries the slug in its last
    column, so ~440 players resolve exactly with no similarity threshold
    involved. That matters because fuzzy matching on these names merges
    Nikola Jokic with Nikola Jovic at 0.92 similarity.
    """
    from nbare.crosswalk.build import build, unresolved
    from nbare.ingest.contracts import crosswalk_seed, load_raw

    seed: list[tuple[str, str]] = []
    if contracts:
        seed = crosswalk_seed(load_raw(contracts))
        console.print(f"seeded {len(seed):,} exact bbref slugs from {contracts}")

    with session() as con:
        df = build(con, bbref=seed, spotrac=[])
        miss = unresolved(con)
    console.print(f"[green]crosswalk rows: {df.height:,}")
    console.print(f"[yellow]unresolved: {miss.height:,} (see overrides.yaml)")


@app.command("check-minutes")
def check_minutes_cmd(season: str = typer.Option(CURRENT_SEASON)) -> None:
    """Stage 2 gate: reconstructed minutes vs official box score.

    Do not start the RAPM work until this passes. A lineup reconstruction
    that is quietly wrong produces a design matrix that is quietly wrong,
    and you will not find out from the regression diagnostics.
    """
    console.print("[yellow]not implemented until stint builder exists (Stage 2)")


if __name__ == "__main__":
    app()
