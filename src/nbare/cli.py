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


@app.command("ingest-box")
def ingest_box_cmd(
    season: str = typer.Option(CURRENT_SEASON),
    limit: int = typer.Option(0, help="0 = all games in the season"),
) -> None:
    """Resumable. Cached games are skipped for free, so rerun freely."""
    from nbare.ingest.nba_stats import ingest_box_scores

    with session() as con:
        rows = con.execute(
            "SELECT game_id FROM stg.game WHERE season = ? ORDER BY game_date",
            [season],
        ).fetchall()
        game_ids = [r[0] for r in rows]
        if limit:
            game_ids = game_ids[:limit]
        console.print(f"backfilling box scores for {len(game_ids):,} games")
        n = ingest_box_scores(con, game_ids)
    console.print(f"[green]box score rows: {n:,}")


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


@app.command("check-trade")
def check_trade_cmd(
    contracts: str = typer.Argument(..., help="BBRef contract CSV"),
    send: str = typer.Option(..., help="outgoing bbref slug, e.g. reaveau01"),
    receive: str = typer.Option(..., help="incoming bbref slug, e.g. vassede01"),
    season: str = typer.Option(CURRENT_SEASON),
) -> None:
    """Check salary-matching legality of a two-player swap between the
    teams the two players currently play for.

    This exercises the full Stage 1 path on real data: it builds each
    team's cap sheet from the contract file, constructs the trade, and
    prints a per-team verdict with the specific CBA band cited. Verdicts
    can be LEGAL, ILLEGAL, or INDETERMINATE -- the last is not a failure,
    it means a fact we cannot observe (incentives, cap room) could change
    the answer, and the engine declines to guess.
    """
    import polars as pl

    from nbare.cba.matching import Severity, check_trade
    from nbare.config import league_year
    from nbare.domain.models import (
        CapSheet, PlayerSalary, TradePiece, TradeProposal,
    )
    from nbare.domain.money import Money
    from nbare.ingest.contracts import load_raw, to_contract_years

    ly = league_year(season)
    years = to_contract_years(load_raw(contracts)).filter(
        pl.col("season") == season
    )

    def one(slug: str) -> dict:
        rows = years.filter(pl.col("bbref_slug") == slug).to_dicts()
        if not rows:
            raise typer.BadParameter(f"slug {slug!r} not found for {season}")
        return rows[0]

    def cap_sheet(team: str) -> CapSheet:
        rs = years.filter(pl.col("team_abbrev") == team).to_dicts()
        sal = tuple(
            PlayerSalary(
                player_id=r["bbref_slug"], name=r["player"],
                cap_hit=Money(r["cap_hit"]), guaranteed=Money(r["guaranteed"]),
                is_dead_money=bool(r["needs_review"]),
            )
            for r in rs
        )
        certain = not any(s.is_dead_money for s in sal)
        return CapSheet(
            team=team, season=season, salaries=sal, certain=certain,
            uncertainty_notes=(
                ("cap sheet contains rows flagged for review (dead money)",)
                if not certain else ()
            ),
        )

    a, b = one(send), one(receive)
    ta, tb = a["team_abbrev"], b["team_abbrev"]

    def to_ps(r: dict) -> PlayerSalary:
        return PlayerSalary(
            player_id=r["bbref_slug"], name=r["player"],
            cap_hit=Money(r["cap_hit"]), guaranteed=Money(r["guaranteed"]),
        )

    prop = TradeProposal(pieces=(
        TradePiece(to_ps(a), ta, tb),
        TradePiece(to_ps(b), tb, ta),
    ))
    report = check_trade({ta: cap_sheet(ta), tb: cap_sheet(tb)}, prop, ly)

    console.print(
        f"\n[bold]{a['player']}[/bold] ({ta}, ${a['cap_hit']:,})  <-->  "
        f"[bold]{b['player']}[/bold] ({tb}, ${b['cap_hit']:,})\n"
    )
    color = {
        Severity.LEGAL: "green",
        Severity.ILLEGAL: "red",
        Severity.INDETERMINATE: "yellow",
    }
    console.print(
        f"OVERALL: [{color[report.severity]}]"
        f"{report.severity.value.upper()}[/]\n"
    )
    for v in report.verdicts:
        console.print(f"[{color[v.severity]}]{v.team}: {v.severity.value.upper()}[/]"
                      f"  [{v.band}]")
        console.print(f"  {v.reason}")
        console.print(f"  [dim]rule: {v.rule}[/dim]")
        for c in v.caveats:
            console.print(f"  [yellow]caveat:[/yellow] {c}")
        console.print()


@app.command("sign")
def sign_cmd(
    contracts: str = typer.Argument(..., help="base BBRef contract CSV"),
    player: str = typer.Option(..., help="display name, e.g. 'LeBron James'"),
    team: str = typer.Option(..., help="team abbreviation, e.g. PHI"),
    cap_hit: int = typer.Option(..., help="cap hit in whole dollars, e.g. 4000000"),
    slug: str = typer.Option(None, help="BBRef slug if known, e.g. jamesle01"),
    guaranteed: int = typer.Option(None, help="guaranteed dollars (defaults to full cap hit)"),
    season: str = typer.Option(CURRENT_SEASON),
    note: str = typer.Option("", help="optional free-text note"),
    show_team: bool = typer.Option(True, help="print the team's cap sheet after signing"),
) -> None:
    """Sign a player to a team and show the cap impact.

    This applies a single signing on top of the base contract snapshot and
    prints the result — no YAML file needed. Under the hood it uses the same
    transaction overlay system as apply-overrides, so the base CSV is never
    modified.

    Examples
    --------
    # LeBron James signing with PHI at $4M (year 1 of 2yr/$8M):
    nbare sign data/raw/bbref_contracts_2026-27.csv \\
        --player "LeBron James" --slug jamesle01 --team PHI --cap-hit 4000000

    # A hypothetical NTMLE signing in Denver:
    nbare sign data/raw/bbref_contracts_2026-27.csv \\
        --player "Free Agent" --team DEN --cap-hit 15000000
    """
    import polars as pl

    from nbare.config import league_year
    from nbare.ingest.contracts import load_raw, to_contract_years
    from nbare.ingest.transactions import Sign, apply_transactions

    ly = league_year(season)
    base = to_contract_years(load_raw(contracts))
    gtd = guaranteed if guaranteed is not None else cap_hit

    txn = Sign(
        player=player,
        slug=slug,
        team=team,
        cap_hit=cap_hit,
        guaranteed=gtd,
        note=note,
    )
    result = apply_transactions(base, [txn], season)

    for w in result.warnings:
        console.print(f"[yellow]![/yellow] {w}")

    console.print(f"\n[green]signed[/green] {player} → {team} @ ${cap_hit:,}/yr  "
                  f"(gtd ${gtd:,})")

    if show_team:
        roster = result.frame.filter(
            pl.col("team_abbrev") == team
        ).sort("cap_hit", descending=True)
        total = int(roster["cap_hit"].sum())
        band = ly.apron_status(total)
        bcolor = {
            "over_second_apron": "red",
            "between_aprons": "yellow",
            "taxpayer": "yellow",
        }.get(band, "green")

        t = Table(
            title=f"{team} after signing  "
                  f"[{bcolor}]${total:,} ({band})[/{bcolor}]  "
                  f"{roster.height} players"
        )
        t.add_column("player")
        t.add_column("cap hit", justify="right")
        t.add_column("gtd", justify="right")
        for r in roster.select("player", "cap_hit", "guaranteed").iter_rows(named=True):
            gtd_str = "✓" if r["guaranteed"] == r["cap_hit"] else f"${r['guaranteed']:,}"
            t.add_row(r["player"], f"${r['cap_hit']:,}", gtd_str)
        console.print(t)
        console.print(
            f"\n  [dim]first apron ${ly.first_apron:,} | "
            f"second apron ${ly.second_apron:,}[/dim]"
        )
        if total > ly.second_apron:
            console.print(f"  [red]⚠ over the second apron by ${total - ly.second_apron:,}[/red]")
        elif total > ly.first_apron:
            console.print(f"  [yellow]⚠ over the first apron by ${total - ly.first_apron:,}[/yellow]")


@app.command("waive")
def waive_cmd(
    contracts: str = typer.Argument(..., help="base BBRef contract CSV"),
    slug: str = typer.Option(..., help="BBRef slug of the player to waive"),
    team: str = typer.Option(..., help="team abbreviation"),
    stretch: bool = typer.Option(False, "--stretch", help="leave guaranteed money as dead money"),
    season: str = typer.Option(CURRENT_SEASON),
    note: str = typer.Option("", help="optional free-text note"),
) -> None:
    """Waive a player and show the team's updated cap sheet.

    By default, the player's salary comes fully off the books (correct for
    non-guaranteed deals). Use --stretch to keep the guaranteed amount as a
    dead-money row (correct for waived guaranteed contracts).

    Example
    -------
    nbare waive data/raw/bbref_contracts_2026-27.csv --slug terryda01 --team PHI
    """
    import polars as pl

    from nbare.config import league_year
    from nbare.ingest.contracts import load_raw, to_contract_years
    from nbare.ingest.transactions import Waive, apply_transactions

    ly = league_year(season)
    base = to_contract_years(load_raw(contracts))

    # Show who we're waiving before we do it.
    row = base.filter(
        (pl.col("bbref_slug") == slug) & (pl.col("season") == season)
    ).to_dicts()
    if row:
        r = row[0]
        console.print(
            f"\nwaiving [bold]{r['player']}[/bold] ({r['team_abbrev']}) "
            f"${r['cap_hit']:,}  gtd=${r['guaranteed']:,}"
            + (" — dead money stays" if stretch and r["guaranteed"] > 0 else " — clean removal")
        )
    else:
        console.print(f"[yellow]slug {slug!r} not found for {season}; applying anyway[/yellow]")

    txn = Waive(slug=slug, team=team, stretch_dead_money=stretch, note=note)
    result = apply_transactions(base, [txn], season)

    for w in result.warnings:
        console.print(f"[yellow]![/yellow] {w}")

    roster = result.frame.filter(
        pl.col("team_abbrev") == team
    ).sort("cap_hit", descending=True)
    total = int(roster["cap_hit"].sum())
    band = ly.apron_status(total)
    bcolor = {
        "over_second_apron": "red",
        "between_aprons": "yellow",
        "taxpayer": "yellow",
    }.get(band, "green")

    t = Table(
        title=f"{team} after waive  "
              f"[{bcolor}]${total:,} ({band})[/{bcolor}]  "
              f"{roster.height} players"
    )
    t.add_column("player")
    t.add_column("cap hit", justify="right")
    for r in roster.select("player", "cap_hit", "needs_review").iter_rows(named=True):
        flag = " [dim](dead)[/dim]" if r["needs_review"] else ""
        t.add_row(r["player"] + flag, f"${r['cap_hit']:,}")
    console.print(t)



@app.command("apply-overrides")
def apply_overrides_cmd(
    contracts: str = typer.Argument(..., help="base BBRef contract CSV"),
    overlay: str = typer.Argument(..., help="transaction overlay YAML"),
    season: str = typer.Option(CURRENT_SEASON),
    team: str = typer.Option(None, help="show only this team's roster after"),
) -> None:
    """Apply a transaction overlay (signings, waives, trades) on top of the
    base contract snapshot and show the result.

    The base CSV is never modified. The overlay is an ordered, diffable
    transaction log; anything suspicious (waiving a player who isn't on the
    named team, amending a row that doesn't exist) is reported as a warning
    rather than silently applied.
    """
    import polars as pl

    from nbare.config import league_year
    from nbare.ingest.contracts import load_raw, to_contract_years
    from nbare.ingest.transactions import apply_transactions, load_transactions

    ly = league_year(season)
    base = to_contract_years(load_raw(contracts))
    txns = load_transactions(overlay)
    res = apply_transactions(base, txns, season)

    console.print(f"[bold]applied {len(txns)} transaction(s)[/bold]")
    for line in res.log:
        console.print(f"  {line}")
    if res.warnings:
        console.print("\n[yellow]warnings:[/yellow]")
        for w in res.warnings:
            console.print(f"  [yellow]![/yellow] {w}")

    teams = [team] if team else sorted(res.frame["team_abbrev"].unique().to_list())
    for tm in teams:
        roster = res.frame.filter(pl.col("team_abbrev") == tm).sort(
            "cap_hit", descending=True
        )
        total = int(roster["cap_hit"].sum())
        count = roster.filter(~pl.col("needs_review")).height
        band = ly.apron_status(total)
        color = "red" if band == "over_second_apron" else (
            "yellow" if band in ("between_aprons", "taxpayer") else "green"
        )
        flags = []
        if count > ly.max_roster:
            flags.append(f"{count} players > {ly.max_roster} max")
        if count < ly.min_roster:
            flags.append(f"{count} players < {ly.min_roster} min")
        flagstr = f"  [red]({'; '.join(flags)})[/red]" if flags else ""
        if team or total > ly.first_apron:
            console.print(
                f"\n[bold]{tm}[/bold]: ${total:,} base  "
                f"[{color}]{band}[/{color}]  {count} players{flagstr}"
            )
            if team:
                for r in roster.select("player", "cap_hit").iter_rows(named=True):
                    console.print(f"  {r['player']:28s} ${r['cap_hit']:>12,}")

@app.command("fit-rapm")
def fit_rapm_cmd(
    season: str = typer.Option(CURRENT_SEASON),
    limit: int = typer.Option(0, help="cap games (0 = all in season)"),
    min_possessions: float = typer.Option(1.0, help="drop tiny blocks"),
    top: int = typer.Option(25, help="print this many top players"),
) -> None:
    """Fit offense/defense RAPM from the warehouse for a season.

    Runs the full real-data chain: reconstruct stints -> split by team ->
    offense blocks -> sparse design -> ridge with grouped CV. Requires a
    play-by-play + box-score backfill (`make pbp`) and should only be
    trusted after `check-minutes` passes for the season.
    """
    from nbare.rapm.blocks import blocks_from_warehouse
    from nbare.rapm.design import build_design
    from nbare.rapm.fit import fit_rapm

    with session() as con:
        rows = con.execute(
            "SELECT game_id FROM stg.game WHERE season = ? ORDER BY game_date"
            + (f" LIMIT {limit}" if limit else ""),
            [season],
        ).fetchall()
        if not rows:
            console.print(
                "[yellow]no games in warehouse for that season. Run "
                "`make games`/`make pbp` first."
            )
            return
        game_ids = [r[0] for r in rows]
        console.print(f"building blocks from {len(game_ids):,} games...")
        blockres = blocks_from_warehouse(con, game_ids)

        # Attach readable names if we have them.
        names = {
            r[0]: r[1]
            for r in con.execute(
                "SELECT nba_player_id, full_name FROM stg.player"
            ).fetchall()
        }

    if not blockres.blocks:
        console.print("[red]no usable blocks produced (check the minutes gate).")
        return
    console.print(
        f"{len(blockres.blocks):,} blocks, {blockres.skipped_stints} stints "
        "skipped (bad team split)"
    )

    design = build_design(blockres.blocks, min_possessions=min_possessions)
    console.print(
        f"design: {design.X.shape[0]:,} rows x {design.X.shape[1]:,} cols; "
        "fitting with grouped CV..."
    )
    result = fit_rapm(design)
    console.print(
        f"selected lambda: {result.best_lambda}  "
        f"(intercept {result.intercept:.1f} pts/100)"
    )

    t = Table(title=f"Top {top} by total RAPM — {season}")
    t.add_column("#", justify="right")
    t.add_column("player")
    t.add_column("total", justify="right")
    t.add_column("off", justify="right")
    t.add_column("def", justify="right")
    for i, (pid, total, off, deff) in enumerate(result.ranking()[:top], 1):
        t.add_row(
            str(i), names.get(pid, str(pid)),
            f"{total:+.1f}", f"{off:+.1f}", f"{deff:+.1f}",
        )
    console.print(t)


@app.command("check-minutes")
def check_minutes_cmd(
    season: str = typer.Option(CURRENT_SEASON),
    limit: int = typer.Option(20, help="check this many games (0 = all)"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="run the gate on generated games (no data needed)"
    ),
) -> None:
    """Stage 2 gate: reconstructed minutes vs official box score.

    Do not build RAPM until this passes on real data. A lineup
    reconstruction that is quietly wrong produces a design matrix that is
    quietly wrong, and the regression will not tell you.

    --synthetic runs the gate on generated games with known ground truth,
    which needs no warehouse and proves the reconstruction logic end to end.
    """
    from nbare.rapm.stints import reconstruct_game, validate_minutes

    if synthetic:
        from nbare.rapm.synthetic import box_frame, make_game

        failures = 0
        for seed in range(limit or 20):
            g = make_game(seed=seed)
            res = reconstruct_game(g.pbp)
            chk = validate_minutes(res, box_frame(g))
            if not chk.passed:
                failures += 1
                console.print(f"[red]seed {seed}: {chk.summary()}")
        n = limit or 20
        if failures:
            console.print(f"[red]{failures}/{n} synthetic games FAILED")
        else:
            console.print(f"[green]all {n} synthetic games passed the minutes gate")
        return

    # Real path: read PBP + box scores from the warehouse.
    with session(read_only=True) as con:
        games = con.execute(
            "SELECT game_id FROM stg.game WHERE season = ? ORDER BY game_date"
            + (f" LIMIT {limit}" if limit else ""),
            [season],
        ).fetchall()
        if not games:
            console.print(
                "[yellow]no games in warehouse. Run `make games`/`make pbp` "
                "first, or try --synthetic to test the logic now."
            )
            return

        passed = failed = 0
        for (gid,) in games:
            pbp = con.execute(
                "SELECT * FROM stg.pbp_event WHERE game_id = ?", [gid]
            ).pl()
            box = con.execute(
                "SELECT * FROM stg.box_player WHERE game_id = ?", [gid]
            ).pl()
            if pbp.is_empty() or box.is_empty():
                continue
            res = reconstruct_game(pbp)
            chk = validate_minutes(res, box)
            if chk.passed:
                passed += 1
            else:
                failed += 1
                console.print(f"[red]{gid}: {chk.summary()}")

    total = passed + failed
    color = "green" if failed == 0 else "red"
    console.print(
        f"[{color}]minutes gate: {passed}/{total} games passed"
        + ("" if failed == 0 else f" -- {failed} FAILED, do not build RAPM yet")
    )


if __name__ == "__main__":
    app()