"""Who differs most, in either direction, between box-score prior and RAPM?

    python scripts/prior_vs_rapm.py
    python scripts/prior_vs_rapm.py --seasons 2025-26 --by z --side defense
    python scripts/prior_vs_rapm.py --csv /path/to/gaps.csv

Runs the same gated fit as `nbare export-app-data` (minutes gate -> design ->
box-score prior -> ridge) because the prior means are not exported to
app/data. The math and its traps live in `nbare.rapm.prior_gap`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import typer
from rich.table import Table

from nbare.cli import _fit_with_prior, console
from nbare.rapm.blocks import blocks_from_warehouse, merge_warehouse_results
from nbare.rapm.design import build_design
from nbare.rapm.fit import REPLACEMENT_LEVEL
from nbare.rapm.prior_gap import PlayerGap, prior_gaps
from nbare.warehouse.db import session

# Combined off + def possessions; 2,000 is the prior's own training bar
# (DEFAULT_RELIABLE_POSSESSIONS = 1,000 per side).
DEFAULT_MIN_POSSESSIONS = 2000.0


def _gap_table(title: str, rows: list[PlayerGap], names: dict, side: str) -> Table:
    t = Table(title=title)
    t.add_column("#", justify="right")
    t.add_column("player")
    for col in ("rapm", "box prior", "gap", "z", "off gap", "def gap", "poss"):
        t.add_column(col, justify="right")
    for i, p in enumerate(rows, 1):
        s = getattr(p, side)
        t.add_row(
            str(i), names.get(p.player_id, str(p.player_id)),
            f"{s.rapm:+.2f}", f"{s.prior:+.2f}", f"{s.gap:+.2f}", f"{s.z:+.1f}",
            f"{p.offense.gap:+.2f}", f"{p.defense.gap:+.2f}", f"{p.possessions:,.0f}",
        )
    return t


def _write_csv(path: Path, rows: list[PlayerGap], names: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "name", "possessions"] + [
            f"{side}_{field}"
            for side in ("total", "offense", "defense")
            for field in ("rapm", "prior", "gap", "se", "z")
        ])
        for p in rows:
            w.writerow([p.player_id, names.get(p.player_id, ""), p.possessions] + [
                getattr(s, field)
                for s in (p.total, p.offense, p.defense)
                for field in ("rapm", "prior", "gap", "se", "z")
            ])


def main(
    seasons: str = typer.Option(
        "2023-24,2024-25,2025-26", help="comma-separated seasons, pooled into one fit"
    ),
    side: str = typer.Option("total", help="total, offense, or defense"),
    by: str = typer.Option("gap", help="rank by 'gap' (size) or 'z' (confidence)"),
    top: int = typer.Option(20, help="players to show in each direction"),
    min_possessions: float = typer.Option(
        DEFAULT_MIN_POSSESSIONS, help="minimum combined off + def possessions"
    ),
    csv_path: Path = typer.Option(None, "--csv", help="also write every player here"),
) -> None:
    season_list = [s.strip() for s in seasons.split(",")]
    results = []
    game_ids: list[str] = []
    with session() as con:
        names = dict(con.execute(
            "SELECT nba_player_id, full_name FROM stg.player"
        ).fetchall())
        for s in season_list:
            ids = [r[0] for r in con.execute(
                "SELECT game_id FROM stg.game WHERE season = ? ORDER BY game_date", [s]
            ).fetchall()]
            if not ids:
                console.print(f"[red]no games in warehouse for {s}")
                raise typer.Exit(1)
            console.print(f"[{s}] gating and building blocks from {len(ids):,} games...")
            results.append(blocks_from_warehouse(con, ids))
            game_ids.extend(ids)

    design = build_design(merge_warehouse_results(results).blocks)
    result, fit_type = _fit_with_prior(design, "box", game_ids, REPLACEMENT_LEVEL)
    gaps = prior_gaps(design, result)
    ranked = gaps.ranked(side=side, by=by, min_possessions=min_possessions)
    if not ranked:
        console.print("[red]no players with a box prior clear the possession minimum.")
        raise typer.Exit(1)

    rapm = np.array([getattr(p, side).rapm for p in ranked])
    prior = np.array([getattr(p, side).prior for p in ranked])
    console.print(
        f"\n{fit_type}, lambda {result.best_lambda}: {len(ranked):,} players with "
        f"{min_possessions:,.0f}+ possessions ({gaps.n_no_box_prior:,} players overall "
        f"left out: no box prior). r(rapm, box prior) = {np.corrcoef(rapm, prior)[0, 1]:.2f}, "
        f"SD rapm {rapm.std():.2f}, SD prior {prior.std():.2f}"
    )
    # A gap is rapm - prior; z is gap / posterior SD (see rapm/prior_gap.py).
    console.print(_gap_table(
        f"RAPM above box score ({side}, by {by})", ranked[:top], names, side
    ))
    console.print(_gap_table(
        f"Box score above RAPM ({side}, by {by})", ranked[::-1][:top], names, side
    ))
    if csv_path is not None:
        _write_csv(csv_path, ranked, names)
        console.print(f"[green]wrote {csv_path} ({len(ranked):,} rows)")


if __name__ == "__main__":
    typer.run(main)
