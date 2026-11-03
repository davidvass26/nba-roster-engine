"""nbare — NBA Roster Construction Engine (Streamlit app, v1).

See docs/app_spec.md for the design brief this implements. Entry point for
Streamlit Community Cloud: `streamlit run app/streamlit_app.py` from the
repo root (so `nbare` and `app/data/*` resolve as relative imports/paths).

Data contract: this file (and data_access.py) reads ONLY the live contract
CSV and the precomputed app/data/ files -- never the DuckDB warehouse or
nba.com cache. See data_access.py's module docstring for why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

# So `import data_access` works whether Streamlit's cwd is the repo root
# (the deploy convention this app requires) or app/ itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# So `import nbare` works when the repo root isn't already on sys.path
# (e.g. a bare `streamlit run app/streamlit_app.py` without `pip install
# -e .` having put it there some other way).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data_access as da  # noqa: E402
from nbare.cba.matching import Severity, check_trade  # noqa: E402
from nbare.config import CURRENT_SEASON, league_year  # noqa: E402
from nbare.domain.models import TradePiece, TradeProposal  # noqa: E402
from nbare.domain.money import Money  # noqa: E402

SEASON = CURRENT_SEASON  # 2026-27 -- the only season the contract CSV covers
REPO_URL = "https://github.com/davidvass26/nba-roster-engine"

SEVERITY_COLOR = {
    Severity.LEGAL: "green",
    Severity.ILLEGAL: "red",
    Severity.INDETERMINATE: "orange",
}
SEVERITY_LABEL = {
    Severity.LEGAL: "LEGAL",
    Severity.ILLEGAL: "ILLEGAL",
    Severity.INDETERMINATE: "INDETERMINATE",
}

st.set_page_config(
    page_title="nbare — NBA Roster Engine", page_icon="🏀", layout="wide"
)


# --- shared helpers ---------------------------------------------------------

def to_ps(row: dict):
    from nbare.domain.models import PlayerSalary

    return PlayerSalary(
        player_id=row["bbref_slug"], name=row["player"],
        cap_hit=Money(row["cap_hit"]), guaranteed=Money(row["guaranteed"]),
    )


def player_label(row: dict) -> str:
    return f"{row['player']} (${row['cap_hit']:,})"


def rapm_badge(name: str, lookup: dict[str, dict]) -> str:
    r = lookup.get(name)
    if r is None:
        return "_no RAPM data (name not matched to the pooled-fit leaderboard)_"
    last = (meta.get("last_season") or {}).get("season")
    n = len(meta.get("seasons_used") or [])
    yr = f"{n}-yr" if n > 1 else "pooled"
    pooled = (
        f"{yr} RAPM {r['total']:+.1f} (off {r['offense']:+.1f} / def {r['defense']:+.1f})"
        f"  ·  {yr} Wins Added {r['wins_added']:+.1f}"
    )
    if last is None or "last_total" not in r:
        return pooled
    if r["last_total"] is None:
        return f"_did not play {last}_  ·  " + pooled
    return (
        f"{last} RAPM {r['last_total']:+.1f}  ·  {last} Wins Added "
        f"{r['last_wins_added']:+.1f}  ·  " + pooled
    )


years = da.load_contract_years()
all_teams = da.teams_in(years, SEASON)
ly = league_year(SEASON)
rapm_board = da.load_rapm_leaderboard()
rapm_lookup = da.rapm_lookup_by_name(rapm_board)
meta = da.load_meta()

st.title("🏀 nbare — NBA Roster Construction Engine")
st.caption(
    "Contracts → CBA rules → RAPM → value → projections. A working system, "
    "not a slide deck. All numbers on this page are computed live or "
    "precomputed by the engine itself — nothing here is hand-typed."
)

# =============================================================================
# LANDING: TRADE CHECKER — no tab, no click required. This is the rarest
# asset (a working CBA legality engine), so it is the first thing a visitor
# sees, pre-filled with a real trade so the page is alive on load.
# =============================================================================

st.header("Trade Checker")
st.caption(
    f"Salary-matching legality under the 2023 CBA, {SEASON} figures. "
    "Pick players on each side; the verdict updates immediately."
)

default_team_a, default_team_b = "PHI", "CLE"
idx_a = all_teams.index(default_team_a) if default_team_a in all_teams else 0
idx_b = all_teams.index(default_team_b) if default_team_b in all_teams else 1

col_a, col_b = st.columns(2)
with col_a:
    team_a = st.selectbox("Team A", all_teams, index=idx_a, key="team_a")
    roster_a = da.players_for_team(years, team_a, SEASON)
    labels_a = {player_label(r): r for r in roster_a.to_dicts()}
    default_a = [lbl for lbl in labels_a if lbl.startswith("Joel Embiid")]
    picks_a = st.multiselect(
        f"{team_a} sends", list(labels_a.keys()),
        default=default_a or list(labels_a.keys())[:1], key="picks_a",
    )
with col_b:
    team_b = st.selectbox("Team B", all_teams, index=idx_b, key="team_b")
    roster_b = da.players_for_team(years, team_b, SEASON)
    labels_b = {player_label(r): r for r in roster_b.to_dicts()}
    default_b = [lbl for lbl in labels_b if lbl.startswith("Ricky Rubio")]
    picks_b = st.multiselect(
        f"{team_b} sends", list(labels_b.keys()),
        default=default_b or list(labels_b.keys())[:1], key="picks_b",
    )

if team_a == team_b:
    st.warning("Pick two different teams to check a trade.")
elif not picks_a or not picks_b:
    st.info("Select at least one player on each side.")
else:
    rows_a = [labels_a[lbl] for lbl in picks_a]
    rows_b = [labels_b[lbl] for lbl in picks_b]
    pieces = tuple(
        [TradePiece(to_ps(r), team_a, team_b) for r in rows_a]
        + [TradePiece(to_ps(r), team_b, team_a) for r in rows_b]
    )
    proposal = TradeProposal(pieces=pieces)
    sheets = {
        team_a: da.cap_sheet_for_team(years, team_a, SEASON),
        team_b: da.cap_sheet_for_team(years, team_b, SEASON),
    }
    report = check_trade(sheets, proposal, ly)

    names_a = ", ".join(r["player"] for r in rows_a)
    names_b = ", ".join(r["player"] for r in rows_b)
    st.markdown(f"**{team_a}** sends *{names_a}*  ⇄  **{team_b}** sends *{names_b}*")
    for r in rows_a:
        st.caption(f"{r['player']} — {rapm_badge(r['player'], rapm_lookup)}")
    for r in rows_b:
        st.caption(f"{r['player']} — {rapm_badge(r['player'], rapm_lookup)}")

    sev = report.severity
    verdict_fn = {
        Severity.LEGAL: st.success,
        Severity.ILLEGAL: st.error,
        Severity.INDETERMINATE: st.warning,
    }[sev]
    verdict_fn(f"### OVERALL: {SEVERITY_LABEL[sev]}")

    st.markdown("**Show the work:**")
    for v in report.verdicts:
        with st.container(border=True):
            badge = SEVERITY_LABEL[v.severity]
            color = SEVERITY_COLOR[v.severity]
            st.markdown(f"**{v.team}** — :{color}[{badge}]  ·  band: `{v.band}`")
            st.write(v.reason)
            c1, c2, c3 = st.columns(3)
            c1.metric("Outgoing", f"${int(v.outgoing):,}")
            c2.metric("Incoming", f"${int(v.actual_incoming):,}")
            c3.metric("Matching ceiling", f"${int(v.max_incoming):,}")
            if v.severity is Severity.ILLEGAL:
                gap = int(v.actual_incoming) - int(v.max_incoming)
                st.error(f"Exceeds the ceiling by ${gap:,}.")
            st.caption(f"Cited rule: {v.rule}")
            if v.caveats:
                for c in v.caveats:
                    st.warning(f"Caveat: {c}")
            if v.severity is Severity.INDETERMINATE:
                st.info(
                    "INDETERMINATE means a fact we cannot observe from this "
                    "data source (likely incentives, cap-room absorption, "
                    "or dead-money attribution) could change this answer. "
                    "The engine declines to guess rather than assert a "
                    "verdict it cannot back up."
                )

st.divider()

# =============================================================================
# TABS
# =============================================================================

tab_cap, tab_impact, tab_proj, tab_opt, tab_method = st.tabs(
    ["Cap Sheets", "Impact", "Projections", "Optimizer (coming soon)", "Methodology"]
)

# --- Cap Sheets --------------------------------------------------------------
with tab_cap:
    st.subheader("Cap Sheets")
    team = st.selectbox("Team", all_teams, key="cap_team")
    sheet = da.cap_sheet_for_team(years, team, SEASON)
    roster_df = da.players_for_team(years, team, SEASON).select(
        "player", "cap_hit", "guaranteed", "needs_review"
    ).rename({
        "player": "Player", "cap_hit": "Cap Hit",
        "guaranteed": "Guaranteed", "needs_review": "Flagged for review",
    })
    apron_payroll = int(sheet.apron_payroll)
    status = ly.apron_status(apron_payroll)
    status_label = {
        "under_cap": "Under the cap",
        "over_cap": "Over the cap, under the tax",
        "taxpayer": "Taxpayer, under the first apron",
        "between_aprons": "Between the first and second apron",
        "over_second_apron": "Over the SECOND apron (hardest-capped)",
    }[status]
    status_color = {
        "under_cap": "#2e7d32", "over_cap": "#f9a825", "taxpayer": "#ef6c00",
        "between_aprons": "#d84315", "over_second_apron": "#b71c1c",
    }[status]

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**{team}** — {SEASON} · apron payroll (base salary + likely incentives, "
                     "which are usually unobserved from this source, so this is a lower bound)")
        thresholds = pl.DataFrame({
            "name": ["Salary cap", "Tax line", "1st apron", "2nd apron"],
            "value": [ly.salary_cap, ly.tax_level, ly.first_apron, ly.second_apron],
        }).to_pandas()
        bar_df = pl.DataFrame({"label": [team], "value": [apron_payroll]}).to_pandas()

        x_max = ly.second_apron * 1.12
        bar = alt.Chart(bar_df).mark_bar(cornerRadiusEnd=4, size=36).encode(
            x=alt.X("value:Q", title="Dollars", scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(format="$,.0f", grid=False)),
            y=alt.Y("label:N", title=None),
            color=alt.value(status_color),
            tooltip=[alt.Tooltip("value:Q", title="Apron payroll", format="$,.0f")],
        )
        rules = alt.Chart(thresholds).mark_rule(
            strokeDash=[4, 4], size=1.5, color="#666666"
        ).encode(x="value:Q")
        labels = alt.Chart(thresholds).mark_text(
            align="left", baseline="bottom", dx=4, dy=-2, fontSize=11, color="#444444"
        ).encode(x="value:Q", y=alt.value(8), text="name:N")
        st.altair_chart((bar + rules + labels).properties(height=120), width="stretch")
    with c2:
        st.metric("Apron payroll", f"${apron_payroll:,}")
        st.markdown(f"**Status:** :orange[{status_label}]" if "apron" in status or status == "taxpayer"
                     else f"**Status:** :green[{status_label}]")
        st.write(f"vs. cap: **${apron_payroll - ly.salary_cap:+,}**")
        st.write(f"vs. tax: **${apron_payroll - ly.tax_level:+,}**")
        st.write(f"vs. 1st apron: **${apron_payroll - ly.first_apron:+,}**")
        st.write(f"vs. 2nd apron: **${apron_payroll - ly.second_apron:+,}**")
        st.write(f"Roster count: {sheet.roster_count}")
        if not sheet.certain:
            for note in sheet.uncertainty_notes:
                st.warning(note)

    st.dataframe(
        roster_df, width="stretch", hide_index=True,
        column_config={
            "Cap Hit": st.column_config.NumberColumn(format="$%d"),
            "Guaranteed": st.column_config.NumberColumn(format="$%d"),
        },
    )
    st.caption(
        "Base salary only — likely incentives, cap holds for free agents/"
        "unsigned picks, and precise dead-money attribution are not in this "
        "source. Apron figures above are therefore a LOWER BOUND, not exact."
    )

# --- Impact -------------------------------------------------------------
with tab_impact:
    st.subheader("Impact — RAPM and Wins Added")
    if rapm_board.is_empty():
        st.warning(
            "No RAPM leaderboard found. Run `nbare export-app-data` to "
            "populate app/data/rapm_leaderboard.csv."
        )
    else:
        search = st.text_input("Search player", key="impact_search")
        board = rapm_board
        if search:
            board = board.filter(pl.col("name").str.contains(f"(?i){search}"))

        last_info = meta.get("last_season") or {}
        last_season = last_info.get("season")
        has_last = last_season is not None and "last_total" in board.columns
        window = (
            f"{meta['seasons_used'][0][:4]}–{meta['seasons_used'][-1][-2:]}"
            if meta.get("seasons_used") else "pooled"
        )

        st.caption(
            "**RAPM** = per-possession impact (points per 100 possessions vs. "
            "league average, offense + defense). **Wins Added** = total "
            "contribution: rate × playing time above replacement level, "
            f"÷ {meta.get('points_per_win', 30):g} points per win. "
            + (f"**{last_season}** columns come from a separate fit on that "
               "season alone (more current, noisier); "
               if has_last else "")
            + f"**{window}** columns pool every season (more stable). "
            "Multi-season Wins Added is a total across all those seasons, "
            "not a per-season average. Blank = did not play that season."
        )
        split = st.toggle("Show offense / defense split", key="impact_split")

        cols = ["name"]
        rename = {"name": "Player"}
        fmt = {}
        if has_last:
            cols += ["last_total"] + (["last_offense", "last_defense"] if split else [])
            rename.update({"last_total": f"RAPM {last_season}",
                           "last_offense": f"Off {last_season}",
                           "last_defense": f"Def {last_season}"})
        cols += ["total"] + (["offense", "defense"] if split else [])
        rename.update({"total": f"RAPM {window}", "offense": f"Off {window}",
                       "defense": f"Def {window}"})
        if has_last:
            cols += ["last_wins_added"]
            rename["last_wins_added"] = f"Wins Added {last_season}"
        cols += ["wins_added"]
        rename["wins_added"] = f"Wins Added {window} (total)"
        if split:
            if has_last:
                cols += ["last_possessions"]
                rename["last_possessions"] = f"Poss {last_season}"
            cols += ["possessions"]
            rename["possessions"] = f"Poss {window}"
        for c in cols[1:]:
            fmt[rename[c]] = st.column_config.NumberColumn(
                format="%d" if "possessions" in c else "%+.1f"
            )

        sort_col = "last_total" if has_last else "total"
        st.dataframe(
            board.select(cols)
            .sort(sort_col, descending=True, nulls_last=True)
            .rename({c: rename[c] for c in cols}),
            width="stretch", hide_index=True, column_config=fmt,
        )
        st.caption("Click any column header to re-sort.")

        seasons_used = ", ".join(meta.get("seasons_used", []))
        st.info(
            f"**Methodology:** Multi-season regularized RAPM ({seasons_used}) "
            f"— {meta.get('fit_type', 'unknown fit type')}, λ={meta.get('lambda', '?')}. "
            "Single-team dominance can inflate individual defensive ratings; "
            "schedule and garbage-time adjustments are in development. "
            f"Replacement level ({meta.get('replacement_level', '?')} pts/100) "
            "is an approximate placeholder, not empirically calibrated yet "
            f"— see the Methodology tab. Generated {meta.get('generated_at', '?')}."
            + (f" {last_season} columns: {last_info.get('fit_type', '?')}, "
               f"λ={last_info.get('lambda', '?')}, "
               f"{last_info.get('n_games', 0):,} games."
               if has_last else "")
        )

# --- Projections ----------------------------------------------------------
with tab_proj:
    st.subheader("Projections")
    proj = da.load_projections()
    if proj.is_empty():
        st.info(
            "**Coming soon.** The projection MODEL exists and is fully "
            "validated on synthetic data (delta-method aging curve → "
            "player/position/league partial pooling → box-score baseline), "
            "but it has not yet been run end-to-end on real multi-season "
            "data — that real-data pipeline (per-season RAPM panels, ages, "
            "position groups feeding the hierarchy) is still being built. "
            "This page will show projected RAPM 1–4 years out with an "
            "always-visible 80% posterior interval once it lands — never a "
            "bare point estimate."
        )
        status = meta.get("projections_status")
        if status:
            st.caption(f"From meta.json: {status}")
    else:
        pinfo = meta.get("projections") or {}
        players = sorted(proj["name"].unique().to_list())
        default_player = "Victor Wembanyama"
        pick = st.selectbox(
            "Player", players,
            index=players.index(default_player) if default_player in players else 0,
            key="proj_player",
        )
        pdf = proj.filter(pl.col("name") == pick).sort("season")
        cap = pinfo.get("max_projected_age")
        st.caption(
            f"{len(players):,} players projected: everyone with a measurable "
            f"{pinfo.get('projected_from', 'last-season')} rating"
            + (f", up to age {cap} — too few players have been observed past "
               f"{cap} to estimate the aging curve there, so older players "
               "(and later seasons for players nearing it) are left out, not "
               "guessed" if cap else "")
            + "."
        )

        first_season = proj["season"].min()
        nxt = proj.filter(pl.col("season") == first_season).sort("proj_total", descending=True)
        names_ranked = nxt["name"].to_list()
        if pick in names_ranked:
            st.caption(
                f"Projected #{names_ranked.index(pick) + 1} of {len(names_ranked):,} "
                f"players for {first_season} (by median total)."
            )

        chart_df = pdf.to_pandas()
        band = alt.Chart(chart_df).mark_area(opacity=0.25).encode(
            x=alt.X("season:N", title="Season"),
            y=alt.Y("lower_80:Q", title="Projected total RAPM (pts/100)"),
            y2="upper_80:Q",
            tooltip=["season", "age",
                     alt.Tooltip("proj_total:Q", format="+.1f", title="median"),
                     alt.Tooltip("lower_80:Q", format="+.1f", title="80% low"),
                     alt.Tooltip("upper_80:Q", format="+.1f", title="80% high")],
        )
        line = alt.Chart(chart_df).mark_line(point=True).encode(
            x="season:N", y="proj_total:Q",
        )
        zero = alt.Chart(chart_df).mark_rule(strokeDash=[4, 4], opacity=0.4).encode(
            y=alt.datum(0)
        )
        st.altair_chart((band + line + zero).properties(height=320), width="stretch")

        st.dataframe(
            pdf.select("season", "age", "proj_total", "lower_80", "upper_80",
                       "proj_offense", "proj_defense")
            .rename({"season": "Season", "age": "Age", "proj_total": "Total (median)",
                     "lower_80": "80% low", "upper_80": "80% high",
                     "proj_offense": "Off (median)", "proj_defense": "Def (median)"}),
            width="stretch", hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%+.1f") for c in
                           ("Total (median)", "80% low", "80% high",
                            "Off (median)", "Def (median)")},
        )
        if len(pdf) < pinfo.get("years_ahead", len(pdf)):
            st.caption(
                "Fewer seasons than requested: the aging curve is only fit up "
                f"to age {pinfo.get('age_grid', ['?', '?'])[1]}, and this "
                "engine truncates rather than extrapolating past it."
            )

        diag = pinfo.get("diagnostics", {})
        rhat = ", ".join(f"{k} {v.get('max_rhat', float('nan')):.3f}" for k, v in diag.items())
        bt = pinfo.get("backtest") or {}
        bt_text = ""
        if bt.get("offense") and bt.get("defense"):
            bt_text = (
                f"**Backtest:** refit without {bt['held_out_season']} and predict it — "
                f"the 80% band contained {bt['offense']['coverage_80']:.0%} of actual "
                f"offensive and {bt['defense']['coverage_80']:.0%} of actual defensive "
                f"{bt['held_out_season']} ratings ({bt['offense']['n_scored']:,} players). "
            )
        st.info(
            "**How to read this.** The line is the median projection of the "
            "player's underlying rating; the shaded band is its 80% posterior "
            "interval — never a bare point estimate. Model: hierarchical "
            "Bayesian aging curve (delta method, so survivorship cannot bias "
            "it), partially pooled player → position → league, with a "
            "box-score-informed baseline; offense and defense projected "
            "separately. Each season's rating enters as a measurement with "
            "its own standard error (single-season RAPM with the ridge prior "
            f"divided out), from {', '.join(pinfo.get('seasons_used', []))} "
            f"({pinfo.get('n_panel_rows', 0):,} player-seasons). "
            f"Convergence: max R-hat {rhat}. " + bt_text +
            "**Scale:** these are the projection model's estimates of "
            "underlying talent, which are shrunk far less than the Impact "
            "tab's ridge RAPM (a ridge fit pulls everyone hard toward zero; "
            "this model learns its own, weaker pooling from the data). So "
            "compare players within this tab — ranks, bands, trajectories — "
            "not against the leaderboard's numbers. "
            "**Limits:** three seasons of history (≤2 age transitions per "
            "player), so bands are wide; no retirement, injury, or "
            "playing-time modeling yet (v2)."
        )

# --- Optimizer (coming soon) -----------------------------------------------
with tab_opt:
    st.subheader("Optimizer — coming soon")
    st.write(
        "Stage 4 will find the roster moves that **maximize projected wins "
        "subject to CBA constraints** — salary-matching bands, hard-cap "
        "triggers, and apron restrictions — using the RAPM value numbers "
        "and multi-year projections already in this app as its objective "
        "and the Trade Checker's legality engine as its constraint set. "
        "Framed as a mixed-integer program (MILP) over realistic trade/"
        "signing combinations, not a greedy heuristic."
    )
    st.caption("Not built yet. This is the roadmap, not a placeholder for its own sake.")

# --- Methodology ------------------------------------------------------------
# Written for a reader (front-office analyst, hiring manager) who wants to
# know how every number on this page was produced and how far to trust it.
# Every figure below is read from meta.json, config.LEAGUE_YEARS, or the
# CBA module's own constants -- nothing is hand-typed, so the prose cannot
# drift out of sync with what the engine actually ran.


def _n(x, spec: str = ",") -> str:
    """Format a meta.json number, or '?' if the export didn't record it."""
    return format(x, spec) if isinstance(x, (int, float)) else "?"


with tab_method:
    from nbare.cba.matching import (
        BAND_1_CEILING, BAND_2_CEILING, BAND_2_FLAT_ADD, MATCHING_ALLOWANCE,
    )

    pinfo = meta.get("projections") or {}
    last_info = meta.get("last_season") or {}
    bt = pinfo.get("backtest") or {}
    seasons = meta.get("seasons_used") or []
    seasons_txt = ", ".join(seasons) if seasons else "multiple seasons"

    st.subheader("Methodology")
    st.markdown(
        """
`nbare` is a roster-construction engine built in stages:
**contracts → CBA rules → player impact (RAPM) → value → multi-year
projections → (next) a CBA-constrained optimizer.** The point is the
*system*. The impact metric is deliberately conventional, so the novel
part, which is finding legal moves that maximize projected wins, sits on
a foundation that's easy to audit.

Three rules run through everything:

1. **Validate against planted truth first.** Each computation has to
   recover a known answer on synthetic data before it touches real data.
2. **Unknown means unknown.** If a fact can't be observed from the source
   data, the engine reports `INDETERMINATE` or a lower bound. It never
   assumes the favorable case.
3. **Exact money math.** Dollars are integers or `Decimal`, never floats,
   and percentages round half-up the way the CBA does.
        """
    )

    # --- 1. Data ------------------------------------------------------------
    st.markdown("### 1. Data")
    st.markdown(
        f"""
- **Contracts:** a Basketball-Reference contracts export for {SEASON}.
  The raw file is never edited. Corrections and transactions (sign, waive,
  trade, amend) are applied as overlays on top, which keeps the original
  point-in-time record intact. Rows the parser can't fully trust are marked
  *Flagged for review* on the Cap Sheets tab.
- **Play-by-play and box scores:** nba.com, fetched through a
  rate-limited, disk-cached client. **{_n(meta.get('n_games'))} games**
  across {seasons_txt} feed the pooled fit.
- **Player bio:** per-season ages and positions from nba.com, used only by
  the projection model.
        """
    )

    # --- 2. CBA engine ------------------------------------------------------
    st.markdown("### 2. Cap sheets and trade legality")
    st.markdown(
        f"""
Cap thresholds are stored as **data keyed by league year**, not as
constants in the rule code. That's what makes it possible to check a
trade against any past season's rules. {SEASON} figures:
        """
    )
    st.dataframe(
        pl.DataFrame({
            "Threshold": ["Salary cap", "Luxury tax", "First apron", "Second apron"],
            "Amount": [ly.salary_cap, ly.tax_level, ly.first_apron, ly.second_apron],
        }),
        hide_index=True,
        column_config={"Amount": st.column_config.NumberColumn(format="$%d")},
    )
    st.markdown(
        f"""
**Salary matching (2023 CBA).** Each side of a trade is checked against
the band for its *post-trade* payroll, where S is the salary the team
sends out:

| Team after trade | Outgoing salary S | May take back up to |
|---|---|---|
| Below 1st apron | ≤ ${BAND_1_CEILING:,} | 200% of S + ${MATCHING_ALLOWANCE:,} |
| Below 1st apron | ${BAND_1_CEILING:,} – ${BAND_2_CEILING:,} | S + ${BAND_2_FLAT_ADD:,} |
| Below 1st apron | > ${BAND_2_CEILING:,} | 125% of S + ${MATCHING_ALLOWANCE:,} |
| Above 1st apron | any | 100% of S |

The bands are fixed dollar amounts in the CBA and don't scale with the
cap.

**Three-valued verdicts.** Every check returns `LEGAL`, `ILLEGAL`, or
`INDETERMINATE`, together with the band used, the dollar ceiling, and the
rule it cites. `INDETERMINATE` means the answer depends on something this
data source doesn't contain, such as likely or unlikely incentives,
cap-room absorption, dead-money attribution, or option type. The engine
states that dependency rather than guessing.

**Apron payroll is a lower bound.** The source has base salary only.
Likely incentives, cap holds, and exact dead money are missing, so every
payroll figure shown is a floor, labeled as one.
        """
    )

    # --- 3. Impact ----------------------------------------------------------
    st.markdown("### 3. Player impact: RAPM")
    st.markdown(
        f"""
**Step 1: Rebuild every lineup from play-by-play.** The play-by-play
records substitutions but not who is on the floor. The engine works out
the five players per team for every stretch between substitutions (a
*stint*). Two safeguards protect this step:
- **Allowlist filter.** An ID is accepted as a player only if it appears
  in the real player table. Team IDs and replay or event codes that leak
  into player slots get removed before reconstruction.
- **Minutes gate.** Each player's reconstructed floor time must match the
  official box-score minutes within 90 seconds. A game that fails is
  **excluded from the fit** rather than trusted. This is what catches the
  classic reversed-substitution bug.

After the gate: **{_n(meta.get('n_stints'))} stints** from
{_n(meta.get('n_games'))} games.

**Step 2: Possessions and the design matrix.** Possessions are estimated
per stint and split by which team had the ball, so each player gets an
**offensive** and a **defensive** coefficient. Each row is weighted by its
possessions.

**Step 3: Ridge regression.** Plain adjusted plus-minus is unstable:
teammates who share the floor can get offsetting extreme ratings that
just fit noise. Ridge regression adds a penalty that pulls ratings toward
a prior unless the data says otherwise. The penalty strength λ is chosen
by **cross-validation grouped by game**. A random split would put stints
from the same game in both train and test (same lineups, pace, and
referees), which leaks information and picks too little shrinkage.
Selected λ = **{_n(meta.get('lambda'), ',.0f')}**.

**Step 4: A box-score prior instead of zero.** Standard ridge pulls every
low-minute player to league average. Here, players are pulled toward
what their **box score predicts**:
- First, fit plain ridge. Then, on players with at least 1,000
  possessions, regress rating on box-score features. Offense uses
  points, assists, and turnovers per 100 possessions plus true shooting.
  Defense uses steals, blocks, defensive-rebound rate, and a big/non-big
  flag, because steals and blocks alone misrank positional defenders.
- That prediction becomes each player's prior mean. A player with 50
  possessions ends up close to his box-score estimate. A player with
  thousands is driven by his on-court results.
        """
    )
    st.latex(r"\hat\beta = \arg\min_\beta \; \|W^{1/2}(X\beta - y)\|^2 + \lambda\|\beta - \mu_{\text{box}}\|^2")
    st.markdown(
        f"""
**Two fits are shown.** The pooled fit over {seasons_txt} is the most
stable. A separate single-season fit on {last_info.get('season', 'the last season')}
({_n(last_info.get('n_games'))} games) is more current and noisier.

**Step 5: From rate to wins.** RAPM is a per-100-possession rate.
*Wins Added* turns it into a volume number by weighting each side by that
side's own possessions above replacement level R:
        """
    )
    st.latex(
        r"\text{Value} = \frac{(\text{off} - \tfrac{R}{2})\cdot\text{poss}_{off}"
        r" + (\text{def} - \tfrac{R}{2})\cdot\text{poss}_{def}}{100},"
        r"\qquad \text{Wins Added} = \frac{\text{Value}}{"
        + f"{_n(meta.get('points_per_win'), 'g')}" + r"}"
    )
    st.markdown(
        f"""
R = **{_n(meta.get('replacement_level'), 'g')}** pts/100, which is half a
point below BPM's published convention. It is a **placeholder and has not
been calibrated** against this project's own rating distribution. The
offense/defense split matters here. An earlier version multiplied total
RAPM by combined possessions, which counted every possession twice and
inflated wins by about 2×. A unit test now pins the corrected formula.
        """
    )

    # --- 4. Projections -----------------------------------------------------
    st.markdown("### 4. Multi-year projections")
    st.markdown(
        f"""
The projection model is a hierarchical Bayesian model (numpyro, NUTS
sampler), fit separately for offense and defense. It was built and tested
one layer at a time:

1. **Aging curve by the delta method.** A naive regression of rating on
   age suffers from survivorship bias: weak players leave the league
   early, so the players still around at 35 look good. The delta method
   compares each player only with himself one season later (age *a* to
   *a*+1). Survivorship then decides which deltas exist but not what they
   say. Gap years are skipped, not interpolated.
2. **Partial pooling, player → position → league.** Each player's curve
   is shrunk toward his position group's curve, and each group's toward
   the league curve. How much to shrink is learned from the data.
3. **Box-score baseline.** Like the RAPM prior, it gives thin-sample
   players a sensible starting level.

**No double-counting.** A ridge rating already includes its prior, so
passing it in as if it were a raw measurement would count that prior
twice. When this happened in development, it flattened every aging curve.
Each season's rating therefore enters with the **ridge prior divided
out**: an unbiased measurement with its own standard error.

**This run:** {_n(pinfo.get('n_panel_rows'))} player-seasons from
{', '.join(pinfo.get('seasons_used') or []) or '?'};
{_n(pinfo.get('n_players_projected'))} players projected up to
{_n(pinfo.get('years_ahead'))} seasons ahead. The sampler used
{_n((pinfo.get('sampler') or {}).get('num_chains'))} chains, and every
parameter's R-hat is ≤ 1.01. Projections stop at age
{_n(pinfo.get('max_projected_age'))} because too few players have been
observed beyond that to estimate the curve. The model leaves those
seasons out rather than extrapolating.
        """
    )
    if bt.get("offense") and bt.get("defense"):
        st.markdown(
            f"""
**Out-of-sample backtest.** The model was refit without
{bt['held_out_season']} and used to predict that season. The 80% intervals
contained **{bt['offense']['coverage_80']:.0%}** of actual offensive
ratings and **{bt['defense']['coverage_80']:.0%}** of actual defensive
ratings ({bt['offense']['n_scored']:,} players), close to the 80% target.
Calibration is judged on held-out data because in-sample coverage looks
good almost automatically when each player has about as many parameters
as observations.
            """
        )

    # --- 5. Validation ------------------------------------------------------
    st.markdown("### 5. How it's validated")
    st.markdown(
        """
Each layer has a synthetic generator with a planted answer, and the test
suite asserts that the layer recovers it:

| Component | Planted truth it must recover |
|---|---|
| Stint reconstruction | exact lineups and minutes, including tricky substitution orderings |
| Stints → design matrix connector | exact scoring per lineup |
| Ridge RAPM | planted offense/defense ratings (correlation > 0.95) |
| Box-score prior | planted box-score → rating relationship |
| Delta-method aging curve | a planted curve under survivorship-biased attrition |
| Hierarchical model | planted player/position effects with known measurement error |
| Salary matching | hand-computed CBA band edge cases, in exact dollars |

Building one layer at a time localizes bugs. For example, a connector bug
in Stage 2 was caught by its own synthetic test before it could show up
as strange RAPM values.
        """
    )

    # --- 6. Limitations -----------------------------------------------------
    st.markdown("### 6. Known limitations")
    st.markdown(
        f"""
- **Single-team defensive confounding.** A player who has played nearly
  all his minutes with one core group is hard to separate from that
  team's scheme, which can inflate individual defensive ratings.
- **No schedule or garbage-time adjustment yet.**
- **Replacement level ({_n(meta.get('replacement_level'), 'g')}) and
  {_n(meta.get('points_per_win'), 'g')} points per win** are conventions,
  not calibrated values.
- **Projections have short histories** (at most two age transitions per
  player so far), so the intervals are wide. They don't model
  retirement, injury, or playing time yet.
- **Projection scale differs from the Impact tab.** The projection model
  shrinks much less than ridge, so compare players within each tab, not
  across tabs.
- **Cap data is base salary only**, so apron figures are lower bounds.
        """
    )

    # --- 7. Next ------------------------------------------------------------
    st.markdown("### 7. What's next")
    st.markdown(
        """
- **Stage 4, the roster optimizer.** A mixed-integer program that
  searches trades and signings for the moves that maximize projected
  wins. The constraints are the same salary-matching, apron, and
  hard-cap rules the Trade Checker enforces, and the objective uses the
  projection distributions above.
- **Stage 5, agent layer and evaluation harness.** A natural-language
  front end over the engine, with an evaluation suite that checks every
  suggested move for legality.
        """
    )

    st.caption(
        f"Pooled RAPM generated {meta.get('generated_at', '?')[:10]} · "
        f"projections generated {(pinfo.get('generated_at') or '?')[:10]} · "
        f"source: [{REPO_URL.split('//')[-1]}]({REPO_URL})"
    )
