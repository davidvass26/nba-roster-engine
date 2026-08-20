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
    return (
        f"RAPM {r['total']:+.1f} (off {r['offense']:+.1f} / def {r['defense']:+.1f})"
        f"  ·  Wins Added {r['wins_added']:+.1f}"
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
    st.subheader("Impact — RAPM rate + value")
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

        st.caption(
            "**Rate** = per-possession quality (points per 100 possessions "
            "above average). **Value** = total contribution "
            "(rate × volume above replacement)."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rate — by total RAPM**")
            st.dataframe(
                board.select("name", "total", "offense", "defense")
                .sort("total", descending=True)
                .rename({"name": "Player", "total": "Total", "offense": "Off", "defense": "Def"}),
                width="stretch", hide_index=True,
                column_config={
                    "Total": st.column_config.NumberColumn(format="%+.1f"),
                    "Off": st.column_config.NumberColumn(format="%+.1f"),
                    "Def": st.column_config.NumberColumn(format="%+.1f"),
                },
            )
        with c2:
            st.markdown("**Value — by value / wins added**")
            st.dataframe(
                board.select("name", "value", "wins_added", "possessions")
                .sort("value", descending=True)
                .rename({"name": "Player", "value": "Value", "wins_added": "Wins Added",
                          "possessions": "Possessions"}),
                width="stretch", hide_index=True,
                column_config={
                    "Value": st.column_config.NumberColumn(format="%+.0f"),
                    "Wins Added": st.column_config.NumberColumn(format="%+.1f"),
                    "Possessions": st.column_config.NumberColumn(format="%d"),
                },
            )

        seasons_used = ", ".join(meta.get("seasons_used", []))
        st.info(
            f"**Methodology:** Multi-season regularized RAPM ({seasons_used}) "
            f"— {meta.get('fit_type', 'unknown fit type')}, λ={meta.get('lambda', '?')}. "
            "Single-team dominance can inflate individual defensive ratings; "
            "schedule and garbage-time adjustments are in development. "
            f"Replacement level ({meta.get('replacement_level', '?')} pts/100) "
            "is an approximate placeholder, not empirically calibrated yet "
            f"— see the Methodology tab. Generated {meta.get('generated_at', '?')}."
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
        players = sorted(proj["name"].unique().to_list())
        pick = st.selectbox("Player", players, key="proj_player")
        pdf = proj.filter(pl.col("name") == pick).sort("season").to_pandas()
        band = alt.Chart(pdf).mark_area(opacity=0.25).encode(
            x=alt.X("season:N", title="Season"),
            y=alt.Y("lower_80:Q", title="Projected total RAPM"),
            y2="upper_80:Q",
        )
        line = alt.Chart(pdf).mark_line(point=True).encode(
            x="season:N", y="proj_total:Q",
        )
        st.altair_chart((band + line).properties(height=320), width="stretch")
        st.caption("Shaded band is the 80% posterior interval — the point estimate alone is never shown.")

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
with tab_method:
    st.subheader("Methodology")
    st.markdown(
        f"""
`nbare` is an NBA roster-construction engine: **contracts → CBA rules →
RAPM → value → projections**, built so that every number on this page can
be traced back to a computation, not an eyeballed guess.

**How it's validated.** Every nontrivial computation in this engine is
proven against a planted, synthetic ground truth before it ever touches
real data — the stint reconstruction recovers planted lineups exactly, the
box-score connector recovers planted scoring exactly, and RAPM recovers
planted player ratings at roughly **0.99 correlation** on synthetic games
with known answers. "Produces plausible output" is never treated as a
substitute for that.

**Why INDETERMINATE is a feature.** The CBA salary-matching engine returns
three-valued verdicts — `LEGAL` / `ILLEGAL` / `INDETERMINATE` — not two.
When a fact the rules need (likely incentives, exact dead-money
attribution, an option type) is not observable from the contract source,
the engine says so rather than silently assuming the favorable case. An
honest "I can't verify this" beats a confident wrong answer.

**What's real:**
- Cap sheets and trade legality run live against the actual 2023 CBA
  salary-matching bands, computed from a real Basketball-Reference
  contracts export.
- RAPM is a real ridge-regression fit (grouped cross-validation on
  {meta.get('n_games', '?'):,} games / {meta.get('n_stints', '?'):,} stints
  across {', '.join(meta.get('seasons_used', []) or ['multiple seasons'])})
  on real play-by-play, gated so any game whose lineup reconstruction fails
  a minutes-accuracy check is excluded from the fit entirely, not silently
  trusted.

**What's approximate:**
- **Replacement level** ({meta.get('replacement_level', '?')} pts/100) is a
  configurable placeholder in the neighborhood of established box-score
  metrics' conventions, not yet empirically calibrated against this
  project's own rating distribution.
- **Single-team defensive confounder**: a player who has spent almost all
  his minutes with the same small set of teammates is hard for ridge
  regression to separate from his team's system, which can inflate
  individual defensive ratings. Schedule and garbage-time adjustments that
  would help are in development, not shipped.
- Apron payroll from the contract source is a **lower bound** — likely
  incentives, cap holds for free agents and unsigned picks, and precise
  dead-money attribution are not in this data source.

**What's next:** a MILP roster optimizer (Stage 4) that finds legal moves
maximizing projected wins, and a real-data run of the projections pipeline
described above (currently validated on synthetic data only).

Source: [{REPO_URL.split('//')[-1]}]({REPO_URL})
        """
    )
