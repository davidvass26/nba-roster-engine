# nbare Streamlit app (v1)

A deployable, resume-ready web app that is an honest window onto the engine.
Goal: easy to interact with, all the info present, not perfect. Single
Streamlit app under `app/`, deployed free to Streamlit Community Cloud.

## Design principles (these are not optional polish — they ARE the point)

- **Lead with the rarest asset.** The landing view is the TRADE CHECKER, not
  a leaderboard. RAPM leaderboards are everywhere; a working CBA trade-
  legality engine is rare. A visitor should be able to check a trade in the
  first ten seconds without reading anything.
- **Show the reasoning, not just the answer.** Every verdict and number
  shows its work. This is what separates a credible tool from a toy.
- **Caveats are features, not apologies.** Surface limitations honestly and
  prominently; it reads as expertise.
- **Do not over-design.** Streamlit's clean default is professional. No
  splash screens, no flashy theming. Functional and honest beats slick.

## Data contract

The app reads ONLY:
- the live contract CSV (`data/raw/bbref_contracts_2026-27.csv`) for cap
  sheets and trade legality — computed live so editing the CSV updates the
  app,
- the exported files in `app/data/` (`rapm_leaderboard.csv`,
  `projections.csv`, `meta.json`) for impact and projections.

It must NOT require the DuckDB warehouse or the cache (they won't exist in
the cloud).

### Auto-refresh on CSV change (required)

The cap sheet and trade checker must reflect edits to the contract CSV on a
browser refresh. Use `@st.cache_data` keyed on the CSV's file modification
time (mtime), so editing the CSV invalidates the cache and a refresh
recomputes. Do not cache on a static key — that would serve stale data.

## Structure

Landing: **Trade Checker**
Tabs: **Cap Sheets** · **Impact** · **Projections** · **Optimizer (coming
soon)** · **Methodology**

### 1. Trade Checker (landing)

- Two multiselects (or team-then-player pickers) to choose players going each
  way. Default to a pre-filled interesting example so the page is alive on
  load (e.g. a star-for-star that returns ILLEGAL, so the first thing a
  visitor sees is the engine doing something non-trivial).
- Include players RAPM and Wins Added rankings
- Big, clear verdict: LEGAL (green) / ILLEGAL (red) / INDETERMINATE (amber).
- Below the verdict, SHOW THE WORK: the CBA rule/band that applied, the
  outgoing vs incoming salary, the matching ceiling, and the exact dollar gap
  if illegal. Cite the provision. Make the INDETERMINATE case visible and
  explained ("cannot verify without incentive data") — do not hide it; it's a
  sophistication signal.

### 2. Cap Sheets

- Pick a team; show its roster with salaries, sorted.
- Show payroll relative to the FOUR thresholds (cap, tax, first apron, second
  apron) — draw them as reference lines/markers and color the team's position.
  The aprons are the story; make them prominent.
- Show apron status label and how far over/under each line.

### 3. Impact (RAPM rate + value)

- Two leaderboards side by side (or toggle): RATE (by total RAPM, with
  offense/defense split) and VALUE (by value / wins-added, with possessions).
- One-line explainer between them: "Rate = per-possession quality; Value =
  total contribution (rate x volume above replacement)."
- Sortable; searchable by player.
- A prominent methodology note: "Multi-season regularized RAPM (2023–26) with
  a box-score-informed prior. Single-team dominance can inflate individual
  defensive ratings; schedule and garbage-time adjustments are in
  development. Replacement level is an approximate placeholder." Pull the
  seasons/lambda/etc from meta.json so it's honest and auto-updating.

### 4. Projections

- Pick a player; show projected RAPM 1–4 years out.
- ALWAYS show the 80% interval as a band, never a bare point estimate. The
  interval is the sophistication — make it visually prominent.
- Note it's a hierarchical Bayesian aging-curve model with a box-score
  baseline.

### 5. Optimizer (coming soon)

- Not a blank placeholder. A short description of what it WILL do: "find the
  roster moves that maximize projected wins subject to CBA constraints
  (aprons, matching rules)." Frame as roadmap/ambition. This shows vision.

### 6. Methodology (prominent, not buried)

This is the highest-leverage text in the app — it converts interest into
respect. A few honest paragraphs, real story not marketing:
- What the engine does end to end (contracts -> CBA rules -> RAPM -> value ->
  projections).
- How it's validated: synthetic-first (every computation recovers planted
  ground truth before real data); RAPM recovers planted ratings at ~0.99
  correlation; the CBA engine returns INDETERMINATE rather than guess.
- What's real vs approximate vs next: real cap/trade/RAPM; approximate
  replacement level and the known single-team defensive confounder;
  optimizer next.
- Link to the GitHub repo.

Tone: confident, restrained, honest. "Validated RAPM," never "revolutionary
metric." The restraint is the signal.

## Deployment

- App at `app/streamlit_app.py`, importing the engine (`nbare.*`) directly
  for the live cap/trade computations and reading `app/data/*` for impact and
  projections.
- Add a `requirements.txt` or ensure `pyproject` install works on Streamlit
  Cloud (it needs the engine importable; the heavy model deps like numpyro
  are NOT needed at app runtime since projections are precomputed — keep the
  cloud install lean: it needs duckdb/polars/pydantic/etc for cap/trade, plus
  streamlit, but not numpyro/jax).
- Deploy from GitHub to Streamlit Community Cloud; produce a public URL.

## Do not

- Do not require the warehouse or cache at runtime.
- Do not show bare player ids (use names).
- Do not present point-estimate projections without intervals.
- Do not oversell; every claim must be defensible.