# CLAUDE.md — working notes for Claude Code

Guidance for AI assistants working in this repo. Read this before making
changes. The rules below are not style preferences; they encode decisions
that took real debugging to get right, and violating them reintroduces bugs
this project already fixed.

## What this project is

`nbare` is an NBA roster-construction engine. The end goal is a system that,
given a team's cap sheet, finds the *legal* roster moves that maximize
projected wins under the 2023 CBA. It is built in stages:

- **Stage 0** — data warehouse (DuckDB), cached ingestion, contract parsing
- **Stage 1** — CBA rule engine (salary matching shipped; hard-cap triggers pending data)
- **Stage 2** — Bayesian RAPM impact metric (ridge pipeline shipped; box-score prior pending)
- **Stage 3** — multi-year player projections (not started)
- **Stage 4** — MILP roster optimizer (not started)
- **Stage 5** — agent layer + eval harness (not started)

The differentiator is the *system*, not any single metric. The RAPM is
deliberately conventional so the novel part (CBA-constrained optimization)
rests on a foundation no one questions.

## Non-negotiable principles

These are the project's spine. When a change would violate one, stop and
flag it rather than working around it.

1. **Honesty over false completeness.** The engine must say "I can't verify
   this" rather than emit a confident wrong number. Unknown facts are
   modeled as `None`/`INDETERMINATE`, never guessed. Examples already in the
   code: `option_type` is `None` when unknown (not `False`); trade verdicts
   are three-valued (`LEGAL`/`ILLEGAL`/`INDETERMINATE`); cap aggregates are
   named `base_salary_total`, not `apron_payroll`, because incentives are
   missing. Preserve this. Do not add code paths that assume the favorable
   case to make output look complete.

2. **Validate against known ground truth, synthetic-first.** Every
   nontrivial computation is proven by recovering a planted answer before it
   touches real data. RAPM recovers planted player ratings; the stint
   builder recovers planted lineups; the connector recovers planted scoring.
   When adding a computation, add a synthetic generator with known truth and
   assert recovery. "Produces plausible output" is not a test.

3. **Exact money math.** Never use `float` for dollars. Use `domain.money`
   (`Money`, `pct_of`, `parse_dollars`). `pct_of` rounds half-UP via
   `Decimal` because the CBA does; Python's `round()` is banker's rounding
   and is wrong here. `Money(1.0)` raises on purpose.

4. **Rules are league-year-relative.** CBA thresholds live in
   `config.LEAGUE_YEARS` keyed by season. Rule code takes a `LeagueYear`; it
   never hardcodes a dollar figure. This is what makes backtesting against
   past seasons possible. Do not inline cap numbers into logic.

5. **Raw data is immutable; changes are overlays.** The contract CSV and the
   nba.com cache are never edited. Corrections and transactions go through
   `ingest/transactions.py` (sign/waive/trade/amend) applied on top. This
   preserves the point-in-time record.

6. **The cache is precious; derived data is disposable.** `data/cache/`
   (nba.com responses) represents hours of rate-limited fetching and is
   content-addressed and immutable-safe. The DuckDB warehouse is rebuildable
   from it in minutes. Never add logic that invalidates or rewrites cache
   entries for completed games — finished games don't change.

## Workflow rules

- **Run the relevant SKILL.md first** if one applies (document/spreadsheet
  creation, etc.). For pure Python module work in `src/`, no skill applies.
- **Tests must pass before committing.** `make test` (currently 119 tests).
  Commit green so every point in history is a working state.
- **Add tests with every behavior change.** This repo's credibility is its
  test suite. A change without a test is incomplete.
- **Keep prose docstrings that explain *why*.** The modules have long
  header docstrings explaining the reasoning and the traps. Preserve and
  extend them; they are the difference between this reading as a serious
  project and a tutorial. Do not strip them for brevity.
- **Never invent NBA facts.** Contracts, cap figures, transactions, and CBA
  rules must come from a real source (the CSV, a verified web search, the
  league release). If a number can't be sourced, flag it as approximate in a
  comment and mark the row `needs_review`, as the contract parser does.

## Environment

- Python 3.10+ (runs on 3.12; `from __future__ import annotations`
  everywhere so `X | None` hints work on 3.10).
- Install: `pip install -e ".[dev]"` (RAPM stack included). The Bayesian
  prior stack is `pip install -e ".[dev,model]"` (adds numpyro/arviz).
- Package is `src/`-layout; import as `nbare.*`.
- **nba.com is only reachable from your local machine, not from CI or
  sandboxes.** All logic is therefore validated on synthetic data; real
  ingestion (`make games`, `make pbp`) runs locally only.

## Module map

\`\`\`
src/nbare/
  config.py            LeagueYear cap figures (data, not constants); paths
  domain/
    money.py           exact integer/Decimal dollar math — floats banned
    models.py          CapSheet, PlayerSalary, TradeProposal (Stage 1 inputs)
  warehouse/
    schema.sql         raw / stg / mart layering
    db.py              DuckDB connection, idempotent schema
  ingest/
    client.py          rate-limited, disk-cached, retrying nba_api wrapper
    nba_stats.py       parse nba.com payloads; idempotent upsert
    contracts.py       BBRef contract CSV parser + data-quality report
    transactions.py    sign/waive/trade/amend overlays on the immutable base
  crosswalk/
    build.py           nba.com <-> BBRef <-> Spotrac id resolution
  cba/
    matching.py        CBA salary-matching rules; 3-valued verdicts
  rapm/
    stints.py          reconstruct on-court lineups from play-by-play
    possessions.py     possession estimation, split by offensive team
    design.py          sparse offense/defense design matrix
    fit.py             ridge RAPM with grouped (by game) cross-validation
    blocks.py          connector: stints -> offense blocks for the design
    synthetic.py       ground-truth game generators for validation
  cli.py               `nbare` command (typer)
\`\`\`

## Common commands

\`\`\`bash
make test                                  # full suite
nbare check-minutes --synthetic            # prove stint logic, no data
nbare ingest-contracts data/raw/<csv>      # contract data-quality report
nbare check-trade <csv> --send X --receive Y   # salary-matching legality
nbare apply-overrides <csv> <overlay.yaml> --team PHI
nbare fit-rapm --season 2024-25            # needs a local backfill first
\`\`\`

## Gotchas already fixed (do not reintroduce)

- **`__len__` makes empty objects falsy.** `NBAStatsCache` defines `__len__`,
  so `cache or NBAStatsCache()` silently discarded a passed-in empty cache.
  Use identity checks (`if x is None`) with any object that defines `__len__`.
- **`Rk` is a display index, not row identity.** Deduping contract rows on
  all columns misses true duplicates (a player at two `Rk` values). Dedupe
  on content columns only.
- **Grouped CV, not random CV, for RAPM lambda.** Stints from one game share
  lineups/pace; random splits leak and pick too-small lambda. Split on
  `game_id`.
- **Reversed substitution direction** is the classic stint bug. The minutes
  gate (`validate_minutes`) exists to catch it — do not build RAPM on a
  reconstruction that hasn't passed it.