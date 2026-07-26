# nbare — NBA Roster Construction Engine

A decision system for NBA roster building under the 2023 CBA: given a team's
cap sheet, find the *legal* moves that maximize projected wins, and explain
what each one costs in future flexibility.

Most public NBA analytics projects predict outcomes. This one models
**decisions under constraints** — which is the actual job.

## Why the aprons make this hard

The 2023 CBA replaced soft penalties with hard caps. For 2026-27 the league
set the salary cap at $164.961M, the tax level at $200.428M, the first apron
at $209.015M, and the second apron at $221.686M.

Those aprons are not tax brackets — they are hard ceilings that lock the
moment you trigger them:

| Level | Triggered by | Effect |
|---|---|---|
| First apron ($209.015M) | Sign-and-trade acquisition; signing above the taxpayer MLE; any BAE use; MLE-based trade or waiver claim; expanded TPE use; use of a trade exception generated before the 2026 offseason | Hard cap for the season |
| Second apron ($221.686M) | Any MLE use; aggregating two or more salaries in a trade; sending cash in a trade; using a signed-and-traded player to take back salary | Hard cap, plus pick freezing |

Encoding this correctly is a mixed-integer program with real business logic.
Almost nobody builds it, because it requires reading the CBA rather than
downloading a CSV. That is the moat.

## Architecture

```
Stage 0  data foundation      DuckDB warehouse, cached ingestion    <- YOU ARE HERE
Stage 1  CBA rule engine      legality + hard-cap triggers          <- ship publicly on its own
Stage 2  impact metric        Bayesian RAPM from possession stints
Stage 3  projections          hierarchical aging curves, censoring-aware
Stage 4  optimizer            MILP over trades/signings under Stage 1 constraints
Stage 5  agent layer          monitor -> analyze -> memo -> critic, with eval harness
Stage 6  interface            Next.js app + methodology writeups
```

The layering matters: **the LLM never computes anything.** Stage 5 agents
orchestrate and explain; Stages 1 and 4 compute. Say this loudly in any
writeup — it is what separates a serious system from a demo.

## Stage 0 — what's built

```
src/nbare/
  config.py              league-year cap figures as DATA, not constants
  domain/money.py        exact integer dollar math (floats are banned)
  warehouse/
    schema.sql           raw / stg / mart layering
    db.py                DuckDB connection + idempotent schema
  ingest/
    client.py            rate-limited, disk-cached, retrying nba_api wrapper
    nba_stats.py         players, teams, games, play-by-play
  crosswalk/
    build.py             NBA.com <-> BBRef <-> Spotrac id resolution
    overrides.yaml       version-controlled manual fixes (always win)
  cli.py                 `nbare` command
tests/test_stage0.py     35 tests incl. doctests
```

### Two design decisions worth defending in an interview

**1. Cap figures are league-year data.** Every rule in Stage 1 will be written
relative to a `LeagueYear` object, never against hardcoded 2026-27 numbers.
This is what makes the Stage 1 validation suite possible: you replay real
2025-26 trades against 2025-26 thresholds. Hardcode the constants and you
can never backtest, which means you can never prove the engine is right.

**2. The cache is the deliverable.** A 12-season play-by-play backfill is
roughly 15,000 requests. At the 0.75s floor that is about three hours, and
stats.nba.com will drop you if you go faster. Every response is written to
disk keyed by `(endpoint, params)`, so parser changes never trigger a
refetch and a backfill that dies at request 11,000 resumes for free.
Back up `data/cache`. Losing it costs hours; losing the DuckDB file costs
minutes.

## Runbook

```bash
pip install -e ".[dev,scrape]"
make test

make teams
make players
make games SEASON=2025-26        # game index
make pbp   SEASON=2025-26        # ~20 min/season, resumable
make status                      # row counts + cache size
```

Backfill order matters: `games` populates the `game_id` list that `pbp`
iterates. Run seasons oldest-first so a partial backfill still gives you a
contiguous training window.

## Where the time actually goes

Two silent time sinks, both of which produce no visible progress:

- **Contract scraping.** Spotrac encodes player options, team options, ETOs,
  and partial guarantees in table *footnotes*. Those footnotes are what
  determine whether a salary is tradeable or a cap hold. Parse them or
  Stage 1 is built on sand.
- **Lineup reconstruction (Stage 2).** Rebuilding who is on the floor from
  substitution events is error-prone in ways the regression will not reveal.
  `nbare check-minutes` exists as a hard gate: reconstructed minutes must
  match the official box score. Do not start the RAPM work until it passes.

Timebox both. Imperfect data that moves forward beats a perfect pipeline
that never reaches Stage 4.

## Contract data: what the BBRef export can and cannot support

`nbare ingest-contracts <csv>` parses a Basketball-Reference contract
export. It carries the **BBRef slug in the last column**, which is the
single most valuable field in the file: it seeds the crosswalk exactly and
removes fuzzy matching entirely. That matters, because on this player pool
fuzzy matching merges *Nikola Jokic* with *Nikola Jovic* at 0.92
similarity, and *Mouhamadou Gueye* (CHA) with *Mouhamed Gueye* (ATL).

**Guarantee structure is partially recoverable.** Walking the salary
columns until the running total equals `Guaranteed` recovers the guarantee
cliff for 402 of 413 rows. Jokic 2027-28, Tatum 2029-30, and Embiid 2028-29
all correctly come back non-guaranteed. The 11 that don't decompose are
partial guarantees or stretched dead money, and are flagged rather than
guessed.

**What the file cannot support, and why Stage 1 must refuse:**

| Missing | Consequence |
|---|---|
| Option type (player / team / ETO) | Non-guaranteed and "player option" are different cap holds and different trade rules. Stored as NULL, not False. |
| Likely incentives | Apron payroll *includes* them. Every apron figure from this source is a **lower bound**. |
| Dead-money attribution | Lillard appears on MIL and POR with identical salary rows; Beal on PHX and LAC. Naive summing double counts. |
| Cap holds / empty roster charges | 8 teams sit below the 14-man minimum — this is a contract list, not a roster. |
| Two-way flags, trade bonuses, no-trade clauses, Bird rights | All required for Stage 1 legality checks. |

So the aggregate is deliberately named `base_salary_total`, **not**
`apron_payroll`. 35 contract-years across 10 teams carry a `needs_review`
flag; Stage 1 should decline to certify a payroll for any team containing
one rather than emit a confident wrong number.

## Status / known gaps

- [x] Warehouse schema, ingestion client, CLI, test suite
- [x] BBRef contract parser + data-quality report + crosswalk seed (444 slugs)
- [ ] `config.LEAGUE_YEARS` — the 2026-27 cap/tax/apron/MLE figures are the
      official league release, but the **BAE and the 2025-26 secondary
      figures are unverified placeholders**. Confirm against the league
      press release before any Stage 1 rule depends on them.
- [ ] Contract + transaction scrapers (Spotrac / Basketball Reference)
- [ ] Box score ingestion (needed for the minutes validation gate)
- [ ] Stage 1 rule engine
