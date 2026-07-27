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

## Stage 1 — CBA salary-matching engine (shipped, scoped)

`nbare check-trade <csv> --send <slug> --receive <slug>` validates the
salary-matching legality of a swap and prints a per-team verdict citing the
specific CBA band.

**Verified rules (2023 CBA, fully phased in from 2024-25).** The matching
bands are fixed dollar figures in the CBA -- they do NOT scale with the cap
-- so 2026-27 uses the same bands as 2024-25. For a team below the first
apron after the trade, sending out salary S:

| Outgoing S | May take back |
|---|---|
| S ≤ $7.5M | 200% of S + $250K |
| $7.5M < S ≤ $29M | S + $7.5M |
| S > $29M | 125% of S + $250K |

A team **above** either apron is held to a flat **100%** of outgoing, with
no aggregation benefit. That single rule is why apron teams can't trade a
$33M player for a $39M player: 100% matching caps the return at what they
send out.

**These ceilings are validated against real published worked examples** and
locked in as regression tests: Tre Mann ($3,191,400 → $6,632,800), Oladipo
($9,450,000 → $16,950,000), Wiggins (~$26.3M → ~$33.8M). Both band
boundaries ($7.5M, $29M) are tested for off-by-one. All money math is exact
integer/Decimal — no float ever touches a cap figure.

**Three-valued verdicts, by design.** A check returns LEGAL, ILLEGAL, or
**INDETERMINATE**. The third is not a failure — it means a fact the current
data can't supply (likely incentives, cap-room absorption, dead-money
attribution) could change the answer, so the engine declines to certify
rather than guess. A team sitting within $5M of an apron with unobserved
incentives is automatically downgraded from LEGAL to INDETERMINATE. This is
the honesty property from Stage 0 carried into the rules: an engine that
says "I can't verify this" is more credible to a front office than one that
always answers.

**Scope boundary.** This module does salary matching only, because that is
what base salaries alone can support. It does NOT yet compute which apron a
move hard-caps a team into, because that needs exact apron payroll
(including likely incentives) and dead-money attribution — both flagged
missing in the contract-data section above. Those are the next data
unblocks, not rule-engine work.

## Updating the data: transaction overlays

The contract CSV is a snapshot and the league doesn't stop moving. Rather
than hand-edit the CSV (which loses your point-in-time record and invites
errors like leaving a player on two teams), changes live in a small,
ordered YAML overlay applied on top of the immutable base:

```bash
nbare apply-overrides data/raw/bbref_contracts_2026-27.csv \
    data/overrides/phi_2026_offseason.yaml --team PHI
```

Overlays are transaction logs, not find-and-replace, because the CBA treats
each move type differently and the rule engine already knows the
difference:

| Type | Meaning |
|---|---|
| `sign` | a free agent joins a team at a stated **cap hit** (not headline salary) |
| `waive` | a player leaves; `stretch_dead_money: true` keeps guaranteed money as a dead-money row |
| `trade` | players (and cash) move between teams |
| `amend` | override one field of one row (escape hatch) |

The base is never mutated. Every action is logged, and anything suspicious
(waiving a player who isn't on the named team, amending a row that doesn't
exist) becomes a **warning** rather than silently applying — a wrong
overlay that runs clean is worse than one that complains.

The shipped example, `phi_2026_offseason.yaml`, encodes the real 2026
Philadelphia moves and is instructive about why structure matters: LeBron
**signed as a free agent** (a `sign`, and he wasn't even in the snapshot
because free agents have no contract row — the system caught this with a
warning when an `amend` no-op'd), and KCP was **bought out and re-signed at
the vet minimum** — his cap hit is ~$2.4M, not the $3.9M headline. Applying
it shows Philadelphia at ~$261.6M with 16 players: over the second apron and
over the roster max, i.e. a roster that can't legally exist yet, which is
exactly what the reporting says. You model the move; the engine tells you
what else has to happen.

## Stage 2 — lineup reconstruction + the minutes gate

RAPM needs one design-matrix row per *stint* (a span where the 10 players
on the floor don't change), so the foundation is reconstructing on-court
lineups from play-by-play. The hard part: PBP gives you substitution
*events*, not lineups, and a player can be on the floor for minutes before
appearing in any event. The reconstructor infers each period's opening five
per side (players who act before their first sub-in), then walks the subs.

**This inference is validated, not trusted.** `nbare check-minutes` compares
reconstructed on-floor seconds against the official box score for every
player. Summed stint seconds must equal box-score minutes within tolerance.
A reversed sub direction or a missed opener fails the gate loudly. **Do not
build RAPM on a reconstruction that hasn't passed** — a lineup that's quietly
wrong yields a design matrix that's quietly wrong, and the regression won't
tell you.

Because stats.nba.com is unreachable from CI, the logic is proven against a
synthetic game generator with known ground truth: we build games forward
with a controlled substitution timeline, then assert reconstruction inverts
them to zero error. The reversed-sub test confirms the gate *catches* the
classic bug, so it's trustworthy on real data where there's no ground truth.

```bash
nbare check-minutes --synthetic          # proves the logic, needs no data
nbare check-minutes --season 2024-25     # the real gate, after a backfill
```

## Status / known gaps

- [x] Warehouse schema, ingestion client, CLI, test suite
- [x] BBRef contract parser + data-quality report + crosswalk seed (444 slugs)
- [x] Stage 1 salary-matching engine (verified bands, 3-valued verdicts, CLI)
- [x] Transaction overlay system (sign/waive/trade/amend on immutable base)
- [x] Stage 2 stint reconstruction + minutes-validation gate (synthetic-proven)
- [ ] `config.LEAGUE_YEARS` — the 2026-27 cap/tax/apron/MLE figures are the
      official league release, but the **BAE and the 2025-26 secondary
      figures are unverified placeholders**. Confirm against the league
      press release before any Stage 1 rule depends on them.
- [ ] Contract + transaction scrapers (Spotrac / Basketball Reference)
- [ ] Box score ingestion (needed for the minutes validation gate)
- [ ] Stage 1 rule engine