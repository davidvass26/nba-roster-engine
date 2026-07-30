# Stage 3 — Player Projections (v1 spec)

Hierarchical Bayesian aging-curve model projecting player RAPM forward.
This is a focused v1: one target, one validation story, layered build.

## Target

Project **offensive and defensive RAPM separately**, 1–4 years forward, for
each player, as **posterior distributions** — not point estimates. Output
posterior samples per player per future year, structured to feed the Stage 4
optimizer. The optimizer needs impact-with-uncertainty, so a point estimate
is not an acceptable output.

## Aging curve — delta method (non-negotiable)

Estimate the aging curve with the **delta method**: compare the *same player*
across consecutive seasons and average the deltas by age. This avoids
survivorship bias.

Do NOT regress performance against age across different players. That is the
classic trap: bad players get cut young, so only good players remain at 35,
making raw age-vs-performance curves far too optimistic about decline. The
delta method never compares across players, so survivorship cannot
contaminate it.

## Hierarchy — partial pooling (the centerpiece)

Partial-pool player aging trajectories: **player → position-group → league**.
A player with lots of history gets his own trajectory; a sparse player gets
pulled toward his position group's typical curve, which is pulled toward the
league curve. Shrinkage proportional to how little data the player has — the
same philosophy as the Bayesian prior in `rapm/prior.py`, applied to
trajectories in the time dimension.

## Box score as predictors — BASELINE ONLY (v1)

A player's box-score profile **shifts where his aging curve sits** (his
baseline level) but does NOT modulate the curve's *shape* in v1. This anchors
young/sparse players the way the RAPM prior anchors low-possession ones —
which is exactly the population aging curves most need help with.

Reuse the existing definitions from `src/nbare/rapm/prior.py`:
- `BoxScoreRow` (do not redefine it)
- `OFFENSE_FEATURES = ("pts_100", "ast_100", "tov_100", "ts_pct")`
- `DEFENSE_FEATURES = ("stl_100", "blk_100", "dreb_rate", "position_big")`

Offense baseline uses the offense features; defense baseline uses the
(deliberately richer) defense features. Same split, same reasoning as the
prior.

## Inputs

- RAPM ratings: output of `rapm/fit.py` (`RAPMResult.offense` / `.defense`).
- Box-score aggregates: `BoxScoreRow` from `rapm/prior.py`.
- Player age and position per season.

## Output

Posterior samples per player, per future year (t+1 … t+4), for offensive and
defensive RAPM. Shape it so Stage 4 can draw from it directly.

## Validation — synthetic-first (non-negotiable)

Before any real data: write a synthetic generator that plants **known aging
curves** and **known box-score → baseline relationships**, then confirm the
model recovers them. Report **R-hat** (convergence) and **posterior
predictive checks**. A hierarchical model that silently fails to converge is
the specific risk here — the recovery test is what catches it.

## Build order (do not build the whole model at once)

1. **Delta-method aging curve** on synthetic data. Prove it recovers a
   planted league-average curve before anything else.
2. **Add the hierarchy** (player → position → league partial pooling). Prove
   it recovers planted per-position curves and correctly shrinks sparse
   players.
3. **Add box-score baselines.** Prove it recovers a planted box-score →
   baseline relationship.

Each layer must recover planted truth before the next is added. When a
recovery test breaks, this ordering tells you which layer broke it.

## Explicitly deferred to v2 — DO NOT build in v1

- Attrition / retirement modeling (non-random exit of declining players)
- Availability / games-played (beta-binomial on durability)
- Box-score stats as their own projected targets
- Box score modulating trajectory *shape* (v1 is baseline-only)

## Stack

`numpyro` + `arviz` (the `[model]` extra). Install first:
`pip install -e ".[dev,model]"`. Follow all CLAUDE.md principles — exact math,
honesty over false completeness, synthetic-first validation.