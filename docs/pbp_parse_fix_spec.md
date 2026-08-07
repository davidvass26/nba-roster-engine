# Fix: ingest_pbp cannot parse real play-by-play (blocking)

## The bug

`ingest_pbp` in `src/nbare/ingest/nba_stats.py` calls
`result_set_to_df(resp.payload)` on `playbyplayv3` responses.
`result_set_to_df` only understands the legacy `resultSets` shape. But
`playbyplayv3` returns the nested `{"game": {"actions": [...]}}` shape (same
class of mismatch as the box-score bug already fixed in this codebase). So
`ingest_pbp` silently returns an empty DataFrame on ALL real data.

Consequence: real play-by-play has never loaded. Every downstream piece —
stint reconstruction, the minutes gate, RAPM, projections — has only ever
run on synthetic play-by-play. This is the last plumbing blocker before the
full real-data chain works.

## What to build

A dedicated parser for the v3 play-by-play shape, mirroring how
`_parse_box_score_payload` was built for box scores. Do NOT route v3 pbp
through `result_set_to_df`.

- VERIFY the real `playbyplayv3` payload shape and field names against (a) a
  real fetched game and (b) nba_api's own parser source. Do not assume field
  names — the box-score task already proved nba_api's real shapes differ
  from the generic parser.
- Map the `actions` array into the existing `stg.pbp_event` columns, which
  stint reconstruction reads:
  `game_id, event_num, period, clock_seconds_left, event_type,
   event_action_type, description, team_id, player1_id, player2_id,
   player3_id, home_score, away_score`.
- Keep using the cached, rate-limited `fetch()` client and the `_upsert`
  helper. Do not bypass either.

## The load-bearing field: substitution events

Stint reconstruction identifies who is on the floor from SUBSTITUTION events,
reading `player1_id` (player going OUT) and `player2_id` (player coming IN).
This attribution is the single most important thing the parser must get
right — it is what the minutes gate ultimately validates.

- Confirm on a REAL substitution event that the outgoing player lands in
  `player1_id` and the incoming player in `player2_id` (or whatever the v3
  shape provides — verify, then map to that convention, because
  `reconstruct_game` defaults to `sub_out_col="player1_id",
  sub_in_col="player2_id"`).
- `clock_seconds_left` must be seconds remaining in the period as a float.
  The existing `parse_clock` handles the "PT11M23.00S" ISO-8601 duration
  format; confirm v3 uses that and reuse it.
- `event_type` must carry a value that `_is_sub` recognizes (it matches the
  lowercased substring "substitution"). Confirm the v3 action-type string
  for subs contains that, or adjust `_is_sub` to match what v3 actually
  emits.

## Validation (synthetic-first, then real)

1. Unit-test the new parser against a small hand-built v3-shaped payload
   with a known substitution, asserting the sub lands in the right columns.
2. Fetch ONE real 2025-26 game, parse it, and manually confirm: non-empty
   DataFrame, substitution events present with correct in/out attribution,
   clock values sane (0–720s in regulation), scores monotonic.
3. Then run `nbare check-minutes --season 2025-26`. It will now actually run
   instead of skipping every game.

## Expect the minutes gate to fail on the first real run — that is correct

Once real pbp loads, real substitution data hits reconstruction for the
first time, and real subs are messier than the synthetic generator (odd sub
labels, period-boundary quirks, players who log time but appear in no
event). A gate failure means the gate is WORKING. Fix the reconstruction to
match real data. NEVER loosen `MINUTES_TOLERANCE_S` to force a pass — that
defeats the entire purpose of the gate.

## After this fix — the full real chain finally runs

    make games SEASON=2025-26
    make pbp   SEASON=2025-26      # now actually populates stg.pbp_event
    make box   SEASON=2025-26
    nbare check-minutes --season 2025-26   # real test; fix reconstruction
    nbare fit-rapm --season 2025-26        # first real ratings

## Scope

Fix the pbp parser and get check-minutes running on real data. Do not change
the RAPM math, the design matrix, or the fit — those read the tables
correctly and only need the tables populated. This is a plumbing fix, not a
modeling change.