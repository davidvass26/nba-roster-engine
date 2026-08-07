# Box-Score Ingestion (Stage 0 gap-fill)

## Why this exists

`stg.box_player` is defined in the schema and READ by two places
(`check-minutes` in cli.py and `blocks_from_warehouse` in rapm/blocks.py),
but NOTHING writes to it. There is no box-score ingestion function. As a
result, on real data `check-minutes` and `fit-rapm` silently skip every
game (the `if pbp.is_empty() or box.is_empty(): continue` guard fires
because the box table is always empty). This must be built before any real
RAPM is possible.

## What to build

An `ingest_box_scores` function in `src/nbare/ingest/nba_stats.py`,
mirroring the structure of the existing `ingest_pbp`:
- Takes a connection and an iterable of game_ids.
- Uses the cached, rate-limited `fetch()` client (same as ingest_pbp — do
  NOT bypass the cache or the rate limiter; a box-score backfill is as many
  requests as the pbp backfill and must be equally resumable).
- Fetches each game's traditional box score. VERIFY the correct nba_api
  endpoint against the installed library — it is most likely
  `boxscoretraditionalv3`, but confirm the class name and its result-set
  shape before relying on it. Do not assume field names.
- Parses per-player rows into the existing `stg.box_player` schema:
  `game_id, nba_player_id, nba_team_id, seconds_played, pts, reb, ast,
  started`.
- Upserts keyed on `(game_id, nba_player_id)` using the existing `_upsert`
  helper.

## The critical field: seconds_played

nba.com reports minutes as a string like "34:12" (MM:SS) or sometimes an
ISO-8601 duration. Parse it to integer SECONDS. This field is what
`validate_minutes` compares reconstructed stint-time against — if it is
wrong, the entire minutes gate is meaningless. Write a small parse function
and unit-test it on the string formats nba.com actually returns (check a
real payload).

## Validation (synthetic-first, per CLAUDE.md)

Before touching nba.com:
- The synthetic scoring-game generator (`rapm/synthetic.py`,
  `make_scoring_game` / `scoring_box_frame`) already produces box scores
  with KNOWN `seconds_played`. Use it to test that the parse-and-upsert path
  round-trips seconds correctly.
- Add a test that the MM:SS parser recovers known seconds exactly
  (e.g. "34:12" -> 2052, "0:00" -> 0, "48:00" -> 2880).

## Then wire it up

- Add a CLI command `ingest-box` (mirror `ingest-pbp`): fetch box scores for
  all games in a season, resumable, cached.
- Add a `make box SEASON=...` target.
- Do NOT change `check-minutes` or `fit-rapm` — they already read the table
  correctly; they were just reading an empty table.

## After it's built (real data)

- make games SEASON=2025-26
- make pbp SEASON=2025-26
- make box SEASON=2025-26        # the new command, once box ingestion is built
- nbare check-minutes --season 2025-26
- nbare fit-rapm --season 2025-26

Expect step 4 to surface real-data reconstruction issues the synthetic
tests didn't (nba.com labels subs and period boundaries in messier ways
than the generator). That is exactly what the gate is for — do not lower the
tolerance to force a pass; fix the reconstruction.