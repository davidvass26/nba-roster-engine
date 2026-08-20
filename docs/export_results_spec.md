# Export precomputed results for the app

## Why

The deployed Streamlit app (Streamlit Community Cloud) will NOT have the
local DuckDB warehouse or the nba.com cache. So RAPM, value, and projections
must be computed locally and exported to small files the app ships with and
reads directly. Cap sheets and trade legality run live from the contract CSV
(which is small and in the repo), so those need no export.

## What to build

A CLI command `nbare export-app-data` that writes small, app-ready files to
`app/data/` (create the folder). All files are derived, so they can be
committed to the repo — they're small (KB-MB), unlike the cache.

Export these:

1. `rapm_leaderboard.csv` — one row per player, columns:
   `player_id, name, offense, defense, total, value, wins_added,
    possessions`. This is the full rate+value leaderboard from the pooled
   multi-season fit. Sorted by total descending.

2. `projections.csv` — one row per player per projected year, columns:
   `player_id, name, season, proj_offense, proj_defense, proj_total,
    lower_80, upper_80` (the 80% posterior interval bounds). If projections
   output posterior samples, summarize to median + 80% interval here.

3. `meta.json` — provenance so the app can show it honestly:
   `{seasons_used: [...], n_games, n_stints, lambda, replacement_level,
     points_per_win, generated_at, fit_type: "pooled multi-season ridge
     with box-score prior"}`.

## Rules

- Do NOT recompute anything novel — read the existing fit path
  (`fit-rapm --seasons ...`) and the projections output, and serialize.
- Money/rating values stay exact per CLAUDE.md; format for display only at
  the app layer, not here.
- The command should print what it wrote and the row counts.
- These files are the app's data contract — the app reads ONLY these plus
  the live contract CSV, never the warehouse.

## Validation

- Round-trip check: the exported RAPM leaderboard top-10 must match what
  `fit-rapm` prints. Don't let export silently reorder or drop players.
- Confirm every player in the leaderboard has a real name (join to
  stg.player); no bare ids in the exported files.