# Fix: phantom non-player entities leak into lineups (allowlist, not denylist)

## Status: partially fixed, still broken

The pbp parser fix added guards for SPECIFIC event types whose personId slot
holds a non-player (timeouts, team rebounds, instant-replay reviews). That is
a denylist, and denylists are incomplete by construction. A 20-game
check-minutes run (AFTER that fix) proved it: games 0022500098 and
0022500093 still reconstruct team IDs and event codes as players on the floor
for entire games.

Evidence from that run:
- `1610612758`, `1610612762`, `1610612745`, `1610612765` reconstructed at
  ~3080s (a full game). These are TEAM IDs — every NBA team id is in the
  `16106127xx` range. Teams are being placed on the floor as players.
- `486`, `510`, `226`, `327`, `605`, `766` reconstructed at 720s (a full
  period). These are small event/replay codes, not player ids.

These phantoms occupy roster slots, which means they also silently corrupt
some games that currently PASS the gate (the errors only need to net out
under tolerance to slip through). So this is not "5 bad games to exclude" —
it is a systemic contamination of the lineup reconstruction.

## The fix: allowlist against the real player table

Do NOT add another event-type guard. Replace the whole approach with a single
allowlist rule:

> A value in a player-id slot (`player1_id`, `player2_id`, `player3_id`) is
> kept only if it is an actual player id present in `stg.player`. Anything
> else is set to NULL before reconstruction.

This closes the entire class in one rule — team ids, replay codes, and any
future leak type nobody has seen yet — instead of patching instances.

Preferred implementation point: filter at reconstruction time, in
`rapm/stints.py`, where the pbp is read and lineups are built, so the guard
protects reconstruction regardless of what upstream parsing let through.
(Filtering in the parser is also acceptable, but the reconstruction-time
guard is the safety net that cannot be bypassed by a future parser change.)
`reconstruct_game` should take the set of valid player ids (from
`stg.player`) and ignore any id not in it when inferring openers and applying
subs.

## Step 1 — audit first, so we see the full scope

Before writing the filter, add a one-off audit that prints EVERY distinct
`player1_id` / `player2_id` / `player3_id` value in `stg.pbp_event` for the
2025-26 season that is NOT in `stg.player`, grouped by the `event_type` it
came from, with counts. This shows the complete set of leak sources the
denylist was missing — not just the two we happened to see. Keep this as a
`nbare audit-phantom-ids --season 2025-26` command or a test-only helper;
it's useful diagnostics to keep.

## Step 2 — apply the allowlist filter, add tests

- Reconstruction ignores any id not in the valid-player set.
- Test with a hand-built payload (mirroring the real games above) that
  contains a team id and a small event code in player slots; assert neither
  appears in any reconstructed lineup and that minutes for the real players
  come out correct.
- Add a test that a game which previously failed ONLY due to phantom ids now
  passes.

## Step 3 — re-validate at 400 games, NOT 20

Run `nbare check-minutes --season 2025-26 --limit 400` and report the new
pass rate. The 20-game sample is what hid this bug in the first place; do not
re-validate on a small sample. Report:
- new pass/fail count out of 400
- the remaining failure taxonomy (which are inherent data gaps — a player
  with box minutes but zero pbp events, a sub whose incoming player never
  appears elsewhere — vs. any still-systematic issue)

## Do not

- Do not add per-event-type guards. Allowlist only.
- Do not loosen MINUTES_TOLERANCE_S to raise the pass rate.
- Do not exclude failing games to make the number look better — diagnose them.