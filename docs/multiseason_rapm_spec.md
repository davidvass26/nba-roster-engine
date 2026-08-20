# Step 1: Multi-Season RAPM (pooled baseline)

Goal: fit RAPM across three seasons (2023-24, 2024-25, 2025-26) instead of
one. More possessions per player = less shrinkage = more spread and fewer
collinearity artifacts. This is the SAME model with more data, not a new
model. It is also the baseline that Steps 2 (home-court adjustment) and 3
(aging-aware RAPM) will be measured against, so it must be clean before
anything is built on top of it.

## Why this is the highest-leverage change

The single-season 2025-26 leaderboard showed compressed ratings (+1.9 to
+2.9, where elite players should reach +4 to +6) and role-player artifacts
(DiVincenzo on both offense and defense lists; Champagnie, AJ Green in the
offensive top 15). Both are symptoms of too few possessions per player, not
model error. Multi-season fitting is the direct fix: stars should spread
out, artifacts should regress toward the mean.

## Part A — backfill the two new seasons

For BOTH 2023-24 and 2024-25:
    make games SEASON=<season>
    make pbp   SEASON=<season>
    make box   SEASON=<season>

This is ~2 hours of rate-limited fetching total. It is resumable (the cache
means a crash resumes for free). Do not rush it.

## Part B — validate each new season through the minutes gate

Do NOT assume the pbp/phantom/allowlist fixes transfer silently. Each season
has its own data and its own quirks. For each new season:
    nbare check-minutes --season <season> --limit 400

Confirm the pass rate holds around 85%+ (2025-26 was 86.7%). Report the
pass rate and failure taxonomy per season. If a new season's pass rate is
dramatically lower, STOP and diagnose — it means that season has a data
issue the 2025-26 fixes didn't cover, and fitting on it would inject bad
stints. Do not proceed to the fit until each season passes at a comparable
rate.

Also run `nbare audit-phantom-ids` on each new season to confirm zero
non-player leaks (the allowlist should hold, but verify).

## Part C — pooled multi-season fit

Extend the fit to consume multiple seasons at once:
- A player who appears in all three seasons is ONE column in the design
  matrix (one pooled rating). His stints from all three seasons stack as
  rows. This is the plain pooled baseline — do NOT split a player into
  per-season columns (that's Step 3's aging-aware version, not this).
- Continue to exclude any game that fails the minutes gate, in every season.
- Grouped cross-validation for lambda still groups by game_id (a game is a
  game regardless of season; no leakage concern across seasons).
- Log: total games across all seasons, games included/excluded per season,
  total stints, final lambda.

Add a CLI option like `nbare fit-rapm --seasons 2023-24,2024-25,2025-26`
(or a sensible multi-season interface) while keeping single-season working.

## Part D — the validation checkpoint (this is the point)

Re-print the top-30 total / top-15 offense / top-15 defense leaderboard with
names. Compare to the single-season 2025-26 result and confirm:

1. **Spread widens.** Elite players should now reach higher magnitudes than
   the compressed +2.9 ceiling — more like +3.5 to +5+. If the spread does
   NOT widen with 3x the data, the pooling is likely wrong (e.g.
   accidentally still per-season, or possessions not accumulating).
2. **Role-player artifacts regress.** DiVincenzo appearing on BOTH lists,
   and role players like Champagnie/AJ Green in the offensive top 15, should
   diminish or vanish as they get pulled toward the mean by more varied
   lineup exposure.
3. **Stars remain at the top.** SGA, Jokić, Dončić, Curry (offense);
   Gobert, Wembanyama, Caruso (defense) should still anchor the top — but
   with more separation from the pack.

Report the before/after comparison explicitly. If (1) and (2) do not
happen, the fit has a bug — do not declare success just because it ran.

## Part E — back up the cache (do this once, it's insurance)

After all three seasons are fetched, `data/cache/` represents ~hours of
rate-limited fetching and is git-ignored (correctly). Copy it somewhere
safe (external drive / cloud) ONCE. Completed games never change, so this
backup is valid forever and means those seasons never need re-fetching even
if the working machine's data is lost. Note this in the run log as a
reminder to the user.

## Do not

- Do not split players into per-season columns — that's Step 3, not this.
- Do not loosen the minutes tolerance to raise any season's pass rate.
- Do not declare success on "it ran" — success is the leaderboard spread
  widening and artifacts regressing, per Part D.
- Do not start home-court or aging adjustments — those are separate specs
  that build on this baseline.