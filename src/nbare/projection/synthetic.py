"""Generate synthetic player-season panels with KNOWN ground truth.

Same philosophy as `rapm/synthetic.py`: build data forward from a planted
truth, then assert the model inverts it. Each generator here is scoped to
the layer it validates (see `docs/stage3_spec.md`'s build order) but they
share one aging-curve shape and one player-population model, so layer 2's
generator is layer 1's generator plus position structure, and layer 3's is
layer 2's plus a box-score relationship -- consistent with how the real
model is built up.

The survivorship trap, modeled on purpose
------------------------------------------
`make_league_curve_players` deliberately gives bad players SHORT careers
and good players LONG ones (career length depends on skill). This is not
an incidental realism detail -- it is the specific contamination the delta
method exists to resist. A naive age-vs-rating regression over this data
would show performance *improving* into the late 30s, because only good
players survive that long. The delta method must not show that; the test
in `test_projection_delta.py` checks both claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbare.projection.types import PlayerSeason
from nbare.rapm.prior import BoxScoreRow

POSITION_GROUPS = ("G", "F", "C")


def true_level(age: int, anchor_age: int, curvature: float) -> float:
    """Planted league-average trajectory shape: a downward parabola peaking
    at `anchor_age`, in rating points relative to the peak."""
    return -curvature * (age - anchor_age) ** 2


def planted_delta_curve(
    age_min: int, age_max: int, anchor_age: int, curvature: float
) -> dict[int, float]:
    """Exact analytic deltas of `true_level` for each age transition
    age -> age + 1, for age in [age_min, age_max)."""
    return {
        age: true_level(age + 1, anchor_age, curvature)
        - true_level(age, anchor_age, curvature)
        for age in range(age_min, age_max)
    }


@dataclass(frozen=True, slots=True)
class LeagueCurveDataset:
    seasons: list[PlayerSeason]
    delta_curve: dict[int, float]     # planted, age_from -> delta
    anchor_age: int
    age_min: int
    age_max: int
    quality_by_player: dict[int, float]  # planted skill baseline


def make_league_curve_players(
    n_players: int = 200,
    age_min: int = 20,
    age_max: int = 38,
    anchor_age: int = 27,
    curvature: float = 0.08,
    season_noise_sd: float = 1.0,
    seed: int = 0,
) -> LeagueCurveDataset:
    """Layer-1 dataset: one league-wide planted curve, no position
    structure, career length coupled to skill (the survivorship trap).
    """
    rng = np.random.default_rng(seed)
    delta_curve = planted_delta_curve(age_min, age_max, anchor_age, curvature)

    seasons: list[PlayerSeason] = []
    quality_by_player: dict[int, float] = {}

    for pid in range(n_players):
        quality = float(rng.normal(0, 8))
        quality_by_player[pid] = quality

        start_age = int(rng.integers(age_min, min(age_min + 5, age_max - 1)))
        # Career length rises with quality: bad players wash out young,
        # good players play into their late 30s.
        expected_years = np.clip(6 + quality * 0.7 + rng.normal(0, 1.5), 2,
                                  age_max - start_age)
        end_age = min(age_max, start_age + int(round(expected_years)))

        pos = POSITION_GROUPS[int(rng.integers(0, len(POSITION_GROUPS)))]
        for age in range(start_age, end_age + 1):
            off = (quality + true_level(age, anchor_age, curvature)
                   + rng.normal(0, season_noise_sd))
            defn = (quality * 0.6 + true_level(age, anchor_age, curvature)
                    + rng.normal(0, season_noise_sd))
            poss = float(rng.uniform(500, 3000))
            seasons.append(PlayerSeason(
                player_id=pid, season=f"s{age}", age=age,
                position_group=pos, off_rating=off, def_rating=defn,
                off_possessions=poss, def_possessions=poss,
            ))

    return LeagueCurveDataset(
        seasons=seasons, delta_curve=delta_curve, anchor_age=anchor_age,
        age_min=age_min, age_max=age_max, quality_by_player=quality_by_player,
    )


def planted_position_delta_curve(
    age_min: int, age_max: int, anchor_age: int, curvature: float, linear: float
) -> dict[int, float]:
    """Like `planted_delta_curve` but with an added linear term, so
    different positions can have visibly different shapes (not just
    different steepness of the same symmetric parabola)."""
    def level(age: int) -> float:
        return linear * (age - anchor_age) - curvature * (age - anchor_age) ** 2

    return {age: level(age + 1) - level(age) for age in range(age_min, age_max)}


@dataclass(frozen=True, slots=True)
class PositionCurveDataset:
    seasons: list[PlayerSeason]
    delta_curve_by_position: dict[str, dict[int, float]]  # planted, per position
    player_position: dict[int, str]
    anchor_age: int
    age_min: int
    age_max: int
    sparse_player_ids: list[int]   # <= 2 seasons
    rich_player_ids: list[int]     # >= 8 seasons
    player_own_shift: dict[int, float]  # planted idiosyncratic deviation from position curve


# Distinct planted shapes per position: guards decline slowly and late,
# bigs decline earlier and faster, forwards in between. Chosen to be
# visibly different so a recovery test can tell "recovered position curve"
# from "recovered league curve" apart.
POSITION_SHAPE_PARAMS: dict[str, dict[str, float]] = {
    "G": dict(curvature=0.05, linear=0.3),
    "F": dict(curvature=0.08, linear=0.0),
    "C": dict(curvature=0.12, linear=-0.3),
}


def make_position_curve_players(
    n_players_per_position: int = 30,
    age_min: int = 20,
    age_max: int = 38,
    anchor_age: int = 27,
    season_noise_sd: float = 1.0,
    own_shift_sd: float = 0.45,
    seed: int = 0,
) -> PositionCurveDataset:
    """Layer-2 dataset: three position groups with distinct planted delta
    curves. Career length still varies (some players sparse: 1-2 seasons;
    some rich: 8+ seasons) so the test can check both curve recovery AND
    shrinkage of sparse players toward their position curve.
    """
    rng = np.random.default_rng(seed)

    delta_curve_by_position = {
        pos: planted_position_delta_curve(age_min, age_max, anchor_age, **params)
        for pos, params in POSITION_SHAPE_PARAMS.items()
    }

    def level_for(pos: str, age: int) -> float:
        params = POSITION_SHAPE_PARAMS[pos]
        return (params["linear"] * (age - anchor_age)
                - params["curvature"] * (age - anchor_age) ** 2)

    seasons: list[PlayerSeason] = []
    player_position: dict[int, str] = {}
    player_own_shift: dict[int, float] = {}
    sparse_ids: list[int] = []
    rich_ids: list[int] = []

    pid = 0
    for pos in POSITION_GROUPS:
        for j in range(n_players_per_position):
            player_position[pid] = pos
            quality = float(rng.normal(0, 6))
            # An idiosyncratic per-player deviation from the position curve
            # (linear in age-from-anchor) -- lets a rich player's own
            # trajectory differ visibly from his position's pooled curve.
            own_shift = float(rng.normal(0, own_shift_sd))
            player_own_shift[pid] = own_shift

            # Alternate sparse and rich careers within each position so
            # both regimes are well represented.
            if j % 3 == 0:
                start_age = int(rng.integers(age_min, age_max - 1))
                end_age = min(age_max, start_age + 1)  # 1-2 seasons
                sparse_ids.append(pid)
            else:
                start_age = int(rng.integers(age_min, age_min + 4))
                end_age = min(age_max, start_age + int(rng.integers(8, 14)))
                rich_ids.append(pid)

            for age in range(start_age, end_age + 1):
                off = (quality + level_for(pos, age) + own_shift * (age - anchor_age)
                       + rng.normal(0, season_noise_sd))
                defn = (quality * 0.5 + level_for(pos, age)
                        + own_shift * (age - anchor_age)
                        + rng.normal(0, season_noise_sd))
                poss = float(rng.uniform(500, 3000))
                seasons.append(PlayerSeason(
                    player_id=pid, season=f"s{age}", age=age,
                    position_group=pos, off_rating=off, def_rating=defn,
                    off_possessions=poss, def_possessions=poss,
                ))
            pid += 1

    return PositionCurveDataset(
        seasons=seasons, delta_curve_by_position=delta_curve_by_position,
        player_position=player_position, anchor_age=anchor_age,
        age_min=age_min, age_max=age_max, sparse_player_ids=sparse_ids,
        rich_player_ids=rich_ids, player_own_shift=player_own_shift,
    )


# Planted box-score -> baseline relationships. Coefficients are chosen to
# mirror the real ones documented (informally) in `rapm/prior.py`'s own
# synthetic test: offense rewards scoring/passing/efficiency and punishes
# turnovers; defense rewards steals, blocks, defensive rebounding, and
# being a big.
PLANTED_OFFENSE_COEF: dict[str, float] = dict(
    pts_100=0.25, ast_100=0.9, tov_100=-1.1, ts_pct=25.0,
)
PLANTED_DEFENSE_COEF: dict[str, float] = dict(
    stl_100=1.4, blk_100=1.8, dreb_rate=12.0, position_big=1.6,
)


@dataclass(frozen=True, slots=True)
class BoxScoreBaselineDataset:
    seasons: list[PlayerSeason]
    box_rows: dict[int, BoxScoreRow]
    delta_curve_by_position: dict[str, dict[int, float]]
    player_position: dict[int, str]
    anchor_age: int
    age_min: int
    age_max: int
    sparse_player_ids: list[int]
    rich_player_ids: list[int]
    player_own_shift: dict[int, float]
    true_off_baseline: dict[int, float]   # planted per-player offense level
    true_def_baseline: dict[int, float]   # planted per-player defense level


def make_boxscore_baseline_players(
    n_players_per_position: int = 18,
    age_min: int = 20,
    age_max: int = 38,
    anchor_age: int = 27,
    season_noise_sd: float = 1.0,
    own_shift_sd: float = 0.45,
    baseline_residual_sd: float = 1.5,
    seed: int = 0,
) -> BoxScoreBaselineDataset:
    """Layer-3 dataset: layer 2's position/career structure, but each
    player's baseline (offense and defense separately) is now generated
    FROM a planted linear box-score relationship plus a residual -- exactly
    the thing layer 3's regression must recover. `baseline_residual_sd` is
    the part of skill the box score does NOT explain (real players are not
    perfectly predictable from a box score either), so recovery is expected
    to be good but not exact.
    """
    rng = np.random.default_rng(seed)

    delta_curve_by_position = {
        pos: planted_position_delta_curve(age_min, age_max, anchor_age, **params)
        for pos, params in POSITION_SHAPE_PARAMS.items()
    }

    def level_for(pos: str, age: int) -> float:
        params = POSITION_SHAPE_PARAMS[pos]
        return (params["linear"] * (age - anchor_age)
                - params["curvature"] * (age - anchor_age) ** 2)

    seasons: list[PlayerSeason] = []
    box_rows: dict[int, BoxScoreRow] = {}
    player_position: dict[int, str] = {}
    player_own_shift: dict[int, float] = {}
    true_off_baseline: dict[int, float] = {}
    true_def_baseline: dict[int, float] = {}
    sparse_ids: list[int] = []
    rich_ids: list[int] = []

    pid = 0
    for pos in POSITION_GROUPS:
        for j in range(n_players_per_position):
            player_position[pid] = pos
            own_shift = float(rng.normal(0, own_shift_sd))
            player_own_shift[pid] = own_shift

            pts = float(rng.normal(20, 6))
            ast = float(rng.normal(4, 2))
            tov = float(rng.normal(2.5, 1))
            ts = float(np.clip(rng.normal(0.56, 0.05), 0.4, 0.7))
            stl = float(rng.normal(1.2, 0.5))
            blk = float(rng.normal(0.6, 0.6))
            dreb = float(np.clip(rng.normal(0.15, 0.06), 0.02, 0.35))
            big = 1 if pos == "C" else (1 if pos == "F" and rng.random() < 0.25 else 0)

            off_baseline = (
                PLANTED_OFFENSE_COEF["pts_100"] * (pts - 20)
                + PLANTED_OFFENSE_COEF["ast_100"] * (ast - 4)
                + PLANTED_OFFENSE_COEF["tov_100"] * (tov - 2.5)
                + PLANTED_OFFENSE_COEF["ts_pct"] * (ts - 0.56)
                + rng.normal(0, baseline_residual_sd)
            )
            def_baseline = (
                PLANTED_DEFENSE_COEF["stl_100"] * (stl - 1.2)
                + PLANTED_DEFENSE_COEF["blk_100"] * (blk - 0.6)
                + PLANTED_DEFENSE_COEF["dreb_rate"] * (dreb - 0.15)
                + PLANTED_DEFENSE_COEF["position_big"] * big
                + rng.normal(0, baseline_residual_sd)
            )
            true_off_baseline[pid] = float(off_baseline)
            true_def_baseline[pid] = float(def_baseline)

            if j % 3 == 0:
                start_age = int(rng.integers(age_min, age_max - 1))
                end_age = min(age_max, start_age + 1)
                sparse_ids.append(pid)
            else:
                start_age = int(rng.integers(age_min, age_min + 4))
                end_age = min(age_max, start_age + int(rng.integers(8, 14)))
                rich_ids.append(pid)

            off_poss_total = 0.0
            def_poss_total = 0.0
            for age in range(start_age, end_age + 1):
                off = (off_baseline + level_for(pos, age) + own_shift * (age - anchor_age)
                       + rng.normal(0, season_noise_sd))
                defn = (def_baseline + level_for(pos, age) + own_shift * (age - anchor_age)
                        + rng.normal(0, season_noise_sd))
                poss = float(rng.uniform(500, 3000))
                off_poss_total += poss
                def_poss_total += poss
                seasons.append(PlayerSeason(
                    player_id=pid, season=f"s{age}", age=age,
                    position_group=pos, off_rating=off, def_rating=defn,
                    off_possessions=poss, def_possessions=poss,
                ))

            box_rows[pid] = BoxScoreRow(
                player_id=pid, off_possessions=off_poss_total,
                def_possessions=def_poss_total, pts_100=pts, ast_100=ast,
                tov_100=tov, ts_pct=ts, stl_100=stl, blk_100=blk,
                dreb_rate=dreb, position_big=big,
            )
            pid += 1

    return BoxScoreBaselineDataset(
        seasons=seasons, box_rows=box_rows,
        delta_curve_by_position=delta_curve_by_position,
        player_position=player_position, anchor_age=anchor_age,
        age_min=age_min, age_max=age_max, sparse_player_ids=sparse_ids,
        rich_player_ids=rich_ids, player_own_shift=player_own_shift,
        true_off_baseline=true_off_baseline, true_def_baseline=true_def_baseline,
    )
