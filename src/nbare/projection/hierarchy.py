"""Layer 2: hierarchical partial pooling of aging trajectories.

Same philosophy as `rapm/prior.py`, applied in the time dimension
--------------------------------------------------------------------
`rapm/prior.py` shrinks a low-possession player's RAPM toward what his box
score predicts, instead of toward zero, because zero throws away
information the box score already has. This module makes the analogous
move for trajectories: a player with a long career gets his own aging
curve; a player with one or two seasons gets pulled toward his position
group's curve, which itself is pulled toward the league curve. The amount
of pooling is not asserted -- it is *learned* from the data, as the
variance ratio between levels (a player who is wildly inconsistent with
his position group's curve resists being pooled into it; a player who is
noisy but unremarkable does not).

How this builds on layer 1 (delta.py) instead of replacing it
---------------------------------------------------------------
`delta.py` computes ONE number: the league-wide average delta at each age
transition. This module keeps the same generative idea -- a player's
rating at age `a` is his own baseline PLUS the cumulative sum of age
transition deltas -- but makes every level of that computation (player,
position, league) a partial-pooling hierarchy instead of a single flat
average. The player's baseline (rating at the anchor age) is a free
parameter here, weakly regularized; layer 3 (`baseline.py`) replaces that
weak prior with a box-score-informed one, exactly mirroring how
`rapm/prior.py` replaces ridge's zero-mean prior.

Why deltas, not raw age-vs-rating levels, are pooled
------------------------------------------------------
Pooling raw age-vs-rating levels across players would reintroduce the
survivorship trap `delta.py`'s docstring describes: a position group's
"typical level at 35" is contaminated by which players are still around
at 35. Pooling the age-TRANSITION DELTA is safe because every delta is
still a within-player comparison; only the shape gets shared across
players, never the level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from nbare.projection.types import PlayerSeason


@dataclass
class HierarchicalCurveResult:
    """Posterior samples for every level of the trajectory hierarchy.

    Shapes (all arrays are pooled across chains, draws first):
      baseline:        (S, N)
      player_delta:     (S, N, K-1)
      position_delta:   (S, G, K-1)
      league_delta:     (S, K-1)
    """

    samples: dict[str, np.ndarray]
    player_index: dict[int, int]
    position_index: dict[str, int]
    player_position: dict[int, str]
    age_grid: list[int]
    ref_idx: int
    anchor_age: int
    idata: az.InferenceData
    # Raw observation arrays, kept for posterior predictive checks.
    obs_i: np.ndarray = field(repr=False)
    obs_a: np.ndarray = field(repr=False)
    obs_y: np.ndarray = field(repr=False)
    obs_w: np.ndarray = field(repr=False)

    def player_curve_samples(self, player_id: int) -> np.ndarray:
        """Posterior samples of the player's rating across the whole age
        grid, shape (S, K). Falls back to the position/league pooled curve
        (via player_delta's prior mean) for players not in the fit -- but
        callers should prefer `position_curve_samples` explicitly for that
        case since it is clearer about what is being returned."""
        i = self.player_index[player_id]
        baseline = self.samples["baseline"][:, i]                 # (S,)
        pdelta = self.samples["player_delta"][:, i, :]             # (S, K-1)
        S = np.concatenate(
            [np.zeros((pdelta.shape[0], 1)), np.cumsum(pdelta, axis=1)], axis=1
        )  # (S, K)
        return baseline[:, None] + S - S[:, [self.ref_idx]]

    def player_relative_curve_samples(self, player_id: int) -> np.ndarray:
        """Like `player_curve_samples` but WITHOUT the player's baseline --
        pure curve shape, pinned to 0 at the anchor age, shape (S, K). This
        is the player-level counterpart to `position_curve_samples` and is
        what should be compared against a planted curve SHAPE (a true value
        that does not include the player's skill level)."""
        i = self.player_index[player_id]
        pdelta = self.samples["player_delta"][:, i, :]             # (S, K-1)
        S = np.concatenate(
            [np.zeros((pdelta.shape[0], 1)), np.cumsum(pdelta, axis=1)], axis=1
        )
        return S - S[:, [self.ref_idx]]

    def position_curve_samples(self, position: str) -> np.ndarray:
        """Posterior samples of the pooled position-group delta curve,
        cumulated to a curve shape (relative, no baseline), shape (S, K)."""
        g = self.position_index[position]
        pdelta = self.samples["position_delta"][:, g, :]           # (S, K-1)
        S = np.concatenate(
            [np.zeros((pdelta.shape[0], 1)), np.cumsum(pdelta, axis=1)], axis=1
        )
        return S - S[:, [self.ref_idx]]

    def r_hat_summary(self) -> dict[str, float]:
        """Max R-hat per site. Values well above 1.01 flag non-convergence
        -- the exact failure mode a hierarchical model risks silently."""
        summary = az.summary(self.idata, var_names=list(self.samples.keys()))
        out: dict[str, float] = {}
        for site in self.samples:
            # Match "site" (scalar) or "site[...]" (indexed) exactly -- a
            # plain startswith would also match e.g. "position_delta_raw"
            # when site == "position_delta".
            mask = (summary.index == site) | summary.index.str.startswith(site + "[")
            rows = summary[mask]
            out[site] = float(rows["r_hat"].max()) if len(rows) else float("nan")
        return out

    def posterior_predictive_coverage(self, prob: float = 0.9, seed: int = 0) -> float:
        """Fraction of observed ratings falling inside their `prob` central
        posterior-predictive interval. Should land near `prob` if the model
        is well specified; systematically low coverage means the model is
        overconfident (posterior too narrow), the specific failure a
        hierarchical model with a mis-set likelihood can produce silently.
        """
        baseline = self.samples["baseline"]                     # (S, N)
        player_delta = self.samples["player_delta"]              # (S, N, K-1)
        S = np.concatenate(
            [np.zeros((*player_delta.shape[:2], 1)), np.cumsum(player_delta, axis=2)],
            axis=2,
        )  # (S, N, K)
        mu_grid = baseline[:, :, None] + S - S[:, :, [self.ref_idx]]  # (S, N, K)
        mu_obs = mu_grid[:, self.obs_i, self.obs_a]                    # (S, n_obs)

        obs_sigma = self.samples["obs_sigma"][:, None]                 # (S, 1)
        eff_sd = obs_sigma / np.sqrt(np.clip(self.obs_w, 1e-6, None) / 1000.0)

        rng = np.random.default_rng(seed)
        draws = rng.normal(mu_obs, eff_sd)  # (S, n_obs), one predictive draw per posterior draw

        lo = np.quantile(draws, (1 - prob) / 2, axis=0)
        hi = np.quantile(draws, 1 - (1 - prob) / 2, axis=0)
        inside = (self.obs_y >= lo) & (self.obs_y <= hi)
        return float(np.mean(inside))


def _build_age_grid(age_min: int, age_max: int, anchor_age: int) -> tuple[list[int], int]:
    age_grid = list(range(age_min, age_max + 1))
    ref_idx = age_grid.index(anchor_age)
    return age_grid, ref_idx


def _model(obs_i, obs_a, obs_y, obs_w, ref_idx, position_of_player,
           n_players, n_positions, n_ages, baseline_prior_mean, baseline_prior_sd):
    sigma_position = numpyro.sample("sigma_position", dist.HalfNormal(2.0))
    sigma_player = numpyro.sample("sigma_player", dist.HalfNormal(2.0))
    obs_sigma = numpyro.sample("obs_sigma", dist.HalfNormal(5.0))

    with numpyro.plate("age_transition", n_ages - 1):
        league_delta = numpyro.sample("league_delta", dist.Normal(0.0, 5.0))

    # Non-centered parameterization: sample a raw N(0,1) and scale/shift it,
    # rather than sampling directly from Normal(parent, sigma). A centered
    # version here produces a classic hierarchical-model funnel -- when the
    # group-level sigma's posterior is small, `parent` and the raw value
    # become tightly correlated and NUTS cannot mix (observed as R-hat > 2
    # on league_delta in the first version of this model). The non-centered
    # form decouples them and is the standard fix.
    #
    # `.expand([n_ages - 1]).to_event(1)` puts the age-transition dimension
    # in the EVENT shape, not the batch shape, so it is invisible to the
    # plate's dimension bookkeeping -- the plate only ever has to reconcile
    # its own (n_positions or n_players) dimension, never collides with
    # another plate's leftover dimension, and needs no explicit `dim=`.
    with numpyro.plate("position", n_positions):
        position_delta_raw = numpyro.sample(
            "position_delta_raw", dist.Normal(0.0, 1.0).expand([n_ages - 1]).to_event(1)
        )
    position_delta = league_delta[None, :] + sigma_position * position_delta_raw
    numpyro.deterministic("position_delta", position_delta)
    # position_delta has shape (n_positions, n_ages - 1).

    with numpyro.plate("player", n_players):
        baseline = numpyro.sample(
            "baseline", dist.Normal(baseline_prior_mean, baseline_prior_sd)
        )
        player_delta_raw = numpyro.sample(
            "player_delta_raw", dist.Normal(0.0, 1.0).expand([n_ages - 1]).to_event(1)
        )
    player_delta = position_delta[position_of_player] + sigma_player * player_delta_raw
    numpyro.deterministic("player_delta", player_delta)

    cum = jnp.concatenate(
        [jnp.zeros((n_players, 1)), jnp.cumsum(player_delta, axis=1)], axis=1
    )
    mu_grid = baseline[:, None] + cum - cum[:, ref_idx][:, None]  # (n_players, n_ages)
    mu_obs = mu_grid[obs_i, obs_a]

    eff_sd = obs_sigma / jnp.sqrt(jnp.clip(obs_w, 1e-6) / 1000.0)
    numpyro.sample("obs", dist.Normal(mu_obs, eff_sd), obs=obs_y)


def fit_hierarchical_curve(
    seasons: list[PlayerSeason],
    rating_attr: str,
    weight_attr: str,
    *,
    age_min: int,
    age_max: int,
    anchor_age: int,
    baseline_prior_mean: dict[int, float] | None = None,
    baseline_prior_sd: float = 20.0,
    num_warmup: int = 400,
    num_samples: int = 400,
    num_chains: int = 2,
    seed: int = 0,
) -> HierarchicalCurveResult:
    """Fit the player -> position -> league partial-pooling model.

    `baseline_prior_mean` defaults to 0 for every player (uninformative --
    layer 2's setting). Layer 3 (`baseline.py`) calls this with a box-score
    -predicted mean instead, which is the only change needed to add the
    box-score baseline on top of this layer.
    """
    players = sorted({s.player_id for s in seasons})
    positions = sorted({s.position_group for s in seasons})
    player_index = {pid: i for i, pid in enumerate(players)}
    position_index = {pos: g for g, pos in enumerate(positions)}
    player_position: dict[int, str] = {}
    for s in seasons:
        player_position[s.player_id] = s.position_group

    age_grid, ref_idx = _build_age_grid(age_min, age_max, anchor_age)
    age_pos = {a: k for k, a in enumerate(age_grid)}

    obs_i = np.array([player_index[s.player_id] for s in seasons], dtype=np.int32)
    obs_a = np.array([age_pos[s.age] for s in seasons], dtype=np.int32)
    obs_y = np.array([getattr(s, rating_attr) for s in seasons], dtype=np.float64)
    obs_w = np.array([getattr(s, weight_attr) for s in seasons], dtype=np.float64)
    position_of_player = np.array(
        [position_index[player_position[pid]] for pid in players], dtype=np.int32
    )

    n_players = len(players)
    n_positions = len(positions)
    n_ages = len(age_grid)

    if baseline_prior_mean is None:
        prior_mean_arr = np.zeros(n_players)
    else:
        prior_mean_arr = np.array(
            [baseline_prior_mean.get(pid, 0.0) for pid in players], dtype=np.float64
        )

    kernel = NUTS(_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    import jax

    mcmc.run(
        jax.random.PRNGKey(seed),
        obs_i=jnp.array(obs_i), obs_a=jnp.array(obs_a),
        obs_y=jnp.array(obs_y), obs_w=jnp.array(obs_w),
        ref_idx=ref_idx, position_of_player=jnp.array(position_of_player),
        n_players=n_players, n_positions=n_positions, n_ages=n_ages,
        baseline_prior_mean=jnp.array(prior_mean_arr),
        baseline_prior_sd=baseline_prior_sd,
    )

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    idata = az.from_numpyro(mcmc)

    return HierarchicalCurveResult(
        samples=samples, player_index=player_index, position_index=position_index,
        player_position=player_position, age_grid=age_grid, ref_idx=ref_idx,
        anchor_age=anchor_age, idata=idata,
        obs_i=obs_i, obs_a=obs_a, obs_y=obs_y, obs_w=obs_w,
    )
