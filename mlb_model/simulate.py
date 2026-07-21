"""Monte Carlo game sim.

Not plain Poisson: runs cluster within innings, so each sim draws a gamma
multiplier on lambda (negative binomial marginal, ~1.3x variance) - plain
Poisson under-prices blowouts. Game structure matters for totals: home team
skips the bottom 9th with a lead, walk-off wins end at +1, extras use
ghost-runner scoring (~1 run/team/inning) in a vectorized loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import config

log = logging.getLogger("mlb_model.simulate")


@dataclass
class SimResult:
    p_home: float
    exp_total: float
    total_dist: np.ndarray   # final combined runs per sim (for any total line)
    p_extra_innings: float

    def p_over(self, line: float) -> float:
        # Pushes (total == integer line) excluded from both sides, as settled.
        return float((self.total_dist > line).mean())

    def p_under(self, line: float) -> float:
        return float((self.total_dist < line).mean())


def simulate_game(lam_home: float, lam_away: float,
                  n_sims: int = config.N_SIMS,
                  rng: np.random.Generator | None = None) -> SimResult:
    rng = rng or np.random.default_rng(config.RNG_SEED)
    n = n_sims

    # --- Overdispersion: per-sim Gamma multiplier (mean 1) on each lambda.
    # shape k = 1/(DISPERSION*lam) gives Var(runs) = lam*(1+DISPERSION*lam).
    def disp_lambda(lam):
        k = 1.0 / max(config.DISPERSION * lam, 1e-9)
        return lam * rng.gamma(k, 1.0 / k, n)

    lh, la = disp_lambda(lam_home), disp_lambda(lam_away)

    # --- Regulation: away bats 9 innings; home bats 8 + (maybe) the 9th.
    away9 = rng.poisson(la, n)
    home8 = rng.poisson(lh * 8.0 / 9.0, n)
    home_9th = rng.poisson(lh / 9.0, n)

    # Bottom 9th played only if home is NOT already ahead after the top.
    plays_b9 = home8 <= away9
    home9 = home8 + np.where(plays_b9, home_9th, 0)

    # Walk-off in the 9th: home stops the moment it leads (cap at away+1).
    walkoff_b9 = plays_b9 & (home9 > away9)
    home9 = np.where(walkoff_b9, np.minimum(home9, away9 + 1), home9)

    home_total = home9.astype(float)
    away_total = away9.astype(float)
    home_win = home9 > away9
    tied = home9 == away9
    p_extras = float(tied.mean())

    # --- Extra innings: loop while tied, placed-runner scoring rate.
    lam_xh = lam_home / 9.0 + config.MANFRED_RUNNER_BONUS
    lam_xa = lam_away / 9.0 + config.MANFRED_RUNNER_BONUS
    for _ in range(config.MAX_EXTRA_INNINGS):
        idx = np.flatnonzero(tied)
        if idx.size == 0:
            break
        xa = rng.poisson(lam_xa, idx.size)  # top half
        xh = rng.poisson(lam_xh, idx.size)  # bottom half
        # Home walk-off cap again: stop when the lead is taken.
        xh = np.where(xh > xa, np.minimum(xh, xa + 1), xh)
        away_total[idx] += xa
        home_total[idx] += xh
        won = xh != xa
        home_win[idx[won]] = xh[won] > xa[won]
        still = idx[~won]
        tied[:] = False
        tied[still] = True
    else:
        # Marathon leftovers (<0.1% of sims): empirical home extras edge.
        idx = np.flatnonzero(tied)
        if idx.size:
            home_win[idx] = rng.random(idx.size) < config.HFA_EXTRAS

    total = home_total + away_total
    return SimResult(
        p_home=float(home_win.mean()),
        exp_total=float(total.mean()),
        total_dist=total,
        p_extra_innings=p_extras,
    )
