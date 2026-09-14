"""Edge-uncertainty-adjusted fractional Kelly staking.

Full Kelly maximizes long-run growth ONLY if the win probability is exactly
right. A model's probability never is, and full Kelly on an *estimated* edge
systematically over-bets: the arithmetic that maximizes growth under a known
edge maximizes ruin under an over-estimated one. So three guards, in order:

  1. Shrink the model's edge toward the market before sizing. The market is a
     strong prior; a 6-point disagreement is more likely 3 points of real edge
     plus 3 of model error. `edge_shrink` = how far to pull model->market.
  2. Fractional Kelly (quarter by default) on the shrunk probability.
  3. A hard per-bet cap, and a CLV gate: until the model's flags are shown to
     beat the closing line, staking is clamped hard no matter what EV says.

None of this is bettable size until CLV is demonstrated (backtest.py). This
module turns "the model says +EV" into a number that survives being wrong.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stake:
    fraction: float        # fraction of bankroll to wager (0 = no bet)
    amount: float          # fraction * bankroll
    kelly_full: float      # full-Kelly on the RAW model prob, for reference
    p_used: float          # uncertainty-shrunk probability actually sized on
    capped: bool           # True if the hard cap bound the stake
    note: str


def american_to_decimal(ml: float) -> float:
    """American odds -> decimal (total return per 1 staked, stake included)."""
    ml = float(ml)
    return 1.0 + (ml / 100.0 if ml > 0 else 100.0 / -ml)


def full_kelly(p: float, dec_odds: float) -> float:
    """Growth-optimal fraction for a simple win/lose bet. b = dec_odds - 1."""
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * b - (1.0 - p)) / b          # = p - (1-p)/b
    return max(0.0, f)


def size_bet(p_model: float,
             american_odds: float,
             *,
             p_market: float | None = None,
             bankroll: float = 1.0,
             kelly_fraction: float = 0.25,
             edge_shrink: float = 0.5,
             max_fraction: float = 0.02,
             clv_proven: bool = False) -> Stake:
    """Return a defensible stake for one bet.

    p_model       : model win probability for the side being bet.
    american_odds : the price on that side (e.g. +235).
    p_market      : de-vigged market prob for that side. If given, the model
                    edge is shrunk toward it by `edge_shrink` BEFORE sizing.
    kelly_fraction: fraction of full Kelly (0.25 = quarter-Kelly).
    edge_shrink   : 0 = trust model fully, 1 = collapse to market (no bet).
    max_fraction  : hard cap on fraction of bankroll for any single bet.
    clv_proven    : until True, kelly_fraction and cap are force-tightened -
                    an unvalidated model does not get full-size stakes.
    """
    dec = american_to_decimal(american_odds)
    kelly_full = full_kelly(p_model, dec)

    # 1) shrink the edge toward the market prior
    if p_market is not None:
        p_used = p_market + (1.0 - edge_shrink) * (p_model - p_market)
    else:
        # no market anchor: penalize by shrinking probability toward the
        # bet's own breakeven (implied) prob instead.
        p_be = 1.0 / dec
        p_used = p_be + (1.0 - edge_shrink) * (p_model - p_be)

    # 2) fractional Kelly on the shrunk probability
    kf = kelly_fraction
    cap = max_fraction
    if not clv_proven:                    # unvalidated -> tighten both guards
        kf = min(kf, 0.25)
        cap = min(cap, 0.01)
    frac = full_kelly(p_used, dec) * kf

    # 3) hard cap
    capped = frac > cap
    frac = min(frac, cap)
    if frac <= 0:
        note = "no bet: edge non-positive after shrink"
    elif not clv_proven:
        note = f"UNVALIDATED model -> clamped to {kf:g}x Kelly, cap {cap:.1%}"
    else:
        note = f"{kf:g}x Kelly on shrunk p={p_used:.3f}"
    return Stake(fraction=frac, amount=frac * bankroll, kelly_full=kelly_full,
                 p_used=p_used, capped=capped, note=note)


# --------------------------------------------------------------- self-check
def _selfcheck() -> None:
    """Sanity assertions (run: python -m mlb_model.bet_sizing)."""
    # Fair coin at even money -> zero Kelly.
    assert abs(full_kelly(0.5, 2.0)) < 1e-12
    # Known Kelly: p=0.6 at even money (b=1) -> f = 2p-1 = 0.20.
    assert abs(full_kelly(0.6, 2.0) - 0.20) < 1e-9
    # KC-style dog: model 0.347 at +235 (dec 3.35), market 0.299, unvalidated.
    s = size_bet(0.347, 235, p_market=0.299, bankroll=120, clv_proven=False)
    assert s.p_used < 0.347                      # shrunk toward market
    assert 0 < s.fraction <= 0.01                # clamped for unvalidated
    assert s.amount <= 1.2 + 1e-9                # <= 1% of 120
    # Negative edge -> no bet.
    assert size_bet(0.28, 235, p_market=0.30).fraction == 0.0
    # Proven model gets a bigger (but still capped) stake than unvalidated.
    proven = size_bet(0.347, 235, p_market=0.299, clv_proven=True,
                      kelly_fraction=0.5, max_fraction=0.05)
    unval = size_bet(0.347, 235, p_market=0.299, clv_proven=False,
                     kelly_fraction=0.5, max_fraction=0.05)
    assert proven.fraction > unval.fraction
    print("bet_sizing self-check passed")
    print(f"  KC $120 example -> stake ${size_bet(0.347, 235, p_market=0.299, bankroll=120).amount:.2f}")


if __name__ == "__main__":
    _selfcheck()
