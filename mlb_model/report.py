"""Output tables + market comparison.

De-vig is proportional (divide both implied probs by their sum) - simple and
fine for MLB moneylines, which are never extreme. Edge = model prob minus
de-vigged market prob; >= 3.5% with positive EV gets the HIGH-EV flag,
roughly where a calibrated model would clear typical vig.

Table layout uses team codes + starter surnames so nothing wraps, and status
is fixed columns (WX/PEN/LU/SP/UMP) instead of a crammed comma string.
"""
from __future__ import annotations

import logging

from . import config
from .engine import GamePrediction

log = logging.getLogger("mlb_model.report")


# ------------------------------------------------------------------- odds
def american_implied(ml: float) -> float:
    """Implied probability (vig included) of American odds."""
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def american_payout(ml: float) -> float:
    """Profit per $1 stake if the bet wins."""
    ml = float(ml)
    return ml / 100.0 if ml > 0 else 100.0 / -ml


def devig_proportional(ml_home: float, ml_away: float) -> tuple[float, float]:
    """Remove the vig proportionally: fair probs that sum to 1."""
    ih, ia = american_implied(ml_home), american_implied(ml_away)
    s = ih + ia
    if s <= 0:
        return 0.5, 0.5
    return ih / s, ia / s


def fair_moneyline(p: float) -> str:
    p = min(max(p, 0.001), 0.999)
    return (f"-{round(100 * p / (1 - p)):d}" if p >= 0.5
            else f"+{round(100 * (1 - p) / p):d}")


# ---------------------------------------------------------- market compare
def market_compare(pred: GamePrediction, odds_row: dict) -> dict | None:
    """Compare model fair value to a market line; stored on pred.market."""
    try:
        ml_h, ml_a = float(odds_row["home_ml"]), float(odds_row["away_ml"])
    except (KeyError, ValueError, TypeError):
        log.warning("Bad odds row for %s @ %s",
                    pred.game.away_name, pred.game.home_name)
        return None
    p_model = pred.p_home_ml if pred.p_home_ml is not None else pred.p_home
    mkt_h, mkt_a = devig_proportional(ml_h, ml_a)
    edge_h = p_model - mkt_h
    edge_a = (1 - p_model) - mkt_a
    ev_h = p_model * american_payout(ml_h) - (1 - p_model)
    ev_a = (1 - p_model) * american_payout(ml_a) - p_model
    side, edge, ev, ml = (("home", edge_h, ev_h, ml_h)
                          if edge_h >= edge_a else ("away", edge_a, ev_a, ml_a))

    # Data-completeness gate: if EITHER starter is prior-only (TBD / no-low
    # season data), the defense side of this game is a guess and the edge is
    # provisional - it can be almost entirely a missing-starter artifact and
    # will move when the arm posts. Hold it to a higher bar before we call it
    # HIGH-EV, so the card doesn't fire on numbers that aren't real yet.
    sp_provisional = ("no/low" in pred.home_starter_note
                      or "no/low" in pred.away_starter_note)
    threshold = (config.EV_THRESHOLD_PROVISIONAL_SP if sp_provisional
                 else config.EV_THRESHOLD)
    result = {
        "home_ml": ml_h, "away_ml": ml_a,
        "market_p_home": mkt_h,
        "edge_home": edge_h, "edge_away": edge_a,
        "ev_home": ev_h, "ev_away": ev_a,
        "best_side": side, "best_edge": edge, "best_ev": ev, "best_ml": ml,
        "sp_provisional": sp_provisional,
        "high_ev": edge >= threshold and ev > 0,
    }
    pred.market = result
    return result


# ------------------------------------------------------------ label helpers
def team_abbr(team_id: int, fallback_name: str = "") -> str:
    return config.TEAM_ABBREV.get(team_id, (fallback_name[:3] or "???").upper())


def _surname(full_name: str) -> str:
    return full_name.split()[-1] if full_name and full_name != "TBD" else "TBD"


def fatigue_pct(fatigue_idx: float) -> float:
    """Excess-relief-IP fatigue index -> the RA9 inflation % it produces,
    using the exact same cap/rate as features.bullpen_fatigue_multiplier so
    the number shown in the table equals the number actually applied to runs."""
    return min(fatigue_idx * config.FATIGUE_RA9_PCT_PER_EXTRA_IP,
               config.FATIGUE_RA9_PCT_CAP)


# A pen tax below this (~+1% RA9) is not worth surfacing; above it we show the
# actual magnitude for EACH side so the away bullpen is never invisible.
FATIGUE_SHOW_IDX = 0.5


def matchup_label(p: GamePrediction, width: int = 30) -> str:
    """'MIN Ryan @ CLE Bibee' - fits a fixed column, never wraps."""
    g = p.game
    label = (f"{team_abbr(g.away_id, g.away_name)} {_surname(g.away_prob_name)}"
             f" @ {team_abbr(g.home_id, g.home_name)} "
             f"{_surname(g.home_prob_name)}")
    return label[:width - 1] + "…" if len(label) > width else label


# ------------------------------------------------- structured status columns
def status_columns(p: GamePrediction) -> dict[str, str]:
    """One short, aligned indicator per data dimension.
    '--' always means 'nothing notable', never 'error'."""
    wx = f"{(p.weather_mult - 1) * 100:+3.0f}%" if p.weather_mult != 1.0 else "  --"
    pen = {(False, False): " --", (True, False): "H ", (False, True): " A",
           (True, True): "HA"}[(p.fatigue_home > FATIGUE_SHOW_IDX,
                                p.fatigue_away > FATIGUE_SHOW_IDX)]
    lu = "Y " if p.game.home_lineup or p.game.away_lineup else "--"
    sp_h = "no/low" in p.home_starter_note
    sp_a = "no/low" in p.away_starter_note
    sp = {(False, False): "ok", (True, False): "H?", (False, True): "A?",
          (True, True): "??"}[(sp_h, sp_a)]
    ump = (f"{(p.umpire_mult - 1) * 100:+3.0f}%" if p.umpire_mult != 1.0
           else "  --")
    return {"wx": wx, "pen": pen, "lu": lu, "sp": sp, "ump": ump}


STATUS_LEGEND = (
    "  WX  weather run effect vs neutral      PEN taxed bullpen (H=home A=away)\n"
    "  LU  posted lineups used (Y) or not(--) SP  starter data (H?/A? = prior only)\n"
    "  UMP home-plate umpire zone effect      --  = nothing notable")


# ------------------------------------------------------------ slate output
def slate_table(preds: list[GamePrediction], total_line: float) -> str:
    """Fixed-width slate table. Column plan (chars):
    matchup 30 | H WIN 6 | FAIR ML 13 | xTOT 5 | O-line 6 | WX 4 | PEN 3 |
    LU 2 | SP 2 | UMP 4 [| MKT H 6 | EDGE 6 | flag]"""
    has_market = any(p.market for p in preds)
    o_hdr = f"O{total_line:g}"
    hdr = (f"  {'MATCHUP':<30}  {'H WIN':>6}  {'FAIR ML (A/H)':^13}  "
           f"{'xTOT':>5}  {o_hdr:>6}   {'WX':>4}  {'PEN':>3}  {'LU':>2}  "
           f"{'SP':>2}  {'UMP':>4}")
    if has_market:
        hdr += f"  {'MKT H':>6}  {'EDGE':>6}"
    lines = [hdr, "  " + "-" * (len(hdr) - 2)]

    for p in preds:
        p_show = p.p_home_ml if p.p_home_ml is not None else p.p_home
        ml_col = f"{fair_moneyline(1 - p_show):>5} /{fair_moneyline(p_show):>5}"
        s = status_columns(p)
        row = (f"  {matchup_label(p):<30}  {p_show:>6.1%}  {ml_col:^13}  "
               f"{p.exp_total:>5.1f}  {p.p_over:>6.1%}   {s['wx']:>4}  "
               f"{s['pen']:>3}  {s['lu']:>2}  {s['sp']:>2}  {s['ump']:>4}")
        if has_market:
            if p.market:
                row += (f"  {p.market['market_p_home']:>6.1%}  "
                        f"{p.market['best_edge']:>+6.1%}")
                if p.market["high_ev"]:
                    row += f"  << HIGH-EV {p.market['best_side'].upper()}"
            else:
                row += f"  {'--':>6}  {'--':>6}"
        lines.append(row)

    lines += ["", STATUS_LEGEND]
    return "\n".join(lines)


def ev_report(preds: list[GamePrediction]) -> str:
    """Detail block for flagged HIGH EV spots - with mandatory caveats."""
    flagged = [p for p in preds if p.market and p.market["high_ev"]]
    if not flagged:
        return ("\n  No HIGH-EV spots vs current lines (model within "
                f"{config.EV_THRESHOLD:.1%} of the de-vigged market\n"
                "  everywhere - which is the normal, expected outcome).")
    lines = ["", f"  HIGH-EV candidates (edge >= {config.EV_THRESHOLD:.1%})",
             "  " + "-" * 66,
             f"  {'SIDE':<26} {'ML':>6}  {'MODEL':>6}  {'MKT':>6}  "
             f"{'EDGE':>6}  {'EV/unit':>8}"]
    for p in sorted(flagged, key=lambda x: -x.market["best_edge"]):
        m = p.market
        side_id = (p.game.home_id if m["best_side"] == "home" else p.game.away_id)
        side_nm = (p.game.home_name if m["best_side"] == "home"
                   else p.game.away_name)
        p_model = p.p_home_ml if p.p_home_ml is not None else p.p_home
        p_side = p_model if m["best_side"] == "home" else 1 - p_model
        p_mkt = (m["market_p_home"] if m["best_side"] == "home"
                 else 1 - m["market_p_home"])
        prov = "  ⚠ SP TBD" if m.get("sp_provisional") else ""
        label = f"{team_abbr(side_id, side_nm)}  ({matchup_label(p, 20)})"
        lines.append(f"  {label:<26} {m['best_ml']:>+6.0f}  {p_side:>6.1%}  "
                     f"{p_mkt:>6.1%}  {m['best_edge']:>+6.1%}  "
                     f"{m['best_ev']:>+8.3f}{prov}")
    lines += ["",
              "  CAVEAT: an 'edge' is model-vs-market disagreement, not free "
              "money. It is only",
              "  real if the model is calibrated (backtest.py) AND these "
              "flags beat the closing",
              "  line on average (CLV). Until both are demonstrated, treat "
              "HIGH-EV as 'find out",
              "  why the market disagrees with us' - the market is usually "
              "right."]
    return "\n".join(lines)


def slate_footer() -> str:
    return ("\n  Caveats: intraday injuries/roster moves not seen; lineup "
            "adjustment needs posted\n  lineups; weather is forecast-based. "
            "Full data trail in mlb_model.log. See\n  backtest.py for "
            "calibration and CLV evidence before acting on any number above.")
