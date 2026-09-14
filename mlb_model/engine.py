"""Expected runs for one game.

    lam_side = league_R/G
             x off_ratio(batting team)       [shrunk, de-parked]
             x lineup_multiplier             [platoon vs SP hand]
             x def_ratio(opposing pitching)  [starter + rest-of-staff blend]
             x park_factor x weather x umpire
             x HFA (x home, / away, once)

All the context inputs are optional - anything missing just drops out as 1.0,
so a dead data source degrades the model instead of killing it, and the
backtest can toggle features off to measure what each one is worth.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field

from . import config
from .data import (BatterSplits, BullpenUsage, GameInfo, PitcherLine, Weather)
from .features import (LeagueContext, bullpen_fatigue_multiplier,
                       game_park_factor, lineup_off_multiplier,
                       starter_innings_share, umpire_run_multiplier,
                       weather_run_multiplier)

log = logging.getLogger("mlb_model.engine")


@dataclass
class GameContext:
    """Optional V2 inputs for one game. Everything defaults to 'absent'."""
    home_sp: PitcherLine | None = None
    away_sp: PitcherLine | None = None
    home_sp_hand: str = "R"
    away_sp_hand: str = "R"
    home_batters: dict[int, BatterSplits] = field(default_factory=dict)
    away_batters: dict[int, BatterSplits] = field(default_factory=dict)
    home_pen: BullpenUsage | None = None
    away_pen: BullpenUsage | None = None
    weather: Weather | None = None
    hp_umpire: str | None = None   # home-plate ump (None until announced)


@dataclass
class GamePrediction:
    game: GameInfo
    lam_home: float
    lam_away: float
    home_starter_note: str
    away_starter_note: str
    park_factor: float
    weather_mult: float = 1.0
    weather_note: str = ""
    home_lineup_note: str = ""
    away_lineup_note: str = ""
    fatigue_home: float = 0.0    # excess-IP fatigue index (feature for ML)
    fatigue_away: float = 0.0
    umpire_mult: float = 1.0
    umpire_note: str = "ump TBD"

    p_home: float | None = None
    exp_total: float | None = None
    p_over: float | None = None
    total_line: float | None = None
    # market comparison (filled by report.market_compare when odds exist)
    market: dict | None = None
    # ML-blended probability (filled by ml_layer when a model is loaded)
    p_home_ml: float | None = None


def expected_runs(ctx: LeagueContext, game: GameInfo,
                  gc: GameContext | None = None) -> GamePrediction | None:
    gc = gc or GameContext()
    hr, ar = ctx.rating(game.home_id), ctx.rating(game.away_id)
    if hr is None or ar is None:
        log.warning("Skipping %s @ %s - missing team rating",
                    game.away_name, game.home_name)
        return None

    pf = game_park_factor(game.home_id)

    # --- V2: bullpen fatigue -> rest-of-staff RA9 inflation
    h_fat_mult, h_fat_idx = bullpen_fatigue_multiplier(gc.home_pen)
    a_fat_mult, a_fat_idx = bullpen_fatigue_multiplier(gc.away_pen)

    def defense_ratio(starter: PitcherLine | None, team_id: int,
                      fat_mult: float) -> tuple[float, str]:
        sp_ra9, note = ctx.starter_ra9_regressed(starter, team_id)
        staff_ra9 = ctx.rest_of_staff_ra9(team_id, starter, fat_mult)
        # Per-pitcher innings share: an ace who works deep gets more of the
        # blend (and less bullpen dilution) than a short-leash starter.
        share = starter_innings_share(starter)
        blended = share * sp_ra9 + (1 - share) * staff_ra9
        return blended / ctx.league_ra9, f"{note} · {share:.0%}IP"

    # home_sp defends AGAINST the away offense and vice versa.
    away_faces, h_note = defense_ratio(gc.home_sp, game.home_id, h_fat_mult)
    home_faces, a_note = defense_ratio(gc.away_sp, game.away_id, a_fat_mult)

    # --- V2: lineup platoon multipliers (vs the OPPOSING starter's hand)
    h_lu_mult, h_lu_note = lineup_off_multiplier(
        gc.home_batters, game.home_lineup, gc.away_sp_hand, ctx.league_woba)
    a_lu_mult, a_lu_note = lineup_off_multiplier(
        gc.away_batters, game.away_lineup, gc.home_sp_hand, ctx.league_woba)

    # --- V2: weather (shared multiplier - it moves BOTH teams' runs, which
    # also induces the total-vs-side correlation v1 lacked)
    wx_mult, wx_note = weather_run_multiplier(gc.weather, game.home_id)

    # --- Umpire zone tendency: also a shared, symmetric total-mover.
    ump_mult, ump_note = umpire_run_multiplier(gc.hp_umpire)

    lam_home = (ctx.league_rs_pg * hr.off_ratio * h_lu_mult * home_faces
                * pf * wx_mult * ump_mult * config.HFA_RUN_MULT)
    lam_away = (ctx.league_rs_pg * ar.off_ratio * a_lu_mult * away_faces
                * pf * wx_mult * ump_mult / config.HFA_RUN_MULT)

    log.info("%s @ %s | lam %.2f-%.2f | PF %.2f wx %.3f ump %.2f | "
             "fat %.2f/%.2f | H:%s | A:%s", game.away_name, game.home_name,
             lam_away, lam_home, pf, wx_mult, ump_mult, h_fat_idx, a_fat_idx,
             h_note, a_note)
    return GamePrediction(
        game=game, lam_home=lam_home, lam_away=lam_away,
        home_starter_note=h_note, away_starter_note=a_note,
        park_factor=pf, weather_mult=wx_mult, weather_note=wx_note,
        home_lineup_note=h_lu_note, away_lineup_note=a_lu_note,
        fatigue_home=h_fat_idx, fatigue_away=a_fat_idx,
        umpire_mult=ump_mult, umpire_note=ump_note,
    )


def build_game_context(client, ctx: LeagueContext, game: GameInfo,
                       season: int, as_of=None,
                       use_lineups: bool = True, use_weather: bool = True,
                       use_fatigue: bool = True,
                       use_umpire: bool = True) -> GameContext:
    """Assemble all V2 inputs for a game with graceful fallbacks.
    Each feature degrades independently: a weather API outage does not cost
    us the lineup adjustment, etc."""
    gc = GameContext()
    gc.home_sp = (client.pitcher_line(game.home_prob_id, season, as_of=as_of)
                  if game.home_prob_id else None)
    gc.away_sp = (client.pitcher_line(game.away_prob_id, season, as_of=as_of)
                  if game.away_prob_id else None)

    hands = client.pitcher_hands([game.home_prob_id, game.away_prob_id])
    gc.home_sp_hand = hands.get(game.home_prob_id, "R")
    gc.away_sp_hand = hands.get(game.away_prob_id, "R")

    if use_lineups and (game.home_lineup or game.away_lineup):
        tag = f"splits-{game.game_date}"
        gc.home_batters = client.batter_splits(game.home_lineup, season, tag)
        gc.away_batters = client.batter_splits(game.away_lineup, season, tag)

    if use_fatigue:
        ref_date = as_of or _dt.date.fromisoformat(game.game_date)
        gc.home_pen = client.bullpen_usage(game.home_id, ref_date)
        gc.away_pen = client.bullpen_usage(game.away_id, ref_date)

    if use_weather and game.game_time_utc:
        gc.weather = client.weather(game.home_id, game.game_time_utc)

    if use_umpire:
        gc.hp_umpire = client.hp_umpire(game.game_pk,
                                        game_final=game.state == "Final")
    return gc
