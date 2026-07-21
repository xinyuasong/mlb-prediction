"""Raw API lines -> regressed, park-adjusted ratings.

Core idea everywhere: shrink toward a prior before trusting anything.
Team rates shrink toward league average, starters toward their projection
(more ballast - projections already encode multi-year data), platoon splits
toward overall wOBA + league platoon offset. Also: de-parking, rest-of-staff
RA9, bullpen fatigue, weather and umpire multipliers, final calibration.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from . import config
from .data import BatterSplits, BullpenUsage, PitcherLine, TeamSeasonLine, Weather

log = logging.getLogger("mlb_model.features")


def shrink(observed_rate: float, sample_size: float,
           prior_rate: float, ballast: float) -> float:
    """Posterior mean rate: (obs*n + prior*ballast) / (n + ballast)."""
    n = max(sample_size, 0.0)
    return (observed_rate * n + prior_rate * ballast) / (n + ballast)


def calibrate_prob(p: float) -> float:
    """sigmoid(a * logit(p)) with a fitted from backtest calibration bins.

    a > 1 stretches probs away from 50%, undoing the compression that input
    shrinkage causes. Runs last so everything downstream sees one number."""
    a = config.CALIBRATION_LOGIT_SCALE
    if a == 1.0:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = a * math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-z))


def park_env(team_id: int) -> float:
    """Average park environment a team plays in (~half home, half neutral)."""
    pf = config.PARK_FACTORS.get(team_id)
    if pf is None:
        log.warning("No park factor for team %s; assuming neutral", team_id)
        pf = 100
    return (pf / 100.0 + 1.0) / 2.0


def game_park_factor(home_team_id: int) -> float:
    pf = config.PARK_FACTORS.get(home_team_id)
    if pf is None:
        log.warning("No park factor for venue of team %s; using neutral",
                    home_team_id)
        pf = 100
    return pf / 100.0


@dataclass
class TeamRating:
    """Park-neutral, regressed team ratings, as ratios to league average."""
    team_id: int
    name: str
    games: int
    off_ratio: float
    def_ratio: float
    raw_rs_pg: float
    raw_ra9: float


class LeagueContext:
    """League baselines + per-team ratings built from one set of team lines."""

    def __init__(self, lines: dict[int, TeamSeasonLine],
                 league_woba: float | None = None,
                 projections: dict[int, float] | None = None):
        self.league_woba = league_woba or config.LEAGUE_WOBA_FALLBACK
        self.projections = projections or {}
        usable = {tid: t for tid, t in lines.items()
                  if t.games >= config.MIN_TEAM_GAMES and t.innings_pitched > 0}
        dropped = set(lines) - set(usable)
        if dropped:
            log.warning("Teams below %d games excluded from league baseline: %s",
                        config.MIN_TEAM_GAMES, sorted(dropped))
        if not usable:
            raise ValueError("No usable team data - too early in season?")

        self.league_rs_pg = (sum(t.runs_scored for t in usable.values())
                             / sum(t.games for t in usable.values()))
        self.league_ra9 = (9.0 * sum(t.runs_allowed for t in usable.values())
                          / sum(t.innings_pitched for t in usable.values()))
        log.info("League: %.2f R/G, %.2f RA9, wOBA %.3f, %d teams, "
                 "%d projections", self.league_rs_pg, self.league_ra9,
                 self.league_woba, len(usable), len(self.projections))

        self.ratings: dict[int, TeamRating] = {}
        for tid, t in usable.items():
            env = park_env(tid)
            rs_pg = t.runs_scored / t.games
            ra9 = 9.0 * t.runs_allowed / t.innings_pitched
            off = shrink(rs_pg / env, t.games,
                         self.league_rs_pg, config.TEAM_BALLAST_GAMES)
            dfn = shrink(ra9 / env, t.games,
                         self.league_ra9, config.TEAM_BALLAST_GAMES)
            self.ratings[tid] = TeamRating(
                team_id=tid, name=t.name, games=t.games,
                off_ratio=off / self.league_rs_pg,
                def_ratio=dfn / self.league_ra9,
                raw_rs_pg=rs_pg, raw_ra9=ra9,
            )
        self._lines = usable

    # ------------------------------------------------------------- pitching
    def starter_ra9_regressed(self, p: PitcherLine | None,
                              team_id: int | None) -> tuple[float, str]:
        """Regressed, park-neutral starter RA9, with projection prior when
        available (V2). Returns (ra9, source_tag) - the tag keeps fallbacks
        visible so a confident number can't hide a data gap."""
        proj = self.projections.get(p.pitcher_id) if p else None
        # Prior: projection if we have one, else league average.
        if proj is not None:
            # Projections arrive on the ERA/FIP scale converted to RA9 but
            # reflect a NEUTRAL park by construction; recentre on the actual
            # league run environment so a hot/cold league doesn't bias us.
            prior = proj * (self.league_ra9 / 4.30) if proj > 0 else self.league_ra9
            ballast = config.STARTER_BALLAST_IP_WITH_PROJ
            prior_tag = f"proj {proj:.2f}"
        else:
            prior = self.league_ra9
            ballast = config.STARTER_BALLAST_IP
            prior_tag = "lg-avg prior"

        if p is None or p.ip < config.STARTER_MIN_IP:
            ip = 0.0 if p is None else p.ip
            log.info("Starter %s: %.1f IP < %.0f - using prior only (%s)",
                     getattr(p, "name", "TBD"), ip, config.STARTER_MIN_IP,
                     prior_tag)
            return prior, f"{prior_tag}, no/low season data"
        env = park_env(team_id) if team_id is not None else 1.0
        neutral = p.ra9 / env
        reg = shrink(neutral, p.ip, prior, ballast)
        return reg, f"{p.ip:.0f} IP raw {p.ra9:.2f} -> reg {reg:.2f} ({prior_tag})"

    def rest_of_staff_ra9(self, team_id: int, starter: PitcherLine | None,
                          fatigue_mult: float = 1.0) -> float:
        """Team RA9 excluding tonight's starter, park-neutral, regressed,
        multiplied by the bullpen fatigue factor (V2)."""
        t = self._lines.get(team_id)
        if t is None:
            return self.league_ra9 * fatigue_mult
        runs, ip = t.runs_allowed, t.innings_pitched
        if starter is not None and starter.ip < ip:
            runs -= starter.runs
            ip -= starter.ip
        if ip <= 50:
            return self.league_ra9 * fatigue_mult
        neutral = (9.0 * runs / ip) / park_env(team_id)
        return shrink(neutral, ip, self.league_ra9, 360.0) * fatigue_mult

    def rating(self, team_id: int) -> TeamRating | None:
        r = self.ratings.get(team_id)
        if r is None:
            log.warning("No rating for team %s", team_id)
        return r


# ===========================================================================
# V2: wOBA + platoon lineup adjustment
# ===========================================================================
def woba_from_counts(st: dict) -> tuple[float | None, float]:
    """(wOBA, PA) from an API counting-stat dict; (None, 0) if unusable."""
    try:
        bb = int(st.get("baseOnBalls", 0))
        ibb = int(st.get("intentionalWalks", 0))
        hbp = int(st.get("hitByPitch", 0))
        h = int(st.get("hits", 0))
        d2 = int(st.get("doubles", 0))
        d3 = int(st.get("triples", 0))
        hr = int(st.get("homeRuns", 0))
        ab = int(st.get("atBats", 0))
        sf = int(st.get("sacFlies", 0))
    except (TypeError, ValueError):
        return None, 0.0
    denom = ab + (bb - ibb) + hbp + sf
    if denom <= 0:
        return None, 0.0
    w = config.WOBA_WEIGHTS
    num = (w["uBB"] * (bb - ibb) + w["HBP"] * hbp + w["1B"] * (h - d2 - d3 - hr)
           + w["2B"] * d2 + w["3B"] * d3 + w["HR"] * hr)
    return num / denom, float(denom)


def batter_woba_vs_hand(b: BatterSplits, pitcher_hand: str,
                        league_woba: float) -> float:
    """Regressed wOBA for this batter against a pitcher of the given hand.

    Two-stage shrinkage, because platoon splits are notoriously noisy:
    1. Overall wOBA (both splits pooled) shrunk toward league (220 PA ballast).
    2. The split itself shrunk toward (overall + league platoon offset) with
       a heavy 600 PA ballast - i.e., unless a player has a big split sample,
       we mostly assume he has a LEAGUE-AVERAGE platoon split. Switch
       hitters always bat with the advantage."""
    w_vl, pa_vl = woba_from_counts(b.vs.get("vl", {}))
    w_vr, pa_vr = woba_from_counts(b.vs.get("vr", {}))
    tot_pa = pa_vl + pa_vr
    if tot_pa <= 0:
        return league_woba  # no data at all: league-average bat
    pooled = ((w_vl or 0) * pa_vl + (w_vr or 0) * pa_vr) / tot_pa
    overall = shrink(pooled, tot_pa, league_woba,
                     config.WOBA_OVERALL_BALLAST_PA)

    has_advantage = (b.bat_side == "S"
                     or (b.bat_side == "L" and pitcher_hand == "R")
                     or (b.bat_side == "R" and pitcher_hand == "L"))
    offset = (config.PLATOON_OFFSET_ADV if has_advantage
              else config.PLATOON_OFFSET_DIS)
    split_prior = overall + offset

    code = "vl" if pitcher_hand == "L" else "vr"
    w_split, pa_split = (w_vl, pa_vl) if code == "vl" else (w_vr, pa_vr)
    if w_split is None:
        return split_prior
    return shrink(w_split, pa_split, split_prior, config.WOBA_SPLIT_BALLAST_PA)


def lineup_off_multiplier(lineup: dict[int, BatterSplits],
                          batting_order: list[int],
                          pitcher_hand: str, league_woba: float) -> tuple[float, str]:
    """Offense multiplier from the actual lineup vs the starter's hand.

    lineup wOBA ratio -> run ratio via elasticity ~1.9 (BaseRuns-scale),
    then blended at LINEUP_ADJ_WEIGHT with 1.0 (the season-team-rate view)
    because bench innings + our approximations deserve humility.
    Returns (multiplier, note)."""
    if not batting_order:
        return 1.0, "no lineup posted"
    wobas = []
    for pid in batting_order[:9]:
        b = lineup.get(pid)
        wobas.append(batter_woba_vs_hand(b, pitcher_hand, league_woba)
                     if b else league_woba)
    if not wobas:
        return 1.0, "no lineup data"
    ratio = (sum(wobas) / len(wobas)) / league_woba
    run_ratio = ratio ** config.WOBA_RUN_ELASTICITY
    mult = 1.0 + config.LINEUP_ADJ_WEIGHT * (run_ratio - 1.0)
    # Clamp: a lineup adjustment beyond +/-12% means bad data, not talent.
    mult = min(max(mult, 0.88), 1.12)
    return mult, (f"lineup wOBA {sum(wobas)/len(wobas):.3f} vs {pitcher_hand}HP "
                  f"-> x{mult:.3f}")


# ===========================================================================
# V2: bullpen fatigue index
# ===========================================================================
def bullpen_fatigue_multiplier(u: BullpenUsage | None) -> tuple[float, float]:
    """(RA9 multiplier >= 1, raw fatigue index in excess IP).

    Excess relief workload over the per-game-played norm (measured league
    average ~3.8 relief IP/game) inflates rest-of-staff RA9: +2%/excess IP,
    capped +12%. Pitch counts refine the index: 20 extra pitches ~ 1 extra
    IP of fatigue. Norm scales with games actually played, so off-days
    count as recovery."""
    if u is None or u.games_counted == 0:
        return 1.0, 0.0
    norm_ip = config.BULLPEN_NORM_RELIEF_IP_PER_GAME * u.games_counted
    norm_pitches = config.BULLPEN_NORM_PITCHES_PER_GAME * u.games_counted
    excess_ip = max(0.0, u.relief_ip_3d - norm_ip)
    excess_pitch_ip = max(0.0, (u.relief_pitches_3d - norm_pitches) / 20.0)
    fatigue = max(excess_ip, excess_pitch_ip)  # take the harsher signal
    pct = min(fatigue * config.FATIGUE_RA9_PCT_PER_EXTRA_IP,
              config.FATIGUE_RA9_PCT_CAP)
    return 1.0 + pct, fatigue


# ===========================================================================
# V2: weather -> run environment multiplier
# ===========================================================================
def weather_run_multiplier(w: Weather | None, home_team_id: int) -> tuple[float, str]:
    """Run multiplier from temperature + wind, open-air parks only.

    Temperature: ~+2.5%/10F vs 70F (ball carry + grip). Wind: component of
    the wind vector blowing OUT toward center field, ~+2%/5mph; blowing in
    is symmetric. wind_dir_deg is where wind comes FROM, so the direction
    it blows TOWARD is dir+180; we compare that to the park's home-plate ->
    center-field bearing."""
    if w is None:
        return 1.0, "no weather (dome/unavailable)"
    venue = config.VENUES.get(home_team_id)
    if venue is None:
        return 1.0, "unknown venue"
    _, _, roof, cf_bearing = venue
    temp_component = (w.temp_f - config.TEMP_BASELINE_F) * config.TEMP_RUN_PCT_PER_F
    blow_toward = (w.wind_dir_deg + 180.0) % 360.0
    # cos of angle between wind vector and CF direction: +1 straight out.
    out_component = math.cos(math.radians(blow_toward - cf_bearing))
    wind_component = w.wind_mph * out_component * config.WIND_RUN_PCT_PER_MPH_OUT
    mult = 1.0 + temp_component + wind_component
    mult = min(max(mult, config.WEATHER_MULT_MIN), config.WEATHER_MULT_MAX)
    note = (f"{w.temp_f:.0f}F, wind {w.wind_mph:.0f}mph "
            f"{'out' if out_component > 0 else 'in'} -> x{mult:.3f} ({w.source})")
    return mult, note


# ===========================================================================
# Umpire tendency -> run environment multiplier
# ===========================================================================
def umpire_run_multiplier(ump_name: str | None) -> tuple[float, str]:
    """Run multiplier for the home-plate umpire's zone tendency.

    Tight-zone umpires walk more hitters -> more runs (>1.0); big-zone
    umpires suppress them (<1.0). Factors live in config.UMPIRE_RUN_FACTORS
    (name-keyed, curated from public umpire data - keep it updated). The
    effect is small and capped at +/-3%: umpire influence is real but
    routinely overestimated by bettors. Unknown or unannounced umpire ->
    exactly neutral, clearly labeled."""
    if not ump_name:
        return 1.0, "ump TBD"
    factor = config.UMPIRE_RUN_FACTORS.get(ump_name.strip().lower())
    if factor is None:
        return 1.0, f"ump {ump_name} (no data, neutral)"
    factor = min(max(factor, config.UMPIRE_MULT_MIN), config.UMPIRE_MULT_MAX)
    return factor, f"ump {ump_name} x{factor:.2f}"
