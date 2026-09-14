"""
Model configuration. Every constant documents WHY it has its value.
Tune these only with backtest evidence (see backtest.py).
"""

API_BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1  # MLB

# ---------------------------------------------------------------------------
# Shrinkage (regression to the mean)
# ---------------------------------------------------------------------------
# Starter RA9 stabilizes very slowly. Standard sabermetric practice is to add
# ~60-70 IP of league-average performance as a Bayesian prior before trusting
# a run-average stat. A 20 IP hot streak then moves the estimate only ~25% of
# the way from league average toward the raw number - which is about right.
STARTER_BALLAST_IP = 60.0

# Minimum real IP before we use a starter's own numbers at all. Below this,
# the regressed estimate would be >80% prior anyway; we log it and fall back
# to a league-average starter rather than pretend we know something.
STARTER_MIN_IP = 10.0

# Team run rates: adding ~60 games of league-average is a common regression
# amount for team-level scoring rates (about 1/3 shrinkage at 120 games,
# 2/3 early in the season - matching how predictive early rates actually are).
TEAM_BALLAST_GAMES = 60.0

# ---------------------------------------------------------------------------
# Starter / rest-of-staff blend
# ---------------------------------------------------------------------------
# Starters throw ~55% of innings league-wide (2024-2026 era). The other 45%
# is priced off "team minus this starter" RA9, NOT whole-team RA9 - whole-team
# RA9 double-counts the starter we already priced separately (audit bug B4).
#
# NOTE: this is now only a FALLBACK. A fixed share treats Skubal and a 4.2-IP
# opener identically, which under-credits aces (who work deep) and over-credits
# short starters - and that systematically inflates model edges on underdogs
# facing elite pitchers. engine.defense_ratio now uses a PER-PITCHER share =
# (his IP/start) / 9, clamped to [MIN,MAX], and only falls back to this
# constant when games-started is unknown (TBD/no season data). See
# features.starter_innings_share.
STARTER_INNINGS_SHARE = 0.55
# Dynamic-share clamp. Floor ~ a short-leash starter (3.6 IP); ceiling ~6.5 IP
# because 7+ IP starts and complete games are too rare to price a full share on
# and the bullpen still throws the rest even on an ace's good night.
STARTER_SHARE_MIN = 0.40
STARTER_SHARE_MAX = 0.72

# ---------------------------------------------------------------------------
# Probability calibration (fitted from backtest evidence)
# ---------------------------------------------------------------------------
# The 2026 full-season backtest (1,384 games, Mar 31-Jul 19) showed the
# model is systematically UNDER-confident: its 38% dogs realized ~19-30%,
# its 66%+ favorites realized ~75% - the classic signature of shrinkage
# pulling probabilities too hard toward 50%. Fitting
#     p_true = sigmoid(a * logit(p_model))
# by weighted MLE on the calibration bins gives a = 1.39. Applied as a
# final post-hoc stretch to every win probability.
#   - 1.0 disables (raw structural output)
#   - fitted IN-SAMPLE on one season: validate on future months and refit
#     (rerun backtest.py, refit, update this constant) every ~4-6 weeks
CALIBRATION_LOGIT_SCALE = 1.39
# Modern MLB home win% is ~52-52.5%. A symmetric run multiplier m applied as
# home*m, away/m gives run ratio m^2; via Pythagorean (exp 1.83),
# m = 1.021 -> ratio 1.043 -> ~52% home win. v1's 1.04 gave ~54% (bug B5).
HFA_RUN_MULT = 1.021

# Home team win rate in extra innings (post-2020 placed-runner era, empirical
# ~52%): used only as a final tiebreak if the vectorized sim caps out.
HFA_EXTRAS = 0.52

# ---------------------------------------------------------------------------
# Park factors  (runs, 100 = neutral, ~3yr blended, REGRESSED toward 100)
# ---------------------------------------------------------------------------
# Approximate values in the spirit of Savant/FanGraphs 3-year run factors,
# already shrunk toward 100 (never trust a raw single-year park factor).
# EDIT/REFRESH ANNUALLY. Keyed by team id of the home team.
# A missing key -> 100 (neutral) with a logged warning, never a crash.
PARK_FACTORS = {
    109: 104,  # ARI Chase Field
    144: 100,  # ATL Truist Park
    110: 102,  # BAL Camden Yards
    111: 106,  # BOS Fenway Park
    112: 101,  # CHC Wrigley Field (wind-dependent; annual avg near neutral+)
    145: 102,  # CWS Rate Field
    113: 106,  # CIN Great American Ball Park
    114: 100,  # CLE Progressive Field
    115: 112,  # COL Coors Field
    116: 100,  # DET Comerica Park
    117: 100,  # HOU Daikin Park
    118: 104,  # KC  Kauffman Stadium
    108: 100,  # LAA Angel Stadium
    119: 101,  # LAD Dodger Stadium
    146: 97,   # MIA loanDepot park
    158: 100,  # MIL American Family Field
    142: 99,   # MIN Target Field
    121: 96,   # NYM Citi Field
    147: 102,  # NYY Yankee Stadium
    133: 103,  # ATH Sutter Health Park (Sacramento, hitter-friendly)
    143: 101,  # PHI Citizens Bank Park
    134: 98,   # PIT PNC Park
    135: 96,   # SD  Petco Park
    137: 99,   # SF  Oracle Park
    136: 93,   # SEA T-Mobile Park
    138: 100,  # STL Busch Stadium
    139: 97,   # TB  Tropicana/home venue
    140: 101,  # TEX Globe Life Field
    141: 100,  # TOR Rogers Centre
    120: 101,  # WSH Nationals Park
}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
N_SIMS = 20_000  # MC std error on win prob ~0.35% - below model error
RNG_SEED = 19

# Runs are NOT Poisson: they cluster within innings ("crooked numbers"), so
# game-level variance runs ~1.2-1.4x the mean even conditional on a good
# lambda. We mix the Poisson rate with a Gamma multiplier -> negative
# binomial marginal: Var = lam * (1 + DISPERSION * lam).
# DISPERSION 0.07 at lam=4.5 -> Var/mean ~= 1.32.
DISPERSION = 0.07

# Extra innings (2020+ rules): placed runner on 2B makes expected scoring
# ~1.0 runs/team/inning vs ~0.5 in regulation. Modeled as a per-inning
# Poisson of (lam/9 + MANFRED_RUNNER_BONUS).
MANFRED_RUNNER_BONUS = 0.45
MAX_EXTRA_INNINGS = 12  # then HFA_EXTRAS coin; affects <0.1% of sims

# ---------------------------------------------------------------------------
# Data hygiene
# ---------------------------------------------------------------------------
MIN_TEAM_GAMES = 15      # below this, team rates are mostly ballast anyway
HTTP_TIMEOUT = 20
HTTP_RETRIES = 3
CACHE_DIR = ".api_cache"  # on-disk cache; safe to delete anytime

# ===========================================================================
# V2: external projections as starter priors
# ===========================================================================
# FanGraphs has no free API; export a projections CSV (Steamer/ZiPS/ATC,
# "Export Data" button) and drop it here. Loader auto-detects the id column
# (MLBAMID/xMLBAMID) and a FIP or ERA column. Missing file -> league-average
# priors (V1 behavior) with a logged warning, never a crash.
PROJECTIONS_CSV = "projections.csv"
# FIP is on the ERA scale; RA9 runs ~8% above ERA (unearned runs).
RA9_FROM_FIP_MULT = 1.08
# A projection is worth far more ballast than a league-average guess:
# it already encodes multi-year data + aging. Observed current-season RA9
# gets LESS weight against a real projection than against the league mean.
STARTER_BALLAST_IP_WITH_PROJ = 110.0

# ===========================================================================
# V2: lineups + platoon (wOBA vs L/R)
# ===========================================================================
# wOBA linear weights (FanGraphs-era scale, roughly stable year to year).
WOBA_WEIGHTS = {"uBB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.25,
                "3B": 1.58, "HR": 2.03}
LEAGUE_WOBA_FALLBACK = 0.312   # used only if league can't be computed live
# Shrinkage: overall wOBA stabilizes ~ a few hundred PA; platoon SPLITS are
# far noisier (Tango: ~1000+ PA to trust a split delta), hence heavy ballast
# toward (overall wOBA + league-average platoon offset).
WOBA_OVERALL_BALLAST_PA = 220.0
WOBA_SPLIT_BALLAST_PA = 600.0
# League-average platoon offsets (wOBA points, batter with platoon adv.):
# hitters gain ~+10-12 pts with the advantage, lose ~-8 without it.
PLATOON_OFFSET_ADV = 0.011
PLATOON_OFFSET_DIS = -0.008
# Team runs scale ~ (wOBA ratio)^1.8-2.0 (BaseRuns elasticity). We use 1.9.
WOBA_RUN_ELASTICITY = 1.9
# Blend weight of the lineup-derived offense vs season team offense.
# <1 because: bench/PH innings, our wOBA->runs map is approximate, and
# 9-man averages ignore order effects. 0.5 is a deliberately humble start.
LINEUP_ADJ_WEIGHT = 0.5

# ===========================================================================
# V2: bullpen fatigue
# ===========================================================================
# League-normal relief workload is ~3.8 IP per game played (measured
# empirically from 2026 boxscores; starters average ~5.2 IP). Excess over
# the norm across the trailing 3 days forces tired arms / worse leverage
# choices: +2% bullpen RA9 per excess IP, capped at +12% (~+0.5 RA9 for a
# wrecked pen). Norm scales with games actually played, so an off-day
# correctly counts as recovery.
BULLPEN_NORM_RELIEF_IP_PER_GAME = 3.8
BULLPEN_NORM_PITCHES_PER_GAME = 63     # ~3.8 IP x ~16.5 pitches/IP
FATIGUE_RA9_PCT_PER_EXTRA_IP = 0.02
FATIGUE_RA9_PCT_CAP = 0.12

# ===========================================================================
# V2: weather (Open-Meteo, free, no key)
# ===========================================================================
# Temperature: ~+2.5% runs per +10F (carry + pitcher grip). Baseline is the
# typical in-season game-time temperature (~75F), NOT room temperature:
# league-average scoring already includes average weather, so a 70F baseline
# double-counted summer heat and ran totals ~+0.3 high in the full-season
# backtest (bias +0.31). Raised 70 -> 75 on that evidence; remeasure.
# Wind: out-to-center component ~+2% runs per 5 mph out, symmetric in.
# Applied only with roof 'open' or 'retract' (retractables assumed open in
# good weather; if in doubt the multiplier is clamped anyway).
TEMP_BASELINE_F = 75.0
TEMP_RUN_PCT_PER_F = 0.0025
WIND_RUN_PCT_PER_MPH_OUT = 0.004
WEATHER_MULT_MIN, WEATHER_MULT_MAX = 0.88, 1.15  # sanity clamp

# Venue table: (lat, lon, roof, cf_bearing_deg from home plate to CF).
# Bearings are APPROXIMATE (wind is a second-order effect; a 15-degree error
# is noise). roof: open | retract | dome. Keyed by home team id.
VENUES = {
    109: (33.445, -112.067, "retract", 0),    # ARI
    144: (33.891, -84.468, "open", 45),       # ATL
    110: (39.284, -76.622, "open", 30),       # BAL
    111: (42.346, -71.097, "open", 45),       # BOS
    112: (41.948, -87.656, "open", 35),       # CHC
    145: (41.830, -87.634, "open", 45),       # CWS
    113: (39.097, -84.507, "open", 60),       # CIN
    114: (41.496, -81.685, "open", 0),        # CLE
    115: (39.756, -104.994, "open", 0),       # COL
    116: (42.339, -83.049, "open", 40),       # DET
    117: (29.757, -95.356, "retract", 345),   # HOU
    118: (39.051, -94.480, "open", 45),       # KC
    108: (33.800, -117.883, "open", 45),      # LAA
    119: (34.074, -118.240, "open", 25),      # LAD
    146: (25.778, -80.220, "retract", 40),    # MIA
    158: (43.028, -87.971, "retract", 55),    # MIL
    142: (44.982, -93.278, "open", 90),       # MIN
    121: (40.757, -73.846, "open", 30),       # NYM
    147: (40.829, -73.926, "open", 75),       # NYY
    133: (38.580, -121.513, "open", 60),      # ATH (Sacramento)
    143: (39.906, -75.166, "open", 10),       # PHI
    134: (40.447, -80.006, "open", 120),      # PIT
    135: (32.707, -117.157, "open", 0),       # SD
    137: (37.778, -122.389, "open", 90),      # SF
    136: (47.591, -122.332, "retract", 45),   # SEA
    138: (38.623, -90.193, "open", 60),       # STL
    139: (27.768, -82.653, "dome", 0),        # TB
    140: (32.747, -97.084, "retract", 135),   # TEX
    141: (43.641, -79.389, "retract", 0),     # TOR
    120: (38.873, -77.007, "open", 30),       # WSH
}

# ===========================================================================
# V3: market evaluation + ML layer
# ===========================================================================
# Minimum model-vs-market probability gap to flag as "High EV". 3.5% is
# roughly the level where, IF the model were perfectly calibrated, edge
# would survive a typical -110/-110 vig. Below that it's noise.
EV_THRESHOLD = 0.035

# Data-completeness gate. When a game has a starter we can't actually price
# (TBD, or so few IP we fell back to a league-average/projection prior), the
# defense side of that game is a guess, so a moneyline "edge" can be almost
# entirely an artifact of the missing starter - it will move materially once
# the arm is announced. Require a bigger gap before flagging HIGH-EV on those
# games (empirically ~the ATL +108 / BAL-TBD spot: a 3.5% edge that mostly
# evaporated on lineup lock). Flags below this on a provisional game are still
# shown in the table but NOT marked HIGH-EV, and are tagged 'SP?'.
EV_THRESHOLD_PROVISIONAL_SP = 0.060

ML_MODEL_FILE = "ml_residual_model.pkl"
ML_FEATURES = ["p_home", "lam_home", "lam_away", "exp_total", "park_factor",
               "weather_mult", "fatigue_home", "fatigue_away", "hfa_extras"]
ML_MIN_TRAIN_GAMES = 400   # below this, a residual model is pure overfit
ML_PARAMS = {               # strict regularization for noisy sports data:
    "n_estimators": 150,    # residual signal on structural outputs is TINY
    "learning_rate": 0.01,  # slow, conservative steps
    "max_depth": 2,         # stumps+1: no memorizing early-season noise
    "num_leaves": 4,        # matches depth-2 trees exactly
    "min_child_samples": 50,  # every leaf must represent ~a week of games
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,       # L1: prunes useless features to zero
    "reg_lambda": 1.0,      # L2: damps what survives
}

# ===========================================================================
# Live odds (The Odds API - free tier, https://the-odds-api.com)
# ===========================================================================
# Key comes from the ODDS_API_KEY environment variable (or --odds-key).
# Free tier = current snapshot only, NO openers/closers/history. So odds.csv
# is maintained as a rolling ledger: the first snapshot we ever see for a
# game is stored as the opener proxy, the latest as the current line, and a
# fetch made near first pitch (daily_runner --close) is stored as the
# closing line. CLV in backtest.py needs those close columns - run --close
# in the evening or you will have no CLV data. That is a real operational
# requirement, not a nice-to-have.
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY_ENV = "ODDS_API_KEY"
# Key resolution order: --odds-key arg > env var > this file (first line).
# The file is the friendliest option on Windows - no env vars needed.
ODDS_API_KEY_FILE = "odds_api_key.txt"
ODDS_SPORT = "baseball_mlb"
ODDS_REGIONS = "us"
ODDS_CSV = "odds.csv"
# The Odds API team names that differ from MLB Stats API names.
ODDS_TEAM_ALIASES = {
    "Oakland Athletics": "Athletics",
    "Sacramento Athletics": "Athletics",
}

# ===========================================================================
# Umpire tendency factor
# ===========================================================================
# Home-plate umpire zones move totals: a big-zone (pitcher-friendly) ump is
# worth roughly -2..-3% runs, a tight-zone ump +2..+3%. Values below are
# ILLUSTRATIVE for well-known extreme umpires, in the spirit of public
# Umpire Scorecards data - REVIEW AND EXTEND with current-season data
# (umpscorecards.com). Anyone not listed is neutral 1.0, and assignments
# are only published shortly before first pitch, so morning runs will show
# 'ump TBD' - that is correct behavior, not a bug.
UMPIRE_RUN_FACTORS = {
    "pat hoberg": 1.00,       # famously accurate: neutral
    "laz diaz": 1.02,         # inconsistent, hitter-leaning
    "angel hernandez": 1.02,  # historical example (retired)
    "doug eddings": 1.01,
    "chad fairchild": 1.02,   # tight zone, hitter-friendly
    "ron kulpa": 1.01,
    "hunter wendelstedt": 0.99,
    "mark carlson": 0.99,
    "phil cuzzi": 0.98,       # big zone, pitcher-friendly
    "bill miller": 0.98,
    "quinn wolcott": 0.98,
}
UMPIRE_MULT_MIN, UMPIRE_MULT_MAX = 0.97, 1.03  # clamp: it's a small effect

# ===========================================================================
# Daily runner / webhook
# ===========================================================================
WEBHOOK_URL_ENV = "MLB_WEBHOOK_URL"   # Discord or Slack incoming webhook
DISCORD_CHAR_LIMIT = 1900             # Discord hard limit is 2000

# ===========================================================================
# Display: team abbreviations (keeps table columns from wrapping)
# ===========================================================================
TEAM_ABBREV = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC", 145: "CWS",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL", 142: "MIN", 121: "NYM",
    147: "NYY", 133: "ATH", 143: "PHI", 134: "PIT", 135: "SD", 137: "SF",
    136: "SEA", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}
