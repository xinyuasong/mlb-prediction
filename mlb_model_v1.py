"""
MLB Poisson Monte Carlo model v1  (Richard's model)
Data: free MLB Stats API (statsapi.mlb.com)
Method: team offense RPG x opponent pitching (starter ERA blended with team RA9),
        home-field bump, Poisson runs, 10k sims/game, extra-inning tiebreak.
"""
import requests, math, numpy as np
from datetime import date

S = requests.Session()
BASE = "https://statsapi.mlb.com/api/v1"
SEASON = 2026
N_SIMS = 10000
rng = np.random.default_rng(19)

def get(url, **params):
    r = S.get(url, params=params, timeout=20); r.raise_for_status(); return r.json()

# ---- 1. League + team rates ----
hit = get(f"{BASE}/teams/stats", sportIds=1, season=SEASON, group="hitting", stats="season")
pit = get(f"{BASE}/teams/stats", sportIds=1, season=SEASON, group="pitching", stats="season")

off, dfn = {}, {}
for sp in hit["stats"][0]["splits"]:
    t = sp["team"]["name"]; st = sp["stat"]
    off[t] = float(st["runs"]) / float(st["gamesPlayed"])
for sp in pit["stats"][0]["splits"]:
    t = sp["team"]["name"]; st = sp["stat"]
    dfn[t] = float(st["runsAllowed" if "runsAllowed" in st else "runs"]) / float(st["gamesPlayed"])

LG = sum(off.values()) / len(off)          # league runs/team/game
print(f"League avg: {LG:.2f} runs/team/game over {len(off)} teams")

# ---- 2. Monday schedule + probables ----
sched = get(f"{BASE}/schedule", sportId=1, date="2026-07-20", hydrate="probablePitcher")
games = sched["dates"][0]["games"]

def pitcher_ra9(pid):
    """starter's season RA9 (runs, not just earned); fallback to ERA*1.08; None if no data"""
    try:
        d = get(f"{BASE}/people/{pid}/stats", stats="season", group="pitching", season=SEASON)
        st = d["stats"][0]["splits"][0]["stat"]
        ip = st["inningsPitched"]; whole, _, frac = ip.partition(".")
        innings = int(whole) + (int(frac) if frac else 0)/3
        if innings < 15: return None, innings   # tiny sample -> ignore
        runs = float(st.get("runs", 0))
        return 9*runs/innings, innings
    except Exception:
        return None, 0

HOME_BUMP = 1.04   # ~54% home win baseline
STARTER_W = 0.55   # starter covers ~55% of innings

def exp_runs(off_rpg, opp_team_ra, opp_starter_ra9):
    opp_def = opp_team_ra if opp_starter_ra9 is None else STARTER_W*opp_starter_ra9 + (1-STARTER_W)*opp_team_ra
    return LG * (off_rpg/LG) * (opp_def/LG)

def simulate(lh, la):
    h = rng.poisson(lh, N_SIMS); a = rng.poisson(la, N_SIMS)
    ties = h == a
    # extra innings: ~1 run per ~2 extra frames; approximate winner by re-rolling small poissons until break
    et_h = rng.poisson(lh/9*1.1, N_SIMS); et_a = rng.poisson(la/9*1.1, N_SIMS)
    coin = rng.random(N_SIMS) < (0.52)  # residual home edge in extras
    h_extra_win = np.where(et_h != et_a, et_h > et_a, coin)
    hw = (h > a) | (ties & h_extra_win)
    total = h + a + ties*(et_h+et_a+0.7)  # extras add some runs on tied games
    return hw.mean(), total

print(f"\n{'MATCHUP':44s} {'HOME WIN':>8s} {'FAIR ML':>14s} {'xTOTAL':>7s} {'O8.5':>6s}")
rows=[]
for g in games:
    ht, at = g["teams"]["home"], g["teams"]["away"]
    hname, aname = ht["team"]["name"], at["team"]["name"]
    hp = ht.get("probablePitcher"); ap = at.get("probablePitcher")
    hp_ra, hp_ip = pitcher_ra9(hp["id"]) if hp else (None,0)
    ap_ra, ap_ip = pitcher_ra9(ap["id"]) if ap else (None,0)
    lh = exp_runs(off[hname], dfn[aname], ap_ra) * HOME_BUMP
    la = exp_runs(off[aname], dfn[hname], hp_ra) / HOME_BUMP
    p_home, totals = simulate(lh, la)
    xt = totals.mean(); o85 = (totals > 8.5).mean()
    def ml(p): 
        return f"-{round(100*p/(1-p)):d}" if p>=.5 else f"+{round(100*(1-p)/p):d}"
    hp_n = hp["fullName"] if hp else "TBD"; ap_n = ap["fullName"] if ap else "TBD"
    label = f"{aname} ({ap_n.split()[-1]}) @ {hname} ({hp_n.split()[-1]})"
    print(f"{label:44s} {p_home:7.1%} {ml(p_home):>6s}/{ml(1-p_home):<6s} {xt:6.1f} {o85:6.1%}")
    rows.append((label, p_home, lh, la, xt, o85))

print("\nNotes: v1 = team rates + confirmed starter RA9 blend. No park factors, weather,")
print("lineups, bullpen fatigue, or umpire. Probabilities are ~±4-5% fuzzy at best.")
