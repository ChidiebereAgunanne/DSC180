# %%
import argparse
import re
import sys
 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# %%
CONFIG = dict(
    half_life_days=45,     # recency weighting of regular-season marks
    k_spread=4,            # shrinkage of athlete spread toward event spread (in "effective marks")
    k_fail=6,              # shrinkage of athlete failure rate toward event rate
    conv_weight=0.5,       # weight of a converted mark relative to a direct mark
    min_conv_pairs=10,     # athlete-seasons needed to fit a conversion
    cal_prior_n=8,         # shrinkage of per-event calibration toward the pooled value
    blowup_z=4.0,          # Big West result this many (robust) spreads below prediction = failure, not noise
    p_fail_source="bigwest",  # "bigwest": past Big West failure rate per event; "athlete": season history
    multi_min_frac=0.6,    # multi-event score below this share of the event median = unfinished (failed)
    day_form_rho="auto",   # athlete form shared across events: "auto" = estimate per fold; a number fixes it (0 = off)
    n_sims=20000,
    drop_wind_aided=False, # True: drop wind-aided sprint/hurdle/horizontal-jump marks
    seed=0,
)

# Walk-forward folds: (training seasons, test season)
FOLDS = [([2021, 2022], 2023),
         ([2021, 2022, 2023], 2024),
         ([2021, 2022, 2023, 2024], 2025)]
FINAL_FOLD = ([2021, 2022, 2023, 2024, 2025], 2026)
 
DEFAULT_POINTS = [10, 8, 6, 5, 4, 3, 2, 1]



# %%
ALIASES = {"10,000": "10000", "2 Mile": "2Mile"}
RELAYS = {"4x100", "4x400", "4x440y", "4x200", "4x800", "DMR", "SMR"}   # relay events
MULTIS = {"Dec", "Hep", "Pent"}                                  # multi-events (scored in points)
FIELD = {"LJ", "TJ", "HJ", "PV", "SP", "DT", "HT", "JT", "WT"}      # field events (higher = better)
WIND_EVENTS = {"100", "200", "100H", "110H", "LJ", "TJ"}        # events where wind is recorded
 # Non-Big West event -> Big West event(s) it informs; marks are converted with a fitted line
CONVERT = {
    "60": ["100"], "300": ["200", "400"], "500": ["400"], "600": ["400", "800"],
    "600y": ["800"], "1000": ["800", "1500"], "Mile": ["1500"], "2000": ["1500"],
    "3000": ["1500", "5000", "3000S"], "2Mile": ["5000"], "2000S": ["3000S"],
    "300H": ["400H"], "WT": ["HT"],
    "5000": ["10000", "3000S"], "10000": ["5000"], "1500": ["5000"],  
}

# exact distance conversions (shift in p-space): 440y = 402.336 m
YARD = {"4x440y": ("4x400", np.log(402.336 / 400))}
DROP = {"4x200", "4x800", "DMR", "SMR"} # relays that don't predict Big West relays; ignored

# Non-mark codes -> status
STATUS_MAP = {
    "DNS": "DID_NOT_COMPETE", "SCR": "DID_NOT_COMPETE",
    "DQ": "FAILED", "FS": "FAILED", "DNF": "FAILED", "NH": "FAILED",
    "NM": "FAILED", "FOUL": "FAILED", "NP": "FAILED",
    "NT": "MISSING",
}

# Distances in meters, for the Riegel fallback between flat distance races
DIST_M = {"1500": 1500, "Mile": 1609.344, "2000": 2000, "3000": 3000,
          "2Mile": 3218.688, "5000": 5000, "10000": 10000}
RIEGEL_EXP = 1.06   # Riegel's formula: t2 = t1 * (d2/d1)^1.06
RIEGEL_SD = 0.03    # extra ~3% uncertainty when using that general formula

def higher_is_better(event):
    return event in FIELD or event in MULTIS    # True for field/multi events, False for timed events
 
def targets_for(event, gender, bw_events):
    """Which Big West event(s) a regular-season event informs, and how: list of (target, kind)."""
    if event in DROP:   # ignored relays -> nothing
        return []
    if event == "60H":   # 60m hurdles -> 100H (women) or 110H (men)
        return [("100H" if gender == "W" else "110H", "convert")]
    if event == "Hep" and gender == "M":    # men's Hep is the INDOOR heptathlon -> Dec
        return [("Dec", "convert")]
    if event == "Pent": # women's indoor pentathlon -> Hep
        return [("Hep", "convert")] if gender == "W" else []
    if event in YARD:    # yard relay -> exact distance conversion
        return [(YARD[event][0], "yard")]
    # If mark is from a big wets event use it directly for that event; otherwise, see if it informs any Big West events
    out = [(event, "direct")] if event in bw_events else []                       # direct if Big West event
    out += [(t, "convert") for t in CONVERT.get(event, []) if t in bw_events]     # plus any conversions
    return out 

# %%
## Cleaning dataset
def parse_mark(raw, unit=None):
    """Turn a mark string into a number (seconds, meters, or points), or NaN."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    s = re.sub(r"[A-Za-z\"]+$", "", str(raw).strip()).strip()   # remove trailing letters (e.g., "m", "pt") and quotes
    s = s.replace(",", "")  # remove commas in numbers (e.g., "1,000" -> "1000")
    ft_in = re.match(r"^(\d+)\s*[-']\s*(\d+(?:\.\d+)?)$", s)    # feet-inches (e.g., 6-2 or 6'2")
    if ft_in:
        return (float(ft_in.group(1)) * 12 + float(ft_in.group(2))) * 0.0254 # convert to meters
    if ":" in s:
        #Turn a time string (e.g., "1:23.45") into seconds. 1:23.45 -> 83.45
        try:
            parts = [float(x) for x in s.split(":")]
        except ValueError:
            return np.nan
        return sum(v * 60 ** (len(parts) - 1 - i) for i, v in enumerate(parts))
    try:
        return float(s) # convert under 1 minute values to float (e.g., "12.34" -> 12.34)
    except ValueError:
        return np.nan
    
def _prep(df, cfg):
    df = df.copy()
    df["event"] = df["event"].astype(str).str.strip().replace(ALIASES)
    g = df["gender"].astype(str).str.strip().str.upper().str[0]
    df["gender"] = g.replace({"F": "W"})
    df["team"] = df["team"].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    # Normalize IDs so both files match: 12345, 12345.0 and " 12345" all become "12345"
    aid = df["athlete_id"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    df["athlete_id"] = aid.fillna(df["athlete_name"].astype("string").str.strip())
 
    # relays: one "athlete" per team relay
    rel = df["event"].isin(RELAYS)
    df.loc[rel, "athlete_id"] = df.loc[rel, "team"].astype(str) + "|" + df.loc[rel, "event"]
    df.loc[rel, "athlete_name"] = df.loc[rel, "team"].astype(str) + " " + df.loc[rel, "event"]
 
    up = df["mark"].astype(str).str.strip().str.upper()
    status = up.map(STATUS_MAP)
    unit = df["mark_unit"] if "mark_unit" in df else pd.Series(None, index=df.index)
    df["value"] = [parse_mark(m, u) if pd.isna(s) else np.nan
                   for m, u, s in zip(df["mark"], unit, status)]
    ok = df["value"].notna() & (df["value"] > 0)
    df["status"] = status.where(status.notna(), np.where(ok, "VALID", "MISSING"))
 
    # Multi-event scores far below the event median are partial scores from an unfinished
    # decathlon/heptathlon/pentathlon (e.g. 821 points). Count them as failed, not as a performance.
    multi = df["event"].isin(MULTIS) & (df["status"] == "VALID")
    if multi.any():
        med = df[multi].groupby(["gender", "event"])["value"].transform("median")
        df.loc[med.index[df.loc[med.index, "value"] < cfg.get("multi_min_frac", 0.6) * med], "status"] = "FAILED"
 
    if cfg["drop_wind_aided"] and "wind" in df:
        w = pd.to_numeric(df["wind"], errors="coerce")
        df.loc[df["event"].isin(WIND_EVENTS) & (w > 2.0), "status"] = "MISSING"
 
    hib = df["event"].map(higher_is_better)
    df["p"] = np.where(df["status"] == "VALID",
                       np.where(hib, np.log(df["value"]), -np.log(df["value"])), np.nan)
    return df

def parse_meet_dates(date):
    """Parse meet dates, keeping the LAST day of multi-day ranges."""
    s = date.astype(str).str.strip()
    # "Mar 31-Apr 1, 2023" -> "Apr 1, 2023"
    s = s.str.replace(r"^[A-Za-z]{3,9}\.?\s+\d{1,2}\s*-\s*(?=[A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4}$)",
                      "", regex=True)
    # "Apr 10-11, 2025" -> "Apr 11, 2025"
    s = s.str.replace(r"^([A-Za-z]{3,9}\.?)\s+\d{1,2}\s*-\s*(\d{1,2}),\s*(\d{4})$",
                      r"\1 \2, \3", regex=True)
    return pd.to_datetime(s, errors="coerce", format="mixed")

def load_regular(path, cfg):
    df = pd.read_csv(path)
    date = df["meet_date"].fillna(df.get("meet_start_date"))    # meet_date, else start date
    df["date"] = parse_meet_dates(date)
    df["season"] = df["date"].dt.year 
    return _prep(df.dropna(subset=["date"]), cfg)    # drop undated rows, clean

# Big West full names -> regular-season codes
BW_NAME_MAP = {
    "100 Meters": "100", "200 Meters": "200", "400 Meters": "400", "800 Meters": "800",
    "1500 Meters": "1500", "5000 Meters": "5000", "10,000 Meters": "10000",
    "100 Hurdles": "100H", "110 Hurdles": "110H", "400 Hurdles": "400H",
    "3000 Steeplechase": "3000S", "4 x 100 Relay": "4x100", "4 x 400 Relay": "4x400",
    "High Jump": "HJ", "Pole Vault": "PV", "Long Jump": "LJ", "Triple Jump": "TJ",
    "Shot Put": "SP", "Discus": "DT", "Hammer": "HT", "Javelin": "JT",
    "Decathlon": "Dec", "Heptathlon": "Hep",
}


def bw_event_code(name):    # strip "Men's " / "Women's " and look up the code; unknown names unchanged
    s = re.sub(r"^(Men|Women)[’']s\s+", "", str(name).strip())
    return BW_NAME_MAP.get(s, s)

def load_bigwest(src, cfg):
    df = pd.read_csv(src)
    df["event"] = df["event"].map(bw_event_code)    # full names -> codes
    unmapped = sorted(set(df["event"]) - set(BW_NAME_MAP.values())) # anything that didn't map
    if unmapped:
        print(f"WARNING: Big West events not mapped to codes: {unmapped}")
    df["season"] = df["meet_year"].astype(int)  # season = meet year
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    return _prep(df, cfg)    # shared cleaning
 
 
def champ_dates(reg, seasons):
    """Reference date per season: the day after its last regular-season meet."""
    last = reg.groupby("season")["date"].max()  # last meet per season
    return {s: last[s] + pd.Timedelta(days=1) for s in seasons if s in last.index}


# %%
# conversional functions
def season_bests(reg):
    """Creates a table of best marks per athlete-season-event, with p-space values(perfromance scale,higher=better).NaN if athlete did not comepete"""
    v = reg[reg["status"] == "VALID"]
    return v.groupby(["season", "gender", "athlete_id", "event"])["p"].max().unstack("event")
 
 
def fit_conversions(reg_train, bw_events, cfg):
    """learns marks from one event to another using linear regression on athlete-seasons that did both events."""
    sb = season_bests(reg_train)
    conv = {}
    for g in ["W", "M"]:
        sbg = sb.xs(g, level="gender") if g in sb.index.get_level_values("gender") else None
        if sbg is None:
            continue
        for src in sbg.columns:
            for tgt, kind in targets_for(src, g, bw_events.get(g, set())):
                if kind != "convert" or tgt not in sbg.columns:
                    continue
                both = sbg[[src, tgt]].dropna()
                if len(both) < cfg["min_conv_pairs"]:
                    continue
                b, a = np.polyfit(both[src], both[tgt], 1)
                resid = both[tgt] - (a + b * both[src])
                conv[(g, src, tgt)] = dict(a=a, b=b, resid_sd=resid.std(ddof=1), n=len(both))
    return conv

def expand_marks(reg, conv, bw_events, cfg):
    """Takes every regular-season mark and files it under the Big West event(s) it tells us something about, converting it if needed and tagging how much to trust it."""
    out = []
    for (g, ev), grp in reg.groupby(["gender", "event"]):
        for tgt, kind in targets_for(ev, g, bw_events.get(g, set())):
            r = grp[["season", "date", "athlete_id", "athlete_name", "team", "status", "p"]].copy()
            r["gender"], r["event"], r["kind"], r["source"] = g, tgt, kind, ev
            if kind == "direct":
                r["w"], r["xv"] = 1.0, 0.0
            elif kind == "yard":
                r = r[r["status"] == "VALID"]
                r["p"] = r["p"] + YARD[ev][1]
                r["w"], r["xv"] = 1.0, 0.0
            else:
                c = conv.get((g, ev, tgt))
                r = r[r["status"] == "VALID"]
                if c is not None:
                    r["p"] = c["a"] + c["b"] * r["p"]
                    r["w"], r["xv"] = cfg["conv_weight"], c["resid_sd"] ** 2
                elif ev in DIST_M and tgt in DIST_M:
                    r["p"] = r["p"] - RIEGEL_EXP * np.log(DIST_M[tgt] / DIST_M[ev])
                    r["w"], r["xv"] = cfg["conv_weight"], RIEGEL_SD ** 2
                else:
                    continue
            out.append(r)
    return pd.concat(out, ignore_index=True)

# %%
KEYS = ["gender", "event", "athlete_id"]   # one profile per athlete per event
 
 
def athlete_profiles(marks, as_of, cfg):
    """Summarize each athlete-event: mean (mu), spread (sd), best, failure chance."""
    m = marks[marks["date"] < pd.Timestamp(as_of)].copy()                     
    age = (pd.Timestamp(as_of) - m["date"]).dt.days                          
    m["wt"] = m["w"] * 0.5 ** (age / cfg["half_life_days"])                   
    v = m[m["status"] == "VALID"].copy()                                     
    d = v[v["kind"] == "direct"]                                              
 
    # event-level priors (what's typical for this event)
    sd_ev = d.groupby(KEYS)["p"].std().groupby(level=[0, 1]).median()         
    sd_global = sd_ev.median() if len(sd_ev) else 0.03                        
    dm = m[m["kind"] == "direct"]                                             
    pf_ev = (dm["status"] == "FAILED").groupby([dm["gender"], dm["event"]]).mean()   
 
    # weighted mean of p per athlete
    v["wp"] = v["wt"] * v["p"]                                                
    agg = v.groupby(KEYS).agg(sw=("wt", "sum"), swp=("wp", "sum"))            
    agg["sw2"] = (v["wt"] ** 2).groupby([v[k] for k in KEYS]).sum()           
    agg["mu"] = agg["swp"] / agg["sw"]                                        
 
    # weighted spread per athlete
    v = v.join(agg["mu"], on=KEYS)                                            
    v["wdev2"] = v["wt"] * (v["p"] - v["mu"]) ** 2                            
    v["wxv"] = v["wt"] * v["xv"]                                             
    agg["var_own"] = v.groupby(KEYS)["wdev2"].sum() / agg["sw"]              
    agg["xv"] = v.groupby(KEYS)["wxv"].sum() / agg["sw"]                      
    agg["n_eff"] = agg["sw"] ** 2 / agg["sw2"]                               
    agg["n_direct"] = d.groupby(KEYS).size().reindex(agg.index).fillna(0)     
    agg["best"] = d.groupby(KEYS)["p"].max().reindex(agg.index)               
    agg["best"] = agg["best"].fillna(v.groupby(KEYS)["p"].max())              
 
    # shrink spread toward the event's typical spread
    idx = agg.index.droplevel("athlete_id")                                  
    prior_sd = pd.Series(sd_ev.reindex(idx).to_numpy(), index=agg.index).fillna(sd_global)
    k = cfg["k_spread"]
    agg["sd"] = np.sqrt((agg["n_eff"] * agg["var_own"] + k * prior_sd ** 2)   
                        / (agg["n_eff"] + k) + agg["xv"])                     
 
    # failure chance, shrunk toward the event rate
    n_all = dm.groupby(KEYS).size().reindex(agg.index).fillna(0)             
    n_fail = dm[dm["status"] == "FAILED"].groupby(KEYS).size().reindex(agg.index).fillna(0)
    prior_pf = pd.Series(pf_ev.reindex(idx).to_numpy(), index=agg.index).fillna(0.03)
    agg["p_fail"] = (n_fail + cfg["k_fail"] * prior_pf) / (n_all + cfg["k_fail"])
 
    last_team = m.sort_values("date").groupby(KEYS)["team"].last()            
    agg["team"] = last_team.reindex(agg.index)
    return agg[["team", "mu", "sd", "best", "p_fail", "n_eff", "n_direct"]].reset_index()

# %%
# Calibration
def _mad_sd(x):
    """Robust standard deviation (1.4826 x median absolute deviation); ignores extreme values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def fit_calibration(marks, bw, seasons, cdates, cfg):
    """Checks how well regular-season profiles predicted past championships,and learns
    offset(average difference, do athletes typically beat or fall short of season average)
    and scale (how much variation is there in that difference) per event."""
    recs = []
    for s in seasons:
        prof = athlete_profiles(marks[marks["season"] == s], cdates[s], cfg)
        act = bw[(bw["season"] == s) & (bw["status"] == "VALID")][KEYS + ["p"]]
        recs.append(act.rename(columns={"p": "p_act"}).merge(prof, on=KEYS).assign(season=s))
    m = pd.concat(recs, ignore_index=True)
    m["res"] = m["p_act"] - m["mu"]
 
    # 1. flag blowups per event using robust (median / MAD) statistics
    m["blowup"] = False
    for _, idx in m.groupby(["gender", "event"]).groups.items():
        g = m.loc[idx]
        z = (g["res"] - g["res"].median()) / g["sd"]
        sc = _mad_sd(z)
        if np.isfinite(sc) and sc > 0:
            m.loc[idx, "blowup"] = z < -cfg["blowup_z"] * sc
    ok = m[~m["blowup"]]
 
    # 2. offset and scale from normal results only
    off_all = ok["res"].mean()
    scale_all = _mad_sd((ok["res"] - off_all) / ok["sd"])
 
    # 3. championship failure rate: FAILED status + blowups, over all entrants in the training seasons
    b = bw[bw["season"].isin(seasons) & bw["status"].isin(["VALID", "FAILED"])]
    fails = b.assign(f=b["status"].eq("FAILED")).groupby(["gender", "event"])["f"].agg(["sum", "size"])
    fails["sum"] += m[m["blowup"]].groupby(["gender", "event"]).size().reindex(fails.index).fillna(0)
    pf_all = fails["sum"].sum() / max(fails["size"].sum(), 1)
 
    n0 = cfg["cal_prior_n"]
    cal = {}
    for (g, e), grp in ok.groupby(["gender", "event"]):
        n = len(grp)
        off = (n * grp["res"].mean() + n0 * off_all) / (n + n0)
        sc = _mad_sd((grp["res"] - off) / grp["sd"]) if n > 4 else scale_all
        sc = (n * sc + n0 * scale_all) / (n + n0)
        nf, nt = fails.loc[(g, e)] if (g, e) in fails.index else (0, 0)
        pf = (nf + n0 * pf_all) / (nt + n0)
        cal[(g, e)] = dict(offset=off, scale=float(np.clip(sc, 0.5, 3.0)), p_fail=float(pf), n=n,
                           n_blowups=int(m[(m["gender"] == g) & (m["event"] == e)]["blowup"].sum()))
    cal["_pooled"] = dict(offset=off_all, scale=float(np.clip(scale_all, 0.5, 3.0)),
                          p_fail=float(pf_all), n=len(ok), n_blowups=int(m["blowup"].sum()),
                          rho=estimate_day_form_rho(ok, cfg))
    return cal
 
def estimate_day_form_rho(ok, cfg):
    """
    Correlation of an athlete's standardized results across their individual events at the same
    Big West (good/bad days carry over). Rank (Spearman) correlation, converted to the normal-scale
    correlation the simulation uses. Fixed value if cfg["day_form_rho"] is a number.
    """
    if cfg.get("day_form_rho", "auto") != "auto":
        return float(cfg["day_form_rho"])
    d = ok[~ok["event"].isin(RELAYS)].copy()
    d["z"] = d["res"] / d["sd"]
    d["season"] = d.get("season", 0)
    pairs = []
    for _, grp in d.groupby(["season", "athlete_id"]):
        z = grp["z"].to_numpy()
        pairs += [(z[i], z[j]) for i in range(len(z)) for j in range(i + 1, len(z))]
    if len(pairs) < 30:
        return 0.0
    pairs = np.array(pairs)
    rs = spearmanr(pairs[:, 0], pairs[:, 1])[0]
    rho = 2 * np.sin(np.pi * rs / 6)            # Spearman -> Pearson for normal data
    return float(np.clip(rho, 0.0, 0.6))
 
def points_tables(bw):
    """Infer place -> points from the Big West results."""
    b = bw[bw["points"] > 0].copy()
    b["place_n"] = pd.to_numeric(b["place"], errors="coerce")
    b["kind"] = np.where(b["event"].isin(RELAYS), "relay", "ind")
    tables = {}
    for kind, grp in b.dropna(subset=["place_n"]).groupby("kind"):
        t = grp.groupby("place_n")["points"].agg(lambda x: x.mode().iloc[0]).sort_index()
        tables[kind] = [float(t.get(i, 0)) for i in range(1, int(t.index.max()) + 1)]
    tables.setdefault("ind", DEFAULT_POINTS)
    tables.setdefault("relay", tables["ind"])
    return tables

# %%
#Simulation
PRIOR_SD_INFLATE = 1.5 
def build_field(prof, entries, gender, event, prof_prev=None):
    """
    Assembles the list of entrants for one event
    No marks this season -> last season's profile (incl. last Big West), with wider spread.
    Still nothing -> weak prior (lower part of the field).
    """
    e = entries[(entries["gender"] == gender) & (entries["event"] == event)]
    e = e[["athlete_id", "athlete_name", "team"]].drop_duplicates("athlete_id")
    p = prof[(prof["gender"] == gender) & (prof["event"] == event)]
    f = e.merge(p.drop(columns="team"), on="athlete_id", how="left")
    f["source"] = np.where(f["mu"].isna(), "none", "current")
    if prof_prev is not None and (f["source"] == "none").any():
        pp = prof_prev[(prof_prev["gender"] == gender) & (prof_prev["event"] == event)]
        pp = pp.set_index("athlete_id")
        use = (f["source"] == "none") & f["athlete_id"].isin(pp.index)
        for col in ["mu", "sd", "best", "p_fail", "n_eff", "n_direct"]:
            f.loc[use, col] = f.loc[use, "athlete_id"].map(pp[col])
        f.loc[use, "sd"] = f.loc[use, "sd"] * PRIOR_SD_INFLATE
        f.loc[use, "source"] = "prior_season"
    f["imputed"] = f["mu"].isna()
    if f["imputed"].all():
        return None
    if f["imputed"].any():   # no regular-season marks: assume lower-field athlete, wide spread
        f.loc[f["imputed"], "mu"] = f["mu"].quantile(0.2)
        f.loc[f["imputed"], "sd"] = f["sd"].median() * 1.5
        f.loc[f["imputed"], "p_fail"] = f["p_fail"].median()
        f.loc[f["imputed"], "best"] = -np.inf
    return f.reset_index(drop=True)
 
def simulate_field(f, cal, pts, n_sims, rng, p_fail_source="athlete", day_form=None, rho=0.0):
    """
    Return (n_sims, n_athletes) array of points.
    day_form(athlete_id) -> length-n_sims array of shared standard-normal draws (same array for every
    event the athlete is in). Mixed with event-specific noise so each event's spread is unchanged:
        noise = sqrt(rho) * day_form + sqrt(1 - rho) * event_noise
    """
    n = len(f)
    if p_fail_source == "bigwest" and "p_fail" in cal:
        f = f.assign(p_fail=cal["p_fail"])
    mu = f["mu"].to_numpy() + cal["offset"]
    sd = f["sd"].to_numpy() * cal["scale"]
    eps = rng.standard_normal((n_sims, n))
    if day_form is not None and rho > 0:
        shared = np.column_stack([day_form(a) for a in f["athlete_id"]])
        eps = np.sqrt(rho) * shared + np.sqrt(1 - rho) * eps
    perf = mu + sd * eps
    fail = rng.random((n_sims, n)) < f["p_fail"].to_numpy()
    perf[fail] = -np.inf
    order = np.argsort(-perf, axis=1)
    places = np.empty_like(order)
    places[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)
    table = np.zeros(n + 1)
    k = min(n, len(pts))
    table[1:k + 1] = pts[:k]
    out = table[places]
    out[fail] = 0
    return out
 
 
def baseline_points(f, pts):
    """Baseline: rank by season best, assign points (no uncertainty)."""
    rank = f["best"].rank(ascending=False, method="first").astype(int)
    table = {i + 1: p for i, p in enumerate(pts)}
    return rank.map(table).fillna(0).to_numpy()
 
 
def crps(samples, y):
    """CRPS: how far a predicted distribution is from the actual value (lower is better)."""
    x = np.sort(samples)
    n = len(x)
    return np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2

def previous_profiles(marks, bw, season, cdates, cfg):
    """Profiles from a past season, including that season's Big West result as a mark."""
    if season not in cdates:
        return None
    prev = marks[marks["season"] == season]
    b = bw[(bw["season"] == season) & bw["status"].isin(["VALID", "FAILED"])]
    b = b[["season", "athlete_id", "athlete_name", "team", "status", "p", "gender", "event"]].copy()
    b["date"] = cdates[season]
    b["kind"], b["source"], b["w"], b["xv"] = "direct", "bigwest", 1.0, 0.0
    both = pd.concat([prev, b], ignore_index=True)
    return athlete_profiles(both, cdates[season] + pd.Timedelta(days=1), cfg)

# %%
def run_fold(reg, bw, cal_seasons, test, cdates, bw_events, cfg, verbose=True):
    rng = np.random.default_rng(cfg["seed"] + test)
    conv = fit_conversions(reg[reg["season"].isin(cal_seasons)], bw_events, cfg)
    marks = expand_marks(reg[reg["season"].isin(cal_seasons + [test])], conv, bw_events, cfg)
    cal = fit_calibration(marks, bw, cal_seasons, cdates, cfg)
    pts_tab = points_tables(bw[bw["season"].isin(cal_seasons)])
    prof = athlete_profiles(marks[marks["season"] == test], cdates[test], cfg)
    prof_prev = previous_profiles(marks, bw, test - 1, cdates, cfg)
 
    entries = bw[bw["season"] == test]
 
    # shared day form for athletes entered in 2+ individual events (same draws across their events)
    rho = cal["_pooled"].get("rho", 0.0)
    ind = entries[~entries["event"].isin(RELAYS)]
    multi_ids = set(ind.groupby("athlete_id")["event"].nunique().loc[lambda x: x >= 2].index)
    form_cache = {}
 
    def day_form(aid):
        if aid not in multi_ids:
            return rng.standard_normal(cfg["n_sims"])
        if aid not in form_cache:
            form_cache[aid] = rng.standard_normal(cfg["n_sims"]).astype(np.float32)
        return form_cache[aid]
 
    ath_rows, te_rows = [], []
    totals = {}   # gender -> team -> samples
    skipped = []
    for (g, e), _ in entries.groupby(["gender", "event"]):
        f = build_field(prof, entries, g, e, prof_prev)
        if f is None:
            skipped.append(f"{g} {e}")
            continue
        c = cal.get((g, e), cal["_pooled"])
        pts = pts_tab["relay" if e in RELAYS else "ind"]
        sims = simulate_field(f, c, pts, cfg["n_sims"], rng, cfg.get("p_fail_source", "athlete"),
                              day_form=day_form, rho=rho)
        base = baseline_points(f, pts)
        n_score = len(pts)
 
        actual = entries[(entries["gender"] == g) & (entries["event"] == e)] \
            .groupby("athlete_id")["points"].sum()
        a = f[["athlete_id", "athlete_name", "team", "source", "imputed"]].copy()
        a["gender"], a["event"] = g, e
        a["exp_pts"] = sims.mean(0)
        a["p_score"] = (sims > 0).mean(0)
        a["base_pts"] = base
        a["actual_pts"] = a["athlete_id"].map(actual).fillna(0).to_numpy()
        ath_rows.append(a)
 
        for t in f["team"].unique():
            mask = (f["team"] == t).to_numpy()
            ts = sims[:, mask].sum(1)
            totals.setdefault(g, {}).setdefault(t, np.zeros(cfg["n_sims"]))
            totals[g][t] += ts
            act = a.loc[mask, "actual_pts"].sum()
            te_rows.append(dict(gender=g, event=e, team=t, exp_pts=ts.mean(),
                                lo80=np.percentile(ts, 10), hi80=np.percentile(ts, 90),
                                base_pts=base[mask].sum(), actual=act, crps=crps(ts, act)))
 
    ath = pd.concat(ath_rows, ignore_index=True)
    te = pd.DataFrame(te_rows)
 
    tot_rows = []
    for g, teams in totals.items():
        names = list(teams)
        mat = np.column_stack([teams[t] for t in names])
        win = np.bincount(mat.argmax(1), minlength=len(names)) / cfg["n_sims"]
        for j, t in enumerate(names):
            act = te[(te["gender"] == g) & (te["team"] == t)]["actual"].sum()
            tot_rows.append(dict(gender=g, team=t, exp_pts=mat[:, j].mean(),
                                 lo80=np.percentile(mat[:, j], 10), hi80=np.percentile(mat[:, j], 90),
                                 p_win=win[j],
                                 base_pts=te[(te["gender"] == g) & (te["team"] == t)]["base_pts"].sum(),
                                 actual=act, crps=crps(mat[:, j], act)))
    tot = pd.DataFrame(tot_rows)
 
    # ---- metrics
    scored = (ath["actual_pts"] > 0).astype(float)
    summ = dict(
        test=test,
        brier_model=np.mean((ath["p_score"] - scored) ** 2),
        brier_base=np.mean(((ath["base_pts"] > 0) - scored) ** 2),
        ath_mae_model=np.mean(np.abs(ath["exp_pts"] - ath["actual_pts"])),
        ath_mae_base=np.mean(np.abs(ath["base_pts"] - ath["actual_pts"])),
        te_crps_model=te["crps"].mean(),
        te_crps_base=np.mean(np.abs(te["base_pts"] - te["actual"])),
        te_cover80=np.mean((te["actual"] >= te["lo80"]) & (te["actual"] <= te["hi80"])),
        n_imputed=int(ath["imputed"].sum()),
        n_prior_season=int((ath["source"] == "prior_season").sum()),
        day_form_rho=rho,
    )
    for g, grp in tot.groupby("gender"):
        summ[f"{g}_total_crps_model"] = grp["crps"].mean()
        summ[f"{g}_total_mae_base"] = np.mean(np.abs(grp["base_pts"] - grp["actual"]))
        summ[f"{g}_spearman_model"] = spearmanr(grp["exp_pts"], grp["actual"])[0]
        summ[f"{g}_spearman_base"] = spearmanr(grp["base_pts"], grp["actual"])[0]
 
    if verbose:
        print(f"\n===== Fold: calibrate {cal_seasons[0]}-{cal_seasons[-1]} -> test {test} =====")
        print(f"conversions fitted: {len(conv)} | events skipped (no marks): {skipped or 'none'}")
        print(f"day-form correlation used: {rho:.2f} | blowups in training: {cal['_pooled']['n_blowups']}")
        print(f"entrants using last season: {summ['n_prior_season']} | "
              f"entrants with no marks (imputed): {summ['n_imputed']}")
        print(f"Brier (scores?)        model {summ['brier_model']:.3f} | base {summ['brier_base']:.3f}")
        print(f"Team-event CRPS        model {summ['te_crps_model']:.2f} | base {summ['te_crps_base']:.2f}"
              f"   80% coverage {summ['te_cover80']:.0%}")
        for g, grp in tot.groupby("gender"):
            print(f"{g} team totals          CRPS model {summ[f'{g}_total_crps_model']:.1f} | "
                  f"MAE base {summ[f'{g}_total_mae_base']:.1f}   Spearman model "
                  f"{summ[f'{g}_spearman_model']:.2f} | base {summ[f'{g}_spearman_base']:.2f}")
 
    ctx = dict(prof=prof, prof_prev=prof_prev, cal=cal, pts=pts_tab, entries=entries, cfg=cfg, season=test)
    return dict(summary=summ, athletes=ath, team_event=te, totals=tot, ctx=ctx)

# %%
def exceedance(ctx, gender, event, team, extra=None, k_max=None, seed=1):
    """P(team scores >= k points) in one event. `extra`: rows to add (e.g. a recruit)."""
    f = build_field(ctx["prof"], ctx["entries"], gender, event, ctx.get("prof_prev"))
    if extra is not None:
        f = pd.concat([f, extra], ignore_index=True)
    pts = ctx["pts"]["relay" if event in RELAYS else "ind"]
    c = ctx["cal"].get((gender, event), ctx["cal"]["_pooled"])
    sims = simulate_field(f, c, pts, ctx["cfg"]["n_sims"], np.random.default_rng(seed),
                          ctx["cfg"].get("p_fail_source", "athlete"))
    ts = sims[:, (f["team"] == team).to_numpy()].sum(1)
    k_max = k_max or int(sum(pts[:3]))
    return pd.Series({k: (ts >= k).mean() for k in range(0, k_max + 1)}, name="P(>=k)")
 
 
def recruit_row(ctx, gender, event, team, mark, name="RECRUIT"):
    """Hypothetical athlete whose typical championship-ready mark is `mark` (s, m, or points)."""
    p = np.log(mark) if higher_is_better(event) else -np.log(mark)
    ev = ctx["prof"][(ctx["prof"]["gender"] == gender) & (ctx["prof"]["event"] == event)]
    return pd.DataFrame([dict(athlete_id=name, athlete_name=name, team=team, mu=p,
                              sd=ev["sd"].median(), best=p, p_fail=ev["p_fail"].median(),
                              n_eff=0, n_direct=0, imputed=False)])
 
 
def whatif(ctx, gender, event, team, mark):
    """Compare a team's exceedance curve with and without a recruit."""
    base = exceedance(ctx, gender, event, team)
    new = exceedance(ctx, gender, event, team, extra=recruit_row(ctx, gender, event, team, mark))
    return pd.DataFrame({"current": base, "with_recruit": new})
 
 
def team_report(result, team="UC SAN DIEGO"):
    """Expected points by event for one team, with the 80% range and P(score >= 1)."""
    te = result["team_event"]
    r = te[te["team"] == team][["gender", "event", "exp_pts", "lo80", "hi80", "actual"]]
    return r.sort_values(["gender", "exp_pts"], ascending=[True, False]).round(2)
 
 
# ============================================================== main
def run_model(regular, bigwest, final=False, team="UC SAN DIEGO", cfg=None, save_csv=True):
    """
    regular, bigwest: CSV paths or DataFrames.
    Returns {test_year: result}; each result has summary, athletes, team_event, totals, ctx.
    """
    cfg = {**CONFIG, **(cfg or {})}
    reg = load_regular(regular, cfg)
    bw = load_bigwest(bigwest, cfg)
    seasons = sorted(bw["season"].unique())
    cdates = champ_dates(reg, seasons)
    bw_events = {g: set(grp["event"]) for g, grp in bw.groupby("gender")}
 
    folds = FOLDS + ([FINAL_FOLD] if final else [])
    results, summaries = {}, []
    for cal_seasons, test in folds:
        if test not in seasons:
            continue
        r = run_fold(reg, bw, cal_seasons, test, cdates, bw_events, cfg)
        results[test] = r
        summaries.append(r["summary"])
        if save_csv:
            r["athletes"].to_csv(f"pred_athletes_{test}.csv", index=False)
            r["team_event"].to_csv(f"pred_team_event_{test}.csv", index=False)
            r["totals"].to_csv(f"pred_team_totals_{test}.csv", index=False)
 
    s = pd.DataFrame(summaries).set_index("test")
    val = s.loc[[t for _, t in FOLDS if t in s.index]]
    print("\n===== Validation folds (mean) =====")
    print(val.mean().round(3).to_string())
    if save_csv:
        s.round(3).to_csv("fold_summary.csv")
 
    last = results[max(results)]
    print(f"\n===== {team} expected points by event, {max(results)} =====")
    print(team_report(last, team).to_string(index=False))
    results["summary"] = s
    return results
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("regular")
    ap.add_argument("bigwest")
    ap.add_argument("--final", action="store_true", help="also run the 2026 held-out test")
    ap.add_argument("--team", default="UCSD")
    args = ap.parse_args()
    return run_model(args.regular, args.bigwest, final=args.final, team=args.team)
 
 
# Command-line entry point only; skipped inside Jupyter so the file can be imported or pasted.
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()
 

# %%
regular_season = "../data/processed/regular_season.csv"
bigwest = "../data/processed/bigwest_2021_2026_results.csv"

# %%
reg = load_regular(regular_season, CONFIG)
bw = load_bigwest(bigwest, CONFIG)
cdates = champ_dates(reg, sorted(bw["season"].unique()))
bw_events = {g: set(grp["event"]) for g, grp in bw.groupby("gender")}


# %%
fold1 = run_fold(reg, bw, [2021, 2022], 2023, cdates, bw_events,CONFIG)

# %%
log = []

def evaluate(label, cfg_overrides=None):
    cfg = {**CONFIG, **(cfg_overrides or {})}
    rows = [run_fold(reg, bw, cs, t, cdates, bw_events, cfg, verbose=False)["summary"]
            for cs, t in FOLDS]
    s = pd.DataFrame(rows).mean(numeric_only=True)
    keep = ["brier_model", "te_crps_model", "M_total_crps_model", "W_total_crps_model",
            "M_spearman_model", "W_spearman_model"]
    log.append(s[keep].rename(label))
    display(pd.DataFrame(log).round(3))

evaluate("current")

# %%
print(reg[reg.status == "MISSING"].mark.value_counts().head(30))

# %%



