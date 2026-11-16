"""
Big West Track & Field championship model
=========================================
Predicts each athlete's (and relay's) Big West championship performance from regular-season marks,
Monte Carlo simulates every event, and scores teams.

Every event is modelled on one "performance" scale p where higher is always better:
    field events / multi-events : p =  log(mark)
    timed events / relays       : p = -log(time in seconds)
Differences on this scale read as percentages (0.02 = 2%), so one method fits every event and
conversions between events (60 m -> 100 m, Mile -> 1500 m) are straight lines.

Pipeline per fold (training seasons -> one test season)
-------------------------------------------------------
 1. fit_conversions    related-event conversions (60 -> 100, Mile -> 1500, ...) from training seasons
 2. expand_marks       every regular-season mark mapped onto Big West events (direct or converted)
 3. fit_calibration    from training championships: per-event offset / spread scale / failure rate, plus
                       the athlete-specific offset (ridge regression) and failure chance (ridge logistic)
 4. athlete_profiles   test-season profiles (recency-weighted mean, spread, best, failure history)
                       from marks before that year's Big West
 5. simulate_field     Monte Carlo each event; prelims + final in the 100-1500 m and hurdles
 6. evaluate           against a TFRRS-style performance-list meet calculator (the baseline):
                       Brier, CRPS, MAE of the median prediction, 80% coverage, Spearman

Final model (chosen on the 2023-2025 validation folds; see notebooks/model_history.ipynb):
    offset_model="regression", pfail_model="logistic", two_rounds=True, round_rho=0.43,
    2019 in the training seasons. Tested and not kept: virtual multi scores, year-to-year
    improvement for last-season profiles, team form. field_mode="everyone" is a planning mode.

Usage
-----
    python bigwest_model.py regular_season.csv bigwest_results.csv            # validation folds
    python bigwest_model.py regular_season.csv bigwest_results.csv --final    # + 2026 test

    import bigwest_model as bm
    r = bm.run_fold(reg, bw, train_seasons, test_season, cdates, bw_events, bm.CONFIG)
"""

import argparse
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================== configuration
CONFIG = dict(
    # ---- athlete profiles
    half_life_days=45,     # recency weighting; "auto" = estimated per fold (~20 days), tested and not kept [8a]
    k_spread=4,            # shrinkage of athlete spread toward event spread; "auto" = estimated per fold (~15),
                           # essentially identical results [8b]
    reg_blowup_z=5,        # regular-season mark this many event spreads below the athlete's median = failure (0 = off)
    multi_min_frac=0.6,    # multi-event score below this share of the event median = unfinished (failed)
    drop_wind_aided=False, # True: drop wind-aided sprint/hurdle/horizontal-jump marks (EDA: not needed)

    # ---- conversions between events
    conv_weight=0.5,       # weight of a converted mark vs a direct one; "auto" = inverse-variance weight per
                           # conversion (~0.25), tested and not kept [8c]
    min_conv_pairs=10,     # athlete-seasons needed to fit a conversion
    conv_trim_z=4,         # refit conversion lines without points this many robust sds off the line (0 = off)

    # ---- calibration to the championship (training seasons only)
    cal_prior_n=8,         # shrinkage of per-event calibration toward the pooled value
    blowup_z=4.0,          # Big West result this many (robust) spreads below prediction = failure, not noise
    dns_as_fail=True,      # Big West DNS / scratches count toward the event's failure rate
    offset_model="regression",  # "event": one offset per event; "regression": athlete-specific offset from
                           # SB level, consistency gap, risk, starts, event group and gender      [kept, V2]
    pfail_model="logistic",     # "event": p_fail from p_fail_source; "logistic": athlete-specific p_fail  [kept, V3]
    p_fail_source="bigwest",    # used when pfail_model="event" or no features: "bigwest" / "athlete" / "blend"
    ridge=1.0,             # small ridge penalty for the two regressions (stabilises small samples)
    mu_shrink=0,           # pull mu toward the field average: "auto" = learned, or a fraction (0 = off)

    # ---- field and simulation
    field_mode="entrants", # "entrants": that year's declared Big West entrants;
                           # "everyone": every athlete with an outdoor mark in the event (planning mode, goals 2/3)
    impute_q="auto",       # where entrants with no marks are placed in the field: "auto" = learned per fold, or a quantile
    impute_sd_mult=1.5,    # spread of imputed athletes = field's median spread x this
    day_form_rho="auto",   # athlete form shared across events: "auto" = estimate per fold; a number fixes it (0 = off)
    two_rounds=True,       # prelims then finals in TWO_ROUND_EVENTS (top 8 advance; only finalists score)  [kept, V5]
    round_rho=0.43,        # share of an athlete's noise shared by prelim and final; measured from 660 finalists,
                           # 2019-2025 (model_history.ipynb, step 2)
    n_sims=20000,
    seed=0,

    # ---- tested and NOT kept (left in, switched off, so the history can be reproduced)
    virtual_multi=False,   # athletes with no multi score get one built from their component events     [V6a: no]
    virtual_multi_min=4,   # component events needed for a virtual multi score
    prior_improve="off",   # last-season profiles + typical year-to-year change: "off" / "group" / "level"  [V6b: no]
    team_form_sd=0,        # shared team good/bad day on the p scale: 0 = off, "auto", or a number            [V6c: no]
)

FOLDS = [([2019, 2021, 2022], 2023),
         ([2019, 2021, 2022, 2023], 2024),
         ([2019, 2021, 2022, 2023, 2024], 2025)]
FINAL_FOLD = ([2019, 2021, 2022, 2023, 2024, 2025], 2026)

DEFAULT_POINTS = [10, 8, 6, 5, 4, 3, 2, 1]

# ============================================================== event definitions
ALIASES = {"10,000": "10000", "2 Mile": "2Mile"}
RELAYS = {"4x100", "4x400", "4x440y", "4x200", "4x800", "DMR", "SMR"}
MULTIS = {"Dec", "Hep", "Pent"}
FIELD = {"LJ", "TJ", "HJ", "PV", "SP", "DT", "HT", "JT", "WT"}
WIND_EVENTS = {"100", "200", "100H", "110H", "LJ", "TJ"}

# related event -> Big West target(s), mark converted with a fitted model
CONVERT = {
    "60": ["100"], "300": ["200", "400"], "500": ["400"], "600": ["400", "800"],
    "600y": ["800"], "1000": ["800", "1500"], "Mile": ["1500"], "2000": ["1500"],
    "3000": ["1500", "5000", "3000S"], "2Mile": ["5000"], "2000S": ["3000S"],
    "300H": ["400H"], "WT": ["HT"],
}
# Big West event -> other Big West events it also informs (used alongside direct marks)
CROSS = {"5000": ["10000", "3000S"], "10000": ["5000"], "1500": ["5000"]}
# Big West events run as prelims + final when the field is bigger than the number of scoring places
TWO_ROUND_EVENTS = {"100", "200", "400", "800", "1500", "100H", "110H", "400H"}

# exact distance conversions (shift in p-space): 440y = 402.336 m
YARD = {"4x440y": ("4x400", np.log(402.336 / 400))}
DROP = {"4x200", "4x800", "DMR", "SMR"}

# Riegel fallback for flat distances when too few pairs to fit a conversion:
# t2 = t1 * (d2/d1)^1.06  ->  p2 = p1 - 1.06 * log(d2/d1)   (p = -log time)
DIST_M = {"1500": 1500, "Mile": 1609.344, "2000": 2000, "3000": 3000,
          "2Mile": 3218.688, "5000": 5000, "10000": 10000}
RIEGEL_EXP = 1.06
RIEGEL_SD = 0.03   # extra uncertainty (~3%) for using a general formula

# Same school under different names (after uppercasing)
TEAM_ALIASES = {"CAL ST. NORTHRIDGE": "CSUN", "CAL STATE NORTHRIDGE": "CSUN",
                "CAL STATE FULLERTON": "CAL ST. FULLERTON", "LONG BEACH STATE": "LONG BEACH ST.",
                "HAWAI'I": "HAWAII", "HAWAIʻI": "HAWAII"}

STATUS_MAP = {
    "DNS": "DID_NOT_COMPETE", "SCR": "DID_NOT_COMPETE",
    "DQ": "FAILED", "FS": "FAILED", "DNF": "FAILED", "NH": "FAILED",
    "NM": "FAILED", "FOUL": "FAILED", "NP": "FAILED",
    "ND": "FAILED",                                   # no distance
    "NT": "MISSING",
}
# Scraped codes can be doubled ("FOULNM", "NHNH", "DNSNM"): use the first code in the string
_CODE_RE = r"^(" + "|".join(sorted(STATUS_MAP, key=len, reverse=True)) + r")"


def higher_is_better(event):
    """
    Whether a bigger mark is better in this event (field and multi events) or a smaller one (timed events).

    Args:
        event: event code, e.g. "100", "LJ", "Dec".
    Returns:
        bool: True for field and multi events, False for track events and relays.
    """
    return event in FIELD or event in MULTIS


def targets_for(event, gender, bw_events):
    """
    The Big West event(s) a regular-season event gives information about, and how.

    Args:
        event: regular-season event code, e.g. "60", "Mile", "400".
        gender: "M" or "W".
        bw_events: set of Big West event codes held for that gender.
    Returns:
        list of (target_event, kind) pairs. kind is "direct" (same event), "convert" (needs a fitted
        conversion line, e.g. 60 -> 100) or "yard" (exact yards -> meters shift). Empty if the event
        informs no Big West event.
    """
    if event in DROP:
        return []
    if event == "60H":
        return [("100H" if gender == "W" else "110H", "convert")]
    if event == "Hep" and gender == "M":          # men's indoor heptathlon
        return [("Dec", "convert")]
    if event == "Pent":                           # women's indoor pentathlon
        return [("Hep", "convert")] if gender == "W" else []
    if event in YARD:
        return [(YARD[event][0], "yard")]
    if event in CONVERT:
        return [(t, "convert") for t in CONVERT[event] if t in bw_events]
    if event in bw_events:
        return [(event, "direct")] + [(t, "convert") for t in CROSS.get(event, []) if t in bw_events]
    return []


# ============================================================== loading & cleaning
def parse_mark(raw, unit=None):
    """
    Turn one raw mark string into a number in natural units.

    Args:
        raw: mark as scraped, e.g. "10.52", "1:52.31", "6.45m", "21-04.5", "5,432", "DNF".
        unit: optional unit column value (not needed; kept for compatibility).
    Returns:
        float: seconds for track, meters for field, points for multis; NaN if it can't be parsed
        (status codes such as "DNF" are handled by _prep, not here).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    metric = re.match(r"^\s*(\d+(?:\.\d+)?)\s*m(?![a-z])", str(raw), flags=re.IGNORECASE)
    if metric:                                                   # '14.21m46\' 7.5"' -> 14.21
        return float(metric.group(1))
    s = re.sub(r"[A-Za-z\"]+$", "", str(raw).strip()).strip()   # drop trailing 'm', 'w', 'h', ...
    s = s.replace(",", "")                                       # multi-event scores like "5,432"
    ft_in = re.match(r"^(\d+)\s*[-']\s*(\d+(?:\.\d+)?)$", s)
    if ft_in:
        return (float(ft_in.group(1)) * 12 + float(ft_in.group(2))) * 0.0254
    if ":" in s:
        try:
            parts = [float(x) for x in s.split(":")]
        except ValueError:
            return np.nan
        return sum(v * 60 ** (len(parts) - 1 - i) for i, v in enumerate(parts))
    try:
        return float(s)
    except ValueError:
        return np.nan


def _prep(df, cfg):
    """
    Shared cleaning for the regular-season and Big West files.

    Standardises event codes, gender ("M"/"W"), team names and athlete IDs; turns each relay into one
    "athlete" per team ("TEAM|4x100"); maps status codes (DNF, DQ, FOUL, NH -> FAILED; DNS, SCR ->
    DID_NOT_COMPETE); parses marks; marks unfinished multis as FAILED; adds the performance scale p.

    Args:
        df: raw DataFrame with at least event, gender, team, athlete_id, athlete_name, mark.
        cfg: settings (uses multi_min_frac and drop_wind_aided).
    Returns:
        DataFrame: the input plus value (parsed mark), status (VALID / FAILED / DID_NOT_COMPETE /
        MISSING) and p (log mark for field/multi, -log time for track; NaN unless VALID).
    """
    df = df.copy()
    df["event"] = df["event"].astype(str).str.strip().replace(ALIASES)
    g = df["gender"].astype(str).str.strip().str.upper().str[0]
    df["gender"] = g.replace({"F": "W"})
    df["team"] = (df["team"].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
                  .replace(TEAM_ALIASES))
    # Normalize IDs so both files match: 12345, 12345.0 and " 12345" all become "12345"
    aid = df["athlete_id"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    df["athlete_id"] = aid.fillna(df["athlete_name"].astype("string").str.strip())

    # relays: one "athlete" per team relay
    rel = df["event"].isin(RELAYS)
    df.loc[rel, "athlete_id"] = df.loc[rel, "team"].astype(str) + "|" + df.loc[rel, "event"]
    df.loc[rel, "athlete_name"] = df.loc[rel, "team"].astype(str) + " " + df.loc[rel, "event"]

    up = df["mark"].astype(str).str.strip().str.upper()
    status = up.map(STATUS_MAP)
    first_code = up.str.extract(_CODE_RE, expand=False)
    status = status.fillna(first_code.map(STATUS_MAP))
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


def _read(src):
    """
    Accept either a CSV path or an already-loaded DataFrame.

    Args:
        src: path to a CSV file, or a DataFrame.
    Returns:
        DataFrame: the file's contents, or a copy of the DataFrame passed in.
    """
    return pd.read_csv(src) if isinstance(src, str) else src.copy()


def parse_meet_dates(date):
    """
    Parse meet date strings, keeping the LAST day of multi-day meets:
      "Mar 31-Apr 1, 2023" -> Apr 1, 2023   (cross-month)
      "Apr 10-11, 2025"    -> Apr 11, 2025  (same month; pandas otherwise reads 11 as the year)

    Args:
        date: Series of date strings.
    Returns:
        Series of Timestamps (NaT where a date can't be parsed).
    """
    s = date.astype(str).str.strip()
    s = s.str.replace(r"^[A-Za-z]{3,9}\.?\s+\d{1,2}\s*-\s*(?=[A-Za-z]{3,9}\.?\s+\d{1,2},\s*\d{4}$)",
                      "", regex=True)
    s = s.str.replace(r"^([A-Za-z]{3,9}\.?)\s+\d{1,2}\s*-\s*(\d{1,2}),\s*(\d{4})$",
                      r"\1 \2, \3", regex=True)
    return pd.to_datetime(s, errors="coerce", format="mixed")


def load_regular(src, cfg):
    """
    Load and clean the regular-season marks file.

    Args:
        src: path to regular_season.csv, or the DataFrame.
        cfg: settings passed on to _prep.
    Returns:
        DataFrame: one row per mark with date, season (August-December meets count toward the next
        spring), status, value and p. Rows with no parseable date are dropped.
    """
    df = _read(src)
    # prefer the already-parsed start date; fall back to the raw meet_date string
    date = (df["meet_start_date"].astype("string").fillna(df["meet_date"].astype("string"))
            if "meet_start_date" in df else df["meet_date"])
    df["date"] = parse_meet_dates(date)
    # Aug-Dec meets count toward the next spring's season (indoor starts in Dec)
    df["season"] = df["date"].dt.year + (df["date"].dt.month >= 8).astype(int)
    return _prep(df.dropna(subset=["date"]), cfg)


# Big West full event names (gender prefix removed) -> regular-season codes
BW_NAME_MAP = {
    "100 Meters": "100", "200 Meters": "200", "400 Meters": "400", "800 Meters": "800",
    "1500 Meters": "1500", "5000 Meters": "5000", "10,000 Meters": "10000",
    "100 Hurdles": "100H", "110 Hurdles": "110H", "400 Hurdles": "400H",
    "3000 Steeplechase": "3000S", "4 x 100 Relay": "4x100", "4 x 400 Relay": "4x400",
    "High Jump": "HJ", "Pole Vault": "PV", "Long Jump": "LJ", "Triple Jump": "TJ",
    "Shot Put": "SP", "Discus": "DT", "Hammer": "HT", "Javelin": "JT",
    "Decathlon": "Dec", "Heptathlon": "Hep",
}


_BW_LOWER = {k.lower(): v for k, v in BW_NAME_MAP.items()}


def bw_event_code(name):
    """
    Convert a Big West event name to the short code used everywhere else.

    Args:
        name: event name from the results file, e.g. "Men's 100 Meters", "Women's Pole Vault".
    Returns:
        str: event code, e.g. "100", "PV". Unknown names are returned unchanged (load_bigwest warns).
    """
    s = re.sub(r"^(men|women)[’']s\s+", "", str(name).strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return _BW_LOWER.get(s.lower(), s)


def load_bigwest(src, cfg):
    """
    Load and clean the Big West championship results file.

    Args:
        src: path to bigwest_2019_2026_results.csv (or the rounds file), or the DataFrame.
        cfg: settings passed on to _prep.
    Returns:
        DataFrame: one row per championship result with event code, season (meet_year), points
        (0 if none), place, status and p.
    """
    df = _read(src)
    df["event"] = df["event"].map(bw_event_code)
    unmapped = sorted(set(df["event"]) - set(BW_NAME_MAP.values()))
    if unmapped:
        print(f"WARNING: Big West events not mapped to codes: {unmapped}")
    df["season"] = df["meet_year"].astype(int)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    return _prep(df, cfg)


def champ_dates(reg, seasons):
    """
    The date each season's athlete profiles are measured at: the day after that season's last
    regular-season meet (the regular-season file only holds meets before Big West).

    Args:
        reg: cleaned regular-season DataFrame (load_regular).
        seasons: list of seasons to get dates for.
    Returns:
        dict {season: Timestamp}. Seasons with no regular-season marks are left out.
    """
    last = reg.groupby("season")["date"].max()
    return {s: last[s] + pd.Timedelta(days=1) for s in seasons if s in last.index}


# ============================================================== conversions
def season_bests(reg):
    """
    Each athlete's season best in every event they did, on the p scale.

    Args:
        reg: cleaned regular-season DataFrame.
    Returns:
        DataFrame indexed by (season, gender, athlete_id), one column per event, NaN where the athlete
        has no valid mark in that event.
    """
    v = reg[reg["status"] == "VALID"]
    return v.groupby(["season", "gender", "athlete_id", "event"])["p"].max().unstack("event")


def fit_conversions(reg_train, bw_events, cfg):
    """
    Fit the straight lines that convert one event's marks to another's (e.g. 60 m -> 100 m),
    p_target = a + b * p_source, on athlete-seasons that did both events.

    Args:
        reg_train: regular-season marks from the training seasons only.
        bw_events: {gender: set of Big West event codes}.
        cfg: settings (min_conv_pairs, conv_trim_z).
    Returns:
        dict {(gender, source_event, target_event): {a, b, resid_sd, n, n_dropped}}. resid_sd is the
        line's scatter (added to a converted athlete's uncertainty); n is the number of athlete-seasons
        used, n_dropped the outliers removed. Pairs with too few athletes are left out.
    """
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
                b, a, keep = robust_line(both[src].to_numpy(), both[tgt].to_numpy(),
                                         cfg.get("conv_trim_z", 0))
                resid = (both[tgt] - (a + b * both[src]))[keep]
                conv[(g, src, tgt)] = dict(a=a, b=b, resid_sd=resid.std(ddof=1),
                                           n=int(keep.sum()), n_dropped=int((~keep).sum()))
    # inverse-variance weight of a converted mark: a direct mark varies by the athlete's race-to-race
    # spread (sigma); a converted one also carries the conversion error (resid_sd)
    sigma = race_to_race_sd(reg_train, bw_events)
    for (g, src, tgt), c in conv.items():
        s2 = sigma.get((g, tgt), np.nan) ** 2
        c["w_iv"] = float(s2 / (s2 + c["resid_sd"] ** 2)) if np.isfinite(s2) and np.isfinite(c["resid_sd"]) else np.nan
    conv["_sigma"] = sigma
    return conv


def race_to_race_sd(reg_train, bw_events):
    """
    Typical race-to-race spread of one athlete in each Big West event (p scale): median over athlete-seasons
    with 3+ valid direct marks of the standard deviation of their marks.

    Args:
        reg_train: cleaned regular-season marks (training seasons).
        bw_events: {gender: set of Big West event codes}.
    Returns:
        dict {(gender, event): sd}.
    """
    m = reg_train[(reg_train["status"] == "VALID") & reg_train["p"].notna()]
    m = m[[e in bw_events.get(g, set()) for g, e in zip(m["gender"], m["event"])]]
    sd = m.groupby(["season", "gender", "event", "athlete_id"])["p"].agg(["std", "size"])
    sd = sd[sd["size"] >= 3]
    return sd.groupby(["gender", "event"])["std"].median().to_dict()


def robust_line(x, y, trim_z=0, n_iter=3):
    """
    Fit y = a + b*x, optionally ignoring outliers.

    trim_z = 0: ordinary least squares.
    trim_z > 0: start from the Theil-Sen line (median of pairwise slopes, which far-out points can't
    drag), drop points more than trim_z robust standard deviations from it, refit least squares on the
    rest; repeated up to n_iter times.

    Args:
        x, y: numpy arrays of the same length.
        trim_z: outlier cut-off in robust standard deviations (0 = off).
        n_iter: maximum refits.
    Returns:
        (b, a, keep): slope, intercept, and a boolean array of the points used in the final fit.
    """
    keep = np.ones(len(x), dtype=bool)
    if not trim_z:
        b, a = np.polyfit(x, y, 1)
        return b, a, keep
    from scipy.stats import theilslopes
    b, a = theilslopes(y, x)[:2]
    for _ in range(n_iter):
        r = y - (a + b * x)
        rs = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        if not np.isfinite(rs) or rs == 0:
            break
        new_keep = np.abs(r - np.median(r[keep])) <= trim_z * rs
        if new_keep.sum() < 5:
            break
        changed = not (new_keep == keep).all()
        keep = new_keep
        b, a = np.polyfit(x[keep], y[keep], 1)
        if not changed:
            break
    return b, a, keep


# ============================================================== virtual multi scores
# World Athletics scoring tables. Track: A * (B - seconds)^C. Field: A * (mark - B)^C (jumps in cm, throws in m).
WA_TABLES = {
    "Dec": {"100": (25.4347, 18, 1.81, "t"), "LJ": (0.14354, 220, 1.40, "cm"), "SP": (51.39, 1.5, 1.05, "m"),
            "HJ": (0.8465, 75, 1.42, "cm"), "400": (1.53775, 82, 1.81, "t"), "110H": (5.74352, 28.5, 1.92, "t"),
            "DT": (12.91, 4, 1.10, "m"), "PV": (0.2797, 100, 1.35, "cm"), "JT": (10.14, 7, 1.08, "m"),
            "1500": (0.03768, 480, 1.85, "t")},
    "Hep": {"100H": (9.23076, 26.7, 1.835, "t"), "HJ": (1.84523, 75, 1.348, "cm"), "SP": (56.0211, 1.5, 1.05, "m"),
            "200": (4.99087, 42.5, 1.81, "t"), "LJ": (0.188807, 210, 1.41, "cm"), "JT": (15.9803, 3.8, 1.04, "m"),
            "800": (0.11193, 254, 1.88, "t")},
}
MULTI_GENDER = {"Dec": "M", "Hep": "W"}


def _wa_points(multi, event, p):
    """
    World Athletics scoring-table points for component marks of a decathlon or heptathlon.

    Args:
        multi: "Dec" or "Hep".
        event: component event code, e.g. "100", "LJ".
        p: mark(s) on the p scale (scalar or array).
    Returns:
        numpy array of points (0 for marks below the table's floor).
    """
    A, B, C, kind = WA_TABLES[multi][event]
    if kind == "t":
        x = B - np.exp(-p)
    else:
        x = np.exp(p) * (100 if kind == "cm" else 1) - B
    return np.where(x > 0, A * np.clip(x, 0, None) ** C, 0.0)


def virtual_scores(reg, multi, cfg):
    """
    A "virtual" multi score built from each athlete's season bests in the component events.
    Component marks made during a multi (same athlete, multi date or the next day) are left out.

    Args:
        reg: cleaned regular-season DataFrame.
        multi: "Dec" (men) or "Hep" (women).
        cfg: settings (virtual_multi_min = component events needed).
    Returns:
        DataFrame, one row per athlete-season with enough component events: season, athlete_id,
        n_events, pts (average points per event), date, p_virtual (log of the scaled-up total),
        athlete_name, team, gender, event. Empty DataFrame if there are none.
    """
    g, comps = MULTI_GENDER[multi], WA_TABLES[multi]
    v = reg[(reg["status"] == "VALID") & (reg["gender"] == g)]
    days = v[v["event"].isin(MULTIS)][["athlete_id", "date"]].drop_duplicates()
    inside = set(zip(days["athlete_id"], days["date"])) | \
        set(zip(days["athlete_id"], days["date"] + pd.Timedelta(days=1)))
    c = v[v["event"].isin(comps)]
    c = c[[(a, d) not in inside for a, d in zip(c["athlete_id"], c["date"])]]
    if c.empty:
        return pd.DataFrame()
    best = c.groupby(["season", "athlete_id", "event"]).agg(p=("p", "max"), date=("date", "max")).reset_index()
    best["pts"] = [float(_wa_points(multi, e, p)) for e, p in zip(best["event"], best["p"])]
    s = best.groupby(["season", "athlete_id"]).agg(n_events=("pts", "size"), pts=("pts", "mean"),
                                                   date=("date", "max")).reset_index()
    s = s[(s["n_events"] >= cfg.get("virtual_multi_min", 4)) & (s["pts"] > 0)]
    s["p_virtual"] = np.log(s["pts"] * len(comps))
    info = c.sort_values("date").groupby("athlete_id")[["athlete_name", "team"]].last()
    return s.join(info, on="athlete_id").assign(gender=g, event=multi)


def fit_virtual_multi(reg_train, cfg):
    """
    Fit the line from virtual score to real season-best multi score, on athletes with both.

    Args:
        reg_train: regular-season marks from the training seasons only.
        cfg: settings (min_conv_pairs, conv_trim_z, virtual_multi_min).
    Returns:
        dict {multi: {a, b, resid_sd, n}} for "Dec" and/or "Hep"; a multi is left out if too few athletes
        have both.
    """
    fits = {}
    for multi, g in MULTI_GENDER.items():
        vs = virtual_scores(reg_train, multi, cfg)
        if vs.empty:
            continue
        real = (reg_train[(reg_train["status"] == "VALID") & (reg_train["gender"] == g) & (reg_train["event"] == multi)]
                .groupby(["season", "athlete_id"])["p"].max().rename("p_real"))
        both = vs.join(real, on=["season", "athlete_id"]).dropna(subset=["p_real"])
        if len(both) < cfg["min_conv_pairs"]:
            continue
        x, y = both["p_virtual"].to_numpy(), both["p_real"].to_numpy()
        b, a, keep = robust_line(x, y, cfg.get("conv_trim_z", 0))
        resid = y[keep] - (a + b * x[keep])
        fits[multi] = dict(a=a, b=b, resid_sd=resid.std(ddof=1), n=int(keep.sum()))
    return fits


def virtual_multi_marks(reg, fits, cfg):
    """
    Converted multi marks for athletes with no real multi score that season (V6a, not in the final model).

    Args:
        reg: cleaned regular-season DataFrame (training + test seasons).
        fits: output of fit_virtual_multi.
        cfg: settings (conv_weight, virtual_multi_min).
    Returns:
        DataFrame in the same format as expand_marks (kind="convert", source="virtual", weight
        conv_weight, extra variance resid_sd**2), or None if there are none.
    """
    out = []
    for multi, fit in fits.items():
        g = MULTI_GENDER[multi]
        vs = virtual_scores(reg, multi, cfg)
        if vs.empty:
            continue
        has_real = reg[(reg["gender"] == g) & (reg["event"] == multi) & (reg["status"] == "VALID")]
        has_real = set(zip(has_real["season"], has_real["athlete_id"]))
        vs = vs[[(s, a) not in has_real for s, a in zip(vs["season"], vs["athlete_id"])]]
        out.append(pd.DataFrame({
            "season": vs["season"], "date": vs["date"], "athlete_id": vs["athlete_id"],
            "athlete_name": vs["athlete_name"], "team": vs["team"], "status": "VALID",
            "p": fit["a"] + fit["b"] * vs["p_virtual"], "gender": g, "event": multi,
            "kind": "convert", "source": "virtual", "w": cfg["conv_weight"], "xv": fit["resid_sd"] ** 2}))
    return pd.concat(out, ignore_index=True) if out else None


def expand_marks(reg, conv, bw_events, cfg):
    """
    Map every regular-season mark onto the Big West event(s) it informs, converting where needed.

    Args:
        reg: cleaned regular-season DataFrame.
        conv: fitted conversions (fit_conversions).
        bw_events: {gender: set of Big West event codes}.
        cfg: settings (conv_weight).
    Returns:
        DataFrame, one row per (mark, target Big West event): season, date, athlete_id, athlete_name,
        team, status, p (converted to the target event), gender, event (the TARGET event), kind
        ("direct" / "convert" / "yard"), source (the original event), w (weight: 1 for direct,
        conv_weight for converted) and xv (extra variance from the conversion).
    """
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
                auto = cfg["conv_weight"] == "auto"
                if c is not None:
                    r["p"] = c["a"] + c["b"] * r["p"]
                    w_iv = c.get("w_iv", np.nan)
                    r["w"] = (w_iv if np.isfinite(w_iv) else 0.25) if auto else cfg["conv_weight"]
                    r["xv"] = c["resid_sd"] ** 2
                elif ev in DIST_M and tgt in DIST_M:
                    r["p"] = r["p"] - RIEGEL_EXP * np.log(DIST_M[tgt] / DIST_M[ev])
                    s2 = conv.get("_sigma", {}).get((g, tgt), np.nan) ** 2
                    r["w"] = (s2 / (s2 + RIEGEL_SD ** 2) if np.isfinite(s2) else 0.5) if auto else cfg["conv_weight"]
                    r["xv"] = RIEGEL_SD ** 2
                else:
                    continue
            out.append(r)
    return pd.concat(out, ignore_index=True)


# ============================================================== athlete profiles
KEYS = ["gender", "event", "athlete_id"]


def flag_season_blowups(m, cfg):
    """
    Relabel regular-season marks far WORSE than the athlete's own median in that event (falls,
    injuries, pulling up) as FAILED. They then count toward the athlete's failure history instead of
    dragging down their average or inflating their spread. "Far" = more than reg_blowup_z times the
    event's typical spread. Needs 2+ valid marks to compare against.

    Args:
        m: marks in expand_marks format.
        cfg: settings (reg_blowup_z; 0 turns this off).
    Returns:
        DataFrame: a copy of m with those rows' status set to "FAILED".
    """
    zt = cfg.get("reg_blowup_z", 0)
    if not zt:
        return m
    ok = m["status"].eq("VALID")
    if not ok.any():
        return m
    v = m[ok]
    g = v.groupby(KEYS)["p"]
    med, n = g.transform("median"), g.transform("size")
    sd_ev = (v[v["kind"] == "direct"].groupby(KEYS)["p"].std()
             .groupby(level=[0, 1]).median())
    sd_typ = pd.Series(sd_ev.reindex(pd.MultiIndex.from_frame(v[["gender", "event"]])).to_numpy(),
                       index=v.index).fillna(sd_ev.median() if len(sd_ev) else 0.03)
    bad = (n >= 2) & (v["p"] < med - zt * sd_typ)
    m = m.copy()
    m.loc[bad[bad].index, "status"] = "FAILED"
    return m


def athlete_profiles(marks, as_of, cfg):
    """
    Summarise each athlete's season in each Big West event, from marks before `as_of`.

    Args:
        marks: marks in expand_marks format (usually one season).
        as_of: Timestamp; only marks before it are used and recency weights count back from it.
        cfg: settings (half_life_days, k_spread, reg_blowup_z).
    Returns:
        DataFrame, one row per (gender, event, athlete_id):
          team       last team the athlete competed for
          mu         recency- and source-weighted mean of p (the athlete's typical level)
          sd         spread, shrunk toward the event's typical spread, plus conversion error
          best       season best (direct marks preferred)
          p_fail     chance of no valid mark, shrunk toward the event's failure rate
          n_fail, n_entries   failed starts and all starts in the event (direct marks)
          n_eff      effective number of marks after weighting
          n_direct   number of direct (unconverted) valid marks
    """
    m = marks[marks["date"] < pd.Timestamp(as_of)].copy()
    m = flag_season_blowups(m, cfg)
    age = (pd.Timestamp(as_of) - m["date"]).dt.days
    m["wt"] = m["w"] * 0.5 ** (age / cfg["half_life_days"])
    v = m[m["status"] == "VALID"].copy()
    d = v[v["kind"] == "direct"]

    # event-level priors
    sd_ev = d.groupby(KEYS)["p"].std().groupby(level=[0, 1]).median()
    sd_global = sd_ev.median() if len(sd_ev) else 0.03
    dm = m[m["kind"] == "direct"]
    pf_ev = (dm["status"] == "FAILED").groupby([dm["gender"], dm["event"]]).mean()

    # weighted mean / spread
    v["wp"] = v["wt"] * v["p"]
    agg = v.groupby(KEYS).agg(sw=("wt", "sum"), swp=("wp", "sum"))
    agg["sw2"] = (v["wt"] ** 2).groupby([v[k] for k in KEYS]).sum()
    agg["mu"] = agg["swp"] / agg["sw"]
    v = v.join(agg["mu"], on=KEYS)
    v["wdev2"] = v["wt"] * (v["p"] - v["mu"]) ** 2
    v["wxv"] = v["wt"] * v["xv"]
    agg["var_own"] = v.groupby(KEYS)["wdev2"].sum() / agg["sw"]
    agg["xv"] = v.groupby(KEYS)["wxv"].sum() / agg["sw"]
    agg["n_eff"] = agg["sw"] ** 2 / agg["sw2"]
    agg["n_direct"] = d.groupby(KEYS).size().reindex(agg.index).fillna(0)
    agg["best"] = d.groupby(KEYS)["p"].max().reindex(agg.index)
    agg["best"] = agg["best"].fillna(v.groupby(KEYS)["p"].max())

    idx = agg.index.droplevel("athlete_id")
    prior_sd = pd.Series(sd_ev.reindex(idx).to_numpy(), index=agg.index).fillna(sd_global)
    k = cfg["k_spread"]
    agg["sd"] = np.sqrt((agg["n_eff"] * agg["var_own"] + k * prior_sd ** 2)
                        / (agg["n_eff"] + k) + agg["xv"])

    # failure rate from direct entries
    n_all = dm.groupby(KEYS).size().reindex(agg.index).fillna(0)
    n_fail = dm[dm["status"] == "FAILED"].groupby(KEYS).size().reindex(agg.index).fillna(0)
    prior_pf = pd.Series(pf_ev.reindex(idx).to_numpy(), index=agg.index).fillna(0.03)
    k_fail = 6  # only used with p_fail_source="athlete" (not the final model); results identical for 3-12
    agg["p_fail"] = (n_fail + k_fail * prior_pf) / (n_all + k_fail)
    agg["n_fail"], agg["n_entries"] = n_fail, n_all

    last_team = m.sort_values("date").groupby(KEYS)["team"].last()
    agg["team"] = last_team.reindex(agg.index)
    return agg[["team", "mu", "sd", "best", "p_fail", "n_fail", "n_entries", "n_eff", "n_direct"]].reset_index()


def estimate_half_life(reg_train, bw, seasons, cdates, bw_events, cfg):
    """
    Recency half-life measured from the training seasons: the value that makes each athlete's weighted
    average land closest to their Big West mark.

    Uses direct marks only (so it measures recency, not conversions) and athletes with 3+ such marks.
    For a candidate half-life h, every athlete's weighted mean mu is built as of each season's Big West;
    the error is |Big West p - mu| after removing each event's median championship shift, ignoring
    results more than 15% off (blowups). h is found by a bounded 1-D search between 3 and 365 days.

    Args:
        reg_train: cleaned regular-season marks from the training seasons.
        bw: cleaned Big West results.
        seasons: training seasons.
        cdates: {season: profile date}.
        bw_events: {gender: set of Big West event codes}.
        cfg: settings (other profile settings are held fixed).
    Returns:
        float: estimated half-life in days.
    """
    from scipy.optimize import minimize_scalar
    direct = expand_marks(reg_train, {}, bw_events, cfg)
    direct = direct[direct["kind"] == "direct"]
    act = bw[bw["season"].isin(seasons) & (bw["status"] == "VALID") & ~bw["event"].isin(RELAYS)]

    def err(log_h):
        c = {**cfg, "half_life_days": float(np.exp(log_h))}
        out = []
        for s in seasons:
            prof = athlete_profiles(direct[direct["season"] == s], cdates[s], c)
            d = act[act["season"] == s].merge(prof[KEYS + ["mu", "n_direct"]], on=KEYS)
            d = d[d["n_direct"] >= 3]
            r = d["p"] - d["mu"]
            out.append(r - r.groupby([d["gender"], d["event"]]).transform("median"))
        r = pd.concat(out)
        return np.abs(r[r.abs() < 0.15]).mean()

    best = minimize_scalar(err, bounds=(np.log(3), np.log(365)), method="bounded", options={"xatol": 0.02})
    return float(np.exp(best.x))

def estimate_k_spread(reg_train, bw, seasons, cdates, bw_events, cfg):
    """
    Spread shrinkage measured from the training seasons: how much to trust an athlete's own spread vs the
    event's typical spread when predicting how variable their Big West result will be.

    For a candidate k, every athlete's sd is built as of each season's Big West (direct marks only). The
    Big West residual (p - mu, minus each event's median shift; results more than 15% off ignored) is scored
    with a normal log-likelihood using sd times one scale per event (as in fit_calibration). k is found by a
    bounded 1-D search between 0.1 and 200.

    Args:
        reg_train: cleaned regular-season marks from the training seasons.
        bw: cleaned Big West results.
        seasons: training seasons.
        cdates: {season: profile date}.
        bw_events: {gender: set of Big West event codes}.
        cfg: settings (other profile settings are held fixed).
    Returns:
        float: estimated k_spread (in "effective marks").
    """
    from scipy.optimize import minimize_scalar
    direct = expand_marks(reg_train, {}, bw_events, cfg)
    direct = direct[direct["kind"] == "direct"]
    act = bw[bw["season"].isin(seasons) & (bw["status"] == "VALID") & ~bw["event"].isin(RELAYS)]

    def nll(log_k):
        c = {**cfg, "k_spread": float(np.exp(log_k))}
        out = []
        for s in seasons:
            prof = athlete_profiles(direct[direct["season"] == s], cdates[s], c)
            d = act[act["season"] == s].merge(prof[KEYS + ["mu", "sd"]], on=KEYS)
            r = d["p"] - d["mu"]
            r = r - r.groupby([d["gender"], d["event"]]).transform("median")
            out.append(pd.DataFrame({"r": r, "sd": d["sd"], "g": d["gender"], "e": d["event"]}))
        x = pd.concat(out)
        x = x[x["r"].abs() < 0.15]
        c2 = (x["r"] ** 2 / x["sd"] ** 2).groupby([x["g"], x["e"]]).transform("mean")
        return float(np.mean(np.log(np.sqrt(c2) * x["sd"]) + 0.5 * x["r"] ** 2 / (c2 * x["sd"] ** 2)))

    best = minimize_scalar(nll, bounds=(np.log(0.1), np.log(200)), method="bounded", options={"xatol": 0.02})
    return float(np.exp(best.x))

# ============================================================== athlete features (hybrid model)
def event_group(event):
    """
    Event group used by the regressions and the by-group tables.

    Args:
        event: event code.
    Returns:
        str: "relay", "multi", "vertical" (HJ/PV), "horizontal" (LJ/TJ), "throw", "sprint" (up to 400 m
        and hurdles) or "distance".
    """
    if event in RELAYS:
        return "relay"
    if event in MULTIS:
        return "multi"
    if event in {"HJ", "PV"}:
        return "vertical"
    if event in {"LJ", "TJ"}:
        return "horizontal"
    if event in {"SP", "DT", "HT", "JT", "WT"}:
        return "throw"
    if event in {"55", "60", "100", "200", "300", "400", "60H", "100H", "110H", "300H", "400H"}:
        return "sprint"
    return "distance"


FEATURE_GROUPS = ["distance", "horizontal", "vertical", "throw", "multi"]   # "sprint" is the reference


def athlete_features(prof, entries, marks_season, as_of, cfg, ref_k=None):
    """
    Features for the athlete-specific offset and failure regressions, one row per individual (non-relay)
    entrant who has a profile. Uses only marks before `as_of`.

    Args:
        prof: athlete profiles (athlete_profiles) for that season.
        entries: the field: DataFrame with gender, event, athlete_id.
        marks_season: that season's marks in expand_marks format (for counting starts and failures).
        as_of: Timestamp profiles are measured at.
        cfg: settings (reg_blowup_z).
        ref_k: None to measure sb_level within the entrants; or {(gender, event): k} to measure it
            against the top k (used in "everyone" mode so it means the same as in training).
    Returns:
        DataFrame with gender, event, athlete_id, mu, best and:
          sb_level  season best as a z-score within the field (+ = better), clipped to +-4
          gap       season best minus weighted mean, in % (bigger = less consistent), clipped to 0-20
          risk      share of the athlete's individual starts that season that failed
          n_starts  individual starts that season
          group     event group (event_group)
    """
    e = entries[~entries["event"].isin(RELAYS)][KEYS].drop_duplicates()
    f = e.merge(prof[KEYS + ["mu", "best"]], on=KEYS, how="inner")
    if f.empty:
        return f.assign(sb_level=[], gap=[], risk=[], n_starts=[], group=[])
    if ref_k is None:
        g = f.groupby(["gender", "event"])["best"]
        sd = g.transform("std").replace(0, np.nan)
        f["sb_level"] = ((f["best"] - g.transform("mean")) / sd).fillna(0.0).clip(-4, 4)
    else:
        # field is everyone with marks: compare with the top k (a typical number of entrants for the event),
        # so sb_level means the same as in training, where it is measured among actual entrants
        def _z(d):
            k = int(ref_k.get(d.name, 16))
            top = d.nlargest(k)
            sd = top.std() if top.std() > 0 else np.nan
            return ((d - top.mean()) / sd).fillna(0.0).clip(-4, 4)
        f["sb_level"] = f.groupby(["gender", "event"])["best"].transform(_z)
    f["gap"] = (100 * (f["best"] - f["mu"])).clip(lower=0, upper=20)
    m = marks_season[(marks_season["date"] < pd.Timestamp(as_of)) & (marks_season["kind"] == "direct")
                     & ~marks_season["event"].isin(RELAYS)]
    m = flag_season_blowups(m, cfg)
    m = m[m["status"].isin(["VALID", "FAILED"])]
    starts = m.drop_duplicates(["athlete_id", "event", "date", "p", "status"])
    r = starts.assign(bad=starts["status"].eq("FAILED")).groupby("athlete_id")["bad"].agg(["mean", "size"])
    f["risk"] = f["athlete_id"].map(r["mean"]).fillna(0.0)
    f["n_starts"] = f["athlete_id"].map(r["size"]).fillna(0.0)
    f["group"] = f["event"].map(event_group)
    return f


def _design(f):
    """
    Design matrix for the two regressions.

    Args:
        f: features from athlete_features (needs group, gender, sb_level, gap, risk, n_starts).
    Returns:
        DataFrame, same rows as f: intercept, one dummy per event group (sprint is the reference), men,
        sb_level, gap, risk x 10 (so the coefficient is per +0.10), log_starts = log(1 + n_starts).
    """
    X = pd.DataFrame({"intercept": 1.0}, index=f.index)
    for grp in FEATURE_GROUPS:
        X[f"group_{grp}"] = (f["group"] == grp).astype(float)
    X["men"] = (f["gender"] == "M").astype(float)
    X["sb_level"] = f["sb_level"]
    X["gap"] = f["gap"]
    X["risk"] = f["risk"] * 10                       # per +0.10
    X["log_starts"] = np.log1p(f["n_starts"])
    return X


def _ridge_ols(X, y, lam):
    """
    Ridge least squares (the intercept is not penalised).

    Args:
        X: design matrix (numpy, n x k), first column = intercept.
        y: target (numpy, length n).
        lam: ridge penalty.
    Returns:
        numpy array of k coefficients.
    """
    A = X.T @ X + lam * np.diag([0.0] + [1.0] * (X.shape[1] - 1))   # don't penalise the intercept
    return np.linalg.solve(A, X.T @ y)


def _ridge_logit(X, y, lam, n_iter=50):
    """
    Ridge logistic regression fitted by Newton's method (the intercept is not penalised).

    Args:
        X: design matrix (numpy, n x k), first column = intercept.
        y: 0/1 outcomes (numpy, length n).
        lam: ridge penalty.
        n_iter: maximum Newton steps.
    Returns:
        numpy array of k coefficients on the log-odds scale.
    """
    beta = np.zeros(X.shape[1])
    beta[0] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
    pen = lam * np.diag([0.0] + [1.0] * (X.shape[1] - 1))
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-6, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + pen, X.T @ (y - p) - pen @ beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


# ============================================================== calibration
def _mad_sd(x):
    """
    Robust standard deviation: 1.4826 x median absolute deviation, so extreme values can't drive it.

    Args:
        x: array-like of numbers (non-finite values ignored).
    Returns:
        float, or NaN if fewer than 3 finite values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def fit_calibration(marks, bw, seasons, cdates, cfg):
    """
    Learn how athletes perform at Big West compared with their regular-season profile, from the
    training seasons' championships.

    For each training season it builds profiles as of that season's Big West and compares them with the
    actual results (residual = p_actual - mu). Results more than blowup_z robust spreads below prediction
    count as failures instead of noise. Per-event values are shrunk toward the pooled value (cal_prior_n).

    Args:
        marks: marks in expand_marks format covering the training seasons.
        bw: cleaned Big West results (all seasons; only `seasons` are used).
        seasons: list of training seasons.
        cdates: {season: profile date} from champ_dates.
        cfg: settings (blowup_z, cal_prior_n, dns_as_fail, offset_model, pfail_model, ridge, ...).
    Returns:
        dict:
          cal[(gender, event)] = {offset, scale, p_fail, n, mu_shrink, n_blowups, offset_left}
              offset      average championship residual (+ = athletes do better than their mean)
              scale       how wide championship results are vs the profile's sd
              p_fail      share of entrants who failed, didn't start or blew up
              offset_left what the offset regression leaves unexplained for this event
          cal["_pooled"] = the same values over all events, plus rho (day form), impute_q,
              team_sd, n_nohist, and the regressions: offset_beta, pfail_beta (pandas Series of
              coefficients named like the _design columns), offset_n, pfail_n.
    """
    recs, nohist, feats = [], [], []
    hybrid = cfg.get("offset_model", "event") == "regression" or cfg.get("pfail_model", "event") == "logistic"
    for s in seasons:
        prof = athlete_profiles(marks[marks["season"] == s], cdates[s], cfg)
        act = bw[(bw["season"] == s) & (bw["status"] == "VALID")][KEYS + ["p"]]
        recs.append(act.rename(columns={"p": "p_act"}).merge(prof, on=KEYS).assign(season=s))
        if hybrid:
            feats.append(athlete_features(prof, bw[bw["season"] == s], marks[marks["season"] == s],
                                          cdates[s], cfg).assign(season=s))
        # where did entrants with no regular-season profile finish within their field? (0 = last, 1 = first)
        a = act.merge(prof[KEYS + ["mu"]], on=KEYS, how="left")
        a["q"] = a.groupby(["gender", "event"])["p"].rank(pct=True)
        nohist.append(a.loc[a["mu"].isna() & ~a["event"].isin(RELAYS), "q"])
    m = pd.concat(recs, ignore_index=True)
    nh = pd.concat(nohist)
    # shrink toward the old default (0.2) when there are few such athletes
    impute_q = float(np.clip((nh.sum() + 10 * 0.2) / (len(nh) + 10), 0.05, 0.6))
    m["res"] = m["p_act"] - m["mu"]

    # 1. flag blowups per event using robust (median / MAD) statistics
    m["blowup"] = False
    for _, idx in m.groupby(["gender", "event"]).groups.items():
        g = m.loc[idx]
        z = (g["res"] - g["res"].median()) / g["sd"]
        sc = _mad_sd(z)
        if np.isfinite(sc) and sc > 0:
            m.loc[idx, "blowup"] = z < -cfg["blowup_z"] * sc
    ok = m[~m["blowup"]].copy()

    # 1b. regression to the mean: do weaker-than-field athletes beat their prediction (and stronger
    #     ones fall short)? res ~ lambda * (field mean mu - mu), within each season-event field.
    ok["gap"] = ok.groupby(["season", "gender", "event"])["mu"].transform("mean") - ok["mu"]
    lam = estimate_mu_shrink(ok, cfg)
    ok["res"] = ok["res"] - lam * ok["gap"]          # calibrate offset/scale on the adjusted prediction

    # 2. offset and scale from normal results only
    off_all = ok["res"].mean()
    scale_all = _mad_sd((ok["res"] - off_all) / ok["sd"])

    # 3. championship failure rate: FAILED status + blowups, over all entrants in the training seasons
    # entrants who start but fail, plus (if listed) entrants who don't start at all
    no_score = ["FAILED", "DID_NOT_COMPETE"] if cfg.get("dns_as_fail", True) else ["FAILED"]
    b = bw[bw["season"].isin(seasons) & bw["status"].isin(["VALID"] + no_score)]
    fails = b.assign(f=b["status"].isin(no_score)).groupby(["gender", "event"])["f"].agg(["sum", "size"])
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
                           mu_shrink=lam,n_blowups=int(m[(m["gender"] == g) & (m["event"] == e)]["blowup"].sum()))
        
    cal["_pooled"] = dict(offset=off_all, scale=float(np.clip(scale_all, 0.5, 3.0)),
                          p_fail=float(pf_all), mu_shrink=lam, n=len(ok), n_blowups=int(m["blowup"].sum()),
                          rho=estimate_day_form_rho(ok, cfg), impute_q=impute_q, n_nohist=int(len(nh)))

    cal["_pooled"]["team_sd"] = estimate_team_sd(ok, cfg)

    if hybrid and feats:
        F = pd.concat(feats, ignore_index=True)
        # (a) athlete-specific offset: regress the championship residual on the features
        if cfg.get("offset_model", "event") == "regression":
            d = ok[~ok["event"].isin(RELAYS)][["season"] + KEYS + ["res"]].merge(F, on=["season"] + KEYS)
            X = _design(d).to_numpy()
            beta = _ridge_ols(X, d["res"].to_numpy(), cfg.get("ridge", 1.0))
            d["res_left"] = d["res"] - X @ beta
            # what's left per event, shrunk toward 0 like the plain offsets
            left = d.groupby(["gender", "event"])["res_left"].agg(["mean", "size"])
            for (g, e), r in left.iterrows():
                if (g, e) in cal:
                    cal[(g, e)]["offset_left"] = float(r["mean"] * r["size"] / (r["size"] + n0))
            cal["_pooled"]["offset_beta"] = pd.Series(beta, index=_design(d).columns)
            cal["_pooled"]["offset_n"] = len(d)
        # (b) athlete-specific failure chance: logistic on the features
        if cfg.get("pfail_model", "event") == "logistic":
            no_score = ["FAILED", "DID_NOT_COMPETE"] if cfg.get("dns_as_fail", True) else ["FAILED"]
            ent = bw[bw["season"].isin(seasons) & ~bw["event"].isin(RELAYS)
                     & bw["status"].isin(["VALID"] + no_score)][["season"] + KEYS + ["status"]]
            ent = ent.merge(F, on=["season"] + KEYS)
            blown = m[m["blowup"]][["season"] + KEYS].assign(blown=True)
            ent = ent.merge(blown, on=["season"] + KEYS, how="left")
            y = (ent["status"].isin(no_score) | ent["blown"].fillna(False).astype(bool)).astype(float).to_numpy()
            beta = _ridge_logit(_design(ent).to_numpy(), y, cfg.get("ridge", 1.0))
            cal["_pooled"]["pfail_beta"] = pd.Series(beta, index=_design(ent).columns)
            cal["_pooled"]["pfail_n"] = len(ent)
    return cal


def estimate_team_sd(ok, cfg):
    """
    How much teammates' Big West results move together (shared team form), on the p scale (V6c, off).

    Args:
        ok: championship residuals without blowups (built inside fit_calibration).
        cfg: settings (team_form_sd: 0 = off, a number = use it, "auto" = estimate).
    Returns:
        float: standard deviation of the shared team effect (0 if off or not detectable).
    """
    setting = cfg.get("team_form_sd", 0)
    if setting != "auto":
        return float(setting or 0)
    d = ok[~ok["event"].isin(RELAYS)].copy()
    if "team" not in d or "season" not in d:
        return 0.0
    d["r"] = d["res"] - d.groupby(["season", "gender", "event"])["res"].transform("mean")
    sc = _mad_sd(d["r"])
    if not np.isfinite(sc) or sc == 0:
        return 0.0
    d["r"] = d["r"].clip(-4 * sc, 4 * sc)
    g = d.groupby(["season", "gender", "team"])["r"].agg(["mean", "size"])
    g = g[g["size"] >= 3]
    between = g["mean"].var() - (d["r"].var() / g["size"]).mean()
    return float(np.sqrt(max(between, 0.0)))


def estimate_mu_shrink(ok, cfg):
    """
    Share of the gap to the field average that athletes close at Big West (regression to the mean).

    Args:
        ok: championship residuals without blowups, with a gap column (built inside fit_calibration).
        cfg: settings (mu_shrink: 0 = off, a number = use it, "auto" = estimate).
    Returns:
        float between 0 and 0.6.
    """
    setting = cfg.get("mu_shrink", 0)
    if setting != "auto":
        return float(setting or 0)
    d = ok[~ok["event"].isin(RELAYS)]
    grp = [d["season"], d["gender"], d["event"]]
    x = d["gap"] - d["gap"].groupby(grp).transform("mean")
    y = d["res"] - d["res"].groupby(grp).transform("mean")
    if (x ** 2).sum() == 0:
        return 0.0
    return float(np.clip((x * y).sum() / (x ** 2).sum(), 0.0, 0.6))


def estimate_day_form_rho(ok, cfg):
    """
    How much an athlete's good or bad day carries across their individual events at the same Big West.

    Args:
        ok: championship residuals without blowups (built inside fit_calibration).
        cfg: settings (day_form_rho: "auto" = estimate, a number = use it).
    Returns:
        float between 0 and 0.6: correlation used to mix shared and event-specific noise (0 if fewer
        than 30 pairs of results).
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
    """
    Read the place -> points tables from the Big West results.

    Args:
        bw: cleaned Big West results (training seasons).
    Returns:
        dict {"ind": [10, 8, 6, 5, 4, 3, 2, 1], "relay": [...]}: points for 1st, 2nd, ... (falls back to
        DEFAULT_POINTS if the file has none).
    """
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


# ============================================================== simulation
PRIOR_SD_INFLATE = 1.5   # widen spread when using last season's profile


def build_field(prof, entries, gender, event, prof_prev=None, impute_q=0.2, impute_sd_mult=1.5):
    """
    Everyone entered in one event, with the profile the simulation will use for each.

    Args:
        prof: this season's athlete profiles.
        entries: the field: DataFrame with gender, event, athlete_id, athlete_name, team.
        gender, event: which event to build.
        prof_prev: last season's profiles (previous_profiles), or None.
        impute_q: field quantile where athletes with no marks are placed (0 = bottom, 1 = top).
        impute_sd_mult: their spread = field's median spread x this.
    Returns:
        DataFrame, one row per entrant: athlete_id, athlete_name, team, the profile columns (mu, sd,
        best, p_fail, ...), source ("current" = this season's marks; "prior_season" = last season's
        profile with sd x 1.5; "none" = no marks) and imputed (True if placed at impute_q).
        None if no entrant has any marks.
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
        for col in ["mu", "sd", "best", "p_fail", "n_fail", "n_entries", "n_eff", "n_direct"]:
            f.loc[use, col] = f.loc[use, "athlete_id"].map(pp[col])
        f.loc[use, "sd"] = f.loc[use, "sd"] * PRIOR_SD_INFLATE
        f.loc[use, "source"] = "prior_season"
    f["imputed"] = f["mu"].isna()
    if f["imputed"].all():
        return None
    if f["imputed"].any():   # no regular-season marks: assume lower-field athlete, wide spread
        f.loc[f["imputed"], "mu"] = f["mu"].quantile(impute_q)
        f.loc[f["imputed"], "sd"] = f["sd"].median() * impute_sd_mult
        f.loc[f["imputed"], "p_fail"] = f["p_fail"].median()
        f.loc[f["imputed"], "best"] = -np.inf
    return f.reset_index(drop=True)


def simulate_field(f, cal, pts, n_sims, rng, p_fail_source="athlete", day_form=None, rho=0.0,
                   two_rounds=False, round_rho=0.5, team_form=None, team_sd=0.0):
    """
    Monte Carlo one event: draw every athlete's performance n_sims times and turn places into points.

    Each athlete's centre = mu + athlete-specific offset (offset_i + offset_left) or the event offset;
    spread = sd x the event's scale. Failures (probability p_fail_i or p_fail) score 0. With day form,
    an athlete's noise is shared across their events: sqrt(rho) * day_form + sqrt(1 - rho) * own.
    With two_rounds, a prelim and a final are drawn (sharing round_rho of the noise), the top len(pts)
    from the prelim advance, and only the final decides their places.

    Args:
        f: the field from build_field (optionally with offset_i, p_fail_i from attach_features).
        cal: this event's calibration dict (offset, scale, p_fail, offset_left, ...).
        pts: points by place, e.g. [10, 8, 6, 5, 4, 3, 2, 1].
        n_sims: number of simulated meets.
        rng: numpy random Generator.
        p_fail_source: where p_fail comes from when there is no p_fail_i: "athlete" (profile),
            "bigwest" (event rate) or "blend".
        day_form: function athlete_id -> array of n_sims shared draws, or None.
        rho: day-form correlation.
        two_rounds: simulate prelims + final.
        round_rho: share of noise shared by prelim and final.
        team_form: function team -> array of n_sims shared draws, or None (V6c, off).
        team_sd: size of the team effect on the p scale.
    Returns:
        numpy array (n_sims, n_athletes): points each athlete scored in each simulated meet.
    """
    n = len(f)
    if p_fail_source == "bigwest" and "p_fail" in cal:
        f = f.assign(p_fail=cal["p_fail"])
    elif p_fail_source == "blend" and "p_fail" in cal:
        # athlete's own failures, shrunk toward the event's Big West failure rate
        nf = f["n_fail"].fillna(0) if "n_fail" in f else 0
        ne = f["n_entries"].fillna(0) if "n_entries" in f else 0
        k = cal.get("k_fail", 6)
        f = f.assign(p_fail=(nf + k * cal["p_fail"]) / (ne + k))
    mu = f["mu"].to_numpy().astype(float)
    lam = cal.get("mu_shrink", 0.0)
    if lam:                                       # regression to the mean: pull toward the field average
        known = ~f["imputed"].fillna(False).astype(bool) if "imputed" in f else np.ones(n, bool)
        mu = mu + lam * (f.loc[known, "mu"].mean() - mu)
    if "offset_i" in f:   # hybrid: athlete-specific offset where features exist, else the event offset
        oi = f["offset_i"].to_numpy(dtype=float)
        mu = mu + np.where(np.isfinite(oi), oi + cal.get("offset_left", 0.0), cal["offset"])
    else:
        mu = mu + cal["offset"]
    if "p_fail_i" in f:
        pi = f["p_fail_i"].to_numpy(dtype=float)
        f = f.assign(p_fail=np.where(np.isfinite(pi), pi, f["p_fail"].to_numpy(dtype=float)))
    sd = f["sd"].to_numpy() * cal["scale"]
    eps = rng.standard_normal((n_sims, n))
    if day_form is not None and rho > 0:
        shared = np.column_stack([day_form(a) for a in f["athlete_id"]])
        eps = np.sqrt(rho) * shared + np.sqrt(1 - rho) * eps
    if team_form is not None and team_sd > 0:   # shared team day: same draw for every athlete on a team
        mu = mu + team_sd * np.column_stack([team_form(t) for t in f["team"]])
    fail = rng.random((n_sims, n)) < f["p_fail"].to_numpy()
    k_pts = len(pts)
    if two_rounds and n > k_pts:
        # Prelim and final each get their own draw; round_rho of the noise is shared (the athlete's form
        # that day), the rest is round-specific. Each round keeps the same overall spread.
        a, b = np.sqrt(round_rho), np.sqrt(1 - round_rho)
        p1 = mu + sd * (a * eps + b * rng.standard_normal((n_sims, n)))
        p2 = mu + sd * (a * eps + b * rng.standard_normal((n_sims, n)))
        in_final_fail = rng.random((n_sims, n)) < 0.5            # a failure happens in either round
        fail1, fail2 = fail & ~in_final_fail, fail & in_final_fail
        p1[fail1] = -np.inf
        # top k_pts prelim marks advance
        rank1 = np.argsort(np.argsort(-p1, axis=1), axis=1)
        finalist = rank1 < k_pts
        p2[fail2] = -np.inf
        # finalists are ranked by the final, everyone else by the prelim, below all finalists
        perf = np.where(finalist, 1e6 + p2, p1)
        fail = fail1 | (fail2 & finalist)
    else:
        perf = mu + sd * eps
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


def crps(samples, y):
    """
    Continuous ranked probability score of a set of simulated values against the actual value. Measures
    how far the whole simulated range is from what happened; equals the absolute error for a single guess.

    Args:
        samples: 1-D numpy array of simulated values (e.g. a team's points in each simulation).
        y: the actual value.
    Returns:
        float (same units as y; lower is better).
    """
    x = np.sort(samples)
    n = len(x)
    return np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2


# ============================================================== year-to-year improvement
def fit_year_change(reg_train, seasons, cfg):
    """
    Typical change in an athlete's season best from one season to the next in the same event (V6b, off).
    2019 -> 2021 counts as consecutive (no 2020 season).

    Args:
        reg_train: regular-season marks from the training seasons.
        seasons: training seasons.
        cfg: settings (unused; kept for a consistent signature).
    Returns:
        dict {by_group: median change per event group (Series), by_level: median change per
        (group, quartile of last season's best) with 20+ athletes (Series), n: pairs used}, or None.
    """
    v = reg_train[(reg_train["status"] == "VALID") & ~reg_train["event"].isin(RELAYS)]
    sb = v.groupby(["season", "gender", "event", "athlete_id"])["p"].max().rename("best").reset_index()
    order = sorted(s for s in seasons if s in set(sb["season"]))
    nxt = {a: b for a, b in zip(order[:-1], order[1:])}
    prev = sb.assign(season=sb["season"].map(nxt)).dropna(subset=["season"])
    pairs = prev.merge(sb, on=["season", "gender", "event", "athlete_id"], suffixes=("_prev", ""))
    if pairs.empty:
        return None
    pairs["change"] = pairs["best"] - pairs["best_prev"]
    pairs["group"] = pairs["event"].map(event_group)
    pct = pairs.groupby(["season", "gender", "event"])["best_prev"].rank(pct=True)
    pairs["level"] = np.ceil(pct * 4).clip(1, 4).astype(int)
    by_group = pairs.groupby("group")["change"].median()
    lvl = pairs.groupby(["group", "level"])["change"].agg(["median", "size"])
    by_level = lvl.loc[lvl["size"] >= 20, "median"]
    return dict(by_group=by_group, by_level=by_level, n=len(pairs))


def apply_year_change(prof_prev, fit, cfg):
    """
    Shift last season's profiles by the typical improvement (V6b, off).

    Args:
        prof_prev: last season's profiles, or None.
        fit: output of fit_year_change, or None.
        cfg: settings (prior_improve: "off", "group" or "level").
    Returns:
        DataFrame: prof_prev with mu and best shifted (relays unchanged); returned unchanged if off.
    """
    mode = cfg.get("prior_improve", "off")
    if prof_prev is None or fit is None or mode == "off":
        return prof_prev
    pp = prof_prev.copy()
    pp["group"] = pp["event"].map(event_group)
    shift = pp["group"].map(fit["by_group"]).fillna(0.0)
    if mode == "level":
        pct = pp.groupby(["gender", "event"])["best"].rank(pct=True)
        pp["level"] = np.ceil(pct * 4).clip(1, 4).astype(int)
        lv = pd.Series([fit["by_level"].get((g, l), np.nan) for g, l in zip(pp["group"], pp["level"])],
                       index=pp.index)
        shift = lv.fillna(shift)
    shift = shift.where(pp["group"] != "relay", 0.0)      # relay lineups change; no shift
    pp["mu"] = pp["mu"] + shift
    pp["best"] = pp["best"] + shift
    return pp.drop(columns=[c for c in ("group", "level") if c in pp])


def previous_profiles(marks, bw, season, cdates, cfg):
    """
    Last season's profiles, used for entrants with no marks this season. That season's Big West result
    counts as one more mark.

    Args:
        marks: marks in expand_marks format (must include `season`).
        bw: cleaned Big West results.
        season: the previous season (test - 1).
        cdates: {season: profile date}.
        cfg: settings passed to athlete_profiles.
    Returns:
        DataFrame in athlete_profiles format, or None if that season has no date (e.g. 2020).
    """
    if season not in cdates:
        return None
    prev = marks[marks["season"] == season]
    b = bw[(bw["season"] == season) & bw["status"].isin(["VALID", "FAILED"])]
    b = b[["season", "athlete_id", "athlete_name", "team", "status", "p", "gender", "event"]].copy()
    b["date"] = cdates[season]
    b["kind"], b["source"], b["w"], b["xv"] = "direct", "bigwest", 1.0, 0.0
    both = pd.concat([prev, b], ignore_index=True)
    return athlete_profiles(both, cdates[season] + pd.Timedelta(days=1), cfg)


# ============================================================== field of everyone with marks
def everyone_entries(reg_season, bw_events, as_of):
    """
    Pre-meet field without entry lists ("everyone" mode): every athlete and relay team with at least one
    valid OUTDOOR mark in a Big West event that season, plus 5000 m runners in the 10,000 m. Indoor marks
    are those with no weather data (`temp` missing); without that column, January-February counts as indoor.

    Args:
        reg_season: cleaned regular-season marks for the test season.
        bw_events: {gender: set of Big West event codes}.
        as_of: Timestamp; only marks before it count.
    Returns:
        DataFrame in the same shape as the Big West entries: gender, event, athlete_id, athlete_name,
        team, season, points (0), status ("VALID"), place (NaN).
    """
    r = reg_season[(reg_season["status"] == "VALID") & (reg_season["date"] < pd.Timestamp(as_of))]
    indoor = r["temp"].isna() if "temp" in r else r["date"].dt.month.isin([1, 2])
    r = r[~indoor]
    rows = []
    for g in ("M", "W"):
        evs = bw_events.get(g, set())
        rg = r[r["gender"] == g]
        rows.append(rg[rg["event"].isin(evs)])
        if "10000" in evs:
            rows.append(rg[rg["event"] == "5000"].assign(event="10000"))
    e = pd.concat(rows, ignore_index=True)
    e = (e.sort_values("date").groupby(["gender", "event", "athlete_id"])
         .agg(athlete_name=("athlete_name", "last"), team=("team", "last"), season=("season", "first"))
         .reset_index())
    return e.assign(points=0.0, status="VALID", place=np.nan)


# ============================================================== performance-list baseline
def performance_list_points(reg_season, bw_events, pts_tab, as_of):
    """
    The baseline: a TFRRS-style meet calculator. Every athlete and relay with an outdoor mark in a Big
    West event is ranked by their outdoor season best in that event (no conversions, no limit on events
    per athlete, wind-aided marks included); places 1-8 get the points table.

    Args:
        reg_season: cleaned regular-season marks for the test season.
        bw_events: {gender: set of Big West event codes}.
        pts_tab: points tables (points_tables).
        as_of: Timestamp; only marks before it count.
    Returns:
        DataFrame, one row per athlete-event on the list: gender, event, athlete_id, best (p scale),
        team, place, points (0 below 8th).
    """
    r = reg_season[(reg_season["status"] == "VALID") & (reg_season["date"] < pd.Timestamp(as_of))]
    indoor = r["temp"].isna() if "temp" in r else r["date"].dt.month.isin([1, 2])
    r = r[~indoor]
    r = r[[e in bw_events.get(g, set()) for g, e in zip(r["gender"], r["event"])]]
    sb = (r.sort_values("date").groupby(["gender", "event", "athlete_id"])
          .agg(best=("p", "max"), team=("team", "last")).reset_index())
    sb["place"] = sb.groupby(["gender", "event"])["best"].rank(ascending=False, method="first").astype(int)
    out = []
    for kind, grp in sb.groupby(sb["event"].isin(RELAYS)):
        table = pts_tab["relay" if kind else "ind"]
        out.append(grp.assign(points=grp["place"].map({i + 1: p for i, p in enumerate(table)}).fillna(0.0)))
    return pd.concat(out, ignore_index=True)


# ============================================================== athlete-specific predictions
def predict_features(prof, entries, marks_season, as_of, pooled, cfg, ref_k=None):
    """
    Each entrant's athlete-specific championship offset and failure chance, from the regressions fitted
    in fit_calibration.

    Args:
        prof: this season's athlete profiles.
        entries: the field (gender, event, athlete_id).
        marks_season: this season's marks in expand_marks format.
        as_of: Timestamp profiles are measured at.
        pooled: cal["_pooled"] from fit_calibration (holds offset_beta / pfail_beta).
        cfg: settings.
        ref_k: see athlete_features (None for entry lists).
    Returns:
        DataFrame from athlete_features plus offset_i (expected championship shift on the p scale, e.g.
        -0.005 = 0.5% worse) and p_fail_i (chance of failing). None if neither regression is on.
    """
    if "offset_beta" not in pooled and "pfail_beta" not in pooled:
        return None
    feats = athlete_features(prof, entries, marks_season, as_of, cfg, ref_k)
    if feats.empty:
        return None
    X = _design(feats)
    if "offset_beta" in pooled:
        feats["offset_i"] = X[pooled["offset_beta"].index].to_numpy() @ pooled["offset_beta"].to_numpy()
    if "pfail_beta" in pooled:
        z = X[pooled["pfail_beta"].index].to_numpy() @ pooled["pfail_beta"].to_numpy()
        feats["p_fail_i"] = 1 / (1 + np.exp(-z))
    return feats


def attach_features(f, feats, gender, event):
    """
    Add offset_i / p_fail_i to one event's field. Only athletes using this season's profile get them
    (the features describe this season's marks); everyone else keeps the event-level values.

    Args:
        f: the field from build_field.
        feats: output of predict_features, or None.
        gender, event: which event f is.
    Returns:
        DataFrame: f with offset_i and p_fail_i columns (NaN for non-current athletes). Unchanged for
        relays or when feats is None.
    """
    if feats is None or event in RELAYS:
        return f
    fx = feats[(feats["gender"] == gender) & (feats["event"] == event)].set_index("athlete_id")
    cur = f["source"].eq("current") if "source" in f else pd.Series(False, index=f.index)
    for col in ("offset_i", "p_fail_i"):
        if col in fx:
            f[col] = np.where(cur, f["athlete_id"].map(fx[col]), np.nan)
    return f


def round_settings(cfg, event):
    """
    The simulate_field arguments for prelims + final.

    Args:
        cfg: settings (two_rounds, round_rho).
        event: event code.
    Returns:
        dict {two_rounds: True only if cfg["two_rounds"] and the event is in TWO_ROUND_EVENTS,
        round_rho: cfg["round_rho"]}.
    """
    return dict(two_rounds=bool(cfg.get("two_rounds", False)) and event in TWO_ROUND_EVENTS,
                round_rho=cfg.get("round_rho", 0.5))


# ============================================================== one fold
def run_fold(reg, bw, cal_seasons, test, cdates, bw_events, cfg, verbose=True):
    """
    Fit the model on the training seasons and predict and score one test season.

    Steps: conversions -> marks -> calibration -> test-season profiles -> field (entry lists or everyone)
    -> athlete-specific offsets -> simulate every event -> team totals -> metrics vs the actual results
    and the performance-list baseline.

    Args:
        reg: cleaned regular-season marks (all seasons).
        bw: cleaned Big West results (all seasons).
        cal_seasons: list of training seasons, e.g. [2019, 2021, 2022].
        test: the season to predict, e.g. 2023.
        cdates: {season: profile date} from champ_dates.
        bw_events: {gender: set of Big West event codes}.
        cfg: settings (CONFIG or a modified copy).
        verbose: print a short summary of the fold.
    Returns:
        dict:
          summary     one row of metrics: mark_mae_pct, mark_bias_pct, brier_model, brier_plist,
                      te_crps_model, te_mae_model, te_mae_plist, te_cover80, {M,W}_total_crps_model,
                      {M,W}_total_mae_model, {M,W}_total_mae_plist, {M,W}_total_cover80,
                      {M,W}_spearman_model, {M,W}_spearman_plist, and fold info (n_imputed, ...)
          athletes    one row per athlete-event: exp_pts, p_score, actual_pts, source, imputed
          team_event  one row per team-event: exp_pts, med, lo80, hi80, actual, crps, plist_pts
          totals      one row per team: exp_pts, med, lo80, hi80, p_win, actual, crps, plist_pts
          ctx         everything the coaching tools need (profiles, calibration, entries, features, cfg)
    """
    rng = np.random.default_rng(cfg["seed"] + test)

    if cfg.get("half_life_days") == "auto":           # measure recency weighting from the training seasons
        cfg = {**cfg, "half_life_days": estimate_half_life(reg[reg["season"].isin(cal_seasons)], bw, cal_seasons,
                                                           cdates, bw_events, cfg)}
        
    if cfg.get("k_spread") == "auto":                 # measure spread shrinkage from the training seasons
        cfg = {**cfg, "k_spread": estimate_k_spread(reg[reg["season"].isin(cal_seasons)], bw, cal_seasons,
                                                    cdates, bw_events, cfg)}   
         
    conv = fit_conversions(reg[reg["season"].isin(cal_seasons)], bw_events, cfg)
    marks = expand_marks(reg[reg["season"].isin(cal_seasons + [test])], conv, bw_events, cfg)
    if cfg.get("virtual_multi", False):
        vfits = fit_virtual_multi(reg[reg["season"].isin(cal_seasons)], cfg)
        vm = virtual_multi_marks(reg[reg["season"].isin(cal_seasons + [test])], vfits, cfg)
        if vm is not None:
            marks = pd.concat([marks, vm], ignore_index=True)
    cal = fit_calibration(marks, bw, cal_seasons, cdates, cfg)
    pts_tab = points_tables(bw[bw["season"].isin(cal_seasons)])
    prof = athlete_profiles(marks[marks["season"] == test], cdates[test], cfg)
    prof_prev = previous_profiles(marks, bw, test - 1, cdates, cfg)
    if cfg.get("prior_improve", "off") != "off":
        prof_prev = apply_year_change(prof_prev, fit_year_change(reg[reg["season"].isin(cal_seasons)],
                                                                 cal_seasons, cfg), cfg)

    bw_test = bw[bw["season"] == test]
    ref_k = None
    if cfg.get("field_mode", "entrants") == "everyone":
        entries = everyone_entries(reg[reg["season"] == test], bw_events, cdates[test])
        # typical number of entrants per event, from the training championships
        ref_k = (bw[bw["season"].isin(cal_seasons)].groupby(["season", "gender", "event"]).size()
                 .groupby(level=["gender", "event"]).median().round().astype(int).to_dict())
    else:
        entries = bw_test
    team_actual = bw_test.groupby(["gender", "event", "team"])["points"].sum().to_dict()
    marks_test = marks[marks["season"] == test]
    feats = predict_features(prof, entries, marks_test, cdates[test], cal["_pooled"], cfg, ref_k)
    mark_rows = []

    # shared day form for athletes entered in 2+ individual events (same draws across their events)
    rho = cal["_pooled"].get("rho", 0.0)
    impute_q = (cal["_pooled"].get("impute_q", 0.2) if cfg.get("impute_q", 0.2) == "auto"
                else float(cfg.get("impute_q", 0.2)))
    ind = entries[~entries["event"].isin(RELAYS)]
    multi_ids = set(ind.groupby("athlete_id")["event"].nunique().loc[lambda x: x >= 2].index)
    form_cache = {}
    team_cache = {}
    team_sd = cal["_pooled"].get("team_sd", 0.0)

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
        f = build_field(prof, entries, g, e, prof_prev, impute_q, cfg.get("impute_sd_mult", 1.5))
        if f is None:
            skipped.append(f"{g} {e}")
            continue
        c = cal.get((g, e), cal["_pooled"])
        f = attach_features(f, feats, g, e)
        pts = pts_tab["relay" if e in RELAYS else "ind"]
        # predicted centre (log scale) for the mark-level metric
        if e not in RELAYS:
            ctr = f["mu"].to_numpy(dtype=float)
            if "offset_i" in f:
                oi = f["offset_i"].to_numpy(dtype=float)
                ctr = ctr + np.where(np.isfinite(oi), oi + c.get("offset_left", 0.0), c["offset"])
            else:
                ctr = ctr + c["offset"]
            act_p = bw_test[(bw_test["gender"] == g) & (bw_test["event"] == e)
                            & (bw_test["status"] == "VALID")].set_index("athlete_id")["p"]
            mr = pd.DataFrame({"athlete_id": f["athlete_id"], "pred": ctr, "source": f["source"]})
            mr["act"] = mr["athlete_id"].map(act_p)
            mark_rows.append(mr.assign(gender=g, event=e))
        def team_form(t, g=g):
            if (g, t) not in team_cache:
                team_cache[(g, t)] = rng.standard_normal(cfg["n_sims"]).astype(np.float32)
            return team_cache[(g, t)]

        sims = simulate_field(f, c, pts, cfg["n_sims"], rng, cfg.get("p_fail_source", "athlete"),
                              day_form=day_form, rho=rho, team_form=team_form, team_sd=team_sd,
                              **round_settings(cfg, e))

        actual = bw_test[(bw_test["gender"] == g) & (bw_test["event"] == e)] \
            .groupby("athlete_id")["points"].sum()
        a = f[["athlete_id", "athlete_name", "team", "source", "imputed"]].copy()
        a["gender"], a["event"] = g, e
        a["exp_pts"] = sims.mean(0)
        a["p_score"] = (sims > 0).mean(0)
        a["actual_pts"] = a["athlete_id"].map(actual).fillna(0).to_numpy()
        ath_rows.append(a)

        for t in f["team"].unique():
            mask = (f["team"] == t).to_numpy()
            ts = sims[:, mask].sum(1)
            totals.setdefault(g, {}).setdefault(t, np.zeros(cfg["n_sims"]))
            totals[g][t] += ts
            act = float(team_actual.get((g, e, t), 0.0))     # real points, even if a scorer isn't in the field
            te_rows.append(dict(gender=g, event=e, team=t, exp_pts=ts.mean(), med=np.median(ts),
                                lo80=np.percentile(ts, 10), hi80=np.percentile(ts, 90),
                                actual=act, crps=crps(ts, act)))

    ath = pd.concat(ath_rows, ignore_index=True)
    te = pd.DataFrame(te_rows)

    tot_rows = []
    for g, teams in totals.items():
        names = list(teams)
        mat = np.column_stack([teams[t] for t in names])
        win = np.bincount(mat.argmax(1), minlength=len(names)) / cfg["n_sims"]
        for j, t in enumerate(names):
            act = te[(te["gender"] == g) & (te["team"] == t)]["actual"].sum()
            tot_rows.append(dict(gender=g, team=t, exp_pts=mat[:, j].mean(), med=np.median(mat[:, j]),
                                 lo80=np.percentile(mat[:, j], 10), hi80=np.percentile(mat[:, j], 90),
                                 p_win=win[j], actual=act, crps=crps(mat[:, j], act)))
    tot = pd.DataFrame(tot_rows)

    # ---- performance-list baseline (meet-calculator style): scored independently of the model's field
    pl = performance_list_points(reg[reg["season"] == test], bw_events, pts_tab, cdates[test])
    pl_te = pl.groupby(["gender", "event", "team"])["points"].sum()
    real_te = bw_test.groupby(["gender", "event", "team"])["points"].sum()
    keys = pl_te.index.union(real_te.index)
    pl_te, real_te = pl_te.reindex(keys, fill_value=0.0), real_te.reindex(keys, fill_value=0.0)
    pl_tot = pl_te.groupby(level=["gender", "team"]).sum()
    real_tot = real_te.groupby(level=["gender", "team"]).sum()
    tkeys = pl_tot.index.union(real_tot.index)
    pl_tot, real_tot = pl_tot.reindex(tkeys, fill_value=0.0), real_tot.reindex(tkeys, fill_value=0.0)
    # the list's points next to the model's, for team reports
    te["plist_pts"] = [float(pl_te.get((g, e, t), 0.0)) for g, e, t in zip(te["gender"], te["event"], te["team"])]
    tot["plist_pts"] = [float(pl_tot.get((g, t), 0.0)) for g, t in zip(tot["gender"], tot["team"])]

    # ---- metrics
    scored = (ath["actual_pts"] > 0).astype(float)
    summ = dict(
        test=test,
        brier_model=np.mean((ath["p_score"] - scored) ** 2),
        ath_mae_model=np.mean(np.abs(ath["exp_pts"] - ath["actual_pts"])),
        te_crps_model=te["crps"].mean(),
        te_mae_model=np.mean(np.abs(te["med"] - te["actual"])),     # single guess, comparable with the list
        te_cover80=np.mean((te["actual"] >= te["lo80"]) & (te["actual"] <= te["hi80"])),
        n_imputed=int(ath["imputed"].sum()),
        n_prior_season=int((ath["source"] == "prior_season").sum()),
        day_form_rho=rho,
        impute_q=impute_q,
        mu_shrink=cal["_pooled"].get("mu_shrink", 0.0),
        te_mae_plist=float(np.mean(np.abs(pl_te - real_te))),
    )
    # performance-list Brier on the model's athletes: "scores" = placed in the list's top 8
    pl_score = pl.assign(s=(pl["points"] > 0).astype(float)).set_index(["gender", "event", "athlete_id"])["s"]
    pl_p = pd.Series(list(zip(ath["gender"], ath["event"], ath["athlete_id"]))).map(pl_score).fillna(0.0).to_numpy()
    summ["brier_plist"] = float(np.mean((pl_p - scored.to_numpy()) ** 2))
    for g in ("M", "W"):
        if g in pl_tot.index.get_level_values("gender"):
            a_, b_ = pl_tot.xs(g, level="gender"), real_tot.xs(g, level="gender")
            summ[f"{g}_total_mae_plist"] = float(np.mean(np.abs(a_ - b_)))
            summ[f"{g}_spearman_plist"] = spearmanr(a_, b_)[0]
    # mark level: how close the predicted centre is to the actual mark (current-season athletes,
    # blowups more than 50% off excluded, like the teammate's model)
    mk = pd.concat(mark_rows, ignore_index=True) if mark_rows else pd.DataFrame(columns=["pred", "act", "source"])
    mk = mk[(mk["source"] == "current") & mk["act"].notna()]
    err = 100 * (mk["act"] - mk["pred"])
    err = err[err > 100 * np.log(0.5)]
    summ["mark_mae_pct"] = float(err.abs().mean()) if len(err) else np.nan
    summ["mark_bias_pct"] = float(err.mean()) if len(err) else np.nan
    for g, grp in tot.groupby("gender"):
        summ[f"{g}_total_crps_model"] = grp["crps"].mean()
        summ[f"{g}_total_mae_model"] = np.mean(np.abs(grp["med"] - grp["actual"]))
        summ[f"{g}_total_cover80"] = np.mean((grp["actual"] >= grp["lo80"]) & (grp["actual"] <= grp["hi80"]))
        summ[f"{g}_spearman_model"] = spearmanr(grp["exp_pts"], grp["actual"])[0]

    if verbose:
        print(f"\n===== Fold: calibrate {cal_seasons[0]}-{cal_seasons[-1]} -> test {test} =====")
        print(f"conversions fitted: {len(conv)} | events skipped (no marks): {skipped or 'none'}")
        print(f"day-form correlation used: {rho:.2f} | imputed athletes placed at field quantile {impute_q:.2f} | "
              f"blowups in training: {cal['_pooled']['n_blowups']} | mu shrink: {cal['_pooled'].get('mu_shrink', 0):.2f}")
        print(f"entrants using last season: {summ['n_prior_season']} | "
              f"entrants with no marks (imputed): {summ['n_imputed']}")
        print(f"Brier (scores?)        model {summ['brier_model']:.3f} | list {summ['brier_plist']:.3f}")
        print(f"Team-event points MAE  model {summ['te_mae_model']:.2f} | list {summ['te_mae_plist']:.2f}"
              f"   (model CRPS {summ['te_crps_model']:.2f}, 80% coverage {summ['te_cover80']:.0%})")
        for g, grp in tot.groupby("gender"):
            print(f"{g} team totals MAE      model {summ[f'{g}_total_mae_model']:.1f} | "
                  f"list {summ.get(f'{g}_total_mae_plist', np.nan):.1f}   Spearman model "
                  f"{summ[f'{g}_spearman_model']:.2f} | list {summ.get(f'{g}_spearman_plist', np.nan):.2f}")

    ctx = dict(prof=prof, prof_prev=prof_prev, impute_q=impute_q, cal=cal, pts=pts_tab, entries=entries, cfg=cfg,
               season=test, feats=feats, marks_season=marks_test, as_of=cdates[test], ref_k=ref_k)
    return dict(summary=summ, athletes=ath, team_event=te, totals=tot, ctx=ctx)


# ============================================================== coaching tools
def exceedance(ctx, gender, event, team, extra=None, k_max=None, seed=1):
    """
    Chance a team scores at least k points in one event, with the same model as run_fold.

    Args:
        ctx: run_fold(...)["ctx"].
        gender, event: which event, e.g. "M", "400".
        team: team name as in the data, e.g. "UC SAN DIEGO".
        extra: rows to add to the field (e.g. a recruit from recruit_row), or None.
        k_max: largest k to report (default: points for 1st + 2nd + 3rd).
        seed: random seed.
    Returns:
        Series indexed by k = 0..k_max: P(team scores >= k points).
    """
    cfg = ctx["cfg"]
    f = build_field(ctx["prof"], ctx["entries"], gender, event, ctx.get("prof_prev"),
                    ctx.get("impute_q", 0.2), cfg.get("impute_sd_mult", 1.5))
    f = attach_features(f, ctx.get("feats"), gender, event)
    if extra is not None:
        f = pd.concat([f, extra], ignore_index=True)
    pts = ctx["pts"]["relay" if event in RELAYS else "ind"]
    c = ctx["cal"].get((gender, event), ctx["cal"]["_pooled"])
    sims = simulate_field(f, c, pts, cfg["n_sims"], np.random.default_rng(seed),
                          cfg.get("p_fail_source", "athlete"), **round_settings(cfg, event))
    ts = sims[:, (f["team"] == team).to_numpy()].sum(1)
    k_max = k_max or int(sum(pts[:3]))
    return pd.Series({k: (ts >= k).mean() for k in range(0, k_max + 1)}, name="P(>=k)")


def recruit_row(ctx, gender, event, team, mark, name="RECRUIT"):
    """
    A hypothetical athlete to add to a field with exceedance.

    Args:
        ctx: run_fold(...)["ctx"].
        gender, event: the event.
        team: team the recruit would score for.
        mark: the recruit's typical championship-ready mark in natural units (seconds, meters, points).
        name: label for the row.
    Returns:
        one-row DataFrame in build_field format: mu = best = the mark on the p scale, sd and p_fail =
        the event's median.
    """
    p = np.log(mark) if higher_is_better(event) else -np.log(mark)
    ev = ctx["prof"][(ctx["prof"]["gender"] == gender) & (ctx["prof"]["event"] == event)]
    return pd.DataFrame([dict(athlete_id=name, athlete_name=name, team=team, mu=p,
                              sd=ev["sd"].median(), best=p, p_fail=ev["p_fail"].median(),
                              n_eff=0, n_direct=0, imputed=False)])


def whatif(ctx, gender, event, team, mark):
    """
    A team's chances in one event with and without a recruit.

    Args:
        ctx: run_fold(...)["ctx"].
        gender, event, team: as in exceedance.
        mark: the recruit's mark in natural units.
    Returns:
        DataFrame indexed by k with columns current and with_recruit: P(team scores >= k) each way.
    """
    base = exceedance(ctx, gender, event, team)
    new = exceedance(ctx, gender, event, team, extra=recruit_row(ctx, gender, event, team, mark))
    return pd.DataFrame({"current": base, "with_recruit": new})


def team_report(result, team="UC SAN DIEGO"):
    """
    One team's expected points by event from a fold's results.

    Args:
        result: output of run_fold.
        team: team name as in the data.
    Returns:
        DataFrame: gender, event, exp_pts, lo80, hi80, actual, sorted by expected points within gender.
    """
    te = result["team_event"]
    r = te[te["team"] == team][["gender", "event", "exp_pts", "lo80", "hi80", "actual"]]
    return r.sort_values(["gender", "exp_pts"], ascending=[True, False]).round(2)


# ============================================================== main
def run_model(regular, bigwest, final=False, team="UC SAN DIEGO", cfg=None, save_csv=True):
    """
    Run the validation folds (and optionally the 2026 test) from files, print a summary and save CSVs.

    Args:
        regular: path to the regular-season CSV, or the DataFrame.
        bigwest: path to the Big West results CSV, or the DataFrame.
        final: also run FINAL_FOLD (the 2026 test).
        team: team for the printed report.
        cfg: settings to change from CONFIG, or None.
        save_csv: write pred_athletes_{year}.csv, pred_team_event_{year}.csv, pred_team_totals_{year}.csv
            and fold_summary.csv to the working directory.
    Returns:
        dict {test_year: run_fold result, ..., "summary": DataFrame of fold metrics}.
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
    """
    Command-line entry point: python bigwest_model.py REGULAR.csv BIGWEST.csv [--final] [--team NAME].

    Returns:
        the output of run_model.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("regular")
    ap.add_argument("bigwest")
    ap.add_argument("--final", action="store_true", help="also run the 2026 held-out test")
    ap.add_argument("--team", default="UC SAN DIEGO")
    args = ap.parse_args()
    return run_model(args.regular, args.bigwest, final=args.final, team=args.team)



# ============================================================== quick evaluation helper
# (model_history.ipynb is the full, reproducible comparison; this is for one-off checks)
LOG = []
EVAL_COLS = ["mark_mae_pct", "mark_bias_pct", "brier_model", "brier_plist",
             "te_crps_model", "te_mae_model", "te_mae_plist",
             "M_total_crps_model", "M_total_mae_model", "M_total_mae_plist",
             "W_total_crps_model", "W_total_mae_model", "W_total_mae_plist",
             "M_spearman_model", "M_spearman_plist", "W_spearman_model", "W_spearman_plist"]


def evaluate(label, reg, bw, cdates, bw_events, cfg_overrides=None, folds=None):
    """
    Quick check of a setting: run the validation folds with CONFIG + cfg_overrides, average the metrics
    and add a row to LOG. (model_history.ipynb is the full, reproducible comparison.)
    Settings used while loading data (multi_min_frac, drop_wind_aided) need reg/bw reloaded.

    Args:
        label: row name in the log.
        reg, bw: cleaned regular-season and Big West DataFrames.
        cdates: {season: profile date}.
        bw_events: {gender: set of Big West event codes}.
        cfg_overrides: dict of settings to change, or None.
        folds: list of (training seasons, test season), default FOLDS.
    Returns:
        DataFrame: every row logged so far (EVAL_COLS), rounded. Clear with LOG.clear().
    """
    cfg = {**CONFIG, **(cfg_overrides or {})}
    rows = [run_fold(reg, bw, cs, t, cdates, bw_events, cfg, verbose=False)["summary"]
            for cs, t in (folds or FOLDS)]
    s = pd.DataFrame(rows).mean(numeric_only=True)
    LOG.append(s.reindex(EVAL_COLS).rename(label))
    return pd.DataFrame(LOG).round(3)


# ============================================================== mock meets
def simulate_meet(ctx, entries, n_sims=None, seed=7):
    """
    Simulate a championship with any entry list, using a fitted fold's model (same steps as run_fold).

    Args:
        ctx: run_fold(...)["ctx"] (the profiles and calibration of that season).
        entries: DataFrame with athlete_id, athlete_name, team, gender, event (Big West codes, e.g. "100",
            "LJ"; relays: athlete_id = "TEAM|4x100"). Athletes need marks that season to get a profile;
            anyone without one is imputed.
        n_sims: number of simulated meets (default cfg["n_sims"]).
        seed: random seed.
    Returns:
        dict:
          athletes    one row per athlete-event: exp_pts, p_score, p_win_event, source
          team_event  one row per team-event: exp_pts, lo80, hi80
          totals      one row per team: exp_pts, lo80, hi80, p_win (sorted by expected points)
    """
    cfg = ctx["cfg"]
    n_sims = n_sims or cfg["n_sims"]
    rng = np.random.default_rng(seed)
    e = entries.copy()
    e["team"] = e["team"].astype(str).str.strip().str.upper().replace(TEAM_ALIASES)
    e["athlete_id"] = e["athlete_id"].astype(str)
    e = e.drop_duplicates(["gender", "event", "athlete_id"])
    # athlete-specific offset / failure chance, with sb_level measured within this entry list
    feats = predict_features(ctx["prof"], e, ctx["marks_season"], ctx["as_of"], ctx["cal"]["_pooled"], cfg)

    rho = ctx["cal"]["_pooled"].get("rho", 0.0)
    ind = e[~e["event"].isin(RELAYS)]
    multi_ids = set(ind.groupby("athlete_id")["event"].nunique().loc[lambda x: x >= 2].index)
    cache = {}

    def day_form(aid):
        if aid not in multi_ids:
            return rng.standard_normal(n_sims)
        if aid not in cache:
            cache[aid] = rng.standard_normal(n_sims).astype(np.float32)
        return cache[aid]

    ath_rows, te_rows, totals = [], [], {}
    for (g, ev), _ in e.groupby(["gender", "event"]):
        f = build_field(ctx["prof"], e, g, ev, ctx.get("prof_prev"),
                        ctx.get("impute_q", 0.2), cfg.get("impute_sd_mult", 1.5))
        if f is None:
            continue
        f = attach_features(f, feats, g, ev)
        c = ctx["cal"].get((g, ev), ctx["cal"]["_pooled"])
        pts = ctx["pts"]["relay" if ev in RELAYS else "ind"]
        sims = simulate_field(f, c, pts, n_sims, rng, cfg.get("p_fail_source", "athlete"),
                              day_form=day_form, rho=rho, **round_settings(cfg, ev))
        a = f[["athlete_id", "athlete_name", "team", "source"]].copy()
        a["gender"], a["event"] = g, ev
        a["exp_pts"] = sims.mean(0)
        a["p_score"] = (sims > 0).mean(0)
        a["p_win_event"] = (sims == pts[0]).mean(0)
        ath_rows.append(a)
        for t in f["team"].unique():
            ts = sims[:, (f["team"] == t).to_numpy()].sum(1)
            totals.setdefault(g, {}).setdefault(t, np.zeros(n_sims))
            totals[g][t] += ts
            te_rows.append(dict(gender=g, event=ev, team=t, exp_pts=ts.mean(),
                                lo80=np.percentile(ts, 10), hi80=np.percentile(ts, 90)))

    tot_rows = []
    for g, teams in totals.items():
        names = list(teams)
        mat = np.column_stack([teams[t] for t in names])
        win = np.bincount(mat.argmax(1), minlength=len(names)) / n_sims
        for j, t in enumerate(names):
            tot_rows.append(dict(gender=g, team=t, exp_pts=mat[:, j].mean(),
                                 lo80=np.percentile(mat[:, j], 10), hi80=np.percentile(mat[:, j], 90),
                                 p_win=win[j]))
    return dict(athletes=pd.concat(ath_rows, ignore_index=True),
                team_event=pd.DataFrame(te_rows),
                totals=pd.DataFrame(tot_rows).sort_values(["gender", "exp_pts"], ascending=[True, False]))


def entries_from_meet(reg, meet_pattern, season, teams=None):
    """
    Build an entry list for simulate_meet from a real regular-season meet.

    Args:
        reg: cleaned regular-season marks.
        meet_pattern: text or regex matched against meet_name (case-insensitive), e.g. "Azusa".
        season: season of the meet.
        teams: list of team names to keep, or None for everyone at the meet.
    Returns:
        DataFrame: athlete_id, athlete_name, team, gender, event, one row per athlete-event (Big West
        events only; relays become "TEAM|event").
    """
    m = reg[(reg["season"] == season)
            & reg["meet_name"].astype(str).str.contains(meet_pattern, case=False, regex=True)]
    if teams is not None:
        m = m[m["team"].isin([str(t).strip().upper() for t in teams])]
    bw_codes = set(BW_NAME_MAP.values())
    m = m[m["event"].isin(bw_codes)]
    return m[["athlete_id", "athlete_name", "team", "gender", "event"]].drop_duplicates()


# Command-line entry point only; skipped inside Jupyter so the file can be imported or pasted.
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    main()