"""Big West Championships mark prediction: train on 2021-2025, test once on 2026.

Predicts each athlete's championship mark from their regular season on a relative scale
(% better or worse than their season best), so every event shares one model.

    python modeling/model_bigwest.py

Writes results.md, predictions_2026.csv, chosen_model.pkl, figures/ and tables/ next to this script.
"""
from __future__ import annotations

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2021, 2022, 2023, 2024, 2025]
TEST_YEAR = 2026
BASE_METRIC = 'sb'            # 'sb' | 'median': the regular-season mark predictions scale from
RISK_SMOOTHING_K = [3, 5, 10]  # pseudo-races for the smoothed risk_score
MAX_OFF_SEASON_BEST = 50.0     # %: championship marks this far below SB are treated as "not a race"
MIN_ATHLETES_FOR_EVENT_R2 = 5  # event-years with fewer athletes are left out of within-event R²
MIN_ROWS_FOR_EVENT_SD = 8      # events with fewer training rows borrow their event group's SD
OUTLIER_IQRS = 3               # risk_score outlier fence, same rule as notebooks/risk_score.ipynb
TEMP_CENTER = 70.0             # °F; temperature enters the model as (temp - 70)
# ───────────────────────────────────────────────────────────────────────────────

import pickle
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data' / 'processed'
FIG_DIR = HERE / 'figures'
TABLE_DIR = HERE / 'tables'
RESULTS_MD = HERE / 'results.md'
PREDICTIONS_CSV = HERE / f'predictions_{TEST_YEAR}.csv'
MODEL_PKL = HERE / 'chosen_model.pkl'

# Championship event names -> regular-season event codes
EVENT_CODES = {
    '100 Meters': '100', '200 Meters': '200', '400 Meters': '400', '800 Meters': '800',
    '1500 Meters': '1500', '5000 Meters': '5000', '10,000 Meters': '10,000',
    '100 Hurdles': '100H', '110 Hurdles': '110H', '400 Hurdles': '400H', '3000 Steeplechase': '3000S',
    'High Jump': 'HJ', 'Pole Vault': 'PV', 'Long Jump': 'LJ', 'Triple Jump': 'TJ',
    'Shot Put': 'SP', 'Discus': 'DT', 'Hammer': 'HT', 'Javelin': 'JT',
}
TRACK_CODES = {'100', '200', '400', '800', '1500', '5000', '10,000', '100H', '110H', '400H', '3000S'}
EVENT_GROUP = {
    '100': 'sprints', '200': 'sprints', '400': 'sprints',
    '100H': 'hurdles', '110H': 'hurdles', '400H': 'hurdles',
    '800': 'distance', '1500': 'distance', '5000': 'distance', '10,000': 'distance', '3000S': 'distance',
    'HJ': 'jumps', 'PV': 'jumps', 'LJ': 'jumps', 'TJ': 'jumps',
    'SP': 'throws', 'DT': 'throws', 'HT': 'throws', 'JT': 'throws',
}
GROUPS = ['sprints', 'hurdles', 'distance', 'jumps', 'throws']
GROUP_LABEL = {'sprints': 'Sprints', 'hurdles': 'Hurdles', 'distance': 'Middle/distance',
               'jumps': 'Jumps', 'throws': 'Throws'}

# risk_score rules (same as notebooks/risk_score.ipynb)
DQ_CODES, DNF_CODES, NO_MARK_CODES = {'DQ', 'FS'}, {'DNF'}, {'FOUL', 'NH', 'NM'}
NOT_STARTED_CODES = {'DNS', 'NT', 'NP'}
RELAY = r'^(?:4x|DMR|SMR)'

# Chart style (dataviz reference palette)
INK, INK_2, SURFACE, GRID, AXIS, DOT = '#0b0b0b', '#52514e', '#fcfcfb', '#e1e0d9', '#c3c2b7', '#2a78d6'


def to_number(mark):
    """'10.74' -> 10.74, '3:44.68' -> 224.68, '7.45m' -> 7.45, 'DNF'/'FOUL' -> NaN."""
    text = str(mark).strip().removesuffix('m')
    try:
        return sum(float(part) * 60**i for i, part in enumerate(reversed(text.split(':'))))
    except ValueError:
        return np.nan


# ── Data ───────────────────────────────────────────────────────────────────────

def load_inputs():
    results = pd.read_csv(DATA_DIR / 'bigwest_2019_2026_results.csv', dtype={'athlete_id': 'Int64'})
    season = pd.read_csv(DATA_DIR / 'regular_season.csv', dtype={'athlete_id': 'Int64'}, low_memory=False)
    weather = pd.read_csv(DATA_DIR / 'meet_weather.csv')

    champs = weather[weather['source'] == 'championship'].copy()
    champs['year'] = champs['meet_year'].astype(int)
    champs = champs.set_index('year')
    champ_start = pd.to_datetime(champs['start_date'])

    temp = results.groupby('meet_year')['temp'].agg(['nunique', 'first'])
    assert (temp['nunique'] == 1).all(), 'Expected one championship temperature per year'

    indoor = weather.loc[weather['source'] == 'regular_season', ['meet_name', 'meet_date', 'is_indoor']]
    season = season.merge(indoor, on=['meet_name', 'meet_date'], how='left', validate='many_to_one')
    assert season['is_indoor'].notna().all(), 'Regular-season meet missing from meet_weather.csv'
    season['is_indoor'] = season['is_indoor'].astype(bool)
    season['gender'] = season['gender'].str.lower()
    season['date'] = pd.to_datetime(season['meet_start_date'])
    season['mark_num'] = season['mark'].map(to_number)
    season['before_champs'] = season['date'] < season['year'].map(champ_start)
    return results, season, champs[['meet_date', 'start_date', 'end_date', 'city']], temp['first']


def build_risk(season, cutoff_years):
    """risk_score per athlete-season, from individual races before that year's Big West.

    Bad race = DQ/FS, DNF, field no-mark (FOUL/NH/NM), or an outlier mark (further off the
    athlete's own season best in that event than Q3 + 3*IQR for the event). Outlier cutoffs
    come from `cutoff_years` only, so the test year never shapes its own features.
    """
    races = season[season['before_champs'] & ~season['event'].str.match(RELAY)].copy()
    races['code'] = races['mark'].astype(str).str.strip().str.upper()
    races = races[~races['code'].isin(NOT_STARTED_CODES)]
    lower = races['event_type'].isin(['track', 'unknown'])
    valid = races['mark_num'].notna()
    grp = races[valid].groupby(['athlete_id', 'year', 'event'])['mark_num']
    best = np.where(lower[valid], grp.transform('min'), grp.transform('max'))
    ratio = races.loc[valid, 'mark_num'] / best
    races.loc[valid, 'pct_off_best'] = np.where(lower[valid], ratio - 1, 1 - ratio) * 100
    races.loc[valid, 'n_marks'] = grp.transform('size')

    ref = races[races['year'].isin(cutoff_years) & (races['n_marks'] >= 2) & (races['pct_off_best'] > 0)]
    q = ref.groupby('event')['pct_off_best'].quantile([0.25, 0.75]).unstack()
    cutoff = q[0.75] + OUTLIER_IQRS * (q[0.75] - q[0.25])
    races['outlier'] = (races['n_marks'] >= 2) & (races['pct_off_best'] > races['event'].map(cutoff))
    races['bad'] = races['code'].isin(DQ_CODES | DNF_CODES | NO_MARK_CODES) | races['outlier']

    risk = (races.groupby(['athlete_id', 'year'])
            .agg(n_starts=('bad', 'size'), n_bad=('bad', 'sum'), last_race=('date', 'max'))
            .reset_index())
    risk['risk_raw'] = risk['n_bad'] / risk['n_starts']
    return risk, cutoff


def build_season_features(season):
    """Season best, season median and number of meets, from outdoor marks before Big West."""
    s = season[season['before_champs'] & ~season['is_indoor'] & season['mark_num'].notna()
               & season['event'].isin(EVENT_GROUP.keys())].rename(columns={'event': 'event_code'})
    keys = ['athlete_id', 'gender', 'year', 'event_code']
    is_track = s['event_code'].isin(TRACK_CODES)
    # One mark per meet (the athlete's best there)
    per_meet = pd.concat([
        s[is_track].groupby(keys + ['date', 'meet_name'])['mark_num'].min(),
        s[~is_track].groupby(keys + ['date', 'meet_name'])['mark_num'].max(),
    ]).reset_index()
    feats = (per_meet.groupby(keys)
             .agg(n_meets=('mark_num', 'size'), median=('mark_num', 'median'),
                  low=('mark_num', 'min'), high=('mark_num', 'max'), last_meet=('date', 'max'))
             .reset_index())
    feats['sb'] = np.where(feats['event_code'].isin(TRACK_CODES), feats['low'], feats['high'])
    return feats.drop(columns=['low', 'high'])


def build_model_data(results, feats, risk, champ_temp):
    """One row per athlete-event at each year's championships, with regular-season features."""
    c = results[results['event_type'].isin(['track', 'field'])].rename(columns={'meet_year': 'year'}).copy()
    c['event_code'] = c['event'].str.replace(r"^(?:Men|Women)'s ", '', regex=True).map(EVENT_CODES)
    assert c['event_code'].notna().all(), 'Unmapped championship event'
    c['final_mark'] = c['mark'].map(to_number)
    assert not c.duplicated(['athlete_id', 'year', 'event']).any()

    d = c.merge(feats, on=['athlete_id', 'gender', 'year', 'event_code'], how='inner', validate='one_to_one')
    d = d.merge(risk[['athlete_id', 'year', 'n_starts', 'n_bad', 'risk_raw', 'last_race']],
                on=['athlete_id', 'year'], how='left', validate='many_to_one')
    assert d['risk_raw'].notna().all()

    d['is_track'] = d['event_code'].isin(TRACK_CODES)
    sign = np.where(d['is_track'], -1.0, 1.0)  # so that positive = better for every event
    d['sign'] = sign
    d['perf_sb'] = sign * (d['final_mark'] - d['sb']) / d['sb'] * 100
    d['perf_median'] = sign * (d['final_mark'] - d['median']) / d['median'] * 100
    d['gap_pct'] = sign * (d['sb'] - d['median']) / d['sb'] * 100  # >= 0; bigger = less consistent
    # SB level: standing among that year's entrants in the event (z-score, positive = better).
    # Uses only pre-meet information, so it includes entrants who later had no mark.
    oriented = pd.Series(sign * d['sb'], index=d.index)
    g = oriented.groupby([d['year'], d['gender'], d['event_code']])
    d['sb_level'] = ((oriented - g.transform('mean')) / g.transform('std')).fillna(0.0)
    d['temp'] = d['year'].map(champ_temp)
    d['temp_c'] = d['temp'] - TEMP_CENTER
    d['event_group'] = pd.Categorical(d['event_code'].map(EVENT_GROUP), categories=GROUPS)
    d['track_field'] = pd.Categorical(np.where(d['is_track'], 'track', 'field'), categories=['track', 'field'])
    d['event_key'] = d['gender'] + ' ' + d['event_code']
    d['gender'] = pd.Categorical(d['gender'], categories=['women', 'men'])
    d['no_mark'] = d['final_mark'].isna()
    d['not_a_race'] = d['perf_sb'] < -MAX_OFF_SEASON_BEST

    flow = pd.DataFrame({
        'Individual championship entries': c.groupby('year').size(),
        'With a mark': c[c['final_mark'].notna()].groupby('year').size(),
        'Also have regular-season marks in the event': d[~d['no_mark']].groupby('year').size(),
        f'Dropped: >{MAX_OFF_SEASON_BEST:.0f}% off SB': d[d['not_a_race']].groupby('year').size(),
        'Modeled': d[~d['no_mark'] & ~d['not_a_race']].groupby('year').size(),
    }).fillna(0).astype(int)
    return d, flow


# ── Models ─────────────────────────────────────────────────────────────────────

GROUP, GENDER = 'C(event_group)', 'C(gender)'
FEATURE_STEPS = [  # (id, label, terms); None = no model (perf = 0)
    ('A', 'Baseline A: mark = SB', None),
    ('B', 'Baseline B: SB + average drop-off', ()),
    ('M3a', 'Event group + gender', (GROUP, GENDER)),
    ('M3', '+ SB level', (GROUP, GENDER, 'sb_level')),
    ('M4', '+ consistency gap', (GROUP, GENDER, 'sb_level', 'gap_pct')),
    ('M5', '+ risk_score + n_starts', (GROUP, GENDER, 'sb_level', 'gap_pct', 'risk', 'n_starts')),
    ('M6', '+ temperature', (GROUP, GENDER, 'sb_level', 'gap_pct', 'risk', 'n_starts', 'temp_c')),
    ('M6x', '+ temperature × event group',
     (GROUP, GENDER, 'sb_level', 'gap_pct', 'risk', 'n_starts', 'temp_c', 'temp_c:C(event_group)')),
]
CONTINUOUS = ['sb_level', 'gap_pct', 'risk', 'n_starts', 'temp_c']
TERM_LABEL = {'sb_level': 'SB level (vs field)', 'gap_pct': 'Consistency gap', 'risk': 'risk_score',
              'n_starts': 'n_starts', 'temp_c': 'Temperature', GROUP: 'Event group', GENDER: 'Gender'}


@dataclass(frozen=True)
class Spec:
    name: str
    step: str
    label: str
    terms: tuple | None
    base: str = 'sb'
    scale: bool = False
    risk_k: int | None = None
    mixed: str | None = None  # None | 'intercept' | 'slope'

    @property
    def formula(self):
        return 'y ~ ' + (' + '.join(self.terms) if self.terms else '1')

    @property
    def is_model(self):
        return self.terms is not None


def make_spec(step, base='sb', scale=False, **kw):
    sid, label, terms = next(s for s in FEATURE_STEPS if s[0] == step)
    spec = Spec(name=sid, step=sid, label=label, terms=terms, base=base, scale=scale)
    return tag(spec, **kw) if kw else retag(spec)


def retag(spec):
    tags = []
    if spec.base == 'median':
        tags.append('median base')
    if spec.scale:
        tags.append('SD-scaled')
    if spec.terms and 'C(track_field)' in spec.terms:
        tags.append('track/field')
    if spec.risk_k is not None:
        tags.append(f'smoothed risk k={spec.risk_k}')
    if spec.terms and any(t.startswith('I(') for t in spec.terms):
        tags.append(' + '.join(t[2:-1].replace(' ** 2', '²') for t in spec.terms if t.startswith('I(')))
    if spec.mixed:
        tags.append('mixed' if spec.mixed == 'intercept' else 'mixed + slope')
    return replace(spec, name=spec.step + (f" [{', '.join(tags)}]" if tags else ''))


def tag(spec, track_field=False, risk_k=None, squares=(), mixed=None):
    terms = list(spec.terms)
    if track_field:
        terms = [t.replace('C(event_group)', 'C(track_field)') for t in terms]
    terms += [f'I({v} ** 2)' for v in squares]
    return retag(replace(spec, terms=tuple(terms), risk_k=risk_k if risk_k is not None else spec.risk_k,
                         mixed=mixed or spec.mixed))


class EventSD:
    """Training-year SD of a perf column per event (falls back to gender x event group, then overall)."""

    def __init__(self, train, col):
        by_event = train.groupby('event_key')[col].agg(['std', 'size'])
        self.by_event = by_event.loc[by_event['size'] >= MIN_ROWS_FOR_EVENT_SD, 'std'].to_dict()
        self.by_group = {(str(g), str(e)): v for (g, e), v in
                         train.groupby(['gender', 'event_group'], observed=True)[col].std().items()}
        self.overall = float(train[col].std())

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj

    def __call__(self, df):
        sd = df['event_key'].map(self.by_event)
        grp = pd.Series([self.by_group.get((str(g), str(e)), np.nan)
                         for g, e in zip(df['gender'], df['event_group'])], index=df.index)
        return sd.fillna(grp).fillna(self.overall)


class SavedModel:
    """Fitted coefficients that predict through patsy directly (statsmodels formula results don't unpickle reliably)."""

    def __init__(self, rhs, params, random_effects=None):
        self.rhs, self.params, self.random_effects = rhs, pd.Series(params), random_effects or {}

    def predict(self, df):
        X = patsy.dmatrix(self.rhs, df, return_type='dataframe')
        missing = set(self.params.index) - set(X.columns)
        assert not missing, f'Design is missing columns: {missing}'
        return X[self.params.index] @ self.params


def save_model(path, spec, model, info, cutoffs):
    """Plain-Python pickle (dicts and Series only), so it loads anywhere with load_model()."""
    params = model.fe_params if spec.mixed else model.params
    effects = {k: v.to_dict() for k, v in model.random_effects.items()} if spec.mixed else {}
    payload = {
        'spec': asdict(spec), 'rhs': spec.formula.split('~', 1)[1].strip(), 'params': params.to_dict(),
        'random_effects': effects, 'risk_rate': float(info['risk_rate']),
        'sd_target': vars(info['sd_target']), 'sd_sb': vars(info['sd_sb']),
        'risk_cutoffs_pct': cutoffs.to_dict(), 'temp_center': TEMP_CENTER, 'train_years': TRAIN_YEARS,
        'usage': 'spec, model, prep = load_model(); preds = predict(spec, model, prep(rows)) where rows come from '
                 'build_model_data() in model_bigwest.py',
    }
    with open(path, 'wb') as fh:
        pickle.dump(payload, fh)


def load_model(path=MODEL_PKL):
    """Returns (spec, model, prep). prep(rows) adds the training-year columns predict() needs."""
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    spec = Spec(**{**payload['spec'], 'terms': tuple(payload['spec']['terms'] or ()) if payload['spec']['terms'] is not None else None})
    model = SavedModel(payload['rhs'], payload['params'], payload['random_effects'])
    sd_target, sd_sb = (EventSD.from_dict(payload[k]) for k in ('sd_target', 'sd_sb'))

    def prep(rows):
        rows = rows.copy()
        rows['sd_target'], rows['sd_sb'] = sd_target(rows), sd_sb(rows)
        k = spec.risk_k
        rows['risk'] = rows['risk_raw'] if k is None else (rows['n_bad'] + k * payload['risk_rate']) / (rows['n_starts'] + k)
        return rows

    return spec, model, prep


def prepare(spec, train, test):
    """Fold-specific columns, computed from the training years only."""
    target = f'perf_{spec.base}'
    sd_target, sd_sb = EventSD(train, target), EventSD(train, 'perf_sb')
    seasons = train.drop_duplicates(['athlete_id', 'year'])
    rate = seasons['n_bad'].sum() / seasons['n_starts'].sum()
    out = []
    for df in (train, test):
        df = df.copy()
        df['sd_target'], df['sd_sb'] = sd_target(df), sd_sb(df)
        df['risk'] = (df['risk_raw'] if spec.risk_k is None
                      else (df['n_bad'] + spec.risk_k * rate) / (df['n_starts'] + spec.risk_k))
        df['y'] = df[target] / df['sd_target'] if spec.scale else df[target]
        out.append(df)
    return out[0], out[1], {'risk_rate': rate, 'sd_target': sd_target, 'sd_sb': sd_sb}


def fit(spec, train):
    if not spec.is_model:
        return None
    if spec.mixed:
        re_formula = '~gap_pct' if spec.mixed == 'slope' else '1'
        model = smf.mixedlm(spec.formula, train, groups=train['event_key'], re_formula=re_formula)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return model.fit(reml=True, method=['lbfgs'])
    return smf.ols(spec.formula, train).fit()


def n_params(spec, model):
    if model is None:
        return 0
    if spec.mixed:
        k = model.cov_re.shape[0]
        return len(model.fe_params) + k * (k + 1) // 2
    return len(model.params)


def predict(spec, model, df):
    """Adds pred_mark and pred_perf_sb (predicted % vs SB, positive = better)."""
    df = df.copy()
    if model is None:
        pred = np.zeros(len(df))
    else:
        pred = np.asarray(model.predict(df), dtype=float)
        if spec.mixed:
            re = pd.DataFrame(model.random_effects).T
            intercept = df['event_key'].map(re.iloc[:, 0]).fillna(0).to_numpy()
            slope = (df['event_key'].map(re['gap_pct']).fillna(0).to_numpy() * df['gap_pct'].to_numpy()
                     if 'gap_pct' in re else 0)
            pred = pred + intercept + slope
    pred_target = pred * df['sd_target'].to_numpy() if spec.scale else pred
    base = df['sb'] if spec.base == 'sb' else df['median']
    df['pred_mark'] = base * (1 + df['sign'] * pred_target / 100)
    df['pred_perf_sb'] = df['sign'] * (df['pred_mark'] - df['sb']) / df['sb'] * 100
    return df


def score(df):
    """Metrics on predictions converted back to marks. Errors are actual - predicted, in % vs SB."""
    e = df['perf_sb'] - df['pred_perf_sb']
    out = {
        'n': len(df),
        'srmse': float(np.sqrt(np.mean((e / df['sd_sb']) ** 2))),
        'mae_pct': float(e.abs().mean()),
        'bias_pct': float(e.mean()),
        'r2_perf': float(1 - (e**2).sum() / ((df['perf_sb'] - df['perf_sb'].mean()) ** 2).sum()),
    }
    events = []
    for _, g in df.groupby(['year', 'event_key']):
        if len(g) < MIN_ATHLETES_FOR_EVENT_R2:
            continue
        sst = ((g['final_mark'] - g['final_mark'].mean()) ** 2).sum()
        sse = ((g['final_mark'] - g['pred_mark']) ** 2).sum()
        if sst > 0:
            events.append((len(g), sse, sst))
    if events:
        n, sse, sst = map(np.array, zip(*events))
        out['event_r2_mean'] = float(np.mean(1 - sse / sst))
        out['event_r2_pooled'] = float(1 - np.sum((n - 1) * sse / sst) / np.sum(n - 1))
    else:
        out['event_r2_mean'] = out['event_r2_pooled'] = np.nan
    abs_err = (df['pred_mark'] - df['final_mark']).abs()
    out['mae_track_s'] = float(abs_err[df['is_track']].mean())
    out['mae_field_m'] = float(abs_err[~df['is_track']].mean())
    return out


def make_folds():
    return {
        'LOYO': [([y for y in TRAIN_YEARS if y != v], v) for v in TRAIN_YEARS],
        'Forward': [(TRAIN_YEARS[:i], TRAIN_YEARS[i]) for i in range(2, len(TRAIN_YEARS))],
        'LOYO excl. 2021': [([y for y in TRAIN_YEARS[1:] if y != v], v) for v in TRAIN_YEARS[1:]],
    }


def run_cv(specs, data, folds, desc):
    """Fit every spec on every fold. Returns per-fold metrics and out-of-fold predictions."""
    rows, oof = [], []
    total = sum(len(f) for f in folds.values()) * len(specs)
    with tqdm(total=total, desc=desc, unit='fit') as bar:
        for spec in specs:
            for scheme, scheme_folds in folds.items():
                for train_years, val_year in scheme_folds:
                    tr, va, _ = prepare(spec, data[data['year'].isin(train_years)], data[data['year'] == val_year])
                    try:
                        va = predict(spec, fit(spec, tr), va)
                    except (np.linalg.LinAlgError, ValueError) as err:
                        tqdm.write(f'  {spec.name} / {scheme} {val_year}: fit failed ({err})')
                        bar.update()
                        continue
                    rows.append({'model': spec.name, 'scheme': scheme, 'fold': val_year, **score(va)})
                    if scheme == 'LOYO':
                        oof.append(va.assign(model=spec.name))
                    bar.update()
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def choose(per_fold, scheme, candidates, complexity):
    """Lowest mean standardized RMSE; then the simplest model within one fold-SD of it.

    'Within one fold-SD' uses the paired per-fold difference to the best model, since all models
    share the same folds. Complexity = (parameters, mixed, smoothed risk, SD-scaled) — fewer wins.
    """
    wide = (per_fold[(per_fold['scheme'] == scheme) & per_fold['model'].isin(candidates)]
            .pivot(index='fold', columns='model', values='srmse').dropna(axis=1))
    means = wide.mean()
    best = means.idxmin()
    diff = wide.sub(wide[best], axis=0)
    table = pd.DataFrame({'mean_srmse': means, 'diff_vs_best': diff.mean(), 'fold_sd_of_diff': diff.std()})
    table['within_1sd'] = (table['diff_vs_best'] <= table['fold_sd_of_diff']) | (table.index == best)
    table['complexity'] = [complexity[m] for m in table.index]
    table = table.sort_values('mean_srmse')
    pick = min(table.index[table['within_1sd']], key=lambda m: (complexity[m], means[m]))
    return pick, best, table


def year_temperature_check(spec, train):
    """Year-level test of temperature: mean residual per championship (model without temperature) vs temp."""
    tr, _, _ = prepare(spec, train, train)
    model = fit(spec, tr)
    tr = predict(spec, model, tr)
    tr['resid'] = tr['perf_sb'] - tr['pred_perf_sb']
    by_year = tr.groupby('year').agg(temp=('temp', 'first'), mean_resid=('resid', 'mean'), n=('resid', 'size'))

    def slope_ci(df):
        res = stats.linregress(df['temp'], df['mean_resid'])
        t = stats.t.ppf(0.975, len(df) - 2)
        return res.slope * 10, (res.slope - t * res.stderr) * 10, (res.slope + t * res.stderr) * 10, res.pvalue

    return by_year, slope_ci(by_year), slope_ci(by_year.drop(index=2022, errors='ignore'))


def coef_table(spec, train, cluster=True):
    """Coefficients (raw and per-SD) with 95% CIs. OLS SEs are clustered by athlete."""
    tr, _, info = prepare(spec, train, train)
    model = fit(spec, tr)
    z = tr.copy()
    sds = {}
    for col in CONTINUOUS:
        if col in z:
            sds[col] = z[col].std()
            z[col] = (z[col] - z[col].mean()) / sds[col]
    std_model = fit(spec, z)
    if not spec.mixed and cluster:
        groups = pd.factorize(tr['athlete_id'])[0]
        model_c = model.get_robustcov_results('cluster', groups=groups)
        std_c = std_model.get_robustcov_results('cluster', groups=groups)
        names = model.model.exog_names
        params, ci, p = (pd.Series(model_c.params, names), pd.DataFrame(model_c.conf_int(), index=names),
                         pd.Series(model_c.pvalues, names))
        std_params, std_ci = pd.Series(std_c.params, names), pd.DataFrame(std_c.conf_int(), index=names)
        vif_names = names
    else:
        names = list(model.fe_params.index)
        params, ci, p = model.fe_params, model.conf_int().loc[names], model.pvalues.loc[names]
        std_params, std_ci = std_model.fe_params, std_model.conf_int().loc[names]
        vif_names = names
    table = pd.DataFrame({
        'coef': params, 'ci_low': ci.iloc[:, 0], 'ci_high': ci.iloc[:, 1], 'p': p,
        'std_coef': std_params, 'std_ci_low': std_ci.iloc[:, 0], 'std_ci_high': std_ci.iloc[:, 1],
    })
    X = patsy.dmatrix(spec.formula.split('~', 1)[1], tr, return_type='dataframe')
    vif = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != 'Intercept'}
    table['vif'] = pd.Series(vif)
    table['sd_of_feature'] = [sds.get(n, np.nan) for n in table.index]
    return table, model, info


# ── Report helpers ─────────────────────────────────────────────────────────────

def md(df, floatfmt='.3f', index=True):
    fmts = ([''] if index else []) + ['.0f' if pd.api.types.is_integer_dtype(df[c]) else floatfmt
                                      for c in df.columns]
    return df.to_markdown(floatfmt=fmts, index=index, missingval='—')


def fmt_mark(value, is_track):
    if pd.isna(value):
        return ''
    if not is_track:
        return f'{value:.2f}m'
    if value >= 60:
        minutes, seconds = divmod(value, 60)
        return f'{int(minutes)}:{seconds:05.2f}'
    return f'{value:.2f}'


def pretty_term(name):
    for key, label in [('C(event_group)[T.', 'Event group: '), ('C(track_field)[T.', 'Field (vs track)'),
                       ('C(gender)[T.', 'Men (vs women)')]:
        if name.startswith(key):
            level = name[len(key):-1]
            if 'gender' in key or 'track_field' in key:
                return label
            return label + GROUP_LABEL.get(level, level) + ' (vs sprints)'
    if name.startswith('temp_c:C(event_group)[T.'):
        level = name[len('temp_c:C(event_group)[T.'):-1]
        return f'Temperature × {GROUP_LABEL.get(level, level)}'
    if name.startswith('temp_c:C(track_field)[T.'):
        return 'Temperature × field'
    if name.startswith('I('):
        return TERM_LABEL.get(name[2:-6], name[2:-6]) + '²'
    return {'Intercept': 'Intercept', **TERM_LABEL}.get(name, name)


def style():
    plt.rcParams.update({
        'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 10,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
        'axes.titlesize': 11, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK,
    })


def headline(fig, title, subtitle):
    fig.text(0.01, 0.99, title, ha='left', va='top', fontsize=13, fontweight='bold', color=INK)
    fig.text(0.01, 0.935, subtitle, ha='left', va='top', fontsize=10, color=INK_2)


def plot_pred_vs_actual(test_pred, path, spec):
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(15, 4.2), sharey=True)
    lo = np.nanpercentile(test_pred['perf_sb'], 1) - 1
    hi = max(test_pred['perf_sb'].max(), test_pred['pred_perf_sb'].max()) + 1
    for ax, group in zip(axes, GROUPS):
        g = test_pred[test_pred['event_group'] == group]
        ax.axline((0, 0), slope=1, color=INK_2, lw=1, ls='--', zorder=1)
        ax.scatter(g['pred_perf_sb'], g['perf_sb'], s=26, color=DOT, alpha=0.55, edgecolor=SURFACE, lw=0.6, zorder=2)
        s = score(g)
        ax.set_title(f'{GROUP_LABEL[group]}  (n={len(g)})', loc='left')
        ax.text(0.02, 0.03, f"within-event R² {s['event_r2_mean']:.2f}\nmiss {s['mae_pct']:.1f}% (avg)",
                transform=ax.transAxes, fontsize=9, color=INK_2, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc=SURFACE, ec='none', alpha=0.9))
        ax.set_xlabel('Predicted, % vs SB')
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel('Actual, % vs SB (positive = better)')
    headline(fig, f'{TEST_YEAR} holdout: predicted vs actual championship marks',
             f'Model {spec.name}, trained on {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}. Dashed line = perfect prediction. '
             f'Marks below {lo:.0f}% are off the chart.')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def strip(ax, values_by_row, labels):
    for i, vals in enumerate(values_by_row):
        jitter = np.random.default_rng(i).uniform(-0.22, 0.22, len(vals))
        ax.scatter(vals, i + jitter, s=14, color=DOT, alpha=0.35, edgecolor='none')
        ax.plot([np.median(vals)] * 2, [i - 0.32, i + 0.32], color=INK, lw=2)
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(0, color=INK_2, lw=1)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)


def plot_residuals(oof, test_pred, path, spec):
    lim = (-15, 8)

    def resid(df):
        return (df['perf_sb'] - df['pred_perf_sb']).clip(*lim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    strip(ax1, [resid(oof[oof['event_group'] == g]) for g in GROUPS], [GROUP_LABEL[g] for g in GROUPS])
    ax1.set_title('By event group (leave-one-year-out)', loc='left')
    years = TRAIN_YEARS + [TEST_YEAR]
    strip(ax2, [resid(test_pred) if y == TEST_YEAR else resid(oof[oof['year'] == y]) for y in years],
          [f'{y} (holdout)' if y == TEST_YEAR else str(y) for y in years])
    ax2.set_title(f'By year ({TEST_YEAR} = holdout)', loc='left')
    for ax in (ax1, ax2):
        ax.set_xlim(lim[0] - 0.5, lim[1] + 0.5)
        ax.set_xlabel('Actual − predicted, % vs SB (positive = better than predicted)')
    headline(fig, 'Where the model misses', f'Model {spec.name}. Dots = athletes, black bar = median. '
             f'Misses beyond {lim[0]}% / +{lim[1]}% are pinned to the edge.')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def binned(x, resid, edges):
    cats = pd.cut(x, edges, include_lowest=True)
    g = resid.groupby(cats, observed=True)
    out = g.agg(['mean', 'sem', 'size'])
    out['mid'] = x.groupby(cats, observed=True).median()
    return out


def plot_residuals_vs_features(oof, path, spec):
    df = oof.copy()
    df['resid'] = df['perf_sb'] - df['pred_perf_sb']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    lim = (-12, 8)

    ax = axes[0]
    jitter = np.random.default_rng(0).uniform(-0.6, 0.6, len(df))
    ax.scatter(df['temp'] + jitter, df['resid'].clip(*lim), s=10, color=DOT, alpha=0.25, edgecolor='none')
    yr = df.groupby('year').agg(temp=('temp', 'first'), m=('resid', 'mean'), se=('resid', 'sem'))
    ax.errorbar(yr['temp'], yr['m'], yerr=1.96 * yr['se'], fmt='o', color=INK, ms=6, capsize=3, lw=1.5)
    last_temp = None
    for y, r in yr.sort_values('temp').iterrows():
        crowded = last_temp is not None and r['temp'] - last_temp < 2
        ax.annotate(str(y), (r['temp'], r['m']), xytext=(8, -14 if crowded else 6), textcoords='offset points',
                    fontsize=9, color=INK_2, bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none', alpha=0.9))
        last_temp = r['temp']
    ax.set_xlabel('Championship temperature (°F, mean daily high)')
    ax.set_title('Temperature: one value per championship', loc='left')

    for ax, col, edges, title, xlabel in [
        (axes[1], 'gap_pct', np.unique(np.quantile(df['gap_pct'], np.linspace(0, 1, 11))),
         'Consistency gap', 'Season median vs SB (% worse)'),
        (axes[2], 'risk_raw', [0, 1e-9, 0.05, 0.10, 0.20, 1.0], 'risk_score', 'Share of bad regular-season races'),
    ]:
        ax.scatter(df[col], df['resid'].clip(*lim), s=10, color=DOT, alpha=0.25, edgecolor='none')
        b = binned(df[col], df['resid'], edges)
        ax.errorbar(b['mid'], b['mean'], yerr=1.96 * b['sem'], fmt='-o', color=INK, ms=5, capsize=3, lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left')
    axes[1].set_xscale('symlog', linthresh=1)
    axes[1].set_xticks([0, 1, 2, 5, 10, 20])
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    for ax in axes:
        ax.axhline(0, color=INK_2, lw=1)
        ax.set_ylim(*lim)
    axes[0].set_ylabel('Actual − predicted, % vs SB')
    headline(fig, 'Residuals vs temperature, consistency gap and risk_score',
             f'Model {spec.name}, leave-one-year-out predictions. Black = mean residual (95% CI) per year or bin.')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coefficients(table, path, spec, units):
    t = table.drop(index='Intercept', errors='ignore').copy()
    t = t[~t.index.str.contains('Var')]
    t['label'] = [pretty_term(n) for n in t.index]
    t['continuous'] = t['sd_of_feature'].notna()
    t = t.sort_values(['continuous', 'std_coef'])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.hlines(y, t['std_ci_low'], t['std_ci_high'], color=DOT, lw=2)
    ax.scatter(t['std_coef'], y, s=40, color=DOT, edgecolor=SURFACE, lw=1, zorder=3)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(y, t['label'])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'Effect on championship result ({units}); continuous features per +1 SD')
    headline(fig, 'What moves the prediction', f'Model {spec.name}, fit on {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}. '
             'Lines = 95% CI (clustered by athlete). Right = better.')
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)
    style()
    print('Loading data')
    results, season, champ_info, champ_temp = load_inputs()
    risk, cutoffs = build_risk(season, TRAIN_YEARS)
    feats = build_season_features(season)
    data_all, flow = build_model_data(results, feats, risk, champ_temp)

    usable = data_all[~data_all['no_mark']]
    train = usable[usable['year'].isin(TRAIN_YEARS) & ~usable['not_a_race']].copy()
    test = usable[(usable['year'] == TEST_YEAR) & ~usable['not_a_race']].copy()
    test_unfiltered = usable[usable['year'] == TEST_YEAR].copy()
    print(f'{len(train)} training rows ({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}), {len(test)} test rows ({TEST_YEAR})')

    # ── Step 0: inspection ──
    csv_risk = (season.groupby(['athlete_id', 'year'])['risk_score'].first().rename('risk_csv').reset_index())
    risk_cmp = data_all.drop_duplicates(['athlete_id', 'year']).merge(csv_risk, on=['athlete_id', 'year'], how='left')
    risk_corr = risk_cmp[['risk_raw', 'risk_csv']].corr().iloc[0, 1]
    risk_same = np.isclose(risk_cmp['risk_raw'], risk_cmp['risk_csv'], atol=1e-4).mean()

    leakage = pd.DataFrame({
        'Championship starts': champ_info['start_date'],
        'Latest meet in SB/median features': data_all.groupby('year')['last_meet'].max().dt.date.astype(str),
        'Latest race in risk_score': data_all.groupby('year')['last_race'].max().dt.date.astype(str),
    })
    assert (pd.to_datetime(leakage['Latest meet in SB/median features']) < pd.to_datetime(leakage['Championship starts'])).all()
    assert (pd.to_datetime(leakage['Latest race in risk_score']) < pd.to_datetime(leakage['Championship starts'])).all()

    modeled = pd.concat([train, test])
    counts_group = modeled.pivot_table(index='year', columns='event_group', values='athlete_id', aggfunc='size',
                                       observed=True).rename(columns=GROUP_LABEL)
    counts_gender = modeled.pivot_table(index='year', columns='gender', values='athlete_id', aggfunc='size', observed=True)
    counts_gender.columns = counts_gender.columns.astype(str)
    counts = counts_group.join(counts_gender).assign(Total=modeled.groupby('year').size())
    counts_event = modeled.pivot_table(index='event_key', columns='year', values='athlete_id', aggfunc='size').fillna(0).astype(int)

    temp_by_year = pd.DataFrame({
        'Main-meet dates': champ_info['meet_date'], 'City': champ_info['city'],
        'Distinct temp values': modeled.groupby('year')['temp'].nunique(),
        'temp (°F)': modeled.groupby('year')['temp'].first(),
    })
    seasons_model = modeled.drop_duplicates(['athlete_id', 'year'])
    risk_by_year = seasons_model.groupby('year').agg(
        athletes=('risk_raw', 'size'), mean=('risk_raw', 'mean'), median=('risk_raw', 'median'),
        p90=('risk_raw', lambda s: s.quantile(0.9)), share_zero=('risk_raw', lambda s: (s == 0).mean()),
        median_starts=('n_starts', 'median'))
    years_per_athlete = modeled.groupby('athlete_id')['year'].nunique()
    multi_year = years_per_athlete.value_counts().sort_index()
    test_ids = set(test['athlete_id'])
    test_seen = len(test_ids & set(train['athlete_id'])) / len(test_ids)

    # risk_score vs championship no-marks (these rows have no mark, so the mark model can't use them)
    entries = data_all[data_all['year'].isin(TRAIN_YEARS)].copy()
    entries['risk_band'] = pd.cut(entries['risk_raw'], [-0.01, 0, 0.1, 1.0], labels=['0', '0–10%', '>10%'])
    nomark_by_risk = entries.groupby('risk_band', observed=True).agg(
        entries=('no_mark', 'size'), no_mark_rate=('no_mark', 'mean'))
    nm_fit = smf.logit('no_mark ~ risk_raw + C(track_field)', entries.assign(no_mark=entries['no_mark'].astype(int))).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(entries['athlete_id'])[0]})
    nm_or = np.exp(nm_fit.params['risk_raw'] * 0.1)
    nm_ci = np.exp(nm_fit.conf_int().loc['risk_raw'] * 0.1)

    # ── Model specs ──
    other_base = 'median' if BASE_METRIC == 'sb' else 'sb'
    primary = [make_spec(s, BASE_METRIC, False) for s, _, _ in FEATURE_STEPS]
    primary += [make_spec(s, BASE_METRIC, True) for s, _, t in FEATURE_STEPS if t]  # no SD-scaled baselines
    primary = [s for s in primary if not (s.scale and s.step == 'B')]
    variants = []
    for scale in (False, True):
        for step in ('M6', 'M6x'):
            variants.append(make_spec(step, BASE_METRIC, scale, track_field=True))
        for k in RISK_SMOOTHING_K:
            variants.append(make_spec('M6', BASE_METRIC, scale, risk_k=k))
    comparison = [make_spec(s, other_base, False) for s, _, _ in FEATURE_STEPS]
    comparison += [make_spec(s, other_base, True) for s, _, t in FEATURE_STEPS if t and t != ()]
    folds = make_folds()

    print('Cross-validating linear models')
    per_fold, oof = run_cv(primary + variants + comparison, train, folds, 'Linear models')

    # Mixed-effects version of the best linear model (by mean LOYO error)
    linear_candidates = [s.name for s in primary + variants if s.step not in ('A', 'B')]
    loyo = per_fold[per_fold['scheme'] == 'LOYO']
    best_linear_name = loyo[loyo['model'].isin(linear_candidates)].groupby('model')['srmse'].mean().idxmin()
    best_linear = next(s for s in primary + variants if s.name == best_linear_name)
    mixed = [tag(best_linear, mixed='intercept'), tag(best_linear, mixed='slope')]
    print(f'Best linear model: {best_linear.name}; fitting mixed-effects versions')
    pf, oo = run_cv(mixed, train, folds, 'Mixed models')
    per_fold, oof = pd.concat([per_fold, pf]), pd.concat([oof, oo])

    all_specs = {s.name: s for s in primary + variants + comparison + mixed}

    def complexity_of(specs):
        out = {}
        for s in specs:
            tr, _, _ = prepare(s, train, train)
            out[s.name] = (n_params(s, fit(s, tr)), bool(s.mixed), s.risk_k is not None, s.scale)
        return out

    complexity = complexity_of(all_specs.values())
    candidates = [s.name for s in primary + variants + mixed if s.step not in ('A', 'B')]
    pick0, best0, _ = choose(per_fold, 'LOYO', candidates, complexity)
    chosen0 = all_specs[pick0]

    # Nonlinearity check: squared terms for the continuous features in the chosen model
    squares = [v for v in ('gap_pct', 'risk') if v in (chosen0.terms or ())]
    nonlinear = [tag(chosen0, squares=(v,)) for v in squares]
    if nonlinear:
        print(f'Chosen so far: {chosen0.name}; testing squared terms')
        pf, oo = run_cv(nonlinear, train, folds, 'Squared terms')
        per_fold, oof = pd.concat([per_fold, pf]), pd.concat([oof, oo])
        all_specs.update({s.name: s for s in nonlinear})
        complexity.update(complexity_of(nonlinear))
        candidates += [s.name for s in nonlinear]

    pick_loyo, best_loyo, choice_loyo = choose(per_fold, 'LOYO', candidates, complexity)
    pick_fc, best_fc, choice_fc = choose(per_fold, 'Forward', candidates, complexity)
    pick_no21, best_no21, _ = choose(per_fold, 'LOYO excl. 2021', candidates, complexity)

    # Forward-chaining check: is the LOYO pick clearly worse when trained only on the past?
    shared = [y for _, y in folds['Forward']]
    pl = per_fold[(per_fold['model'] == pick_loyo) & per_fold['fold'].isin(shared)].pivot(
        index='fold', columns='scheme', values='srmse')
    fc_gap = (pl['Forward'] / pl['LOYO'] - 1) * 100
    fc_clearly_worse = bool((fc_gap > 2).all())
    chosen_name = pick_fc if (pick_fc != pick_loyo and fc_clearly_worse) else pick_loyo
    chosen = all_specs[chosen_name]
    print(f'LOYO pick: {pick_loyo} | forward-chaining pick: {pick_fc} | chosen: {chosen.name}')

    # ── Influence: drop-one on the full main-effects model ──
    full = make_spec('M6', BASE_METRIC, chosen.scale)
    drop_terms = {TERM_LABEL[t]: t for t in full.terms}
    drop_specs = []
    for label, term in drop_terms.items():
        s = replace(full, terms=tuple(t for t in full.terms if t != term), name=f'M6 − {label}', step='drop')
        drop_specs.append(s)
    pf, _ = run_cv(drop_specs, train, {'LOYO': folds['LOYO']}, 'Drop-one')
    full_loyo = per_fold[(per_fold['scheme'] == 'LOYO') & (per_fold['model'] == full.name)].set_index('fold')['srmse']
    drop_rows = []
    for s in drop_specs:
        d = pf[pf['model'] == s.name].set_index('fold')['srmse'] - full_loyo
        drop_rows.append({'Feature dropped': s.name.split(' − ')[1], 'LOYO sRMSE increase': d.mean(),
                          'fold SD': d.std(), 'Folds where dropping hurts': f'{(d > 0).sum()}/{len(d)}'})
    drop_table = pd.DataFrame(drop_rows).sort_values('LOYO sRMSE increase', ascending=False)

    full_coefs, _, _ = coef_table(full, train)
    chosen_coefs, chosen_model, chosen_info = coef_table(chosen, train)
    units = 'event SDs of % vs SB' if chosen.scale else 'percentage points of % vs SB'

    # Risk: raw vs smoothed coefficient (in the full model, at the chosen scaling)
    risk_rows = []
    for k in [None] + RISK_SMOOTHING_K:
        s = replace(full, risk_k=k)
        t, _, _ = coef_table(s, train)
        r = t.loc['risk']
        name = 'raw' if k is None else f'smoothed k={k}'
        m = per_fold[(per_fold['scheme'] == 'LOYO') & (per_fold['model'] == retag(s).name)]['srmse'].mean()
        risk_rows.append({'risk_score version': name, 'coef per +0.10': r['coef'] / 10,
                          'CI low': r['ci_low'] / 10, 'CI high': r['ci_high'] / 10, 'p': r['p'],
                          'per +1 SD': r['std_coef'], 'per-SD CI low': r['std_ci_low'],
                          'per-SD CI high': r['std_ci_high'], 'LOYO sRMSE (M6)': m})
    risk_table = pd.DataFrame(risk_rows)

    # Temperature: coefficient, year-level check, and bias in the hot year
    temp_naive = smf.ols(full.formula, prepare(full, train, train)[0]).fit()
    temp_naive_ci = temp_naive.conf_int().loc['temp_c'] * 10
    no_temp = make_spec('M5', BASE_METRIC, chosen.scale)
    by_year, (slope10, lo10, hi10, p_year), (slope10_no22, lo10_no22, hi10_no22, _) = year_temperature_check(no_temp, train)

    # ── Final: refit on all training years, evaluate once on the holdout ──
    final_specs = {'Baseline A': make_spec('A', BASE_METRIC), 'Baseline B': make_spec('B', BASE_METRIC),
                   'Chosen': chosen}
    holdout, holdout_preds = [], {}
    for label, spec in tqdm(final_specs.items(), desc=f'Final fit + {TEST_YEAR} test', unit='model'):
        tr, te, info = prepare(spec, train, test)
        model = fit(spec, tr)
        _, te_all, _ = prepare(spec, train, test_unfiltered)
        holdout_preds[label] = predict(spec, model, te)
        holdout.append({'model': f'{label}: {spec.name}', 'test set': 'outlier filter on (default)',
                        **score(holdout_preds[label])})
        holdout.append({'model': f'{label}: {spec.name}', 'test set': 'no filter',
                        **score(predict(spec, model, te_all))})
        if label == 'Chosen':
            final_model, final_info = model, info
            pred_unfiltered = predict(spec, model, te_all)
    holdout = pd.DataFrame(holdout)
    test_pred = holdout_preds['Chosen']

    # Breakdowns by event group / track-field (LOYO out-of-fold and holdout)
    def breakdown(frames, by):
        rows = []
        for label, df in frames.items():
            for key, g in df.groupby(by, observed=True):
                s = score(g)
                rows.append({'group': GROUP_LABEL.get(key, key), 'model': label, 'n': s['n'], 'sRMSE': s['srmse'],
                             'MAE %': s['mae_pct'], 'MAE (s or m)': (g['pred_mark'] - g['final_mark']).abs().mean(),
                             'bias %': s['bias_pct'], 'within-event R²': s['event_r2_mean']})
        out = pd.DataFrame(rows).pivot(index='group', columns='model')
        return out

    oof_frames = {'A': oof[oof['model'] == make_spec('A', BASE_METRIC).name],
                  'B': oof[oof['model'] == make_spec('B', BASE_METRIC).name],
                  'Chosen': oof[oof['model'] == chosen.name]}
    test_frames = {'A': holdout_preds['Baseline A'], 'B': holdout_preds['Baseline B'], 'Chosen': test_pred}
    group_order = [GROUP_LABEL[g] for g in GROUPS]
    bd_loyo = breakdown(oof_frames, 'event_group').reindex(group_order)
    bd_test = breakdown(test_frames, 'event_group').reindex(group_order)
    tf_loyo = breakdown(oof_frames, 'track_field')
    tf_test = breakdown(test_frames, 'track_field')

    def group_table(bd):
        t = pd.DataFrame({'n': bd[('n', 'Chosen')],
                          'sRMSE A': bd[('sRMSE', 'A')], 'sRMSE B': bd[('sRMSE', 'B')],
                          'sRMSE chosen': bd[('sRMSE', 'Chosen')],
                          'MAE % chosen': bd[('MAE %', 'Chosen')], 'MAE s/m chosen': bd[('MAE (s or m)', 'Chosen')],
                          'bias % chosen': bd[('bias %', 'Chosen')],
                          'R² A': bd[('within-event R²', 'A')], 'R² chosen': bd[('within-event R²', 'Chosen')]})
        t['worse than A?'] = np.where(t['sRMSE chosen'] > t['sRMSE A'], 'yes', '')
        return t

    # ── Save predictions and model ──
    out = pred_unfiltered.copy()
    out['predicted'] = [fmt_mark(v, t) for v, t in zip(out['pred_mark'], out['is_track'])]
    out['actual'] = [fmt_mark(v, t) for v, t in zip(out['final_mark'], out['is_track'])]
    out['season_best'] = [fmt_mark(v, t) for v, t in zip(out['sb'], out['is_track'])]
    out['error'] = out['final_mark'] - out['pred_mark']
    out['beat_prediction_by_pct'] = out['perf_sb'] - out['pred_perf_sb']
    out['event_group'] = out['event_group'].map(GROUP_LABEL)
    pred_cols = ['athlete_id', 'athlete_name', 'team', 'gender', 'event', 'event_group', 'season_best', 'sb', 'median',
                 'predicted', 'pred_mark', 'actual', 'final_mark', 'error', 'pred_perf_sb', 'perf_sb',
                 'beat_prediction_by_pct', 'risk_raw', 'n_starts', 'temp', 'not_a_race']
    out = out[pred_cols].rename(columns={
        'sb': 'sb_value', 'median': 'median_value', 'pred_mark': 'predicted_value', 'final_mark': 'actual_value',
        'pred_perf_sb': 'predicted_pct_vs_sb', 'perf_sb': 'actual_pct_vs_sb', 'risk_raw': 'risk_score',
        'not_a_race': f'dropped_by_{MAX_OFF_SEASON_BEST:.0f}pct_filter'})
    out = out.sort_values(['gender', 'event', 'actual_value'])
    out.to_csv(PREDICTIONS_CSV, index=False, float_format='%.4f')

    save_model(MODEL_PKL, chosen, final_model, final_info, cutoffs)

    per_fold.to_csv(TABLE_DIR / 'cv_per_fold_all_models.csv', index=False)
    holdout.to_csv(TABLE_DIR / f'holdout_{TEST_YEAR}.csv', index=False)
    chosen_coefs.to_csv(TABLE_DIR / 'chosen_model_coefficients.csv')

    # ── Figures ──
    print('Drawing figures')
    chosen_oof = oof[oof['model'] == chosen.name]
    plot_pred_vs_actual(test_pred, FIG_DIR / 'pred_vs_actual_2026.png', chosen)
    plot_residuals(chosen_oof, test_pred, FIG_DIR / 'residuals_by_group_and_year.png', chosen)
    plot_residuals_vs_features(chosen_oof, FIG_DIR / 'residuals_vs_features.png', chosen)
    plot_coefficients(chosen_coefs, FIG_DIR / 'coefficients.png', chosen, units)

    # ── Report ──
    print('Writing results.md')
    write_report(**locals())
    print(f'Done. Chosen model: {chosen.name}')


def write_report(**v):
    """results.md, built from the computed numbers."""
    per_fold, chosen, train, test = v['per_fold'], v['chosen'], v['train'], v['test']
    holdout, all_specs = v['holdout'], v['all_specs']
    L = []
    w = L.append

    def fold_table(scheme, names):
        sub = per_fold[(per_fold['scheme'] == scheme) & per_fold['model'].isin(names)]
        wide = sub.pivot(index='model', columns='fold', values='srmse')
        wide.columns = [str(c) for c in wide.columns]
        wide['mean'] = wide.mean(axis=1)
        wide['fold SD'] = wide.drop(columns='mean').std(axis=1)
        other = sub.groupby('model')[['event_r2_mean', 'event_r2_pooled', 'r2_perf', 'mae_pct',
                                      'mae_track_s', 'mae_field_m', 'bias_pct']].mean()
        other.columns = ['within-event R² (mean)', 'within-event R² (pooled)', 'R² on % vs SB', 'MAE %',
                         'MAE track (s)', 'MAE field (m)', 'bias %']
        return wide.join(other).reindex([n for n in names if n in wide.index])

    def label_of(name):
        s = all_specs.get(name)
        if not s:
            return name
        label = s.label.replace('SB', 'median') if s.base == 'median' and s.step in ('A', 'B') else s.label
        return f'{name}: {label}'

    ladder_raw = [make_spec(s, BASE_METRIC, False).name for s, _, _ in FEATURE_STEPS]
    ladder_sd = [make_spec(s, BASE_METRIC, True).name for s, _, t in FEATURE_STEPS if t]
    variant_names = [s.name for s in v['variants']] + [s.name for s in v['mixed']] + [s.name for s in v['nonlinear']]
    main_rows = ladder_raw + [n for n in ladder_sd if n not in ladder_raw] + variant_names

    loyo_tab = fold_table('LOYO', main_rows)
    fc_tab = fold_table('Forward', main_rows)
    no21_tab = fold_table('LOYO excl. 2021', main_rows)
    for t in (loyo_tab, fc_tab, no21_tab):
        t.index = [label_of(n) for n in t.index]

    cmp_names = [make_spec(s, b, sc).name for b in ('sb', 'median') for sc in (False, True)
                 for s, _, t in FEATURE_STEPS if not (sc and t in (None, ()))]
    cmp = fold_table('LOYO', cmp_names)[['mean', 'within-event R² (pooled)', 'MAE %', 'bias %']]
    cmp.index = [label_of(n) for n in cmp.index]

    ho = holdout.set_index(['model', 'test set'])[['n', 'srmse', 'event_r2_mean', 'event_r2_pooled', 'r2_perf',
                                                   'mae_pct', 'mae_track_s', 'mae_field_m', 'bias_pct']]
    ho.columns = ['n', 'sRMSE', 'within-event R² (mean)', 'within-event R² (pooled)', 'R² on % vs SB', 'MAE %',
                  'MAE track (s)', 'MAE field (m)', 'bias %']
    ho_main = ho.xs('outlier filter on (default)', level='test set')
    ho_nofilter = ho.xs('no filter', level='test set')
    filter_matters = not np.allclose(ho_main.to_numpy(dtype=float), ho_nofilter.to_numpy(dtype=float))
    ho_show = ho if filter_matters else ho_main
    chosen_ho = ho_main.loc[f'Chosen: {chosen.name}']
    a_ho, b_ho = ho_main.iloc[0], ho_main.iloc[1]

    loyo_mean = loyo_tab.loc[label_of(chosen.name)]
    a_name, b_name = make_spec('A', BASE_METRIC).name, make_spec('B', BASE_METRIC).name
    loyo_a, loyo_b = loyo_tab.loc[label_of(a_name)], loyo_tab.loc[label_of(b_name)]

    # Bias by held-out year
    bias_models = [a_name, b_name] + [make_spec(s, BASE_METRIC, chosen.scale).name for s in ('M5', 'M6', 'M6x')]
    if chosen.name not in bias_models:
        bias_models.append(chosen.name)
    bias = (per_fold[(per_fold['scheme'] == 'LOYO') & per_fold['model'].isin(bias_models)]
            .pivot(index='fold', columns='model', values='bias_pct')[bias_models])
    test_bias = {a_name: a_ho['bias %'], b_name: b_ho['bias %'], chosen.name: chosen_ho['bias %']}
    bias.loc[f'{TEST_YEAR} (holdout)'] = [test_bias.get(m, np.nan) for m in bias_models]
    bias.index = [str(i) for i in bias.index]
    bias.columns = [f'{m}' for m in bias.columns]
    m5, m6, m6x = [make_spec(s, BASE_METRIC, chosen.scale).name for s in ('M5', 'M6', 'M6x')]
    b22 = bias.loc['2022']

    # Selection details
    choice = v['choice_loyo'].copy()
    choice.index = [label_of(n) for n in choice.index]
    choice['complexity'] = [c[0] for c in choice['complexity']]
    choice = choice.rename(columns={'complexity': 'parameters'})

    coefs = v['chosen_coefs'].copy()
    coefs.index = [pretty_term(n) for n in coefs.index]
    coefs = coefs[['coef', 'ci_low', 'ci_high', 'p', 'std_coef', 'std_ci_low', 'std_ci_high', 'vif']]
    coefs.columns = ['coef', '95% CI low', '95% CI high', 'p', 'per-SD coef', 'per-SD CI low', 'per-SD CI high', 'VIF']

    full_coefs = v['full_coefs']
    cont = full_coefs.loc[[c for c in CONTINUOUS if c in full_coefs.index]]
    rank_std = pd.DataFrame({'Feature': [pretty_term(n) for n in cont.index], 'per-SD coef': cont['std_coef'],
                             '95% CI': [f'[{a:.3f}, {b:.3f}]' for a, b in zip(cont['std_ci_low'], cont['std_ci_high'])],
                             'p': cont['p']})
    rank_std = rank_std.reindex(rank_std['per-SD coef'].abs().sort_values(ascending=False).index)

    unit_word = 'event SDs' if chosen.scale else 'percentage points'
    risk_t = v['risk_table']
    raw_risk = risk_t.iloc[0]
    risk_sig = raw_risk['p'] < 0.05
    # Compare per-SD: smoothing squeezes risk toward the average, which inflates the raw-unit coefficient
    smooth_changes = any(not (raw_risk['per-SD CI low'] <= r['per +1 SD'] <= raw_risk['per-SD CI high'])
                         for _, r in risk_t.iloc[1:].iterrows())

    # Temperature summaries
    tc = full_coefs.loc['temp_c']
    temp_ci_inc0 = v['lo10'] <= 0 <= v['hi10']
    slope_no22 = v['slope10_no22']

    # Base comparison
    sb_mean = cmp.loc[label_of(make_spec('M6', 'sb').name), 'mean']
    med_mean = cmp.loc[label_of(make_spec('M6', 'median').name), 'mean']
    cmp_per_fold = per_fold[(per_fold['scheme'] == 'LOYO')
                            & per_fold['model'].isin([make_spec('M6', 'sb').name, make_spec('M6', 'median').name])]
    cmp_wide = cmp_per_fold.pivot(index='fold', columns='model', values='srmse')
    base_diff = cmp_wide[make_spec('M6', 'median').name] - cmp_wide[make_spec('M6', 'sb').name]

    # Transfer vs the 2021->2022 single-feature lines
    r2_2022 = per_fold[(per_fold['scheme'] == 'LOYO') & (per_fold['model'] == chosen.name)
                       & (per_fold['fold'] == 2022)]['event_r2_pooled'].iloc[0]

    # Coach numbers
    train_perf = train.groupby('event_group', observed=True)['perf_sb'].mean()
    drop_all = train['perf_sb'].mean()
    tr_c, _, _ = prepare(chosen, train, test)
    coach_model = fit(chosen, tr_c)
    base_pred = predict(chosen, coach_model, tr_c)['pred_perf_sb']
    effects = {}
    if 'temp_c' in (chosen.terms or ()):
        hot = tr_c.copy()
        hot['temp_c'] += 10
        effects['temp'] = (predict(chosen, coach_model, hot)['pred_perf_sb'] - base_pred).mean()
    if 'risk' in (chosen.terms or ()):
        risky = tr_c.copy()
        risky['risk'] += 0.10
        effects['risk'] = (predict(chosen, coach_model, risky)['pred_perf_sb'] - base_pred).mean()

    examples = []
    for key, label in [('men 100', "Men's 100m"), ('women 400', "Women's 400m"), ('women 1500', "Women's 1500m"),
                       ('men 110H', "Men's 110m hurdles"), ('men LJ', "Men's long jump"),
                       ('women SP', "Women's shot put")]:
        rows = train[train['event_key'] == key]
        if rows.empty:
            continue
        sb = rows['sb'].median()
        p = rows['perf_sb'].mean()
        is_track = bool(rows['is_track'].iloc[0])
        exp = sb * (1 + rows['sign'].iloc[0] * p / 100)
        examples.append({'Event': label, 'Typical SB': fmt_mark(sb, is_track), 'Average result vs SB': f'{p:+.1f}%',
                         'Expected mark': fmt_mark(exp, is_track)})
    examples = pd.DataFrame(examples)

    # ── Write ──
    w(f'# Big West Championships mark prediction: train {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}, test {TEST_YEAR}\n')
    w(f'Generated by `modeling/model_bigwest.py` (base metric: **{BASE_METRIC.upper()}**). '
      f'Rerun the script to regenerate every number, table and figure.\n')

    w('## Bottom line\n')
    w(f'- **Chosen model: {chosen.name}** ({describe(chosen)}). On the {TEST_YEAR} holdout it explains '
      f'**{chosen_ho["within-event R² (pooled)"] * 100:.0f}%** of the differences between athletes in the same event '
      f'(pooled within-event R²), vs {a_ho["within-event R² (pooled)"] * 100:.0f}% for SB as-is and '
      f'{b_ho["within-event R² (pooled)"] * 100:.0f}% for SB minus the average drop-off. Typical miss: '
      f'{chosen_ho["MAE %"]:.1f}% of the mark. Expect closer to {loyo_mean["within-event R² (pooled)"] * 100:.0f}% in a '
      'typical year (the LOYO average).')
    w(f'- **Most of that R² is the SB itself.** Of the variation in how far athletes finish above or below their SB, the '
      f'model explains {loyo_mean["R² on % vs SB"] * 100:.0f}% (LOYO) and {chosen_ho["R² on % vs SB"] * 100:.0f}% '
      f'({TEST_YEAR}). The strongest signal is consistency: athletes whose season median sits far behind their SB fall '
      'further short of it.')
    gain_total = loyo_a['mean'] - loyo_mean['mean']
    gain_b = loyo_a['mean'] - loyo_b['mean']
    w(f'- **Where the gain over "SB as-is" comes from (LOYO sRMSE {loyo_a["mean"]:.3f} → {loyo_mean["mean"]:.3f}):** '
      f'{gain_b / gain_total * 100:.0f}% from applying the average drop-off (Baseline B, {loyo_b["mean"]:.3f}) and '
      f'{(1 - gain_b / gain_total) * 100:.0f}% from the athlete/event features on top.')
    w(f'- **risk_score:** {"significant" if risk_sig else "no significant"} effect '
      f'({raw_risk["coef per +0.10"]:+.2f} {unit_word} per +0.10, 95% CI {raw_risk["CI low"]:+.2f} to {raw_risk["CI high"]:+.2f}).')
    w(f'- **Temperature:** {v["slope10"]:+.2f}% per +10°F at the year level (95% CI {v["lo10"]:+.2f} to {v["hi10"]:+.2f}), '
      f'from only {len(v["by_year"])} championships — {"not distinguishable from zero" if temp_ci_inc0 else "a real but fragile signal"}'
      + (f', and it comes entirely from 2022 (without 2022: {slope_no22:+.2f}%)' if abs(slope_no22) < abs(v['slope10']) / 3 else '')
      + '. Adding it does not lower out-of-sample error.\n')

    w('## Assumptions\n')
    w(f'- **Data:** `data/processed/bigwest_2019_2026_results.csv` (championship marks, one row per athlete-event) and '
      f'`regular_season.csv`. Relays and multi-events are excluded (no individual season best on the same scale).')
    w('- **Column names:** the prompt\'s `temperature` is the `temp` column: the mean daily high (°F, Open-Meteo, '
      'city-level) over the championship\'s main-meet days. 2026 was corrected to **May 15–16** (TFRRS lists May 8–16 '
      'because it folds in the multi-events weekend; Big West athletes raced at Titan Tune Up on May 8). That moved 2026 '
      'from 69.9°F to 68.1°F in the CSV. 2021 uses TFRRS\'s May 12–15 (dropping May 12 would give 69.5°F instead of 70.1°F).')
    w('- **risk_score:** rebuilt inside the script, not read from the CSV column: same rules as '
      '`notebooks/risk_score.ipynb` (DQ/FS, DNF, FOUL/NH/NM, and outlier marks beyond Q3 + 3·IQR of % off the athlete\'s '
      'SB for the event — not a fixed 50% rule), but using only races before that year\'s Big West and outlier cutoffs '
      f'from {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} only. Correlation with the CSV column: {v["risk_corr"]:.3f} '
      f'({v["risk_same"] * 100:.0f}% of athlete-seasons identical to 4 decimals). It covers indoor and outdoor races.')
    w('- **SB and median:** outdoor marks before the championship (one best mark per meet), all winds, as in the '
      'earlier single-feature work.')
    w('- **SB level** (added in M3): the athlete\'s SB as a z-score among that year\'s championship entrants in the '
      'event (positive = better). It lets the model learn that the best seeds drop off differently from the rest — the '
      'relative-scale version of a fitted slope. Entry lists are known before the meet, so it isn\'t leakage.')
    w('- **Consistency gap** = how far the season median is behind the SB, in % (≥ 0; bigger = less consistent).')
    w(f'- **Target:** `perf_pct` = championship mark vs SB in %, positive = better (track (SB − mark)/SB, field '
      f'(mark − SB)/SB). Marks more than {MAX_OFF_SEASON_BEST:.0f}% worse than SB are dropped from training.')
    w('- **Errors** are always *actual − predicted* in % vs SB, so a negative bias means athletes did worse than predicted. '
      '**sRMSE** (selection metric) = root-mean-square error after dividing each miss by its event\'s training-year SD '
      'of % vs SB, so a 100m and a javelin count equally.')
    w(f'- **Within-event R²** is computed on marks (predictions converted back), per event-year with ≥ '
      f'{MIN_ATHLETES_FOR_EVENT_R2} athletes. "Mean" averages events equally; "pooled" weights by athletes and matches the '
      'earlier 2021→2022 figure (~0.46).\n')

    w('## Step 0: data checks\n')
    w('**Rows from championship entry to model** (individual track and field events):\n')
    w(md(v['flow'], index=True))
    w('\n**Modeled rows by year, event group and gender:**\n')
    w(md(v['counts'].fillna(0).astype(int)))
    w(f'\nPer-event counts are in `tables/rows_by_event.csv`. Smallest event-years: '
      + ', '.join(f'{k} {y} ({n})' for (k, y), n in v['counts_event'].stack().loc[lambda s: s > 0].nsmallest(4).items()) + '.\n')
    v['counts_event'].to_csv(TABLE_DIR / 'rows_by_event.csv')
    w('**Temperature is one value per championship.** It does not vary by day, session or event within a year, so its '
      f'effect rests on {len(TRAIN_YEARS)} training data points (4 inside any CV fold):\n')
    w(md(v['temp_by_year'], floatfmt='.1f'))
    w('\n**risk_score by year** (one row per athlete-season in the model):\n')
    w(md(v['risk_by_year']))
    w('\n**No leakage:** every feature uses only races that started before that year\'s Big West:\n')
    w(md(v['leakage']))
    w(f'\n**Repeat athletes:** {int(v["multi_year"].loc[v["multi_year"].index > 1].sum())} of '
      f'{int(v["multi_year"].sum())} athletes appear in more than one year '
      f'({", ".join(f"{n} in {k} years" for k, n in v["multi_year"].items() if k > 1)}). '
      f'{v["test_seen"] * 100:.0f}% of {TEST_YEAR} athletes also appear in training years, so the holdout is new '
      'races, not entirely new people. Coefficient CIs are clustered by athlete for this reason.\n')

    w('## Model comparison\n')
    w('Ladder: each row adds to the one above. The selection metric is **sRMSE** (lower = better). The year columns are the '
      'held-out year.\n')
    w('### Leave-one-year-out (train on 4 years, validate on the 5th)\n')
    w(md(loyo_tab))
    w('\n### Forward-chaining (train on earlier years only)\n')
    w(md(fc_tab))
    w(f'\n### {TEST_YEAR} holdout (refit on {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}, evaluated once)\n')
    w(md(ho_show))
    w('\n### How the model was chosen\n')
    w(f'- LOYO: lowest error **{label_of(v["best_loyo"])}**; simplest within one fold-SD: **{label_of(v["pick_loyo"])}**.')
    w(f'- Forward-chaining: lowest error **{label_of(v["best_fc"])}**; simplest within one fold-SD: **{label_of(v["pick_fc"])}**.')
    gap_txt = ', '.join(f'{y}: {g:+.1f}%' for y, g in v['fc_gap'].items())
    w(f'- The LOYO pick\'s forward-chaining error vs its LOYO error on the same years: {gap_txt}. '
      + ('Forward-chaining is clearly worse (> 2% on every shared year), so the forward-chaining pick is used.'
         if v['fc_clearly_worse'] and v['pick_fc'] != v['pick_loyo'] else
         'Not clearly worse, so the LOYO pick stands.'))
    w(f'- **Chosen: {label_of(chosen.name)}.**\n')
    w('<details><summary>Selection table (LOYO)</summary>\n')
    w(md(choice))
    w('\n</details>\n')

    w(f'## Chosen model: coefficients (fit on {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})\n')
    w(f'Target units: **{unit_word}** of % vs SB, positive = better. "per-SD coef" = effect of +1 SD of a continuous '
      'feature. CIs and p-values are clustered by athlete. VIF > 5 would flag collinearity.\n')
    w(md(coefs))
    w('\n![Coefficients](figures/coefficients.png)\n')

    w('## By event group\n')
    w('### Leave-one-year-out (out-of-fold predictions, all 5 folds)\n')
    w(md(v['group_table'](v['bd_loyo'])))
    w(f'\n### {TEST_YEAR} holdout\n')
    w(md(v['group_table'](v['bd_test'])))
    tf = pd.concat({'LOYO': v['group_table'](v['tf_loyo']), str(TEST_YEAR): v['group_table'](v['tf_test'])})
    tf.index = [f'{a}: {b}' for a, b in tf.index]
    w('\n### Track vs field\n')
    w(md(tf))
    worse = [g for g in group_order_list() if g in v['group_table'](v['bd_test']).index
             and v['group_table'](v['bd_test']).loc[g, 'worse than A?'] == 'yes']
    worse_cv = [g for g in group_order_list() if v['group_table'](v['bd_loyo']).loc[g, 'worse than A?'] == 'yes']
    w(f'\nGroups where the chosen model is worse than SB as-is: LOYO {", ".join(worse_cv) or "none"}; '
      f'{TEST_YEAR} {", ".join(worse) or "none"}.')
    if worse:
        gt = v['group_table'](v['bd_test'])
        w(f'In {TEST_YEAR} these athletes finished closer to their SB than in the training years (bias '
          + ', '.join(f'{g.lower()} {gt.loc[g, "bias % chosen"]:+.2f}%' for g in worse)
          + '), so the average drop-off the model subtracts was too pessimistic and SB as-is happened to be the better '
            'guess. Their training-year drop-off is under 1%, so the two are close either way.')
    w('')
    w('![Predicted vs actual 2026](figures/pred_vs_actual_2026.png)\n')
    w('![Residuals by group and year](figures/residuals_by_group_and_year.png)\n')

    w('## Bias by held-out year\n')
    w(f'Mean of actual − predicted (% vs SB). Negative = athletes did worse than the model expected.\n')
    w(md(bias))
    w(f'\nIn the hot 2022 fold (89.8°F), the model without temperature ({m5}) is off by {b22[m5]:+.2f}%; adding '
      f'temperature ({m6}) gives {b22[m6]:+.2f}%, and temperature × event group ({m6x}) {b22[m6x]:+.2f}%. '
      'In this fold the temperature slope is learned from four championships between 65 and 76°F and then '
      'extrapolated to 89.8°F.')
    b25 = bias.loc['2025'] if '2025' in bias.index else None
    if abs(b22[m6]) >= abs(b22[m5]) - 0.05:
        w(f'**Temperature does not reduce the 2022 bias out of sample:** without 2022, the other four championships show '
          f'no temperature trend ({v["slope10_no22"]:+.2f}% per +10°F), so there is nothing to extrapolate.'
          + (f' It also adds bias in 2025, the coolest meet ({b25[m5]:+.2f}% → {b25[m6]:+.2f}%), because the slope '
             'learned with 2022 included predicts a cool meet should go better than it did.' if b25 is not None
             and abs(b25[m6]) > abs(b25[m5]) + 0.1 else ''))
    else:
        w(f'Temperature reduces the 2022 bias by {abs(b22[m5]) - abs(b22[m6]):.2f} percentage points.')
    w('')

    w('## Nonlinearity check\n')
    nl = v['nonlinear']
    if nl:
        nl_rows = fold_table('LOYO', [chosen_or(v['chosen0'])] + [s.name for s in nl])[['mean', 'fold SD']]
        nl_rows.index = [label_of(n) for n in nl_rows.index]
        w(f'Squared terms added to {v["chosen0"].name} (LOYO sRMSE):\n')
        w(md(nl_rows))
        kept = [s.name for s in nl if s.name == chosen.name]
        none_kept = ('The squared term did' if len(nl) == 1 else 'Neither squared term did') + \
            ' not beat the selection rule, so the model stays linear.'
        w(f'\n{"Kept: " + kept[0] if kept else none_kept} '
          'Temperature has only 5 distinct values, so a curve in temperature can\'t be estimated.\n')
    else:
        w('The chosen model has no continuous gap/risk terms, so no squared terms were tested.\n')
    w('![Residuals vs features](figures/residuals_vs_features.png)\n')

    w('## Sensitivity checks\n')
    w(f'**Excluding 2021 (COVID season):** LOYO over 2022–2025 only. Chosen under this scheme: **{label_of(v["pick_no21"])}**.\n')
    keep = [label_of(n) for n in ladder_raw + ladder_sd if label_of(n) in no21_tab.index]
    same_years = per_fold[(per_fold['scheme'] == 'LOYO') & per_fold['fold'].isin(TRAIN_YEARS[1:])].groupby('model')['srmse'].mean()
    sens = pd.DataFrame({'LOYO incl. 2021 (2022–25 folds)': [same_years.get(n.split(':')[0], np.nan) for n in keep],
                         'LOYO excl. 2021': no21_tab.loc[keep, 'mean'].values}, index=keep)
    sens['change'] = sens.iloc[:, 1] - sens.iloc[:, 0]
    w(md(sens))
    w(f'\n**{TEST_YEAR} with and without the {MAX_OFF_SEASON_BEST:.0f}% filter** (the filter uses the outcome, so the '
      'unfiltered row is the honest worst case):\n')
    n_dropped = int(v['test_unfiltered']['not_a_race'].sum())
    if n_dropped:
        w(f'{n_dropped} {TEST_YEAR} mark(s) were more than {MAX_OFF_SEASON_BEST:.0f}% off SB; the holdout table shows both '
          'versions.\n')
    else:
        w(f'No {TEST_YEAR} mark was more than {MAX_OFF_SEASON_BEST:.0f}% off SB, so the filter changes nothing on the '
          'holdout (both versions score identically).\n')

    w('## Influence analysis\n')
    w(f'Full main-effects model ({make_spec("M6", BASE_METRIC, chosen.scale).name}) fit on {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}:\n')
    w('**By standardized coefficient** (effect of +1 SD, clustered CIs):\n')
    w(md(rank_std, index=False))
    w('\n**By drop-one increase in LOYO sRMSE** (positive = the model is worse without it):\n')
    w(md(v['drop_table'], floatfmt='.4f', index=False))

    w('\n### (a) Does risk_score hurt championship performance?\n')
    w(md(risk_t, floatfmt='.3f', index=False))
    w(f'\n{"Yes" if risk_sig and raw_risk["coef per +0.10"] < 0 else "No"}: with the raw rate, +0.10 risk_score '
      f'(one extra bad race in ten) changes the result by {raw_risk["coef per +0.10"]:+.3f} {unit_word} '
      f'(95% CI {raw_risk["CI low"]:+.3f} to {raw_risk["CI high"]:+.3f}, p = {raw_risk["p"]:.2f}). '
      f'{"The estimate is slightly positive, the opposite of the hypothesis. " if raw_risk["coef per +0.10"] > 0 else ""}'
      f'Smoothing {"does" if smooth_changes else "does not"} change it: per SD the coefficient goes from '
      f'{raw_risk["per +1 SD"]:+.3f} (raw) to {risk_t["per +1 SD"].iloc[1:].min():+.3f} to {risk_t["per +1 SD"].iloc[1:].max():+.3f} '
      f'(smoothed), {"outside" if smooth_changes else "inside"} the raw CI. The per-0.10 coefficient grows only because smoothing squeezes '
      f'risk_score toward the average. {"" if smooth_changes else "The raw version is kept for interpretability. "}'
      f'LOYO error is the same for every version.')
    nm = v['nomark_by_risk']
    w(f'\nrisk_score may matter more for *whether* an athlete gets a mark at all, which the mark model can\'t see. Among '
      f'{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} championship entries, the no-mark rate (DNF/DQ/foul-out) is '
      + ', '.join(f'{r.no_mark_rate * 100:.1f}% for risk {k} (n={int(r.entries)})' for k, r in nm.iterrows())
      + f'. Each +0.10 risk_score multiplies the odds of a championship no-mark by {v["nm_or"]:.2f} '
        f'(95% CI {v["nm_ci"].iloc[0]:.2f}–{v["nm_ci"].iloc[1]:.2f}, logistic model with track/field, clustered by athlete).\n')

    w('### (b) Does temperature explain part of the drop-off?\n')
    w(f'Year-level check: mean residual of {no_temp_name(chosen)} (no temperature) for each championship vs its temperature:\n')
    w(md(v['by_year'], floatfmt='.2f'))
    w(f'\n- Slope: **{v["slope10"]:+.2f}% per +10°F** (95% CI {v["lo10"]:+.2f} to {v["hi10"]:+.2f}, p = {v["p_year"]:.2f}, '
      f'5 points, 3 degrees of freedom).')
    w(f'- Without 2022: {v["slope10_no22"]:+.2f}% per +10°F (95% CI {v["lo10_no22"]:+.2f} to {v["hi10_no22"]:+.2f}).')
    w(f'- The athlete-level OLS coefficient in M6 is {tc["coef"] * 10:+.3f} {unit_word} per +10°F with a naive CI of '
      f'{v["temp_naive_ci"].iloc[0]:+.3f} to {v["temp_naive_ci"].iloc[1]:+.3f}; that CI treats ~{len(train)} rows as '
      'independent when temperature has only 5 values, so it is far too narrow. Use the year-level CI.')
    w(f'- Temperature is also confounded with everything else that changes by year (venue, track, 2021 COVID '
      'season, field depth). Treat it as a hypothesis with weak support, not a conclusion.\n')

    w('### (c) SB vs median as the base\n')
    w(md(cmp))
    w(f'\nFull model (M6) LOYO sRMSE: SB base {sb_mean:.3f} vs median base {med_mean:.3f} '
      f'(median − SB per fold: {", ".join(f"{d:+.3f}" for d in base_diff)}). '
      + ('The choice barely matters once the consistency gap is in the model, because SB + gap carries the same '
         'information as median + gap.' if abs(sb_mean - med_mean) < base_diff.std() else
         ('SB is the better base.' if sb_mean < med_mean else 'Median is the better base.')) + '\n')

    w('### (d) Does it transfer better than the 2021→2022 single-feature lines (~0.46)?\n')
    w(f'- LOYO 2022 fold (trained on the other four years): pooled within-event R² **{r2_2022:.2f}**.')
    w(f'- LOYO mean: {loyo_mean["within-event R² (pooled)"]:.2f}; forward-chaining mean: '
      f'{fc_tab.loc[label_of(chosen.name), "within-event R² (pooled)"]:.2f}; {TEST_YEAR} holdout: '
      f'{chosen_ho["within-event R² (pooled)"]:.2f}.')
    w(f'- {TEST_YEAR} was an easier year to predict than the CV years: even SB as-is scores '
      f'{a_ho["within-event R² (pooled)"]:.2f} there vs {loyo_a["within-event R² (pooled)"]:.2f} in LOYO. '
      'Use the LOYO mean as the expected accuracy for a new year.')
    w(f'- {"Yes" if chosen_ho["within-event R² (pooled)"] > 0.46 else "No"}, compared with ~0.46. Most of the improvement '
      'comes from training on several years and from the relative scale, which shares one drop-off across all events '
      'instead of fitting 37 separate lines.\n')

    w('### (e) Do LOYO and forward-chaining agree?\n')
    agree = v['pick_loyo'] == v['pick_fc']
    w(f'{"Yes" if agree else "No"}: LOYO picks {label_of(v["pick_loyo"])}, forward-chaining picks {label_of(v["pick_fc"])}. '
      f'The LOYO pick\'s forward-chaining error differs from its LOYO error on the same years by {gap_txt}. '
      + ('' if agree else 'Forward-chaining trains on 2–4 years, so it favors simpler models; with only 3 folds the '
         'disagreement is weak evidence of a time trend. ') +
      'The year-to-year swings in bias (table above) are larger than any steady trend.\n')

    w('## Plain-language summary for coaches\n')
    w(f'- **Expect athletes to finish about {abs(drop_all):.1f}% worse than their season best at Big West.** That\'s the '
      f'average over {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}. It varies by event group: '
      + ', '.join(f'{GROUP_LABEL[g].lower()} {train_perf[g]:+.1f}%' for g in GROUPS) + '.')
    if not examples.empty:
        w('\n' + md(examples, index=False) + '\n')
    cc = v['chosen_coefs']
    if 'gap_pct' in cc.index and not chosen.scale:
        frac = -cc.loc['gap_pct', 'coef']
        dist = cc.loc['C(event_group)[T.distance]', 'coef'] if 'C(event_group)[T.distance]' in cc.index else np.nan
        lvl = cc.loc['sb_level', 'coef'] if 'sb_level' in cc.index else np.nan
        w(f'- **Rule of thumb from the model:** predict a mark about **{frac * 100:.0f}% of the way from the athlete\'s SB '
          f'to their season median**. Consistent athletes stay close to their SB; streaky ones fall back toward their '
          f'typical mark.'
          + (f' Middle-distance and distance runners finish about another {abs(dist):.1f}% slower than that '
             '(tactical championship races).' if pd.notna(dist) else '')
          + (f' Top seeds give back a little more: {abs(lvl):.1f}% per 1 SD of SB above the field.' if pd.notna(lvl) and lvl < 0 else ''))
    if 'temp' in effects:
        w(f'- **Heat:** each +10°F at the meet shifts the prediction by {effects["temp"]:+.2f}% '
          f'(year-level 95% CI {v["lo10"]:+.2f} to {v["hi10"]:+.2f}%). That rests on 5 championships, so treat it as a hint.')
    else:
        w(f'- **Heat:** no reliable effect. The year-level estimate is {v["slope10"]:+.2f}% per +10°F, but the 95% CI '
          f'({v["lo10"]:+.2f} to {v["hi10"]:+.2f}%) is wide because there are only 5 championships. The hot 2022 meet '
          'did have the biggest drop-off.')
    if 'risk' in effects:
        w(f'- **Risky athletes:** one extra bad race in ten during the season (+0.10 risk_score) shifts the prediction by '
          f'{effects["risk"]:+.2f}%.')
    else:
        w(f'- **Risky athletes:** DQs, DNFs and no-marks in the regular season don\'t predict a worse *mark* at Big West '
          f'(+0.10 risk_score ≈ {raw_risk["coef per +0.10"]:+.2f} {unit_word}, not significant), but they do predict a higher '
          f'chance of no mark at all (odds × {v["nm_or"]:.2f} per +0.10).')
    w(f'- **How close are predictions?** On {TEST_YEAR}, the typical miss was {chosen_ho["MAE %"]:.1f}% of the mark '
      f'({chosen_ho["MAE track (s)"]:.2f} s on track, {chosen_ho["MAE field (m)"]:.2f} m in the field). Field events, '
      'especially throws, are much less predictable than sprints.\n')

    w('## Files\n')
    w(f'- `predictions_{TEST_YEAR}.csv`: every {TEST_YEAR} athlete-event with SB, predicted and actual marks, error '
      f'(actual − predicted, s or m) and the {MAX_OFF_SEASON_BEST:.0f}% filter flag.')
    w('- `chosen_model.pkl`: coefficients, spec, and the training-year SD and risk-rate tables. Load it with '
      '`spec, model, prep = load_model()` from `model_bigwest.py`, then `predict(spec, model, prep(rows))`.')
    w('- `tables/`: every model\'s per-fold metrics, holdout metrics, coefficients, row counts by event.')
    w('- `figures/`: the plots above.')
    RESULTS_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')


def describe(spec):
    if not spec.is_model:
        return 'season best as-is'
    if not spec.terms:
        return 'season best + average drop-off'
    parts = []
    for t in spec.terms:
        if t.startswith('temp_c:'):
            parts.append('temperature × ' + ('track/field' if 'track_field' in t else 'event group'))
        elif t.startswith('I('):
            parts.append(pretty_term(t))
        else:
            parts.append({'C(track_field)': 'track/field', **TERM_LABEL}.get(t, t))
    extra = [x for x in [spec.base == 'median' and 'median base', spec.scale and 'SD-scaled target',
                         spec.risk_k is not None and f'smoothed risk (k={spec.risk_k})',
                         spec.mixed and ('random intercept by event' + (' + slope on gap' if spec.mixed == 'slope' else ''))] if x]
    return ', '.join(parts) + (f' ({"; ".join(extra)})' if extra else '')


def group_order_list():
    return [GROUP_LABEL[g] for g in GROUPS]


def chosen_or(spec):
    return spec.name


def no_temp_name(chosen):
    return make_spec('M5', BASE_METRIC, chosen.scale).name


if __name__ == '__main__':
    main()
