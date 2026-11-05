# Big West Track & Field Championship Predictor (DSC 180)

We predict each athlete's mark at the **Big West Outdoor Track & Field Championships** from their regular-season results. Models are trained on the 2021–2025 championships and tested once on **2026**, which is never used for fitting or model choice.

- [Repository layout](#repository-layout)
- [Data: what it is and how to get it](#data-what-it-is-and-how-to-get-it)
- [Software dependencies](#software-dependencies)
- [Reproducing our results](#reproducing-our-results)
- [What we explored](#what-we-explored)
- [Findings so far](#findings-so-far)
- [Next steps](#next-steps)

## Repository layout

Everything lives in `big-west-predictor/`:

```
big-west-predictor/
├── data/
│   ├── raw/                          # data as collected
│   │   ├── championships/            # saved TFRRS Big West results pages, men + women, 2021–2026
│   │   ├── bigwest_regular_season_partial.csv
│   │   └── national_event_ranking.ndjson
│   └── processed/                    # cleaned tables used by all analysis code
├── notebooks/
│   ├── bigwest_2021_2026_data_wrangling.ipynb   # parses the saved championship pages
│   ├── meet_temperatures.ipynb                  # adds `temp`, builds meet_weather.csv
│   ├── risk_score.ipynb                         # adds `risk_score`
│   └── bigwest_2021_predictor.ipynb             # single-feature models (SB, median, ...)
└── modeling/
    ├── model_bigwest.py              # combined model: cross-validation, selection, 2026 test
    ├── results.md                    # full generated report (tables, answers, figures)
    ├── predictions_2026.csv          # every 2026 athlete-event: predicted vs actual
    ├── chosen_model.pkl              # saved model (load with load_model() in model_bigwest.py)
    ├── figures/                      # plots used in results.md
    └── tables/                       # per-fold metrics for every model, coefficients, counts
```

## Data: what it is and how to get it

### Getting the data

**All data needed to reproduce our results is committed to this repo.** Cloning it gives you everything (about 130 MB, mostly saved web pages). Keep the folder structure as is: the notebooks read `../data/processed/...` relative to `notebooks/`, and `model_bigwest.py` finds the data relative to its own location.

```bash
git clone https://github.com/ChidiebereAgunanne/DSC180.git
cd DSC180/big-west-predictor
```

### Sources

| Source | What we took | How |
|---|---|---|
| [TFRRS](https://www.tfrrs.org) | Big West championship results, 2021–2026 | Results pages saved as HTML in `data/raw/championships/` |
| TFRRS (via the parse.bot TFRRS scraper API) | Every regular-season mark for athletes who competed at Big West | Athlete-page scrape → `regular_season.csv` |
| TFRRS meet pages | Venue and city of each meet | `meet_temperatures.ipynb` |
| [Open-Meteo](https://open-meteo.com) (free, no key) | City coordinates and daily high temperatures | `meet_temperatures.ipynb` |

### Processed tables (`big-west-predictor/data/processed/`)

| File | Rows | Contents |
|---|---|---|
| `bigwest_2021_2026_results.csv` | 4,579 | One row per athlete-event at each championship: the final if the athlete ran one, otherwise their prelim. Includes `temp`. **This is the prediction target.** |
| `bigwest_2021_2026_audit.csv` | 5,303 | Every round kept separately (prelims and finals). `results` is a de-duplicated version of this. |
| `regular_season.csv` | 41,493 | Every regular-season mark (indoor and outdoor) for those athletes, with `temp` and `risk_score`. |
| `meet_weather.csv` | 378 | One row per meet (372 regular-season + 6 championships): TFRRS link, venue, city, coordinates, indoor flag, `temp`. Lets anyone check each temperature. |
| `bigwest_2021_2026_team_scores.csv` | 118 | Team scores per championship. |
| `bigwest_roster.csv` | 5,338 | `athlete_id`, year, school. |
| `national_event_ranking.csv` | 642 | National event rankings for Big West teams (2024–2026 only). |

**Features we added:**
- **`temp`:** the mean daily high (°F) at the meet's city over the meet's days, from Open-Meteo's historical weather. It's modeled weather for the city, not a thermometer at the track. Indoor meets are left blank.
  - Championship values: 70.1 (2021, Irvine), 89.8 (2022, Davis), 75.6 (2023, Fullerton), 75.2 (2024, San Luis Obispo), 65.2 (2025, Long Beach), 68.1 (2026, Long Beach).
  - TFRRS lists the 2026 meet as May 8–16 because it includes the multi-events weekend. We use the main meet, **May 15–16**.
- **`risk_score`:** each athlete-season's share of "bad" individual races: DQ or false start, DNF, field no-mark (FOUL/NH/NM), or an outlier mark far off their own season best.
  - The outlier cutoff is per event: beyond Q3 + 3·IQR of % off season best.
  - Relays and did-not-start entries are excluded.
  - The modeling script rebuilds it from pre-championship races only.

### Rebuilding the data from scratch (optional)

You don't need this to reproduce our results.

- **Championship tables:** `bigwest_2021_2026_data_wrangling.ipynb` parses the saved pages in `data/raw/championships/`. TFRRS field-event tables have several `MARK` columns, so the notebook has you approve the correct column per event before exporting.
- **Regular-season marks:** scraping these needs a parse.bot API key and the team's scraping notebooks, which are kept out of git. Ask the team.
- **`temp` and `risk_score`:** run `meet_temperatures.ipynb`, then `risk_score.ipynb` (commands below). Each one only rewrites its own column.
  - `meet_temperatures.ipynb` reuses the meet locations saved in `meet_weather.csv`. Set `REFRESH_LOCATIONS = True` at the top to look every meet up on TFRRS again (about 400 page requests, one per second).

## Software dependencies

- **Python 3.13** (tested with 3.13.5).
- The packages pinned in [`requirements.txt`](requirements.txt): numpy, pandas, scipy, statsmodels, scikit-learn, patsy, matplotlib, tabulate, tqdm, requests, beautifulsoup4, lxml, jupyter, nbconvert, ipykernel.
- An internet connection only for `meet_temperatures.ipynb`. Everything else runs offline.

Set up an environment from the repo root (`DSC180/`) with **conda**:

```bash
conda create -n bigwest python=3.13 -y
conda activate bigwest
pip install -r requirements.txt
```

or with **venv**:

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing our results

Run from `DSC180/big-west-predictor/` with the environment active. Steps 2 and 3 reproduce every number in our findings. Step 1 is only needed if you change the raw data.

```bash
cd big-west-predictor

# 1. (Optional) Rebuild the added features. Their outputs are already committed.
jupyter nbconvert --to notebook --execute --inplace notebooks/meet_temperatures.ipynb   # needs internet, ~5 min
jupyter nbconvert --to notebook --execute --inplace notebooks/risk_score.ipynb          # < 1 min

# 2. Combined model: cross-validation, model choice, one-time 2026 test, report (~20 s)
python modeling/model_bigwest.py

# 3. Single-feature models: 2021 analysis and the 2021–2025 → 2026 test (~15 s)
#    Run after step 2: it compares against modeling/predictions_2026.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/bigwest_2021_predictor.ipynb
```

| Step | Writes |
|---|---|
| 1 | `temp` in `regular_season.csv` and `bigwest_2021_2026_results.csv`; `meet_weather.csv`; `risk_score` in `regular_season.csv` |
| 2 | `modeling/results.md`, `predictions_2026.csv`, `chosen_model.pkl`, `figures/*.png`, `tables/*.csv` |
| 3 | The notebook's tables and charts (saved in the notebook) |

**Notes**
- Results are deterministic: rerunning gives identical files.
- Settings are at the top of `modeling/model_bigwest.py`: `TRAIN_YEARS`, `TEST_YEAR`, `BASE_METRIC` (`'sb'` or `'median'`), and the outlier threshold.
- Progress bars (tqdm) show the cross-validation fits and the weather downloads.
- To open the notebooks interactively instead, run `jupyter lab` from `big-west-predictor/` and run cells top to bottom.

## What we explored

1. **Understanding the data.** We compared the `audit` (every round) and `results` (one row per athlete-event) tables. We also flagged non-races, such as a 100.56 s 400m (likely an injury). Championship marks more than 50% worse than the athlete's season best are dropped.
2. **Choosing a fair metric.** One R² across all events comes out around 0.99 only because it separates 100m times from 5000m times. All comparisons use **within-event R²**: how well a predictor ranks athletes *within the same event*.
3. **Single-feature models** (`bigwest_2021_predictor.ipynb`). We compared season best (SB), season median, mean of the last 3 meets, most recent meet, and wind-legal SB.
   - Each was fit as one line per event, first on 2021 alone. The main test fits on 2021–2025 and tests on 2026.
   - We split results by track vs field and by event, and compared "SB as-is" (no fitting).
4. **New features.** We added meet temperature (`temp`) and an athlete reliability measure (`risk_score`).
5. **Combined model** (`modeling/model_bigwest.py`). It predicts each athlete's mark as **% better or worse than their SB**, so one model covers every event.
   - Feature ladder: average drop-off → event group + gender → SB level (where the athlete's SB ranks among that event's championship field) → **consistency gap** (how far the season median is behind the SB) → `risk_score` → temperature → temperature × event group. We also tried mixed-effects models, squared terms, an SD-scaled target, track/field grouping and smoothed `risk_score`.
   - Model choice used **leave-one-year-out** cross-validation within 2021–2025. **Forward-chaining** (train on past years only) served as a time check, plus a sensitivity check excluding the 2021 COVID season.
   - The simplest model within one fold-SD of the best was kept, then tested once on 2026.

## Findings so far

| Predictor (one line per event) | Fit 2021–25 | **Test 2026** |
|---|---|---|
| Season best | 0.550 | **0.589** |
| Wind-legal season best | 0.550 | 0.586 |
| Season median | 0.544 | 0.555 |
| Mean of last 3 meets | 0.525 | 0.542 |
| Most recent meet | 0.437 | 0.386 |
| Season best as-is (no fitting) | — | 0.352 |
| **Combined model (M4)** | — | **0.602** |

*Within-event R² on the same 701 athletes in 2026.*

- **Season best is the best single predictor**, tied with wind-legal SB. Median and last-3 are close behind; the most recent meet is clearly worst.
- **Athletes usually finish below their SB at Big West:** about 70% of them, by a median of 1.2%. It's worse in distance (~2%) and throws (~2.5%) than in sprints (~0.8%). Using SB as-is is too optimistic, and learning that drop-off is most of what fitting adds.
- **Consistency is the strongest signal.** The chosen model (event group, gender, SB level, consistency gap) predicts about **halfway between an athlete's SB and their season median**. Distance runners finish about another 1.4% slower, likely because championship races are tactical.
- **More training years help.** Fit on 2021 alone, the SB line scored 0.46 on 2022. Fit on 2021–2025, it scores 0.59 on 2026. Both cross-validation methods choose the same model.
- **Most accuracy comes from the SB itself.** The model explains only 8–10% of *who beats or misses* their SB on the day.
- **Temperature isn't supported.** The estimate is −0.3% per +10°F (95% CI −0.7 to +0.1). It comes entirely from the hot 2022 meet, rests on only 5 championships, and doesn't improve predictions for new years.
- **`risk_score` doesn't predict the mark, but it does predict a no-mark.** Championship DNF/DQ/foul-out rates are 2.4% for athletes with zero risk vs 6.9% above 0.10.

Full details, per-fold tables, coefficients and plots are in [`big-west-predictor/modeling/results.md`](big-west-predictor/modeling/results.md).

## Next steps

1. **No-mark model:** model DNF/DQ/no-mark separately using `risk_score`, then combine it with the mark model to predict places and team scores.
2. **Uncertainty:** report a likely range for each prediction, not just a point estimate.
3. **New features:**
   - season trend (improving or declining)
   - days since last race
   - number of meets (single-meet athletes get a gap of 0 automatically)
   - prior-year Big West results (58% of 2026 athletes competed in earlier years)
   - national rankings (2024–2026 only)
4. **Distance events:** treat them differently (e.g., predict place or pace instead of time).
5. **Temperature:** test it properly with more championships (other conferences) or hourly, per-session weather.
6. **Build script:** wrap the reproduction commands in a single build script / environment file (conda or Docker).
