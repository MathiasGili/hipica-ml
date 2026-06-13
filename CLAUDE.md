# CLAUDE.md — Hipica-ML knowledge base

> Living reference for the Maroñas Trifecta classifier project. Captures
> everything we learned: dataset shape, scraping endpoints, feature
> contract, model performance, gotchas, and the commands that work.
>
> **Project goal:** binary classifier `in_trifecta = 1` if a horse
> finishes 1st, 2nd or 3rd; otherwise 0. Data sourced from
> `https://hipica.maronas.com.uy/`.

---

## 1. Final dataset

| Property | Value |
|---|---|
| Raw Tabuladas downloaded | **1 301** files |
| Raw size | ~1.3 GB of `.xls` |
| Long-form rows | **98 398** |
| Unique horses | **8 610** |
| Date range | **2013-06-30 → 2026-05-31** (~13 years) |
| Racetracks observed (history) | 20 abbreviations: MRÑ, L.PD, COL, FLS, FLD, MEL, PAY, RCH, CHS, GAV, HCH, SAL, SI, TAR, VSC, MON, LP, PAL, CRI, C.JD |
| Label balance (`in_trifecta`) | **35.76%** positive |
| Build time (parquet from raw) | ~47 s |
| Loader cache | `data/processed/history.parquet` |

**Why so many tracks?** The scraper only requests Maroñas-day Tabuladas
(racetrack_id = 1), but each Tabulada includes the **full career history
table** of every horse in the program — which lists *every* race that
horse ever ran, including races at other tracks. The loader picks all of
that up. For training we only use historical rows; the column
`racetrack_id` is the track of the historical race, not the target race.

**Files skipped during loading** (HTTP error pages or empty files
returned by the source):
- `Tabulada_RT1_20160603.xls`
- `Tabulada_RT1_20160604.xls`
- `Tabulada_RT1_20180608.xls`
- `Tabulada_RT1_20190831.xls` (0 bytes)

The loader logs and skips these — does not crash.

---

## 2. Final model performance

Trained with `python -m src.training.train --cache --device cpu`,
test_size=0.2, temporal split (random splits explicitly disabled).

- Cutoff date: **2024-04-14** (test set is everything ≥ this date)
- n_train = 65 990 · n_test = 16 605 · 313 jockeys indexed at fit time
- Algorithm: XGBoost `binary:logistic`, `tree_method=hist`,
  `n_estimators=600`, `max_depth=6`, `lr=0.05`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=2`, `reg_lambda=1.0`.
- **v4 features (current)**: 33 numeric + 2 categorical = 35 total.
  New in v4 (7 added on top of v3): `dividend_career_mean`,
  `dividend_last3_mean`, `dividend_career_min`, `dist_diff_from_avg`,
  `weight_change_from_last`, `jockey_career_runs`,
  `jockey_career_show_rate`. The jockey features are the only ones that
  index across horses (every horse a jockey rode contributes).

### v4 vs v3 vs v1 (test set, same cutoff and config)

| Metric | v1 test | v3 test | **v4 test** | v1→v4 |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.682 | 0.684 | **0.704** | **+0.022** |
| PR-AUC | 0.619 | 0.620 | **0.634** | +0.015 |
| Log-loss | 0.603 | 0.603 | **0.592** | −0.011 |
| Brier | 0.208 | 0.207 | **0.203** | −0.005 |
| Precision @0.5 | 0.729 | 0.727 | 0.691 | −0.038 |
| Recall @0.5 | 0.272 | 0.278 | **0.338** | +0.066 |
| F1 @0.5 | 0.396 | 0.402 | **0.453** | +0.057 |
| Positive rate | 0.378 | 0.378 | 0.378 | — |

MLflow run_id for v4: `98d0a5cce6024f86aa69cfece693d9be`.
Train ROC-AUC: 0.849 (gap to test: 0.145, basically unchanged from v3's
0.145 — not new overfitting, just more signal extracted on both sides).

**Reading the numbers**

- Test ROC-AUC 0.704 — close to the practical ceiling for trifecta
  prediction on this kind of card (~0.75 with bookmaker odds; we're
  approximating that with `dividend_career_*` features).
- Test precision @0.5 dropped from 0.729 to 0.691, but recall jumped
  from 0.278 to 0.338. F1 went from 0.402 to **0.453**. The model is
  more confident overall — if you want the old precision back, raise
  the threshold to ~0.55 and you still keep more recall than v3.
- Test precision @0.5 = 0.691 vs base rate 0.378 → the model's
  positive calls hit the trifecta ~**1.83×** more often than a random
  horse from the field.
- Log-loss and Brier improved without explicit calibration — the new
  features genuinely sharpen the probability distribution.

### What moved the needle in v4 (and why earlier rounds didn't)

- **Dividend (market signal)**: the bookmakers aggregate everything we
  don't know about (workouts, vet reports, jockey form). Even a lagged
  career-mean dividend captures "this horse is generally favoured /
  generally a longshot". This was the single biggest contribution.
- **Jockey features**: `jockey_career_show_rate` indexes across every
  horse the jockey rode. Top jockeys carry signal even on unfamiliar
  horses. 313 jockeys in the training index.
- **Distance fit & weight change**: small but free. Captures
  "horse running unusually long today" and "jockey carries 3 kg more
  than last race" (both classic handicapping inputs).
- v2 added 5 features and moved metrics by 0.001 because XGBoost was
  already extracting that signal from career features.
- v3 fixed `sex_code` / `horse_age` coverage at training time (0 →
  100 %) and moved metrics by 0.002.
- v4 added market and cross-horse jockey signal — information the
  model could not have extracted from horse-only career stats. **+0.020
  ROC-AUC, ~10× the v2/v3 gain.**

---

## 3. Architecture (anti-skew, anti-leakage)

```
data/raw/<Track>/Tabulada_RT<id>_<YYYYMMDD>.xls
                │
                ▼
   src/ingestion/loader.py  (parses Crystal-Reports BIFF .xls)
                │
                ▼
   data/processed/history.parquet     ← shared by training and serving
                │
        ┌───────┴────────┐
        ▼                ▼
 training:           serving:
 src/training/       api/main.py + api/model_loader.py
 train.py
        │                │
        └─── SAME ───────┘
       FeatureEngineeringPipeline
       (src/features/pipeline.py)
```

**Anti-skew guarantees**

1. **Single source of truth** for feature engineering: the
   `FeatureEngineeringPipeline` class. Both training and the FastAPI
   service import it from `src.features.pipeline` and call
   `.fit(history_df)` then `.transform(targets_df)`. There is no
   alternative path to features anywhere in the codebase.
2. **Hard runtime guard** inside `transform()`: if the concatenation of
   "pass-through" columns and "historical" columns ever produces a
   duplicate column name, it raises `RuntimeError`. This is the
   canonical training-serving skew failure mode and we trip it the
   moment it happens.
3. **Full feature contract** in `src/config.py`:
   `ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES`
   (currently 33 numeric + 2 categorical = 35 columns). The
   `transform()` method ensures every one of these columns exists in
   the output (NaN if no history) so the downstream sklearn
   `ColumnTransformer` always sees the expected schema.
4. **Same `requirements.txt`** in every container (training, API,
   Streamlit) so library versions cannot drift.

**Anti-leakage guarantees**

1. **Strict `<` filter**: `_history_for(horse_name, race_date)` returns
   only rows with `race_date < target_date`. A row never sees its own
   outcome. Verified by `test_no_self_leakage`: at the i-th appearance
   of a horse, `career_runs == i`, never `i+1`.
2. **Temporal split only** in `src/training/split.py`:
   `temporal_train_test_split` raises if `strategy != "quantile"`.
   Random splits are explicitly disabled.
3. **Rookie defaults**: a horse appearing for the first time in the
   data has `career_runs = career_wins = ... = 0` and all rates / last-N
   stats are NaN (downstream median imputer fills them deterministically).
   No leakage of "future" history into historical features.

---

## 4. Scraping endpoints (Maroñas)

The public site `https://hipica.maronas.com.uy/` is an AngularJS SPA. We
discovered the back-end by reading its JS bundle (`/bundles/app`).

| Service | Base URL |
|---|---|
| REST | `https://mobile-rest-services-v3.azurewebsites.net/XTurfRestService.svc` |
| Resources | `https://mobile-rest-services-v3.azurewebsites.net/XTurfResourcesService.svc` |

**Methods used by the scraper**

- `POST GetRacingCalendarHistory` → list of race days for a track in a
  date range.
- `POST GetDocuments` (`documentType=2` for Tabulada) → URI of the
  per-race-day Tabulada document.
- `GET GetRacingDocument?...&exportFormat=Excel` → downloads the
  `.xls` file.

**Racetrack IDs** (defined in `src/config.py`)

| ID | Name |
|---|---|
| 1 | Maroñas |
| 4 | Colonia |
| 8 | Flores |
| 9 | Florida |
| 13 | Las Piedras |
| 16 | Melo |
| 21 | Paysandú |
| 22 | Rocha |

**DocumentType enum**: `Program=1`, `Tabulada=2`, `RaceResult=6`,
`RaceResultWeb=10`, `Statistics=11`. Only `2` is used here.

### Scraper gotchas

- The Azure service occasionally emits a **UTF-8 BOM** on JSON
  responses, which breaks `requests.Response.json()`. The scraper
  decodes with `utf-8-sig` defensively.
- Some old dates (2016, 2018, 2019) return HTML error pages or 0-byte
  files instead of an Excel. The scraper writes them to disk anyway;
  the loader skips them with a warning.
- The scraper is **idempotent**: existing files at the target path are
  skipped unless `--force` is passed. Safe to re-run.
- ThreadPoolExecutor with 8 workers; tenacity retries with exponential
  backoff on `requests.RequestException`.

---

## 5. Loader gotchas (Crystal Reports BIFF .xls)

The Tabulada is a Crystal-Reports-generated `.xls`, NOT a clean tabular
spreadsheet. `xlrd==2.0.1` is **mandatory** because newer xlrd dropped
`.xls` support and `openpyxl` can't open BIFF.

Key column offsets used by `_iter_blocks` and `_parse_history_row` (0-indexed):

| Offset | Field |
|---|---|
| 1 | History race date (dd/mm/yy) |
| 4 | Track abbreviation |
| 6 | Finish position (1..25, "DSC", "RTD") |
| 7 | Race winner name |
| 18 | Weight carried (kg) |
| 22 | Distance (m) |
| 25 | Total time (e.g. `1'34''37` → 94.37 s) |
| 28 | Dividend |
| 30 | Jockey name |
| 33 | Body weight (kg) |

Block detection: a "leader row" is identified as `col1 ∈ 1..25 (int) AND
col4 is non-empty AND col30 ∈ 40..70 (kg, sane jockey weight)`. The
following "fecha" row marks the start of the per-block history table.

---

## 6. Feature contract

**Pass-through (from request / target):** `weight_kg`,
`weight_kg_zscore_in_race`, `n_field`, `racetrack_id`, `sex_code`,
`horse_age`, `post_position`, `distance_m`, `jockey_name`.

**Historical (per-horse, leakage-safe):** `career_runs`,
`career_wins`, `career_places`, `career_shows`,
`career_win_rate`, `career_show_rate`, `year_runs`, `year_wins`,
`year_places`, `year_shows`, `year_win_rate`, `year_show_rate`,
`last_finish_pos`, `avg_finish_last3`, `best_finish_last3`,
`rest_days`, `track_runs`, `track_show_rate`, `dist_bucket_runs`,
`dist_bucket_show_rate`, `days_since_last_win`,
`dividend_career_mean`, `dividend_last3_mean`, `dividend_career_min`,
`dist_diff_from_avg`, `weight_change_from_last`.

**Cross-horse (per-jockey, leakage-safe via `race_date < target_date`):**
`jockey_career_runs`, `jockey_career_show_rate`.

**Within-race (computed at transform time):**
`weight_kg_zscore_in_race`, `n_field` — z-score of `kg` within the
target race and the field size.

Total: **33 numeric + 2 categorical = 35 features**. List authoritatively
defined in `src/config.py`.

---

## 7. API contract

**`GET /health`**
```json
{"status": "ok", "model_name": "local", "model_version": null}
```

**`POST /predict_online`** — single horse
```json
{
  "race": {"race_date": "2026-06-08", "racetrack_id": 1, "distance_m": 1600},
  "entry": {"horse_name": "NOSTRADAMUS", "kg": 56.0,
            "post_position": 1, "horse_age": 6, "sex_code": "M"}
}
```

**`POST /predict_batch`** — full field (1..25 horses)
```json
{
  "race": {"race_date": "2026-06-08", "racetrack_id": 1, "distance_m": 1600},
  "entries": [
    {"horse_name": "...", "kg": 56.0, "post_position": 1, "horse_age": 6, "sex_code": "M"},
    ...
  ]
}
```

Pydantic v2 validation (in `api/schemas.py`):
- `kg`: `(30, 80)` exclusive
- `post_position`: `1..25`
- `horse_age`: `2..20`
- `sex_code`: `Literal["M", "H"]`
- `distance_m`: `600..4000`
- `entries`: `1..25` items, **horse names must be unique** within the
  race (uppercased)
- `ValueError` raised by the FE pipeline → HTTP 422 with the message.

---

## 8. Bugs caught and fixed (lessons learned)

### 8.1 MLflow rejects `@` in metric names
First training run crashed with
`MlflowException: Invalid value "test_f1@0.5" for parameter 'name'`.
MLflow only accepts alphanumerics, `_`, `-`, `.`, ` `, `:`, `/`.
**Fix:** rename to `f1_at_05`, `precision_at_05`, `recall_at_05`.
**Lesson:** validate metric names upfront, don't use punctuation.

### 8.2 Real training-serving skew bug (the one we want to catch)
The first version of `_features_from_history` always emitted
`horse_age` and `weight_kg`. At training time the targets frame did
NOT have those columns; at serving time the API request DID include
them. After `pd.concat([targets, feats], axis=1)` the serving frame
had **two columns named `horse_age`**. XGBoost then saw two columns
with the same name and threw:
`The feature names should match those that were passed during fit.`
**Fix:** derive pass-through columns once at the top of `transform()`,
make `_features_from_history` emit ONLY historical columns, and add
a hard `RuntimeError` guard in `transform()` that fires on any
duplicate column name. Two new tests in `tests/test_features.py`
(`test_serving_input_has_no_duplicate_columns`,
`test_serving_pass_through_columns_are_preserved`) lock this in.
**Lesson:** the FE pipeline must work with both shapes — training
(history-only) and serving (history + request fields). The shapes are
different; the contract is "no duplicate columns and every
`ALL_FEATURES` column exists in the output".

### 8.3 FastAPI exception handler signature
First version had:
```python
@app.exception_handler(ValueError)
async def value_error_handler(_, exc):
    return HTTPException(status_code=422, detail=str(exc))
```
Wrong. Exception handlers must **return a `Response`**, not an
`HTTPException`. The framework error was misleading
(`'HTTPException' object is not callable`).
**Fix:** `return JSONResponse(status_code=422, content={"detail": str(exc)})`.

### 8.4 UTF-8 BOM in scraper
The Azure REST service prepends `\ufeff` to some JSON responses,
breaking `resp.json()`.
**Fix:** decode with `resp.content.decode("utf-8-sig")` then
`json.loads`.

### 8.5 Test fixture date overflow
`datetime(2024, 1, 1+i*30)` for i=2 = `datetime(2024, 1, 61)` — invalid.
**Fix:** `base + timedelta(days=30*i)`. Python's `datetime` constructor
does NOT roll over.

### 8.6 Loader UserWarning at training time (fixed in v3)
sklearn's `SimpleImputer` used to warn:
`Skipping features without any observed values: ['horse_age',
'post_position', 'sex_code']`. Root cause: loader emitted those columns
as NaN; they only got values at request time. **Fix:** loader's
`_extract_leader_meta` now reads each block's leader row and attaches
`sex_code` (truly constant per horse) and `horse_age_at_race`
(`leader_age - (tabulada_year - race_year)`, clamped to `>=2`). Only
`post_position` is still NaN at training — deliberately, because the
leader's post position is "today", not what the horse had in past
races, so back-attaching it would inject future info.

---

## 9. Commands that actually work

### Scrape Maroñas full archive
```bash
python -m src.ingestion.scraper --racetrack 1 --from 2010-01-01 --to 2026-12-31
# 1 301 Tabuladas, ~30 min on a residential connection
```

### Build the long-form parquet
```bash
python -c "
from src.config import RAW_DIR, PROCESSED_DIR
from src.ingestion.loader import build_long_form_dataset
build_long_form_dataset(RAW_DIR, cache_path=PROCESSED_DIR / 'history.parquet', use_cache=False)
"
```

### Train (local file MLflow, CPU)
```bash
MLFLOW_TRACKING_URI=file:///tmp/mlruns_smoke XGB_DEVICE=cpu \
  python -m src.training.train --cache --device cpu --test-size 0.2
```

### Train + register against running MLflow service
```bash
MLFLOW_TRACKING_URI=http://mlflow:5000 \
  python -m src.training.train --cache --register --device cuda
```

### Run the test suite
```bash
python -m pytest tests/test_features.py -v
# 7 passed in <1s
```

### Smoke-test the API in-process
```bash
python -c "
from fastapi.testclient import TestClient
from api.main import app
with TestClient(app) as c:
    print(c.get('/health').json())
    print(c.post('/predict_online', json={'race': {...}, 'entry': {...}}).json())
"
```

### Bring the Docker stack up
```bash
docker compose build api streamlit
docker compose up -d postgres mlflow api streamlit
# UI: http://localhost:8501
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
```

### Train inside the GPU profile
```bash
docker compose --profile training run --rm training
# Requires NVIDIA Container Toolkit on the host
```

---

## 10. Open work / known caveats

1. **`post_position` still NaN at training** — by design (see §8.6).
   The leader's post is "today", not the past. Imputer warns; harmless.
   Real fix would require parsing the per-row history table for a post
   column (not present in the Tabulada).
2. **Streamlit Dockerfile** written but image not yet built (the
   `docker compose build streamlit` step). Code is identical pattern
   to the API image; should "just work".
3. **Model registry** never exercised end-to-end against the Postgres
   MLflow service — only against the file backend. The `register=True`
   path through MLflow's sklearn flavor is wired in
   `src/training/train.py` and ready.
4. **GPU training** never actually run; CPU training was sufficient
   for this dataset size (~120s on 65k rows). The training Dockerfile
   uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` and the XGBoost CUDA
   wheels — should work directly when run with NVIDIA Container
   Toolkit.
5. **Feature improvements that should help**:
   - jockey win-rate features (we have the jockey column in the
     long-form dataset, never used).
   - dividend features as a proxy for prior odds.
   - distance-bucket performance.
   - track-bucket performance.
   - days-since-last-win, not just days-since-last-race.
6. **Calibration**: the model is mildly overconfident
   (Brier 0.208, log-loss 0.603 on a well-balanced label). A
   `CalibratedClassifierCV` wrapper or isotonic post-fit could tighten
   probability calibration without changing rankings.

---

## 11. File map

```
.
├── api/
│   ├── main.py             FastAPI app, /health, /predict_online, /predict_batch
│   ├── model_loader.py     MLflow primary, local joblib fallback
│   ├── schemas.py          Pydantic v2 request/response models
│   └── __init__.py
├── app/
│   └── streamlit_app.py    UI: race form + data_editor + bar chart
├── data/
│   ├── raw/Maroñas/        1 301 Tabulada_RT1_*.xls
│   └── processed/history.parquet   98k-row long-form dataset
├── docker/
│   ├── api.Dockerfile
│   ├── streamlit.Dockerfile
│   ├── training.Dockerfile        nvidia/cuda base
│   └── postgres/init.sql           creates `racing` DB
├── docker-compose.yml             postgres / mlflow / api / streamlit / training
├── models/trifecta_pipeline/
│   ├── estimator.joblib            sklearn pipeline (preproc + xgb)
│   └── feature_pipeline.joblib     fitted FeatureEngineeringPipeline
├── src/
│   ├── config.py                   paths, racetrack IDs, feature lists
│   ├── ingestion/
│   │   ├── scraper.py              MaronasScraper, idempotent, parallel
│   │   └── loader.py               Crystal Reports .xls parser
│   ├── features/pipeline.py        FeatureEngineeringPipeline (THE shared one)
│   ├── training/
│   │   ├── train.py                XGBoost + MLflow + temporal split
│   │   └── split.py                temporal_train_test_split
│   └── __init__.py
├── tests/test_features.py          7/7 passing, 2 anti-skew tests
├── requirements.txt                pinned: pandas, sklearn, xgboost, mlflow,
│                                   fastapi, streamlit, xlrd==2.0.1, ...
├── Obligatorio_Machine_Learning_en_Produccion.pdf
└── CLAUDE.md                       this document
```
