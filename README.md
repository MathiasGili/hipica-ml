# hipica-ml — Maroñas Trifecta Classifier

End-to-end ML system for predicting whether a horse will finish in the
**Trifecta** (1st, 2nd or 3rd) at Hipódromo Nacional de Maroñas
(Montevideo, Uruguay). Trained on ~13 years of public race history
scraped from the Maroñas mobile-services API.

> **Status**: working end-to-end. Model v4 hits **0.704 ROC-AUC** /
> **0.453 F1** on a held-out temporal test set (cutoff 2024-04-14).

## Highlights

- **Anti-skew architecture**: a single `FeatureEngineeringPipeline`
  shared by training and the FastAPI service. Hard `RuntimeError`
  guard on duplicate columns to catch training-serving drift the
  moment it happens.
- **Anti-leakage by construction**: strict `<` filter on `race_date`,
  temporal train/test split (random splits explicitly disabled), and
  per-row `horse_age` computed as
  `max(2, leader_age - (tabulada_year - race_year))`.
- **MLflow tracking + model registry** wired in (file + Postgres
  backends both supported).
- **35 features** (33 numeric + 2 categorical), including market
  signal (lagged dividends), cross-horse jockey features (313 jockeys
  indexed), within-race z-scored weight, and distance-fit metrics.
- **Docker Compose stack**: Postgres, MLflow, FastAPI, Streamlit, and
  an optional GPU training container.

## Architecture

```
data/raw/Maroñas/Tabulada_RT1_<YYYYMMDD>.xls   ← 1 301 Crystal Reports BIFF files
        │
        ▼
src/ingestion/loader.py    ← parses .xls into long-form parquet
        │
        ▼
data/processed/history.parquet     ← shared by training and serving
        │
   ┌────┴─────┐
   ▼          ▼
training:    serving:
src/training/train.py   api/main.py
   │          │
   └─ SAME ───┘
   FeatureEngineeringPipeline (src/features/pipeline.py)
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Scrape (1 301 Tabuladas, ~30 min on a residential connection)
python -m src.ingestion.scraper --racetrack 1 --from 2010-01-01 --to 2026-12-31

# 3. Build the long-form parquet from the raw .xls files (~50 s)
python -c "
from src.config import RAW_DIR, PROCESSED_DIR
from src.ingestion.loader import build_long_form_dataset
build_long_form_dataset(RAW_DIR, cache_path=PROCESSED_DIR / 'history.parquet', use_cache=False)
"

# 4. Train (~5 min on CPU)
MLFLOW_TRACKING_URI=file:///tmp/mlruns_smoke XGB_DEVICE=cpu \
  python -m src.training.train --cache --device cpu --test-size 0.2

# 5. Serve
docker compose up -d postgres mlflow api streamlit
# Streamlit UI:  http://localhost:8501
# API docs:      http://localhost:8000/docs
# MLflow:        http://localhost:5000
```

## API

`POST /predict_batch` — full field, recommended (in-race z-score is
computed correctly only when all entries are submitted together).

```json
{
  "race": {"race_date": "2026-06-08", "racetrack_id": 1, "distance_m": 1600},
  "entries": [
    {"horse_name": "NOSTRADAMUS", "kg": 56.0, "post_position": 1, "horse_age": 6, "sex_code": "M", "jockey_name": "PABLO RODRÍGUEZ"},
    {"horse_name": "DEVIL AVENUE", "kg": 57.0, "post_position": 2, "horse_age": 4, "sex_code": "M"}
  ]
}
```

`POST /predict_online` — single horse (in-race z-score falls back to NaN).

`GET /health` — liveness + loaded model version.

## Model performance (v4, current)

Temporal cutoff 2024-04-14. n_train = 65 990, n_test = 16 605, base
trifecta rate = 37.8 %.

| Metric | Train | Test |
|---|---:|---:|
| ROC-AUC | 0.849 | **0.704** |
| PR-AUC | 0.779 | 0.634 |
| Log-loss | 0.473 | 0.592 |
| Brier | 0.154 | 0.203 |
| Precision @0.5 | 0.807 | 0.691 |
| Recall @0.5 | 0.489 | 0.338 |
| F1 @0.5 | 0.609 | **0.453** |

Test precision is **1.83×** the base rate at threshold 0.5. Raise the
threshold to ~0.55 to trade recall for tighter precision.

## Tests

```bash
python -m pytest tests/test_features.py -v
# 7 passed in <1s, including 2 explicit anti-skew regression tests.
```

## Project layout

```
api/                    FastAPI service (/health, /predict_online, /predict_batch)
app/                    Streamlit UI
docker/                 Dockerfiles for api, streamlit, training (CUDA), postgres init
src/
├── config.py           paths, racetrack ids, feature contract
├── ingestion/
│   ├── scraper.py      Maroñas REST client (idempotent, parallel, BOM-safe)
│   └── loader.py       Crystal Reports BIFF .xls parser
├── features/
│   └── pipeline.py     FeatureEngineeringPipeline (THE shared one)
└── training/
    ├── train.py        XGBoost + MLflow + temporal split
    └── split.py        temporal_train_test_split
tests/                  Pytest suite, includes anti-skew regressions
docker-compose.yml      Postgres + MLflow + API + Streamlit + (GPU) training
requirements.txt        Pinned: xlrd==2.0.1 mandatory for the .xls parser
CLAUDE.md               Living engineering notebook (gotchas, decisions, lessons)
```

For the deep dive — every gotcha, dataset stat, scraping endpoint,
loader column offset, and bug we caught — see
[CLAUDE.md](CLAUDE.md).

## License

Code is unlicensed (all rights reserved by the author) at this stage —
this is academic coursework. The race history data scraped from the
public Maroñas API belongs to its respective owners and is not
redistributed in this repository (`data/raw/` is gitignored).

## Acknowledgements

Coursework for **Machine Learning en Producción** at Universidad ORT
Uruguay. Data source: the public AngularJS frontend at
`https://hipica.maronas.com.uy/`.
