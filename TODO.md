# TODO — Obligatorio ML en Producción

> Roadmap of the remaining work to deliver the assignment.
> Source of requirements: `Obligatorio_Machine_Learning_en_Produccion.pdf`.
> Deadline: **2026-07-15 21:00 (Uruguay time)** via `gestion.ort.edu.uy`.
> Submission size cap: **40 MB** (zip / rar / pdf).

---

## Status snapshot

- ✅ Code complete end-to-end (scrape → train → API → Streamlit → Docker).
- ✅ Repo public: <https://github.com/MathiasGili/hipica-ml>
- ✅ v4 model: test ROC-AUC 0.704, F1@0.5 0.453.
- ✅ EDA notebook (`notebooks/01_eda.ipynb`) — 8 figures saved.
- ✅ SHAP notebook (`notebooks/02_explainability.ipynb`) — 4 figures saved.
- ✅ Feature selection (`notebooks/03_feature_selection.ipynb`) — dropped 3 raw features, ROC-AUC +0.0018.
- ✅ DVC — `data/processed/history.parquet` tracked, local remote at `~/.dvc-store`, `dvc pull` round-trip verified.
- ✅ Report — `reports/informe.md` (19 sections) rendered to `reports/informe.pdf` (24 pages, 1.46 MB) via WeasyPrint.
- ✅ Calibration plot, `/predict_explain` endpoint, full Docker stack live with Streamlit screenshots, MLflow Registry on Postgres exercised end-to-end.
- ✅ Live race-day predictions — `src/ingestion/program.py` (Programa scrape + Tesseract OCR of distance badges) + `POST /predict_program` endpoint + Streamlit "🗓️ Race day" tab. End-to-end verified on 2026-06-19 (9 races, OCR 9/9 correct).
- ✅ Daily scheduler — `scheduler/main.py` (APScheduler 06:30 UY) + `docker/scheduler.Dockerfile` + compose service `hipica_scheduler`. Pre-warms the API cache for today + tomorrow on every configured racetrack.
- 🟡 Hyperparameter tuning — script ready (`src/training/tune.py`), 3-trial smoke test green; full 50-trial run pending.

---

## 1. Mandatory — still missing

### 1.1 EDA notebook
**Why**: The PDF explicitly requires `Análisis Exploratorio de los Datos` and
the rubric grades on it ("Análisis Exploratorio y Preparación de los datos").

**Where**: create `notebooks/01_eda.ipynb`.

**Content checklist**:
- [ ] Load `data/processed/history.parquet` (98 398 rows).
- [ ] Dataset shape, date range, missingness per column.
- [ ] Label balance (`in_trifecta` ≈ 35.76 % positive).
- [ ] Distribution of `kg`, `distance_m`, `n_field`, `horse_age`.
- [ ] Top racetracks, top jockeys, top trainers by row count.
- [ ] Career-runs distribution per horse (long tail).
- [ ] Correlation heatmap of numeric features.
- [ ] Time series: races per year, label rate per year (drift check).
- [ ] Class balance per racetrack and per distance bucket.
- [ ] Save 3-5 plots into `reports/figures/` for inclusion in the report.

### 1.2 Written report (`informe.pdf`)
**Why**: Section "Entrega" of the PDF — "incluye un informe en conjunto con el
código base". Without the report the submission is incomplete.

**Suggested structure**:
1. Portada — name, group, date, link to repo.
2. Resumen ejecutivo (½ page).
3. Definición del problema y target (`in_trifecta`).
4. Dataset y EDA — figures from the notebook.
5. Arquitectura del sistema — copy the diagram from `CLAUDE.md` §3.
6. Feature engineering — the 35-feature contract, anti-leakage strategy.
7. Modelo y entrenamiento — XGBoost choice, temporal split, MLflow tracking.
8. Resultados — v1 → v4 progression table from `CLAUDE.md` §2.
9. Anti-skew y anti-leakage — single shared FE pipeline + tests.
10. API — endpoints, Pydantic validation, examples.
11. Despliegue — `docker-compose.yml`, services, ports.
12. Trazabilidad — MLflow runs, model registry, joblib fallback.
13. Explicabilidad — SHAP values (after §2.1).
14. Optimización — Optuna tuning + feature selection (after §2.2 and §2.3).
15. Streamlit UI — screenshot.
16. Trade-offs y mejoras posibles — what we'd do next.
17. Uso de IA generativa — declare GitHub Copilot / Claude (the PDF requires
    explicit citation).
18. Anexo: comandos para reproducir, links a runs de MLflow.

**Format**: PDF, ≤ 40 MB total (zip with code + report).

### 1.3 Deployment platform
The PDF says **AWS is recommended, not required** — *"Si ya están familiarizados
con otras plataformas... pueden optar por usarlas"*. Local Docker Compose
counts. We will keep this as-is and document the deployment story in the report.

If we want extra points: deploy the API to a free tier (Render, Fly.io, AWS
EC2 t3.micro). Optional.

---

## 2. Electives — recommended additions

The minimum is 3 electives. We already have 3 done (scraper, ML traceability
partial, Streamlit). Adding 2-3 more raises the grade.

### 2.1 Explainability with SHAP — ✅ done
**Cheap, high-visibility win.** Notebook `notebooks/02_explainability.ipynb`:
- [x] `pip install shap` (already in `requirements.txt`).
- [x] Load fitted pipeline from `models/trifecta_pipeline/`.
- [x] Compute SHAP via `booster.predict(..., pred_contribs=True)` (SHAP 0.49
  is incompatible with XGBoost 2.x's array-format `base_score` when loading
  from joblib — `TreeExplainer(clf)` raises `could not convert string to
  float: '[3.5253826E-1]'`. The booster path produces identical TreeSHAP
  values).
- [x] `shap.plots.bar`, `beeswarm`, `scatter` (top 5), `waterfall`.
- [x] Single-prediction example for the report.
- [x] Logged 4 PNGs + importance CSV as MLflow artifact under `shap/`.

Top SHAP features (mean(|SHAP|), log-odds, sample n=2000):
1. `weight_kg_zscore_in_race` (0.32) — relative weight in race
2. `n_field` (0.18)
3. `jockey_career_show_rate` (0.17) — cross-entity signal validated
4. `racetrack_id_1.0` (0.16) — Maroñas indicator
5. `avg_finish_last3` (0.13)

**Surprising finding for the report**: `dividend_*` features land at #13–#15,
not the top, despite being the change that moved the metric most v3 → v4.
SHAP measures contribution magnitude per prediction, while v3 → v4 gain came
from **orthogonal new information**. A feature can move ROC-AUC without
dominating SHAP magnitude.

**Bonus**: expose `/explain` endpoint in FastAPI returning SHAP values for one
prediction. Optional.

### 2.1.bis Dead feature flagged for §2.3
`post_position` is 100 % NaN at training (see `CLAUDE.md` §8.6); model never
learns from it. Hard candidate to drop in feature selection.

### 2.2 Hyperparameter tuning with Optuna — 🟡 in progress
**Why**: PDF lists "ajuste de hiperparámetros" as an explicit option, and the
rubric asks to "evalúen su impacto en el rendimiento del modelo y sistema".

Script: `src/training/tune.py` (`python -m src.training.tune --cache --n-trials 50`).
- [x] `optuna>=3.6.0` in `requirements.txt` (installed: 4.9.0).
- [x] Search space: `n_estimators ∈ [200, 1200] step 50`, `max_depth ∈ [3, 10]`,
  `lr ∈ [0.01, 0.2]` (log), `min_child_weight ∈ [1, 10]`, `reg_lambda ∈ [0, 5]`,
  `reg_alpha ∈ [0, 2]`, `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6, 1.0]`,
  `gamma ∈ [0, 5]`.
- [x] Inner temporal split inside the train slice; test held out, never seen
  during search. Optimises **PR-AUC** on the val fold.
- [x] TPESampler(seed=42); every trial logs as an MLflow child run; best
  params + final test metrics logged on the parent run.
- [x] Refit on full train with best params, persist to
  `models/trifecta_pipeline_tuned/`.
- [x] Smoke test: 3 trials on CPU green — best val PR-AUC 0.6306, refit test
  ROC-AUC 0.7046 / PR-AUC 0.6350 (parity with v4 already).
- [ ] **Full run (50 trials)** — pending. CPU ≈ 5h, GPU ≈ 1h.
- [ ] Compare final tuned model to v4 in the report.
- [ ] Measure latency impact: `n_estimators` doubling roughly doubles inference time.

### 2.3 Feature selection — ✅ done
**Why**: PDF lists "selección de características para datos tabulares" as an
explicit option. We have 35 features; some may be noise.

Notebook `notebooks/03_feature_selection.ipynb`:
- [x] Permutation importance on test sample (n=5000, 5 repeats).
- [x] XGBoost `gain` importance.
- [x] Mutual information vs target on train (n=20 000).
- [x] SHAP mean(|·|) loaded from `02_explainability.ipynb` CSV.
- [x] **Conservative pass** (`max_rank<0.25` in all 4 metrics): 0 drops.
- [x] **Aggressive pass** (`mean_rank<0.25`): drop 3 raw features
  (`career_shows`, `year_shows`, `track_runs`) — all redundant aggregate counts.
- [x] Retrain with reduced set: **ROC-AUC 0.7035 → 0.7053 (+0.0018)**,
  log-loss 0.5907 → 0.5905. The model is at least as good with 32 features.
- [x] Heatmap saved as `reports/figures/13_feature_rank_heatmap.png`.
- [x] Ranking + summary CSVs saved in `reports/`.

**Recommendation for the report**: adopt the 32-feature subset and remove
`post_position` from the training contract entirely (it's 100 % NaN at fit
time → silently dropped by `SimpleImputer`). At serving time the API still
supplies it, so we'd need a small refactor in `src/config.py` to keep two
lists (`NUMERIC_FEATURES_TRAIN` vs `NUMERIC_FEATURES_SERVE`) — *not done in
this pass* because the gain is marginal and risks complicating the anti-skew
contract.

**Decision**: keep the change reversible. We do **not** flip
`src/config.py` to 32 features yet — we document the finding in the report
and the original `NUMERIC_FEATURES` list stays untouched. If Optuna also
shows the reduced model is more robust, then we flip.

### 2.4 Data versioning — ✅ done
**Why**: PDF lists three things to version under "Trazabilidad de ML":
experiments ✅, models ✅, **data ✅** (now closed).

- [x] `dvc>=3.50.0` in `requirements.txt` (installed 3.67.1).
- [x] `dvc init` (commits `.dvc/` and `.dvcignore`).
- [x] `dvc add data/processed/history.parquet` → md5 `a5edaea5…`,
  pointer file `data/processed/history.parquet.dvc` (98 bytes).
- [x] `.gitignore` adjusted: `data/processed/*` ignored, but
  `!data/processed/*.dvc` whitelisted so the pointer is committable.
- [x] Local default remote: `dvc remote add -d localstore ~/.dvc-store`.
  `dvc push` succeeded (1 file pushed).
- [x] Round-trip verified: deleted `history.parquet` and `dvc pull`
  restored the exact bytes from the local store.
- [x] README § "Data versioning (DVC)" documents the workflow,
  including the regenerate-from-raw fallback if no remote is
  configured.

### 2.5 Live race-day predictions + daily scheduler — ✅ done
**Why**: replaces the manual demo form with a real "scrape today's
program and predict every race" flow. Demonstrates the full operational
loop (scrape → parse → OCR → feature engineering → inference → UI)
running on a schedule.

Modules:
- [x] `src/ingestion/program.py` — `fetch_program(racetrack_id, race_date)`
  downloads `DocumentType=1`, parses entries (col offsets:
  0=post, 2=horse, 11=kg, 13=track_pref, 14=sex, 15=age, 16=jockey),
  detects HTML error pages by BIFF OLE2 magic, converts .xls → .xlsx
  via LibreOffice headless, extracts the embedded distance badges
  (~972×520) ordered by drawing anchor row, and OCRs them with
  Tesseract (voting across thresholds + PSMs + polarity, sanity
  filter 800–3000 m).
- [x] `POST /predict_program` in `api/main.py` — wraps fetch + parse +
  per-race prediction; returns ranked horses per race with
  `race_index`, `distance_m`, `post_time`, `predictions[]`.
  Returns 404 when no Programa is published for the requested date.
- [x] `app/streamlit_app.py` — new "🗓️ Race day (scrape)" tab with
  date picker + racetrack dropdown; old manual flow preserved under
  "✏️ Manual" tab.
- [x] `scheduler/main.py` + `docker/scheduler.Dockerfile` — APScheduler
  cron (06:30 America/Montevideo by default) that calls
  `/predict_program` for today + tomorrow on every configured
  racetrack (`RACETRACK_IDS=1` by default).
- [x] Compose service `hipica_scheduler` wired in; API mount of
  `./data:/app/data` flipped to rw (the API now persists Programas
  under `data/raw/Maroñas/`).
- [x] Tesseract + LibreOffice added to `docker/api.Dockerfile`;
  `pytesseract`, `Pillow`, `APScheduler` added to `requirements.txt`.

End-to-end verified on 2026-06-19:
- `/predict_program` returns 9 races with OCR 9/9 correct
  (2000, 1100, 1200, 1000, 1200, 1400, 1600, 1100, 1300 mts).
- Scheduler logs: `rt=1 date=2026-06-19 → 9 races (model=mlflow v1)`.
- 2026-06-18 (no Programa) → 404 with clean detail message.

---

## 3. Hardening / nice-to-have

These are **not** required by the rubric but raise the polish level:

- [x] **LICENSE file** — MIT License added at repo root.
- [x] **GitHub Actions CI** — `.github/workflows/ci.yml` runs `pytest`
  on push/PR to `main` (Ubuntu, Python 3.10, pip cache, Playwright
  browser download skipped).
- [x] **`/predict_explain` endpoint** — returns prediction +
  base value + top-k SHAP contributions via
  `booster.predict(..., pred_contribs=True)` (same workaround as
  the SHAP notebook). Smoke-tested via `TestClient`.
- [x] **Calibration plot** — saved to
  `reports/figures/14_calibration.png` and added as §7.1 in the
  informe. Reliability gap < 0.02 in the operating range.
- [x] **Build the Streamlit Docker image and bring the full stack up live** —
  `hipica-ml/api:latest` and `hipica-ml/streamlit:latest` built;
  pila completa levantada con `POSTGRES_PORT=15432 API_PORT=18000
  docker compose up -d`; screenshots end-to-end en
  `reports/figures/15_streamlit_ui.png` y `16_streamlit_predictions.png`.
- [x] **MLflow Model Registry exercised against Postgres backend** —
  modelo `trifecta-classifier` v1 registrado y promovido a
  `Production` contra Postgres; API verificada via `/health`
  reportando `model_name=mlflow, model_version=1`.

---

## 4. Suggested execution order

| Day | Task | Expected output |
|---|---|---|
| 1 | EDA notebook (§1.1) + figures saved | `notebooks/01_eda.ipynb`, 5 PNGs |
| 1 | SHAP notebook (§2.1) | `notebooks/02_explainability.ipynb`, 3 plots |
| 2 | Optuna tuning (§2.2) | `src/training/tune.py`, MLflow run with best params |
| 2 | Feature selection (§2.3) | `notebooks/03_feature_selection.ipynb`, decision |
| 3 | DVC for data (§2.4) | `data/processed/history.parquet.dvc` |
| 3 | LICENSE + CI (§3) | `LICENSE`, `.github/workflows/test.yml` |
| 4-5 | Write the report (§1.2) | `informe.pdf` |
| 6 | Final review, package zip, submit | `entrega.zip` ≤ 40 MB |

> Order rationale: EDA first because the report needs its figures. SHAP and
> Optuna next because they generate report material. DVC and LICENSE are
> mechanical. Report last because it depends on all the above.

---

## 5. Out of scope (do not pursue unless asked)

- ❌ Image data / CNNs — the PDF says "imágenes y/o datos tabulares" and we
  chose tabular only. That is allowed.
- ❌ AutoML — would auto-trigger Streamlit as mandatory (already done) but
  would not move the grade.
- ❌ Quantization / pruning / distillation — not meaningful for XGBoost on
  tabular data.
- ❌ AWS deployment — recommended but not required, and Docker Compose counts.

---

## 6. Submission checklist (the day before deadline)

- [ ] All electives implemented ≥ 3 (we will have 5-6).
- [ ] `pytest` green.
- [ ] README updated with how to reproduce every step.
- [ ] `informe.pdf` finalised, uses Copilot/Claude declaration section.
- [ ] Zip contains: `informe.pdf` + repo snapshot (no `data/raw/`, no
  `mlruns/`, no `.venv/`).
- [ ] Total size ≤ 40 MB.
- [ ] Repo is **public** OR ready to invite the docente if private.
- [ ] Submitted on `gestion.ort.edu.uy` before 21:00 on 2026-07-15.
