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
- ✅ **v5-datafix-tuned model** (2026-07-11): test ROC-AUC **0.7171**, PR-AUC **0.6538**, F1@0.5 0.4224, precision@0.5 **0.765**. Rebuilt after fixing two loader bugs (pre-2019 Crystal Reports column drift + 2021-2023 leader-row kg column). Dataset grew from 53 k → **98 623** rows; `n_field` mean 39.2 → 13.0. See `CLAUDE.md` §8.7.
- ✅ EDA notebook (`notebooks/01_eda.ipynb`) — 8 figures saved.
- ✅ SHAP notebook (`notebooks/02_explainability.ipynb`) — 4 figures saved.
- ✅ Feature selection (`notebooks/03_feature_selection.ipynb`) — dropped 3 raw features, ROC-AUC +0.0018.
- ✅ DVC — `data/processed/history.parquet` tracked, local remote at `~/.dvc-store`, `dvc pull` round-trip verified.
- ✅ Report — `reports/informe.md` (19 sections) rendered to `reports/informe.pdf` (24 pages, 1.46 MB) via WeasyPrint.
- ✅ Calibration plot, `/predict_explain` endpoint, full Docker stack live with Streamlit screenshots, MLflow Registry on Postgres exercised end-to-end.
- ✅ Live race-day predictions — `src/ingestion/program.py` (Programa scrape + Tesseract OCR of distance badges) + `POST /predict_program` endpoint + Streamlit "🗓️ Race day" tab. End-to-end verified on 2026-06-19 (9 races, OCR 9/9 correct).
- ✅ Per-horse SHAP explanations in the UI — every Race-day row has a "🔍 Explicar" button that calls `/predict_explain` and renders the top contributions as a green/red Altair bar chart inside the same race block. Bug fixed: predictions are now persisted in `st.session_state["prog_data"]` so per-race button reruns don't wipe the field; also switched both date pickers to `America/Montevideo` so the default doesn't roll a day too early when the container clock is in UTC.
- ✅ Daily scheduler — `scheduler/main.py` (APScheduler 06:30 UY) + `docker/scheduler.Dockerfile` + compose service `hipica_scheduler`. Pre-warms the API cache for today + tomorrow on every configured racetrack.
- ✅ **Cloud deployment on AWS Elastic Beanstalk** — `api + streamlit` stack live on `t3.small` (Docker running on AL2023) in the AWS Academy Learner Lab default VPC. Streamlit public on port 80, FastAPI Swagger on port 8080. Deploy artifacts: root `docker-compose.yml` (EBS-facing), `docker-compose.local.yml` (renamed local stack), `.ebignore`, `.ebextensions/01-open-api-port.config`. URLs: <http://hipica-ml-prod.eba-d63jdkhp.us-east-1.elasticbeanstalk.com> and `:8080/docs`.
- ✅ **Hyperparameter tuning (v4-tuned)** — 50 Optuna trials done (~9 min CPU). Test ROC-AUC 0.704 → **0.7093** (+0.005), PR-AUC 0.634 → **0.6428** (+0.009), overfit gap 0.145 → **0.109**. MLflow parent run `bfbdada5deec4c98bbf4b519dc4642d1`.
- ✅ **Hyperparameter tuning (v5-datafix-tuned)** — 50 fresh Optuna trials on the rebuilt dataset (~16 min CPU). Test ROC-AUC **0.7171** (+0.008 vs v4-tuned), PR-AUC **0.6538** (+0.011), precision@0.5 **0.765**. Winning params: `n_estimators=850`, `max_depth=8`, `lr=0.01229`, `min_child_weight=4`, `reg_lambda=4.44`, `reg_alpha=1.87`, `subsample=0.83`, `colsample_bytree=0.85`, `gamma=2.25`. Threshold sweep: F1-optimal cutoff on val is **0.25** (F1=0.587, P=0.489, R=0.732). Artifacts: [`models/trifecta_pipeline_tuned/`](models/trifecta_pipeline_tuned/), promoted to [`models/trifecta_pipeline/`](models/trifecta_pipeline/). Rollback backup at [`models/trifecta_pipeline_v4tuned_predatafix/`](models/trifecta_pipeline_v4tuned_predatafix/).

---

## 1. Mandatory — still missing

### 1.1 EDA notebook — ✅ done
**Why**: The PDF explicitly requires `Análisis Exploratorio de los Datos` and
the rubric grades on it ("Análisis Exploratorio y Preparación de los datos").

**Where**: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (647 KB,
executed end-to-end) and 8 PNGs under [`reports/figures/`](reports/figures/).

**Content checklist**:
- [x] Load `data/processed/history.parquet` (98 398 rows).
- [x] Dataset shape, date range, missingness per column
  → [01_missingness.png](reports/figures/01_missingness.png).
- [x] Label balance (`in_trifecta` ≈ 35.76 % positive)
  → [02_label_balance.png](reports/figures/02_label_balance.png).
- [x] Distribution of `kg`, `distance_m`, `n_field`, `horse_age`
  → [03_numeric_distributions.png](reports/figures/03_numeric_distributions.png).
- [x] Top racetracks, top jockeys, top trainers by row count
  → [04_top_tracks_jockeys.png](reports/figures/04_top_tracks_jockeys.png).
- [x] Career-runs distribution per horse (long tail)
  → [05_runs_per_horse.png](reports/figures/05_runs_per_horse.png).
- [x] Correlation heatmap of numeric features
  → [07_correlation.png](reports/figures/07_correlation.png).
- [x] Time series: races per year, label rate per year (drift check)
  → [06_temporal_trends.png](reports/figures/06_temporal_trends.png).
- [x] Class balance per racetrack and per distance bucket
  → [08_balance_by_segment.png](reports/figures/08_balance_by_segment.png).
- [x] 8 plots saved to `reports/figures/` and embedded in the report.

### 1.2 Written report (`informe.pdf`) — ✅ done
**Why**: Section "Entrega" of the PDF — "incluye un informe en conjunto con el
código base".

**Where**: [`reports/informe.md`](reports/informe.md) (37 KB, 19 sections)
rendered to [`reports/informe.pdf`](reports/informe.pdf) (1.77 MB) via
WeasyPrint using the CSS embedded in [`reports/informe.html`](reports/informe.html).

**Sections shipped**:
1. Portada + resumen ejecutivo.
2. Definición del problema y target (`in_trifecta`).
3. Dataset y EDA — embeds the 8 figures from §1.1.
4. Arquitectura del sistema (con diagrama).
5. Feature engineering — contrato de 35 features + anti-leakage.
6. Modelo y entrenamiento — XGBoost, split temporal, MLflow.
7. Resultados — progresión v1 → v4 (+0.022 ROC-AUC, +0.057 F1).
8. API — endpoints, Pydantic, ejemplos.
9. Despliegue — **§9.1 Local (Docker Compose)** + **§9.2 Nube (AWS EBS)**.
10. Trazabilidad — MLflow + DVC + joblib fallback.
11. Explicabilidad — SHAP top-5 + waterfall.
12. Selección de features (aggressive pass +0.0018 ROC-AUC).
13. Optuna — smoke 3 trials en paridad con v4.
14. UI Streamlit + Race-day tab con OCR.
15. Tests y CI (7/7 passing).
16. Bugs encontrados — incluye §16.7 sobre el episodio LibreOffice en EBS.
17. Trade-offs y mejoras posibles.
18. Uso de IA generativa (declaración GitHub Copilot / Claude).
19. Anexo — comandos de reproducción.

**Format**: PDF, 1.77 MB. Well under the 40 MB submission cap.

### 1.3 Deployment platform — ✅ done
The PDF says **AWS is recommended, not required**. We went the recommended
route and shipped the `api + streamlit` stack to **AWS Elastic Beanstalk**
(Docker on AL2023) inside the AWS Academy Learner Lab, following the
instructor's `Ejemplo de despliegue EBS.pdf`.

- [x] `docker-compose.yml` at the repo root, only `api` (internal) + `streamlit`
  (host port 80). `docker-compose.local.yml` keeps the full 5-service dev stack.
- [x] `.ebignore` — excludes `data/raw/` (1.3 GB), `notebooks/`, `mlruns/`,
  PDFs. Final deploy bundle ≈ 35 MB.
- [x] `.ebextensions/01-open-api-port.config` — opens TCP 8080 on the
  EBS-managed security group so `/docs` is publicly reachable.
- [x] Env created with the Learner Lab-friendly flags:
  `--instance_type t3.small --single --instance_profile LabInstanceProfile
  --service-role LabRole --vpc.id <default-VPC> --vpc.ec2subnets <3 public subnets>
  --vpc.publicip`.
- [x] End-to-end verified: `/health`, `/predict_online`, `/predict_batch`
  respond < 1 s from the public URL. `/predict_program` works after restoring
  `libreoffice-calc` in `docker/api.Dockerfile` (regression from image-slim pass).
- [x] Documented as `§9.2 Nube — AWS Elastic Beanstalk` in `reports/informe.md`
  and PDF regenerated.

---

## 2. Electives — recommended additions

The minimum is 3 electives. We already have 3 done (scraper, ML traceability
partial, Streamlit). Adding 2-3 more raises the grade.

Final count at submission time: **7 electives done** — scraper,
trazabilidad (MLflow + DVC), Streamlit UI, SHAP explainability,
feature selection, Optuna (script ready + smoke), and **AWS Elastic
Beanstalk cloud deployment**.

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

### 2.2 Hyperparameter tuning with Optuna — ✅ done
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
- [x] **Full run (50 trials)** — done 2026-07-11 (v4-tuned, ~9 min CPU) and
  re-run **2026-07-11** on the datafix rebuild (**v5-datafix-tuned**, ~16 min
  CPU on 12 cores, `tree_method=hist`, `nthread=-1`). Both used experiment
  `trifecta-classifier`, local file store.
  MLflow parent runs: v4-tuned `bfbdada5deec4c98bbf4b519dc4642d1`,
  v5-datafix baseline `57660386534b4990a3df363350495e45`.
- [x] Compare final tuned model to v4 (v4-tuned):
  - **ROC-AUC** 0.7040 → **0.7093** (+0.0053)
  - **PR-AUC** 0.6340 → **0.6428** (+0.0088)
  - **Log-loss** 0.5920 → **0.5856** (−0.0064)
  - **Brier** 0.2030 → **0.2003** (−0.0027)
  - Overfit gap (train − test ROC) 0.145 → **0.109** (−0.036)
  - F1@0.5 dips (0.4530 → 0.4418) because the tuned model is more
    conservative at that cutoff; threshold sweep on held-out val placed the
    F1-optimal cutoff at **0.30**, giving test F1 = **0.5708** (P=0.49, R=0.68).
- [x] Compare **v5-datafix-tuned** to v4-tuned (after the loader fix):
  - **ROC-AUC** 0.7093 → **0.7171** (+0.0078)
  - **PR-AUC** 0.6428 → **0.6538** (+0.0110)
  - **Precision @0.5** 0.7140 → **0.7650** (+0.0510)
  - **Recall @0.5** 0.3198 → 0.2917 (−0.0281)
  - **F1 @0.5** 0.4418 → 0.4224 (−0.0194) — shipped cutoff is more conservative;
    val-fold sweep places the F1-optimal cutoff at **0.25** (F1=0.587, P=0.489,
    R=0.732). See `reports/threshold_sweep_v5_datafix.csv`.
  - Lift comes entirely from data quality — no new features. Cleaner
    leader-row detection recovered ~45 k historical rows from 2021-2023 that
    the v4 loader silently dropped, and the plausibility filter removed
    ~20 k noise rows from the pre-2019 column drift.
- [x] Winning hyperparams (v4-tuned): `n_estimators=1150`, `max_depth=7`,
  `learning_rate=0.0194`, `min_child_weight=10`, `reg_lambda=2.13`,
  `reg_alpha=1.84`, `subsample=0.90`, `colsample_bytree=0.66`, `gamma=1.51`.
- [x] Winning hyperparams (**v5-datafix-tuned**): `n_estimators=850`, `max_depth=8`,
  `learning_rate=0.01229`, `min_child_weight=4`, `reg_lambda=4.44`, `reg_alpha=1.87`,
  `subsample=0.83`, `colsample_bytree=0.85`, `gamma=2.25`. Pattern: **fewer but
  deeper trees, slower learning, more regularised**.
- [x] Latency impact: v4-tuned had 1150 trees, v5-datafix-tuned has 850 — so
  inference is slightly *faster* per row than v4-tuned on the same 12-horse
  batches. No user-visible regression.

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
- [x] **AWS Elastic Beanstalk cloud deployment** — see §1.3.

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
- ~~AWS deployment~~ — done, see §1.3.

---

## 6. Submission checklist (the day before deadline)

- [x] All electives implemented ≥ 3 (we have **7**: scraper, trazabilidad,
  Streamlit, SHAP, feature selection, Optuna, AWS EBS deployment).
- [ ] `pytest` green.
- [ ] README updated with how to reproduce every step, including the new
  `docker compose -f docker-compose.local.yml up -d` for local dev and
  `eb deploy` for the cloud path.
- [x] `informe.pdf` finalised (19 sections, 1.77 MB), Copilot/Claude
  declaration in §18.
- [ ] Zip contains: `informe.pdf` + repo snapshot (no `data/raw/`, no
  `mlruns/`, no `.venv/`, no `.elasticbeanstalk/logs/`).
- [ ] Total size ≤ 40 MB.
- [x] Repo is **public**: <https://github.com/MathiasGili/hipica-ml>.
- [ ] `eb terminate hipica-ml-prod` after the docente grades the live URL, to
  release the EC2 + EIP and preserve the 50 USD Academy budget.
- [ ] Submitted on `gestion.ort.edu.uy` before 21:00 on 2026-07-15.
