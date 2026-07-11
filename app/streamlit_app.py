"""Streamlit UI for the Trifecta classifier.

Two modes:

* **Race day** — pick a published Maroñas race day, the API scrapes the
  Programa, OCRs the distance badges and predicts every race.
* **Manual** — fill a custom field by hand (the original demo flow).

The UI is a thin HTTP client over the FastAPI service. It does NOT load
the model directly: that keeps the container lightweight and lets us
swap inference backends without touching the UI.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
UY_TZ = ZoneInfo("America/Montevideo")


def _today_uy() -> date:
    """Today in the local racing calendar's timezone (Montevideo).

    Avoids a UTC-vs-UY off-by-one when the container's clock has already
    rolled past midnight UTC but UY is still on the previous day.
    """
    return datetime.now(UY_TZ).date()

st.set_page_config(
    page_title="Trifecta Classifier — Maroñas",
    page_icon="🏇",
    layout="wide",
)

st.title("🏇 Trifecta Classifier")
st.caption(
    "Predicts the probability that a horse finishes in the Trifecta "
    "(1st, 2nd or 3rd) using historical race data from "
    "hipica.maronas.com.uy."
)


# ---------------------------------------------------------------------------
# Sidebar — health + model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Service")
    st.code(API_URL)
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"Online — model: {h.get('model_name')} v{h.get('model_version') or '—'}")
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")
        st.stop()


RACETRACKS = {
    1: "Maroñas",
    13: "Las Piedras",
    4: "Colonia",
    9: "Florida",
    16: "Melo",
    21: "Paysandú",
    22: "Rocha",
    8: "Flores",
}

# Friendly Spanish labels for the SHAP feature names. The keys are the
# canonical feature names produced by the FE pipeline (and one-hot OHE
# variants like ``racetrack_id_1.0``). Anything not in the map falls back
# to the raw name.
FEATURE_LABELS: dict[str, str] = {
    # Pass-through / per-entry
    "weight_kg": "Peso del jinete (kg)",
    "weight_kg_zscore_in_race": "Peso vs. resto del field (z-score)",
    "n_field": "Tamaño del field",
    "distance_m": "Distancia (m)",
    "post_position": "Partidor",
    "horse_age": "Edad del caballo",
    "sex_code_M": "Sexo: Macho",
    "sex_code_H": "Sexo: Hembra",
    "racetrack_id_1.0": "Track: Maroñas",
    "racetrack_id_4.0": "Track: Colonia",
    "racetrack_id_8.0": "Track: Flores",
    "racetrack_id_9.0": "Track: Florida",
    "racetrack_id_13.0": "Track: Las Piedras",
    "racetrack_id_16.0": "Track: Melo",
    "racetrack_id_21.0": "Track: Paysandú",
    "racetrack_id_22.0": "Track: Rocha",
    # Per-horse career history
    "career_runs": "Carreras totales",
    "career_wins": "Victorias totales",
    "career_places": "2dos puestos totales",
    "career_shows": "3ros puestos totales",
    "career_win_rate": "% de victorias (carrera)",
    "career_show_rate": "% en trifecta (carrera)",
    "year_runs": "Carreras (último año)",
    "year_wins": "Victorias (último año)",
    "year_places": "2dos puestos (último año)",
    "year_shows": "3ros puestos (último año)",
    "year_win_rate": "% de victorias (último año)",
    "year_show_rate": "% en trifecta (último año)",
    "last_finish_pos": "Puesto en última carrera",
    "avg_finish_last3": "Puesto promedio (últimas 3)",
    "best_finish_last3": "Mejor puesto (últimas 3)",
    "rest_days": "Días desde la última carrera",
    "days_since_last_win": "Días desde última victoria",
    # Track / distance fit
    "track_runs": "Carreras previas en el track",
    "track_show_rate": "% en trifecta en el track",
    "dist_bucket_runs": "Carreras previas a esta distancia",
    "dist_bucket_show_rate": "% en trifecta a esta distancia",
    "dist_diff_from_avg": "Distancia hoy vs. promedio del caballo",
    "weight_change_from_last": "Cambio de peso vs. última carrera",
    # Market signal
    "dividend_career_mean": "Dividend promedio (carrera)",
    "dividend_last3_mean": "Dividend promedio (últimas 3)",
    "dividend_career_min": "Mejor dividend de su carrera",
    # Cross-horse jockey
    "jockey_career_runs": "Carreras del jockey",
    "jockey_career_show_rate": "% del jockey en trifecta",
}


def _pretty_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name)

tab_program, tab_manual = st.tabs(["🗓️ Race day (scrape)", "✏️ Manual"])


# ===========================================================================
# Tab 1: Race day — scrape the Programa and predict every race
# ===========================================================================
with tab_program:
    st.subheader("Pick a published race day")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        prog_date = st.date_input(
            "Race date",
            value=_today_uy() + timedelta(days=1),
            key="prog_date",
        )
    with c2:
        prog_track = st.selectbox(
            "Racetrack",
            options=list(RACETRACKS.keys()),
            format_func=lambda i: f"{i} — {RACETRACKS[i]}",
            index=0,
            key="prog_track",
        )
    with c3:
        prog_force = st.checkbox(
            "Force refresh", value=False,
            help="Re-download the Programa even if cached",
        )

    st.caption(
        "The API will download the published Programa (DocumentType=1), OCR "
        "the distance badges, and rank every horse in every race."
    )

    if st.button("Cargar y predecir", type="primary", key="prog_btn"):
        with st.spinner("Scraping + OCR + predicting…"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict_program",
                    json={
                        "race_date": prog_date.isoformat(),
                        "racetrack_id": int(prog_track),
                        "force_refresh": prog_force,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                # Persist across reruns so the per-race Explicar buttons keep
                # access to the predictions (clicking any Streamlit button
                # triggers a full script re-run and drops local variables).
                st.session_state["prog_data"] = resp.json()
                # New race day → drop any cached SHAP explanations.
                for k in list(st.session_state.keys()):
                    if k.startswith("exp_") and not k.startswith("exp_pick_") and not k.startswith("exp_btn_"):
                        st.session_state.pop(k, None)
            except requests.HTTPError as exc:
                st.error(
                    f"API error {exc.response.status_code}: "
                    f"{exc.response.text}"
                )
                st.session_state.pop("prog_data", None)
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
                st.session_state.pop("prog_data", None)

    data = st.session_state.get("prog_data")
    if data is not None:
        st.success(
            f"Served by model **{data['model_name']} v{data['model_version'] or '—'}** "
            f"— {data['n_races']} carreras · "
            f"{RACETRACKS.get(data['racetrack_id'], data['racetrack_id'])} · "
            f"{data['race_date']}"
        )

        for race in data["races"]:
            header = f"Carrera {race['race_index']}"
            if race.get("post_time"):
                header += f"  ·  {race['post_time']}"
            header += f"  ·  {race['distance_m']} mts"
            st.markdown(f"### {header}")

            df = pd.DataFrame(race["predictions"])
            df = df.sort_values("p_trifecta", ascending=False).reset_index(drop=True)
            df["p_trifecta"] = df["p_trifecta"].astype(float)

            colA, colB = st.columns([3, 2])
            with colA:
                show = df[
                    ["rank", "post_position", "horse_name", "kg",
                     "horse_age", "sex_code", "jockey_name", "p_trifecta"]
                ].rename(columns={
                    "rank": "Puesto pred.",
                    "post_position": "Partidor",
                    "horse_name": "Caballo",
                    "kg": "Peso (kg)",
                    "horse_age": "Edad",
                    "sex_code": "Sexo",
                    "jockey_name": "Jockey",
                    "p_trifecta": "P(trifecta)",
                })
                st.dataframe(
                    show.style.format({"P(trifecta)": "{:.3f}", "Peso (kg)": "{:.1f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            with colB:
                chart_df = df.set_index("horse_name")["p_trifecta"]
                st.bar_chart(chart_df)

            # ---- SHAP explanation: click a horse, get top contributions
            exp_col1, exp_col2 = st.columns([3, 1])
            options = df["horse_name"].tolist()
            with exp_col1:
                picked = st.selectbox(
                    "Explicar",
                    options=options,
                    index=0,
                    key=f"exp_pick_{race['race_index']}",
                    label_visibility="collapsed",
                )
            with exp_col2:
                explain_clicked = st.button(
                    "🔍 Explicar",
                    key=f"exp_btn_{race['race_index']}",
                    use_container_width=True,
                )

            if explain_clicked:
                # Find the entry row for the picked horse
                row = df[df["horse_name"] == picked].iloc[0]
                entry_payload = {
                    "horse_name": picked,
                    "kg": float(row["kg"]),
                    "post_position": int(row["post_position"]) if pd.notna(row["post_position"]) else None,
                    "horse_age": int(row["horse_age"]) if pd.notna(row["horse_age"]) else None,
                    "sex_code": row["sex_code"] if pd.notna(row["sex_code"]) else None,
                    "jockey_name": row["jockey_name"] if pd.notna(row["jockey_name"]) else None,
                }
                race_payload = {
                    "race_date": data["race_date"],
                    "racetrack_id": int(data["racetrack_id"]),
                    "distance_m": int(race["distance_m"] or 1100),
                }
                try:
                    er = requests.post(
                        f"{API_URL}/predict_explain",
                        json={"race": race_payload, "entry": entry_payload, "top_k": 10},
                        timeout=15,
                    )
                    er.raise_for_status()
                    st.session_state[f"exp_{race['race_index']}"] = er.json()
                except requests.HTTPError as exc:
                    st.error(f"API error {exc.response.status_code}: {exc.response.text}")
                except requests.RequestException as exc:
                    st.error(f"Network error: {exc}")

            # If an explanation was previously requested for this race, render it
            exp = st.session_state.get(f"exp_{race['race_index']}")
            if exp is not None:
                # Verify it matches the currently picked horse
                contrib_df = pd.DataFrame(exp["top_contributions"])
                contrib_df["feature_label"] = contrib_df["feature"].map(_pretty_feature)
                # Sort by absolute contribution so the most influential land on top
                contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
                contrib_df = contrib_df.sort_values("abs_contribution", ascending=True)
                contrib_df["color"] = contrib_df["contribution"].apply(
                    lambda c: "↑ trifecta" if c > 0 else "↓ trifecta"
                )

                horse_label = exp["horse_name"]
                p_pct = exp["p_trifecta"] * 100
                bias_pct = 1 / (1 + 2.71828 ** -exp["base_value"]) * 100
                st.caption(
                    f"**{horse_label}** — p(trifecta) = **{p_pct:.1f}%** "
                    f"vs. bias del modelo (~{bias_pct:.1f}%). Cada barra es la "
                    f"contribución de esa feature al log-odds final."
                )

                try:
                    import altair as alt
                    chart = (
                        alt.Chart(contrib_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("contribution:Q", title="contribución (log-odds)"),
                            y=alt.Y("feature_label:N", sort=None, title=None),
                            color=alt.Color(
                                "color:N",
                                scale=alt.Scale(
                                    domain=["↑ trifecta", "↓ trifecta"],
                                    range=["#16a34a", "#dc2626"],
                                ),
                                legend=alt.Legend(title=None),
                            ),
                            tooltip=[
                                alt.Tooltip("feature_label:N", title="Feature"),
                                alt.Tooltip("feature:N", title="Nombre técnico"),
                                alt.Tooltip("value:Q", title="Valor", format=".3f"),
                                alt.Tooltip("contribution:Q", title="Contribución", format=".3f"),
                            ],
                        )
                        .properties(height=max(220, 28 * len(contrib_df)))
                    )
                    st.altair_chart(chart, use_container_width=True)
                except ImportError:
                    st.bar_chart(contrib_df.set_index("feature_label")["contribution"])

            st.divider()


# ===========================================================================
# Tab 2: Manual — fill a field by hand
# ===========================================================================
with tab_manual:
    st.subheader("1) Race context")
    c1, c2, c3 = st.columns(3)
    with c1:
        race_date = st.date_input("Race date", value=_today_uy(), key="man_date")
    with c2:
        rt_id = st.selectbox(
            "Racetrack",
            options=list(RACETRACKS.keys()),
            format_func=lambda i: f"{i} — {RACETRACKS[i]}",
            index=0,
            key="man_track",
        )
    with c3:
        distance_m = st.number_input(
            "Distance (m)", min_value=600, max_value=4000, value=1600, step=100,
            key="man_dist",
        )

    st.subheader("2) Field — one row per horse")
    default_entries = pd.DataFrame(
        [
            {"horse_name": "ALPHA", "kg": 55.0, "post_position": 1,
             "horse_age": 4, "sex_code": "M", "jockey_name": ""},
            {"horse_name": "BRAVO", "kg": 56.0, "post_position": 2,
             "horse_age": 5, "sex_code": "H", "jockey_name": ""},
            {"horse_name": "CHARLIE", "kg": 54.5, "post_position": 3,
             "horse_age": 4, "sex_code": "M", "jockey_name": ""},
        ]
    )
    edited = st.data_editor(
        default_entries,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "horse_name": st.column_config.TextColumn("Horse", required=True),
            "kg": st.column_config.NumberColumn("Kg", min_value=40, max_value=70, step=0.5, format="%.1f"),
            "post_position": st.column_config.NumberColumn("Post", min_value=1, max_value=25, step=1),
            "horse_age": st.column_config.NumberColumn("Age", min_value=2, max_value=20, step=1),
            "sex_code": st.column_config.SelectboxColumn("Sex", options=["M", "H"]),
            "jockey_name": st.column_config.TextColumn("Jockey (optional)"),
        },
        key="man_editor",
    )

    st.subheader("3) Predictions")
    mode = st.radio(
        "Mode",
        options=["batch (recommended)", "online (one horse)"],
        horizontal=True,
        key="man_mode",
    )

    if st.button("Predict", type="primary", use_container_width=True, key="man_btn"):
        entries = [
            {
                "horse_name": str(r.horse_name).strip(),
                "kg": float(r.kg),
                "post_position": int(r.post_position) if pd.notna(r.post_position) else None,
                "horse_age": int(r.horse_age) if pd.notna(r.horse_age) else None,
                "sex_code": (r.sex_code or None) if pd.notna(r.sex_code) else None,
                "jockey_name": (str(r.jockey_name).strip() or None) if pd.notna(getattr(r, "jockey_name", None)) else None,
            }
            for r in edited.itertuples(index=False)
            if str(r.horse_name).strip()
        ]
        if not entries:
            st.error("Please add at least one horse.")
            st.stop()

        race_payload = {
            "race_date": race_date.isoformat(),
            "racetrack_id": int(rt_id),
            "distance_m": int(distance_m),
        }

        try:
            if mode.startswith("batch"):
                resp = requests.post(
                    f"{API_URL}/predict_batch",
                    json={"race": race_payload, "entries": entries},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                preds = pd.DataFrame(data["predictions"])
                preds["p_trifecta"] = preds["p_trifecta"].astype(float)
                preds = preds.sort_values("p_trifecta", ascending=False).reset_index(drop=True)
                st.success(f"Served by model {data['model_name']} v{data['model_version'] or '—'}")
                show = preds.rename(columns={
                    "horse_name": "Caballo",
                    "p_trifecta": "P(trifecta)",
                    "rank": "Puesto pred.",
                })
                st.dataframe(
                    show.style.format({"P(trifecta)": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.bar_chart(preds.set_index("horse_name")["p_trifecta"])
            else:
                preds = []
                for entry in entries:
                    resp = requests.post(
                        f"{API_URL}/predict_online",
                        json={"race": race_payload, "entry": entry},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    preds.append(resp.json())
                df = pd.DataFrame(preds)
                df["p_trifecta"] = df["p_trifecta"].astype(float)
                df = df.sort_values("p_trifecta", ascending=False).reset_index(drop=True)
                show = df[["horse_name", "p_trifecta", "model_version"]].rename(columns={
                    "horse_name": "Caballo",
                    "p_trifecta": "P(trifecta)",
                    "model_version": "Modelo (v)",
                })
                st.dataframe(
                    show.style.format({"P(trifecta)": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
        except requests.HTTPError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"Network error: {exc}")
