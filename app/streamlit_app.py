"""Streamlit UI for the Trifecta classifier.

This app talks to the FastAPI service via plain HTTP — it does *not* import
the model directly. That keeps the UI container lightweight and lets us
swap the inference backend without touching the UI.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

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


# ---------------------------------------------------------------------------
# Inputs — race context + entries
# ---------------------------------------------------------------------------
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

st.subheader("1) Race context")
c1, c2, c3 = st.columns(3)
with c1:
    race_date = st.date_input("Race date", value=date.today())
with c2:
    rt_id = st.selectbox(
        "Racetrack",
        options=list(RACETRACKS.keys()),
        format_func=lambda i: f"{i} — {RACETRACKS[i]}",
        index=0,
    )
with c3:
    distance_m = st.number_input("Distance (m)", min_value=600, max_value=4000,
                                 value=1600, step=100)

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
)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
st.subheader("3) Predictions")
mode = st.radio(
    "Mode",
    options=["batch (recommended)", "online (one horse)"],
    horizontal=True,
)

if st.button("Predict", type="primary", use_container_width=True):
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
            st.dataframe(preds, use_container_width=True)
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
            st.dataframe(df[["horse_name", "p_trifecta", "model_version"]],
                         use_container_width=True)
    except requests.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except requests.RequestException as exc:
        st.error(f"Network error: {exc}")
