"""kNN Router Lab — interactive Streamlit demo.

Tabs:
  Demo        : route your own query with a lambda slider
  Benchmarks  : sample-complexity curve (kNN vs Linear, Theorem 1)
  Locality    : UMAP view of the embedding space
  Why kNN?    : the locality property, explained
"""
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="kNN Router Lab", page_icon="🧭", layout="wide")

from src.config import (SUPPORT_SET_PATH, MODEL_POOL_PATH, EMBEDDING_MODEL,
                        K_NEIGHBORS, LAMBDA_PRESETS, SEED)
from src.router import KNNRouter
from src.baselines import LinearRouter, OracleRouter
from src import viz


# ── data loading (cached once per session) ──────────────────────────────
@st.cache_data
def load_support_set():
    rows = [json.loads(l) for l in open(SUPPORT_SET_PATH)]
    train = [r for r in rows if r["split"] == "train"]
    models = list(rows[0]["scores"].keys())
    return {
        "texts":   [r["text"] for r in train],
        "emb":     np.array([r["embedding"] for r in train], dtype=np.float32),
        "scores":  np.array([[r["scores"][m] for m in models] for r in train]),
        "costs":   np.array([[r["costs"][m] for m in models] for r in train]),
        "models":  models,
        "all":     rows,
    }


data = load_support_set()
c1, c2 = st.columns([3, 1])
c1.title("kNN Router Lab")
c2.caption(f"Embedding: `{EMBEDDING_MODEL}` · k = {K_NEIGHBORS} · seed {SEED}")

tab_demo, tab_bench, tab_locality, tab_why = st.tabs(
    ["Demo", "Benchmarks", "Locality", "Why kNN?"])

# ────────────────────────── Tab 1: Demo ──────────────────────────────────
with tab_demo:
    left, right = st.columns([1, 1])
    with left:
        query = st.text_area("Your query",
            value="Prove that the square root of 2 is irrational")
        lam_label = st.select_slider(
            "Cost ↔ performance preference (λ, normalized by c_max)",
            options=list(LAMBDA_PRESETS), value="Balanced")
        lam = LAMBDA_PRESETS[lam_label]
        k = st.slider("k neighbors", 10, 200, K_NEIGHBORS, 10)
        route_clicked = st.button("Route!", type="primary")

    if route_clicked:
        from src.embedding_service import encode
        q_emb = encode([query], EMBEDDING_MODEL)
        router = KNNRouter(data["emb"], data["scores"], data["costs"],
                           data["models"], data["texts"])
        res = router.route(q_emb[0], lam=lam, k=k, return_neighbors=True)

        with right:
            st.subheader(f"➡️ Routed to: `{res.selected_model}`")
            st.plotly_chart(viz.utility_bars(res.utility_by_model,
                                             res.selected_model),
                            use_container_width=True)
        st.subheader(f"Top-{min(k, 15)} neighbors that voted")
        st.dataframe(pd.DataFrame(res.neighbors).head(15),
                     use_container_width=True, hide_index=True)

# ────────────────────── Tab 2: Benchmarks ────────────────────────────────
with tab_bench:
    st.subheader("Sample complexity: how much training data does each router need?")
    from src.metrics import sample_complexity_curve
    fractions = st.multiselect("Support-set fractions",
        [0.05, 0.10, 0.25, 0.50, 1.00], default=[0.10, 0.25, 0.50, 1.00])
    if st.button("Run experiment"):
        rows_all = data["all"]
        test = [r for r in rows_all if r["split"] == "test"]
        df = sample_complexity_curve(
            None,
            {"kNN": KNNRouter, "Linear": LinearRouter},
            data["emb"], data["scores"], data["costs"],
            data["models"], data["texts"],
            lambdas=[0.5], fractions=fractions, seed=SEED)
        st.plotly_chart(viz.sample_complexity_plot(df), use_container_width=True)
        st.caption("kNN should plateau earlier — that is Theorem 1 in action.")

# ────────────────────── Tab 3: Locality ──────────────────────────────────
with tab_locality:
    st.subheader("Queries that are close in embedding space prefer the same model")
    with st.spinner("Computing UMAP projection…"):
        import umap
        proj = umap.UMAP(n_neighbors=15, random_state=SEED).fit_transform(data["emb"][:1500])
    best = [data["models"][i] for i in np.argmax(data["scores"][:1500], axis=1)]
    st.plotly_chart(viz.scatter_2d(proj, best, data["models"]),
                    use_container_width=True)

# ────────────────────── Tab 4: Why kNN? ──────────────────────────────────
with tab_why:
    st.markdown("""
### The core idea (paper §7)

Model performance exhibits **δ-locality**: if two queries are close in
embedding space, every model scores them similarly. Formally:

> d(x₁, x₂) < δ  ⟹  |u(x₁, m) − u(x₂, m)| < ε(δ)

So the k nearest neighbors of a query are a nearly-unbiased estimate of how
each model would perform on it — no training required.

### Why this beats parametric routers (Theorem 1)

| Router | Samples needed for regret O(ε(δ)) |
|---|---|
| kNN | Θ(C/δᵈ · log(1/α)) — logarithmic in confidence |
| Parametric (L layers) | Ω(L/ε(δ)²) — quadratic in accuracy |

With a low-dimensional, high-locality embedding space, kNN needs far less
data — which is exactly what the Benchmarks tab demonstrates.
    """)
