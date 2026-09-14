# 🧭 kNN Router Lab

**An interactive demonstration of how a simple k-Nearest Neighbors algorithm can intelligently route queries between Large Language Models — no training required.**

![Live Demo](https://img.shields.io/badge/🔗%20Live%20Demo-Streamlit%20Cloud-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Paper](https://img.shields.io/badge/Based%20on-arXiv%3A2505.12601-green)

---

## 📖 For Everyone: What Does This Project Do?

Imagine you run a customer-service platform, and you have access to **five different AI models**. Some are cheap and fast but not very smart. Others are expensive but brilliant. Every time a user sends a question, you must pick **one** model to answer it.

Picking the expensive model every time blows your budget. Picking the cheap one every time gives bad answers. So... how do you choose *per question*?

This project implements and demonstrates a surprisingly simple answer from a 2025 research paper: **let the question's neighbors decide.**

> **The core idea:** convert the question into a numerical "fingerprint" (an *embedding*), find the most similar questions you've seen before, and check which model performed best on *those*. That model probably performs well on this new question too.

That's it. No neural network training, no reinforcement learning, no complex infrastructure. Just a well-organized lookup table and a distance calculation. And the paper shows this simple approach **often beats sophisticated AI routers**.

This repository contains a working, deployed web application where you can type your own questions and watch the router decide — with full transparency about *why* it chose what it chose.

---

## 🎯 The Problem: LLM Routing (Technical Overview)

Modern deployments have access to a pool of models with different **capabilities, costs, and latencies**. *Routing* means selecting one model per query to maximize a utility function:

```
m* = argmax over m of  [ score(x, m) − λ · c_max · cost(x, m) ]
```

- `score(x, m)` — how well model `m` answers query `x` (accuracy, win rate...)
- `cost(x, m)` — the price of invoking model `m` on `x` (API tokens)
- `λ` (lambda) — a dial controlling the trade-off: **high λ = cheap-first, low λ = quality-first**

State-of-the-art routers learn this mapping with trained neural networks. This project asks: **is that complexity necessary?**

---

## 💡 The Key Insight: δ-Locality

The paper proves this works because model performance exhibits **locality in embedding space**:

> If two queries are close together in the embedding space (`d(x₁, x₂) < δ`), then every model scores them similarly: `|u(x₁, m) − u(x₂, m)| < ε(δ)`.

In plain terms: **similar questions get similar answers from the same model.** So the *k* most similar past questions are a nearly-unbiased estimate of how each model would perform on a new one.

This gives kNN a **sample-complexity advantage** (paper's Theorem 1):

| Router | Training samples needed for regret O(ε(δ)) |
|---|---|
| **kNN** | Θ(C/δᵈ · log(1/α)) — *logarithmic* in confidence |
| Parametric (L layers) | Ω(L/ε(δ)²) — *quadratic* in accuracy |

With a good embedding space, kNN needs far less data — which the **Benchmarks tab of the app demonstrates experimentally**.

---

## 🗂️ How the System Works

```
Your query
    │
    ▼
[1] Embedding        → convert text to a 384-dim vector (MiniLM, runs on CPU)
    │
    ▼
[2] kNN Retrieval    → find the 100 most similar past queries (cosine similarity)
    │
    ▼
[3] Utility Estimate → for each candidate model:
    │                    û(x, m) = mean_score − λ · c_max · mean_cost
    ▼
[4] Selection        → pick the model with the highest utility
    │
    └──► Show: chosen model, utility comparison, the neighbors that voted
```

Everything is **transparent and interpretable** — you can see exactly which past queries influenced the decision. That's a feature no black-box router offers.

---

## 🏗️ Project Structure

```
knn-router-lab/
├── app.py                       # Streamlit web application (4 tabs)
├── requirements.txt             # Python dependencies
├── packages.txt                 # System libs for Streamlit Cloud (libgomp1)
├── README.md                    # ← you are here
├── scripts/
│   └── generate_support_set.py  # Builds the support set + model pool (run once)
├── notebooks/
│   └── (space for your experiments)
└── src/
    ├── config.py                # Constants: k, λ presets, seeds, splits
    ├── embedding_service.py     # Text → embedding encoder (session-cached)
    ├── knn_index.py             # kNN search (FAISS, numpy fallback)
    ├── router.py                # KNNRouter — the star of the show
    ├── baselines.py             # LinearRouter & OracleRouter (comparisons)
    ├── metrics.py               # Pareto AUC + sample-complexity curves
    ├── viz.py                   # Plotly charts
    └── data/
        ├── model_pool.json      # 5 simulated models with cost profiles
        └── support_set.jsonl    # 3,000 queries with scores/costs per model
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **CPU-only** | Free deployment tiers (Streamlit Cloud, HF Spaces) have no GPU |
| **Small embedding model** (`all-MiniLM-L6-v2`, ~90MB) | Fast enough for interactive demos, free to download |
| **FAISS with numpy fallback** | Robustness in minimal environments |
| **Pre-computed support set** | The app never embeds 3,000 items at runtime — only your live query |
| **Uniform router interface** | `route(query_emb, lam) → RouteResult` for fair kNN vs. baseline comparison |
| **Reproducible** | Fixed seeds, 70/10/20 splits — same protocol as the paper |

---

## 🚀 Running Locally

```bash
# 1. Clone and enter the project
git clone https://github.com/SEU_USUARIO/knn-router-lab.git
cd knn-router-lab

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. (Re)generate the support set — only needed if you changed the data logic
python scripts/generate_support_set.py

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### The four tabs

| Tab | What it shows |
|---|---|
| 🎯 **Demo** | Type any query, slide λ between *cheap-first* and *quality-first*, watch the router choose — with the top-k neighbors that voted |
| 📊 **Benchmarks** | The sample-complexity experiment: how much training data kNN needs vs. a parametric baseline (Theorem 1, visualized) |
| 🔬 **Locality** | A UMAP projection of the embedding space, colored by each region's best model — see δ-locality with your own eyes |
| ❓ **Why kNN?** | The theory, explained: δ-locality, the utility formula, and the sample-complexity table |

---

## ☁️ Deployment (Streamlit Community Cloud)

This app is deployed for free on [Streamlit Community Cloud](https://share.streamlit.io).

**One-time setup:**

1. Push this repository to a **public** GitHub repo
2. Log in at [share.streamlit.io](https://share.streamlit.io) with GitHub
3. **Create app** → select the repo → main file path: `app.py` → **Deploy**
4. Wait ~5–10 min (first build installs ML dependencies; first query downloads the embedding model)

**Every `git push` to `main` triggers an automatic redeploy.**

> **Note on the free tier:** apps sleep after ~7 days without visitors. The first visit after sleep takes ~2 minutes (container restart + model download). Consider rebooting from the *Manage app* panel before sharing the link in your portfolio.

### Alternative: Hugging Face Spaces

Also free. Create a Space with SDK = **Streamlit**, push the repo, done. The `README.md` header metadata for HF Spaces is documented in the deployment notes of this project.

---

## 🧪 The Data: A Honest Note

The support set ships with **synthetic data**: 3,000 queries across 5 domains (math, code, creative, QA, reasoning), with per-model scores and costs simulated so that different models genuinely specialize in different domains — creating the locality structure the method relies on.

This is deliberate: it keeps the demo fully self-contained, reproducible, and API-cost-free. To make the routing "real" end-to-end, one line in `scripts/generate_support_set.py` swaps the placeholder embeddings for the real encoder (`embedding_service.encode`), and real evaluation scores from public benchmarks (RouterBench, HELM, vHELM) can replace the simulated ones.

---

## 📚 References

- Li, Y. (2025). *Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers.* arXiv:2505.12601 — the paper this project is based on
- Ong et al. (2024). *RouteLLM: Learning to Route LLMs from Preference Data.* ICLR — the matrix-factorization baseline family
- Hu et al. (2024). *RouterBench: A Benchmark for Multi-LLM Routing System.* — evaluation protocol inspiration
- Reimers & Gurevych (2019). *Sentence-BERT* — the embedding model used

---

## 📝 License

MIT — use it, learn from it, show it in your portfolio.

---

*Built as a portfolio project demonstrating applied understanding of embedding spaces, non-parametric methods, and the cost/performance trade-offs of real LLM systems.*
