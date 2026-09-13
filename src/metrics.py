"""Evaluation: Pareto-front AUC (paper §B.3.1) and sample-complexity curves."""
import numpy as np
import pandas as pd


def evaluate_at_lambdas(router, test_embeddings, test_scores, test_costs,
                        lambdas, k=100) -> pd.DataFrame:
    """Route each test query at each lambda; return realized (score, cost)."""
    records = []
    scores = np.asarray(test_scores, dtype=np.float32)
    costs = np.asarray(test_costs, dtype=np.float32)
    c_max = float(costs.max())
    for lam in lambdas:
        for i, q in enumerate(test_embeddings):
            r = router.route(q, lam=lam, k=k)
            j = router.models.index(r.selected_model)
            records.append({
                "lambda": lam,
                "query_id": i,
                "model": r.selected_model,
                "realized_score": float(scores[i, j]),
                "realized_cost": float(costs[i, j]),
                "utility": float(scores[i, j] - lam * c_max * costs[i, j]),
            })
    return pd.DataFrame(records)


def sample_complexity_curve(make_router, routers: dict, embeddings, scores,
                            costs, models, texts, lambdas, fractions,
                            seed=42) -> pd.DataFrame:
    """Train each router on a fraction of the support set, evaluate utility.

    Implements the paper's Theorem-1 experiment: how many samples does each
    router need? `routers` maps a name to a class with the KNNRouter signature.
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    test_idx = rng.choice(n, size=int(0.2 * n), replace=False)
    train_pool = np.setdiff1d(np.arange(n), test_idx)

    rows = []
    pool_size = len(train_pool)          # interpret fractions relative to the train pool
    for frac in fractions:
        sub = rng.choice(train_pool, size=max(1, int(frac * pool_size)), replace=False)
        for name, cls in routers.items():
            r = cls(embeddings[sub], scores[sub], costs[sub], models,
                    [texts[i] for i in sub])
            df = evaluate_at_lambdas(r, embeddings[test_idx], scores[test_idx],
                                     costs[test_idx], lambdas)
            rows.append({"router": name, "fraction": frac,
                         "mean_utility": df["utility"].mean()})
    return pd.DataFrame(rows)
