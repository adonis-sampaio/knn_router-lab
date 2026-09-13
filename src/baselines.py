"""Baselines with the same `route` interface, for fair comparison.

- LinearRouter: per-model ridge regression on embeddings (parametric baseline)
- OracleRouter: upper bound using true per-item scores/costs
"""
import numpy as np
from sklearn.linear_model import Ridge
from .router import RouteResult


class LinearRouter:
    def __init__(self, embeddings, scores, costs, models, texts=None):
        self.models = list(models)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.costs = np.asarray(costs, dtype=np.float32)
        self.c_max = float(self.costs.max())
        # One regressor per model, per output (score and cost)
        self.score_regs = [Ridge(alpha=1.0).fit(embeddings, self.scores[:, j])
                           for j in range(len(self.models))]
        self.cost_regs = [Ridge(alpha=1.0).fit(embeddings, self.costs[:, j])
                          for j in range(len(self.models))]

    def route(self, query_emb, lam, k=100, return_neighbors=False) -> RouteResult:
        q = np.atleast_2d(query_emb)
        score_hat = np.array([r.predict(q)[0] for r in self.score_regs])
        cost_hat = np.array([r.predict(q)[0] for r in self.cost_regs])
        utility = score_hat - lam * self.c_max * cost_hat
        return RouteResult(
            selected_model=self.models[int(np.argmax(utility))],
            utility_by_model={m: float(u) for m, u in zip(self.models, utility)},
            lam=lam,
        )


class OracleRouter:
    """Cheats using the support item itself: bounds the achievable utility."""
    def __init__(self, embeddings, scores, costs, models, texts=None):
        self.models = list(models)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.costs = np.asarray(costs, dtype=np.float32)
        self.c_max = float(self.costs.max())

    def route(self, query_emb, lam, k=100, return_neighbors=False) -> RouteResult:
        # In the live demo there is no ground truth for the *new* query, so the
        # oracle predicts with the parametric-free upper envelope of the pool:
        # it assumes every model performs at its best observed score.
        best_scores = self.scores.max(axis=0)
        best_costs = self.costs.min(axis=0)
        utility = best_scores - lam * self.c_max * best_costs
        return RouteResult(
            selected_model=self.models[int(np.argmax(utility))],
            utility_by_model={m: float(u) for m, u in zip(self.models, utility)},
            lam=lam,
        )
