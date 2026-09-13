"""kNN router: retrieve neighbors, aggregate outcomes, pick argmax utility.

Implements the utility-prediction formulation of the paper:
    m* = argmax_m  mean_score(m) - lambda * mean_cost(m)
over the k nearest neighbors in embedding space.
"""
from dataclasses import dataclass, field
import numpy as np
from .knn_index import KNNIndex


@dataclass
class RouteResult:
    selected_model: str
    utility_by_model: dict
    neighbors: list = field(default_factory=list)
    lam: float = 0.0


class KNNRouter:
    """Non-parametric router over a support set (paper §3, §5).

    Args:
        embeddings:  (N, D) support-set embeddings
        scores:      (N, M) performance score of each model on each support item
        costs:       (N, M) invocation cost of each model on each support item
        models:      list of M model names
        texts:       list of N support texts (for the neighbor view)
    """
    def __init__(self, embeddings, scores, costs, models, texts=None):
        self.index = KNNIndex(embeddings)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.costs = np.asarray(costs, dtype=np.float32)
        self.models = list(models)
        self.texts = list(texts) if texts is not None else [""] * len(embeddings)
        self.c_max = float(self.costs.max())

    def route(self, query_emb: np.ndarray, lam: float, k: int = 100,
              return_neighbors: bool = False) -> RouteResult:
        """Route one query. `lam` is normalized by c_max (paper §4.3)."""
        idx, sims = self.index.search(query_emb, k)
        mean_score = self.scores[idx].mean(axis=0)          # (M,)
        mean_cost = self.costs[idx].mean(axis=0)            # (M,)
        utility = mean_score - lam * self.c_max * mean_cost

        result = RouteResult(
            selected_model=self.models[int(np.argmax(utility))],
            utility_by_model={m: float(u) for m, u in zip(self.models, utility)},
            lam=lam,
        )
        if return_neighbors:
            best_per_item = np.argmax(self.scores[idx] - lam * self.c_max * self.costs[idx], axis=1)
            result.neighbors = [
                {
                    "rank": r + 1,
                    "text": self.texts[i][:120],
                    "similarity": round(float(sims[r]), 4),
                    "best_model": self.models[best_per_item[r]],
                }
                for r, i in enumerate(idx)
            ]
        return result
