"""Generate a synthetic-but-realistic support set and model pool.

Run once locally (not needed at app runtime):
    python scripts/generate_support_set.py

Creates:
  - src/data/model_pool.json      : 5 models with API-like cost profiles
  - src/data/support_set.jsonl    : N queries with embeddings, per-model scores/costs

The simulator encodes "capability clusters": models specialize in different
domains (math, code, creative, QA), so utility genuinely varies per query —
exactly the locality structure the paper relies on.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from config import (SUPPORT_SET_PATH, MODEL_POOL_PATH, EMBEDDING_MODEL,
                    EMBEDDING_DIM, SEED, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

rng = np.random.default_rng(SEED)

MODEL_POOL = [
    {"name": "minima-3b",        "input_price": 0.10, "output_price": 0.20, "strength": ["qa", "creative"]},
    {"name": "workhorse-8b",     "input_price": 0.30, "output_price": 0.60, "strength": ["qa", "math", "creative"]},
    {"name": "coder-7b",         "input_price": 0.30, "output_price": 0.90, "strength": ["code"]},
    {"name": "frontier-pro",     "input_price": 3.00, "output_price": 15.0, "strength": ["math", "code", "reasoning"]},
    {"name": "frontier-mini",    "input_price": 0.15, "output_price": 0.60, "strength": ["qa", "math", "creative", "reasoning"]},
]
DOMAINS = ["math", "code", "creative", "qa", "reasoning"]
SEEDS = {
    "math":     ["Solve step by step", "Compute the integral", "Prove that", "What is the derivative of"],
    "code":     ["Write a function that", "Debug this Python code", "Implement", "Explain what this code does"],
    "creative": ["Write a short story about", "Compose a poem on", "Continue this passage", "Describe a world where"],
    "qa":       ["What is the capital of", "Who wrote", "Explain the difference between", "When did"],
    "reasoning":["If all bloops are razzies", "Which statement must be true", "Evaluate the argument", "Given the constraints"],
}

def main(n_queries: int = 3000):
    # 1) synthetic queries as domain seed + random words
    queries, labels = [], []
    for _ in range(n_queries):
        d = DOMAINS[rng.integers(len(DOMAINS))]
        labels.append(d)
        queries.append(SEEDS[d][rng.integers(len(SEEDS[d]))] + " " +
                       " ".join(rng.choice(["alpha", "beta", "gamma", "delta",
                                            "omega", "prime", "vector", "logic",
                                            "market", "garden"], size=rng.integers(2, 6))))
    labels = np.array(labels)

    # 2) embeddings — placeholder random correlated with domain so locality holds.
    #    In production replace with:  from embedding_service import encode
    # domain_centers = rng.normal(0, 1, (len(DOMAINS), EMBEDDING_DIM)).astype(np.float32)
    # domain_centers /= np.linalg.norm(domain_centers, axis=1, keepdims=True)
    # embs = domain_centers[[DOMAINS.index(l) for l in labels]] + 0.25 * rng.normal(0, 1, (n_queries, EMBEDDING_DIM)).astype(np.float32)
    # embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    from embedding_service import encode
    embs = encode(queries, EMBEDDING_MODEL).astype(np.float32)


    # 3) per-model scores: strong in strength domains, weak elsewhere
    scores = np.zeros((n_queries, len(MODEL_POOL)), dtype=np.float32)
    costs = np.zeros_like(scores)
    for j, m in enumerate(MODEL_POOL):
        for i, d in enumerate(labels):
            strong = d in m["strength"]
            base = rng.uniform(0.75, 0.95) if strong else rng.uniform(0.25, 0.55)
            scores[i, j] = base
            costs[i, j] = m["input_price"] + rng.uniform(0.5, 1.0) * m["output_price"]

    # 4) splits (paper §B.4): 70/10/20
    perm = rng.permutation(n_queries)
    n_tr, n_va = int(SPLIT_TRAIN * n_queries), int(SPLIT_VAL * n_queries)
    split = np.array(["test"] * n_queries, dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr:n_tr + n_va]] = "val"

    MODEL_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_POOL_PATH.write_text(json.dumps(MODEL_POOL, indent=2))
    with open(SUPPORT_SET_PATH, "w") as f:
        for i in range(n_queries):
            f.write(json.dumps({
                "id": i, "text": queries[i], "domain": labels[i], "split": split[i],
                "embedding": embs[i].tolist(),
                "scores": {m["name"]: float(scores[i, j]) for j, m in enumerate(MODEL_POOL)},
                "costs":   {m["name"]: float(costs[i, j]) for j, m in enumerate(MODEL_POOL)},
            }) + "\n")
    print(f"wrote {n_queries} items -> {SUPPORT_SET_PATH}")


if __name__ == "__main__":
    main()
