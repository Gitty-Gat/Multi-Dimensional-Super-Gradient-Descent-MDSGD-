"""Alpha‑naut search controller for multi‑dimensional super gradient descent."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .model_graph import Node, PredictionCache


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation between two vectors, handling constant cases."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Candidate:
    node: Node
    novelty: float
    quality: float


class AlphaNaut:
    """A minimal novelty‑gated, tempered search over (u, m) nodes.

    The AlphaNaut orchestrates a set of LightGBM boosters through a
    backend interface. Each universe maintains its own booster and
    progresses along boosting rounds. At each step the controller
    proposes forward/backward and lateral moves, computes their
    predictive novelty and quality, and accepts a candidate if it
    passes the novelty threshold and either improves quality or passes
    a Metropolis criterion according to its universe temperature.
    """

    def __init__(self, backend, U: int, M_max: int, *, tau_nov: float = 0.15,
                 delta: float = 0.0, temperatures: Optional[List[float]] = None) -> None:
        self.backend = backend  # backend must implement fit_forward, prune, clone_to, predict_val
        self.U = U
        self.M_max = M_max
        self.tau_nov = tau_nov
        self.delta = delta
        self.temps = temperatures or [0.0] * U
        self.active: List[Node] = [Node(u, 0) for u in range(U)]
        self.cache = PredictionCache(maxlen=64)

    def _neighbors(self, node: Node) -> List[Node]:
        """Return valid neighbor nodes reachable in one move."""
        cand: List[Node] = []
        u, m = node.u, node.m
        if m + 1 <= self.M_max:
            cand.append(Node(u, m + 1))
        if m - 1 >= 0:
            cand.append(Node(u, m - 1))
        if u - 1 >= 0:
            cand.append(Node(u - 1, m))
        if u + 1 < self.U:
            cand.append(Node(u + 1, m))
        if u - 1 >= 0 and m + 1 <= self.M_max:
            cand.append(Node(u - 1, m + 1))
        if u + 1 < self.U and m + 1 <= self.M_max:
            cand.append(Node(u + 1, m + 1))
        return cand

    def _predict(self, node: Node) -> np.ndarray:
        """Return cached or fresh predictions for a node on the validation set."""
        cached = self.cache.get(node)
        if cached is not None:
            return cached
        yhat = self.backend.predict_val(node)
        self.cache.put(node, yhat)
        return yhat

    def _quality(self, node: Node, yval: np.ndarray) -> float:
        """Compute quality: logloss plus size penalty."""
        yhat = self._predict(node)
        eps = 1e-12
        logloss = -np.mean(yval * np.log(yhat + eps) + (1 - yval) * np.log(1 - yhat + eps))
        size_penalty = node.m / max(1, self.M_max)
        return float(logloss + 0.02 * size_penalty)

    def _generate_candidates(self, cur: Node) -> List[Candidate]:
        """Generate candidate nodes with computed novelty and quality."""
        cands: List[Candidate] = []
        yhat_cur = self._predict(cur)
        q_cur = self._quality(cur, self.backend.y_val)

        for nb in self._neighbors(cur):
            # Realize candidate by fitting/pruning/cloning as needed
            if nb.m == cur.m + 1 and nb.u == cur.u:
                self.backend.fit_forward(cur)
            elif nb.m == cur.m - 1 and nb.u == cur.u:
                self.backend.prune(cur)
            elif nb.u != cur.u and nb.m == cur.m:
                self.backend.clone_to(src=nb, dst=cur)
            elif nb.u != cur.u and nb.m == cur.m + 1:
                self.backend.clone_to(src=Node(nb.u, cur.m), dst=cur)
                self.backend.fit_forward(cur)

            yhat_nb = self._predict(nb)
            nov = 1.0 - _corr(yhat_cur, yhat_nb)
            q_nb = self._quality(nb, self.backend.y_val)
            cands.append(Candidate(nb, nov, q_nb))
        return cands

    def step_universe(self, u: int) -> None:
        """Perform one novelty‑gated search step for universe u."""
        cur = self.active[u]
        q_cur = self._quality(cur, self.backend.y_val)
        T = self.temps[u]

        candidates = self._generate_candidates(cur)
        # Sort by quality to try better moves first
        for cand in sorted(candidates, key=lambda c: c.quality):
            accept = False
            if cand.novelty >= self.tau_nov and cand.quality <= q_cur - self.delta:
                accept = True
            elif cand.novelty >= self.tau_nov and T > 0.0:
                import math, random
                p = math.exp(-(cand.quality - q_cur) / (T + 1e-12))
                accept = random.random() < p
            if accept:
                self.active[u] = cand.node
                return

    def run(self, steps: int = 100, swap_every: int = 0) -> List[Node]:
        """Run the search for a number of steps."""
        for _ in range(steps):
            for u in range(self.U):
                self.step_universe(u)
            # (Optional) implement parallel tempering swaps here
        return self.active
