from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class Node:
    """Represents a universe index and boosting round."""
    u: int  # universe index
    m: int  # boosting round


class PredictionCache:
    """LRU cache of validation predictions keyed by (u, m)."""

    def __init__(self, maxlen: int = 64) -> None:
        self.maxlen = maxlen
        self._keys: List[Tuple[int, int]] = []
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get(self, node: Node) -> Optional[np.ndarray]:
        return self._cache.get((node.u, node.m))

    def put(self, node: Node, yhat: np.ndarray) -> None:
        key = (node.u, node.m)
        if key not in self._cache and len(self._keys) >= self.maxlen:
            k0 = self._keys.pop(0)
            self._cache.pop(k0, None)
        if key not in self._cache:
            self._keys.append(key)
        self._cache[key] = yhat
