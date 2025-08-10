from typing import Dict, Tuple
import numpy as np
import lightgbm as lgb
from .model_graph import Node

class LGBBackend:
    def __init__(self, X_train, y_train, X_val, params=None, lr=0.05):
        self.Xtr, self.ytr = X_train, y_train
        self.Xval = X_val
        self.lr = lr
        self.params = params or dict(objective="binary", learning_rate=lr,
                                     num_leaves=31, max_depth=-1, min_data_in_leaf=20,
                                     verbose=-1)
        # one booster per universe
        self.boosters: Dict[int, lgb.Booster] = {}
        self.rounds: Dict[int, int] = {}

    def _ensure(self, u: int):
        if u not in self.boosters:
            dtrain = lgb.Dataset(self.Xtr, label=self.ytr, free_raw_data=False)
            booster = lgb.train(dict(self.params, num_boost_round=0), dtrain=dtrain)
            self.boosters[u] = booster
            self.rounds[u] = 0

    def fit_forward(self, node: Node):
        self._ensure(node.u)
        booster = self.boosters[node.u]
        dtrain = lgb.Dataset(self.Xtr, label=self.ytr, free_raw_data=False)
        booster = lgb.train(dict(self.params, num_boost_round=1), train_set=dtrain, init_model=booster)
        self.boosters[node.u] = booster
        self.rounds[node.u] += 1

    def prune(self, node: Node):
        self._ensure(node.u)
        r = self.rounds[node.u]
        if r <= 0:
            return
        booster = self.boosters[node.u]
        booster.drop_constraint()  # no-op; LightGBM lacks native "pop last tree"
        # For a real implementation, rebuild booster up to r-1 from stored snapshots.
        self.rounds[node.u] = max(0, r-1)

    def clone_to(self, src: Node, dst: Node):
        self._ensure(src.u)
        self._ensure(dst.u)
        # no-op placeholder; extend to copy leaves/splits or average
        return

    def predict_val(self, node: Node) -> np.ndarray:
        self._ensure(node.u)
        booster = self.boosters[node.u]
        # ignore node.m here for brevity; real impl should slice to m trees
        return 1.0 / (1.0 + np.exp(-booster.predict(self.Xval, raw_score=True)))
