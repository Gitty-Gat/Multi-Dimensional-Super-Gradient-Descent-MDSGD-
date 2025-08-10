import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from .lightgbm_backend import LGBBackend
from .alphanaut import AlphaNaut
from .stacking import logistic_stack


def main():
    ap = argparse.ArgumentParser("mdsgd quickstart")
    ap.add_argument("--universes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    X, y = make_classification(n_samples=20000, n_features=30, n_informative=12, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=123)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, random_state=456)

    backend = LGBBackend(Xtr, ytr, Xval)
    temps = [0.0, 0.02, 0.05, 0.1][:args.universes]
    ctrl = AlphaNaut(backend, U=args.universes, M_max=300, tau_nov=0.15, temperatures=temps)
    active = ctrl.run(yval=yval, steps=args.steps)

    # stack final universes
    P = []
    for node in active:
        P.append(backend.predict_val(node))
    P = np.vstack(P)
    w = logistic_stack(P, yval)
    print("Active nodes:", active)
    print("Stacking weights:", np.round(w, 3))

if __name__ == "__main__":
    main()
