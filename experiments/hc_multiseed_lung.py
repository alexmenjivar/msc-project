import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from de import differential_evolution
from hill_climbing import hill_climbing

def mi_score(X, y, seed):
    return mutual_info_classif(X, y, random_state=seed)

df = pd.read_csv("data/Lung.csv")
y = df["class"].values
X = df.drop(columns=["class"]).values
print(f"CNS: {X.shape[0]} samples, {X.shape[1]} genes\n")

SEEDS = [0, 1, 2, 3, 4]          # five different random splits
per_seed = {"longDE": [], "shortDE": [], "shortDE+HC": []}

for seed in SEEDS:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_res = {"longDE": [], "shortDE": [], "shortDE+HC": []}

    for tr, te in outer.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        k = max(1, int(Xtr.shape[1]*0.05))
        sel = SelectKBest(partial(mi_score, seed=seed), k=k).fit(Xtr, ytr)
        fidx = sel.get_support(indices=True)
        Xf = Xtr[:, fidx]

        long_mask, _  = differential_evolution(Xf, ytr, generations=20, seed=seed)
        short_mask, _ = differential_evolution(Xf, ytr, generations=6, seed=seed)
        hc_mask, _    = hill_climbing(short_mask, Xf, ytr, seed=seed, max_steps=15)

        for label, mask in [("longDE", long_mask), ("shortDE", short_mask), ("shortDE+HC", hc_mask)]:
            idx = fidx[mask == 1]
            m = Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="linear", random_state=seed))])
            m.fit(Xtr[:, idx], ytr)
            fold_res[label].append(accuracy_score(yte, m.predict(Xte[:, idx])))

    for label in fold_res:
        per_seed[label].append(np.mean(fold_res[label]))
    print(f"seed {seed}: longDE={per_seed['longDE'][-1]:.3f}  "
          f"shortDE={per_seed['shortDE'][-1]:.3f}  "
          f"shortDE+HC={per_seed['shortDE+HC'][-1]:.3f}")

print("\n=== Averaged over 5 seeds (Lung, SVM) ===")
for label in ["longDE", "shortDE", "shortDE+HC"]:
    arr = np.array(per_seed[label])
    print(f"{label:<12} {arr.mean():.3f}  (+/- {arr.std():.3f})")

# how often did HC actually help?
wins = sum(1 for h, s in zip(per_seed["shortDE+HC"], per_seed["shortDE"]) if h > s)
ties = sum(1 for h, s in zip(per_seed["shortDE+HC"], per_seed["shortDE"]) if h == s)
print(f"\nHC beat short-DE in {wins}/5 seeds, tied in {ties}/5")
