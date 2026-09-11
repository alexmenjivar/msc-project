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

def mi_score(X, y, seed=42):
    return mutual_info_classif(X, y, random_state=seed)

df = pd.read_csv("data/CNS.csv")
y = df["class"].values
X = df.drop(columns=["class"]).values
print(f"CNS: {X.shape[0]} samples, {X.shape[1]} genes\n")

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {"longDE": [], "shortDE": [], "shortDE+HC": []}

for fold, (tr, te) in enumerate(outer.split(X, y), 1):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    k = max(1, int(Xtr.shape[1]*0.05))
    sel = SelectKBest(partial(mi_score, seed=42), k=k).fit(Xtr, ytr)
    fidx = sel.get_support(indices=True)
    Xf = Xtr[:, fidx]

    # long DE alone (30 generations)
    long_mask, _ = differential_evolution(Xf, ytr, generations=30, seed=42)
    # short DE alone (8 generations)
    short_mask, _ = differential_evolution(Xf, ytr, generations=8, seed=42)
    # short DE + HC
    hc_mask, _ = hill_climbing(short_mask, Xf, ytr, seed=42, max_steps=15)

    for label, mask in [("longDE", long_mask), ("shortDE", short_mask), ("shortDE+HC", hc_mask)]:
        idx = fidx[mask == 1]
        m = Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="linear", random_state=42))])
        m.fit(Xtr[:, idx], ytr)
        res[label].append(accuracy_score(yte, m.predict(Xte[:, idx])))
    print(f"fold {fold} done")

print("\n=== Is the hybrid worth it? (CNS, SVM) ===")
for label in ["longDE", "shortDE", "shortDE+HC"]:
    print(f"{label:<12} accuracy {np.mean(res[label]):.3f}")
print(f"\nHC rescued short DE by: {np.mean(res['shortDE+HC'])-np.mean(res['shortDE']):+.3f}")
print(f"short+HC vs long DE:    {np.mean(res['shortDE+HC'])-np.mean(res['longDE']):+.3f}")
