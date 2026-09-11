import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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

# collect accuracy + gene count for both versions, using SVM (our best classifier)
res = {"DE": {"acc": [], "genes": []}, "DE+HC": {"acc": [], "genes": []}}

for fold, (tr, te) in enumerate(outer.split(X, y), 1):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # shared: filter
    k = max(1, int(Xtr.shape[1]*0.05))
    sel = SelectKBest(partial(mi_score, seed=42), k=k).fit(Xtr, ytr)
    fidx = sel.get_support(indices=True)
    Xf = Xtr[:, fidx]

    # shared: DE (moderate length so HC has room to help)
    de_mask, _ = differential_evolution(Xf, ytr, generations=10, seed=42)
    de_idx = fidx[de_mask == 1]

    # Version B only: HC refines DE's result
    hc_mask, _ = hill_climbing(de_mask, Xf, ytr, seed=42, max_steps=15)
    hc_idx = fidx[hc_mask == 1]

    # evaluate both on the SAME held-out fold with SVM
    for label, idx in [("DE", de_idx), ("DE+HC", hc_idx)]:
        m = Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="linear", random_state=42))])
        m.fit(Xtr[:, idx], ytr)
        acc = accuracy_score(yte, m.predict(Xte[:, idx]))
        res[label]["acc"].append(acc)
        res[label]["genes"].append(len(idx))
    print(f"fold {fold}: DE={res['DE']['acc'][-1]:.3f} ({len(de_idx)}g)  "
          f"DE+HC={res['DE+HC']['acc'][-1]:.3f} ({len(hc_idx)}g)")

print("\n=== Does Hill Climbing add value? (CNS, SVM) ===")
for label in ["DE", "DE+HC"]:
    a = np.mean(res[label]["acc"]); g = np.mean(res[label]["genes"])
    print(f"{label:<6} accuracy {a:.3f}   avg genes {g:.0f}")
diff = np.mean(res["DE+HC"]["acc"]) - np.mean(res["DE"]["acc"])
print(f"\nHC effect on accuracy: {diff:+.3f}")
