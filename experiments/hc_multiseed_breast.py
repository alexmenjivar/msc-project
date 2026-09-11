import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from functools import partial
from sklearn.impute import SimpleImputer
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

df = pd.read_csv("data/Breast_clean.csv", na_values=["?"], low_memory=False)
y = df["class"].values
X = df.drop(columns=["class"]).values.astype(float)
print(f"Breast: {X.shape[0]} samples, {X.shape[1]} genes\n")

SEEDS = [0, 1, 2]                       # 3 seeds (Breast is slow); add more if you have time
per_seed = {"longDE": [], "shortDE": [], "shortDE+HC": []}

for seed in SEEDS:
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)   # 3-fold for speed
    fold_res = {"longDE": [], "shortDE": [], "shortDE+HC": []}
    for tr, te in outer.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)
        k = max(1, int(Xtr_i.shape[1]*0.05))
        sel = SelectKBest(partial(mi_score, seed=seed), k=k).fit(Xtr_i, ytr)
        fidx = sel.get_support(indices=True)
        Xf = Xtr_i[:, fidx]

        long_mask, _  = differential_evolution(Xf, ytr, generations=15, pop_size=25, seed=seed)
        short_mask, _ = differential_evolution(Xf, ytr, generations=5, pop_size=25, seed=seed)
        hc_mask, _    = hill_climbing(short_mask, Xf, ytr, seed=seed, max_steps=8)

        for label, mask in [("longDE",long_mask),("shortDE",short_mask),("shortDE+HC",hc_mask)]:
            idx = fidx[mask == 1]
            m = Pipeline([("scale",StandardScaler()),("svm",SVC(kernel="linear",random_state=seed))])
            m.fit(Xtr_i[:, idx], ytr)
            fold_res[label].append(accuracy_score(yte, m.predict(Xte_i[:, idx])))
    for label in fold_res:
        per_seed[label].append(np.mean(fold_res[label]))
    print(f"seed {seed}: longDE={per_seed['longDE'][-1]:.3f}  "
          f"shortDE={per_seed['shortDE'][-1]:.3f}  shortDE+HC={per_seed['shortDE+HC'][-1]:.3f}")

print("\n=== Averaged over seeds (Breast, SVM) ===")
for label in ["longDE","shortDE","shortDE+HC"]:
    arr=np.array(per_seed[label]); print(f"{label:<12} {arr.mean():.3f}  (+/- {arr.std():.3f})")
wins = sum(1 for h,s in zip(per_seed["shortDE+HC"],per_seed["shortDE"]) if h>s)
ties = sum(1 for h,s in zip(per_seed["shortDE+HC"],per_seed["shortDE"]) if h==s)
print(f"\nHC beat short-DE in {wins}/{len(SEEDS)} seeds, tied in {ties}/{len(SEEDS)}")
