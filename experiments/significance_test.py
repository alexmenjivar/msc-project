
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd
from functools import partial
from scipy.stats import wilcoxon
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

df = pd.read_csv("data/Breast_clean.csv")     # change to Lung / Breast as needed
y = df["class"].values
X = df.drop(columns=["class"]).values
print(f"CNS: {X.shape[0]} samples, {X.shape[1]} genes\n")

SEEDS = [0, 1, 2, 3, 4]
de_scores, hc_scores = [], []        # one mean accuracy per seed

for seed in SEEDS:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    de_fold, hc_fold = [], []
    for tr, te in outer.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        k = max(1, int(Xtr.shape[1]*0.05))
        sel = SelectKBest(partial(mi_score, seed=seed), k=k).fit(Xtr, ytr)
        fidx = sel.get_support(indices=True); Xf = Xtr[:, fidx]
        short_mask, _ = differential_evolution(Xf, ytr, generations=8, seed=seed)
        hc_mask, _    = hill_climbing(short_mask, Xf, ytr, seed=seed, max_steps=15)
        for mask, store in [(short_mask, de_fold), (hc_mask, hc_fold)]:
            idx = fidx[mask == 1]
            m = Pipeline([("scale",StandardScaler()),("svm",SVC(kernel="linear",random_state=seed))])
            m.fit(Xtr[:, idx], ytr)
            store.append(accuracy_score(yte, m.predict(Xte[:, idx])))
    de_scores.append(np.mean(de_fold))
    hc_scores.append(np.mean(hc_fold))
    print(f"seed {seed}: DE={de_scores[-1]:.3f}  DE+HC={hc_scores[-1]:.3f}")

de_scores, hc_scores = np.array(de_scores), np.array(hc_scores)
print(f"\nDE mean:    {de_scores.mean():.3f}")
print(f"DE+HC mean: {hc_scores.mean():.3f}")

print("\n--- Significance test (DE vs DE+HC) ---")
diff = hc_scores - de_scores
if np.all(diff == 0):
    print("All differences are zero -> HC changed nothing. Not significant (p = 1.0).")
else:
    stat, p = wilcoxon(hc_scores, de_scores)
    print(f"statistic = {stat:.3f}, p-value = {p:.4f}")
    if p <= 0.05:
        print("p <= 0.05: the difference IS statistically significant.")
    else:
        print("p > 0.05: the difference is NOT statistically significant")
        print("(adding HC does not produce a real improvement).")
