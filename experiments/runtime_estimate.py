"""
Runtime estimate for the DE and HC feature-selection stages.

Times a single run of DE and of HC on each dataset, at the settings used in
the experiments. Reports seconds per run, and scales up to estimate the cost
of a full experiment (all classifiers x configurations x seeds x folds).

The running time depends heavily on how many genes survive the filter
(CNS ~356, Lung ~630, Breast ~1,200), because DE and HC search within those.

Run:  python3 experiments/runtime_estimate.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import time, numpy as np, pandas as pd
from functools import partial
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from de import differential_evolution
from hill_climbing import hill_climbing

def mi(X, y, seed): return mutual_info_classif(X, y, random_state=seed)

# (path, needs_impute, short_gen, pop, hc_steps)
DATASETS = {
    "CNS":    ("data/CNS.csv",    False, 8, 50, 15),
    "Lung":   ("data/Lung.csv",   False, 6, 25, 4),
    "Breast": ("data/Breast.csv", True,  5, 25, 4),
}

print(f"{'Dataset':8s}{'Filtered genes':>16s}{'DE time (s)':>14s}{'HC time (s)':>14s}")
print("-"*52)
results = {}
for name,(path,impute,gen,pop,steps) in DATASETS.items():
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    y = df["class"].values
    X = df.drop(columns=["class"]).values.astype(float)
    if impute:
        keep = ~np.isnan(X).all(axis=0); X = X[:,keep]
        X = SimpleImputer(strategy="median").fit_transform(X)
    k = max(1, int(X.shape[1]*0.05))
    sel = SelectKBest(partial(mi,seed=42),k=k).fit(X,y)
    fidx = sel.get_support(indices=True); Xf = X[:,fidx]

    # time DE
    t0=time.time()
    de_mask,_=differential_evolution(Xf,y,generations=gen,pop_size=pop,seed=42)
    de_t=time.time()-t0

    # time HC (starting from DE result)
    t0=time.time()
    hill_climbing(de_mask,Xf,y,seed=42,max_steps=steps)
    hc_t=time.time()-t0

    results[name]=(len(fidx),de_t,hc_t)
    print(f"{name:8s}{len(fidx):>16d}{de_t:>14.1f}{hc_t:>14.1f}")

# scale-up estimate
print("\n--- Full-experiment estimate ---")
print("A full run = 5 classifiers dont affect DE/HC (search is classifier-independent")
print("here), but each configuration x seed x outer-fold re-runs DE and/or HC.")
print("Approx per dataset: (configs using DE or HC) x seeds x folds x (DE+HC time).\n")
SEEDS=3
for name,(ng,de_t,hc_t) in results.items():
    folds = 3 if name=="Breast" else 5
    # configs that run DE: LongDE, ShortDE, ShortDE_HC (3). HC: ShortDE_HC, FilterHC (2)
    de_runs = 3*SEEDS*folds
    hc_runs = 2*SEEDS*folds
    total = (de_runs*de_t + hc_runs*hc_t)/60
    print(f"{name:8s}: ~{de_runs} DE runs + ~{hc_runs} HC runs  ->  ~{total:.0f} min")
print("\n(Estimates are approximate; actual time varies with machine and load.)")
print("A full-scale run with more seeds/generations was left as future work due to")
print("the computational cost of DE and HC on the large filtered gene sets.")
