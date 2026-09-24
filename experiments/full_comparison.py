import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd, csv, os
from functools import partial
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from de import differential_evolution
from hill_climbing import hill_climbing

def mi(X, y, seed): return mutual_info_classif(X, y, random_state=seed)

def classifiers(seed):
    return {
        "SVM": SVC(kernel="linear", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB":  GaussianNB(),
        "DT":  DecisionTreeClassifier(random_state=seed),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=seed),
    }

# --- dataset settings: (path, needs_imputation, n_splits, long_gen, short_gen, pop, hc_steps) ---
DATASETS = {
    "CNS":    ("data/CNS.csv",    False, 5, 30, 8,  50, 15),
    "Lung":   ("data/Lung.csv",   False, 5, 20, 6,  25, 4),
    "Breast": ("data/Breast.csv", True,  3, 15, 5,  25, 4),
}
SEEDS = [0, 1, 2]
OUT = "results_full.csv"

if os.path.exists(OUT):
    os.remove(OUT)

def load(path, impute):
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    y = df["class"].values
    X = df.drop(columns=["class"]).values.astype(float)
    return X, y

def evaluate(config, X, y, cfg, seed):
    """Run one configuration across the outer folds; return (accuracy, macroF1) per classifier."""
    path, impute, nsplits, long_gen, short_gen, pop, hc_steps = cfg
    outer = StratifiedKFold(nsplits, shuffle=True, random_state=seed)
    names = ["SVM","KNN","NB","DT","RF"]
    acc = {n: [] for n in names}; f1 = {n: [] for n in names}

    for tr, te in outer.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        # drop all-missing genes (per fold) and impute if needed
        if impute:
            keep = ~np.isnan(Xtr).all(axis=0)
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]
            imp = SimpleImputer(strategy="median").fit(Xtr)
            Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)

        # choose which genes to keep, per configuration
        if config == "NoSelection":
            idx = np.arange(Xtr.shape[1])
        else:
            k = max(1, int(Xtr.shape[1] * 0.05))
            sel = SelectKBest(partial(mi, seed=seed), k=k).fit(Xtr, ytr)
            fidx = sel.get_support(indices=True); Xf = Xtr[:, fidx]
            if config == "LongDE":
                mask, _ = differential_evolution(Xf, ytr, generations=long_gen, pop_size=pop, seed=seed)
            elif config == "ShortDE":
                mask, _ = differential_evolution(Xf, ytr, generations=short_gen, pop_size=pop, seed=seed)
            elif config == "ShortDE_HC":
                m, _ = differential_evolution(Xf, ytr, generations=short_gen, pop_size=pop, seed=seed)
                mask, _ = hill_climbing(m, Xf, ytr, seed=seed, max_steps=hc_steps)
            elif config == "FilterHC":
                # HC starts from the full filtered set (no DE)
                start = np.ones(Xf.shape[1], dtype=int)
                mask, _ = hill_climbing(start, Xf, ytr, seed=seed, max_steps=hc_steps)
            idx = fidx[mask == 1]

        for name, clf in classifiers(seed).items():
            m = Pipeline([("s", StandardScaler()), ("c", clf)]).fit(Xtr[:, idx], ytr)
            pred = m.predict(Xte[:, idx])
            acc[name].append(accuracy_score(yte, pred))
            f1[name].append(f1_score(yte, pred, average="macro"))
    return {n: (np.mean(acc[n]), np.mean(f1[n])) for n in names}

def main():
    # write header if file is new
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset","config","seed","classifier","accuracy","macro_f1"])

    configs = ["NoSelection","LongDE","ShortDE","ShortDE_HC","FilterHC"]
    for dname, cfg in DATASETS.items():
        X, y = load(cfg[0], cfg[1])
        print(f"\n=== {dname} ({X.shape[0]} samples, {X.shape[1]} genes) ===")
        for config in configs:
            for seed in SEEDS:
                res = evaluate(config, X, y, cfg, seed)
                with open(OUT, "a", newline="") as f:
                    w = csv.writer(f)
                    for clf,(a,fscore) in res.items():
                        w.writerow([dname, config, seed, clf, f"{a:.4f}", f"{fscore:.4f}"])
                print(f"  {config:12s} seed {seed} done")
    print(f"\nAll done. Results in {OUT}")

if __name__ == "__main__":
    main()
