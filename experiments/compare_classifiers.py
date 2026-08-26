import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from functools import partial
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

def mi_score(X, y, seed=42):
    return mutual_info_classif(X, y, random_state=seed)

def make_classifiers(seed=42):
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB":  GaussianNB(),
        "SVM": SVC(kernel="linear", random_state=seed),
        "DT":  DecisionTreeClassifier(random_state=seed),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=seed),
    }

def select_genes(X_train, y_train, seed=42):
    k = max(1, int(X_train.shape[1] * 0.05))
    sel = SelectKBest(partial(mi_score, seed=seed), k=k).fit(X_train, y_train)
    fidx = sel.get_support(indices=True)
    Xf = X_train[:, fidx]
    de_mask, _ = differential_evolution(Xf, y_train, generations=10, seed=seed)
    hc_mask, _ = hill_climbing(de_mask, Xf, y_train, seed=seed, max_steps=15)
    return fidx[hc_mask == 1]

df = pd.read_csv("data/CNS.csv")
y = df["class"].values
X = df.drop(columns=["class"]).values
print(f"CNS: {X.shape[0]} samples, {X.shape[1]} genes\n")

outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
names = ["KNN","NB","SVM","DT","RF"]
acc = {n: [] for n in names}
f1  = {n: [] for n in names}

for fold, (tr, te) in enumerate(outer.split(X, y), 1):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    idx = select_genes(Xtr, ytr, seed=42)
    for name, clf in make_classifiers().items():
        model = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        model.fit(Xtr[:, idx], ytr)
        pred = model.predict(Xte[:, idx])
        acc[name].append(accuracy_score(yte, pred))
        f1[name].append(f1_score(yte, pred, average="macro"))
    print(f"fold {fold} done")

print("\n=== Honest nested-CV results on CNS (same DE genes, different classifiers) ===")
print(f"{'Classifier':<6} {'Accuracy':>10} {'Macro-F1':>10}")
for n in names:
    print(f"{n:<6} {np.mean(acc[n]):>10.3f} {np.mean(f1[n]):>10.3f}")
print(f"\nHashmi reported CNS = 1.00 (with IG + Naive Bayes)")
