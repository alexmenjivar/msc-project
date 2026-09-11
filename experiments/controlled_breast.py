"""
Controlled comparison: does cleaning help Breast?
Identical settings for both files - ONLY the input data differs.
Same seed, same folds, same DE/HC settings, same classifiers.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
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

def mi_score(X, y, seed=42):
    return mutual_info_classif(X, y, random_state=seed)

def clfs(seed=42):
    return {"KNN":KNeighborsClassifier(5),"NB":GaussianNB(),
            "SVM":SVC(kernel="linear",random_state=seed),
            "DT":DecisionTreeClassifier(random_state=seed),
            "RF":RandomForestClassifier(n_estimators=100,random_state=seed)}

def select_genes(Xtr, ytr, seed=42):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i = imp.transform(Xtr)
    k = max(1, int(Xtr_i.shape[1]*0.05))
    sel = SelectKBest(partial(mi_score, seed=seed), k=k).fit(Xtr_i, ytr)
    fidx = sel.get_support(indices=True)
    Xf = Xtr_i[:, fidx]
    de_mask,_ = differential_evolution(Xf, ytr, generations=4, pop_size=25, seed=seed)
    hc_mask,_ = hill_climbing(de_mask, Xf, ytr, seed=seed, max_steps=2)
    return fidx[hc_mask==1], imp

def run(path, label):
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    y = df["class"].values
    X = df.drop(columns=["class"]).values.astype(float)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # SAME for both
    names=["KNN","NB","SVM","DT","RF"]
    acc={n:[] for n in names}
    for tr,te in outer.split(X,y):
        Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te]
        idx,imp=select_genes(Xtr,ytr)
        Xte_i=imp.transform(Xte); Xtr_i=imp.transform(Xtr)
        for name,clf in clfs().items():
            m=Pipeline([("scale",StandardScaler()),("clf",clf)]).fit(Xtr_i[:,idx],ytr)
            acc[name].append(accuracy_score(yte,m.predict(Xte_i[:,idx])))
    print(f"{label}: {X.shape[1]} genes")
    return {n:np.mean(acc[n]) for n in names}

print("Running ORIGINAL...")
orig = run("data/Breast.csv", "Original")
print("Running CLEANED...")
clean = run("data/Breast_clean.csv", "Cleaned")

print("\n=== CONTROLLED comparison (identical settings, only data differs) ===")
print(f"{'Classifier':<6}{'Original':>10}{'Cleaned':>10}{'Change':>10}")
for n in ["KNN","NB","SVM","DT","RF"]:
    d = clean[n]-orig[n]
    print(f"{n:<6}{orig[n]:>10.3f}{clean[n]:>10.3f}{d:>+10.3f}")
