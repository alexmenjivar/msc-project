import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from de import differential_evolution
from hill_climbing import hill_climbing

def mi(X, y, seed): return mutual_info_classif(X, y, random_state=seed)

# (path, impute, folds, short_gen, pop, hc_steps, labels)
CFG = {
    "CNS":    ("data/CNS.csv",    False, 5, 8, 50, 15, ["Tumour","Normal"]),
    "Lung":   ("data/Lung.csv",   False, 5, 6, 25, 4,  ["ADEN","SQUA","COID","NORMAL","SCLC"]),
    "Breast": ("data/Breast.csv", True,  3, 5, 25, 4,  ["relapse","non-relapse"]),
}
SEED = 0   # matches the first seed used in full_comparison.py

def load(path, impute):
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    return df.drop(columns=["class"]).values.astype(float), df["class"].values

print("=== GENE COUNTS (filter -> DE -> HC), pipeline = Short DE + HC ===")
gene_counts = {}
for name,(path,impute,folds,sg,pop,steps,labels) in CFG.items():
    X, y = load(path, impute)
    if impute:
        keep = ~np.isnan(X).all(axis=0); X = X[:,keep]
        X = SimpleImputer(strategy="median").fit_transform(X)
    k = max(1, int(X.shape[1]*0.05))
    sel = SelectKBest(partial(mi,seed=SEED), k=k).fit(X,y)
    fidx = sel.get_support(indices=True); Xf = X[:,fidx]
    de_mask,_ = differential_evolution(Xf, y, generations=sg, pop_size=pop, seed=SEED)
    hc_mask,_ = hill_climbing(de_mask, Xf, y, seed=SEED, max_steps=steps)
    gene_counts[name] = (len(fidx), int(de_mask.sum()), int(hc_mask.sum()))
    print(f"  {name}: filter={len(fidx)}  DE={int(de_mask.sum())}  HC={int(hc_mask.sum())}")

print("\n=== CONFUSION MATRICES (SVM, pipeline genes, across folds) ===")
def plot_cm(cm, labels, title, fname):
    cm=np.array(cm)
    fig,ax=plt.subplots(figsize=(4.2,3.8) if len(labels)>2 else (4.0,3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45 if len(labels)>2 else 0, ha="right" if len(labels)>2 else "center")
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    th=cm.max()/2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>th else "#1F3864",fontsize=14 if len(labels)>2 else 15,fontweight="bold")
    ax.set_title(title, fontsize=11, color="#333", pad=10)
    plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()

for name,(path,impute,folds,sg,pop,steps,labels) in CFG.items():
    X, y = load(path, impute)
    outer = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    allt, allp = [], []

    for tr,te in outer.split(X,y):
        Xtr,Xte,ytr,yte = X[tr],X[te],y[tr],y[te]

        if impute:
            keep=~np.isnan(Xtr).all(axis=0)
            Xtr,Xte=Xtr[:,keep],Xte[:,keep]
            imp=SimpleImputer(strategy="median").fit(Xtr)
            Xtr,Xte=imp.transform(Xtr),imp.transform(Xte)

        k=max(1,int(Xtr.shape[1]*0.05))
        sel=SelectKBest(partial(mi,seed=SEED),k=k).fit(Xtr,ytr)
        fidx=sel.get_support(indices=True)
        Xf=Xtr[:,fidx]

        de_mask,_=differential_evolution(
            Xf,ytr,generations=sg,pop_size=pop,seed=SEED
        )
        hc_mask,_=hill_climbing(
            de_mask,Xf,ytr,seed=SEED,max_steps=steps
        )

        idx=fidx[hc_mask==1]

        m=Pipeline([
            ("s",StandardScaler()),
            ("svm",SVC(kernel="linear",random_state=SEED))
        ]).fit(Xtr[:,idx],ytr)

        allt.extend(yte)
        allp.extend(m.predict(Xte[:,idx]))

    cm = confusion_matrix(allt, allp, labels=labels)
    print(f"  {name}: {cm.tolist()}")

    display_labels = ["Class 0", "Class 1"] if name == "CNS" else labels

    plot_cm(
        cm,
        display_labels,
        f"Confusion matrix — {name} (SVM)",
        f"cm_{name.lower()}.png"
    )

print("\nSaved: cm_cns.png, cm_lung.png, cm_breast.png")

