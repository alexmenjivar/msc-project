
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

df = pd.read_csv("data/Lung.csv")
y = df["class"].values
X = df.drop(columns=["class"]).values
print(f"Lung: {X.shape[0]} samples, {X.shape[1]} genes, classes={dict(pd.Series(y).value_counts())}\n")

def select_genes(Xtr, ytr, seed=42):
    k = max(1, int(Xtr.shape[1]*0.05))
    sel = SelectKBest(partial(mi_score, seed=seed), k=k).fit(Xtr, ytr)
    fidx = sel.get_support(indices=True)
    Xf = Xtr[:, fidx]
    de_mask,_ = differential_evolution(Xf, ytr, generations=5, pop_size=30, seed=seed)  
    hc_mask,_ = hill_climbing(de_mask, Xf, ytr, seed=seed, max_steps=3)                  
    return fidx[hc_mask==1]

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
names=["KNN","NB","SVM","DT","RF"]
acc={n:[] for n in names}; f1={n:[] for n in names}
def clfs(seed=42):
    return {"KNN":KNeighborsClassifier(5),"NB":GaussianNB(),
            "SVM":SVC(kernel="linear",random_state=seed),
            "DT":DecisionTreeClassifier(random_state=seed),
            "RF":RandomForestClassifier(n_estimators=100,random_state=seed)}

for fold,(tr,te) in enumerate(outer.split(X,y),1):
    Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te]
    idx=select_genes(Xtr,ytr)
    for name,clf in clfs().items():
        m=Pipeline([("scale",StandardScaler()),("clf",clf)]).fit(Xtr[:,idx],ytr)
        p=m.predict(Xte[:,idx])
        acc[name].append(accuracy_score(yte,p))
        f1[name].append(f1_score(yte,p,average="macro"))
    print(f"fold {fold} done ({len(idx)} genes)")

print("\n=== Honest results on Lung (lighter search) ===")
print(f"{'Classifier':<6}{'Accuracy':>10}{'Macro-F1':>10}")
for n in names:
    print(f"{n:<6}{np.mean(acc[n]):>10.3f}{np.mean(f1[n]):>10.3f}")
print("\nHashmi reported Lung = 0.98 | majority-class baseline = 0.68")
