
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
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from de import differential_evolution
from hill_climbing import hill_climbing

def mi(X,y,seed): return mutual_info_classif(X,y,random_state=seed)

def per_fold(path, gens, pop, steps, na=False, nsplits=5, seeds=(0,1,2)):
    df=pd.read_csv(path, na_values=["?"] if na else None, low_memory=False)
    y=df["class"].values; X=df.drop(columns=["class"]).values.astype(float)
    if na:
        keep=~np.isnan(X).all(axis=0); X=X[:,keep]
    accs=[]
    for seed in seeds:
        for tr,te in StratifiedKFold(nsplits,shuffle=True,random_state=seed).split(X,y):
            Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te]
            if na:
                imp=SimpleImputer(strategy="median").fit(Xtr); Xtr=imp.transform(Xtr); Xte=imp.transform(Xte)
            k=max(1,int(Xtr.shape[1]*0.05))
            sel=SelectKBest(partial(mi,seed=seed),k=k).fit(Xtr,ytr); fidx=sel.get_support(indices=True); Xf=Xtr[:,fidx]
            de_mask,_=differential_evolution(Xf,ytr,generations=gens,pop_size=pop,seed=seed)
            hc_mask,_=hill_climbing(de_mask,Xf,ytr,seed=seed,max_steps=steps)
            idx=fidx[hc_mask==1]
            m=Pipeline([("s",StandardScaler()),("svm",SVC(kernel="linear",random_state=seed))]).fit(Xtr[:,idx],ytr)
            accs.append(accuracy_score(yte,m.predict(Xte[:,idx])))
    return accs

print("Running CNS...");   cns  = per_fold("data/CNS.csv",   8, 50, 10)
print("Running Lung...");  lung = per_fold("data/Lung.csv",  5, 25, 3)
print("Running Breast..."); brst = per_fold("data/Breast.csv",5, 25, 3, na=True, nsplits=3)

fig,ax=plt.subplots(figsize=(6.5,3.8))
bp=ax.boxplot([cns,lung,brst], tick_labels=["CNS","Lung","Breast"], patch_artist=True)
for patch,c in zip(bp["boxes"], ["#1F3864","#5B8BD0","#2E5A9C"]):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.set_ylabel("Accuracy per fold (across seeds)")
ax.set_title("Spread of accuracy across folds and seeds", fontsize=10, color="#333")
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig("boxplot.png", dpi=150, bbox_inches="tight")
print("saved boxplot.png")
