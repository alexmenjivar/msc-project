import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


df = pd.read_csv("data/CNS.csv")
y = df["class"]
X = df.drop(columns=["class"])

print(f"Loaded CNS: {X.shape[0]} samples, {X.shape[1]} genes")

pipeline = Pipeline([
    ("select", SelectPercentile(score_func=mutual_info_classif, percentile=5)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

n_selected = int(X.shape[1] * 0.05)
print(f"Keeping top 5% of genes: about {n_selected} genes\n")


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print(f"Accuracy per fold: {[f'{s:.2f}' for s in scores]}")
print(f"Mean accuracy: {scores.mean():.3f}  (+/- {scores.std():.3f})")
print("\nCompare to baseline (all genes): 0.717")
