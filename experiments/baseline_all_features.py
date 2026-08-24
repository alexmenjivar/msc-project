
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


df = pd.read_csv("data/CNS.csv")

y = df["class"]                
X = df.drop(columns=["class"]) 

print(f"Loaded CNS: {X.shape[0]} samples, {X.shape[1]} genes")
print(f"Classes: {dict(y.value_counts())}")


knn = KNeighborsClassifier(n_neighbors=5)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")

print(f"\nAccuracy per fold: {[f'{s:.2f}' for s in scores]}")
print(f"Mean accuracy: {scores.mean():.3f}  (+/- {scores.std():.3f})")
