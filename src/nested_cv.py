

import numpy as np
import pandas as pd
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from de import differential_evolution
from hill_climbing import hill_climbing

def mi_score(X, y, seed=42):
    return mutual_info_classif(X, y, random_state=seed)


def run_pipeline_on_training_data(X_train, y_train, percentile=5,
                                  de_generations=10, seed=42):

    k = max(1, int(X_train.shape[1] * percentile / 100))
    selector = SelectKBest(partial(mi_score, seed=seed), k=k)
    selector.fit(X_train, y_train)
    filter_idx = selector.get_support(indices=True)   # which genes survived
    X_filtered = X_train[:, filter_idx]

    
    de_mask, _ = differential_evolution(X_filtered, y_train,
                                        generations=de_generations, seed=seed)

    # Stage 3: HC refines DE's subset
    hc_mask, _ = hill_climbing(de_mask, X_filtered, y_train,
                               seed=seed, max_steps=15)

    
    selected_idx = filter_idx[hc_mask == 1]
    return selected_idx


def nested_cv(X, y, outer_splits=10, seed=42):
    """Run the honest nested cross-validation and return per-fold accuracies."""
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True,
                               random_state=seed)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        selected_idx = run_pipeline_on_training_data(X_train, y_train, seed=seed)

        model = Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ])
        model.fit(X_train[:, selected_idx], y_train)
        y_pred = model.predict(X_test[:, selected_idx])
        acc = accuracy_score(y_test, y_pred)

        fold_scores.append(acc)
        print(f"Outer fold {fold:2d}: accuracy {acc:.3f}  "
              f"({len(selected_idx)} genes selected)")

    return np.array(fold_scores)


if __name__ == "__main__":
    df = pd.read_csv("data/CNS.csv")
    y = df["class"].values
    X = df.drop(columns=["class"]).values
    print(f"Loaded CNS: {X.shape[0]} samples, {X.shape[1]} genes\n")

    print("Running nested cross-validation (this takes a few minutes)...\n")
    scores = nested_cv(X, y)

    print(f"\n--- Honest result ---")
    print(f"Mean accuracy: {scores.mean():.3f}  (+/- {scores.std():.3f})")
    print(f"(Leaky scripts reported ~0.867 - expect this to be lower and honest.)")
