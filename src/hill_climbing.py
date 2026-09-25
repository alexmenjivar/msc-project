
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold


def fitness(mask, X, y, inner_cv):
    
    if mask.sum() == 0:
        return 1.0
    X_subset = X[:, mask == 1]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    acc = cross_val_score(model, X_subset, y, cv=inner_cv, scoring="accuracy").mean()
    return 1.0 - acc


def hill_climbing(start_mask, X, y, seed=42, max_steps=50):
    
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    n_features = X.shape[1]

    current = start_mask.copy()
    current_score = fitness(current, X, y, inner_cv)
    print(f"  start: accuracy {1-current_score:.3f}, genes = {int(current.sum())}")

    for step in range(max_steps):
        best_neighbour = None
        best_score = current_score

        for g in range(n_features):
            neighbour = current.copy()
            neighbour[g] = 1 - neighbour[g]      # flip this one gene
            if neighbour.sum() == 0:
                continue
            score = fitness(neighbour, X, y, inner_cv)
            if score < best_score:               # strictly better?
                best_score = score
                best_neighbour = neighbour

        # If no neighbour improved, we're at a local peak -> stop.
        if best_neighbour is None:
            print(f"  step {step+1}: no improvement, stopping.")
            break

        # Otherwise move to the best neighbour and continue.
        current = best_neighbour
        current_score = best_score
        print(f"  step {step+1}: accuracy {1-current_score:.3f}, "
              f"genes = {int(current.sum())}")

    return current, current_score


if __name__ == "__main__":
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from de import differential_evolution

    df = pd.read_csv("data/CNS.csv")
    y = df["class"].values
    X_full = df.drop(columns=["class"]).values
    print(f"Loaded CNS: {X_full.shape[0]} samples, {X_full.shape[1]} genes")

    k = int(X_full.shape[1] * 0.05)
    X_filtered = SelectKBest(mutual_info_classif, k=k).fit_transform(X_full, y)
    print(f"After filter: {X_filtered.shape[1]} genes\n")

    print("Differential Evolution:")
    de_mask, de_score = differential_evolution(X_filtered, y)
    print(f"DE result: accuracy {1-de_score:.3f}, genes = {int(de_mask.sum())}\n")

    print("Hill Climbing (refining DE's subset):")
    hc_mask, hc_score = hill_climbing(de_mask, X_filtered, y)

    print(f"\n--- Summary ---")
    print(f"Baseline (all genes): 0.717")
    print(f"Filter only:          0.633")
    print(f"Filter + DE:          {1-de_score:.3f}  ({int(de_mask.sum())} genes)")
    print(f"Filter + DE + HC:     {1-hc_score:.3f}  ({int(hc_mask.sum())} genes)")
