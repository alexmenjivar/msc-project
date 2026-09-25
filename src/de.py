
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


def differential_evolution(X, y,
                           pop_size=50,
                           generations=30,
                           crossover_rate=0.9,
                           seed=42):

    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    
    population = rng.integers(0, 2, size=(pop_size, n_features))
    scores = np.array([fitness(ind, X, y, inner_cv) for ind in population])

    for gen in range(generations):
        for i in range(pop_size):
            
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            
            diff = (population[r1] != population[r2]).astype(int)
            mutant = np.where(diff == 1, population[r1], population[r3])

            cross_points = rng.random(n_features) <= crossover_rate
            
            if not cross_points.any():
                cross_points[rng.integers(0, n_features)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial_score = fitness(trial, X, y, inner_cv)
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score

        best = scores.argmin()
        print(f"  gen {gen+1:2d}: best misclass = {scores[best]:.3f}  "
              f"(accuracy {1-scores[best]:.3f}, genes = {int(population[best].sum())})")

    best = scores.argmin()
    return population[best], scores[best]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    df = pd.read_csv("data/CNS.csv")
    y = df["class"].values
    X_full = df.drop(columns=["class"]).values
    print(f"Loaded CNS: {X_full.shape[0]} samples, {X_full.shape[1]} genes")
 
    k = int(X_full.shape[1] * 0.05)
    X_filtered = SelectKBest(mutual_info_classif, k=k).fit_transform(X_full, y)
    print(f"After filter: {X_filtered.shape[1]} genes going into DE\n")

    print("Running Differential Evolution...")
    best_mask, best_score = differential_evolution(X_filtered, y)

    print(f"\nBest subset: {int(best_mask.sum())} genes, "
          f"accuracy {1-best_score:.3f}")
