
import pandas as pd
import numpy as np

DATASETS = {
    "CNS":    "data/CNS.csv",
    "Lung":   "data/Lung.csv",
    "Breast": "data/Breast.csv",
}


def inspect(name, df):
    """Step 1 - basic inspection."""
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Shape: {df.shape[0]} samples x {df.shape[1]-1} genes (+1 class column)")
    print(f"Classes: {dict(df['class'].value_counts())}")


def profile_missing(name, X):
    """Step 2 - missing values per gene, worst-affected."""
    na_per_gene = X.isnull().sum()
    total_missing = int(na_per_gene.sum())
    genes_with_na = int((na_per_gene > 0).sum())
    print(f"\nMissing values: {total_missing} cells across "
          f"{genes_with_na} of {X.shape[1]} genes")
    if total_missing > 0:
        worst = na_per_gene[na_per_gene > 0].sort_values(ascending=False).head(3)
        print("Worst-affected genes (missing count):")
        for gene, count in worst.items():
            print(f"  {gene}: {count} of {X.shape[0]} samples missing")


def clean(name, df):
    """
    Steps 3-5 - numeric conversion, drop dead/duplicate genes.
    Returns the cleaned dataframe and a small report dict.
    """
    y = df["class"]
    X = df.drop(columns=["class"])

    # Step 3: force everything numeric; '?' / text / blank -> NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    before_genes = X.shape[1]
    missing_before = int(X.isnull().sum().sum())

    # Step 4a: drop genes that are entirely missing (can't impute from nothing)
    all_missing = X.columns[X.isnull().all()]
    X = X.drop(columns=all_missing)

    # Step 4b: drop constant (zero-variance) genes - no information
    #          (computed ignoring NaN; a gene with one value + NaNs is constant)
    variances = X.var(axis=0, skipna=True)
    constant = variances[variances == 0].index
    X = X.drop(columns=constant)

    # Step 5: drop duplicate probe columns (identical values), keep first
    #         transpose so duplicated() compares columns, on non-NaN-filled view
    dup_mask = X.T.duplicated()
    duplicate_genes = X.columns[dup_mask.values]
    X = X.drop(columns=duplicate_genes)

    after_genes = X.shape[1]
    missing_after = int(X.isnull().sum().sum())

    report = {
        "genes_before": before_genes,
        "genes_after": after_genes,
        "all_missing_dropped": len(all_missing),
        "constant_dropped": len(constant),
        "duplicate_dropped": len(duplicate_genes),
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    cleaned = pd.concat([y, X], axis=1)
    return cleaned, report


def report(name, rep):
    """Step 6 - before/after data-quality report."""
    print(f"\n--- Data-quality report: {name} ---")
    print(f"Genes before cleaning: {rep['genes_before']}")
    print(f"Genes after cleaning:  {rep['genes_after']}  "
          f"(removed {rep['genes_before']-rep['genes_after']})")
    print(f"  - entirely-missing genes dropped: {rep['all_missing_dropped']}")
    print(f"  - constant (zero-variance) genes dropped: {rep['constant_dropped']}")
    print(f"  - duplicate probe genes dropped: {rep['duplicate_dropped']}")
    print(f"Missing cells before: {rep['missing_before']}")
    print(f"Missing cells after:  {rep['missing_after']}  "
          f"(remaining gaps handled by in-fold imputation, not here)")


def main():
    for name, path in DATASETS.items():
        df = pd.read_csv(path, na_values=["?"], low_memory=False)
        inspect(name, df)
        # profile missing on the numeric view
        X_numeric = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce")
        profile_missing(name, X_numeric)
        cleaned, rep = clean(name, df)
        report(name, rep)
        # save cleaned copy
        out = f"data/{name}_clean.csv"
        cleaned.to_csv(out, index=False)
        print(f"Saved cleaned dataset -> {out}")


if __name__ == "__main__":
    main()
