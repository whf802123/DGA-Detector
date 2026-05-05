import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
)


INPUT_PATH = "vectorized_feature_w_ranks_norm.txt"

FEATURE_START = "bi_rank"
FEATURE_END = "vowel_ratio"
LABEL_COLUMN = "class"
DOMAIN_COLUMN = "ip"

N_SPLITS = 10
RANDOM_SEED = 12345


def load_dataset(input_path):
    """Load features, labels, and domains from the input CSV file."""
    raw = pd.read_csv(input_path)

    X = raw.loc[:, FEATURE_START:FEATURE_END].to_numpy()
    y = raw.loc[:, LABEL_COLUMN].to_numpy()
    domains = raw.loc[:, DOMAIN_COLUMN].to_numpy()

    return X, y, domains


def run_cross_validation(X, y):
    """Run stratified 10-fold cross-validation."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    classifier = SVC(
        kernel="linear",
        probability=True,
        random_state=RANDOM_SEED,
    )

    results = {
        "precision": [],
        "recall": [],
        "pr_auc": [],
        "fpr": [],
        "tpr": [],
        "roc_auc": [],
        "accuracy": [],
    }

    for fold_index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold_index}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_score = classifier.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)

        results["precision"].append(precision)
        results["recall"].append(recall)
        results["pr_auc"].append(auc(recall, precision))
        results["fpr"].append(fpr)
        results["tpr"].append(tpr)
        results["roc_auc"].append(roc_auc_score(y_test, y_score))
        results["accuracy"].append(accuracy_score(y_test, y_pred))

    return results


def summarize_results(results):
    """Print mean and standard deviation of metrics."""
    accuracy = np.array(results["accuracy"])
    pr_auc = np.array(results["pr_auc"])
    roc_auc = np.array(results["roc_auc"])

    print("\nCross-validation Summary")
    print("------------------------")
    print(f"Accuracy : {accuracy.mean():.4f} ± {accuracy.std():.4f}")
    print(f"PR-AUC   : {pr_auc.mean():.4f} ± {pr_auc.std():.4f}")
    print(f"ROC-AUC  : {roc_auc.mean():.4f} ± {roc_auc.std():.4f}")


def main():
    X, y, domains = load_dataset(INPUT_PATH)

    results = run_cross_validation(X, y)

    summarize_results(results)


if __name__ == "__main__":
    main()
  
