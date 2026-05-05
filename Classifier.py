import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

N_SPLITS = 5

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DomainFeatureDataset(Dataset):
    """Dataset for tabular domain features."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class TabularTransformerClassifier(nn.Module):
    """
    Transformer classifier for tabular numeric features.

    Each numeric feature is treated as one token.
    """

    def __init__(
        self,
        num_features,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.num_features = num_features

        self.feature_projection = nn.Linear(1, d_model)

        self.feature_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        """
        x shape:
            [batch_size, num_features]
        """
        x = x.unsqueeze(-1)

        x = self.feature_projection(x)
        x = x + self.feature_embedding

        x = self.transformer_encoder(x)

        # Mean pooling over feature tokens
        x = x.mean(dim=1)

        logits = self.classifier(x).squeeze(-1)

        return logits


def load_dataset(input_path):
    """Load features, labels, and domains from the input CSV file."""
    raw = pd.read_csv(input_path)

    X = raw.loc[:, FEATURE_START:FEATURE_END].to_numpy(dtype=np.float32)
    y = raw.loc[:, LABEL_COLUMN].to_numpy()
    domains = raw.loc[:, DOMAIN_COLUMN].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y).astype(np.float32)

    if len(label_encoder.classes_) != 2:
        raise ValueError(
            f"Expected binary classification, got classes: {label_encoder.classes_}"
        )

    print("Label mapping:")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name} -> {index}")

    return X, y, domains


def train_one_epoch(model, dataloader, optimizer, criterion):
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def predict(model, dataloader):
    """Generate prediction probabilities and labels."""
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)

            all_scores.extend(probabilities.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_score = np.array(all_scores)
    y_true = np.array(all_labels)

    y_pred = (y_score >= 0.5).astype(int)

    return y_true, y_score, y_pred


def run_cross_validation(X, y):
    """Run stratified 10-fold cross-validation using Transformer classifier."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
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

    num_features = X.shape[1]

    for fold_index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold_index}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit scaler only on training data to avoid data leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = DomainFeatureDataset(X_train, y_train)
        test_dataset = DomainFeatureDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = TabularTransformerClassifier(
            num_features=num_features,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
            )

            if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                print(f"  Epoch {epoch:02d} | loss = {train_loss:.4f}")

        y_true, y_score, y_pred = predict(model, test_loader)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_true, y_score)
        accuracy = accuracy_score(y_true, y_pred)

        results["precision"].append(precision)
        results["recall"].append(recall)
        results["pr_auc"].append(pr_auc)
        results["fpr"].append(fpr)
        results["tpr"].append(tpr)
        results["roc_auc"].append(roc_auc)
        results["accuracy"].append(accuracy)

        print(f"  Accuracy = {accuracy:.4f}")
        print(f"  PR-AUC   = {pr_auc:.4f}")
        print(f"  ROC-AUC  = {roc_auc:.4f}")

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
    print(f"Using device: {DEVICE}")

    X, y, domains = load_dataset(INPUT_PATH)

    results = run_cross_validation(X, y)

    summarize_results(results)


if __name__ == "__main__":
    main()
    
