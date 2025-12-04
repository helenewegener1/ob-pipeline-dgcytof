#!/usr/bin/env python
"""
Omnibenchmark runner that mirrors run_agglomerative.py but drives the DGCyTOF
pipeline instead (https://github.com/lijcheng12/DGCyTOF/).

Input/output contract:
* Accepts the same CLI args as run_agglomerative.py (`--data.matrix`,
  `--data.true_labels`, `--output_dir`, `--name`).
* Emits a plain-text file with one predicted label per line (matching the
  formatting of `true_labels`).
"""

import argparse
import gzip
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "PyTorch is required to run DGCyTOF. Install with `pip install torch`."
    ) from exc

try:
    import dgcytof_local as DGCyTOF
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Missing dgcytof_local module. Ensure dgcytof_local.py is present."
    ) from exc


def _read_first_line(path):
    """Read the first line of a (possibly gzipped) file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return handle.readline()


def _has_header(first_line):
    """Heuristically decide whether the first line is a header row."""
    tokens = [tok for tok in first_line.replace(",", " ").split() if tok]
    if not tokens:
        return False
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return True
    return False


def load_labels(data_file):
    """
    Load labels as 1D array; keeps missing labels as NaN (needed for
    semi-supervised handling in preprocessing).
    """
    opener = gzip.open if data_file.endswith(".gz") else open
    with opener(data_file, "rt") as handle:
        series = pd.read_csv(
            handle,
            header=None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        ).iloc[:, 0]

    try:
        labels = pd.to_numeric(series, errors="coerce").to_numpy()
    except Exception as exc:
        raise ValueError("Invalid data structure, cannot parse labels.") from exc

    if labels.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    return labels


def load_dataset(data_file):
    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)
    df = pd.read_csv(
        data_file,
        sep=",",
        header=0 if has_header else None,
        compression="infer",
    )
    try:
        df = df.apply(pd.to_numeric)
    except ValueError as exc:
        raise ValueError("Data matrix contains non-numeric values.") from exc

    if not has_header:
        df.columns = [f"f{i}" for i in range(df.shape[1])]
    else:
        df.columns = [str(col) for col in df.columns]
    return df


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):  # pragma: no cover - passthrough
        return self.model(x)


def run_dgcytof(train_data, train_labels, test_data):
    """
    Train on the provided training dataset and predict labels for the test dataset.
    """

    # --- Sanity check ---
    if len(train_data) != len(train_labels):
        raise ValueError(
            f"Training data rows ({len(train_data)}) do not match "
            f"training labels length ({len(train_labels)})."
        )

    # Convert labels -> numeric -> 0-based
    train_labels_series = pd.to_numeric(pd.Series(train_labels), errors="coerce")
    train_labels_zero = train_labels_series - 1

    # Build dataframe for DGCyTOF preprocessing (requires a 'label' column)
    df_train = train_data.copy()
    df_train["label"] = train_labels_zero

    # --- DGCyTOF preprocessing (filters unlabeled rows, normalizes, etc.) ---
    X_train_proc, y_train_proc, _ = DGCyTOF.preprocessing(df_train, [])

    if y_train_proc.empty:
        raise ValueError("No labeled rows available after preprocessing.")

    y_train_proc = y_train_proc.astype(int)
    classes = sorted(y_train_proc.unique())
    num_classes = len(classes)

    if num_classes < 2:
        raise ValueError("Need at least two classes to train the classifier.")

    # --- Build PyTorch training dataset with 100% of training rows ---
    train_dataset = TensorDataset(
        torch.tensor(X_train_proc.values, dtype=torch.float32),
        torch.tensor(y_train_proc.values.astype(np.int64)),
    )

    # --- Initialize model ---
    model = SimpleClassifier(
        input_dim=X_train_proc.shape[1],
        num_classes=num_classes
    )

    train_params = {
        "batch_size": min(128, len(train_dataset)),
        "shuffle": True,
        "num_workers": 0,
    }

    # --- Train using all training data ---
    DGCyTOF.train_model(
        model,
        train_dataset,
        max_epochs=20,
        params_train=train_params
    )

    # --- Predict on TEST data ONLY ---
    model.eval()
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Return predictions in 1-based form
    return preds + 1


def main():
    parser = argparse.ArgumentParser(description="clustbench DGCyTOF runner")
    parser.add_argument(
        "--train.data.matrix",
        type=str,
        help="gz-compressed textfile containing the comma-separated data to be clustered.",
        required=True,
    )
    parser.add_argument(
        "--labels_train",
        type=str,
        help="gz-compressed textfile containing the comma-separated data to be clustered.",
        required=True,
    )
    parser.add_argument(
        "--test.data.matrix",
        type=str,
        help="gz-compressed textfile containing the comma-separated data to be clustered.",
        required=True,
    )
    parser.add_argument(
        "--labels_test",
        type=str,
        help="gz-compressed textfile containing the comma-separated data to be clustered.",
        required=True,
    )
    # parser.add_argument(
    #     "--data.matrix",
    #     type=str,
    #     help="gz-compressed textfile containing the comma-separated data to be clustered.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--data.true_labels",
    #     type=str,
    #     help="gz-compressed textfile with the true labels; used to select a range of ks.",
    #     required=True,
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory to store data files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of the dataset",
        default="clustbench",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    # Load training set
    train_matrix = load_dataset(args.train_data_matrix)
    train_labels = load_labels(args.labels_train)
    
    # Load test set
    test_matrix = load_dataset(args.test_data_matrix)
    test_labels = load_labels(args.labels_test)
    
    # Predict
    predictions = run_dgcytof(
      train_data=train_matrix,
      train_labels=train_labels,
      test_data=test_matrix,
    )

    if len(predictions) != len(test_labels):
        sys.stderr.write(
            f"[dgcytof_cli] Length mismatch: predictions={len(predictions)}, "
            f"truth={len(truth)}, data_rows={len(data_df)}, "
            f"nan_labels={int(pd.isna(truth).sum())}\n"
        )
        raise ValueError("Predictions and true labels have mismatched lengths.")

    name = args.name
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_predicted_labels.txt")
    output_labels = [
        "" if pd.isna(t) else f"{float(p):.1f}"
        for p, t in zip(predictions, test_labels)
    ]
    np.savetxt(output_path, np.array(output_labels, dtype=str), fmt="%s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime guard
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"\nError: {exc}\n")
        sys.exit(1)
