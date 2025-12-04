#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${script_dir}/.venv/bin/python"
[ -x "$python_bin" ] || python_bin="python"

"${python_bin}" "${script_dir}/dgcytof_cli.py" \
  --name "dgcytof" \
  --output_dir "${script_dir}/out/data/analysis/default/dgcytof" \
  --train.data.matrix "${script_dir}/out/data/data_preprocessing/default/data_preprocessing.csv"
  --labels_train "${script_dir}/out/data/data_preprocessing/default/data_preprocessing_labels.txt"
  --test.data.matrix "${script_dir}/out/data/data_preprocessing/default/data_preprocessing.csv"
  --labels_test "${script_dir}/out/data/data_preprocessing/default/data_preprocessing_labels.txt"
  
data_import.test.labels.gz
data_import.test.matrix.gz
data_import.train.labels.gz
data_import.train.matrix.gz
  
  
  
  # --data.matrix "${script_dir}/out/data/data_preprocessing/default/data_preprocessing.csv" \
  # --data.true_labels "${script_dir}/out/data/data_preprocessing/default/data_preprocessing_labels.txt"