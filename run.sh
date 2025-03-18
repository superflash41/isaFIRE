#! /usr/bin/env bash
set -e  # Exit on first error

echo ">>> training Xception..."
python src/pipeline/train_xception.py

echo ">>> training DenseNet..."
python src/pipeline/train_dense.py

echo ">>> training ResNet..."
python src/pipeline/train_res.py

echo ">>> building ensemble..."
python src/pipeline/ensemble.py

echo ">>> evaluating models..."
python src/eval/evaluate.py

echo "pipeline complete!"
