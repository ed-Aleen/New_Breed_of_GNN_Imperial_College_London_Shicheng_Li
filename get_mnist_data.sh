#!/usr/bin/env bash
# Link MNIST-75sp raw data from the existing GSAT project into this repo's
# expected location, then this repo's process() will build its own .pt files.
# Usage:  bash get_mnist_data.sh
set -e
cd /vol/bitbucket/sl8025/gnn_deg_expl_clean
SRC=/vol/bitbucket/sl8025/GSAT/data/mnist/raw
DST=storage/datasets/MNIST/raw
mkdir -p "$DST"

for f in mnist_75sp_train.pkl mnist_75sp_test.pkl \
         mnist_75sp_train_superpixels.pkl mnist_75sp_test_superpixels.pkl; do
  if [ -f "$SRC/$f" ]; then
    ln -sf "$SRC/$f" "$DST/$f"
    echo "  [linked] $f"
  else
    echo "  [MISSING in source] $f"
  fi
done
echo "=== MNIST raw data linked into $DST ==="
ls -l "$DST"
