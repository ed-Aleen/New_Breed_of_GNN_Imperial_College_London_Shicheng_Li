#!/usr/bin/env bash
# Download MUTAG (Mutagenicity) raw data from PGExplainer and place it where the
# loader expects: storage/datasets/Mutagenicity/raw/
# Source: https://github.com/flyingdoog/PGExplainer/tree/master/dataset
# Usage:  bash get_mutag_data.sh
set -e
cd /vol/bitbucket/sl8025/gnn_deg_expl_clean
RAW=storage/datasets/Mutagenicity/raw
mkdir -p "$RAW"
cd "$RAW"

BASE=https://github.com/flyingdoog/PGExplainer/raw/master/dataset

echo "=== downloading Mutagenicity.zip ==="
wget -q --show-progress -O Mutagenicity.zip      "$BASE/Mutagenicity.zip"
echo "=== downloading Mutagenicity.pkl.zip ==="
wget -q --show-progress -O Mutagenicity.pkl.zip  "$BASE/Mutagenicity.pkl.zip"

echo "=== unzipping ==="
unzip -o Mutagenicity.zip
unzip -o Mutagenicity.pkl.zip

# Some zips extract into a nested Mutagenicity/ folder; flatten it.
if [ -d Mutagenicity ]; then
  mv -f Mutagenicity/* . 2>/dev/null || true
  rmdir Mutagenicity 2>/dev/null || true
fi

echo "=== checking expected files ==="
NEED=(Mutagenicity_A.txt Mutagenicity_edge_gt.txt Mutagenicity_edge_labels.txt \
      Mutagenicity_graph_indicator.txt Mutagenicity_graph_labels.txt \
      Mutagenicity_label_readme.txt Mutagenicity_node_labels.txt Mutagenicity.pkl)
miss=0
for f in "${NEED[@]}"; do
  if [ -f "$f" ]; then echo "  [ok] $f"; else echo "  [MISSING] $f"; miss=1; fi
done
[ $miss -eq 0 ] && echo "=== MUTAG raw data ready ===" \
                || echo "=== Some files missing -- check the zip contents (ls $RAW) ==="
