#!/usr/bin/env bash
# Download Graph-SST2 (DIG sentiment graph data) needed by SST2P / GraphSST2Planted.
# DIG's SentiGraphDataset does NOT auto-download; we fetch from the Drive link in
# dig/xgraph/dataset/nlp_dataset.py and place files as storage/datasets/GraphSST2/raw/GraphSST2_*.pkl
# Usage:  bash get_sst2_data.sh
set -e
cd "$(dirname "$(readlink -f "$0")")"          # repo root, wherever it is cloned
# activate a virtualenv if VENV points at one, otherwise use the current interpreter
[ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export PATH="$PWD:$PATH"  # bare goodtg -> clean-repo wrapper

RAW=storage/datasets/GraphSST2/raw
mkdir -p "$RAW"
STAGE=$(mktemp -d)
FILEID=1-PiLsjepzT8AboGMYLdVHmmXPpgR8eK1

echo "=== downloading Graph-SST2 from Google Drive (id $FILEID) ==="
gdown "https://drive.google.com/uc?id=$FILEID" -O "$STAGE/graphsst2.zip" || {
  echo "gdown failed (Drive quota or no internet on this node). Try another node, or"
  echo "download manually from https://drive.google.com/file/d/$FILEID/view and unzip into $RAW"
  exit 1
}

echo "=== extracting ==="
unzip -o "$STAGE/graphsst2.zip" -d "$STAGE" >/dev/null

echo "=== staging tree (for inspection) ==="
find "$STAGE" -maxdepth 3 -type f | head -30

# Find the *_node_features.pkl and copy the whole sibling set, normalizing the
# prefix to 'GraphSST2_' (DIG ships them as 'Graph-SST2_*' with a hyphen).
SRCDIR=$(dirname "$(find "$STAGE" -iname "*node_features.pkl" | head -1)")
if [ -z "$SRCDIR" ]; then echo "ERROR: no *_node_features.pkl in the zip"; exit 1; fi
echo "=== source dir: $SRCDIR ==="
for f in "$SRCDIR"/*; do
  base=$(basename "$f")
  newname=$(echo "$base" | sed -E 's/^Graph[-_]?SST2_/GraphSST2_/')
  cp -f "$f" "$RAW/$newname"
  echo "  placed $newname"
done

echo "=== checking required files ==="
miss=0
for f in node_features node_indicator sentence_tokens edge_index graph_labels split_indices; do
  if ls "$RAW"/GraphSST2_${f}.pkl >/dev/null 2>&1; then echo "  [ok] GraphSST2_${f}.pkl"; else echo "  [missing?] GraphSST2_${f}.pkl"; miss=1; fi
done
rm -rf "$STAGE"
[ $miss -eq 0 ] && echo "=== Graph-SST2 raw data ready ===" || echo "=== Some files missing; ls $RAW and tell me the names ==="
ls -l "$RAW"
