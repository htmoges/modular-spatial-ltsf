#!/bin/bash
# Table 5: PatchTST attribution study
# Runs Base and +R+S for each dataset.
# To reproduce all four columns, use the attribution flags below.
#
# Attribution flags:
#   Base:   --no-use-input-residual --n-bands 0
#   +R:     --use-input-residual    --n-bands 0
#   +R+S:   --use-input-residual    --n-bands 1  (default config)
#   +S:     --no-use-input-residual --n-bands 1

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
SEEDS="42 1 2 3"

echo "Table 5 — PatchTST attribution study"
echo "Data: ${DATA_ROOT}"
echo ""

for DATASET in electricity traffic weather; do
  CFG="configs/patchtst/${DATASET}.yaml"
  echo "── patchtst/${DATASET} ──"

  echo "  [Base]"
  python -m gdf.train --config ${CFG} --seeds ${SEEDS} --data-root ${DATA_ROOT} \
    --no-use-input-residual --n-bands 0 \
    --experiment-name "patchtst_${DATASET}_Base"

  echo "  [+R+S]"
  python -m gdf.train --config ${CFG} --seeds ${SEEDS} --data-root ${DATA_ROOT} \
    --use-input-residual --n-bands 1 \
    --experiment-name "patchtst_${DATASET}_RS"

done

echo ""
echo "Done. Results in results/"
