#!/bin/bash
# Table 4: Linear backbone attribution study (DLinear / NLinear / RLinear)
# Runs L and L+PH+R+S for each backbone × dataset combination.
# To reproduce all four columns, add the --use/no variants shown in the comments.
#
# Attribution flags:
#   L:         --no-use-temporal-head --no-use-input-residual --n-bands 0
#   L+PH:      --use-temporal-head    --no-use-input-residual --n-bands 0
#   L+PH+R:    --use-temporal-head    --use-input-residual    --n-bands 0
#   L+PH+R+S:  --use-temporal-head    --use-input-residual    --n-bands 1  (default config)

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
SEEDS="42 1 2 3"

echo "Table 4 — Linear attribution study"
echo "Data: ${DATA_ROOT}"
echo ""

for BACKBONE in dlinear nlinear rlinear; do
  for DATASET in electricity traffic weather; do
    CFG="configs/linear/${BACKBONE}_${DATASET}.yaml"
    echo "── ${BACKBONE}/${DATASET} ──"

    echo "  [L]"
    python -m gdf.train --config ${CFG} --seeds ${SEEDS} --data-root ${DATA_ROOT} \
      --no-use-temporal-head --no-use-input-residual --n-bands 0 \
      --experiment-name "${BACKBONE}_${DATASET}_L"

    echo "  [L+PH+R+S]"
    python -m gdf.train --config ${CFG} --seeds ${SEEDS} --data-root ${DATA_ROOT} \
      --use-temporal-head --use-input-residual --n-bands 1 \
      --experiment-name "${BACKBONE}_${DATASET}_LPHRS"

  done
done

echo ""
echo "Done. Results in results/"
