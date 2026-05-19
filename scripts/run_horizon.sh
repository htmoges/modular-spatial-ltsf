#!/bin/bash
# Table 6: Horizon sweep — NLinear on Electricity at H in {96, 192, 336, 720}
# Runs L+PH+R+S at each horizon.

set -e

DATA_ROOT="${DATA_ROOT:-./data}"
SEEDS="42 1 2 3"
CFG="configs/horizon/nlinear_electricity.yaml"

echo "Table 6 — Horizon sweep (NLinear, Electricity)"
echo "Data: ${DATA_ROOT}"
echo ""

for PRED_LEN in 96 192 336 720; do
  echo "── H=${PRED_LEN} ──"
  python -m gdf.train --config ${CFG} --seeds ${SEEDS} --data-root ${DATA_ROOT} \
    --pred-len ${PRED_LEN} \
    --use-temporal-head --use-input-residual --n-bands 1 \
    --experiment-name "horizon_h${PRED_LEN}_LPHRS"
done

echo ""
echo "Done. Results in results/"
