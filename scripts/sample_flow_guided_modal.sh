#!/bin/bash
# Guided Flow Matching sampling/evaluation on Modal

set -e

MODE="sample"
if [ "$1" = "--eval" ]; then
  MODE="eval"
  shift
fi

FLOW_CKPT=${1:-/data/logs/flow_matching_modal/flow_matching_20260131_192117/checkpoints/flow_matching_final.pt}
CLASSIFIER_CKPT=${2:-/data/logs/classifier_20260216_040118/checkpoints/classifier_final.pt}
ATTR_NAME=${3:-Smiling}
shift 3 || true
OUTPUT_PATH=""
EXTRA_ARGS="$@"

if [ -n "$1" ] && [[ "$1" != "-"* ]]; then
  OUTPUT_PATH="$1"
  shift
  EXTRA_ARGS="$@"
fi

if [ "$MODE" = "eval" ]; then
  modal run modal_app.py --action evaluate_guided_torch_fidelity \
    --checkpoint "$FLOW_CKPT" \
    --classifier-checkpoint "$CLASSIFIER_CKPT" \
    --attr-name "$ATTR_NAME" \
    --guidance-scale 2.0 \
    --num-steps 200 \
    --num-samples 1000 \
    --batch-size 128 \
    --metrics kid \
    --report-classifier \
    --classifier-threshold 0.5 \
    $EXTRA_ARGS
else
  modal run modal_app.py --action sample_flow_guided \
    --checkpoint "$FLOW_CKPT" \
    --classifier-checkpoint "$CLASSIFIER_CKPT" \
    --attr-name "$ATTR_NAME" \
    --guidance-scale 2.0 \
    --num-steps 200 \
    --num-samples 64 \
    --batch-size 64 \
    ${OUTPUT_PATH:+--output "$OUTPUT_PATH"} \
    $EXTRA_ARGS
fi
