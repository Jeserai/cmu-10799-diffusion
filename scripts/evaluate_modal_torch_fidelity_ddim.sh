#!/bin/bash
# Modal Torch-Fidelity Evaluation Script (DDIM)

set -e

METHOD="ddpm"
CHECKPOINT="YOUR_PATH"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=100
OVERRIDE=false
SAMPLER="ddim"

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --override) OVERRIDE=true; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

echo "=========================================="
echo "Modal Torch-Fidelity Evaluation (DDIM)"
echo "=========================================="
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Num steps: $NUM_STEPS"
echo "Override: $OVERRIDE"
echo "Sampler: $SAMPLER"
echo "=========================================="
echo ""
echo "Submitting to Modal..."
echo ""

MODAL_CMD="modal run modal_app.py::main --action evaluate_torch_fidelity \
    --method $METHOD \
    --checkpoint $CHECKPOINT \
    --metrics $METRICS \
    --num-samples $NUM_SAMPLES \
    --batch-size $BATCH_SIZE \
    --num-steps $NUM_STEPS \
    --sampler $SAMPLER"

if [ "$OVERRIDE" = true ]; then
    MODAL_CMD="$MODAL_CMD --override"
fi

eval $MODAL_CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
