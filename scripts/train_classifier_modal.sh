#!/bin/bash
# Train time-dependent classifier on Modal

set -e

modal run modal_app.py --action train_classifier \
  --config configs/classifier_modal.yaml
