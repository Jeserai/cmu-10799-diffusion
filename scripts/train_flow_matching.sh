#!/bin/bash
# Train flow matching with Modal config

set -e

modal run modal_app.py --action train \
  --method flow_matching \
  --config configs/flow_matching_modal.yaml
