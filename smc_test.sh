#!/bin/bash

for SEED in 1 2 3 4 5; do
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num_timesteps 256
done
python design_baselines/smcdiffopt/compute_scores.py