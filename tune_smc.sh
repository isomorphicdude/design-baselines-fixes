#!/bin/bash

for SEED in 1 2 3; do
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 10 --task "Superconductor-RandomForest-v0" --noise-sample-size 1
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 100 --task "Superconductor-RandomForest-v0" --noise-sample-size 1
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 100 --task "Superconductor-RandomForest-v0" --noise-sample-size 10
done
python design_baselines/smcdiffopt/compute_scores.py --task "Superconductor-RandomForest-v0"



