#!/bin/bash

for SEED in 1 2 3 4 5; do
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num_timesteps 1000 --beta-scaling 10 --task "Superconductor-RandomForest-v0"
done
python design_baselines/smcdiffopt/compute_scores.py --task "Superconductor-RandomForest-v0"

for SEED in 1 2 3 4 5; do
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num_timesteps 256 --beta-scaling 200 --task "TFBind8-Exact-v0" --no-task-relabel
done
python design_baselines/smcdiffopt/compute_scores.py --task "TFBind8-Exact-v0" 

for SEED in 1 2 3 4 5; do
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num_timesteps 256 --beta-scaling 200 --task "TFBind10-Exact-v0" --no-task-relabel
done
python design_baselines/smcdiffopt/compute_scores.py --task "TFBind10-Exact-v0" 