#!/bin/bash

for SEED in 1 2 3 4 5; do
#     python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 10 --task "Superconductor-RandomForest-v0" --noise-sample-size 1
    # python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 100 --task "Superconductor-RandomForest-v0" --noise-sample-size 1
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 10 --task "Superconductor-RandomForest-v0" --noise-sample-size 10 --smooth-schedule "flow"
    python design_baselines/smcdiffopt/__init__.py --seed $SEED --num-timesteps 100 --beta-scaling 10 --task "Superconductor-RandomForest-v0" --noise-sample-size 10 --smooth-schedule "diffusion"
    # python design_baselines/smcdiffopt/__init__.py --retrain-model False  --task "TFBind8-Exact-v0" --no-task-relabel
    
done
python design_baselines/smcdiffopt/compute_scores.py --task "Superconductor-RandomForest-v0"



