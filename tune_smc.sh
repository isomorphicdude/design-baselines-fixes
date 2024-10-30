#!/bin/bash

TASK="Superconductor-RandomForest-v0"
# TASK="TFBind8-Exact-v0"
METHOD="smcdiffopt"
# SCALING=1
for SEED in 10 20 30 40 50; do
    python design_baselines/smcdiffopt/__init__.py \
        --seed $SEED \
        --num-timesteps 100 \
        --beta-scaling 10 \
        --task $TASK  \
        --anneal False \
        --noise-sample-size 1 \
        --use-x0 True
    python design_baselines/smcdiffopt/__init__.py \
        --seed $SEED \
        --num-timesteps 100 \
        --beta-scaling 10 \
        --task $TASK  \
        --anneal False \
        --noise-sample-size 1 \
        --use-x0 False

    python design_baselines/smcdiffopt/__init__.py \
        --seed $SEED \
        --num-timesteps 100 \
        --beta-scaling 10 \
        --task $TASK  \
        --anneal True \
        --noise-sample-size 1 \
        --use-x0 True
        
    python design_baselines/smcdiffopt/__init__.py \
        --seed $SEED \
        --num-timesteps 100 \
        --beta-scaling 10 \
        --task $TASK  \
        --anneal True \
        --noise-sample-size 1 \
        --use-x0 False

    # python design_baselines/smcdiffopt/__init__.py \
    #     --seed $SEED \
    #     --num-timesteps 100 \
    #     --beta-scaling $SCALING \
    #     --task $TASK  \
    #     --anneal False \
    #     --use-x0 True \
    #     --method "svdd" \
    #     --noise-sample-size 100 \
    #     --evaluation-samples 2

#     # python design_baselines/smcdiffopt/__init__.py --retrain-model False  --task "TFBind8-Exact-v0" --no-task-relabel
    
done
python design_baselines/smcdiffopt/compute_scores.py --task $TASK --method $METHOD



