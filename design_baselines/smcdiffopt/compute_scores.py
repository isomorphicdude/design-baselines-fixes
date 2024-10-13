"""Implements utility functions for computing scores for SMCDiffOpt from saved scores."""

import os

import click
import json
import numpy as np
import pandas as pd
import tensorflow as tf 

@click.command()
@click.option("--method", default="smcdiffopt", help="Method name.")
@click.option("--task", default="Superconductor-RandomForest-v0", help="Task name.")
def compute_scores_from_dir(method, task):
    """Computes mean and std of scores from a directory containing saved scores."""
    percentile_keys = ["50", "75", "90", "100"]
    scores = {k: [] for k in percentile_keys}
    num_files = 0
    
    # experiments with dir names: smcdiffopt_{seed}_{beta_scaling}
    exps = []
    hyperparams = {} # {dir: (hyperparams, 100th percentile)}
    for dir in os.listdir("."):
        if os.path.isdir(dir) and dir.startswith(method):
            exps.append(dir)
            temp = None
            with open(os.path.join(dir, "hyperparams.json"), "r") as f:
                temp = json.load(f)
            hyperparams[dir] = [temp, None]
            
    for base_dir in exps:
        sub_dir = os.path.join(base_dir, task)
        for filename in os.listdir(sub_dir):
            if filename.endswith(".json") and filename.startswith("norm_score"):
                with open(os.path.join(sub_dir, filename), "r") as f:
                    data = json.load(f)
                    for k in percentile_keys:
                        scores[k].append(data[k])
                num_files += 1
                        
        # compute mean and std
        mean_scores = {k: np.mean(scores[k]) for k in percentile_keys}
        std_scores = {k: np.std(scores[k]) for k in percentile_keys}
        ci_scores = {k: 1.96 * std_scores[k] / np.sqrt(num_files) for k in percentile_keys}
        
        # write to txt
        with open(os.path.join(sub_dir, "aggregated_scores.txt"), "w") as f:
            f.write("Mean, Std, CI\n")
            for k in percentile_keys:
                f.write(f"{mean_scores[k]}, {std_scores[k]}, {ci_scores[k]}\n")
                
        # update hyperparams
        hyperparams[base_dir][1] = mean_scores["100"]
                
    # find best hyperparameters by 100th percentile
    tf.io.gfile.makedirs(os.path.join(method, task))
    with open(os.path.join(method, task, "collection.txt"), "w") as f:
        f.write(",".join(list(hyperparams.values())[0].keys()) + ",100th percentile\n")
        for k, v in hyperparams.items():
            f.write(",".join([str(x) for x in v[0].values()]) + f",{v[1]}\n")
    
    #TODO: automatically find best hyperparameters
    return mean_scores, std_scores, ci_scores
            

if __name__ == "__main__":
    compute_scores_from_dir()
               
               
               
               
    