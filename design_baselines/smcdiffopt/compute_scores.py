"""Implements utility functions for computing scores for SMCDiffOpt from saved scores."""

import os
import click
import json
import numpy as np

@click.command()
@click.option("--dir", default="smcdiffopt", help="Directory containing saved scores.")
@click.option("--task", default="Superconductor-RandomForest-v0", help="Task name.")
def compute_scores_from_dir(dir, task):
    """Computes mean and std of scores from a directory containing saved scores."""
    percentile_keys = ["50", "75", "90", "100"]
    scores = {k: [] for k in percentile_keys}
    num_files = 0
    dir = os.path.join(dir, task)
    for filename in os.listdir(dir):
        if filename.endswith(".json") and filename.startswith("norm_score"):
            with open(os.path.join(dir, filename), "r") as f:
                data = json.load(f)
                for k in percentile_keys:
                    scores[k].append(data[k])
            num_files += 1
                    
    # compute mean and std
    mean_scores = {k: np.mean(scores[k]) for k in percentile_keys}
    std_scores = {k: np.std(scores[k]) for k in percentile_keys}
    ci_scores = {k: 1.96 * std_scores[k] / np.sqrt(num_files) for k in percentile_keys}
    
    # write to txt
    with open(os.path.join(dir, "aggregated_scores.txt"), "w") as f:
        f.write("Mean, Std, CI\n")
        for k in percentile_keys:
            f.write(f"{mean_scores[k]}, {std_scores[k]}, {ci_scores[k]}\n")
            
    return mean_scores, std_scores, ci_scores
            

if __name__ == "__main__":
    compute_scores_from_dir()
               
               
               
               
    