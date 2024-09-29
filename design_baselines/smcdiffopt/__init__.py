import os
import math


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import click
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.smcdiffopt.diffusion import create_sampler
from design_baselines.smcdiffopt.guided_samplers import SMCDiffOpt
from design_baselines.smcdiffopt.nets import FullyConnectedWithTime
from design_baselines.smcdiffopt.trainer import train_model

# set up the logger for info, different from the design_baselines logger
info_logger = logging.getLogger("info_logger")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.command()
@click.option(
    "--logging-dir",
    default="smcdiffopt",
    type=str,
    help="The directory in which tensorboard data is logged " "during the experiment.",
)
@click.option(
    "--task",
    type=str,
    default="Superconductor-RandomForest-v0",
    help="The name of the design-bench task to use during " "the experiment.",
)
@click.option(
    "--task-relabel/--no-task-relabel",
    default=True,
    type=bool,
    help="Whether to relabel the real Offline MBO data with "
    "predictions made by the oracle (this eliminates a "
    "train-test discrepency if the oracle is not an "
    "adequate model of the data).",
)
@click.option(
    "--task-max-samples",
    default=None,
    type=int,
    help="The maximum number of samples to include in the task "
    "visible training set, which can be left as None to not "
    "further subsample the training set.",
)
@click.option(
    "--task-distribution",
    default=None,
    type=str,
    help="The empirical distribution to be used when further "
    "subsampling the training set to the specific "
    "task_max_samples from the previous run argument.",
)
@click.option(
    "--normalize-ys/--no-normalize-ys",
    default=True,
    type=bool,
    help="Whether to normalize the y values in the Offline MBO "
    "dataset before performing model-based optimization.",
)
@click.option(
    "--normalize-xs/--no-normalize-xs",
    default=False,
    type=bool,
    help="Whether to normalize the x values in the Offline MBO "
    "dataset before performing model-based optimization. "
    "(note that x must not be discrete)",
)
@click.option('--evaluation-samples',
              default=128, type=int,
              help='The samples to generate when solving the model-based '
                   'optimization problem.')
def smcdiffopt(
    logging_dir,
    task,
    task_relabel,
    task_max_samples,
    task_distribution,
    normalize_ys,
    normalize_xs,
    evaluation_samples,
) -> None:
    """Main function for smcdiff_opt for model-based optimization."""
    params = dict(
        logging_dir=logging_dir,
        task=task,
        task_relabel=task_relabel,
        task_max_samples=task_max_samples,
        task_distribution=task_distribution,
        normalize_ys=normalize_ys,
        normalize_xs=normalize_xs,
    )

    logger = Logger(logging_dir)
    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(params, f)

    # create task
    logging.info("Creating task...")
    logging.info(f"Task is: {task}")
    task = StaticGraphTask(
        task,
        relabel=task_relabel,
        dataset_kwargs=dict(
            max_samples=task_max_samples, distribution=task_distribution
        ),
    )
    logging.info(f"Dimension is {task.x.shape[1]}")

    if task.is_discrete:
        # raise NotImplementedError(
        #     "SMC-DIFF-OPT does not support discrete x values for now."
        # )
        task.map_to_logits()
        
        
        
    # instantiate the diffusion model
    # data preprocessing
    train_x, val_x, train_y, val_y = train_test_split(task.x, task.y, test_size=0.1)
    
    train_x = train_x.reshape(train_x.shape[0], -1)
    val_x = val_x.reshape(val_x.shape[0], -1)

    # NOTE: we are standardise using the training data;
    # this is not done in the design baselines code, but we are using
    # the full dataset for MBO, so doesn't matter too much
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_x)
    val_data = scaler.transform(val_x)

    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_y))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_y))

    training_config = {
        "batch_size": 256,
        "num_epochs": 15_001,
        "learning_rate": 1e-3,
    }
    
    train_loader = DataLoader(
        train_dataset, batch_size=training_config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config["batch_size"], shuffle=False
    )

    
    # initialise the model
    model_config = {
        "steps": 1000,
        "shape": (1, np.prod(task.x.shape[1:])),
        "noise_schedule": "linear",
        "model_mean_type": "epsilon",
        "model_var_type": "fixed_large",
        "dynamic_threshold": False,
        "clip_denoised": False,
        "rescale_timesteps": False,
        "timestep_respacing": 1000,
        "device": "cuda",
        "scaler": scaler,
        "sampling_task": "optimisation",
        "objective_fn": lambda x: task.predict(scaler.inverse_transform(x)),
    }
    
    dim_x = np.prod(task.x.shape[1:])
    nn_model = FullyConnectedWithTime(dim_x, time_embed_size=4, max_t=999)
    diffusion_model = create_sampler(
        sampler="smcdiffopt", model=nn_model, **model_config
    )

    # try load weights of pre-trained diffusion model from logging directory
    try:
        logging.info("Loading pre-trained weights.")
        nn_model.load_state_dict(
            torch.load(os.path.join(logging_dir, f"model_{15001}.pt"), map_location="cpu")
        )
    except FileNotFoundError:
        logging.info("No pre-trained weights found, training model from scratch.")
        # if no pre-trained weights, train the model
        writer = SummaryWriter(log_dir=os.path.join(logging_dir, "logs"))
        losses = train_model(
            diffusion_model=diffusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=torch.optim.Adam(nn_model.parameters(), lr=training_config["learning_rate"]),
            num_epochs=training_config["num_epochs"],
            writer=writer,
            device=model_config["device"],
            logging_dir=logging_dir,
        )
        # load
        nn_model.load_state_dict(
            torch.load(os.path.join(logging_dir, f"model_{training_config['num_epochs']}.pt"), map_location="cpu")
        )
        
        
    # perform model-based optimization
    logging.info("Performing model-based optimization...")
    x_start = torch.randn(evaluation_samples, task.x.shape[1]).to(model_config["device"])
    diffusion_model.model.to(model_config["device"])
    diffusion_model.model.eval()
    x = diffusion_model.sample(
        x_start=x_start,
        y_obs=None,
        num_particles=evaluation_samples,
        sampling_method="default",
        resampling_method="systematic",
    )

    # evaluate and save the results
    solution = x if isinstance(x, np.ndarray) else x.cpu().detach().numpy()
    try:
        np.save(os.path.join(f"{logging_dir}", f"solution.npy"), solution)
    except FileNotFoundError:
        # pickle
        import pickle
        with open(os.path.join(f"{logging_dir}", f"solution.pkl"), "wb") as f:
            pickle.dump(solution, f)

    score = task.predict(solution)
    if task.is_normalized_y:
        score = task.denormalize_y(score)
    
    logger.record("score", score, 1000, percentile=True)
    logging.info(score)


if __name__ == "__main__":
    smcdiffopt()
