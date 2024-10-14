"""
Implements training functions for the diffusion models.
"""

import os
import time

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import numpy as np
import logging
from tqdm import tqdm

from .diffusion import GaussianDiffusion, SpacedDiffusion

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# TODO: Add checkpointing and other useful features
def train_model(
    diffusion_model: SpacedDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    writer: SummaryWriter,
    device: str = "cpu",
    ckpt_dir: str = "smcdiffopt",
    print_every: int = 100,
    num_time_steps: int = 1000,
    verbose: bool = False,
):
    # load latest checkpoint
    if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
        logging.info("Loading latest checkpoint")
        latest_ckpt = max(
            [int((f.split("_")[1].split("."))[0]) for f in os.listdir(ckpt_dir)]
        )
        checkpoint = torch.load(
            os.path.join(ckpt_dir, f"checkpoint_{latest_ckpt}.pt")
        )
        diffusion_model.network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        logging.info(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    logging.info(f"Training for {num_epochs} epochs")
    logging.info(f"Model has {sum(p.numel() for p in diffusion_model.network.parameters())} parameters")
    tf.io.gfile.makedirs(ckpt_dir)
    
    losses = []
    diffusion_model.network.to(device)
    diffusion_model.network.train()
    
    global_step = 0
    for epoch in range(start_epoch, num_epochs):
        if verbose:
            progress_bar = tqdm(total=len(train_loader))
            progress_bar.set_description(f"Epoch {epoch}")
        diffusion_model.network.train()   
        
        for step, batch in enumerate(train_loader):
            batch = batch[0].to(device)
            
            # batched timesteps (helps convergence)
            timesteps = torch.randint(0, num_time_steps, (batch.shape[0],)).long().to(device)            
            loss = diffusion_model.train_loss_fn(batch, timesteps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            if verbose:
                progress_bar.update(1)
                progress_bar.set_postfix(**loss_logs)
            global_step += 1
            
            writer.add_scalar("Loss/train", loss.detach().item(), epoch)
        
        progress_bar.close()
        
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            # logging.info(f"Epoch {epoch} - Loss: {loss.item()}")
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': diffusion_model.network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }
            torch.save(
            checkpoint,
            os.path.join(ckpt_dir, f"checkpoint_{epoch}.pt"),
            )
    return losses
