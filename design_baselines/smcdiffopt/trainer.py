"""
Implements training functions for the diffusion models.
"""

import os
import time

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging

from .diffusion import GaussianDiffusion

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# TODO: Add checkpointing and other useful features
def train_model(
    diffusion_model: GaussianDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    writer: SummaryWriter,
    device: str = "cpu",
    logging_dir: str = "smcdiffopt",
):
    losses = []
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            # random integer time between 0 and 999
            t = torch.randint(0, 1000, size=(1,), device=device)
            loss = torch.mean(diffusion_model.train_loss_fn(x, t))
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), epoch)
            losses.append(loss.item())
        if epoch % 10 == 0:
            val_loss = 0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_loss += diffusion_model.train_loss_fn(x, t).item()
            writer.add_scalar("Loss/val", val_loss, epoch)

        logging.info(f"Epoch {epoch} - Loss: {loss.item()}")
        if epoch % 100 == 0:
            torch.save(
                diffusion_model.state_dict(),
                os.path.join(logging_dir, f"model_{epoch+1}.pt"),
            )
    return losses
