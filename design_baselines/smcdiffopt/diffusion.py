"""
Implements the base unconditional diffusion models.
Code is adapted from Chung et al. (2023). 
A continuous version will be implemented in diffopt.
"""

import os
from abc import abstractmethod, ABC

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from .diffusion_utils import (
    get_mean_processor,
    get_var_processor,
    extract_and_expand,
    space_timesteps,
    get_named_beta_schedule,
)

__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(
    sampler,
    model,
    steps,
    shape,
    noise_schedule,
    model_mean_type,
    model_var_type,
    dynamic_threshold,
    clip_denoised,
    rescale_timesteps,
    timestep_respacing="",
    device="cpu",
    **kwargs,
):

    sampler = get_sampler(name=sampler)

    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]

    return sampler(
        model,
        betas=betas,
        shape=shape,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        dynamic_threshold=dynamic_threshold,
        clip_denoised=clip_denoised,
        rescale_timesteps=rescale_timesteps,
        # use_timesteps=space_timesteps(steps, timestep_respacing),
        device=device,
        **kwargs,
    )


#TODO: Extend this class to be a base class for all discrete-time diffusion models.
class GaussianDiffusion(ABC):
    """"""
    def __init__(
        self,
        model,
        betas,
        shape,
        model_mean_type,
        model_var_type,
        dynamic_threshold,
        clip_denoised,
        rescale_timesteps,
        device="cpu",
        eta=1.0,  # for DDIM
        scaler=None, # sklearn object #TODO: implement for images
    ):
        self.model = model
        self.device = device
        self.scaler = scaler
        self.shape = shape

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (
            self.betas <= 1
        ).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0]) # 1000 
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = np.sqrt(1.0 - self.alphas_cumprod_prev)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(
            model_mean_type,
            betas=betas,
            dynamic_threshold=dynamic_threshold,
            clip_denoised=clip_denoised,
        )

        self.var_processor = get_var_processor(model_var_type, betas=betas)

        self.eta = eta

    def forward_mean_var(self, x_0, t):
        """
        Get the forward process mean and variance q(x_t | x_0) for a given t.
        Args:
            x_0: the [N x C x ...] tensor of noiseless inputs.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
        Returns:
            tuple (mean, variance, log_variance), all of x_0's shape.
        """

        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_0) * x_0
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_0)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_0)

        return mean, variance, log_variance

    def forward_sample(self, x_0, t):
        """
        Diffuses the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        Args:
            x_0 (torch.Tensor): the [N x C x ...] tensor of noiseless inputs.
            t (int): the integer timestep.
            
        Returns:
            tuple (x_t, noise), both of x_0's shape.
        """
        noise = torch.randn_like(x_0, device=self.device)
        assert noise.shape == x_0.shape

        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_0)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_0)

        return coef1 * x_0 + coef2 * noise, noise


    def posterior_mean_var(self, x_0, x_t, t):
        """
        Computes the mean and variance of the backward process q(x_{t-1} | x_t, x_0).
        """
        assert x_0.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_0)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_0 + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(
            self.posterior_log_variance_clipped, t, x_t
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_0.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def backward_one_step(self, x_t, t):
        """
        Computes the mean and variance of the variational Markov chain p_theta(x_{t-1} | x_t).
        """
        model_output = self.model(x_t, self._scale_timesteps(t))
    
        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output, model_var_values = torch.split(model_output, x_t.shape[1], dim=1)
        else:
            # The name of variable is wrong. 
            # This will just provide shape information, and 
            # will not be used for calculating something important in variance.
            model_var_values = model_output
        
        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x_t, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}
    @abstractmethod
    def sample(self,
                    model,
                    x_start):
        """
        The function used for sampling from noise.
        """ 
        pass
    
    @abstractmethod
    def p_sample(self, x, t, model):
        """
        The function used for sampling from noise.
        """ 
        pass
    
    
    def train_loss_fn(self, data, t):
        """
        Computes the training loss.
        
        Args:
            data (torch.Tensor): The data batch.
            t (int): integer timestep.
        Returns:
            torch.Tensor: The training loss.
        """
        diffused, noise = self.forward_sample(data, t)
        eps_pred = self.model(diffused, self._scale_timesteps(t))   
        loss = torch.mean((eps_pred - noise) ** 2)
        return loss
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t.float()
    
    def inverse_scaler(self, x):
        if self.scaler is not None:
            if isinstance(self.scaler, StandardScaler):
                # convert to numpy and detaching from the graph
                x = x.detach().cpu().numpy()
                return self.scaler.inverse_transform(x)
            else:
                return self.scaler.inverse_transform(x)
        else:
            return x
    
    
@register_sampler("ddpm")
class DDPM(GaussianDiffusion):
    def sample(self, x_start):
        """
        The function used for sampling from noise.
        """ 
        x = x_start
        device = x_start.device
        with torch.no_grad():    
            int_timesteps = list(range(self.num_timesteps))[::-1]
            for idx in int_timesteps:
                time = torch.tensor([idx], device=device)
                out = self.p_sample(x=x, t=time)
                x = out['sample']
        return self.inverse_scaler(x)
    
    def p_sample(self, x, t):
        """
        The function used for sampling from noise.
        """ 
        out = self.backward_one_step(x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}