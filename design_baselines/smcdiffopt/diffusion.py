"""
Implements the base unconditional diffusion models.
Code is adapted from Chung et al. (2023). 
A continuous version will be implemented in diffopt.
"""

import os
import math
from abc import abstractmethod, ABC

import torch
from torch.nn import functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

from .diffusion_utils import (
    get_mean_processor,
    get_var_processor,
    extract_and_expand,
    expand_as,
    space_timesteps,
    get_named_beta_schedule,
)

__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            # raise NameError(f"Name {name} is already registered!")
            print(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        # raise NameError(f"Name {name} is not defined!")
        print(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(
    sampler,
    network,
    steps,
    shape,
    noise_schedule,
    model_mean_type,
    model_var_type,
    dynamic_threshold,
    clip_denoised,
    rescale_timesteps,
    timestep_respacing="ddim100",
    device="cpu",
    sde=None,
    eps=1e-3,
    saving_dir=None,
    **kwargs,
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if "sgm" not in sampler:
        base_model_kwargs = dict(
            network=network,
            betas=betas,
            shape=shape,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            dynamic_threshold=dynamic_threshold,
            clip_denoised=clip_denoised,
            rescale_timesteps=rescale_timesteps,
            use_timesteps=space_timesteps(steps, timestep_respacing),
            device=device,
        )
    else:
        base_model_kwargs = dict(
            network=network,
            shape=shape,
            sde=sde,
            device=device,
            eps=eps,
            saving_dir=saving_dir,
        )
        
    sampler = get_sampler(name=sampler)
    merged_kwargs = {**base_model_kwargs, **kwargs}
    
    return sampler(**merged_kwargs)



# TODO: Extend this class to be a base class for all discrete-time diffusion models.
class GaussianDiffusion(ABC):
    """"""

    def __init__(
        self,
        network,
        betas,
        shape,
        model_mean_type,
        model_var_type,
        dynamic_threshold,
        clip_denoised,
        rescale_timesteps,
        device="cpu",
        eta=1.0,  # for DDIM
        scaler=None,  # sklearn object #TODO: implement for images
    ):
        self.network = network
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

        self.num_timesteps = int(self.betas.shape[0])  # 1000
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_one_minus_alphas_cumprod_next = np.sqrt(1.0 - self.alphas_cumprod_next)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = np.sqrt(
            1.0 - self.alphas_cumprod_prev
        )
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
        noise = torch.randn_like(x_0).to(self.device)
        
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
        model_output = self.network(x_t, self._scale_timesteps(t))

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x_t.shape[1]:
            model_output, model_var_values = torch.split(
                model_output, x_t.shape[1], dim=1
            )
        else:
            # The name of variable is wrong.
            # This will just provide shape information, and
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(
            x_t, t, model_output
        )
        model_variance, model_log_variance = self.var_processor.get_variance(
            model_var_values, t
        )

        assert (
            model_mean.shape
            == model_log_variance.shape
            == pred_xstart.shape
            == x_t.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # @abstractmethod
    def sample(self, model, x_start):
        """
        The function used for sampling from noise.
        """
        pass

    # @abstractmethod
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
        eps_pred = self.network(diffused, t)
        loss = F.mse_loss(eps_pred, noise)
        return loss

    def _scale_timesteps(self, t):
        # if self.rescale_timesteps:
        #     return t.float() * (1000.0 / self.num_timesteps)
        # return t.float()
        raise NotImplementedError

    def inverse_scaler(self, x):
        if self.scaler is not None:
            if isinstance(self.scaler, StandardScaler):
                # convert to numpy and detaching from the graph
                x = x.detach().cpu().numpy()
                return self.scaler.inverse_transform(x)
            else:
                x = x.detach().cpu()
                return self.scaler.inverse_transform(x)
        else:
            return x


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_sample(self, x, t, model):
        raise NotImplementedError

    def sample(self, model, x_start):
        raise NotImplementedError



@register_sampler("ddpm")
class DDPM(SpacedDiffusion):
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
                x = out["sample"]
        return self.inverse_scaler(x)

    def p_sample(self, x, t):
        """
        The function used for sampling from noise.
        """
        out = self.backward_one_step(x, t)
        sample = out["mean"]

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}    

class ScoreBased(ABC):
    def __init__(
        self,
        network,
        sde,
        shape,
        device="cpu",
        scaler=None,  # sklearn object #TODO: implement for images,
        eps=1e-3,
        **kwargs,
    ):
        self.network = network
        self.sde_object = sde
        self.eps = eps
        self.device = device
        self.scaler = scaler
        self.shape = shape
        
        reduce_mean = True
        self.reduce_op = (
            torch.mean
            if reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )
        self.rsde_object = self.sde_object.reverse(self.network, probability_flow=False)
        
        
    def forward_diffusion(self, x_0, t):
        noise = torch.randn_like(x_0).to(self.device)
        mean, std = self.sde_object.marginal_prob(x_0, t)
        x_t = mean + expand_as(std, noise) * noise
        return x_t, noise, mean, std
        
        
    def train_loss_fn(self, data, t):
        """
        Computes the training loss.

        Args:
            data (torch.Tensor): The data batch.
            t (int): integer timestep.
        Returns:
            torch.Tensor: The training loss.
        """
        timesteps = torch.linspace(self.eps, 1.0, self.sde_object.N, device=self.device)
        perturbed_x, noise, mean, std = self.forward_diffusion(data, timesteps[t])
        # vec_t = torch.ones(x_0.shape[0], device=self.device) * t
        assert (
            t.shape[0] == data.shape[0]
        ), "Time steps batch must be the same length as the data batch"
        vec_t = t
        score = self.network(perturbed_x, vec_t)
        noise_pred = score * expand_as(std, noise) * (-1)
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    def tweedie_projection(self, x, t):
        std = self.sde_object.marginal_prob(x, t)[1]
        t = (t * (self.sde_object.N - 1) / self.sde_object.T).long()
        return x + expand_as(std, x) ** 2 * self.network(x, t)
    
    
    def predictor_update_fn(self, x, t):
        """Euler-Maruyama update."""
        dt = -1.0 / self.rsde_object.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde_object.sde(x, t)
        x_mean = x + drift * dt

        x = x_mean + diffusion * z * math.sqrt(-dt)
        return x, x_mean
    
    
    def sample(self, sample_shape, x_start=None, return_list=False):
        """
        The function used for sampling from noise.
        """
        assert sample_shape[1:] == self.shape, "Sample shape must match the model shape"
        
        if x_start is None:
            x_start = self.sde_object.prior_sampling(sample_shape).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            x = x_start
            timesteps = torch.linspace(
                self.sde_object.T, self.eps, self.sde_object.N, device=self.device
            )
            list_of_samples = []
            for i in range(self.sde_object.N):
                # print(f"Solving for timestep {i}")
                t = timesteps[i]
                vec_t = torch.ones(sample_shape[0], device=self.device) * t
                x, x_mean = self.predictor_update_fn(
                    x,
                    vec_t,
                )
                if return_list:
                    list_of_samples.append(x_mean.detach().cpu().numpy())
        if return_list:
            for i in range(len(list_of_samples)):
                list_of_samples[i] = self.inverse_scaler(list_of_samples[i])
            return x_mean, list_of_samples
        else:
            return self.inverse_scaler(x_mean)
        
        
    def _half_denoising_update(self, x, t, step_size=None):
        std = self.sde_object.marginal_prob(x, t)[1]
        if step_size is None:
            step_size = std**2 / 2

        new_noise = torch.randn_like(x)

        x_tilde = x + expand_as(std, new_noise) * new_noise

        grad_tilde = self.network(x_tilde, torch.round(t * 999))
        noise_tilde = torch.randn_like(x)
        x = (
            x_tilde
            + step_size * grad_tilde
            + expand_as(torch.sqrt(2 * step_size - std**2), noise_tilde) * noise_tilde
        )

        return x, x_tilde

    def half_denoising_sample(
        self,
        sample_shape,
        noise_level=0.4,
        num_iter=1000,
        burnin=0,
        thinning=1,
        annealing=False,
        step_size=None,
        in_notebook=False,
        final_time=1,
        inner_iters=100,
    ):
        if in_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        with torch.no_grad():
            
            x = self.sde_object.prior_sampling(sample_shape).to(self.device)
            list_of_samples = torch.zeros((num_iter, *x.shape), device='cpu')
            if not annealing:
                for i in tqdm(range(num_iter)):
                    vec_t = (
                        torch.ones(sample_shape[0], device=self.device) * noise_level
                    )
                                        
                    x, x_tilde = self._half_denoising_update(
                        x,
                        vec_t,
                        step_size=step_size,
                    )

                    # x = self.tweedie_projection(x, vec_t)
                    if i > burnin:
                        list_of_samples[i, ...] = self.tweedie_projection(x, vec_t).detach().cpu()
                        # list_of_samples[i, ...] = x.detach().cpu()
            else:
                list_of_samples = torch.zeros(
                    ((final_time-1) * inner_iters + num_iter, *x.shape), device='cpu'
                )
                timesteps = torch.linspace(
                self.sde_object.T, self.eps, self.sde_object.N, device=self.device
                )
                for i in tqdm(range(final_time)):
                    if i < final_time:
                        t = timesteps[i]
                        vec_t = torch.ones(sample_shape[0], device=self.device) * t
                    else:
                        t = self.sde_object.T * (final_time / self.sde_object.N)
                        vec_t = torch.ones(sample_shape[0], device=self.device) * t
                    if i >= final_time-1:
                        inner_iters = num_iter
                    for j in range(inner_iters):                        
                        x, x_tilde = self._half_denoising_update(
                            x,
                            vec_t,
                            step_size=None,
                        )
                        # if i * inner_iters + j > burnin and i < final_time-1:
                            # list_of_samples[i * inner_iters + j, ...] = x
                        if i >= final_time - 1:
                            list_of_samples[j, ...] = self.tweedie_projection(x, vec_t).detach().cpu()
                            
        return x, list_of_samples[burnin::thinning]
    
    def _baoab_corrector_update_fn(self, x, t, prev_noise=None, step_size=None):
        """
        Here t determines the noise level.
        """
        # includes the case of VE which is 1.0
        std = self.sde_object.marginal_prob(x, t)[1]

        grad = self.network(x, torch.round(t * 999))
        new_noise = torch.randn_like(x)
        if step_size is None:
            step_size = std**2 * 2
        x_mean = x + step_size[:, None, None, None] * grad
        noise = new_noise + prev_noise

        # NOTE: this step is actually wrong
        # should be std[:, None, None, None] instead
        # but works surprisingly well
        # x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        x = x_mean + noise * std[:, None, None, None]
        return x, x_mean, new_noise
    
    
    def baoab_sample(
        self,
        sample_shape,
        noise_level=0.4,
        num_iter=1000,
        burnin=0,
        thinning=1,       
        in_notebook=False, 
    ):
        if in_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        with torch.no_grad():
            x = torch.randn(sample_shape, device=self.device)
            prev_noise = torch.zeros_like(x)
            
            list_of_samples = torch.zeros((num_iter, *x.shape), device='cpu')
            
            for i in tqdm(range(num_iter)):
                vec_t = (
                    torch.ones(sample_shape[0], device=self.device) * noise_level
                )
                
                x, x_mean, prev_noise = self._baoab_corrector_update_fn(
                    x,
                    vec_t,
                    prev_noise=prev_noise,
                )

                # NOTE: this is not present in the original paper
                # but works surprisingly well combined with the wrong update above
                # x = self.tweedie_projection(x, vec_t)

                if i > burnin:
                    # list_of_samples[i, ...] = x.detach().cpu()
                    list_of_samples[i, ...] = self.tweedie_projection(x, vec_t).detach().cpu()
                    
        return x_mean, list_of_samples[burnin::thinning]
    
    
    def _corrector_update_fn(self, x, t, step_size=None, n_steps=1):
        """Updates the state with Langevin dynamics."""
        target_snr = 0.25
        timestep = (t * (self.sde_object.N - 1) / self.sde_object.T).long()
        alpha = self.sde_object.alphas.to(t.device)[timestep]
        
        for i in range(n_steps):
            grad = self.network(x, torch.round(t * 999))
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            if step_size is None:
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + expand_as(step_size, grad) * grad
            x = x_mean + expand_as(torch.sqrt(step_size * 2), noise) * noise
        return x, x_mean
    
    def corrector_sample(self, 
                         sample_shape, 
                         noise_level=0.4, 
                         num_iter=1000, 
                         burnin=0, 
                         thinning=1, 
                         n_steps=1,
                         in_notebook=False):
        if in_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        with torch.no_grad():
            x = self.sde_object.prior_sampling(sample_shape).to(self.device)
            list_of_samples = torch.zeros((num_iter, *x.shape), device='cpu')
            for i in tqdm(range(num_iter)):
                vec_t = (
                    torch.ones(sample_shape[0], device=self.device) * noise_level
                )
                x, x_mean = self._corrector_update_fn(
                    x,
                    vec_t,
                    n_steps=n_steps,
                )

                if i > burnin:
                    list_of_samples[i, ...] = self.tweedie_projection(x, vec_t).detach().cpu()
        return x_mean, list_of_samples[burnin::thinning]
                
    
    def inverse_scaler(self, x):
        if self.scaler is not None:
            if isinstance(self.scaler, StandardScaler):
                # convert to numpy and detaching from the graph
                return self.scaler.inverse_transform(x)
            else:
                x = x.detach().cpu()
                return self.scaler.inverse_transform(x)
        else:
            return x
