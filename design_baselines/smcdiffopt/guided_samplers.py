"""
Implements guided samplers with a base diffusion model.

The base diffusion is a GaussianDiffusion class and the guided samplers
will override p_sample and sample functions. Those choices are made to 
accomodate for both SMC and DPS-like samplers.
"""

import os
import math

import torch
import numpy as np

from .diffusion import GaussianDiffusion, register_sampler
from .diffusion_utils import extract_and_expand, get_model_fn


@register_sampler("smcdiffopt")
class SMCDiffOpt(GaussianDiffusion):

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
        eta=1,
        scaler=None,
        H_func=None,  # the inverse problem operator
        noiser=None,  # the noise model (as in DPS)
        objective_fn=None,  # the objective function
        sampling_task="inverse_problem",  # the task to be performed
    ):
        super().__init__(
            model,
            betas,
            shape,
            model_mean_type,
            model_var_type,
            dynamic_threshold,
            clip_denoised,
            rescale_timesteps,
            device,
            eta,
            scaler,
        )
        self.H_func = H_func
        self.noiser = noiser
        self.objective_fn = objective_fn
        self.task = sampling_task

    @property
    def anneal_schedule(self):
        if self.task == "inverse_problem":
            return lambda t: 1.0
        elif self.task == "optimisation":
            return lambda t: 1 - self.sqrt_one_minus_alphas_cumprod[-(t + 1)]

    # TODO: this is direct import from flows, should clean up
    def get_proposal_X_t(self, num_t, x_t, eps_pred, method="default", **kwargs):
        if method == "default":
            return self._proposal_X_t(num_t, x_t, eps_pred)
        elif method == "conditional":
            c_t = self.sqrt_alphas_cumprod[-(num_t + 1)]
            d_t = self.sqrt_1m_alphas_cumprod[-(num_t + 1)]
            return self._proposal_X_t_y(num_t, x_t, eps_pred, kwargs["y_t"], c_t, d_t)
        else:
            raise ValueError("Invalid proposal method.")

    def get_log_potential(
        self,
        x_new,
        x_old,
        obs_new,
        obs_old,
        time_step: int,
        beta_scaling: float = 200.0,
        eps_pred_new=None,
        eps_pred_old=None,
        writer=None,
        seed=None,
    ):
        """
        Computes the log G(x_t, x_{t+1}) in FK model.

        Args:
            x_new (torch.Tensor): shape (batch*num_particles, dim_x)
            x_old (torch.Tensor): shape (batch*num_particles, dim_x)
            obs_new (torch.Tensor): shape (batch, dim_y)
            obs_old (torch.Tensor): shape (batch, dim_y)
            time_step (int): time step
            beta_scaling (float): scaling factor for beta
        """
        if self.task == "inverse_problem":
            c_new = self.sqrt_alphas_cumprod[-(time_step + 1)]
            c_old = self.sqrt_alphas_cumprod[-time_step]
            d_new = self.sqrt_one_minus_alphas_cumprod[-(time_step + 1)]
            d_old = self.sqrt_one_minus_alphas_cumprod[-time_step]
            numerator = self._log_gauss_liklihood(x_new, obs_new, c_new, d_new)
            denominator = self._log_gauss_liklihood(x_old, obs_old, c_old, d_old)

        elif self.task == "optimisation":
            # x_new_0_pred = self._proposal_X_t(
            #     time_step, x_new, eps_pred_new, return_std=True
            # )[-1]
            
            # x_old_0_pred = self._proposal_X_t(
            #     time_step - 1, x_old, eps_pred_old, return_std=True
            # )[-1]
            x_new_0_pred = x_new
            x_old_0_pred = x_old
            
            numerator = self.objective_fn(x_new_0_pred)
            denominator = self.objective_fn(x_old_0_pred)

            # to device
            numerator = torch.tensor(numerator.numpy(), device=self.device)
            denominator = torch.tensor(denominator.numpy(), device=self.device)
            print(f"Iteration {time_step}, mean value: {numerator.mean()}")
            writer.add_scalar(f"Objective_seed{seed}/mean", numerator.mean(), time_step)
        else:
            raise ValueError("Invalid task.")

        return (
            numerator * self.anneal_schedule(time_step) * beta_scaling
            - denominator * self.anneal_schedule(time_step - 1) * beta_scaling
        )

    def resample(self, weights, method="systematic"):
        if method == "systematic":
            return self._systematic_resample(weights)
        elif method == "multinomial":
            return self._multinomial_resample(weights)
        else:
            raise ValueError("Invalid resampling method.")

    def sample(
        self,
        y_obs,
        return_list=False,
        **kwargs,
    ):
        """
        Returns both samples and weights.
        Adapted from smcdiffopt codebase.
        """
        samples = []

        num_particles = kwargs.get("num_particles", 10)
        score_output = kwargs.get("score_output", False)
        sampling_method = kwargs.get("sampling_method", "conditional")
        resampling_method = kwargs.get("resampling_method", "systematic")
        beta_scaling = kwargs.get("beta_scaling", 200.0)
        writer = kwargs.get("writer", None)
        seed = kwargs.get("seed", None)
        assert seed is not None, "Seed must be provided."

        ts = list(range(self.num_timesteps))[::-1]
        reverse_ts = ts[::-1]

        d_t_func = lambda t: self.sqrt_1m_alphas_cumprod[-(t + 1)]

        model_fn = get_model_fn(self.model, train=False)

        # flattened initial x, shape (batch * num_particles, dim_x)
        # where for images dim = 3*256*256
        x_t = torch.randn(
            self.shape[0] * num_particles, np.prod(self.shape[1:]), device=self.device
        )

        with torch.no_grad():
            for i, num_t in enumerate(reverse_ts):
                if self.task == "inverse_problem":
                    y_new = self._get_obs(y_obs, i, num_particles, method="default")

                    y_old = self._get_obs(y_obs, i - 1, num_particles, method="default")
                elif self.task == "optimisation":
                    y_new = None
                    y_old = None

                vec_t = (torch.ones(self.shape[0]) * (reverse_ts[i - 1])).to(x_t.device)

                model_input_shape = (self.shape[0] * num_particles, *self.shape[1:])

                if score_output:
                    # score predicting model
                    eps_pred = (
                        model_fn(x_t.view(*model_input_shape), vec_t)
                        * (-1)
                        * d_t_func(i)
                    )  # (batch * num_particles, 3, 256, 256)
                else:
                    # noise predicting model
                    eps_pred = model_fn(x_t.view(*model_input_shape), vec_t)
                    if eps_pred.shape[1] == 2 * self.shape[1]:
                        eps_pred, model_var_values = torch.split(
                            eps_pred, self.shape[1], dim=1
                        )

                x_new, x_mean_new = self.get_proposal_X_t(
                    num_t,
                    x_t.view(*model_input_shape),
                    eps_pred,
                    method=sampling_method,
                )  # (batch * num_particles, 3, 256, 256)
                
                new_vec_t = (torch.ones(self.shape[0]) * (reverse_ts[i])).to(x_t.device)
                eps_pred_new = model_fn(x_new.view(*model_input_shape), new_vec_t)

                # x_new = x_new.clamp(-clamp_to, clamp_to)
                x_input_shape = (self.shape[0] * num_particles, -1)
                log_weights = self.get_log_potential(
                    x_new.view(*x_input_shape),
                    x_t.view(*x_input_shape),
                    y_new,
                    y_old,
                    i,
                    beta_scaling=beta_scaling,
                    eps_pred_new=eps_pred_new,
                    eps_pred_old=eps_pred,
                    writer=writer,
                    seed=seed,
                ).view(self.shape[0], num_particles)

                # normalise weights
                log_weights = log_weights - torch.logsumexp(
                    log_weights, dim=1, keepdim=True
                )
                
                # ESS = 1 / sum(w_i^2)
                ess = torch.exp(-torch.logsumexp(2 * log_weights, dim=1)).item()  
                writer.add_scalar(f"ESS_seed{seed}", ess, i)

                if i != len(reverse_ts) - 1 and ess < 0.5 * num_particles:
                    resample_idx = self.resample(
                        torch.exp(log_weights).view(-1), method=resampling_method
                    )
                    x_new = (x_new.view(self.shape[0], num_particles, -1))[
                        torch.arange(self.shape[0])[:, None], resample_idx.unsqueeze(0)
                    ]

                x_t = x_new

                if return_list:
                    samples.append(
                        x_t.reshape(self.shape[0] * num_particles, *self.shape[1:])
                    )

        # apply inverse scaler
        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i].squeeze())
            return samples, torch.exp(log_weights)
        else:
            return self.inverse_scaler(
                x_t.view(num_particles, self.shape[0], *self.shape[1:]).squeeze()
            )

    def p_sample(self, x, t, model):
        raise NotImplementedError("p_sample not implemented for SMCDiffOpt.")

    def _proposal_X_t(self, timestep, x_t, eps_pred, return_std=False):
        """
        Sample x_{t-1} from x_{t} in the diffusion model as a naive proposal.
        Args:
            timestep (int): time step
            x_t (torch.Tensor): x_t
            eps_pred (torch.Tensor): epsilon_t

        Returns:
            (tuple): tuple containing: new_x, x_mean; and 
            if return_std is True, also returns std and x_0
        """

        m = extract_and_expand(self.sqrt_alphas_cumprod, timestep, x_t)
        sqrt_1m_alpha = extract_and_expand(
            self.sqrt_one_minus_alphas_cumprod, timestep, x_t
        )

        v = sqrt_1m_alpha**2

        alpha_cumprod = extract_and_expand(self.alphas_cumprod, timestep, x_t)

        alpha_cumprod_prev = extract_and_expand(self.alphas_cumprod_prev, timestep, x_t)

        m_prev = extract_and_expand(self.sqrt_alphas_cumprod_prev, timestep, x_t)

        v_prev = (
            extract_and_expand(self.sqrt_one_minus_alphas_cumprod_prev, timestep, x_t)
            ** 2
        )

        x_0 = (x_t - sqrt_1m_alpha * eps_pred) / m

        coeff1 = (
            torch.sqrt((v_prev / v) * (1 - alpha_cumprod / alpha_cumprod_prev))
            * self.eta
        )
        coeff2 = torch.sqrt(v_prev - coeff1**2)
        x_mean = m_prev * x_0 + coeff2 * eps_pred
        std = coeff1

        new_x = x_mean + std * torch.randn_like(x_mean, device=x_t.device)
        if return_std:
            return new_x, x_mean, std, x_0
        else:
            return new_x, x_mean

    def _proposal_X_t_y(self, timestep, x_t, eps_pred, y_t, c_t, d_t):
        """
        The optimal proposal \proto g^y(x_t) p(x_t | x_{t+1}) used in Dou & Song 2024.
        But the covariance is different, as y_t = c_t * y_0
        which introduces complex dependencies.
        """
        new_x, x_mean, std = self._proposal_X_t(
            timestep, x_t, eps_pred, return_std=True
        )

        if std == 0:
            std = 1e-6
        # Sigma^-1 y
        rescaled_y = self.H_func.HHt_inv(
            vec=y_t.view(y_t.shape[0], -1),
            r_t_2=d_t**2,
            sigma_y_2=c_t**2 * self.noiser.sigma**2,
        )
        _mean = 1 / std**2 * x_mean.reshape(x_mean.shape[0], -1) + self.H_func.Ht(
            rescaled_y
        )

        # some similar hack to get the inverse of the covariance
        mean = self.H_func.cov_inv_diff_opt(
            vec=_mean.view(_mean.shape[0], -1),
            std=std,
            d_t=d_t,
            c_t=c_t,
            sigma_y_2=self.noiser.sigma**2,
        )

        # noise
        z = torch.randn_like(x_t.view(x_t.shape[0], -1), device=x_t.device)
        scaled_z = self.H_func.scale_noise_diff_opt(
            z, std=std, d_t=d_t, c_t=c_t, sigma_y_2=self.noiser.sigma**2
        )

        new_x = mean + scaled_z

        return new_x, mean

    def _log_gauss_liklihood(self, x_t, y_t, c_t, d_t):
        """
        Computes (22) in thesis:
        N(y_t; mean=Ax_t, cov = c_t^2 * sigma_y^2 I + d_t^2 A A^T)

        Args:
            x_t (torch.Tensor): shape (batch*num_particles, dim_x)
            y_t (torch.Tensor): shape (batch, dim_y)
            c_t (float): drift
            d_t (float): diffusion
        Returns:
            log_prob (torch.Tensor): shape (batch, )
        """
        sigma_y = self.noiser.sigma

        # noiseless may cause division by zero
        modified_singulars = c_t**2 * sigma_y**2 + d_t**2 * self.H_func.add_zeros(
            self.H_func.singulars() ** 2
        )

        logdet = torch.sum(torch.log(modified_singulars))

        # matrix vector product of (Cov)^-1 @ (y - Ax)
        diff = (
            y_t.unsqueeze(1)
            - self.H_func.H(x_t).reshape(y_t.shape[0], -1, y_t.shape[1])
        ).reshape(
            -1, y_t.shape[1]
        )  # (batch*num_particles, dim_y)

        cov_y_xt = self.H_func.HHt_inv(
            vec=diff,
            r_t_2=d_t**2,
            sigma_y_2=c_t**2 * sigma_y**2,
        )  # (batch*num_particles, dim_y)

        norm_diff = torch.sum(diff * cov_y_xt, dim=1)

        return -0.5 * logdet - 0.5 * norm_diff

    def _systematic_resample(self, weights):
        """
        Perform systematic resampling of the given weights.

        Args:
            weights (torch.Tensor): shape (N, )
        Returns:
            idx (torch.Tensor): shape (N, )
        """
        N = len(weights)

        positions = ((torch.rand(1) + torch.arange(N)) / N).to(weights.device)

        cumulative_sum = torch.cumsum(weights, dim=0)
        idx = torch.searchsorted(cumulative_sum, positions)

        idx = torch.min(idx, torch.tensor(N - 1, device=weights.device))
        return idx

    def _multinomial_resample(self, weights):
        """
        Perform multinomial resampling of the given weights.

        Args:
            weights (torch.Tensor): shape (N, )
        Returns:
            idx (torch.Tensor): shape (N, )
        """
        N = len(weights)
        idx = torch.multinomial(weights, N, replacement=True)
        return idx

    def _get_obs(self, obs, t, num_particles, method="conditional"):
        """
        Get y_obs at time t.
        """
        # if method == "default":
        return obs * self.sqrt_alphas_cumprod[-(t + 1)]
