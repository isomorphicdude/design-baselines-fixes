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
        anneal_schedule=None,  # function to anneal
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
        self.anneal_schedule = anneal_schedule

    # TODO: this is direct import from flows, should clean up
    def get_proposal_X_t(self, num_t, x_t, eps_pred, method="default", **kwargs):
        if method == "default":
            return self._proposal_X_t(num_t, x_t, eps_pred)
        elif method == "conditional":
            return self._proposal_X_t_y(num_t, x_t, eps_pred, **kwargs)
        else:
            raise ValueError("Invalid proposal method.")

    def get_log_potential(
        self,
        x_new,
        x_old,
        obs_new,
        obs_old,
        c_new,
        c_old,
        d_new,
        d_old,
        task="inverse_problem",
    ):
        """
        Computes the log G(x_t, x_{t+1}) in FK model.

        Args:
            x_new (torch.Tensor): shape (batch*num_particles, dim_x)
            x_old (torch.Tensor): shape (batch*num_particles, dim_x)
            obs_new (torch.Tensor): shape (batch, dim_y)
            obs_old (torch.Tensor): shape (batch, dim_y)
            task (str): the task to be performed
        """
        if task == "inverse_problem":
            numerator = self._log_gauss_liklihood(x_new, obs_new, c_new, d_new)
            denominator = self._log_gauss_liklihood(x_old, obs_old, c_old, d_old)
        elif task == "optimisation":
            numerator = self.model_log_prob(x_new)
            denominator = self.model_log_prob(x_old)

        return numerator - denominator

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

        # create y_obs sequence for filtering
        # data = self._construct_obs_sequence(y_obs)
        num_particles = kwargs.get("num_particles", 10)
        score_output = kwargs.get("score_output", False)
        sampling_method = kwargs.get("sampling_method", "conditional")
        resampling_method = kwargs.get("resampling_method", "systematic")

        ts = list(range(self.num_timesteps))[::-1]
        reverse_ts = ts[::-1]
        c_t_func = lambda t: self.sqrt_alphas_cumprod[-(t + 1)]
        c_t_prev_func = lambda t: self.sqrt_alphas_cumprod[-(t)]
        d_t_func = lambda t: self.sqrt_1m_alphas_cumprod[-(t + 1)]
        d_t_prev_func = lambda t: self.sqrt_1m_alphas_cumprod[-(t)]

        model_fn = get_model_fn(self.model, train=False)

        # flattened initial x, shape (batch * num_particles, dim_x)
        # where for images dim = 3*256*256
        x_t = torch.randn(
            self.shape[0] * num_particles, np.prod(self.shape[1:]), device=self.device
        )

        with torch.no_grad():
            for i, num_t in enumerate(reverse_ts):
                print(f"Sampling at time {num_t}.")
                y_new = self._get_obs(y_obs, i, num_particles, method="default")

                y_old = self._get_obs(y_obs, i - 1, num_particles, method="default")

                vec_t = (torch.ones(self.shape[0]) * (reverse_ts[i - 1])).to(x_t.device)

                # get model prediction
                # assume input is (N, 3, 256, 256)
                # here N = batch * num_particles
                model_input_shape = (self.shape[0] * num_particles, *self.shape[1:])

                if score_output:
                    eps_pred = (
                        model_fn(x_t.view(model_input_shape), vec_t)
                        * (-1)
                        * d_t_func(i)
                    )  # (batch * num_particles, 3, 256, 256)
                else:
                    eps_pred = model_fn(x_t.view(model_input_shape), vec_t)
                    # print(vec_t)
                    if eps_pred.shape[1] == 2 * self.shape[1]:
                        eps_pred, model_var_values = torch.split(
                            eps_pred, self.shape[1], dim=1
                        )

                x_new, x_mean_new = self.get_proposal_X_t(
                    num_t,
                    x_t.view(model_input_shape),
                    eps_pred,
                    method=sampling_method,
                    y_t=y_new,
                    c_t=c_t_func(i),
                    d_t=d_t_func(i),
                )  # (batch * num_particles, 3, 256, 256)

                # x_new = x_new.clamp(-clamp_to, clamp_to)
                x_input_shape = (self.shape[0] * num_particles, -1)
                log_weights = self.log_potential(
                    x_new.view(x_input_shape),
                    x_t.view(x_input_shape),
                    y_new,
                    y_old,
                    c_new=c_t_func(i),
                    c_old=c_t_prev_func(i),
                    d_new=d_t_func(i),
                    d_old=d_t_prev_func(i),
                ).view(self.shape[0], num_particles)

                # normalise weights
                log_weights = log_weights - torch.logsumexp(
                    log_weights, dim=1, keepdim=True
                )

                if i != len(reverse_ts) - 1:
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
                    # samples.append(
                    #     x_t.view(num_particles, self.shape[0], *self.shape[1:])[-1]
                    # )

        # average and apply inverse scaler
        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i])
            return samples, torch.exp(log_weights)
        else:
            return self.inverse_scaler(
                x_t.view(num_particles, self.shape[0], *self.shape[1:])
            )

    def p_sample(self, x, t, model):
        raise NotImplementedError("p_sample not implemented for SMCDiffOpt.")

    def _proposal_X_t(self, timestep, x_t, eps_pred, obs=None, return_std=False):
        """
        Sample x_{t-1} from x_{t} in the diffusion model as a naive proposal.
        Args:
            timestep (int): time step
            x_t (torch.Tensor): x_t
            eps_pred (torch.Tensor): epsilon_t

        Returns:
            x_{t-1} (torch.Tensor): x_{t-1}
            x_mean (torch.Tensor): mean of x_{t-1}
        """

        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]

        v = sqrt_1m_alpha**2

        alpha_cumprod = self.alphas_cumprod[timestep]

        alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]

        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep] ** 2

        x_0 = (x_t - sqrt_1m_alpha.to(x_t.device) * eps_pred) / m.to(x_t.device)

        coeff1 = (
            torch.sqrt((v_prev / v) * (1 - alpha_cumprod / alpha_cumprod_prev))
            * self.eta
        )
        coeff2 = torch.sqrt(v_prev - coeff1**2)
        x_mean = m_prev.to(x_t.device) * x_0 + coeff2.to(x_t.device) * eps_pred
        std = coeff1.to(x_t.device)

        new_x = x_mean + std * torch.randn_like(x_mean)
        if return_std:
            return new_x, x_mean, std
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
