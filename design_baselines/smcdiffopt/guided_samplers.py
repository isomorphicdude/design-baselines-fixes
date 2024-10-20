"""
Implements guided samplers with a base diffusion model.

The base diffusion is a GaussianDiffusion class and the guided samplers
will override p_sample and sample functions. Those choices are made to 
accomodate for both SMC and DPS-like samplers.
"""

import os
import math
from functools import partial

import torch
import numpy as np

from .diffusion import GaussianDiffusion, SpacedDiffusion, register_sampler
from .diffusion_utils import extract_and_expand, get_model_fn



@register_sampler("smcdiffopt")
class SMCDiffOpt(SpacedDiffusion):
    def __init__(
        self,
        H_func=None,  # the inverse problem operator
        noiser=None,  # the noise model (as in DPS)
        objective_fn=None,  # the objective function
        sampling_task="inverse_problem",  # the task to be performed
        noise_sample_size=10,
        anneal=False,
        use_x0=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.H_func = H_func
        self.noiser = noiser
        self.objective_fn = objective_fn
        self.task = sampling_task
        self.noise_sample_size = noise_sample_size
        self.num_timesteps = int(self.betas.shape[0])
        self.anneal = anneal
        self.use_x0 = use_x0
       
    @property
    def anneal_schedule(self):
        if self.task == "inverse_problem":
            return lambda t: 1.0
        elif self.task == "optimisation":
            # assume t starts from last time step (e.g. 999) and goes to 0
            if self.anneal:
                return lambda t:  (1 - self.sqrt_one_minus_alphas_cumprod[(t)])
            else:
                return lambda t: 1.0

    # TODO: this is direct import from flows, should clean up
    def get_proposal_X_t(self, num_t, x_t, eps_pred, method="default", **kwargs):
        return self._proposal_X_t(num_t, x_t, eps_pred)

    def get_log_potential(
        self,
        x_0_new,
        x_0_old,
        obs_new,
        obs_old,
        x_t_new,
        x_t_old,
        time_step,
        beta_scaling: float = 200.0,
    ):
        """
        Computes the log G(x_t, x_{t+1}) in FK model.

        Args:
            x_0_new (torch.Tensor): shape (batch*num_particles, dim_x)
            x_0_old (torch.Tensor): shape (batch*num_particles, dim_x)
            obs_new (torch.Tensor): shape (batch, dim_y)
            obs_old (torch.Tensor): shape (batch, dim_y)
            x_t_new (torch.Tensor): same but current x_t
            x_t_old (torch.Tensor): same but previous x_t
            time_step (int): time step
            beta_scaling (float): scaling factor for beta
            writer (SummaryWriter): tensorboard writer
            seed (int): random seed
            
        Returns:
            tuple of (weights, xt_mean, xt_max, x0_mean, x0_max)
        """
        if self.task == "inverse_problem":
            c_new = self.sqrt_alphas_cumprod[-(time_step + 1)]
            c_old = self.sqrt_alphas_cumprod[-time_step]
            d_new = self.sqrt_one_minus_alphas_cumprod[-(time_step + 1)]
            d_old = self.sqrt_one_minus_alphas_cumprod[-time_step]
            numerator = self._log_gauss_liklihood(x_t_new, obs_new, c_new, d_new)
            denominator = self._log_gauss_liklihood(x_t_old, obs_old, c_old, d_old)

        elif self.task == "optimisation":
            # sample new random vectors
            # expanded_shape = (x_new.shape[0], self.noise_sample_size, x_new.shape[1])
            # new_noise = torch.randn(expanded_shape, device=x_new.device)
            # old_noise = torch.randn(expanded_shape, device=x_old.device)
            
            # new_obj_input = (
            #     x_new_mean[:, None, :].repeat(1, self.noise_sample_size, 1)
            #     + std_new * new_noise
            # )
            # old_obj_input = (
            #     x_old_mean[:, None, :].repeat(1, self.noise_sample_size, 1)
            #     + std_old * old_noise
            # )

            # squeezed_shape = (x_new.shape[0] * self.noise_sample_size, -1)
            # _numerator = self.objective_fn(new_obj_input.reshape(*squeezed_shape))
            # _denominator = self.objective_fn(old_obj_input.reshape(*squeezed_shape))
                
            x_t_new_numerator = self.objective_fn(x_t_new).numpy()
            x_t_old_numerator = self.objective_fn(x_t_old).numpy()
            
            x_0_new_numerator = self.objective_fn(x_0_new).numpy()
            x_0_old_numerator = self.objective_fn(x_0_old).numpy()

            if self.use_x0:
                _numerator = x_0_new_numerator
                _denominator = x_0_old_numerator
            else:
                _numerator = x_t_new_numerator
                _denominator = x_t_old_numerator
                
            # to device
            numerator = torch.tensor(_numerator, device=self.device).squeeze(-1)
            denominator = torch.tensor(
                _denominator, device=self.device
            ).squeeze(-1)
            # numerator = numerator.reshape(*expanded_shape[:-1]).mean(dim=1)
            # denominator = denominator.reshape(*expanded_shape[:-1]).mean(dim=1)
        else:
            raise ValueError("Invalid task.")
        
        
        if time_step > 0:
            return (
                # original
                numerator * self.anneal_schedule(time_step - 1) * beta_scaling
                - denominator * self.anneal_schedule(time_step) * beta_scaling, 
                x_t_new_numerator.mean(), x_t_new_numerator.max(), x_0_old_numerator.mean(), x_0_old_numerator.max()
            )
        else:
            return (torch.zeros_like(numerator), x_t_new_numerator.mean(), x_t_new_numerator.max(), x_0_old_numerator.mean(), x_0_old_numerator.max())

        # return (
        #     # with dilation path
        #     numerator * self.anneal_schedule(time_step) * beta_scaling - math.log(self.anneal_schedule(time_step))
        #     - denominator * self.anneal_schedule(time_step - 1) * beta_scaling + math.log(self.anneal_schedule(time_step - 1))
        # )

        # return (
        #     # without annealing
        #     numerator * beta_scaling - denominator * beta_scaling
        # )

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

        ts = [i for i in range(1000) if i in self.use_timesteps]

        model_fn = get_model_fn(self.network, train=False)

        # flattened initial x, shape (batch * num_particles, dim_x)
        # where for images dim = 3*256*256
        model_input_shape = (self.shape[0] * num_particles, *self.shape[1:])
        x_t = torch.randn(
            (self.shape[0] * num_particles, np.prod(self.shape[1:])), device=self.device
        )
        
        # initial sampling step P(x_T)exp(beta * f(x_T))
        with torch.no_grad():
            if self.use_x0:
                vec_0 = torch.ones(self.shape[0] * num_particles).to(x_t.device) * ts[-1]
                eps_pred = model_fn(x_t.view(*model_input_shape), vec_0)
                # get x_0
                x_0 = self.get_tweedie_est(self.num_timesteps-1, x_t, eps_pred)
                f_val = self.objective_fn(x_0.view(*model_input_shape)).numpy()
                log_weights = beta_scaling * f_val
            else:
                # directly evaluate f(x_T)
                log_weights = beta_scaling * self.objective_fn(x_t.view(*model_input_shape)).numpy()
            
            if self.anneal:
                log_weights = log_weights * self.anneal_schedule(self.num_timesteps - 1)
                
            log_weights = torch.tensor(log_weights, device=x_t.device).view(
                self.shape[0], num_particles
            )
            log_weights = log_weights - torch.logsumexp(
                    log_weights, dim=1, keepdim=True
            )
            
            # resample the initial particles
            resample_idx = self.resample(
                            torch.exp(log_weights).view(-1), method=resampling_method
            )
            x_t = (x_t.view(self.shape[0], num_particles, -1))[
                torch.arange(self.shape[0])[:, None], resample_idx.unsqueeze(0)
            ]
            
        
        with torch.no_grad():
            for i, num_t in enumerate(list(range(self.num_timesteps))[::-1]):
                if self.task == "inverse_problem":
                    y_new = self._get_obs(y_obs, i, num_particles, method="default")
                    y_old = self._get_obs(y_obs, i - 1, num_particles, method="default")
                elif self.task == "optimisation":
                    y_new = None
                    y_old = None

                vec_t = (torch.ones(self.shape[0]) * (ts[num_t])).to(x_t.device)

                # noise predicting model
                eps_pred = model_fn(x_t.view(*model_input_shape), vec_t)
                if eps_pred.shape[1] == 2 * self.shape[1]:
                    eps_pred, model_var_values = torch.split(
                        eps_pred, self.shape[1], dim=1
                    )

                x_new, x_mean_new, std_new, x_0_old = self._proposal_X_t(
                    num_t,
                    x_t.view(*model_input_shape),
                    eps_pred,
                    return_std=True
                )  # (batch * num_particles, 3, 256, 256)

                new_vec_t = (torch.ones(self.shape[0]) * (ts[num_t - 1])).to(x_t.device)
                eps_pred_new = model_fn(x_new.view(*model_input_shape), new_vec_t)
                
                x_0_new = self.get_tweedie_est(num_t-1, x_new, eps_pred_new)
                
                # new_noise = (self.sqrt_one_minus_alphas_cumprod)**2 * (1 - self.sqrt_one_minus_alphas_cumprod**2)
                # new_noise = self.sqrt_one_minus_alphas_cumprod if i <= self.num_timesteps//10000 else np.zeros_like(self.sqrt_one_minus_alphas_cumprod)
                # x_0_new += torch.randn_like(x_0_new, device=x_t.device) * extract_and_expand(new_noise,
                #     num_t, x_0_new)
                # x_0_old += torch.randn_like(x_0_old, device=x_t.device) * extract_and_expand(new_noise,
                #     num_t, x_0_old)
                
                
                x_input_shape = (self.shape[0] * num_particles, -1)
                if self.use_x0:
                    log_weights, xt_mean, xt_max, x0_mean, x0_max = self.get_log_potential(
                        x_0_new.view(*x_input_shape),
                        x_0_old.view(*x_input_shape),
                        y_new,
                        y_old,
                        x_new.view(*x_input_shape),
                        x_t.view(*x_input_shape),
                        num_t,
                        beta_scaling=beta_scaling,
                    )
                    log_weights = log_weights.view(self.shape[0], num_particles)
                else:
                    log_weights, xt_mean, xt_max, x0_mean, x0_max = self.get_log_potential(
                        x_new.view(*x_input_shape),
                        x_t.view(*x_input_shape),
                        y_new,
                        y_old,
                        x_new.view(*x_input_shape),
                        x_t.view(*x_input_shape),
                        num_t,
                        beta_scaling=beta_scaling,                    )
                    log_weights = log_weights.view(self.shape[0], num_particles)    

                # normalise weights
                log_weights = log_weights - torch.logsumexp(
                    log_weights, dim=1, keepdim=True
                )

                # ESS = 1 / sum(w_i^2)
                ess = torch.exp(-torch.logsumexp(2 * log_weights, dim=1)).item()
                writer.add_scalar(f"ESS_seed{seed}", ess, i)
                writer.add_scalar(f"SMCDiff_seed{seed}/xt_mean", xt_mean.item(), i)
                writer.add_scalar(f"SMCDiff_seed{seed}/xt_max", xt_max.item(), i)
                writer.add_scalar(f"SMCDiff_seed{seed}/x0_mean", x0_mean.item(), i)
                writer.add_scalar(f"SMCDiff_seed{seed}/x0_max", x0_max.item(), i)
                print(f"Iteration {num_t}, max value: {xt_max.item()}")
                
                if i != len(ts) - 1:
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

    def get_tweedie_est(self, timestep, x_t, eps_pred):
        """
        Returns the Tweedie estimator E[x_0|x_t].
        
        Args:
            timestep (int): time step, from 999 to 0
            x_t (torch.Tensor): x_t the current state
            eps_pred (torch.Tensor): epsilon_t the noise prediction
        """
        m = extract_and_expand(self.sqrt_alphas_cumprod, timestep, x_t)
        sqrt_1m_alpha = extract_and_expand(
            self.sqrt_one_minus_alphas_cumprod, timestep, x_t
        )
        x_0 = (x_t - sqrt_1m_alpha * eps_pred) / m
        return x_0

    def _proposal_X_t(self, timestep, x_t, eps_pred, return_std=False):
        """
        Sample x_{t-1} from x_{t} in the diffusion model as a naive proposal.
        Args:
            timestep (int): time step, from 999 to 0
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


# implements SVDD in Li et al. 2024
# TODO: use abstract class
# Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding
@register_sampler("svdd")
class SVDD(SMCDiffOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # here the number of particles refers to the MC samples for Importance Sampling
        num_particles = kwargs.get("num_particles", 10)
        score_output = kwargs.get("score_output", False)
        sampling_method = kwargs.get("sampling_method", "conditional")
        resampling_method = kwargs.get("resampling_method", "systematic")
        beta_scaling = kwargs.get("beta_scaling", 200.0)
        writer = kwargs.get("writer", None)
        seed = kwargs.get("seed", None)
        assert seed is not None, "Seed must be provided."

        ts = [i for i in range(1000) if i in self.use_timesteps]
        
        model_fn = get_model_fn(self.network, train=False)

        # initial x
        x_t = torch.randn((self.shape[0], 1, np.prod(self.shape[1:])), device=self.device)
        # initial sampling step P(x_T)exp(beta * f(x_T))
        
        model_input_shape = (self.shape[0] * num_particles, *self.shape[1:])
        split_input_shape = (self.shape[0], num_particles, *self.shape[1:])
        with torch.no_grad():
            x_t = (
                    x_t
                    .repeat(1, num_particles, 1)
                    .reshape(*model_input_shape)
            )
            if self.use_x0:
                vec_0 = torch.ones(self.shape[0] * num_particles).to(x_t.device) * ts[-1]
                eps_pred = model_fn(x_t.view(*model_input_shape), vec_0)
                # get x_0
                x_0 = self.get_tweedie_est(self.num_timesteps-1, x_t, eps_pred)
                f_val = self.objective_fn(x_0.view(*model_input_shape)).numpy()
                log_weights = beta_scaling * f_val
            else:
                # directly evaluate f(x_T)
                log_weights = beta_scaling * self.objective_fn(x_t.view(*model_input_shape)).numpy()
            
            if self.anneal:
                log_weights = log_weights * self.anneal_schedule(self.num_timesteps - 1)
                
            log_weights = torch.tensor(log_weights, device=x_t.device).view(
                self.shape[0], num_particles
            )
            
            # resample the initial particles
            log_weights = log_weights - torch.logsumexp(
                    log_weights, dim=1, keepdim=True
            )
            resample_idx = torch.multinomial(
                    torch.exp(log_weights), 1, replacement=True
                )
            x_t = (x_t.view(*split_input_shape))[
                    torch.arange(self.shape[0])[:, None], resample_idx.unsqueeze(0)
            ]
            
            x_t = x_t.squeeze(0)

        with torch.no_grad():
            for i, num_t in enumerate(list(range(self.num_timesteps))[::-1]):
                y_new = None
                y_old = None

                # need to repeat x_t to get from (batch, dim_x) to (batch*num_particles, dim_x)
                x_t = (
                    x_t
                    .repeat(1, num_particles, 1)
                    .reshape(*model_input_shape)
                )

                vec_t = (torch.ones(model_input_shape[0]) * (ts[num_t])).to(
                    x_t.device
                )
                
                
                # noise predicting model
                eps_pred = model_fn(x_t.view(*model_input_shape), vec_t)

                x_new, x_mean_new, std, x0 = self._proposal_X_t(
                    num_t,
                    x_t.view(*model_input_shape),
                    eps_pred,
                    return_std=True
                )  # (batch * num_particles, 3, 256, 256)
                # print(f"Time: {num_t}, std: {std}")
                # get tweedie estimates
                if num_t > 0:
                    new_vec_t =  (torch.ones(model_input_shape[0]) * (ts[num_t-1])).to(
                    x_t.device
                )
                    eps_pred_new = model_fn(x_new.view(*model_input_shape), new_vec_t)
                    x_0_new = self.get_tweedie_est(num_t-1, x_new, eps_pred_new)
                    # x_0_new += torch.randn_like(x_0_new, device=x_t.device) * self.sqrt_one_minus_alphas_cumprod[num_t-1] 
                else:
                    x_0_new = x_new
                
                
                objective_val = self.objective_fn(x_0_new.view(*model_input_shape))
                
                writer.add_scalar(f"SVDD_seed{seed}/x0_mean", objective_val.numpy().mean().item(), i)
                writer.add_scalar(f"SVDD_seed{seed}/x0_max", objective_val.numpy().max().item(), i)
                
                # additional evaluation
                xt_objective_val = self.objective_fn(x_new.view(*model_input_shape))
                writer.add_scalar(f"SVDD_seed{seed}/xt_mean", xt_objective_val.numpy().mean().item(), i)
                writer.add_scalar(f"SVDD_seed{seed}/xt_max", xt_objective_val.numpy().max().item(), i)
                
                print(f"Iteration {num_t}, max value: {xt_objective_val.numpy().max().item()}")
                log_weights = beta_scaling * objective_val
                
                log_weights = torch.tensor(log_weights.numpy(), device=x_t.device).view(
                    self.shape[0], num_particles
                )

                # normalise weights
                log_weights = log_weights - torch.logsumexp(
                    log_weights, dim=1, keepdim=True
                )

                # No ESS here
                # ess = torch.exp(-torch.logsumexp(2 * log_weights, dim=1)).item()
                # writer.add_scalar(f"ESS_seed{seed}", ess, i)
                resample_idx = torch.multinomial(
                    torch.exp(log_weights), 1, replacement=True
                )
                
                # resample_idx = torch.argmax(
                #     log_weights, dim=1, keepdim=True
                # )

                # only sample the first particle
                x_new = (x_new.view(*split_input_shape))[
                    torch.arange(self.shape[0])[:, None], resample_idx.unsqueeze(0)
                ] # somehow this is (1, batch, 1, dim_x)
                
                x_t = x_new.squeeze(0)

                if return_list:
                    samples.append(
                        x_t.reshape(self.shape[0], *self.shape[1:])
                    )
                

        # apply inverse scaler
        if return_list:
            for i in range(len(samples)):
                samples[i] = self.inverse_scaler(samples[i].squeeze())
            return samples, torch.exp(log_weights)
        else:
            return self.inverse_scaler(
                x_t.view(self.shape[0], *self.shape[1:]).squeeze()
            )

@register_sampler("unconditional")
class Unconditional(SMCDiffOpt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
        x0_samples = []
        xt_samples = []
        val_samples = kwargs.get("val_samples", None)

        ts = [i for i in range(1000) if i in self.use_timesteps]

        model_fn = get_model_fn(self.network, train=False)

        # flattened initial x, shape (batch * num_particles, dim_x)
        # where for images dim = 3*256*256
        x_start = kwargs.get("x_start", None)
        if x_start is not None:
            x_t = x_start
        else:
            x_t = torch.randn(
                (self.shape[0], np.prod(self.shape[1:])), device=self.device
            )

        with torch.no_grad():
            for i, num_t in enumerate(list(range(self.num_timesteps))[::-1]):
                vec_t = (torch.ones(x_t.shape[0]) * (ts[num_t])).to(x_t.device)
        
                model_input_shape = (self.shape[0], *self.shape[1:])

                # noise predicting model
                eps_pred = model_fn(x_t, vec_t)

                x_new, x_mean_new, std_new, x_0_old = self._proposal_X_t(
                    num_t,
                    x_t,
                    eps_pred,
                    return_std=True
                )  

                x_t = x_new

                if return_list:
                    x0_samples.append(x_0_old.detach().cpu().numpy())
                    xt_samples.append(x_t.detach().cpu().numpy())
                    

        # apply inverse scaler
        if return_list:
            for i in range(len(xt_samples)):
                xt_samples[i] = self.inverse_scaler(xt_samples[i].squeeze())
                x0_samples[i] = self.inverse_scaler(x0_samples[i].squeeze())
            return x0_samples, xt_samples
        else:
            return self.inverse_scaler(x_t)