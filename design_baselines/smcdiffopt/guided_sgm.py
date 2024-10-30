"""
Implements a guided Langevin Monte Carlo sampler.
"""

import os
import math
from functools import partial

import torch
import numpy as np

from .diffusion import ScoreBased, register_sampler
from .diffusion_utils import extract_and_expand, expand_as, get_model_fn

@register_sampler('sgm_smcmc')
class LangevinSMCMC(ScoreBased):
    def __init__(
        self,
        H_func=None,  # the inverse problem operator
        noiser=None,  # the noise model (as in DPS)
        objective_fn=None,  # the objective function
        sampling_task="inverse_problem",  # the task to be performed
        noise_sample_size=10,
        anneal=False,
        use_x0=False,
        saving_dir=None,
        max_hist = 100,
        **kwargs,
    ):
        #NOTE: this is kept the same as SMCDiffOpt
        super().__init__(**kwargs)
        self.H_func = H_func
        self.noiser = noiser
        self.objective_fn = objective_fn
        self.task = sampling_task
        self.anneal = anneal
        self.model_fn = get_model_fn(self.network, train=False)
        self.saving_dir = saving_dir
        self.max_hist = max_hist
        
    def save_history(self, x, idx):
        """
        Saves the history to disk.
        """
        filename = os.path.join(self.saving_dir, "history.pth")
        x = x.detach().cpu()
        
        # load
        if os.path.exists(filename):
            history = torch.load(filename)
        else:
            history = {idx: x}
            
        # save
        history[idx] = x
        
        # if self.max_hist is not None:
        #     if len(history) > self.max_hist:
        #         history.pop(min(history.keys()))
        torch.save(history, filename)
        
    def resample_from_history(self, all_weights, resample_method="max"):        
        """
        Pick the best particle from the history.
        Assuming the current particle is added to the history.
        
        Args:
            filename: the file to load the history from
            all_weights (dict): the weights of the particles, 
            keys are the indices of the particles and values are the weights.
        """
        filename = os.path.join(self.saving_dir, "history.pth")
        # max over the weights
        if resample_method == "max":
            max_idx = max(all_weights, key=all_weights.get)
        elif resample_method == "multinomial":
            keys = list(all_weights.keys())
            values = list(all_weights.values())
            values = np.array([v.item() for v in values])
            max_idx = np.random.choice(keys, p=values/values.sum())
        
        print(f"Resampling from {max_idx}")
        
        # get the history
        history = torch.load(filename)
        return history[max_idx].to(self.device)
    
    def eval_weight(self, x, t):
        """
        Evaluates the weight of the particle.
        """
        with torch.no_grad():
            denoised_x = self.tweedie_projection(x, t)
            model_input_shape = (self.sample_shape[0], *self.sample_shape[1:])
            objective = self.objective_fn(denoised_x.reshape(*model_input_shape)).numpy()
        
        return objective
    
    def get_top_k(self, all_weights, k=10, vec_t=None):
        """
        Get the top k particles.
        """
        top_samples = sorted(all_weights, key=all_weights.get, reverse=True)[:k]
        history = torch.load(os.path.join(self.saving_dir, "history.pth"))
        
        # top largest weights
        ret = [history[i].to(self.device) for i in top_samples]
        
        return np.array(
            [self.tweedie_projection(x, vec_t).detach().cpu().numpy() for x in ret]
        )
        
    def sample(self, 
               sample_shape, 
               noise_level=0.2,
               restart_interval=100,
               resample_interval=1,
               num_iter=1000,
               **kwargs):
        """
        Construsts a MCMC-PF sampler.
        """
        step_size = kwargs.get("step_size", None)
        # step_size = torch.tensor([0.1], device=self.device) 
        self.sample_shape = sample_shape
        evaluation_samples = kwargs.get("evaluation_samples", 10)
        
        with torch.no_grad():
            # intialize the state
            x_start = self.sde_object.prior_sampling(sample_shape).to(self.device)
            self.save_history(x_start, 0)
            
            w_start = self.eval_weight(x_start, torch.ones(sample_shape[0], device=self.device))
            
            weight_dict  = {0: w_start}
            x = x_start
            
            timesteps = torch.linspace(
                self.sde_object.T, self.eps, self.sde_object.N, device=self.device
            )
            
            for i in range(num_iter):
                vec_t = (
                        torch.ones(sample_shape[0], device=self.device) * noise_level
                )
                # langevin proposal
                x, x_tilde = self._corrector_update_fn(
                    x, t=vec_t, step_size=step_size, 
                )
                # x, x_tilde = self._half_denoising_update(
                #     x, t=vec_t, step_size=step_size, 
                # )
                
                self.save_history(x, i+1)
                w = self.eval_weight(x, vec_t)
                weight_dict[i+1] = w
                
                # resample
                if i % resample_interval == 0:
                    x = self.resample_from_history(weight_dict)
                
                # restart
                # device are handled by the other functions
                if i % restart_interval == 0:
                    x = self.sde_object.prior_sampling(sample_shape).to(self.device)
                    self.save_history(x, i+1)
                    w = self.eval_weight(x, vec_t)
                    weight_dict[i+1] = w
                    
                print(f"Iteration {i+1}/{num_iter}: objective {w}")
        
        
        ret = self.get_top_k(weight_dict, k=evaluation_samples, vec_t=vec_t)
        
        # inverse scaler
        return np.array([self.inverse_scaler(x) for x in ret])
        
    