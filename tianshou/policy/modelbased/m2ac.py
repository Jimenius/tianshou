from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import BasePolicy, MBPOPolicy


class M2ACPolicy(MBPOPolicy):
    """Implementation of Masked Model-based Actor Critic. arXiv:2010.04893.

    M2AC builds on MBPO, but uses rollout samples with low uncertainty.

    :param BasePolicy policy: a model-free method for policy optimization.
    :param nn.Module model: the transition dynamics model.
    :param torch.optim.Optimizer model_optim: the optimizer for model training.
    :param Type[ReplayBuffer] model_buffer_type: type of buffer to store
        model rollouts.
    :param Callable loss_fn: loss function for model training.
    :param Callable terminal_fn: terminal function for model rollout.
    :param float real_ratio: ratio of data from real environment interactions when
        learning the policy.
    :param int virtual_env_num: number of parallel rollouts.
    :param int ensemble_size: number of sub-networks in the ensemble.
    :param int num_elites: number of elite sub-networks.
    :param bool deterministic_model_eval: whether to rollout deterministically.
    :param float rew_penalty_coeff: reward penalty coefficient.
    :param int max_rollout_length: maximum rollout length in each rollout.
    :param bool stop_uncertain: stop rollouts for high uncertainty states.
    :param Dict[str, Any] model_args: arguments for model learning.
    :param Dict[str, Any] model_buffer_args: Extra arguments for the
        model replay buffer.
    :param Optional[Union[str, int, torch.device]] device: device for training.

    .. seealso::

        Please refer to :class:`~tianshou.policy.MBPOPolicy`
    """

    def __init__(
        self,
        policy: BasePolicy,
        model: torch.nn.Module,
        model_optim: torch.optim.Optimizer,
        model_buffer_type: Type[ReplayBuffer],
        loss_fn: Callable,
        terminal_fn: Callable,
        real_ratio: float = 0.1,
        virtual_env_num: int = 100000,
        ensemble_size: int = 7,
        num_elites: int = 5,
        deterministic_model_eval: bool = False,
        rew_penalty_coeff: float = 0.001,
        max_rolloout_length: int = 1,
        stop_uncertain: bool = False,
        model_args: Dict[str, Any] = {},
        model_buffer_args: Dict[str, Any] = {},
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            policy,
            model,
            model_optim,
            model_buffer_type,
            loss_fn,
            terminal_fn,
            real_ratio,
            virtual_env_num,
            ensemble_size,
            num_elites,
            deterministic_model_eval,
            model_args,
            model_buffer_args,
            device,
        )
        self._rew_penalty_coeff = rew_penalty_coeff
        self.set_rollout_length(max_rolloout_length)
        self._stop_uncertain = stop_uncertain

    def _rollout_step(
        self, obs: np.ndarray, info: Dict[str, Any], **kwargs: Any
    ) -> Tuple[Batch, Dict[str, Any]]:
        batch = Batch(obs=obs, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
        act = to_numpy(self(batch).act)
        inputs = np.concatenate((obs, act), axis=-1)
        inputs = self.normalizer.transform(inputs)
        mean, logvar, _, _ = self.model(inputs)
        delta_obs, rew, _, uncertainty = self._choose_from_ensemble(mean, logvar)
        obs_next = obs + delta_obs
        done = self._terminal_fn(obs, act, obs_next)

        length = info.get("length", 1)
        if self._rollout_length > 1:
            keep_ratio = (self._rollout_length - length) / \
                (2 * (self._rollout_length + 1))
        else:
            keep_ratio = 0.5
        keep_num = int(len(obs) * keep_ratio)
        _, indices = torch.topk(-uncertainty, keep_num)
        uncertainty = to_numpy(uncertainty)
        indices = to_numpy(indices)
        rew -= self._rew_penalty_coeff * uncertainty

        batch.update(
            obs=obs[indices],
            act=act[indices],
            obs_next=obs_next[indices],
            rew=rew[indices],
            done=done[indices],
        )
        num_samples = info.get("num_samples", 0)
        if self._stop_uncertain:
            obs_next = obs_next[indices]
            done = done[indices]
        info.update(
            num_samples=num_samples + len(obs),
            obs_next=obs_next.copy(),
            done=done.copy(),
        )

        return batch, info

    # def _rollout_step(
    #     self, obs: np.ndarray, info: Dict[str, Any], **kwargs: Any
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    #     batch = Batch(obs=obs, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
    #     act = to_numpy(self(batch).act)
    #     inputs = np.concatenate((obs, act), axis=-1)
    #     inputs = self.normalizer.transform(inputs)
    #     mean, logvar, _, _ = self.model(inputs)
    #     _, batch_size, _ = mean.shape
    #     var = torch.exp(logvar)

    #     # Choose an output sample from the ensemble and compute its statistics.
    #     choice_indice = np.random.randint(self.num_elites, size=batch_size)
    #     batch_indice = np.arange(batch_size)
    #     sample_mean = mean[choice_indice, batch_indice, :]
    #     sample_var = var[choice_indice, batch_indice, :]
    #     sample_std = torch.sqrt(sample_var)
    #     sample_dist = Independent(Normal(sample_mean, sample_std), 1)
    #     # Compute the statistics of rest outputs
    #     rest_mean = (torch.sum(mean, dim=0) - sample_mean) / (self.ensemble_size - 1)
    #     rest_var = (torch.sum(var + mean**2, dim=0) - sample_mean**2 -
    #                 sample_var) / (self.ensemble_size - 1) - rest_mean**2
    #     rest_std = torch.sqrt(rest_var)
    #     rest_dist = Independent(Normal(rest_mean, rest_std), 1)
    #     # OvR uncertainty
    #     uncertainty = torch.distributions.kl.kl_divergence(sample_dist, rest_dist)
    #     # Masking
    #     length = info.get("length", 1)
    #     if self._rollout_length > 1:
    #         keep_ratio = (self._rollout_length - length) / \
    #             (2 * (self._rollout_length + 1))
    #     else:
    #         keep_ratio = 0.5
    #     keep_num = int(len(obs) * keep_ratio)
    #     _, indices = torch.topk(-uncertainty, keep_num)
    #     uncertainty = to_numpy(uncertainty)
    #     indices = to_numpy(indices)

    #     if self._deterministic_model_eval:
    #         sample = sample_mean
    #     else:
    #         sample = sample_dist.rsample()
    #     log_prob = sample_dist.log_prob(sample)
    #     sample = to_numpy(sample)
    #     obs_next = obs + sample[:, :-1]
    #     rew = sample[:, -1] - self._rew_penalty_coeff * uncertainty
    #     done = self._terminal_fn(obs, act, obs_next)
    #     step_info = np.full(batch_size, fill_value={})
    #     batch.update(
    #         obs=obs[indices],
    #         act=act[indices],
    #         obs_next=obs_next[indices],
    #         rew=rew[indices],
    #         done=done[indices],
    #         info=step_info[indices],
    #     )
    #     num_samples = info.get("num_samples", 0)
    #     if self._stop_uncertain:
    #         obs_next = obs_next[indices]
    #         done = done[indices]

    #     info.update(
    #         num_samples=num_samples + keep_num,
    #         obs_next=obs_next.copy(),
    #         done=done.copy(),
    #     )

    #     return batch, info
