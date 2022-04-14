import itertools
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import BasePolicy, DynaPolicy
from tianshou.utils.statistics import Normalizer


class MBPOPolicy(DynaPolicy):
    """Implementation of Model-Based Policy Optimization. arXiv:1906.08253.

    MBPO builds on Dyna framework with branch rollout.

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
    :param Dict[str, Any] model_args: arguments for model learning.
    :param Dict[str, Any] model_buffer_args: Extra arguments for the
        model replay buffer.
    :param Optional[Union[str, int, torch.device]] device: device for training.

    .. seealso::

        Please refer to :class:`~tianshou.policy.DynaPolicy`
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
            real_ratio=real_ratio,
            virtual_env_num=virtual_env_num,
            model_args=model_args
        )

        self.reset_model_buffer(virtual_env_num, **model_buffer_args)
        self.ensemble_size = ensemble_size
        self.num_elites = num_elites
        self.elite_indices = np.arange(num_elites)
        self._deterministic_model_eval = deterministic_model_eval
        self._loss_fn = loss_fn
        self._terminal_fn = terminal_fn
        self.normalizer = Normalizer()
        self.device = device

    def update_model_buffer(
        self,
        new_size: int,
        **kwargs: Any
    ) -> None:
        """Expand the size of the model buffer."""
        temp_buffer = self.model_buffer
        self.reset_model_buffer(new_size, **kwargs)
        self.model_buffer.update(temp_buffer)

    def _form_model_train_io(
        self,
        batch: Batch,
        fit_normalizer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Form the input and output tensor for model training."""
        obs = batch.obs
        act = batch.act
        obs_next = batch.obs_next
        delta_obs = obs_next - obs
        rew = batch.rew
        inputs = np.concatenate((obs, act), axis=-1)
        if fit_normalizer:
            self.normalizer.fit(inputs)
        inputs = self.normalizer.transform(inputs)
        inputs = torch.as_tensor(
            inputs,
            device=self.device,  # type: ignore
            dtype=torch.float32
        )
        target = np.concatenate((delta_obs, rew[..., None]), axis=-1)
        target = torch.as_tensor(
            target,
            device=self.device,  # type: ignore
            dtype=torch.float32
        )
        return inputs, target

    def _learn_model(
        self,
        buffer: ReplayBuffer,
        val_ratio: float = 0.2,
        max_val_num: int = 5000,
        model_learn_batch_size: int = 256,
        max_epoch: Optional[int] = None,
        max_static_epoch: int = 5,
        **kwargs: Any
    ) -> Dict[str, Union[float, int]]:
        """Learn the transition model.
        
        """
        total_num = len(buffer)
        val_num = min(int(total_num * val_ratio), max_val_num)
        train_num = total_num - val_num
        permutation = np.random.permutation(total_num)
        train_batch = buffer[permutation[:train_num]]
        train_inputs, train_target = self._form_model_train_io(
            train_batch,
            fit_normalizer=True,
        )
        train_size = (self.ensemble_size, train_num)
        train_indices = np.random.randint(
            train_num,
            size=train_size,
        )
        val_batch = buffer[permutation[train_num:]]
        val_inputs, val_target = self._form_model_train_io(val_batch)

        epoch_iter: Iterable
        if max_epoch is None:
            epoch_iter = itertools.count()
        else:
            epoch_iter = range(max_epoch)

        epochs_since_update = 0
        epochs_this_train = 0
        train_iter = 0
        train_loss = 0.
        best = np.full(self.ensemble_size, 1e10, dtype=float)
        for _ in epoch_iter:
            self.model.train()
            # Shuffle along the batch axis
            idx = np.random.random(train_size).argsort(axis=-1)
            train_indices = np.take_along_axis(train_indices, idx, axis=-1)

            num_iter = int(np.ceil(train_num / model_learn_batch_size))
            for iteration in range(num_iter):
                start = iteration * model_learn_batch_size
                end = (iteration + 1) * model_learn_batch_size
                indice = torch.as_tensor(train_indices[:, start:end])
                inputs = train_inputs[indice]
                target = train_target[indice]
                mean, logvar, max_logvar, min_logvar = self.model(inputs)
                loss = self._loss_fn(mean, logvar, max_logvar, min_logvar, target)
                self.model_optim.zero_grad()
                loss.backward()
                self.model_optim.step()
                train_loss = (train_loss * train_iter + loss.item()) / (train_iter + 1)
                train_iter += 1

            self.model.eval()
            num_iter = int(np.ceil(val_num / model_learn_batch_size))
            val_mse = np.zeros(self.ensemble_size)
            with torch.no_grad():
                mean, _, _, _ = self.model(val_inputs)
            val_mse = torch.mean(
                torch.square(mean - val_target),
                dim=(1, 2)
            ).cpu().numpy()

            epochs_this_train += 1
            improvement = (best - val_mse) / best
            update_flags = improvement > 0.01
            updated = np.any(update_flags)
            best[update_flags] = val_mse[update_flags]

            if updated:
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update > max_static_epoch:
                break

        self.elite_indices = np.argsort(val_mse)[:self.num_elites]
        elite_mse = val_mse[self.elite_indices].mean().item()

        max_logvar = max_logvar.detach().cpu().mean().item()
        min_logvar = min_logvar.detach().cpu().mean().item()

        # Collect training info to be logged.
        train_metrics = {
            "loss/model_train": train_loss,
            "loss/model_val": elite_mse,
            "model_train_epoch": epochs_this_train,
            "train_num": train_num,
            "val_num": val_num,
            "max_logvar": max_logvar,
            "min_logvar": min_logvar,
        }

        return train_metrics

    def _rollout_reset(
        self,
        buffer: ReplayBuffer,
        num_envs: Optional[int] = None,
        **kwargs: Any
    ) -> np.ndarray:
        if num_envs is None:
            num_envs = self._virtual_env_num
        batch, _ = buffer.sample(num_envs)
        obs = batch.obs.copy()
        return obs

    def _choose_from_ensemble(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        _, batch_size, _ = mean.shape
        var = torch.exp(logvar)

        # Choose an output from the elite subnetworks and compute its statistics.
        choice_indice = np.random.choice(self.elite_indices, size=batch_size)
        batch_indice = np.arange(batch_size)
        sample_mean = mean[choice_indice, batch_indice, :]
        sample_var = var[choice_indice, batch_indice, :]
        sample_std = torch.sqrt(sample_var)
        sample_dist = Independent(Normal(sample_mean, sample_std), 1)
        # Compute the statistics of rest outputs
        rest_mean = (torch.sum(mean, dim=0) - sample_mean) / (self.ensemble_size - 1)
        rest_var = (torch.sum(var + mean ** 2, dim=0) - sample_mean ** 2 -
                    sample_var) / (self.ensemble_size - 1) - rest_mean ** 2
        rest_std = torch.sqrt(rest_var)
        rest_dist = Independent(Normal(rest_mean, rest_std), 1)
        # OvR uncertainty
        uncertainty = torch.distributions.kl.kl_divergence(sample_dist, rest_dist)
        # Sample
        if self._deterministic_model_eval:
            sample = sample_mean
        else:
            sample = sample_dist.rsample()
        log_prob = sample_dist.log_prob(sample)
        sample = to_numpy(sample)
        delta_obs = sample[:, :-1]
        rew = sample[:, -1]

        return delta_obs, rew, log_prob, uncertainty

    def _rollout_step(
        self, obs: np.ndarray, info: Dict[str, Any], **kwargs: Any
    ) -> Tuple[Batch, Dict[str, Any]]:
        batch = Batch(obs=obs, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
        act = to_numpy(self(batch).act)
        inputs = np.concatenate((obs, act), axis=-1)
        inputs = self.normalizer.transform(inputs)
        mean, logvar, _, _ = self.model(inputs)
        delta_obs, rew, _, _ = self._choose_from_ensemble(mean, logvar)
        obs_next = obs + delta_obs
        done = self._terminal_fn(obs, act, obs_next)

        batch.update(
            act=act,
            obs_next=obs_next,
            rew=rew,
            done=done,
        )
        num_samples = info.get("num_samples", 0)
        info.update(
            num_samples=num_samples + len(obs),
            obs_next=obs_next.copy(),
            done=done.copy(),
        )

        return batch, info
