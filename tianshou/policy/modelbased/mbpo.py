import itertools
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer, to_numpy
from tianshou.policy import BasePolicy, DynaPolicy


class MBPOPolicy(DynaPolicy):
    """Implementation of Model-Based Policy Optimization. arXiv:1906.08253.

    MBPO builds on Dyna framework with branch rollout.

    :param BasePolicy policy: a model-free base policy.
    :param nn.Module model: the transition model.
    :param torch.optim.Optimizer: the optimizer for the model.
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
    :param Optional[Union[str, int, torch.device]] device: device for training.
    :param Dict[str, Any] model_args: arguments for model learning.
    :param Dict[str, Any] model_buffer_args: Extra arguments for the
        model replay buffer.

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
        self._best = np.full(ensemble_size, 1e10, dtype=float)
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
        batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Form the input and output tensor for model training."""
        obs = batch.obs
        act = batch.act
        obs_next = batch.obs_next
        delta_obs = obs_next - obs
        rew = batch.rew
        inputs = np.concatenate((obs, act), axis=-1)
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
        holdout_ratio: float = 0.8,
        model_learn_batch_size: int = 256,
        max_epoch: Optional[int] = None,
        max_static_epoch: int = 5,
        **kwargs: Any
    ) -> Dict[str, Union[float, int]]:
        """Learn the transition model."""
        total_num = len(buffer)
        train_num = int(total_num * holdout_ratio)
        val_num = total_num - train_num
        permutation = np.random.permutation(total_num)
        train_batch = buffer[permutation[:train_num]]
        train_size = (self.ensemble_size, train_num)
        val_batch = buffer[permutation[train_num:]]
        train_indices = np.random.randint(
            train_num,
            size=train_size,
        )

        epoch_iter: Iterable
        if max_epoch is None:
            epoch_iter = itertools.count()
        else:
            epoch_iter = range(max_epoch)

        epochs_since_update = 0
        epochs_this_train = 0
        train_iter = 0
        train_loss = 0.
        for _ in epoch_iter:
            self.model.train()
            # Shuffle along the batch axis
            idx = np.random.random(train_size).argsort(axis=-1)
            train_indices = np.take_along_axis(train_indices, idx, axis=-1)

            num_iter = int(np.ceil(train_num / model_learn_batch_size))
            for iteration in range(num_iter):
                start = iteration * model_learn_batch_size
                end = (iteration + 1) * model_learn_batch_size
                batch = train_batch[train_indices[:, start:end]]
                inputs, target = self._form_model_train_io(batch)
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
            for iteration in range(num_iter):
                start = iteration * model_learn_batch_size
                end = (iteration + 1) * model_learn_batch_size
                batch = val_batch[start:end]
                inputs, target = self._form_model_train_io(batch)
                with torch.no_grad():
                    mean, _, _, _ = self.model(inputs)
                batch_mse = torch.mean(torch.square(mean - target),
                                       dim=(1, 2)).cpu().numpy()
                val_mse = (val_mse * iteration + batch_mse) / (iteration + 1)

            epochs_this_train += 1
            improvement = (self._best - val_mse) / self._best
            update_flags = improvement > 0.01
            updated = np.any(update_flags)
            self._best[update_flags] = val_mse[update_flags]

            if updated:
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            if epochs_since_update > max_static_epoch:
                break

        self.elite_indices = np.argsort(val_mse)[:self.num_elites]
        elite_mse = val_mse[self.elite_indices].mean().item()

        # Collect training info to be logged.
        train_info = {
            "loss/model_train": train_loss,
            "loss/model_val": elite_mse,
            "model_train_epoch": epochs_this_train,
            "rollout_length": self._rollout_length,
        }

        return train_info

    def _rollout_reset(self, buffer: ReplayBuffer, **kwargs: Any) -> np.ndarray:
        batch, _ = buffer.sample(self._virtual_env_num)
        obs = batch.obs.copy()
        return obs

    def _rollout_step(
        self, obs: np.ndarray, act: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        inputs = np.concatenate((obs, act), axis=-1)
        with torch.no_grad():
            mean, logvar, _, _ = self.model(inputs)
            std = torch.sqrt(torch.exp(logvar))
            dist = Independent(Normal(mean, std), 1)
            if self._deterministic_model_eval:
                sample = mean
            else:
                sample = dist.rsample()
            log_prob = dist.log_prob(sample)
            # For each input, choose a network from the ensemble
            _, batch_size, _ = sample.shape
            sample = to_numpy(sample)
            indices = np.random.randint(self.num_elites, size=batch_size)
            choice_indices = self.elite_indices[indices]
            batch_indices = np.arange(batch_size)
            obs_next = obs + sample[choice_indices, batch_indices, :-1]
            rew = sample[choice_indices, batch_indices, -1]
            done = self._terminal_fn(obs, act, obs_next)
            log_prob = log_prob[choice_indices, batch_indices]
            info = np.array(
                list(map(lambda x: {"log_prob": x.item()}, torch.split(log_prob, 1)))
            )

        return obs_next, rew, done, info
