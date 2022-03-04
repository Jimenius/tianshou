from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import BasePolicy


class DynaPolicy(BasePolicy):
    """Abstraction of policies in the Dyna family.

    Dyna is a model-based algorithmic framework which builds on a model-free method
        with augmented data from rollouts in the learned model.

    :param BasePolicy policy: a model-free base policy.
    :param nn.Module model: the transition model.
    :param torch.optim.Optimizer: the optimizer for the model.
    :param Type[ReplayBuffer] model_buffer_type: type of buffer to store
        model rollouts.
    :param float real_ratio: ratio of data from real environment interactions when
        learning the policy.
    :param int virtual_env_num: number of parallel rollouts.
    :param model_args: arguments for model learning.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy`
    """

    def __init__(
        self,
        policy: BasePolicy,
        model: torch.nn.Module,
        model_optim: torch.optim.Optimizer,
        model_buffer_type: Type[ReplayBuffer],
        real_ratio: float = 0.1,
        virtual_env_num: int = 1,
        model_args: Dict[str, Any] = {},
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.policy = policy
        self.model = model
        self.model_optim = model_optim
        self._model_buffer_type = model_buffer_type
        self._real_ratio = real_ratio
        self._virtual_env_num = virtual_env_num
        self._model_args = model_args
        self._rollout_length = 0
        self._learn_model_flag = False
        self.results: Dict[str, Any] = {}

    def reset_model_buffer(self, max_size: int, *args: Any, **kwargs: Any) -> None:
        """Initialize or reset model buffer.

        :param max_size: buffer capacity.
        """
        self.model_buffer = self._model_buffer_type(max_size, *args, **kwargs)

    @abstractmethod
    def _learn_model(self, buffer: ReplayBuffer, **kwargs: Any) -> Dict[str, Any]:
        """Learn the transition model."""
        pass

    @abstractmethod
    def _rollout_reset(self, buffer: ReplayBuffer, **kwargs: Any) -> np.ndarray:
        """Get initial states."""
        pass

    @abstractmethod
    def _rollout_step(
        self, obs: np.ndarray, act: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Take a step in the learned model.

        :param np.ndarray obs: observation.
        :param np.ndarray act: action.

        :return: A tuple of 4 numpy arrays.
        """
        pass

    def _rollout(self, buffer: ReplayBuffer, **kwargs: Any) -> None:
        """Collect data to model buffer by rolling out the learned model.

        :param buffer: replay buffer for the real environment.
        """
        data = Batch(obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
        obs = self._rollout_reset(buffer)
        data.update(obs=obs)
        for _ in range(self._rollout_length):
            with torch.no_grad():
                act = to_numpy(self(data).act)
                obs_next, rew, done, info = self._rollout_step(
                    obs,
                    act  # type: ignore
                )
            data.update(
                act=act,
                obs_next=obs_next,
                rew=rew,
                done=done,
                info=info,
            )
            self.model_buffer.add(data)

            if np.any(done):
                done_indices = np.where(done)[0]
                batch, _ = buffer.sample(len(done_indices))
                obs_reset = batch.obs.copy()
                obs_next[done_indices] = obs_reset
            obs = obs_next
            data.update(obs=obs)

    def train(self, mode: bool = True) -> "DynaPolicy":
        """Set the base policy in training mode."""
        self.policy.train(mode)
        return self

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return self.policy.exploration_noise(act, batch)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data by inner policy."""
        return self.policy.forward(batch, state, **kwargs)

    def set_rollout_length(self, length: int) -> None:
        """Rollout length setter."""
        self._rollout_length = length

    def set_learn_model_flag(self, flag: bool = False) -> None:
        """Model learning flag setter."""
        self._learn_model_flag = flag

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        return self.policy.process_fn(batch, buffer, indices)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """Update policy with a given batch of data by inner policy."""
        return self.policy.learn(batch, **kwargs)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer."""
        if buffer is None:
            return {}
        if self._learn_model_flag:
            result = self._learn_model(buffer, **self._model_args)
            self.results.update(result)
            self._rollout(buffer)
            self.set_learn_model_flag(False)
        env_sample_size = int(sample_size * self._real_ratio)
        model_sample_size = sample_size - env_sample_size
        env_batch, env_indice = buffer.sample(env_sample_size)
        model_batch, model_indice = self.model_buffer.sample(model_sample_size)

        self.updating = True
        env_batch = self.process_fn(env_batch, buffer, env_indice)
        model_batch = self.process_fn(model_batch, self.model_buffer, model_indice)
        batch = Batch.cat([env_batch, model_batch])
        result = self.learn(batch, **kwargs)
        self.results.update(result)
        env_batch = batch[:env_sample_size]
        model_batch = batch[env_sample_size:]
        self.post_process_fn(env_batch, buffer, env_indice)
        self.post_process_fn(model_batch, self.model_buffer, model_indice)
        self.updating = False

        return self.results.copy()
