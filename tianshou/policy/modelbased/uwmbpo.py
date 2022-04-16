from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import MBPOPolicy


class UWMBPOPolicy(MBPOPolicy):
    """
    """

    def _rollout_step(
        self, obs: np.ndarray, info: Dict[str, Any], **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        batch = Batch(obs=obs, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
        act = to_numpy(self(batch).act)
        inputs = np.concatenate((obs, act), axis=-1)
        inputs = self.normalizer.transform(inputs)
        mean, logvar, _, _ = self.model(inputs)
        delta_obs, rew, log_prob, uncertainty = self._choose_from_ensemble(mean, logvar)
        obs_next = obs + delta_obs
        done = self._terminal_fn(obs, act, obs_next)
        self.min_uncertainty = min(self.min_uncertainty, torch.min(uncertainty).item())
        print(
            'min: {:.4f}, max: {:.4f}, mean: {:.4f}'.format(
                torch.min(uncertainty).item(),
                torch.max(uncertainty).item(),
                torch.mean(uncertainty).item(),
            )
        )
        step_info = [
            {
                "log_prob": x.item(),
                "uncertainty": y.item()
            } for x, y in zip(torch.split(log_prob, 1), torch.split(uncertainty, 1))
        ]
        step_info = np.array(step_info)

        batch.update(
            act=act,
            obs_next=obs_next,
            rew=rew,
            done=done,
            info=step_info,
        )
        num_samples = info.get("num_samples", 0)
        info.update(
            num_samples=num_samples + len(obs),
            obs_next=obs_next.copy(),
            done=done.copy(),
        )

        return batch, info

    def _postprocess_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        metrics = super()._postprocess_info(info)
        metrics.update(
            ref_uncertainty=self.min_uncertainty,
        )
        return metrics

    def _rollout(
        self,
        buffer: ReplayBuffer,
        auto_reset: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        # # Calculate reference uncertainty.
        # buffer_size = len(buffer)
        # sample_size = min(buffer_size, self._virtual_env_num)
        # indices = np.random.permutation(buffer_size)[:sample_size]
        # batch, _ = buffer.sample(batch_size=0)
        # batch = batch[indices]
        # inputs = np.concatenate((batch.obs, batch.act), axis=-1)
        # inputs = self.normalizer.transform(inputs)
        # mean, logvar, _, _ = self.model(inputs)
        # _, _, _, uncertainty = self._choose_from_ensemble(mean, logvar)
        # self.ref_uncertainty = torch.min(uncertainty).item()
        self.min_uncertainty = float("inf")
        return super()._rollout(buffer, auto_reset, **kwargs)

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.policy.process_fn(batch, buffer, indices)
        info = batch.info
        if hasattr(info, "uncertainty"):
            weight = self.min_uncertainty / torch.tensor(
                info["uncertainty"],
                dtype=torch.float32,
                device=self.device,
            )
            batch.weight = weight.clamp(max=1.)
        else:
            batch.weight = torch.ones(
                len(batch),
                dtype=torch.float32,
                device=self.device,
            )
        return batch

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        batch_size = len(batch)
        batch.weight = batch.weight * batch_size / batch.weight.sum()
        return self.policy.learn(batch, **kwargs)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer."""
        if buffer is None:
            return {}
        if self._learn_model_flag:
            result = self._learn_model(buffer, **self._model_args)
            self.results.update(result)
            with torch.no_grad():
                result = self._rollout(buffer, **self._model_args)
            self.results.update(result)
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
        env_td = env_batch.weight.detach().abs()
        env_td_mean = torch.mean(env_td).item()
        env_td_std = torch.std(env_td)
        model_batch = batch[env_sample_size:]
        model_td = model_batch.weight.detach().abs()
        model_td_mean = torch.mean(model_td).item()
        model_td_std = torch.std(model_td).item()
        self.results.update(
            env_td_mean=env_td_mean,
            env_td_std=env_td_std,
            model_td_mean=model_td_mean,
            model_td_std=model_td_std,
        )
        self.updating = False

        return self.results.copy()
