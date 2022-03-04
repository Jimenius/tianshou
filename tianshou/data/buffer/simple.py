from typing import Any, List, Optional, Tuple, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import _alloc_by_keys_diff, _create_value


class SimpleReplayBuffer(ReplayBuffer):
    """:class:`~tianshou.data.SimpleReplayBuffer` stores data generated from interaction \
    between the policy and environment.

    SimpleReplayBuffer adds a sequence of data by directly filling in samples. It \
    ignores sequence information in an episode.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(
        self,
        size: int,
        **kwargs: Any
    ) -> None:
        self._meta: Batch
        self._index: int
        self._size: int
        super().__init__(size)

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        return np.arange(self._size)[~self.done[:self._size]]

    def prev(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the input index."""
        return np.array(index)

    def next(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the input index."""
        return np.array(index)

    def update(self, buffer: "ReplayBuffer") -> np.ndarray:
        """Move the data from the given buffer to current buffer.

        Return the updated indices. If update fails, return an empty array.
        """
        if len(buffer) == 0 or self.maxsize == 0:
            return np.array([], int)
        self.add(buffer._meta)
        num_samples = len(buffer)
        to_indices = np.arange(self._index, self._index + num_samples) % self.maxsize
        return to_indices

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into SimpleReplayBuffer.

        :param Batch batch: the input data batch. Its keys must belong to the 7
            reserved keys, and "obs", "act", "rew", "done" is required.

        Return current_index and constants to keep compatability
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())

        num_samples = len(batch)
        ptr = self._index
        indices = np.arange(self._index, self._index + num_samples) % self.maxsize
        self._size = min(self._size + num_samples, self.maxsize)
        self._index = (self._index + num_samples) % self.maxsize
        try:
            self._meta[indices] = batch
        except ValueError:
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, False
                )
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._meta[indices] = batch
        return np.array([ptr]), np.array([0.]), np.array([0]), np.array([0])
