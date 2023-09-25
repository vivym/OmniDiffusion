from typing import Any

import numpy as np
import ray.data


class MultiSourceIterator:
    def __init__(self, sources: list[tuple[ray.data.DataIterator, float]]):
        self.source_iters = [s[0] for s in sources]

        weights = np.array([s[1] for s in sources])
        self.probs: np.ndarray =  weights / weights.sum()

        self.indices = np.arange(len(sources))

        self.remaining_iters = len(sources)

    def __next__(self) -> Any:
        data = None
        while self.remaining_iters > 0:
            try:
                idx = np.random.choice(self.indices, p=self.probs)
                data = next(self.source_iters[idx])
                break
            except StopIteration:
                self.remaining_iters -= 1
                self.probs[idx] = 0.0
                if self.remaining_iters > 0:
                    self.probs = self.probs / self.probs.sum()

        if data is None:
            raise StopIteration
        else:
            return data


class MultiSourceDataLoader:
    def __init__(
        self,
        batch_size: int = 8,
        local_shuffle_buffer_size: int | None = None,
        local_shuffle_seed: int | None = None,
        prefetch_batches: int = 1,
    ):
        self.batch_size = batch_size
        self.local_shuffle_buffer_size = local_shuffle_buffer_size
        self.local_shuffle_seed = local_shuffle_seed
        self.prefetch_batches = prefetch_batches

        self.datasets: list[ray.data.Dataset] = []
        self.weights: list[float] = []

    def add_dataset(self, dataset: ray.data.Dataset, weight: float = 1.0):
        self.datasets.append(dataset)
        self.weights.append(weight)

    def __iter__(self) -> MultiSourceIterator:
        return MultiSourceIterator([
            (
                iter(ds.iter_torch_batches(
                    batch_size=self.batch_size,
                    drop_last=True,
                    local_shuffle_buffer_size=self.local_shuffle_buffer_size,
                    local_shuffle_seed=(
                        self.local_shuffle_seed if self.local_shuffle_seed is None else self.local_shuffle_seed + i
                    ),
                    prefetch_batches=self.prefetch_batches,
                )),
                w,
            )
            for i, (ds, w) in enumerate(zip(self.datasets, self.weights))
        ])
