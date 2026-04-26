from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable, Iterator, Sequence

from torch.utils.data import Sampler

from frcnet.data.manifest import SampleManifestRecord


def _distribute_count(total_count: int, num_buckets: int) -> tuple[int, ...]:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive.")
    base_count = total_count // num_buckets
    remainder = total_count % num_buckets
    return tuple(base_count + (1 if bucket_index < remainder else 0) for bucket_index in range(num_buckets))


class _CyclingIndexPool:
    def __init__(self, indices: Sequence[int], *, rng: random.Random, shuffle: bool) -> None:
        if not indices:
            raise ValueError("Source-balanced pools must not be empty.")
        self.indices = list(indices)
        self.rng = rng
        self.shuffle = shuffle
        self.cursor = 0
        self.order: list[int] = []
        self._reset()

    def _reset(self) -> None:
        self.order = list(self.indices)
        if self.shuffle:
            self.rng.shuffle(self.order)
        self.cursor = 0

    def take(self, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            if self.cursor >= len(self.order):
                self._reset()
            remaining = count - len(selected)
            available = len(self.order) - self.cursor
            take_count = min(remaining, available)
            selected.extend(self.order[self.cursor : self.cursor + take_count])
            self.cursor += take_count
        return selected


class SourceBalancedBatchSampler(Sampler[list[int]]):
    """Build batches with fixed ID/ambiguous/OOD fractions and balanced OOD sources."""

    def __init__(
        self,
        manifest_records: Sequence[SampleManifestRecord],
        *,
        batch_size: int,
        batches_per_epoch: int | None = None,
        seed: int = 7,
        shuffle: bool = True,
        id_fraction: float = 0.25,
        ambiguous_fraction: float = 0.25,
        ood_fraction: float = 0.50,
        id_cohorts: Iterable[str] = ("easy_id", "hard_id"),
        ambiguous_cohorts: Iterable[str] = ("ambiguous_id",),
        ood_cohorts: Iterable[str] = ("unknown_supervision", "ood"),
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        fraction_sum = float(id_fraction) + float(ambiguous_fraction) + float(ood_fraction)
        if not math.isclose(fraction_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("source-balanced batch fractions must sum to 1.0.")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.id_count = int(round(self.batch_size * float(id_fraction)))
        self.ambiguous_count = int(round(self.batch_size * float(ambiguous_fraction)))
        self.ood_count = self.batch_size - self.id_count - self.ambiguous_count
        if min(self.id_count, self.ambiguous_count, self.ood_count) <= 0:
            raise ValueError("source-balanced batches require positive ID, ambiguous, and OOD counts.")

        self.id_cohorts = frozenset(str(value) for value in id_cohorts)
        self.ambiguous_cohorts = frozenset(str(value) for value in ambiguous_cohorts)
        self.ood_cohorts = frozenset(str(value) for value in ood_cohorts)
        self.records = list(manifest_records)
        if not self.records:
            raise ValueError("manifest_records must not be empty.")

        self.batches_per_epoch = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else max(1, len(self.records) // self.batch_size)
        )
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive.")
        self._epoch_index = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _build_pools(
        self,
        *,
        rng: random.Random,
    ) -> tuple[_CyclingIndexPool, _CyclingIndexPool, dict[str, _CyclingIndexPool]]:
        id_indices: list[int] = []
        ambiguous_indices: list[int] = []
        ood_indices_by_source: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            if record.cohort_name in self.id_cohorts:
                id_indices.append(index)
            elif record.cohort_name in self.ambiguous_cohorts:
                ambiguous_indices.append(index)
            elif record.cohort_name in self.ood_cohorts:
                ood_indices_by_source[record.source_dataset_name].append(index)

        if not id_indices:
            raise ValueError("source-balanced sampling requires at least one ID sample.")
        if not ambiguous_indices:
            raise ValueError("source-balanced sampling requires at least one ambiguous sample.")
        if not ood_indices_by_source:
            raise ValueError("source-balanced sampling requires at least one OOD/unknown source.")

        return (
            _CyclingIndexPool(id_indices, rng=rng, shuffle=self.shuffle),
            _CyclingIndexPool(ambiguous_indices, rng=rng, shuffle=self.shuffle),
            {
                source_name: _CyclingIndexPool(indices, rng=rng, shuffle=self.shuffle)
                for source_name, indices in sorted(ood_indices_by_source.items())
            },
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch_index)
        self._epoch_index += 1
        id_pool, ambiguous_pool, ood_pools = self._build_pools(rng=rng)
        ood_source_names = tuple(sorted(ood_pools))
        ood_counts = _distribute_count(self.ood_count, len(ood_source_names))
        for _ in range(self.batches_per_epoch):
            batch_indices = []
            batch_indices.extend(id_pool.take(self.id_count))
            batch_indices.extend(ambiguous_pool.take(self.ambiguous_count))
            for source_name, source_count in zip(ood_source_names, ood_counts, strict=True):
                batch_indices.extend(ood_pools[source_name].take(source_count))
            if self.shuffle:
                rng.shuffle(batch_indices)
            yield batch_indices
