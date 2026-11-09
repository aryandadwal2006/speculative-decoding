import torch
from torch import Tensor


class DynamicKVCache:
    def __init__(self):
        self.keys: list[Tensor | None] = []
        self.values: list[Tensor | None] = []

    @property
    def sequence_length(self) -> int:
        if not self.keys or self.keys[0] is None:
            return 0
        return self.keys[0].shape[-2]

    def update(self, key: Tensor, value: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        if key.ndim < 3 or value.ndim < 3:
            raise ValueError("key and value tensors must have a sequence dimension at -2")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("key and value sequence lengths must match")

        while len(self.keys) <= layer_idx:
            self.keys.append(None)
            self.values.append(None)

        cached_key = self.keys[layer_idx]
        cached_value = self.values[layer_idx]

        if cached_key is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            if cached_value is None:
                raise RuntimeError("cache key/value state is inconsistent")
            self.keys[layer_idx] = torch.cat((cached_key, key), dim=-2)
            self.values[layer_idx] = torch.cat((cached_value, value), dim=-2)

        return self.keys[layer_idx], self.values[layer_idx]

    def crop(self, sequence_length: int) -> None:
        if sequence_length < 0 or sequence_length > self.sequence_length:
            raise ValueError("sequence_length is outside the cached range")

        for index, key in enumerate(self.keys):
            if key is not None:
                self.keys[index] = key[..., :sequence_length, :]
                value = self.values[index]
                if value is None:
                    raise RuntimeError("cache key/value state is inconsistent")
                self.values[index] = value[..., :sequence_length, :]

    def reorder(self, batch_indices: Tensor) -> None:
        for index, key in enumerate(self.keys):
            if key is not None:
                indices = batch_indices.to(key.device, dtype=torch.long)
                self.keys[index] = key.index_select(0, indices)
                value = self.values[index]
                if value is None:
                    raise RuntimeError("cache key/value state is inconsistent")
                self.values[index] = value.index_select(0, indices)

    def clear(self) -> None:
        self.keys.clear()
        self.values.clear()
