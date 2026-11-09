from collections import deque
from dataclasses import dataclass

from torch import Tensor


@dataclass
class DecodeRequest:
    request_id: str
    input_ids: Tensor
    max_new_tokens: int


class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.pending: deque[DecodeRequest] = deque()
        self.active: dict[str, DecodeRequest] = {}
        self.known_ids: set[str] = set()

    def submit(self, request: DecodeRequest) -> None:
        if not request.request_id:
            raise ValueError("request_id must not be empty")
        if request.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if request.request_id in self.known_ids:
            raise ValueError("request_id is already queued or active")
        self.known_ids.add(request.request_id)
        self.pending.append(request)

    def next_batch(self) -> list[DecodeRequest]:
        while self.pending and len(self.active) < self.max_batch_size:
            request = self.pending.popleft()
            self.active[request.request_id] = request
        return list(self.active.values())

    def complete(self, request_id: str) -> list[DecodeRequest]:
        if request_id not in self.active:
            raise KeyError(request_id)
        del self.active[request_id]
        self.known_ids.remove(request_id)
        return self.next_batch()

    def cancel(self, request_id: str) -> None:
        if request_id in self.active:
            del self.active[request_id]
            self.known_ids.remove(request_id)
            return

        for request in self.pending:
            if request.request_id == request_id:
                self.pending.remove(request)
                self.known_ids.remove(request_id)
                return

        raise KeyError(request_id)

    def __len__(self) -> int:
        return len(self.pending) + len(self.active)
