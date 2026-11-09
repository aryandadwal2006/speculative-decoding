import torch
from torch import Tensor

from .tree import build_candidate_tree, model_device, verify_tree_greedy


class SpeculativeDecoder:
    def __init__(
        self,
        target_model: torch.nn.Module,
        draft_model: torch.nn.Module,
        width: int = 4,
        depth: int = 3,
        max_nodes: int = 64,
    ):
        if width < 1 or depth < 1 or max_nodes < 1:
            raise ValueError("width, depth, and max_nodes must be positive")

        self.target_model = target_model.eval()
        self.draft_model = draft_model.eval()
        self.width = width
        self.depth = depth
        self.max_nodes = max_nodes

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generate currently supports one sequence at a time")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")

        output = input_ids.to(model_device(self.target_model)).clone()

        while output.shape[1] - input_ids.shape[1] < max_new_tokens:
            draft_prompt = output.to(model_device(self.draft_model))
            nodes = build_candidate_tree(
                self.draft_model,
                draft_prompt,
                width=self.width,
                depth=self.depth,
                max_nodes=self.max_nodes,
            )
            tokens = verify_tree_greedy(self.target_model, output, nodes)
            remaining = max_new_tokens - (output.shape[1] - input_ids.shape[1])
            extension = torch.tensor(
                tokens[:remaining],
                dtype=output.dtype,
                device=output.device,
            ).unsqueeze(0)
            if extension.numel() == 0:
                break
            output = torch.cat((output, extension), dim=1)

        return output
