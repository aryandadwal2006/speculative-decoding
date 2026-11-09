from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TreeNode:
    token_id: int
    parent: int
    depth: int


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def build_candidate_tree(
    draft_model: torch.nn.Module,
    prompt_ids: Tensor,
    width: int = 4,
    depth: int = 3,
    max_nodes: int = 64,
) -> list[TreeNode]:
    if width < 1 or depth < 1 or max_nodes < 1:
        raise ValueError("width, depth, and max_nodes must be positive")

    prompt = prompt_ids.reshape(-1).to(model_device(draft_model))
    if prompt.numel() == 0:
        raise ValueError("prompt_ids must contain at least one token")

    nodes: list[TreeNode] = []
    frontier = [-1]

    for level in range(1, depth + 1):
        next_frontier = []
        for parent in frontier:
            if len(nodes) >= max_nodes:
                break

            path = []
            cursor = parent
            while cursor >= 0:
                path.append(nodes[cursor].token_id)
                cursor = nodes[cursor].parent
            path.reverse()

            if path:
                suffix = torch.tensor(path, device=prompt.device, dtype=prompt.dtype)
                context = torch.cat((prompt, suffix))
            else:
                context = prompt

            logits = draft_model(input_ids=context.unsqueeze(0)).logits[0, -1]
            count = min(width, max_nodes - len(nodes), logits.numel())
            token_ids = torch.topk(logits, k=count).indices.tolist()

            for token_id in token_ids:
                nodes.append(TreeNode(int(token_id), parent, level))
                next_frontier.append(len(nodes) - 1)

        frontier = next_frontier
        if not frontier or len(nodes) >= max_nodes:
            break

    return nodes


def build_tree_attention_mask(
    prompt_length: int,
    nodes: list[TreeNode],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")

    total_length = prompt_length + len(nodes)
    mask = torch.full(
        (1, 1, total_length, total_length),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=device,
    )
    positions = torch.empty((1, total_length), dtype=torch.long, device=device)
    positions[0, :prompt_length] = torch.arange(prompt_length, device=device)

    for query in range(prompt_length):
        mask[0, 0, query, : query + 1] = 0

    for index, node in enumerate(nodes):
        query = prompt_length + index
        mask[0, 0, query, :prompt_length] = 0
        positions[0, query] = prompt_length + node.depth - 1

        ancestor = index
        while ancestor >= 0:
            mask[0, 0, query, prompt_length + ancestor] = 0
            ancestor = nodes[ancestor].parent

    return mask, positions


@torch.inference_mode()
def verify_tree_greedy(
    target_model: torch.nn.Module,
    prompt_ids: Tensor,
    nodes: list[TreeNode],
) -> list[int]:
    prompt = prompt_ids.reshape(-1).to(model_device(target_model))
    if prompt.numel() == 0:
        raise ValueError("prompt_ids must contain at least one token")

    if not nodes:
        logits = target_model(input_ids=prompt.unsqueeze(0)).logits[0, -1]
        return [int(logits.argmax().item())]

    tree_tokens = torch.tensor(
        [node.token_id for node in nodes],
        dtype=prompt.dtype,
        device=prompt.device,
    )
    input_ids = torch.cat((prompt, tree_tokens)).unsqueeze(0)
    attention_mask, position_ids = build_tree_attention_mask(
        prompt.numel(), nodes, prompt.device
    )
    logits = target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ).logits

    children: dict[int, list[int]] = {-1: []}
    for index, node in enumerate(nodes):
        children.setdefault(node.parent, []).append(index)
        children.setdefault(index, [])

    accepted = []
    parent = -1

    while True:
        query_position = prompt.numel() - 1 if parent == -1 else prompt.numel() + parent
        token_id = int(logits[0, query_position].argmax().item())
        accepted.append(token_id)

        match = next(
            (child for child in children[parent] if nodes[child].token_id == token_id),
            None,
        )
        if match is None:
            break
        parent = match

    return accepted
