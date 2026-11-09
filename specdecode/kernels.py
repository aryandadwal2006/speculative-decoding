import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _accepted_prefix_kernel(
        target_ptr,
        draft_ptr,
        token_ptr,
        uniform_ptr,
        output_ptr,
        SEQ: tl.constexpr,
        VOCAB: tl.constexpr,
    ):
        row = tl.program_id(0)
        accepted = 0
        failed = 0

        for step in range(SEQ):
            if failed == 0:
                token = tl.load(token_ptr + row * SEQ + step).to(tl.int32)
                offset = row * SEQ * VOCAB + step * VOCAB + token
                target_prob = tl.load(target_ptr + offset)
                draft_prob = tl.load(draft_ptr + offset)
                uniform = tl.load(uniform_ptr + row * SEQ + step)
                ratio = tl.minimum(1.0, target_prob / tl.maximum(draft_prob, 1.0e-12))
                if uniform <= ratio:
                    accepted += 1
                else:
                    failed = 1

        tl.store(output_ptr + row, accepted)


def accepted_prefix(
    target_probs: Tensor,
    draft_probs: Tensor,
    candidate_ids: Tensor,
    uniforms: Tensor,
) -> Tensor:
    if target_probs.ndim != 3 or target_probs.shape != draft_probs.shape:
        raise ValueError("target_probs and draft_probs must share shape [batch, sequence, vocab]")
    expected = target_probs.shape[:2]
    if candidate_ids.shape != expected or uniforms.shape != expected:
        raise ValueError("candidate_ids and uniforms must have shape [batch, sequence]")
    if target_probs.device != draft_probs.device:
        raise ValueError("probability tensors must be on the same device")

    target = target_probs.contiguous()
    draft = draft_probs.contiguous()
    tokens = candidate_ids.to(device=target.device, dtype=torch.int32).contiguous()
    draws = uniforms.to(device=target.device, dtype=torch.float32).contiguous()
    batch, sequence, vocab = target.shape

    if triton is not None and target.is_cuda:
        output = torch.empty((batch,), device=target.device, dtype=torch.int32)
        _accepted_prefix_kernel[(batch,)](
            target,
            draft,
            tokens,
            draws,
            output,
            SEQ=sequence,
            VOCAB=vocab,
            num_warps=4,
        )
        return output.to(torch.long)

    selected_target = target.gather(-1, tokens.to(torch.long).unsqueeze(-1)).squeeze(-1)
    selected_draft = draft.gather(-1, tokens.to(torch.long).unsqueeze(-1)).squeeze(-1)
    ratios = (selected_target / selected_draft.clamp_min(1.0e-12)).clamp(max=1.0)
    accepted = draws <= ratios
    return torch.cumprod(accepted.to(torch.int64), dim=-1).sum(dim=-1)
