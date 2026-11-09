# Speculative Decoding

A minimal greedy tree-speculative decoding implementation for compatible PyTorch causal language models.

- The draft model expands top-k continuations into a bounded token tree.
- The target model verifies all tree nodes in one forward pass with an ancestor attention mask.
- Decoding follows target-model greedy choices and emits a fallback token at the first mismatch.
- `DynamicKVCache` supports per-layer key/value append, truncation, and batch reordering.
- `ContinuousBatchScheduler` manages pending requests and fills freed batch slots.
- An optional Triton kernel computes accepted prefixes for linear probabilistic speculation.
- Tree decoding currently recomputes contexts and does not yet integrate the KV cache or scheduler.
- Models must support 4D additive attention masks and explicit `position_ids`.

Install with `pip install -e .` or `pip install -e '.[triton]'`; run `python examples/generate.py --target MODEL --draft MODEL --prompt 'Hello'`.
