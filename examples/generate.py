import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdecode import SpeculativeDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=dtype,
    ).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft,
        torch_dtype=dtype,
    ).to(device).eval()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    decoder = SpeculativeDecoder(
        target,
        draft,
        width=args.width,
        depth=args.depth,
        max_nodes=args.max_nodes,
    )
    output = decoder.generate(input_ids, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
