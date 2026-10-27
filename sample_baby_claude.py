"""Sample from a trained Baby Claude checkpoint.

Buffett persona prompts (the default): a shareholder-letter
opening, a value-investing question, and a plain-English
business question — the three voices Buffett uses most in
the letters. For the original Shakespeare checkpoint, pass
--out-dir out-baby-claude --samples-file samples.txt
--prompt "ROMEO:".
"""
import argparse
import os

import torch

from model import GPT, GPTConfig

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(BASE, "out-baby-claude-buffett")

PROMPTS = [
    "To the Shareholders of Berkshire Hathaway Inc.:\n\n",
    "We look for businesses that we can understand, with favorable long-term prospects, operated by honest and competent people, and available at a very attractive price. ",
    "Charlie and I believe that accounting is the language of business, and ",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--samples-file", type=str, default="samples-buffett.txt")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--prompt", type=str, default=None, help="single prompt; default: one Buffett prompt per sample")
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    itos = {int(k): v for k, v in ckpt["itos"].items()}
    stoi = {v: k for k, v in itos.items()}
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [args.prompt] * args.num_samples if args.prompt else PROMPTS[: args.num_samples]
    outputs = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(100 + i)
        start_ids = [stoi.get(c, 0) for c in prompt]
        x = torch.tensor([start_ids], dtype=torch.long)
        with torch.no_grad():
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        text = "".join(itos[int(t)] for t in y[0].tolist())
        outputs.append((prompt, text))
        print(f"\n--- sample {i+1} (prompt: {prompt!r}) ---\n{text}")

    with open(os.path.join(BASE, args.samples_file), "w", encoding="utf-8") as f:
        for i, (prompt, text) in enumerate(outputs, 1):
            f.write(f"\n--- sample {i} (prompt: {prompt!r}) ---\n{text}\n")
    print(f"\nwrote {args.samples_file} (dataset: {ckpt.get('dataset')}, final_loss: {ckpt.get('final_loss')})")


if __name__ == "__main__":
    main()
