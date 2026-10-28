"""Test Baby Claude: fresh loss check + generation tests.

Not a training script. Loads a saved checkpoint, measures
loss on fresh random batches (train and val), then
generates from several prompts at two temperatures.
Also loads the Shakespeare checkpoint for comparison
on a neutral prompt, to show the dataset actually
changed what the model learned.

Usage:
    .venv/bin/python test_baby_claude.py
Writes: test-report.md (same folder)
"""
import argparse
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

BASE = os.path.dirname(os.path.abspath(__file__))


def load_checkpoint(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    itos = {int(k): v for k, v in ckpt["itos"].items()}
    stoi = {v: k for k, v in itos.items()}
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt, itos, stoi, model


def get_batch(data, batch_size, block_size, rng):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def fresh_loss(model, data_dir, iters=20, batch_size=32, block_size=128):
    rng = torch.Generator().manual_seed(999)
    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    out = {}
    for name, data in [("train", train), ("val", val)]:
        losses = []
        for _ in range(iters):
            x, y = get_batch(data, batch_size, block_size, rng)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    return out


@torch.no_grad()
def generate(model, stoi, itos, prompt, max_new_tokens, temperature, top_k, seed):
    missing = sorted(set(prompt) - set(stoi))
    if missing:
        return None, missing
    torch.manual_seed(seed)
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return "".join(itos[int(t)] for t in y[0].tolist()), []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=os.path.join(BASE, "out-baby-claude-buffett"))
    p.add_argument("--data-dir", type=str, default=os.path.join(BASE, "data", "buffett"),
                   help="corpus the checkpoint was trained on; must match its vocab")
    p.add_argument("--report-file", type=str, default="test-report.md")
    args = p.parse_args()
    lines = ["# Baby Claude — test report", "", "Run: 2026-10-27, CPU only, in this VM.", ""]
    t0 = time.time()

    # --- Buffett checkpoint ---
    ckpt, itos, stoi, model = load_checkpoint(args.out_dir)
    n_params = sum(p.numel() for p in model.parameters())
    lines.append(f"Buffett checkpoint: dataset={ckpt.get('dataset')}, params={n_params:,}, "
                 f"vocab={len(itos)}, checkpoint final_loss={ckpt.get('final_loss'):.4f}")
    losses = fresh_loss(model, args.data_dir)
    lines.append(f"Fresh loss check (20 new batches each, seed 999): train {losses['train']:.4f}, val {losses['val']:.4f}")
    lines.append("The fresh check should land close to the checkpoint final_loss above, "
                 "which confirms the saved checkpoint is the trained model, not a random one.")
    lines.append("")

    prompts = [
        "To the Shareholders of Berkshire Hathaway Inc.:\n\n",
        "A wonderful business at a fair price is ",
        "Price is what you pay. Value is what you get. ",
        "Berkshire",
    ]
    lines.append("## Buffett generation tests (150 new tokens each)")
    for i, prompt in enumerate(prompts):
        for temp in (0.7, 1.0):
            text, missing = generate(model, stoi, itos, prompt, 150, temp, 10, seed=200 + i * 10 + int(temp * 10))
            if text is None:
                lines.append(f"- prompt {prompt!r}: MISSING CHARS {missing}")
                print(f"missing chars for {prompt!r}: {missing}")
                continue
            continuation = text[len(prompt):]
            lines.append(f"### Prompt {i+1}, temperature {temp}: {prompt!r}")
            lines.append("")
            lines.append(continuation)
            lines.append("")
            print(f"--- prompt {i+1} temp {temp} ---\n{continuation}\n")

    # --- Shakespeare checkpoint, comparison ---
    ckpt2, itos2, stoi2, model2 = load_checkpoint(os.path.join(BASE, "out-baby-claude"))
    text, _ = generate(model2, stoi2, itos2, "ROMEO:", 150, 0.8, 10, seed=300)
    lines.append("## Shakespeare checkpoint, for comparison")
    lines.append(f"Checkpoint final_loss={ckpt2.get('final_loss'):.4f}. Prompt 'ROMEO:', temperature 0.8:")
    lines.append("")
    lines.append(text or "(generation failed)")
    lines.append("")
    print(f"--- Shakespeare ROMEO ---\n{text}\n")

    # --- Edge case: unknown character (an emoji is not in the Buffett vocab) ---
    text, missing = generate(model, stoi, itos, "Berkshire \U0001f600", 10, 0.8, 10, seed=400)
    lines.append("## Edge case: character outside the vocab")
    lines.append(f"Prompt 'Berkshire 😀' (emoji): missing chars = {missing}. "
                 "The sampler reports them instead of silently mapping them to a wrong character.")
    lines.append("")
    lines.append(f"Total test time: {time.time() - t0:.0f}s")
    with open(os.path.join(BASE, args.report_file), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.report_file}")


if __name__ == "__main__":
    main()
