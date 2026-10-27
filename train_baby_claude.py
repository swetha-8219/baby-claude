"""Baby Claude — tiny character-level GPT, Karpathy-style.

Model: the GPT in local model.py (nanoGPT, MIT, credited there):
token + position embeddings, N transformer Blocks
(causal self-attention + MLP, residual connections,
LayerNorm), final norm, tied output head, cross-entropy
loss on the next token. Training: AdamW, warmup + cosine
decay, gradient clipping. All credit to Andrej Karpathy;
this file is a small, CPU-friendly training wrapper so
Swetha can run and inspect the whole loop in one place.

Datasets: --data-dir data/buffett (Warren Buffett /
Berkshire Hathaway shareholder letters, built locally by
prepare_buffett.py) or data/shakespeare_char (the original
tiny-Shakespeare run). Use --out-dir to keep the two
checkpoints apart.
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE, "data", "buffett")
DEFAULT_OUT_DIR = os.path.join(BASE, "out-baby-claude-buffett")


def load_data(data_dir):
    with open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    itos = {i: ch for i, ch in enumerate(chars)}
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return chars, itos, train_data, val_data


def get_batch(data, batch_size, block_size, device, rng, fixed_ix=None):
    if fixed_ix is not None:
        ix = fixed_ix
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=rng)
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, rng):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device, rng)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * it / max(1, args.warmup_iters)
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=300)
    p.add_argument("--overfit", action="store_true", help="train on one fixed batch only")
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--lr-decay-iters", type=int, default=300)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--dataset", type=str, default="buffett")
    args = p.parse_args()
    args.lr_decay_iters = max(args.lr_decay_iters, args.max_iters)

    device = "cpu"
    torch.manual_seed(1337)
    rng = torch.Generator().manual_seed(1337)
    chars, itos, train_data, val_data = load_data(args.data_dir)
    print(f"dataset={args.dataset} vocab={len(chars)} train_tokens={len(train_data)} val_tokens={len(val_data)} device={device}")

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=len(chars),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Baby Claude params={n_params:,} layers={args.n_layer} heads={args.n_head} embd={args.n_embd}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.01)

    fixed_ix = None
    if args.overfit:
        fixed_ix = torch.randint(len(train_data) - args.block_size, (args.batch_size,), generator=rng)
        print("OVERFIT MODE: one fixed batch only")

    x, y = get_batch(train_data, args.batch_size, args.block_size, device, rng, fixed_ix)
    _, loss0 = model(x, y)
    print(f"initial loss on first batch: {loss0.item():.4f}")
    t0 = time.time()
    for it in range(args.max_iters):
        lr = get_lr(it, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, args.block_size, device, rng, fixed_ix)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (it + 1) % 25 == 0 or it == 0:
            print(f"iter {it+1}/{args.max_iters} loss {loss.item():.4f} lr {lr:.2e} elapsed {time.time()-t0:.0f}s")
        if (it + 1) % args.eval_interval == 0 and not args.overfit:
            losses = estimate_loss(model, train_data, val_data, args, device, rng)
            print(f"  eval iter {it+1}: train {losses['train']:.4f} val {losses['val']:.4f}")
    print(f"final loss: {loss.item():.4f} (started {loss0.item():.4f}) in {time.time()-t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "model_args": dict(config.__dict__),
        "itos": itos,
        "chars": chars,
        "dataset": args.dataset,
        "final_loss": loss.item(),
        "initial_loss": loss0.item(),
        "max_iters": args.max_iters,
        "overfit": args.overfit,
    }
    torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
    print(f"saved checkpoint to {args.out_dir}/ckpt.pt")
    if args.overfit:
        print(f"OVERFIT RESULT: {loss0.item():.4f} -> {loss.item():.4f} "
              f"({'PASS: loss fell' if loss.item() < loss0.item() else 'FAIL: loss did not fall'})")


if __name__ == "__main__":
    main()
