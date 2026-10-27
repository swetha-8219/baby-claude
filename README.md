# Baby Claude

A tiny GPT built the Karpathy way, named by Swetha on
2026-10-27. It is a *learning* model, not a chat
assistant and not Warren Buffett: a character-level
Transformer trained, in this version, on Berkshire
Hathaway shareholder letters. It learns letter-like
character patterns; after a short run it does not
produce coherent investing arguments, and nothing it
generates is Buffett's views or investment advice.

An earlier version was trained on tiny Shakespeare
with the same model shape. See `notes.md` for both
runs, with the honest sample assessment.

Model: GPT (nanoGPT, MIT, credited in `model.py`),
4 layers, 4 heads, 128-dim embeddings, 128-token
context, 817,280 parameters on the Buffett data
(vocab 104 characters). Trained on CPU in a VM with
no GPU. A full GPT-2 run is out of scope here; see
`../weekend-build-nanogpt.md` in the local workspace
for the GPU / cloud version.

## Data — read this before running

`prepare_buffett.py` downloads, at run time only,
Berkshire Hathaway shareholder letters published for
free on berkshirehathaway.com: PDF letters 2005–2024
(1,490,128 characters; HTML letters 1977–2004 were
attempted and skipped when they returned a bot-check
page). The letters remain Berkshire Hathaway's. No
letter text is included in this repository — only the
download-and-prepare recipe, the code, the lab notes,
and short generated samples. Run the script to build
the corpus locally in `data/buffett/`; it is ignored
by git.

## How to run (CPU only)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # CPU PyTorch; see requirements.txt

# 1. data (downloads the letters, builds train/val bins)
.venv/bin/python prepare_buffett.py

# 2. overfit checkpoint — one batch, loss must fall
.venv/bin/python train_baby_claude.py --overfit --max-iters 100

# 3. short training run — 300 steps, CPU
.venv/bin/python train_baby_claude.py --max-iters 300

# 4. generate, with Buffett-style prompts
.venv/bin/python sample_baby_claude.py --num-samples 3 --max-new-tokens 200
```

Checkpoints (`out-baby-claude-buffett/ckpt.pt`) and
the corpus are ignored by git and are not published.

To reproduce the original Shakespeare run instead,
prepare nanoGPT's `shakespeare_char` data and run
with `--data-dir <that dir> --dataset shakespeare
--out-dir out-baby-claude`, then sample with
`--out-dir out-baby-claude --samples-file samples.txt
--prompt "ROMEO:"`.

## Results

See `notes.md` — filled in from the actual runs on
2026-10-27, with train/val loss and the sample
assessment. Generated Buffett samples are in
`samples-buffett.txt`; Shakespeare samples are in
`samples.txt`.
