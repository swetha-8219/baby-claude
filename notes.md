# Baby Claude — lab notes

Run date: 2026-10-27, in this VM (CPU only, no GPU).
Model: character-level GPT, 4 layers / 4 heads / 128-dim
embeddings / 128-token context, 128-token context.
Two datasets were trained with the same model shape;
only the data and the sampling prompts changed.

## Shakespeare run (first version)

Vocab 65 characters, tiny Shakespeare
(1,003,854 train tokens, 111,540 val tokens),
812,288 parameters.

## Setup

- CPU PyTorch 2.14.1+cpu in `.venv/`, 2 CPU threads.
- Data prepared with nanoGPT's `data/shakespeare_char/prepare.py`
  (public tiny-Shakespeare text only).
- Upstream code credited in `README.md`; Baby Claude's own
  training wrapper is `train_baby_claude.py`.

## Checkpoint 1 — overfit one batch (PASS)

Command: `train_baby_claude.py --overfit --max-iters 100`
- Initial loss: 4.1882
- After 100 steps on the same fixed batch: 1.1169
- Time: 59 seconds.

Loss fell clearly, so the model + optimizer can learn.
It did not reach ~0 in 100 steps — more steps on the
same batch would be needed to fully memorize it. The
direction, not the exact value, is the checkpoint here.

## Checkpoint 2 — short training run (300 steps)

- Command: `train_baby_claude.py --max-iters 300 --eval-interval 50 --eval-iters 20`
- Initial loss: 4.1882. Final batch loss: 2.3075 after 300 steps, in 216 seconds total.
- Eval at step 50: train 2.5926 / val 2.6044. At step 300: train 2.3123 / val 2.3385.
- Loss fell steadily and train/val stayed close — no overfitting yet at 300 steps. More steps would keep improving it, slowly, on CPU.
- Checkpoint saved to `out-baby-claude/ckpt.pt` (812,288 parameters).

## Samples

See `samples.txt` (3 samples, prompt "ROMEO:", 200 new
tokens, temperature 0.8, top-k 10).

## TPM lens

- First risk when scaling 10x: compute (CPU here took
  ~0.6s/step for a 0.8M-parameter model; a 124M model
  needs a GPU and ~1 hour / ~$10 on a cloud box, per
  Karpathy's build-nanogpt README).
- Second: evaluation — train loss alone flatters the
  model; val loss and an external check (HellaSwag in
  the full build) are the honest numbers.
- Third: data — Shakespeare is 1.1M characters; GPT-2
  scale needs billions of tokens, with curation and
  deduplication work that dwarfs the model code.

---

## Warren Buffett run (second version, 2026-10-27)

Swetha asked to retrain Baby Claude on Warren Buffett
instead of Shakespeare, rewrite the code for that
persona, and publish it. The model shape is unchanged;
the dataset, the data-preparation script, and the
sampling prompts are new.

### Corpus

- Source: Berkshire Hathaway shareholder letters
  published for free on berkshirehathaway.com,
  downloaded at run time by `prepare_buffett.py`.
  No letter text is committed to GitHub.
- Years actually used: PDF letters 2005–2024 only
  (20 letters, 1,490,128 characters in total).
  The HTML letters for 1977–2004 were attempted too,
  but every one returned a 117-character bot-check
  page in this run and was skipped by the script.
  The PDFs include the annual performance table at
  the front of each letter, so the corpus is prose
  plus tables, numbers, and dotted leaders — noisier
  than Shakespeare.
- Split: 1,341,115 train tokens / 149,013 val tokens
  (90/10, chronological — val is the tail of the
  concatenated text, so it is mostly the most recent
  letters). Vocab: 104 characters.
  817,280 parameters (slightly more than Shakespeare,
  because the larger vocab grows the embedding).

### Overfit checkpoint (PASS)

Command: `train_baby_claude.py --overfit --max-iters 100`
(data/buffett, default out-dir out-baby-claude-buffett)
- Initial loss: 4.6222
- After 100 steps on the same fixed batch: 1.1712
- Time: 60 seconds. Loss fell clearly; as with
  Shakespeare, it did not reach ~0 in 100 steps.

### Short training run (300 steps)

- Command: `train_baby_claude.py --max-iters 300 --eval-interval 50 --eval-iters 20`
- Initial loss: 4.6222. Final batch loss: 2.4058 after
  300 steps, in 229 seconds total.
- Eval at step 50: train 2.5878 / val 2.4681.
  At step 300: train 2.3438 / val 2.2748.
  Val loss is lower than train loss here; that is
  probably an artefact of the chronological split
  (val = recent letters, which repeat the same
  phrasing and table format), not proof the model
  generalises better than it fits.
- Checkpoint saved to
  `out-baby-claude-buffett/ckpt.pt` (not committed).

### Samples — honest assessment

See `samples-buffett.txt` (3 samples, 200 new tokens,
temperature 0.8, top-k 10), with Buffett-style prompts:
a shareholder-letter opening, a value-investing
criterion sentence, and a "Charlie and I believe…"
accounting sentence.

After 300 steps the samples are mostly character-level
gibberish with word fragments ("Ondicalld ar inceed hin
on fongen sithalur…"), noticeably less word-like than
the Shakespeare samples at the same step count. The
model has learned letter-like surface statistics, not
a Buffett persona: no coherent investing argument,
no usable advice, and nothing here should be read as
Buffett's views or as investment advice. The likely
reasons are the noisier corpus (tables and numbers)
and too few steps for dense business prose. A longer
run, and cleaning the tables out of the corpus, are
the two concrete next steps if Swetha wants the
persona to actually emerge.

### Longer training run (2,000 steps, 2026-10-27)

Swetha approved a longer run after the 300-step test.
Same model and corpus, trained from scratch for
2,000 steps into a separate checkpoint so the
300-step one is preserved:

- Command: `train_baby_claude.py --max-iters 2000
  --eval-interval 200 --eval-iters 20
  --lr-decay-iters 2000
  --out-dir out-baby-claude-buffett-2000`
- 2,000 steps in 1,193s (~20 min) on CPU.
  Final batch loss 1.3173 (started 4.6222).
  Eval at step 2,000: train 1.3456 / val 1.3503
  (vs 2.3438 / 2.2748 at 300 steps). Train and val
  are now close, so the earlier val-below-train gap
  has closed; the model fits the letters much better.
- Fresh test (`test_baby_claude.py --out-dir
  out-baby-claude-buffett-2000`, report
  `test-report-2000.md`): new batches, train 1.3410
  / val 1.3358, confirming the checkpoint.
- Samples (`samples-buffett-2000.txt`, and the 8
  test generations in `test-report-2000.md`) are
  mostly real words now, with Buffett vocabulary:
  shareholders, operating, insurance, Berkshire,
  "Charlie and I". Grammar still breaks down and
  invented words remain ("bighamen", "corectaters",
  "ellyicative"), and there is still no coherent
  investing argument or persona. Temperature 0.7 is
  more word-like than 1.0, as before.
- Verdict: longer training worked — loss roughly
  halved and fragments became mostly-words — but
  2,000 steps is still not a Buffett voice. Next
  levers, not yet agreed: clean the performance
  tables out of the corpus, train longer, or both.
  Checkpoint saved to
  `out-baby-claude-buffett-2000/ckpt.pt`
  (not committed).

### Cleaned corpus, longer run (5,000 steps, 2026-10-27)

Swetha approved cleaning the tables and training
longer. `prepare_buffett.py` gained `clean_tables()`
plus `--clean-tables` / `--out-dir` (default
behaviour unchanged): front performance table of
each letter and mid-letter table rows removed,
prose with numbers kept. Verified: 1,490,128 →
1,307,882 chars (12.2% removed), 1,361 front-table
lines + 1,652 table rows, 0 dotted-leader lines
left, all 20 letters intact, vocab 104 → 103.
Cleaned corpus in `data/buffett-clean/` (train
1,177,093 / val 130,789 tokens).

- Command: `train_baby_claude.py --data-dir
  data/buffett-clean --dataset buffett-clean
  --max-iters 5000 --eval-interval 500
  --eval-iters 20 --lr-decay-iters 5000
  --out-dir out-baby-claude-buffett-clean-5000`
- 5,000 steps in 2,853s (~48 min) on CPU.
  Final batch loss 1.0989 (started 4.6594).
  Eval at step 5,000: train 1.0916 / val 1.3571
  (at 4,500: train 1.0823 / val 1.3474).
- Fresh test (`test_baby_claude.py --data-dir
  data/buffett-clean`, report
  `test-report-clean-5000.md`): new batches,
  train 1.0867 / val 1.3781.
- Honest comparison with the 2,000-step run
  (train 1.3456 / val 1.3503; fresh val 1.3358):
  train fit improved a lot, but held-out val loss
  did NOT improve — it is flat to slightly worse.
  The extra steps mostly memorised the training
  letters (chronological split: val is the most
  recent letters). Samples
  (`samples-buffett-clean-5000.txt`, and the 8
  test generations) are uneven: some are the most
  letter-like yet ("the operating earnings for the
  company that our shareholders…", "insurance
  operation of Berkshire. Charlie and I…"), others
  degenerate into blank lines or invented words
  ("watcomy", "importanced", "burdged"). Still no
  coherent investing argument or persona.
- Verdict: cleaning + longer training hit the
  ceiling of this 0.8M-parameter char-level model.
  More steps on the same setup are unlikely to
  help. The levers that would actually change
  the result are a bigger model, token-level
  (BPE) tokenisation instead of characters, or
  much more data — which is what the weekend
  GPT-2 build is for. Checkpoint saved to
  `out-baby-claude-buffett-clean-5000/ckpt.pt`
  (not committed).
