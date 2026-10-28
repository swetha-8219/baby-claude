"""Prepare the Warren Buffett / Berkshire Hathaway training corpus.

Downloads, at run time only, the shareholder letters that
Berkshire Hathaway publishes for free on its own website:
  - HTML letters, 1977-2004 (kept only when the page returns
    real letter text; several years return a short bot-check
    page and are skipped)
  - PDF letters, 2005-2024, from berkshirehathaway.com/letters/

Nothing downloaded here is committed to GitHub. This script
is the reproducible recipe: run it, and the corpus is built
locally in data/buffett/. The letters remain Berkshire
Hathaway's; this project trains a tiny learning model on
them locally and does not redistribute the letter text.

Output, in data/buffett/ (same layout as nanoGPT's
data/shakespeare_char/):
  input.txt  — all letters, one after another, with a
               "--- Berkshire Hathaway letter, YEAR ---" header
  meta.pkl   — vocab (sorted unique characters), stoi/itos
  train.bin / val.bin — uint16 token ids, 90/10 split

Usage:
    .venv/bin/python prepare_buffett.py [--clean-tables] [--out-dir data/buffett-clean]
    Default (no flags) rebuilds data/buffett/ exactly as before.
    --clean-tables removes each letter's front performance table
    and mid-letter table rows; see clean_tables().
Requires: requests, pypdf, fonttools, numpy (see requirements.txt).
"""
import io
import os
import pickle
import re
import time
from html.parser import HTMLParser

import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "buffett")
HEADERS = {"User-Agent": "Mozilla/5.0 (BabyClaude-learning; personal educational use)"}
PDF_YEARS = list(range(2005, 2025))
HTML_YEARS = list(range(1977, 2005))


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False
        if tag in ("p", "br", "div", "tr", "li"):
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(html):
    p = _TextExtractor()
    p.feed(html)
    return " ".join("".join(p.out).split())


def clean(text):
    # Collapse repeated blank lines, normalise line endings, keep
    # the characters the letters actually use ($, %, quotes, dashes).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_TABLE_HEADER_PHRASES = (
    "Corporate Performance vs",
    "Annual Percentage Change",
    "Compounded Annual Gain",
    "Overall Gain",
    "Year (1) (2)",
)


def _is_table_row(line):
    """True for lines that are table rows, not prose.

    Conservative on purpose: a prose sentence that happens
    to contain numbers (a phone number, a date, a dollar
    figure) must survive. Only lines that are dotted
    leaders, almost entirely numbers, or mostly numeric
    tokens with a table-like shape are dropped.
    """
    s = line.strip()
    if not s:
        return False
    if "...." in s:  # dotted leader, e.g. "1965 .... 23.8 10.0 13.8"
        return True
    if s.isdigit():  # stray page number
        return True
    if len(s) < 80 and any(p in s for p in _TABLE_HEADER_PHRASES):
        return True
    nums = re.findall(r"-?\(?\d[\d,\.]*\)?%?", s)
    if not nums:
        return False
    tokens = s.split()
    # Almost entirely digits/punctuation: a table row or a
    # bare list of figures, not a sentence.
    non_space = [c for c in s if c != " "]
    if non_space and sum(1 for c in non_space if c.isdigit() or c in ".,()%$+-") / len(non_space) > 0.8:
        return True
    # Mostly numeric tokens, short line: e.g. a holdings row
    # "2,338,961,000 PetroChina ... 1.3 488 1,915".
    if len(nums) >= 4 and len(nums) / max(len(tokens), 1) >= 0.4:
        return True
    return False


def clean_tables(text):
    """Remove the performance tables from the letters.

    Two passes, both line-based so prose is never reworded:

    1. Front table: in each letter, drop everything between
       the letter header and "To the Shareholders", which is
       where the letter proper starts. The header and the
       "To the Shareholders" line itself are kept.
    2. Mid-letter table rows: drop individual lines that
       _is_table_row identifies as table rows.

    Returns (cleaned_text, stats) where stats counts what
    was removed, so a caller can sanity-check the result.
    """
    lines = text.split("\n")
    kept = []
    i = 0
    front_removed = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- Berkshire Hathaway letter"):
            kept.append(line)
            i += 1
            # Look ahead to the next header (or end) for the
            # "To the Shareholders" marker.
            j = i
            marker = None
            while j < len(lines) and not lines[j].startswith("--- Berkshire Hathaway letter"):
                if "To the Shareholders" in lines[j]:
                    marker = j
                    break
                j += 1
            if marker is not None:
                front_removed += marker - i
                i = marker
            continue
        kept.append(line)
        i += 1

    out = []
    rows_removed = 0
    for line in kept:
        if line.startswith("--- Berkshire Hathaway letter") or "To the Shareholders" in line:
            out.append(line)
        elif _is_table_row(line):
            rows_removed += 1
        else:
            out.append(line)
    cleaned = clean("\n".join(out))
    stats = {
        "front_table_lines_removed": front_removed,
        "table_rows_removed": rows_removed,
        "chars_before": len(text),
        "chars_after": len(cleaned),
    }
    return cleaned, stats


def fetch_pdf(session, year):
    from pypdf import PdfReader

    url = f"https://www.berkshirehathaway.com/letters/{year}ltr.pdf"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    reader = PdfReader(io.BytesIO(r.content))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def fetch_html(session, year):
    url = f"https://www.berkshirehathaway.com/letters/{year}.html"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return html_text(r.text)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default=OUT_DIR)
    p.add_argument("--clean-tables", action="store_true",
                   help="remove the front performance table of each letter "
                        "and mid-letter table rows (see clean_tables)")
    args = p.parse_args()
    out_dir = args.out_dir

    session = requests.Session()
    session.headers.update(HEADERS)
    parts = []
    used, skipped = [], []
    for year in HTML_YEARS:
        try:
            text = clean(fetch_html(session, year))
        except Exception as e:  # noqa: BLE001 - report and continue
            skipped.append((year, f"html: {e}"))
            continue
        if len(text) < 5000 or "Berkshire" not in text:
            skipped.append((year, f"html: only {len(text)} chars, skipped"))
            continue
        parts.append(f"\n\n--- Berkshire Hathaway letter, {year} ---\n\n{text}")
        used.append((year, len(text)))
        print(f"{year} html: {len(text)} chars")
        time.sleep(0.3)
    for year in PDF_YEARS:
        try:
            text = clean(fetch_pdf(session, year))
        except Exception as e:  # noqa: BLE001
            skipped.append((year, f"pdf: {e}"))
            continue
        if len(text) < 5000:
            skipped.append((year, f"pdf: only {len(text)} chars, skipped"))
            continue
        parts.append(f"\n\n--- Berkshire Hathaway letter, {year} ---\n\n{text}")
        used.append((year, len(text)))
        print(f"{year} pdf: {len(text)} chars")
        time.sleep(0.3)

    text = "\n".join(parts).strip() + "\n"
    print(f"years used: {[y for y, _ in used]}")
    print(f"years skipped: {skipped}")
    print(f"total chars: {len(text):,}")
    if args.clean_tables:
        text, stats = clean_tables(text)
        print(f"clean_tables: {stats}")
        print(f"total chars after cleaning: {len(text):,}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    print(f"vocab size: {len(chars)}")
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(chars), "itos": itos, "stoi": stoi}, f)

    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = len(data)
    train_ids = data[: int(n * 0.9)]
    val_ids = data[int(n * 0.9):]
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))
    print(f"train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")


if __name__ == "__main__":
    main()
