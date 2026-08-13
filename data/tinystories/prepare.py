"""
Prepare the TinyStories dataset for PRISM/nanoGPT training.

Downloads the `roneneldan/TinyStories` dataset from the Hugging Face Hub,
tokenizes it with the GPT-2 BPE tokenizer (via tiktoken), and writes the
token ids to `train.bin` and `val.bin` as flat uint16 memmaps, matching the
data loader in `train.py` (`np.memmap(..., dtype=np.uint16)`).

Usage (from repo root):
    python data/tinystories/prepare.py                 # full dataset
    python data/tinystories/prepare.py --limit 2000    # quick subset (dev/CI)

Environment variables (fallbacks for the argument flags):
    PRISM_PREPARE_LIMIT   -> --limit
    PRISM_PREPARE_NUMPROC -> --num_proc
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

import tiktoken
from datasets import load_dataset


HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "roneneldan/TinyStories"


def _int_from_env(name):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TinyStories (GPT-2 BPE) for PRISM training.")
    parser.add_argument(
        "--limit",
        type=int,
        default=_int_from_env("PRISM_PREPARE_LIMIT"),
        help="If set, only use the first N training documents (and a small val slice). "
        "Great for quick local/CI runs. Default: use the entire dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=_int_from_env("PRISM_PREPARE_NUMPROC") or max(1, (os.cpu_count() or 2) // 2),
        help="Number of worker processes for tokenization.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.02,
        help="Fraction of documents used for validation when the dataset has no validation split.",
    )
    return parser.parse_args()


def load_splits(limit):
    """Return (train_ds, val_ds) with a 'text' column."""
    if limit is not None and limit > 0:
        # Slice with the datasets split API so we never materialize the full dataset.
        train_ds = load_dataset(DATASET_NAME, split=f"train[:{limit}]")
        val_count = max(1, int(limit * 0.05))
        try:
            val_ds = load_dataset(DATASET_NAME, split=f"validation[:{val_count}]")
        except (ValueError, KeyError):
            val_ds = load_dataset(DATASET_NAME, split=f"train[{limit}:{limit + val_count}]")
        return train_ds, val_ds

    train_ds = load_dataset(DATASET_NAME, split="train")
    try:
        val_ds = load_dataset(DATASET_NAME, split="validation")
    except (ValueError, KeyError):
        val_ds = None
    return train_ds, val_ds


def main():
    args = parse_args()
    enc = tiktoken.get_encoding("gpt2")

    train_ds, val_ds = load_splits(args.limit)

    if val_ds is None:
        split = train_ds.train_test_split(test_size=args.val_ratio, seed=1337, shuffle=True)
        train_ds, val_ds = split["train"], split["test"]

    splits = {"train": train_ds, "val": val_ds}
    print(f"Preparing TinyStories: train={len(train_ds)} docs, val={len(val_ds)} docs")

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)  # end-of-text delimiter between stories
        return {"ids": ids, "len": len(ids)}

    for split_name, ds in splits.items():
        tokenized = ds.map(
            process,
            remove_columns=ds.column_names,
            desc=f"tokenizing {split_name}",
            num_proc=args.num_proc,
        )

        arr_len = int(np.sum(tokenized["len"], dtype=np.uint64))
        filename = os.path.join(HERE, f"{split_name}.bin")
        # GPT-2 BPE vocab (50257) fits in uint16.
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        total_batches = min(1024, max(1, len(tokenized)))
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"]) if len(batch) else np.array([], dtype=np.uint16)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"wrote {filename}: {arr_len:,} tokens")


if __name__ == "__main__":
    main()
