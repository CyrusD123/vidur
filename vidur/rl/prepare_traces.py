"""
Build request-length traces from chat datasets, as used by Fu et al.,
"Efficient LLM Scheduling by Learning to Rank" (arXiv 2408.15792):

    python vidur/rl/prepare_traces.py sharegpt ShareGPT_V3_unfiltered_cleaned_split.json
    python vidur/rl/prepare_traces.py lmsys path/to/lmsys-chat-1m/data/
    python vidur/rl/prepare_traces.py azure AzureLLMInferenceTrace_conv.csv

Run it as a file: it does not import vidur, so it works from a separate
environment with `tokenizers`, `pandas` and `pyarrow` (lmsys only), none of
which requirements-rl.txt needs. ShareGPT is ShareGPT_V3_unfiltered_cleaned_split.json
from the anon8231489123/ShareGPT_Vicuna_unfiltered HF dataset; LMSYS-Chat-1M
(lmsys/lmsys-chat-1m) is gated, so accept its terms and log in first.

Each conversation contributes its first user turn (prompt) and the reply to it
(output), counted with the Llama-3 tokenizer. Like the paper, prompt lengths
include the Llama-3 chat template (--template-tokens, 10 for one user turn
plus the assistant header). Pairs with an empty side, or whose total exceeds --max-tokens
(the simulator's 4096-token limit), are dropped rather than clipped, so the
length distribution is not distorted. The output is a CSV with
num_prefill_tokens, num_decode_tokens, the format TraceRequestLengthGenerator
reads.

The paper instead regenerates every output with Llama-3 at temperature 1.0;
we use the dataset's recorded replies (as vLLM's serving benchmark does), so
output lengths follow the original chatbots, not Llama-3.

The Azure 2023 conversation trace already ships as splitwise_conv.csv (the
same rows, with timestamps converted to seconds from the first request); the
`azure` command only checks that.
"""

import argparse
import glob
import json
import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# <|begin_of_text|> + user header, <|eot_id|>, assistant header + "\n\n"
LLAMA3_TEMPLATE_TOKENS = 10

# ungated copy of the Meta-Llama-3-8B tokenizer (the simulated model)
DEFAULT_TOKENIZER = "NousResearch/Meta-Llama-3-8B"


def load_tokenizer(name: str):
    from tokenizers import Tokenizer

    if os.path.exists(name):
        return Tokenizer.from_file(name)
    return Tokenizer.from_pretrained(name)


def count_tokens(tokenizer, texts: List[str], batch_size: int = 4096) -> np.ndarray:
    counts = np.empty(len(texts), dtype=np.int64)
    for start in range(0, len(texts), batch_size):
        batch = tokenizer.encode_batch(
            texts[start : start + batch_size], add_special_tokens=False
        )
        counts[start : start + len(batch)] = [len(e.ids) for e in batch]
    return counts


def sharegpt_pairs(path: str) -> Iterable[Tuple[str, str]]:
    with open(path) as f:
        data = json.load(f)
    for item in data:
        turns = item.get("conversations") or []
        if len(turns) >= 2 and turns[0]["from"] == "human" and turns[1]["from"] == "gpt":
            yield turns[0]["value"], turns[1]["value"]


def lmsys_pairs(path: str) -> Iterable[Tuple[str, str]]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(path, "*.parquet"))) if os.path.isdir(path) else [path]
    assert files, f"no parquet files under {path}"
    for file in files:
        table = pq.read_table(file, columns=["conversation"])
        for turns in table.column("conversation").to_pylist():
            if len(turns) >= 2 and turns[0]["role"] == "user" and turns[1]["role"] == "assistant":
                yield turns[0]["content"], turns[1]["content"]


def build_trace(
    pairs: Iterable[Tuple[str, str]],
    tokenizer,
    max_tokens: int,
    max_rows: int,
    seed: int,
    template_tokens: int = LLAMA3_TEMPLATE_TOKENS,
) -> pd.DataFrame:
    prompts, outputs = map(list, zip(*pairs))
    df = pd.DataFrame(
        {
            "num_prefill_tokens": count_tokens(tokenizer, prompts),
            "num_decode_tokens": count_tokens(tokenizer, outputs),
        }
    )
    num_pairs = len(df)
    keep = (df["num_prefill_tokens"] > 0) & (df["num_decode_tokens"] > 0)
    df["num_prefill_tokens"] += template_tokens
    keep &= df["num_prefill_tokens"] + df["num_decode_tokens"] <= max_tokens
    df = df[keep]
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    print(
        f"{num_pairs} first-turn pairs, kept {keep.sum()} "
        f"({keep.mean():.1%}) with total <= {max_tokens}, wrote {len(df)}"
    )
    print(df.describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())
    return df


def check_azure(raw_path: str, processed_path: str) -> None:
    raw = pd.read_csv(raw_path, parse_dates=["TIMESTAMP"])
    processed = pd.read_csv(processed_path)
    seconds = (raw["TIMESTAMP"] - raw["TIMESTAMP"].iloc[0]).dt.total_seconds()
    assert len(raw) == len(processed), (len(raw), len(processed))
    assert (raw["ContextTokens"].values == processed["num_prefill_tokens"].values).all()
    assert (raw["GeneratedTokens"].values == processed["num_decode_tokens"].values).all()
    assert np.allclose(seconds.values, processed["arrived_at"].values, atol=1e-3)
    print(
        f"{processed_path} matches {raw_path}: {len(raw)} requests over "
        f"{seconds.iloc[-1]:.0f} s ({len(raw) / seconds.iloc[-1]:.2f} qps)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["sharegpt", "lmsys", "azure"])
    parser.add_argument("path", help="dataset JSON / parquet dir / raw Azure CSV")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-rows", type=int, default=100_000, help="subsample (0 = keep all)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--template-tokens", type=int, default=LLAMA3_TEMPLATE_TOKENS)
    args = parser.parse_args()

    if args.dataset == "azure":
        check_azure(args.path, args.output or "data/processed_traces/splitwise_conv.csv")
        return

    pairs = sharegpt_pairs(args.path) if args.dataset == "sharegpt" else lmsys_pairs(args.path)
    df = build_trace(
        pairs, load_tokenizer(args.tokenizer), args.max_tokens,
        args.max_rows,
        args.seed,
        args.template_tokens,
    )
    output = args.output or f"data/processed_traces/{args.dataset}_llama3_tokenizer.csv"
    df.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
