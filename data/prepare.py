"""会話コーパスを作る.

公開データセット kunishou/oasst1-89k-ja (Apache-2.0) から
「ユーザーの発言 → アシスタントの返答」のペアを取り出し、
1行1会話のテキストファイルに整形する。

    <|user|>おすすめの本はありますか？<|assistant|>SFがお好きなら...<|end|>

data/raw/ に自分のデータを置けば、そのまま混ぜたり、
--no-hf を付けて自分のデータだけで学習させることもできる。
対応形式は次の2つ。

    data/raw/mydata.jsonl : {"user": "...", "assistant": "..."} を1行1件
    data/raw/mydata.tsv   : ユーザー発言<TAB>アシスタント返答 を1行1件

使い方:
    python data/prepare.py                 # 既定の設定でコーパス生成
    python data/prepare.py --max-a 200     # もっと短い返答だけを使う
    python data/prepare.py --no-hf         # data/raw/ のデータだけ使う
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tokenizer import ASSISTANT, END, USER  # noqa: E402

HF_REPO = "kunishou/oasst1-89k-ja"
HF_FILE = "oasst1_89k_ja_20231027.json"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """空白と改行をすべて半角スペース1個に潰し、1会話=1行を保証する.

    箇条書きの改行は失われるが、そのぶん小さなモデルには学習しやすい形になる。
    """
    return _WS_RE.sub(" ", text).strip()


def load_hf_pairs() -> list[tuple[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    messages = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {m["message_id"]: m for m in messages}

    pairs: list[tuple[str, str]] = []
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        parent = by_id.get(msg["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue
        # ng_translation=="1" は翻訳が破綻していると報告されている行。
        if msg["ng_translation"] == "1" or parent["ng_translation"] == "1":
            continue
        q, a = normalize(parent["text_ja"] or ""), normalize(msg["text_ja"] or "")
        if q and a:
            pairs.append((q, a))
    return pairs


def load_local_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not RAW_DIR.exists():
        return pairs
    for path in sorted(RAW_DIR.iterdir()):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q, a = normalize(obj["user"]), normalize(obj["assistant"])
                if q and a:
                    pairs.append((q, a))
        elif path.suffix == ".tsv":
            for line in path.read_text(encoding="utf-8").splitlines():
                if "\t" not in line:
                    continue
                q, a = (normalize(x) for x in line.split("\t", 1))
                if q and a:
                    pairs.append((q, a))
    return pairs


def drop_rare_char_pairs(
    pairs: list[tuple[str, str]], min_char_freq: int
) -> tuple[list[tuple[str, str]], int]:
    """出現回数が少ない文字を含む会話を丸ごと捨てる.

    絵文字や一度しか出てこない漢字を残すと、語彙が増えるだけで学習には寄与しない。
    未知文字を <|unk|> に置き換える手もあるが、教材としては「捨てる」方が挙動が読みやすい。
    """
    if min_char_freq <= 1:
        return pairs, 0
    counts: Counter[str] = Counter()
    for q, a in pairs:
        counts.update(q)
        counts.update(a)
    rare = {c for c, n in counts.items() if n < min_char_freq}
    kept = [(q, a) for q, a in pairs if not (rare & set(q)) and not (rare & set(a))]
    return kept, len(rare)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus.txt"))
    ap.add_argument("--max-q", type=int, default=100, help="ユーザー発言の最大文字数")
    ap.add_argument("--max-a", type=int, default=300, help="アシスタント返答の最大文字数")
    ap.add_argument("--min-a", type=int, default=4, help="アシスタント返答の最小文字数")
    ap.add_argument("--min-char-freq", type=int, default=10)
    ap.add_argument("--no-hf", action="store_true", help="公開データセットを使わない")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = [] if args.no_hf else load_hf_pairs()
    print(f"公開データセット: {len(pairs)} 会話")
    local = load_local_pairs()
    if local:
        print(f"data/raw/       : {len(local)} 会話")
    pairs += local
    if not pairs:
        raise SystemExit("会話が0件です。--no-hf を外すか data/raw/ にデータを置いてください。")

    pairs = [
        (q, a)
        for q, a in pairs
        if len(q) <= args.max_q and args.min_a <= len(a) <= args.max_a
    ]
    print(f"長さフィルタ後   : {len(pairs)} 会話")

    seen: set[tuple[str, str]] = set()
    deduped = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    print(f"重複除去後       : {len(deduped)} 会話")

    kept, n_rare = drop_rare_char_pairs(deduped, args.min_char_freq)
    print(f"低頻度文字除去後 : {len(kept)} 会話 (捨てた文字種: {n_rare})")

    import random

    random.Random(args.seed).shuffle(kept)

    lines = [f"{USER}{q}{ASSISTANT}{a}{END}" for q, a in kept]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vocab: Counter[str] = Counter()
    for q, a in kept:
        vocab.update(q)
        vocab.update(a)
    n_chars = sum(len(q) + len(a) for q, a in kept)
    meta = {
        "conversations": len(kept),
        "chars": n_chars,
        "unique_chars": len(vocab),
        "avg_user_len": round(sum(len(q) for q, _ in kept) / len(kept), 1),
        "avg_assistant_len": round(sum(len(a) for _, a in kept) / len(kept), 1),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n書き出し: {out}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
