"""文字レベルトークナイザ.

日本語をサブワードに分割するには通常 SentencePiece の学習が必要になるが、
1文字=1トークンと割り切れば辞書は「コーパスに出てきた文字の集合」だけで済む。
そのかわり語彙は数千種類に膨らみ、1トークンあたりの情報量は小さくなる。

会話の役割を表すマーカー (<|user|> など) は、文字に分解せず
1トークンとして扱う。8文字に分解してしまうとモデルが「開始記号」を
学ぶために無駄な容量を使い、生成時に途中まで壊れた記号を出す原因になる。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
UNK = "<|unk|>"

# UNK は語彙に必ず入れるが、コーパス側には現れない (推論時の未知文字用)。
SPECIAL_TOKENS = (UNK, USER, ASSISTANT, END)
_MARKER_RE = re.compile("(" + "|".join(re.escape(t) for t in (USER, ASSISTANT, END)) + ")")


class CharTokenizer:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]
        self.user_id = self.stoi[USER]
        self.assistant_id = self.stoi[ASSISTANT]
        self.end_id = self.stoi[END]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def train(cls, text: str, min_freq: int = 1) -> CharTokenizer:
        """コーパスから語彙を作る. マーカーは除いた上で文字を数える."""
        counts = Counter()
        for chunk in _MARKER_RE.split(text):
            if chunk in SPECIAL_TOKENS:
                continue
            counts.update(chunk)
        chars = sorted(c for c, n in counts.items() if n >= min_freq)
        return cls(list(SPECIAL_TOKENS) + chars)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _MARKER_RE.split(text):
            if chunk in SPECIAL_TOKENS:
                ids.append(self.stoi[chunk])
            else:
                ids.extend(self.stoi.get(c, self.unk_id) for c in chunk)
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"])
