"""ミニGPT本体 (MLX実装).

GPT-2と同じ「デコーダのみのTransformer」を、教材として読める最小構成で書いたもの。
やっていることは1つだけ: これまでの文字列から「次の1文字」の確率分布を出す。
会話が成立するのは、この予測を1文字ずつ繰り返しているだけである。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256  # 一度に見られる文脈の長さ (文字数)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GPTConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class CausalSelfAttention(nn.Module):
    """因果マスク付きの自己注意.

    「未来の文字を見てはいけない」という制約 (causal mask) が言語モデルの心臓部。
    ここを外すとカンニングになり、学習損失は下がるのに生成は破綻する。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        # (B, T, C) -> (B, n_head, T, head_dim)
        shape = (B, T, self.n_head, self.head_dim)
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.drop(self.proj(nn.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN + 残差接続. この形が深くしても学習が壊れにくい."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # 出力層は埋め込み行列を転用する (weight tying)。
        # 語彙2000×次元384ぶんのパラメータを節約でき、小さいモデルでは効きが良い。

    def __call__(self, idx: mx.array) -> mx.array:
        _, T = idx.shape
        pos = mx.arange(T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.tok_emb.as_linear(self.ln_f(x))

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), reduction="mean"
        )

    @property
    def n_params(self) -> int:
        from mlx.utils import tree_flatten

        return sum(p.size for _, p in tree_flatten(self.parameters()))
