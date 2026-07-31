"""runs/loss.csv から学習曲線の画像を作る (記事用).

    pip install matplotlib
    python tools/plot_loss.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def keep_last_run(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    """step が単調増加になるように、末尾から遡って行を残す.

    学習を二重起動してしまうと1つのCSVに2つの run が混ざり、step が前後する。
    最後に生き残った run の軌跡だけを取り出すための後始末。
    """
    kept: list[tuple[int, float]] = []
    last = None
    for step, value in reversed(points):
        if last is None or step < last:
            kept.append((step, value))
            last = step
    kept.reverse()
    return [s for s, _ in kept], [v for _, v in kept]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/loss.csv")
    ap.add_argument("--out", default="docs/images/loss-curve.png")
    args = ap.parse_args()

    train_points: list[tuple[int, float]] = []
    val_points: list[tuple[int, float]] = []
    with open(args.csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row["train_loss"]:
                train_points.append((step, float(row["train_loss"])))
            if row["val_loss"]:
                val_points.append((step, float(row["val_loss"])))

    train_x, train_y = keep_last_run(train_points)
    val_x, val_y = keep_last_run(val_points)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor("#0b1024")
    ax.set_facecolor("#0b1024")
    ax.plot(train_x, train_y, color="#6ea8ff", lw=1.6, label="train loss")
    ax.plot(val_x, val_y, color="#ff7ba8", lw=1.8, marker="o", ms=3, label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy (nats / char)")
    ax.set_title("1LM training curve", loc="left", fontsize=11)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}  (train {len(train_x)}点 / val {len(val_x)}点)")


if __name__ == "__main__":
    main()
