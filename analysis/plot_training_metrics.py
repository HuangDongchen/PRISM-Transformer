import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class Metrics:
    eval_steps: List[int]
    eval_train_loss: List[float]
    eval_val_loss: List[float]
    iter_steps: List[int]
    iter_grad_norm: List[float]


def _read_metrics_jsonl(path: str) -> Metrics:
    eval_steps: List[int] = []
    eval_train: List[float] = []
    eval_val: List[float] = []
    iter_steps: List[int] = []
    grad_norms: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r_type = r.get("type")
            if r_type == "eval":
                if "train_loss" in r and "val_loss" in r and "iter" in r:
                    eval_steps.append(int(r["iter"]))
                    eval_train.append(float(r["train_loss"]))
                    eval_val.append(float(r["val_loss"]))
            elif r_type == "iter":
                if "grad_norm" in r and r["grad_norm"] is not None and "iter" in r:
                    iter_steps.append(int(r["iter"]))
                    grad_norms.append(float(r["grad_norm"]))

    return Metrics(
        eval_steps=eval_steps,
        eval_train_loss=eval_train,
        eval_val_loss=eval_val,
        iter_steps=iter_steps,
        iter_grad_norm=grad_norms,
    )


def _downsample(xs: List[int], ys: List[float], max_points: int) -> Tuple[List[int], List[float]]:
    if max_points <= 0:
        return xs, ys
    if len(xs) <= max_points:
        return xs, ys
    stride = max(1, len(xs) // max_points)
    return xs[::stride], ys[::stride]


def _moving_average(ys: List[float], window: int) -> List[float]:
    if window <= 1:
        return ys
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for y in ys:
        q.append(float(y))
        s += float(y)
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _setup_mpl_style(*, latex_2col: bool):
    # Avoid relying on a LaTeX installation; choose sensible publication defaults.
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600 if latex_2col else 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 9 if latex_2col else 14,
            "axes.labelsize": 8 if latex_2col else 12,
            "xtick.labelsize": 7 if latex_2col else 10,
            "ytick.labelsize": 7 if latex_2col else 10,
            "legend.fontsize": 7 if latex_2col else 10,
            "lines.linewidth": 1.2 if latex_2col else 2.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            # Publication-friendly: show full box and use inward ticks.
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def plot_loss(
    metrics: Metrics,
    save_path: str,
    *,
    figsize: Tuple[float, float],
    dpi: int,
):
    if not metrics.eval_steps:
        raise RuntimeError("No eval records found (train_loss/val_loss).")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # High-contrast colors: R/G/B first
    ax.plot(metrics.eval_steps, metrics.eval_train_loss, label="train loss", color="red")
    ax.plot(metrics.eval_steps, metrics.eval_val_loss, label="val loss", color="blue")

    # Reference baselines (horizontal dashed lines)
    baselines = [
        (2.39, "GPT-2 (22M)", "black"),
        (1.8, "Gemma3 (train)", "green"),
        (2.0, "Gemma3 (val)", "purple"),
    ]
    for y, label, color in baselines:
        ax.axhline(
            y=y,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.85,
            label=label,
        )
    ax.set_title("Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=False, loc="best")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_grad_norm(
    metrics: Metrics,
    save_path: str,
    *,
    figsize: Tuple[float, float],
    dpi: int,
    max_points: int,
    smooth_window: int,
):
    if not metrics.iter_steps:
        raise RuntimeError("No iter records found (grad_norm).")

    steps, grads = _downsample(metrics.iter_steps, metrics.iter_grad_norm, max_points=max_points)
    grads_smooth = _moving_average(grads, window=smooth_window)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # High-contrast colors: R/G/B first (use blue for raw, green for smoothed)
    ax.plot(steps, grads, label="grad norm (raw)", color="blue", alpha=0.25, linewidth=1.0)
    ax.plot(steps, grads_smooth, label=f"grad norm (MA{smooth_window})", color="green")
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 grad norm")
    ax.legend(frameon=False, loc="best")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PRISM training curves from metrics.jsonl (publication-ready).")
    parser.add_argument(
        "--metrics",
        type=str,
        default=os.path.join("out-prism-tiny", "metrics.jsonl"),
        help="Path to metrics.jsonl written by train.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("out-prism-tiny", "analysis"),
        help="Directory to write figures",
    )
    parser.add_argument("--latex_2col", action="store_true", help="Paper-friendly defaults for 2-column \\linewidth.")
    parser.add_argument("--figsize", type=str, default=None, help="Figure size 'width,height' in inches.")
    parser.add_argument("--dpi", type=int, default=None, help="Output DPI.")
    parser.add_argument("--max_points", type=int, default=2000, help="Downsample grad_norm curve to this many points.")
    parser.add_argument("--smooth_window", type=int, default=200, help="Moving average window for grad_norm.")
    args = parser.parse_args()

    _setup_mpl_style(latex_2col=args.latex_2col)

    if args.figsize is None:
        # Slightly narrower than a typical 3.4in column width to leave breathing room in multi-subfigure layouts.
        figsize = (3.06, 2.2) if args.latex_2col else (8.0, 4.5)
    else:
        w_str, h_str = args.figsize.split(",")
        figsize = (float(w_str), float(h_str))

    dpi = args.dpi if args.dpi is not None else (600 if args.latex_2col else 300)

    metrics = _read_metrics_jsonl(args.metrics)

    loss_path = os.path.join(args.out_dir, "loss.png")
    grad_path = os.path.join(args.out_dir, "grad_norm.png")

    plot_loss(metrics, loss_path, figsize=figsize, dpi=dpi)
    plot_grad_norm(
        metrics,
        grad_path,
        figsize=figsize,
        dpi=dpi,
        max_points=args.max_points,
        smooth_window=args.smooth_window,
    )

    print(f"Saved loss plot: {loss_path}")
    print(f"Saved grad norm plot: {grad_path}")


if __name__ == "__main__":
    main()
