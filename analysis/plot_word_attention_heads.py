import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Allow running as `python analysis/plot_word_attention_heads.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tiktoken

from model import GPT, GPTConfig
from analysis.viz_prism import HookCapture, recompute_attention_weights


def _load_checkpoint_model(out_dir: str, device: str):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_args" in checkpoint:
        gptconf = GPTConfig(**checkpoint["model_args"])
    else:
        raise RuntimeError("Checkpoint missing model_args; cannot reliably reconstruct model config.")

    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, checkpoint


def _decode_token_labels(enc, token_ids: List[int]) -> List[str]:
    labels = []
    for tid in token_ids:
        s = enc.decode([tid])
        s = s.replace("\n", "\\n")
        s = s.replace("\t", "\\t")
        # Make spaces visible-ish
        s = s.replace(" ", "·")
        # Clamp very long tokens for tick labels
        if len(s) > 12:
            s = s[:9] + "..."
        labels.append(s)
    return labels


def _find_target_token_index(labels: List[str], target_word: str, occurrence: int = 0) -> Optional[int]:
    """
    Best-effort matching: compare normalized decoded token strings to target_word.
    labels are already escaped and have spaces replaced by '·', so normalize that too.
    """
    target = target_word.strip().lower()
    hits = []
    for i, lab in enumerate(labels):
        norm = lab.replace("·", " ").strip().lower()
        # Remove surrounding punctuation to handle "door," etc.
        norm = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", norm)
        if norm == target:
            hits.append(i)
    if not hits:
        return None
    if occurrence < 0 or occurrence >= len(hits):
        return hits[0]
    return hits[occurrence]


def _plot_attention_grid(
    att: torch.Tensor,
    labels: List[str],
    query_index: int,
    title_prefix: str,
    save_path: str,
    *,
    figsize: Tuple[float, float] = (16.0, 9.0),
    dpi: int = 300,
    tick_fontsize: int = 8,
    title_fontsize: int = 12,
    axis_label_fontsize: int = 10,
    cmap: str = "viridis",
    highlight_query: bool = True,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
):
    """
    att: (H, T, T) attention probabilities
    """
    H, T, _ = att.shape
    cols = min(4, H)
    rows = (H + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    data_vmax = float(att.detach().max().item())
    vmax = data_vmax if vmax is None else float(vmax)
    for h in range(H):
        r = h // cols
        c = h % cols
        ax = axes[r][c]
        data = att[h].detach().cpu().float().numpy()
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=float(vmin), vmax=vmax)
        ax.set_title(f"{title_prefix} head {h}", fontsize=title_fontsize)
        ax.set_xlabel("Key token", fontsize=axis_label_fontsize)
        ax.set_ylabel("Query token", fontsize=axis_label_fontsize)

        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels(labels, fontsize=tick_fontsize)

        if highlight_query:
            ax.axhline(query_index, color="white", linewidth=1.0, alpha=0.9)
            ax.axvline(query_index, color="white", linewidth=1.0, alpha=0.4)

        # Colorbar axis with the same height as the heatmap axes.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im, cax=cax)

    # hide unused axes
    for k in range(H, rows * cols):
        r = k // cols
        c = k % cols
        axes[r][c].axis("off")

    # Use tight layout to reduce clipping of tick labels/titles/colorbars.
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

def _plot_attention_single(
    att: torch.Tensor,
    labels: List[str],
    query_index: int,
    title: str,
    save_path: str,
    *,
    figsize: Tuple[float, float] = (8.0, 7.0),
    dpi: int = 300,
    tick_fontsize: int = 10,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    cmap: str = "viridis",
    highlight_query: bool = True,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
):
    """
    att: (T, T) attention probabilities for a single head
    """
    if att.dim() != 2 or att.shape[0] != att.shape[1]:
        raise ValueError(f"expected att to be (T,T), got {tuple(att.shape)}")

    T = att.shape[0]
    data_vmax = float(att.detach().max().item())
    vmax = data_vmax if vmax is None else float(vmax)
    data = att.detach().cpu().float().numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=float(vmin), vmax=vmax)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Key token", fontsize=axis_label_fontsize)
    ax.set_ylabel("Query token", fontsize=axis_label_fontsize)

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)

    if highlight_query:
        ax.axhline(query_index, color="white", linewidth=1.2, alpha=0.9)
        ax.axvline(query_index, color="white", linewidth=1.0, alpha=0.4)
    # Colorbar axis with the same height as the image.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PRISM attention matrices for all signal/noise heads at a target token position."
    )
    parser.add_argument("--out_dir", type=str, default="out-prism-tiny", help="Directory containing ckpt.pt")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda (default: auto)")
    parser.add_argument("--layer", type=int, default=4, help="Which transformer layer/block to hook (ignored if --all_layers)")
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help="If set, plot attention grids for every layer in the model (0..n_layer-1) in a single forward pass.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="If set, plot only this head index (single-head PNGs for signal+noise) instead of the full head grid.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text (BPE tokenized with GPT-2 tokenizer)")
    parser.add_argument("--target_word", type=str, default="door", help="Token string to locate, e.g. door")
    parser.add_argument(
        "--target_occurrence",
        type=int,
        default=0,
        help="If target word appears multiple times, choose occurrence index (0-based).",
    )
    parser.add_argument(
        "--target_token_index",
        type=int,
        default=None,
        help="Override: directly specify the token index to analyze (0-based).",
    )
    parser.add_argument("--max_tokens", type=int, default=64, help="Max prompt tokens to run through the model")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join("analysis", "word_attention"),
        help="Output directory for PNGs",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI")
    parser.add_argument("--tick_fontsize", type=int, default=8, help="Tick label font size")
    parser.add_argument("--title_fontsize", type=int, default=12, help="Subplot title font size")
    parser.add_argument("--axis_label_fontsize", type=int, default=10, help="Axis label font size")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name (e.g. viridis, Blues)")
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fix color scale max (e.g. 0.7). Default: per-plot max.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Fix color scale min. Default: 0.0.",
    )
    parser.add_argument(
        "--no_highlight",
        action="store_true",
        help="Disable the white query-token highlight lines (paper cleaner).",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="20,10",
        help="Figure size as 'width,height' in inches (applies to each signal/noise grid).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper-friendly defaults (bigger fonts, bigger figure, dpi=300). CLI values still override if passed.",
    )
    parser.add_argument(
        "--latex_2col",
        action="store_true",
        help="Defaults sized for a 2-column LaTeX figure at \\linewidth (single column width).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    if args.paper:
        # Only set defaults if user didn't explicitly pass them.
        if args.dpi == 300:
            args.dpi = 300
        if args.tick_fontsize == 8:
            args.tick_fontsize = 20
        if args.title_fontsize == 12:
            args.title_fontsize = 26
        if args.axis_label_fontsize == 10:
            args.axis_label_fontsize = 22
        if args.figsize == "20,10":
            args.figsize = "30,18"
        if args.cmap == "viridis":
            args.cmap = "Blues"

    if args.latex_2col:
        # Aim for a single-column figure included at width=\\linewidth in a 2-column layout.
        # Typical column widths are ~3.3-3.5 inches depending on template.
        if args.dpi == 300:
            args.dpi = 600
        if args.tick_fontsize == 8:
            args.tick_fontsize = 7
        if args.title_fontsize == 12:
            args.title_fontsize = 9
        if args.axis_label_fontsize == 10:
            args.axis_label_fontsize = 8
        if args.figsize == "20,10":
            # Wider aspect ratio for better token readability; still fits width=\\linewidth.
            args.figsize = "3.4,2.6"
        if args.cmap == "viridis":
            args.cmap = "Blues"

    try:
        w_str, h_str = args.figsize.split(",")
        fig_w, fig_h = float(w_str), float(h_str)
    except Exception as e:
        raise ValueError(f"Invalid --figsize {args.figsize!r}, expected 'width,height'") from e

    model, _ = _load_checkpoint_model(args.out_dir, device=device)

    if args.all_layers:
        layers = list(range(len(model.transformer.h)))
    else:
        if not (0 <= args.layer < len(model.transformer.h)):
            raise ValueError(f"--layer out of range, got {args.layer}, model has {len(model.transformer.h)} layers")
        layers = [args.layer]

    prompt_ids = enc.encode(args.prompt, allowed_special={"<|endoftext|>"})
    if not prompt_ids:
        raise ValueError("Prompt encodes to empty token list")
    prompt_ids = prompt_ids[: args.max_tokens]

    labels = _decode_token_labels(enc, prompt_ids)
    target_idx = args.target_token_index
    if target_idx is None:
        target_idx = _find_target_token_index(labels, args.target_word, occurrence=args.target_occurrence)

    if target_idx is None:
        raise RuntimeError(
            f"Could not locate target_word={args.target_word!r} in tokenized prompt. "
            f"Pass --target_token_index to override."
        )
    if not (0 <= target_idx < len(prompt_ids)):
        raise ValueError(f"target token index out of range: {target_idx} (T={len(prompt_ids)})")

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]

    # Register hooks on all requested layers and run a single forward pass.
    captures = {}
    handles = []
    for layer in layers:
        attn = model.transformer.h[layer].attn
        cap = HookCapture()
        captures[layer] = cap
        handles.append(attn.register_forward_pre_hook(cap))

    try:
        with torch.no_grad():
            _ = model(idx)
    finally:
        for h in handles:
            h.remove()

    safe_word = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.target_word.strip()) or "target"
    print(f"[PRISM] out_dir={args.out_dir} layers={layers} T={len(prompt_ids)} device={device}")
    print(f"[PRISM] target_word={args.target_word!r} target_token_index={target_idx} token_label={labels[target_idx]!r}")

    for layer in layers:
        attn = model.transformer.h[layer].attn
        cap = captures[layer]
        if cap.x is None:
            raise RuntimeError(f"Hook failed to capture attention input x for layer {layer}")

        att_sig, att_noise = recompute_attention_weights(attn, cap.x)
        att_sig = att_sig[0]
        att_noise = att_noise[0]

        base = os.path.join(args.save_dir, f"{os.path.basename(args.out_dir)}_layer{layer}_token{target_idx}_{safe_word}")
        if args.head is not None:
            if not (0 <= args.head < att_sig.size(0)):
                raise ValueError(f"--head out of range, got {args.head}, n_head is {att_sig.size(0)}")
            sig_path = base + f"_signal_head{args.head}.png"
            noise_path = base + f"_noise_head{args.head}.png"
            print(f"[PRISM] saving layer {layer} head {args.head}:\n  {sig_path}\n  {noise_path}")
            _plot_attention_single(
                att_sig[args.head],
                labels,
                query_index=target_idx,
                title=f"Signal L{layer} head {args.head}",
                save_path=sig_path,
                figsize=(fig_w, fig_h),
                dpi=args.dpi,
                tick_fontsize=args.tick_fontsize,
                title_fontsize=args.title_fontsize,
                axis_label_fontsize=args.axis_label_fontsize,
                cmap=args.cmap,
                highlight_query=not args.no_highlight,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            _plot_attention_single(
                att_noise[args.head],
                labels,
                query_index=target_idx,
                title=f"Noise L{layer} head {args.head}",
                save_path=noise_path,
                figsize=(fig_w, fig_h),
                dpi=args.dpi,
                tick_fontsize=args.tick_fontsize,
                title_fontsize=args.title_fontsize,
                axis_label_fontsize=args.axis_label_fontsize,
                cmap=args.cmap,
                highlight_query=not args.no_highlight,
                vmin=args.vmin,
                vmax=args.vmax,
            )
        else:
            sig_path = base + "_signal.png"
            noise_path = base + "_noise.png"

            print(f"[PRISM] saving layer {layer}:\n  {sig_path}\n  {noise_path}")
            _plot_attention_grid(
                att_sig,
                labels,
                query_index=target_idx,
                title_prefix=f"Signal L{layer}",
                save_path=sig_path,
                figsize=(fig_w, fig_h),
                dpi=args.dpi,
                tick_fontsize=args.tick_fontsize,
                title_fontsize=args.title_fontsize,
                axis_label_fontsize=args.axis_label_fontsize,
                cmap=args.cmap,
                highlight_query=not args.no_highlight,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            _plot_attention_grid(
                att_noise,
                labels,
                query_index=target_idx,
                title_prefix=f"Noise L{layer}",
                save_path=noise_path,
                figsize=(fig_w, fig_h),
                dpi=args.dpi,
                tick_fontsize=args.tick_fontsize,
                title_fontsize=args.title_fontsize,
                axis_label_fontsize=args.axis_label_fontsize,
                cmap=args.cmap,
                highlight_query=not args.no_highlight,
                vmin=args.vmin,
                vmax=args.vmax,
            )


if __name__ == "__main__":
    main()
