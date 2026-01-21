import argparse
import math
import os
import pickle
import sys
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import GPT, GPTConfig


@dataclass
class HookCapture:
    x: Optional[torch.Tensor] = None

    def __call__(self, module, inputs):
        # forward_pre_hook signature: (module, inputs) where inputs is a tuple
        self.x = inputs[0].detach()


def _apply_rotary_emb(x: torch.Tensor, base: float) -> torch.Tensor:
    """
    Re-implementation of the RoPE used inside PrismCausalSelfAttention.
    x: (B, H, T, D) with even D.
    """
    if x.dim() != 4:
        raise ValueError(f"expected x to be 4D (B,H,T,D), got shape {tuple(x.shape)}")
    B, H, T, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"RoPE requires even D, got D={D}")

    device = x.device
    half = D // 2

    # Use float32 for stable trig/softmax, then cast back.
    x_float = x.float()
    inv_freq = 1.0 / (float(base) ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.einsum('t,f->tf', t, inv_freq)  # (T, half)
    cos = freqs.cos()[None, None, :, :]  # (1, 1, T, half)
    sin = freqs.sin()[None, None, :, :]  # (1, 1, T, half)

    x1 = x_float[..., ::2]
    x2 = x_float[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y.to(dtype=x.dtype)


def recompute_attention_weights(attn_module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replay the internal PRISM attention math to recover the attention probability matrices.

    Returns:
      att_sig_probs: (B, n_head, T, T)
      att_noise_probs: (B, n_head, T, T)
    """
    if x.dim() != 3:
        raise ValueError(f"expected x to be 3D (B,T,C), got shape {tuple(x.shape)}")

    B, T, _ = x.shape
    n_head = int(attn_module.n_head)
    prism_head_dim = int(attn_module.prism_head_dim)

    # 1) Project with U
    h = attn_module.u_proj(x)  # (B, T, total_dim)
    h = h.view(B, T, 2 * n_head, prism_head_dim)
    h_sig = h[:, :, :n_head, :]
    h_noise = h[:, :, n_head:, :]

    # 2) (B, H, T, D)
    v_sig = h_sig.permute(0, 2, 1, 3).contiguous()
    v_noise = h_noise.permute(0, 2, 1, 3).contiguous()

    # 3) RoPE for Q/K
    signal_base = getattr(attn_module, "signal_rope_base", None)
    noise_base = getattr(attn_module, "noise_rope_base", None)
    if signal_base is None or noise_base is None:
        freq_mult = float(getattr(getattr(attn_module, "config", None), "prism_freq_multiplier", math.pi))
        signal_base = 10000.0 * freq_mult
        noise_base = 10000.0 / freq_mult

    qk_sig = _apply_rotary_emb(v_sig, base=float(signal_base))
    qk_noise = _apply_rotary_emb(v_noise, base=float(noise_base))

    # 4) Attention scores + causal masking + softmax
    scale = 1.0 / math.sqrt(prism_head_dim)

    def _masked_softmax(scores: torch.Tensor) -> torch.Tensor:
        if not hasattr(attn_module, "bias"):
            raise AttributeError("attn_module must have a causal mask buffer `bias`")
        bias = attn_module.bias[:, :, :T, :T]  # (1,1,T,T)
        scores = scores.masked_fill(bias == 0, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1)
        return probs

    scores_sig = (qk_sig @ qk_sig.transpose(-2, -1)) * scale
    scores_noise = (qk_noise @ qk_noise.transpose(-2, -1)) * scale
    att_sig = _masked_softmax(scores_sig)
    att_noise = _masked_softmax(scores_noise)

    return att_sig, att_noise


def _try_load_meta(dataset: str) -> Optional[dict]:
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def _encode_prompt_bpe(prompt: str) -> list:
    try:
        import tiktoken
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "BPE tokenization requested but tiktoken is not installed. "
            "Install tiktoken or use a dataset with meta.pkl (char-level)."
        ) from e
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(prompt, allowed_special={"<|endoftext|>"})


def _make_input_ids(
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    meta: Optional[dict] = None,
    prompt: Optional[str] = None,
    use_bpe: bool = False,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    if prompt:
        if use_bpe:
            prompt_ids = _encode_prompt_bpe(prompt)[:seq_len]
            if prompt_ids:
                x[:, : len(prompt_ids)] = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
        elif meta is not None:
            stoi = meta.get("stoi", {})
            prompt_ids = []
            for ch in prompt:
                if ch in stoi:
                    prompt_ids.append(int(stoi[ch]))
            prompt_ids = prompt_ids[:seq_len]
            if prompt_ids:
                x[:, : len(prompt_ids)] = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]

    return x

def _read_metrics_jsonl(out_dir: str) -> list:
    path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def plot_loss_curve(out_dir: str, save_path: str):
    records = _read_metrics_jsonl(out_dir)
    if not records:
        print(f"No metrics found at {os.path.join(out_dir, 'metrics.jsonl')}; skipping loss curve")
        return

    eval_recs = [r for r in records if r.get("type") == "eval" and "val_loss" in r]
    if not eval_recs:
        print("No eval records with val_loss found; skipping loss curve")
        return

    steps = [int(r["iter"]) for r in eval_recs]
    train_losses = [float(r.get("train_loss", float("nan"))) for r in eval_recs]
    val_losses = [float(r.get("val_loss", float("nan"))) for r in eval_recs]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(steps, train_losses, label="train/loss_eval")
    ax.plot(steps, val_losses, label="val/loss")
    ax.set_title("Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved loss curve to: {save_path}")


def plot_lambda_schedule(out_dir: str, save_path: str):
    records = _read_metrics_jsonl(out_dir)
    if not records:
        print(f"No metrics found at {os.path.join(out_dir, 'metrics.jsonl')}; skipping lambda schedule")
        return

    iter_recs = [r for r in records if r.get("type") == "iter" and r.get("prism_lambda") is not None]
    if not iter_recs:
        print("No iter records with prism_lambda found; skipping lambda schedule")
        return

    steps = [int(r["iter"]) for r in iter_recs]
    lambdas = [float(r["prism_lambda"]) for r in iter_recs]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(steps, lambdas, label="prism/lambda")
    ax.set_title("PRISM Lambda Schedule")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("lambda")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved lambda schedule to: {save_path}")


def _load_model_or_dummy(out_dir: str, device: str) -> Tuple[GPT, dict, str]:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model, checkpoint, "shakespeare_checkpoint"

    # Dummy fallback
    gptconf = GPTConfig(
        block_size=64,
        vocab_size=97,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        prism_expansion_ratio=2,
        prism_freq_multiplier=math.pi,
        prism_lambda_start=0.01,
        prism_lambda_end=0.1,
    )
    model = GPT(gptconf)
    return model, {}, "dummy_model"


def main():
    parser = argparse.ArgumentParser(description="PRISM attention visualization (Signal vs Noise) via hook + replay.")
    parser.add_argument("--out_dir", type=str, default="out-prism-char", help="Checkpoint directory (expects ckpt.pt).")
    parser.add_argument("--layer", type=int, default=0, help="Transformer block index to visualize.")
    parser.add_argument("--head", type=int, default=0, help="Attention head index to visualize.")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch element index to visualize.")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for visualization forward pass.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for visualization forward pass.")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda (default: auto).")
    parser.add_argument("--save_path", type=str, default=None, help="Defaults to <out_dir>/analysis/attention_map_preview.png")
    parser.add_argument("--prompt", type=str, default="\\nCAMILLO:\\n", help="Probe prompt (char-level or BPE).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path or os.path.join(args.out_dir, "analysis", "attention_map_preview.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    model, checkpoint, mode = _load_model_or_dummy(args.out_dir, device=device)
    model.eval().to(device)

    if not (0 <= args.layer < len(model.transformer.h)):
        raise ValueError(f"--layer out of range, got {args.layer}, model has {len(model.transformer.h)} layers")

    attn = model.transformer.h[args.layer].attn
    capture = HookCapture()
    hook_handle = attn.register_forward_pre_hook(capture)

    dataset = None
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        dataset = checkpoint["config"].get("dataset")
    dataset = dataset or "shakespeare_char"

    meta = _try_load_meta(dataset)
    use_bpe = dataset == "tinystories"
    vocab_size = int(model.config.vocab_size)
    idx = _make_input_ids(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        seq_len=min(args.seq_len, int(model.config.block_size)),
        device=device,
        meta=meta,
        prompt=args.prompt if mode == "shakespeare_checkpoint" else None,
        use_bpe=use_bpe,
    )

    with torch.no_grad():
        _ = model(idx)

    hook_handle.remove()
    if capture.x is None:
        raise RuntimeError("Hook did not capture attention input x; unexpected forward path")

    att_sig, att_noise = recompute_attention_weights(attn, capture.x)

    b = args.batch_idx
    h = args.head
    if not (0 <= b < att_sig.size(0)):
        raise ValueError(f"--batch_idx out of range, got {b}, batch size is {att_sig.size(0)}")
    if not (0 <= h < att_sig.size(1)):
        raise ValueError(f"--head out of range, got {h}, n_head is {att_sig.size(1)}")

    sig_map = att_sig[b, h].detach().cpu()
    noise_map = att_noise[b, h].detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(sig_map, aspect="auto", cmap="viridis")
    axes[0].set_title("Signal Head (Low Freq)")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(noise_map, aspect="auto", cmap="viridis")
    axes[1].set_title("Noise Head (High Freq)")
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"PRISM Attention Replay ({mode}) | layer={args.layer} head={args.head} T={sig_map.size(0)}")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"Saved attention comparison to: {save_path}")

    # Automation-friendly extras if training metrics exist
    plot_loss_curve(args.out_dir, os.path.join(args.out_dir, "analysis", "loss_curve.png"))
    plot_lambda_schedule(args.out_dir, os.path.join(args.out_dir, "analysis", "lambda_schedule.png"))


if __name__ == "__main__":
    main()
