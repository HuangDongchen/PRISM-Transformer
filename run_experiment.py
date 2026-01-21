import argparse
import os
import subprocess
import sys


def run_step(cmd, cwd=None):
    print(f"[PRISM Pipeline] Running: {cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def main():
    parser = argparse.ArgumentParser(description="One-click PRISM TinyStories pipeline: prep -> train -> visualize.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use (default: current).")
    parser.add_argument(
        "--no_install_deps",
        action="store_true",
        help="Disable auto-install of Python deps (tiktoken/datasets/tqdm/matplotlib/tensorboard).",
    )
    parser.add_argument("--skip_prep", action="store_true", help="Skip data preparation step.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step.")
    parser.add_argument("--skip_viz", action="store_true", help="Skip visualization step.")
    parser.add_argument("--out_dir", default="out-prism-tiny", help="Training output dir.")
    parser.add_argument("--prompt", default="Alice had a little dog...", help="Probe prompt for visualization.")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing checkpoint and start from scratch.")
    parser.add_argument("--device", default=None, help="Override device (cpu|cuda). Default: auto-detect.")
    args = parser.parse_args()

    py = f"\"{args.python}\""

    # Dependency bootstrap (keep torch installation explicit; it's platform/CUDA specific).
    def py_has_pkg(pkg: str) -> bool:
        check = (
            "import importlib.util as u; "
            f"print('1' if u.find_spec('{pkg}') is not None else '0')"
        )
        result = subprocess.run(f"{py} -c \"{check}\"", shell=True, capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip().endswith("1")

    want_install = not args.no_install_deps
    missing = []
    for pkg in ["tiktoken", "datasets", "tqdm", "matplotlib", "tensorboard"]:
        if not py_has_pkg(pkg):
            missing.append(pkg)

    if missing:
        if want_install:
            run_step(f"{py} -m pip install -U " + " ".join(missing))
        else:
            raise RuntimeError(
                "Missing required packages: "
                + ", ".join(missing)
                + ". Install them manually or re-run without --no_install_deps."
            )

    if not py_has_pkg("torch"):
        raise RuntimeError(
            "PyTorch is not installed in this environment. Install a CUDA-enabled torch build first, then re-run."
        )

    # Device auto-detect (used as an override for train.py).
    device_override = args.device
    if device_override is None:
        detect = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
        result = subprocess.run(f"{py} -c \"{detect}\"", shell=True, capture_output=True, text=True)
        device_override = result.stdout.strip() if result.returncode == 0 else "cpu"
        if device_override not in ("cpu", "cuda"):
            device_override = "cpu"

    # 1) Data prep
    train_bin = os.path.join("data", "tinystories", "train.bin")
    val_bin = os.path.join("data", "tinystories", "val.bin")
    if not args.skip_prep:
        if os.path.exists(train_bin) and os.path.exists(val_bin):
            print("[PRISM Pipeline] Data already prepared; skipping prep.")
        else:
            run_step(f"{py} data/tinystories/prepare.py")

    # 2) Train
    if not args.skip_train:
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        if (not args.fresh) and os.path.exists(ckpt_path):
            print(f"[PRISM Pipeline] Found checkpoint at {ckpt_path}; resuming training.")
            run_step(
                f"{py} train.py config/train_prism_tiny.py --out_dir={args.out_dir} --init_from=resume --device={device_override}"
            )
        else:
            if args.fresh and os.path.exists(ckpt_path):
                print(f"[PRISM Pipeline] --fresh set; starting from scratch despite existing {ckpt_path}.")
            run_step(
                f"{py} train.py config/train_prism_tiny.py --out_dir={args.out_dir} --init_from=scratch --device={device_override}"
            )

    # 3) Visualization
    if not args.skip_viz:
        print("[PRISM Pipeline] Generating visualizations...")
        viz_layer = 4
        # If there's no checkpoint yet, viz_prism will fall back to a small dummy model;
        # in that case, layer 4 may be out of range, so default to layer 0.
        if not os.path.exists(os.path.join(args.out_dir, "ckpt.pt")):
            viz_layer = 0
        run_step(
            f"{py} analysis/viz_prism.py --out_dir {args.out_dir} --prompt \"{args.prompt}\" --layer {viz_layer} --device={device_override}"
        )


if __name__ == "__main__":
    main()
