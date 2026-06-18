#!/usr/bin/env python3
"""
Generate dummy weight JSON files with random weights matching the
CNN architectures. Produces the same computational profile as real
trained weights for benchmarking inference performance.

Usage (from repo root):
  python scripts/generate_dummy_weights.py
  python scripts/generate_dummy_weights.py --model standard
  python scripts/generate_dummy_weights.py --model compact --num_classes 3
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "explainer-app" / "public" / "models"


def _add_conv_bn(out, rng, conv_name, bn_name, kernel_size, in_ch, out_ch):
    """Add conv + batchnorm weights in TF.js format."""
    out[f"{conv_name}/kernel"] = rng.standard_normal((kernel_size, in_ch, out_ch)).tolist()
    out[f"{conv_name}/bias"] = np.zeros(out_ch).tolist()
    out[f"{bn_name}/gamma"] = np.ones(out_ch).tolist()
    out[f"{bn_name}/beta"] = np.zeros(out_ch).tolist()
    out[f"{bn_name}/moving_mean"] = np.zeros(out_ch).tolist()
    out[f"{bn_name}/moving_variance"] = np.ones(out_ch).tolist()


def _add_dense(out, rng, name, in_features, out_features):
    """Add dense layer weights in TF.js format."""
    out[f"{name}/kernel"] = rng.standard_normal((in_features, out_features)).tolist()
    out[f"{name}/bias"] = np.zeros(out_features).tolist()


def generate_compact_weights(num_classes=2):
    """Generate random weights matching CompactSeismicCNN (~9K params).

    Architecture: conv1(1→16,k7) → conv2(16→32,k5) → conv3(32→64,k3) → fc(64→C)
    """
    rng = np.random.default_rng(42)
    out = {"num_classes": num_classes}
    _add_conv_bn(out, rng, "conv1", "bn1", 7, 1, 16)
    _add_conv_bn(out, rng, "conv2", "bn2", 5, 16, 32)
    _add_conv_bn(out, rng, "conv3", "bn3", 3, 32, 64)
    _add_dense(out, rng, "fc", 64, num_classes)
    return out


def generate_standard_weights(num_classes=2):
    """Generate random weights matching SeismicCNN (~94K params).

    Architecture: conv1(1→32,k7) → conv2(32→64,k5) → conv3(64→128,k3) →
                  conv4(128→128,k3) → fc1(128→64) → fc2(64→C)
    """
    rng = np.random.default_rng(42)
    out = {"num_classes": num_classes}
    _add_conv_bn(out, rng, "conv1", "bn1", 7, 1, 32)
    _add_conv_bn(out, rng, "conv2", "bn2", 5, 32, 64)
    _add_conv_bn(out, rng, "conv3", "bn3", 3, 64, 128)
    _add_conv_bn(out, rng, "conv4", "bn4", 3, 128, 128)
    _add_dense(out, rng, "fc1", 128, 64)
    _add_dense(out, rng, "fc2", 64, num_classes)
    return out


GENERATORS = {
    "compact": ("compact_weights.json", generate_compact_weights),
    "standard": ("standard_weights.json", generate_standard_weights),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy weights for seismic CNN benchmarking"
    )
    parser.add_argument(
        "--model", choices=list(GENERATORS.keys()) + ["all"], default="all",
        help="Which model architecture (default: all)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of output classes (default: 2)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=MODELS_DIR,
        help="Output directory for weights JSON files"
    )
    args = parser.parse_args()

    models = list(GENERATORS.keys()) if args.model == "all" else [args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        filename, gen_fn = GENERATORS[model_name]
        weights = gen_fn(args.num_classes)
        output_path = args.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(weights, f, separators=(",", ":"))
        print(f"Generated {model_name} dummy weights ({args.num_classes} classes) -> {output_path}")


if __name__ == "__main__":
    main()
