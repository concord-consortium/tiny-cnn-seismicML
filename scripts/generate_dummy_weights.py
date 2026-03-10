#!/usr/bin/env python3
"""
Generate dummy compact_weights.json with random weights matching the
CompactSeismicCNN architecture. Produces the same computational profile
as real trained weights for benchmarking inference performance.

Usage (from repo root):
  python scripts/generate_dummy_weights.py
  python scripts/generate_dummy_weights.py --num_classes 3 --output explainer-app/public/models/compact_weights.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_dummy_weights(num_classes=2):
    """Generate random weights matching CompactSeismicCNN layer shapes.

    Architecture (from seismicModel.js):
      conv1: 1 -> 16, kernel 7, stride 2
      bn1: 16 channels
      pool1: size 2, stride 2
      conv2: 16 -> 32, kernel 5, stride 1
      bn2: 32 channels
      pool2: size 2, stride 2
      conv3: 32 -> 64, kernel 3, stride 1
      bn3: 64 channels
      global_pool: global average
      fc: 64 -> num_classes
    """
    rng = np.random.default_rng(42)
    out = {"num_classes": num_classes}

    # Conv layers: TF.js format (kernelSize, inChannels, outChannels)
    conv_specs = [
        ("conv1", 7, 1, 16),
        ("conv2", 5, 16, 32),
        ("conv3", 3, 32, 64),
    ]
    for name, k, in_ch, out_ch in conv_specs:
        out[f"{name}/kernel"] = rng.standard_normal((k, in_ch, out_ch)).tolist()
        out[f"{name}/bias"] = np.zeros(out_ch).tolist()

    # BatchNorm layers
    bn_specs = [("bn1", 16), ("bn2", 32), ("bn3", 64)]
    for name, channels in bn_specs:
        out[f"{name}/gamma"] = np.ones(channels).tolist()
        out[f"{name}/beta"] = np.zeros(channels).tolist()
        out[f"{name}/moving_mean"] = np.zeros(channels).tolist()
        out[f"{name}/moving_variance"] = np.ones(channels).tolist()

    # Dense layer: TF.js format (inFeatures, outFeatures)
    out["fc/kernel"] = rng.standard_normal((64, num_classes)).tolist()
    out["fc/bias"] = np.zeros(num_classes).tolist()

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy weights for CompactSeismicCNN benchmarking"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2,
        help="Number of output classes (default: 2)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "explainer-app" / "public" / "models" / "compact_weights.json",
        help="Output path for weights JSON"
    )
    args = parser.parse_args()

    weights = generate_dummy_weights(args.num_classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(weights, f, separators=(",", ":"))

    print(f"Generated dummy weights ({args.num_classes} classes) -> {args.output}")


if __name__ == "__main__":
    main()
