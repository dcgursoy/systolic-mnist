"""Package a simulation trace into viz/trace_data.js for the visualizer.

Combines:
  - the per-cycle RTL trace (sim/traces/img0.json)
  - the input image + label
  - golden INT32 logits and float32 PyTorch logits (correctness side panel)
  - quantization metadata (for dequantizing logits to confidences)

into a single self-contained JS file, so viz/index.html works from file://
with no server.

Usage: python viz/build_demo.py [--trace sim/traces/img0.json]
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from golden import QuantizedMLP, load_mem  # noqa: E402


def float_logits(img_q):
    """Float32 model logits for the (reconstructed) image."""
    import torch
    from net import MnistMLP
    model = MnistMLP()
    model.load_state_dict(
        torch.load(ROOT / "model" / "checkpoints" / "mlp_float.pt"))
    model.eval()
    x = torch.tensor(img_q.astype(np.float32) / 127.0).unsqueeze(0)
    with torch.no_grad():
        return model(x)[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="sim/traces/img0.json")
    args = ap.parse_args()

    meta = json.loads((ROOT / "model" / "export" / "meta.json").read_text())
    frames, img, label = [], None, None
    for line in (ROOT / args.trace).open():
        d = json.loads(line)
        if d.get("ev") == "img":
            img, label = d["pix"], d["label"]
        else:
            frames.append(d)

    m = QuantizedMLP.from_export(ROOT / "model" / "export")
    img_q = np.array(img, dtype=np.int8)
    int_logits = m.forward(img_q).tolist()

    demo = {
        "n": 4,
        "img": img,
        "label": label,
        "goldenIntLogits": int_logits,
        "floatLogits": float_logits(img_q),
        "logitScale": meta["scales"]["logit"],
        "layerDims": meta["layer_dims"],
        "inTiles": [d[0] // 4 for d in meta["layer_dims"]],
        "outBlocks": [-(-d[1] // 4) for d in meta["layer_dims"]],
        "frames": frames,
    }
    out = ROOT / "viz" / "trace_data.js"
    out.write_text("window.DEMO = " + json.dumps(demo, separators=(",", ":"))
                   + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} cycles, label={label})")


if __name__ == "__main__":
    main()
