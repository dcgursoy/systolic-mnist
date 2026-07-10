"""Package model weights + config + example digits into viz/model_data.js.

The visualizer no longer replays a recorded trace: it runs a cycle-accurate
JavaScript twin of the accelerator (viz/sim.js) so users can classify their
own drawn/uploaded digits. This script ships everything the twin needs:

  - the packed weight/bias memories (exactly the RTL's wmem/bmem contents,
    same controller streaming order)
  - per-layer tile counts and requantization constants
  - example MNIST test digits (with labels and true float32 logits from the
    PyTorch checkpoint, for the correctness side panel)
  - per-layer float scales so the twin can also compute a dequantized
    float reference for user-drawn digits

The twin is validated frame-exact against the Icarus Verilog trace by
viz/validate_sim.py — run it after any RTL or sim.js change.

Usage: python viz/build_demo.py
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from golden import QuantizedMLP, load_mem  # noqa: E402

N = 4


def read_packed(path, word_hexdigits, lanes, lane_bits):
    """Read a packed .mem file into a list of per-word lane arrays."""
    words = []
    mask = (1 << lane_bits) - 1
    sign = 1 << (lane_bits - 1)
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = int(line, 16)
        lanes_out = []
        for c in range(lanes):
            x = (v >> (lane_bits * c)) & mask
            if x >= sign:
                x -= 1 << lane_bits
            lanes_out.append(x)
        words.append(lanes_out)
    return words


def float_logits(imgs_q):
    import torch
    from net import MnistMLP
    model = MnistMLP()
    model.load_state_dict(
        torch.load(ROOT / "model" / "checkpoints" / "mlp_float.pt"))
    model.eval()
    x = torch.tensor(imgs_q.astype(np.float32) / 127.0)
    with torch.no_grad():
        return model(x).tolist()


def main():
    export = ROOT / "model" / "export"
    build = ROOT / "sim" / "build"
    meta = json.loads((export / "meta.json").read_text())

    wmem = read_packed(build / "weights.mem", N * 2, N, 8)
    bmem = read_packed(build / "biases.mem", N * 8, N, 32)

    imgs = load_mem(export / "test_images.mem", bits=8).reshape(-1, 784)
    labels = [int(l) for l in (export / "test_labels.txt").read_text().split()]
    m = QuantizedMLP.from_export(export)

    # index 0 first (the RTL-trace reference image), then the first
    # occurrence of each other digit
    example_idx = [0]
    for d in range(10):
        if d == labels[0]:
            continue
        example_idx.append(labels.index(d))
    ex_imgs = imgs[example_idx]
    examples = [{
        "label": labels[i],
        "pix": imgs[i].tolist(),
        "floatLogits": [round(v, 3) for v in fl],
        "goldenIntLogits": m.forward(imgs[i]).tolist(),
    } for i, fl in zip(example_idx, float_logits(ex_imgs))]

    nl = len(meta["layer_dims"])
    data = {
        "n": N,
        "layerDims": meta["layer_dims"],
        "inTiles": [d[0] // N for d in meta["layer_dims"]],
        "outBlocks": [-(-d[1] // N) for d in meta["layer_dims"]],
        "M": meta["requant_M"] + [0],
        "shift": meta["requant_shift"] + [1],
        "wScale": meta["scales"]["weight"],
        "actScale": [meta["scales"]["input"]] + meta["scales"]["activation"],
        "logitScale": meta["scales"]["logit"],
        "wmem": wmem,
        "bmem": bmem,
        "examples": examples,
        "referenceExample": 0,   # examples[0] must match sim/traces/img0.json
    }
    assert data["M"][nl - 1] == 0

    out = ROOT / "viz" / "model_data.js"
    out.write_text("window.MODEL = " + json.dumps(data, separators=(",", ":"))
                   + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB, "
          f"{len(wmem)} weight words, {len(examples)} examples)")


if __name__ == "__main__":
    main()
