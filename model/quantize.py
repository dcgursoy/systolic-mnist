"""Post-training INT8 quantization of the trained MLP + export for RTL.

Steps:
  1. Load the float checkpoint and measure float32 test accuracy.
  2. Calibrate activation ranges on training images (max-abs of float
     activations), derive per-tensor symmetric scales.
  3. Quantize weights (int8), biases (int32), fold requantization scales
     into integer multiplier/shift pairs (gemmlowp/TFLite style).
  4. Evaluate the pure-integer golden model on the full 10k test set.
  5. Export weights/biases/test-vectors as $readmemh hex files + meta.json.

Usage: python model/quantize.py
"""

import json
import pathlib

import numpy as np
import torch

from net import MnistMLP, LAYER_DIMS
from train import get_loaders, evaluate
from golden import QuantizedMLP, quantize_input, save_mem

ROOT = pathlib.Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "export"
N_CALIB = 2000       # calibration images from the training set
N_EXPORT = 200       # test images exported for the RTL testbench
REQUANT_M_BITS = 15  # normalize M into [2^14, 2^15) for precision


def calibrate_activation_scales(model, images):
    """Max-abs of the float pre-activation of each hidden layer output
    (post-ReLU range is what the next layer consumes)."""
    scales = []
    with torch.no_grad():
        x = images.flatten(1)
        for i, layer in enumerate(model.fc[:-1]):
            x = torch.relu(layer(x))
            scales.append(x.abs().max().item() / 127.0)
    return scales


def to_multiplier_shift(m_real):
    """Represent a real-valued requant scale as (int M, int shift) with
    M in [2^(bits-1), 2^bits). out = (acc * M) >> shift."""
    assert 0 < m_real < 1
    shift = 0
    while m_real * (1 << shift) < (1 << (REQUANT_M_BITS - 1)):
        shift += 1
    M = int(round(m_real * (1 << shift)))
    if M == (1 << REQUANT_M_BITS):  # rounding overflow edge case
        M >>= 1
        shift -= 1
    return M, shift


def main():
    torch.manual_seed(0)
    model = MnistMLP()
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "mlp_float.pt"))
    model.eval()

    train_loader, test_loader = get_loaders()
    float_acc = evaluate(model, test_loader)
    print(f"float32 test accuracy: {float_acc:.4f}")

    # ---- calibration ----
    calib = torch.cat([x for x, _ in train_loader], 0)[:N_CALIB]
    act_scales = calibrate_activation_scales(model, calib)
    s_in = 1.0 / 127.0  # input image scale (pixels mapped to 0..127)
    layer_in_scales = [s_in] + act_scales

    # ---- quantize weights / biases / requant params ----
    weights_q, biases_q, Ms, shifts, w_scales = [], [], [], [], []
    for l, layer in enumerate(model.fc):
        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        s_w = float(np.abs(w).max()) / 127.0
        w_scales.append(s_w)
        weights_q.append(np.clip(np.round(w / s_w), -127, 127).astype(np.int8))
        biases_q.append(np.round(b / (layer_in_scales[l] * s_w)).astype(np.int32))
        if l < len(model.fc) - 1:
            m_real = layer_in_scales[l] * s_w / layer_in_scales[l + 1]
            M, shift = to_multiplier_shift(m_real)
            Ms.append(M)
            shifts.append(shift)

    scales = {
        "input": s_in,
        "weight": w_scales,
        "activation": act_scales,
        "logit": layer_in_scales[-1] * w_scales[-1],
    }
    qmodel = QuantizedMLP(weights_q, biases_q, Ms, shifts, scales)

    # ---- evaluate integer model on full test set ----
    correct = total = 0
    disagree = 0
    all_imgs, all_labels = [], []
    for x, y in test_loader:
        x_q = quantize_input(x.numpy().reshape(len(x), -1))
        pred = qmodel.predict(x_q)
        correct += (pred == y.numpy()).sum()
        total += len(y)
        float_pred = model(x).argmax(1).numpy()
        disagree += (pred != float_pred).sum()
        all_imgs.append(x_q)
        all_labels.append(y.numpy())
    int8_acc = correct / total
    print(f"int8 golden-model test accuracy: {int8_acc:.4f}")
    print(f"float vs int8 prediction disagreements: {disagree}/{total}")

    # ---- export ----
    EXPORT_DIR.mkdir(exist_ok=True)
    for l, (w, b) in enumerate(zip(weights_q, biases_q)):
        save_mem(EXPORT_DIR / f"layer{l}_w.mem", w, bits=8)   # row-major (out, in)
        save_mem(EXPORT_DIR / f"layer{l}_b.mem", b, bits=32)

    imgs = np.concatenate(all_imgs)[:N_EXPORT]
    labels = np.concatenate(all_labels)[:N_EXPORT]
    save_mem(EXPORT_DIR / "test_images.mem", imgs, bits=8)
    (EXPORT_DIR / "test_labels.txt").write_text(
        "\n".join(str(int(v)) for v in labels) + "\n")

    meta = {
        "layer_dims": LAYER_DIMS,
        "requant_M": Ms,
        "requant_shift": shifts,
        "scales": scales,
        "n_test_images": int(len(imgs)),
        "float32_test_accuracy": float(float_acc),
        "int8_test_accuracy": float(int8_acc),
    }
    (EXPORT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
