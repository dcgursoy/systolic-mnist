"""Bit-exact integer golden model of the quantized MLP.

This file is the *specification* for the RTL datapath. Every arithmetic
operation here has a direct hardware counterpart:

    acc      = W_q @ x_q + b_q                    # int8 x int8 MACs into int32
    hidden   = clamp((acc * M + 2^(shift-1)) >> shift, 0, 127)   # requantize+ReLU
    logits   = acc of final layer (raw int32)     # argmax needs no requantize

Conventions (must match RTL exactly):
  - x_q, W_q are int8; x_q is non-negative (0..127) after ReLU / input scaling.
  - Accumulation in int32 (no saturation needed: worst case 784 * 127 * 127
    ~= 1.26e7 << 2^31).
  - Requantize multiply is int32 * int32 -> int64, then arithmetic right
    shift with round-half-up bias, then clamp to [0, 127]. The clamp-at-0
    folds ReLU into requantization.
"""

import json
import pathlib

import numpy as np


def requantize(acc, M, shift):
    """int32 accumulator -> int8 activation. Matches RTL requantize unit."""
    prod = acc.astype(np.int64) * np.int64(M) + (np.int64(1) << (shift - 1))
    q = prod >> shift  # numpy >> on signed ints is arithmetic, same as SV >>>
    return np.clip(q, 0, 127).astype(np.int8)


class QuantizedMLP:
    def __init__(self, weights, biases, Ms, shifts, scales):
        self.weights = weights  # list of int8 (out, in)
        self.biases = biases    # list of int32 (out,)
        self.Ms = Ms            # per-hidden-layer requant multiplier
        self.shifts = shifts    # per-hidden-layer requant shift
        self.scales = scales    # dict of float scales (for dequant/display)

    @classmethod
    def from_export(cls, export_dir):
        export_dir = pathlib.Path(export_dir)
        meta = json.loads((export_dir / "meta.json").read_text())
        weights, biases = [], []
        for l, (in_f, out_f) in enumerate(meta["layer_dims"]):
            w = load_mem(export_dir / f"layer{l}_w.mem", bits=8)
            weights.append(w.reshape(out_f, in_f).astype(np.int8))
            biases.append(load_mem(export_dir / f"layer{l}_b.mem", bits=32))
        return cls(weights, biases, meta["requant_M"], meta["requant_shift"],
                   meta["scales"])

    def forward(self, x_q, return_intermediates=False):
        """x_q: int8 array (784,) or (N, 784). Returns int32 logits."""
        x = np.atleast_2d(x_q).astype(np.int32)
        inter = []
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            acc = x @ w.T.astype(np.int32) + b.astype(np.int32)
            inter.append(acc.copy())
            if l < len(self.weights) - 1:
                x = requantize(acc, self.Ms[l], self.shifts[l]).astype(np.int32)
            else:
                x = acc
        logits = x if x_q.ndim > 1 else x[0]
        if return_intermediates:
            return logits, inter
        return logits

    def predict(self, x_q):
        return self.forward(x_q).argmax(axis=-1)


def quantize_input(img_float):
    """float image in [0,1] -> int8 in [0,127]."""
    return np.clip(np.round(img_float * 127), 0, 127).astype(np.int8)


def load_mem(path, bits):
    """Read a $readmemh-style hex file into a signed numpy array."""
    vals = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.split("//")[0].strip()
        if not line:
            continue
        v = int(line, 16)
        if v >= 1 << (bits - 1):
            v -= 1 << bits
        vals.append(v)
    dtype = np.int8 if bits == 8 else np.int32
    return np.array(vals, dtype=dtype)


def save_mem(path, arr, bits):
    """Write a signed numpy array as a $readmemh-compatible hex file."""
    mask = (1 << bits) - 1
    digits = bits // 4
    lines = [format(int(v) & mask, f"0{digits}x") for v in arr.flatten()]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")
