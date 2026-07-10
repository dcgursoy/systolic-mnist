"""Validate the JavaScript hardware twin against the RTL.

Two checks, both run in headless Chromium via Playwright:

  1. FRAME-EXACT: simulate the reference image in sim.js and diff every
     field of every per-cycle frame against the Icarus Verilog trace
     (sim/traces/img0.json) — controller state, layer/tile/block, every
     PE's activation/weight/partial-sum, drain bits, accumulators, and
     requantize events. One mismatch anywhere fails the build.
  2. LOGIT-EXACT: run the twin (frames off) over all exported test images
     and compare INT32 logits against the Python golden model.

Usage: python viz/validate_sim.py
"""

import json
import pathlib
import sys

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from golden import QuantizedMLP, load_mem  # noqa: E402


def load_trace(path):
    frames = []
    for line in path.open():
        d = json.loads(line)
        if "ev" not in d:
            frames.append(d)
    return frames


def diff_frames(rtl, js):
    if len(rtl) != len(js):
        return f"frame count differs: RTL {len(rtl)} vs JS {len(js)}"
    for i, (a, b) in enumerate(zip(rtl, js)):
        for key in ("c", "st", "l", "t", "j", "dv", "acc", "pe", "req"):
            av, bv = a.get(key), b.get(key)
            if av != bv:
                return (f"frame {i} (cycle {a['c']}) field '{key}':\n"
                        f"  RTL: {json.dumps(av)[:300]}\n"
                        f"  JS : {json.dumps(bv)[:300]}")
    return None


def main():
    trace = load_trace(ROOT / "sim" / "traces" / "img0.json")
    imgs = load_mem(ROOT / "model" / "export" / "test_images.mem",
                    bits=8).reshape(-1, 784)
    golden = QuantizedMLP.from_export(ROOT / "model" / "export")
    golden_logits = golden.forward(imgs.astype(np.int8))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("about:blank")
        page.add_script_tag(path=str(ROOT / "viz" / "model_data.js"))
        page.add_script_tag(path=str(ROOT / "viz" / "sim.js"))

        # 1. frame-exact vs RTL trace
        js_frames = page.evaluate(
            "() => AccelSim.simulate(MODEL.examples[MODEL.referenceExample]"
            ".pix, MODEL, {record: true}).frames")
        err = diff_frames(trace, js_frames)
        if err:
            print("FRAME MISMATCH:\n" + err)
            sys.exit(1)
        print(f"frame-exact: {len(js_frames)} frames match the RTL trace")

        # 2. logit-exact vs Python golden over all exported images
        js_logits = page.evaluate(
            "(imgs) => imgs.map(im => AccelSim.simulate(im, MODEL, "
            "{record: false}).logits)", imgs.tolist())
        mism = 0
        for i, (jl, gl) in enumerate(zip(js_logits, golden_logits)):
            if list(jl) != [int(v) for v in gl]:
                mism += 1
                if mism <= 3:
                    print(f"logit mismatch img {i}: JS {jl} vs golden "
                          f"{list(gl)}")
        if mism:
            print(f"LOGIT MISMATCHES: {mism}/{len(imgs)}")
            sys.exit(1)
        print(f"logit-exact: {len(imgs)}/{len(imgs)} images match the "
              f"Python golden model")
        browser.close()
    print("JS twin VALIDATED")


if __name__ == "__main__":
    main()
