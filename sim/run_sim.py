"""Compile and run the RTL simulation against the Python golden model.

Generates expected-value files (logits + hidden activations, bit-exact) from
model/golden.py, compiles the design with Icarus Verilog, runs the
self-checking testbench, and summarizes results into results/.

Usage:
  python sim/run_sim.py                      # 200 images, batch 16
  python sim/run_sim.py --images 1 --batch 1 --trace sim/traces/img0.json
  python sim/run_sim.py --images 20 --batch 1
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from golden import QuantizedMLP, load_mem, save_mem, requantize  # noqa: E402

RTL = [
    "rtl/pe.sv", "rtl/skew_buffer.sv", "rtl/systolic_array.sv",
    "rtl/requantize.sv", "rtl/acc_bank.sv", "rtl/act_bank.sv",
    "rtl/accel_top.sv",
]

OSS_ROOT = pathlib.Path.home() / "Downloads" / "oss-cad-suite"

ENV = os.environ.copy()
if OSS_ROOT.exists():
    ENV["PATH"] = (f"{OSS_ROOT / 'bin'}{os.pathsep}{OSS_ROOT / 'lib'}"
                   f"{os.pathsep}{ENV['PATH']}")


def tool(name):
    exe = OSS_ROOT / "bin" / f"{name}.exe"
    return str(exe) if exe.exists() else name


def synthetic_images():
    """Corner-case stimulus: drives requant saturation, all-zero rows, and
    alternating extremes that real calibrated data never produces."""
    imgs = np.zeros((4, 784), dtype=np.int8)
    imgs[1, :] = 127
    imgs[2, ::2] = 127
    imgs[3, :] = np.linspace(0, 127, 784).astype(np.int8)
    return imgs


def gen_expected(build, n_images, stress=False):
    """Golden-model logits + hidden activations for the first n images."""
    export = ROOT / "model" / "export"
    m = QuantizedMLP.from_export(export)
    if stress:
        imgs = synthetic_images()[:n_images]
        save_mem(build / "stress_images.mem", imgs, bits=8)
        labels = list(m.predict(imgs))  # no true labels; use golden preds
    else:
        imgs = load_mem(export / "test_images.mem", bits=8).reshape(-1, 784)[:n_images]
        labels = [int(l) for l in
                  (export / "test_labels.txt").read_text().split()][:n_images]

    logits, inter = m.forward(imgs.astype(np.int8), return_intermediates=True)
    act0 = requantize(inter[0], m.Ms[0], m.shifts[0])
    act1 = requantize(inter[1], m.Ms[1], m.shifts[1])

    save_mem(build / "exp_logits.mem", logits.astype(np.int32), bits=32)
    save_mem(build / "exp_act0.mem", act0, bits=8)
    save_mem(build / "exp_act1.mem", act1, bits=8)
    (build / "labels.mem").write_text(
        "\n".join(format(l, "02x") for l in labels) + "\n")
    return logits, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--trace", type=str, default=None)
    ap.add_argument("--vcd", type=str, default=None)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--stress", action="store_true",
                    help="synthetic corner-case images instead of MNIST")
    args = ap.parse_args()
    if args.stress:
        args.images = min(args.images, 4)

    build = ROOT / "sim" / "build"
    build.mkdir(parents=True, exist_ok=True)
    (ROOT / "sim" / "traces").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    gen_expected(build, args.images, stress=args.stress)

    vvp_file = build / "accel.vvp"
    if not args.skip_compile or not vvp_file.exists():
        cmd = [tool("iverilog"), "-g2012", "-I", str(build),
               "-o", str(vvp_file)] + RTL + ["tb/tb_accel.sv"]
        print("+", " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
        if r.returncode != 0:
            print(r.stdout + r.stderr)
            sys.exit("compile failed")

    plusargs = [f"+images={args.images}", f"+batch={args.batch}"]
    if args.stress:
        plusargs.append("+imgfile=sim/build/stress_images.mem")
    if args.trace:
        plusargs.append(f"+trace={args.trace}")
    if args.vcd:
        plusargs.append(f"+vcd={args.vcd}")
    cmd = [tool("vvp"), str(vvp_file)] + plusargs
    print("+", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
    elapsed = time.time() - t0
    out = r.stdout
    print(out[-4000:] if len(out) > 4000 else out)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit("simulation failed")

    # parse summary
    def grab(pattern, cast=int):
        mt = re.search(pattern, out)
        return cast(mt.group(1)) if mt else None

    summary = {
        "images": args.images,
        "batch": args.batch,
        "act_errors": grab(r"ACTCHECK errors=(\d+)"),
        "act_checked": grab(r"ACTCHECK errors=\d+ checked=(\d+)"),
        "logit_errors": grab(r"LOGITCHECK errors=(\d+)"),
        "logit_checked": grab(r"LOGITCHECK errors=\d+ checked=(\d+)"),
        "preds_total": grab(r"PREDS total=(\d+)"),
        "preds_match_label": grab(r"PREDS total=\d+ match_label=(\d+)"),
        "preds_match_golden": grab(r"PREDS total=\d+ match_label=\d+ match_golden=(\d+)"),
        "total_cycles": grab(r"TOTALCYCLES runs=\d+ cycles=(\d+)"),
        "runs": grab(r"TOTALCYCLES runs=(\d+)"),
        "coverage_pct": grab(r"COVERAGE hit=\d+ total=\d+ pct=([\d.]+)", float),
        "coverage_hit": grab(r"COVERAGE hit=(\d+)"),
        "coverage_total": grab(r"COVERAGE hit=\d+ total=(\d+)"),
        "tb_pass": "TB PASS" in out,
        "wall_seconds": round(elapsed, 1),
        "stress": args.stress,
        "cov_bins": dict(re.findall(r"(\w+)=(\d+)", " ".join(
            re.findall(r"COVBIN (.*)", out)))),
    }
    if summary["preds_total"]:
        summary["rtl_accuracy"] = summary["preds_match_label"] / summary["preds_total"]
        summary["rtl_vs_golden_agreement"] = (
            summary["preds_match_golden"] / summary["preds_total"])
    if summary["total_cycles"] and summary["preds_total"]:
        summary["cycles_per_image"] = summary["total_cycles"] / summary["preds_total"]

    tag = f"i{args.images}_b{args.batch}"
    out_file = ROOT / "results" / f"sim_{tag}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_file}")
    if not summary["tb_pass"]:
        sys.exit("TB FAIL")


if __name__ == "__main__":
    main()
