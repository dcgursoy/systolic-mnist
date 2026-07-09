"""Synthesis flows: gate-count estimates (Yosys generic) and a real
place-and-route timing study (Yosys synth_ecp5 + nextpnr-ecp5, Lattice
ECP5-85k).

Runs:
  1. Generic synth of pe / systolic_array (N=4 and N=8) / accel_top:
     technology-independent gate counts + longest topological path.
  2. ECP5 synth + nextpnr P&R of the wrapped accel_top: LUT/FF/DSP/BRAM
     utilization and Fmax from nextpnr's static timing analysis.

Results land in results/synth.json and synth/build/.

Usage: python synth/run_synth.py [--skip-pnr]
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "synth" / "build"
OSS = pathlib.Path.home() / "Downloads" / "oss-cad-suite"

ENV = os.environ.copy()
ENV["PATH"] = f"{OSS / 'bin'}{os.pathsep}{OSS / 'lib'}{os.pathsep}{ENV['PATH']}"

RTL = ["rtl/pe.sv", "rtl/skew_buffer.sv", "rtl/systolic_array.sv",
       "rtl/requantize.sv", "rtl/acc_bank.sv", "rtl/act_bank.sv",
       "rtl/accel_top.sv"]


def yosys(script, log_name):
    log = BUILD / log_name
    r = subprocess.run([str(OSS / "bin" / "yosys.exe"), "-p", script],
                       cwd=ROOT, capture_output=True, text=True, env=ENV)
    log.write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:] + r.stderr[-2000:])
        sys.exit(f"yosys failed ({log_name})")
    return r.stdout


def parse_stat(out, module=None):
    """Parse the last `stat` block (optionally for a given module)."""
    blocks = re.split(r"=== ", out)
    text = None
    for b in blocks:
        if module is None or b.startswith(module):
            text = b
    if text is None:
        return {}
    stats = {"cells": {}}
    for line in text.splitlines():
        m = re.match(r"\s+(\d+) cells\s*$", line)
        if m:
            stats["total_cells"] = int(m.group(1))
        m = re.match(r"\s+(\d+) wire bits\s*$", line)
        if m:
            stats["wire_bits"] = int(m.group(1))
        m = re.match(r"\s+(\d+)\s+(\$[\w$]+|\w+)\s*$", line)
        if m and not m.group(2).isdigit():
            stats["cells"][m.group(2)] = int(m.group(1))
    return stats


def parse_ltp(out):
    m = re.findall(r"Longest topological path in .* \(length=(\d+)\)", out)
    return int(m[-1]) if m else None


def generic_flows():
    res = {}
    read_all = f"read_verilog -sv -I sim/build {' '.join(RTL)}"
    read_arr = "read_verilog -sv rtl/pe.sv rtl/systolic_array.sv"

    # single PE
    out = yosys("read_verilog -sv rtl/pe.sv; hierarchy -top pe; "
                "synth -flatten; ltp; stat", "generic_pe.log")
    res["pe"] = parse_stat(out, "pe")
    res["pe"]["logic_depth"] = parse_ltp(out)

    # arrays at N=4 and N=8. `hierarchy -chparam` asserts in this yosys
    # build, so N=8 goes through a wrapper with N=8 as its default.
    tops = {4: ("systolic_array", read_arr),
            8: ("array_top_n8", read_arr + " synth/array_top_n8.sv")}
    for n, (top, read) in tops.items():
        out = yosys(f"{read}; hierarchy -top {top}; synth -flatten; ltp; stat",
                    f"generic_array_n{n}.log")
        res[f"array_n{n}"] = parse_stat(out)
        res[f"array_n{n}"]["logic_depth"] = parse_ltp(out)

    # full accelerator (memories included)
    out = yosys(f"{read_all}; hierarchy -top accel_top; synth -flatten; "
                f"ltp; stat", "generic_top.log")
    res["accel_top"] = parse_stat(out)
    res["accel_top"]["logic_depth"] = parse_ltp(out)
    return res


def ecp5_flow():
    js = BUILD / "accel_ecp5.json"
    out = yosys(
        f"read_verilog -sv -I sim/build {' '.join(RTL)} synth/accel_synth_wrap.sv; "
        f"hierarchy -top accel_synth_wrap; "
        f"synth_ecp5 -top accel_synth_wrap -json {js}",
        "ecp5_synth.log")
    stats = parse_stat(out, "accel_synth_wrap")

    r = subprocess.run(
        [str(OSS / "bin" / "nextpnr-ecp5.exe"), "--85k", "--json", str(js),
         "--freq", "50", "--placer", "heap", "--timing-allow-fail"],
        cwd=ROOT, capture_output=True, text=True, env=ENV)
    log = r.stdout + r.stderr
    (BUILD / "ecp5_pnr.log").write_text(log)
    if r.returncode != 0:
        print(log[-3000:])
        sys.exit("nextpnr failed")

    fmax = {}
    for m in re.finditer(
            r"Max frequency for clock\s+'([^']+)': ([\d.]+) MHz", log):
        fmax[m.group(1)] = float(m.group(2))
    util = {}
    for m in re.finditer(r"\s+(TRELLIS_(?:FF|COMB|IO)|DP16KD|MULT18X18D):"
                         r"\s+(\d+)/\s*(\d+)", log):
        util[m.group(1)] = {"used": int(m.group(2)), "avail": int(m.group(3))}
    return {"synth_cells": stats, "fmax_mhz": fmax, "utilization": util}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pnr", action="store_true")
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)

    res = {"generic": generic_flows()}
    if not args.skip_pnr:
        res["ecp5"] = ecp5_flow()

    out_file = ROOT / "results" / "synth.json"
    out_file.write_text(json.dumps(res, indent=2))

    g = res["generic"]
    print(f"PE:          {g['pe'].get('total_cells')} cells, "
          f"depth {g['pe'].get('logic_depth')}")
    for n in (4, 8):
        a = g[f"array_n{n}"]
        print(f"array {n}x{n}:  {a.get('total_cells')} cells, "
              f"depth {a.get('logic_depth')}")
    print(f"accel_top:   {g['accel_top'].get('total_cells')} cells, "
          f"depth {g['accel_top'].get('logic_depth')}")
    if "ecp5" in res:
        print("ECP5 fmax:", res["ecp5"]["fmax_mhz"])
        print("ECP5 util:", {k: v["used"] for k, v in
                             res["ecp5"]["utilization"].items()})
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
