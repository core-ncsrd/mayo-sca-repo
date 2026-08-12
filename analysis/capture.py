#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""
TVLA (fixed-vs-random) capture for the MAYO-1 secret linear map on ChipWhisperer.

Target firmware: firmware/mayo-linearmap-CW308_STM32F3.hex
  - device holds a fixed secret oil matrix O (v x o over GF(16))
  - 'p' command sends x_i (o = 8 GF(16) nibbles); device computes
        Ox = O * x_i        (== MAYO-C src/mayo.c:486, mat_mul(sk.O, x_i, Ox))
    under the trigger and returns 16 output nibbles.

TVLA methodology (non-specific fixed-vs-random t-test):
  - FIXED group : x_i = FIXED_X (constant)          -> leakage of O*FIXED_X
  - RANDOM group: x_i = fresh random 8 nibbles       -> also the CPA dataset
  Groups are interleaved with a random per-trace coin (best practice) so slow
  environmental drift cannot bias one class.

Outputs (in --out dir, default ./traces):
  fixed_traces.npy  (Nf, samples)   float32
  rand_traces.npy   (Nr, samples)   float32
  rand_pt.npy       (Nr, 8)         uint8   x_i used for each random trace (for CPA)
  fixed_pt.npy      (1, 8)          uint8   the fixed x_i
  tvla_t.npy        (samples,)      float32 Welch t-statistic
  tvla.png                          plot of |t| with +/-4.5 thresholds

Usage:
  python3 capture.py --flash            # program the .hex first, then capture
  python3 capture.py -n 1000 -s 5000    # 1000 traces/group, 5000 samples
"""
import os
import sys
import argparse
import numpy as np

import chipwhisperer as cw

HERE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ repository root (scripts live in analysis/); override with MAYO_SCA_ROOT.
FW_HEX = os.path.join(HERE, "firmware", "mayo-linearmap-CW308_STM32F3.hex")

PARAM_O = 8                                   # MAYO-1 oil variables
FIXED_X = bytes([0x0A, 0x03, 0x0C, 0x05, 0x0F, 0x01, 0x07, 0x09])  # constant nibbles


def setup_scope(samples, gain_db, freq):
    scope = cw.scope()
    scope.default_setup()
    scope.clock.clkgen_freq = freq
    # sample the ADC at 4x the device clock (standard CW-Lite synchronous capture)
    try:
        scope.clock.adc_mul = 4
    except Exception:
        scope.clock.adc_src = "clkgen_x4"
    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.gain.mode = "high"
    scope.gain.db = gain_db
    scope.adc.basic_mode = "rising_edge"
    scope.trigger.triggers = "tio4"
    scope.io.tio1 = "serial_rx"
    scope.io.tio2 = "serial_tx"
    # let clocks settle
    if hasattr(scope.clock, "clkgen_locked"):
        _ = scope.clock.clkgen_locked
    return scope


def flash(scope, hexpath):
    if not os.path.exists(hexpath):
        sys.exit(f"[!] firmware hex not found: {hexpath}\n    build it: "
                 f"cd firmware && make PLATFORM=CW308_STM32F3 CRYPTO_TARGET=NONE")
    print(f"[*] programming {os.path.basename(hexpath)} ...")
    prog = cw.programmers.STM32FProgrammer
    cw.program_target(scope, prog, hexpath)
    print("[*] programming done")


def capture_one(scope, target, x):
    scope.arm()
    target.simpleserial_write('p', bytearray(x))
    if scope.capture():
        return None  # timeout
    target.simpleserial_read('r', 16, timeout=250)
    return scope.get_last_trace()


def welch_t(a, b):
    """t = (mean_a - mean_b) / sqrt(var_a/na + var_b/nb), per sample point."""
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    na, nb = a.shape[0], b.shape[0]
    denom = np.sqrt(va / na + vb / nb)
    denom[denom == 0] = np.finfo(np.float64).eps
    return (ma - mb) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--ntraces", type=int, default=1000, help="traces per group")
    ap.add_argument("-s", "--samples", type=int, default=5000)
    ap.add_argument("-g", "--gain", type=int, default=25, help="ADC gain (dB)")
    ap.add_argument("-f", "--freq", type=float, default=7.37e6)
    ap.add_argument("--flash", action="store_true", help="program firmware first")
    ap.add_argument("--out", default=os.path.join(HERE, "traces"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scope = setup_scope(args.samples, args.gain, args.freq)
    target = cw.target(scope, cw.targets.SimpleSerial)
    if args.flash:
        flash(scope, FW_HEX)

    rng = np.random.default_rng(0xC0FFEE)
    fixed, rnd, rnd_pt = [], [], []
    n = args.ntraces
    print(f"[*] capturing {n} fixed + {n} random traces, {args.samples} samples")

    got_f = got_r = 0
    while got_f < n or got_r < n:
        use_fixed = (rng.random() < 0.5 and got_f < n) or got_r >= n
        if use_fixed:
            x = FIXED_X
        else:
            x = bytes(int(v) & 0x0F for v in rng.integers(0, 16, PARAM_O))
        w = capture_one(scope, target, x)
        if w is None:
            print("  [!] capture timeout, retrying")
            continue
        if use_fixed:
            fixed.append(w); got_f += 1
        else:
            rnd.append(w); rnd_pt.append(list(x)); got_r += 1
        if (got_f + got_r) % 200 == 0:
            print(f"    fixed={got_f} random={got_r}")

    fixed = np.asarray(fixed, dtype=np.float32)
    rnd = np.asarray(rnd, dtype=np.float32)
    rnd_pt = np.asarray(rnd_pt, dtype=np.uint8)

    np.save(os.path.join(args.out, "fixed_traces.npy"), fixed)
    np.save(os.path.join(args.out, "rand_traces.npy"), rnd)
    np.save(os.path.join(args.out, "rand_pt.npy"), rnd_pt)
    np.save(os.path.join(args.out, "fixed_pt.npy"), np.frombuffer(FIXED_X, np.uint8)[None, :])
    print(f"[*] saved traces to {args.out}")

    t = welch_t(fixed, rnd).astype(np.float32)
    np.save(os.path.join(args.out, "tvla_t.npy"), t)
    n_leak = int(np.sum(np.abs(t) > 4.5))
    print(f"[*] TVLA: max|t| = {np.nanmax(np.abs(t)):.2f}  "
          f"samples over 4.5 = {n_leak}/{t.size}")
    if n_leak:
        print(f"    LEAKAGE DETECTED (>4.5) at first sample index "
              f"{int(np.argmax(np.abs(t) > 4.5))}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 4))
        plt.plot(t, lw=0.7)
        plt.axhline(4.5, color="r", ls="--", lw=0.8)
        plt.axhline(-4.5, color="r", ls="--", lw=0.8)
        plt.title("TVLA fixed-vs-random t-test - MAYO-1 secret linear map (O * x)")
        plt.xlabel("sample"); plt.ylabel("Welch t")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "tvla.png"), dpi=110)
        print(f"[*] wrote {os.path.join(args.out, 'tvla.png')}")
    except Exception as e:
        print(f"[!] plot skipped: {e}")

    scope.dis(); target.dis()


if __name__ == "__main__":
    main()
