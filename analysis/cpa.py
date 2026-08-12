#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""
CPA on the MAYO-1 secret linear map  Ox = O * x_i   (MAYO-C src/mayo.c:486).

Leakage model
-------------
Inside mat_mul -> lincomb (simple_arithmetic.h) the elementary secret operation is
        prod = mul_f(O[r][j], x_i[j])          # GF(16) multiply, 4-bit result
where O[r][j] is a SECRET key nibble and x_i[j] is KNOWN (x_i is copied into the
signature at mayo.c:488).  We model power ~ HammingWeight(prod), 16 guesses/nibble.

Two effects make a naive CPA fail on this LINEAR map, both handled here:

  1. Input-load aliasing.  mul_f(1, x) = x (GF(16) identity), so the guess g=1
     has hypothesis HW(x) -- which matches the strong, pervasive leakage of the
     known input x being (re)loaded from memory every row.  A global-max CPA then
     returns g=1 for every nibble.  FIX: *partial correlation* -- linearly regress
     HW(x_j) out of the traces before correlating, which annihilates the g=1
     alias and exposes the true product leakage.  (A nibble whose true value IS 1
     is detected by the plain CPA + a flat partial spectrum -> reported as 1.)

  2. Temporal mixing.  Each x_j multiplies O[r][j] for ALL 78 rows r, so a
     full-trace search mixes rows.  But a 5000-sample window only spans the first
     few rows, and row 0's eight products are separated in time at a fixed cadence
     POI(j) = TRIG_OFFSET + j*SPACING (calibrated: 130 + 100*j at ADCx4).  FIX:
     correlate each nibble only inside a tight window around its POI.

Ground truth (firmware init_secret_O): O_flat[i]=(7*i+3)&0xF, so
O[0][j] = (7*j+3)&0xF -> [3, A, 1, 8, F, 6, D, 4].

Usage:
  python3 cpa.py                                  # attack row 0, calibrated POIs
  python3 cpa.py --trig-offset 130 --spacing 100 --halfwin 45
  python3 cpa.py --plain                          # naive global-max CPA (shows the g=1 alias)
"""
import os
import argparse
import numpy as np

HERE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ repository root (scripts live in analysis/); override with MAYO_SCA_ROOT.
PARAM_O = 8


def mul_f(a, b):
    """GF(16) multiply mod x^4+x+1, replicating MAYO-C simple_arithmetic.h:mul_f."""
    a &= 0xF; b &= 0xF
    p = ((a & 1) * b) ^ ((a & 2) * b) ^ ((a & 4) * b) ^ ((a & 8) * b)
    top = p & 0xF0
    return (p ^ (top >> 4) ^ (top >> 3)) & 0x0F


MUL = np.array([[mul_f(g, x) for x in range(16)] for g in range(16)], dtype=np.uint8)
HWv = np.array([bin(v).count("1") for v in range(16)], dtype=np.float64)


def corr_cols(traces, hyp):
    """Pearson corr of 1-D hyp (N,) with each column of centered traces (N,S)->(S,)."""
    h = hyp - hyp.mean()
    hn = np.sqrt((h * h).sum())
    if hn == 0:
        return np.zeros(traces.shape[1])
    Tn = np.sqrt((traces * traces).sum(0))
    Tn[Tn == 0] = np.finfo(np.float64).eps
    return (h @ traces) / (hn * Tn)


def residualize(T, ctrl):
    """Remove the component of every trace column linearly predicted by ctrl (N,)."""
    c = ctrl - ctrl.mean()
    d = (c * c).sum()
    if d == 0:
        return T
    return T - np.outer(c, (c @ T) / d)


def guess_scores(T, x_col, lo, hi, partial):
    """Peak |corr| for each of the 16 key guesses within window [lo:hi]."""
    Tw = T[:, lo:hi]
    if partial:
        Tw = residualize(Tw, HWv[x_col])
    return np.array([np.max(np.abs(corr_cols(Tw, HWv[MUL[g, x_col]]))) for g in range(16)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(HERE, "traces", "rand_traces.npy"))
    ap.add_argument("--pt", default=os.path.join(HERE, "traces", "rand_pt.npy"))
    ap.add_argument("--row", type=int, default=0, help="output row r of O to attack")
    ap.add_argument("--trig-offset", type=int, default=130, help="sample of row0/col0 product")
    ap.add_argument("--spacing", type=int, default=100, help="samples between successive products")
    ap.add_argument("--halfwin", type=int, default=45, help="half-width of per-nibble window")
    ap.add_argument("--plain", action="store_true", help="naive global-max CPA (no POI/partial)")
    args = ap.parse_args()

    traces = np.load(args.traces).astype(np.float64)
    pt = np.load(args.pt).astype(np.uint8)
    N, S = traces.shape
    T = traces - traces.mean(0)
    print(f"[*] loaded {N} traces x {S} samples; known inputs {pt.shape}")

    r = args.row
    true_O = [(7 * (r * PARAM_O + j) + 3) & 0xF for j in range(PARAM_O)]

    if args.plain:
        print("\n[plain global-max CPA] -- expect g=1 (input-load alias) to dominate")
        hdr = f"  {'j':>2} {'best':>5} {'true':>5} {'ok':>3} {'peak':>7} {'2nd':>7}"
        print(hdr)
        ok = 0
        for j in range(PARAM_O):
            sc = np.array([np.max(np.abs(corr_cols(T, HWv[MUL[g, pt[:, j]]]))) for g in range(16)])
            b = int(np.argmax(sc)); ok += (b == true_O[j])
            s = np.sort(sc)[::-1]
            print(f"  {j:>2} 0x{b:X}   0x{true_O[j]:X}   {'Y' if b==true_O[j] else '.':>3} {s[0]:7.3f} {s[1]:7.3f}")
        print(f"[*] plain CPA correct: {ok}/8 (the g=1 alias is why POI+partial is needed)")
        return

    print(f"\n[POI + partial-correlation CPA]  POI(j) = {args.trig_offset} + {args.spacing}*j"
          f"  (+/-{args.halfwin})")
    print(f"  {'j':>2} {'POI':>5} {'plain':>6} {'partial':>7} {'pick':>5} {'true':>5} {'ok':>3} {'peak':>6} {'margin':>7}")
    recovered, ok = [], 0
    for j in range(PARAM_O):
        ctr = args.trig_offset + args.spacing * j
        lo, hi = max(0, ctr - args.halfwin), min(S, ctr + args.halfwin)
        x_col = pt[:, j]
        plain = guess_scores(T, x_col, lo, hi, partial=False)
        part = guess_scores(T, x_col, lo, hi, partial=True)
        bp, bq = int(np.argmax(plain)), int(np.argmax(part))
        # partial regresses out HW(x); a true value of 1 is invisible to it but
        # dominant in the plain spectrum -> disambiguate.
        pick = 1 if (bp == 1 and part.max() < 0.30) else bq
        s = np.sort(part)[::-1]
        margin = s[0] - s[1]
        good = (pick == true_O[j]); ok += good
        recovered.append(pick)
        print(f"  {j:>2} {ctr:>5} 0x{bp:X}    0x{bq:X}     0x{pick:X}   0x{true_O[j]:X}   "
              f"{'Y' if good else '.':>3} {part.max():6.3f} {margin:7.3f}")

    print(f"\n[*] recovered O[{r}] = " + " ".join(f"{v:X}" for v in recovered))
    print(f"[*] ground truth O[{r}] = " + " ".join(f"{v:X}" for v in true_O))
    print(f"[*] correct nibbles (top-1): {ok}/{PARAM_O}")

    # Rank of the true key per nibble -- the leakage metric even when not top-1.
    ranks, true_corr = [], []
    for j in range(PARAM_O):
        ctr = args.trig_offset + args.spacing * j
        lo, hi = max(0, ctr - args.halfwin), min(S, ctr + args.halfwin)
        x_col = pt[:, j]
        partial = (true_O[j] != 1)          # O=1 is invisible to partial corr
        sc = guess_scores(T, x_col, lo, hi, partial=partial)
        order = np.argsort(sc)[::-1]
        ranks.append(int(np.where(order == true_O[j])[0][0]) + 1)
        true_corr.append(float(sc[true_O[j]]))
    keyspace = int(np.prod(ranks))
    print(f"[*] true-key rank per nibble: {ranks}  (random ~ 8.5 each)")
    print(f"[*] mean true-key rank: {np.mean(ranks):.2f}/16")
    print(f"[*] row-0 key-space reduced 16^8 = {16**8:,} -> prod(ranks) = {keyspace} "
          f"candidates (brute-forceable)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        ax[0].bar(range(PARAM_O), ranks, color=["#2a9d8f" if rk == 1 else "#e76f51" for rk in ranks])
        ax[0].axhline(8.5, color="gray", ls="--", lw=0.8, label="random guess (8.5)")
        ax[0].set_xlabel("secret nibble O[0][j]"); ax[0].set_ylabel("rank of true key (1=best)")
        ax[0].set_title(f"CPA true-key rank (mean {np.mean(ranks):.2f}/16)"); ax[0].legend()
        # per-guess partial-corr spectrum for nibble 0
        ctr = args.trig_offset; lo, hi = max(0, ctr - args.halfwin), ctr + args.halfwin
        sc0 = guess_scores(T, pt[:, 0], lo, hi, partial=(true_O[0] != 1))
        cols = ["#264653"] * 16; cols[true_O[0]] = "#e9c46a"
        ax[1].bar(range(16), sc0, color=cols)
        ax[1].set_xlabel("key guess (0..F); gold = true 0x%X" % true_O[0])
        ax[1].set_ylabel("peak |corr| @ POI"); ax[1].set_title("nibble j=0 guess spectrum")
        plt.tight_layout()
        out = os.path.join(HERE, "traces", "cpa_ranks.png")
        plt.savefig(out, dpi=110); print(f"[*] wrote {out}")
    except Exception as e:
        print(f"[!] plot skipped: {e}")


if __name__ == "__main__":
    main()
