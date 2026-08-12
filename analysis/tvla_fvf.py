# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""
E0.2 — Fixed-vs-Fixed TVLA on the secret O matrix (MAYO-1)
===========================================================
Both groups use the SAME x sequence (identical PRNG seed, interleaved A/B).
Any |t| > 4.5 is unambiguously secret-dependent leakage, not public-input handling.

Capture parameters (E0.3 corrections applied):
  ADC x1  (7.37 MS/s ≈ 1 sample/cycle) so 20 000 samples covers all
  78 × 8 = 624 GF(16) multiplications (~15 600 cycles) plus margins.

Protocol: 78 'k' commands to load one row of O each, then 'p' commands.
  'k' payload: 9 bytes = row_index (0-77) + 8 nibbles
  'p' payload: 8 bytes = x_i
"""

import chipwhisperer as cw
import numpy as np
import os
import time

PARAM_o   = 8      # cols of O (oil variables)
PARAM_v   = 78     # rows of O (vinegar variables)
N         = 500    # traces per group  (500 each = 1000 total for first run)
SAMPLES   = 20000  # 1 sample/cycle, covers full mat_mul + margin
GAIN_DB   = 45
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR    = os.path.join(BASE, 'traces_fvf')
os.makedirs(OUTDIR, exist_ok=True)

RNG_SEED  = 42     # same seed → identical x sequence for both groups

# ── Connect ────────────────────────────────────────────────────────────────────
print('[*] Connecting...', flush=True)
scope  = cw.scope()
target = cw.target(scope)
scope.default_setup()

# ADC x1: 1 sample per target clock cycle
scope.clock.adc_src   = 'clkgen_x1'
scope.adc.samples     = SAMPLES
scope.adc.timeout     = 10
scope.gain.db         = GAIN_DB

scope.io.nrst = 'low';  time.sleep(0.1)
scope.io.nrst = 'high'; time.sleep(0.5)
print(f'[*] ADC src={scope.clock.adc_src}  rate={scope.clock.adc_rate/1e6:.2f} MS/s  '
      f'samples={scope.adc.samples}', flush=True)

# ── Helper: load full O matrix via 78 'k' commands ────────────────────────────
def ss1_send(cmd_char, data_bytes):
    target.write(cmd_char + data_bytes.hex() + '\n')

def ss1_read(timeout_ms=200):
    t0 = time.time()
    buf = ''
    while (time.time() - t0) < timeout_ms / 1000:
        buf += target.read()
        if '\n' in buf:
            break
        time.sleep(0.003)
    return buf.strip()

def load_O(O_mat):
    """Load O (PARAM_v × PARAM_o uint8 array) row by row via 'k' command."""
    for row in range(PARAM_v):
        payload = bytes([row]) + bytes(O_mat[row])   # 1 + 8 = 9 bytes
        ss1_send('k', payload)
        time.sleep(0.003)
    # Quick sanity: send last row again and read ACK
    ss1_read(100)

def capture_one(x_bytes):
    target.flush()
    scope.arm()
    ss1_send('p', x_bytes)
    ret = scope.capture(poll_done=True)
    if ret:
        return None
    return scope.get_last_trace()

# ── Generate two fixed random secrets O_A and O_B ────────────────────────────
rng = np.random.default_rng(0xDEADBEEF)
O_A = (rng.integers(0, 16, size=(PARAM_v, PARAM_o))).astype(np.uint8)
O_B = (rng.integers(0, 16, size=(PARAM_v, PARAM_o))).astype(np.uint8)
np.save(f'{OUTDIR}/O_A.npy', O_A)
np.save(f'{OUTDIR}/O_B.npy', O_B)
print(f'[*] O_A[0,0]={O_A[0,0]}  O_B[0,0]={O_B[0,0]}  (sanity: should differ)', flush=True)

# ── Generate the SHARED x sequence (same PRNG seed for both groups) ──────────
x_rng    = np.random.default_rng(RNG_SEED)
x_seq    = (x_rng.integers(0, 16, size=(N, PARAM_o))).astype(np.uint8)
np.save(f'{OUTDIR}/x_seq.npy', x_seq)

# ── Group A: load O_A, capture N traces ──────────────────────────────────────
print('[*] Loading O_A...', flush=True)
load_O(O_A)
time.sleep(0.05)

traces_A = []
print(f'[*] Capturing {N} traces (group A, secret=O_A)...', flush=True)
for i in range(N):
    t = capture_one(bytearray(x_seq[i]))
    if t is not None:
        traces_A.append(t)
    if i % 100 == 0:
        print(f'    A {i}/{N}  ok={len(traces_A)}', flush=True)

# ── Group B: load O_B, capture N traces with the SAME x sequence ─────────────
print('[*] Loading O_B...', flush=True)
load_O(O_B)
time.sleep(0.05)

traces_B = []
print(f'[*] Capturing {N} traces (group B, secret=O_B)...', flush=True)
for i in range(N):
    t = capture_one(bytearray(x_seq[i]))
    if t is not None:
        traces_B.append(t)
    if i % 100 == 0:
        print(f'    B {i}/{N}  ok={len(traces_B)}', flush=True)

nA, nB = len(traces_A), len(traces_B)
print(f'[+] Collected A={nA}  B={nB}', flush=True)

if nA < 10 or nB < 10:
    print('[!] Too few traces — aborting.', flush=True)
    target.dis()
    import sys; sys.exit(1)

# ── Fixed-vs-fixed Welch t-test ───────────────────────────────────────────────
A = np.array(traces_A, dtype=np.float32)
B = np.array(traces_B, dtype=np.float32)
np.save(f'{OUTDIR}/traces_A.npy', A)
np.save(f'{OUTDIR}/traces_B.npy', B)

mA, mB   = A.mean(0), B.mean(0)
vA, vB   = A.var(0, ddof=1), B.var(0, ddof=1)
denom    = np.sqrt(vA/nA + vB/nB)
denom[denom < 1e-12] = 1e-12
t        = (mA - mB) / denom
np.save(f'{OUTDIR}/t_stat_fvf.npy', t)

abs_t    = np.abs(t)
leaking  = np.where(abs_t > 4.5)[0]
print(f'\n[+] Fixed-vs-Fixed TVLA result:', flush=True)
print(f'    Leaking samples |t|>4.5 : {len(leaking)} / {SAMPLES}', flush=True)
print(f'    Max |t|                 : {abs_t.max():.2f}', flush=True)
if len(leaking):
    top5 = np.argsort(abs_t)[-5:][::-1]
    print(f'    Top-5 samples           : {list(top5)}', flush=True)
    print(f'    Top-5 |t|               : {list(abs_t[top5].round(2))}', flush=True)
    # Estimate which cycle region (assuming 1 sample/cycle, trigger at sample 0)
    first_cycle = leaking[0]
    print(f'    First leaking cycle     : ~{first_cycle}', flush=True)
    # Each inner product loop takes ~25 cycles, 8 iterations per row
    row_est = first_cycle // (25 * PARAM_o)
    col_est = (first_cycle % (25 * PARAM_o)) // 25
    print(f'    Approx location         : row~{row_est}, col~{col_est}', flush=True)
else:
    print('    No leakage detected at this trace count.', flush=True)

target.dis()
print('[+] DONE', flush=True)
