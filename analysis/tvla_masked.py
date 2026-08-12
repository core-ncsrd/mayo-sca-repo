# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""
E2.1 — Fixed-vs-Fixed TVLA on the MASKED MAYO-1 linear map.
Same protocol as E0.2 but firmware uses additive O=O0⊕O1 masking.
Mask is re-randomised internally (LCG PRNG) each 'p' call.

Expected result: max|t| < 4.5 (first-order masking removes secret dependency).
Compare with unmasked: max|t| = 195.66.
"""
import chipwhisperer as cw
import numpy as np
import os, time

PARAM_o  = 8
PARAM_v  = 78
N        = 1000    # more traces for masked (weaker signal expected)
SAMPLES  = 24000   # CW1173 Lite hw max ~24400; 24000 safe, covers O0*x + start of O1*x
GAIN_DB  = 45
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR   = os.path.join(BASE, 'traces_masked')
os.makedirs(OUTDIR, exist_ok=True)

RNG_SEED = 42

print('[*] Connecting...', flush=True)
scope  = cw.scope()
target = cw.target(scope)
scope.default_setup()
scope.clock.adc_src = 'clkgen_x1'
scope.adc.samples   = SAMPLES
scope.adc.timeout   = 15
scope.gain.db       = GAIN_DB
scope.io.nrst = 'low';  time.sleep(0.1)
scope.io.nrst = 'high'; time.sleep(0.5)
print(f'[*] ADC rate={scope.clock.adc_rate/1e6:.2f} MS/s  samples={SAMPLES}', flush=True)

def ss1_send(cmd_char, data_bytes):
    target.write(cmd_char + data_bytes.hex() + '\n')

def ss1_read(timeout_ms=300):
    t0 = time.time()
    buf = ''
    while (time.time()-t0) < timeout_ms/1000:
        buf += target.read()
        if '\n' in buf: break
        time.sleep(0.003)
    return buf.strip()

def load_O(O_mat):
    for row in range(PARAM_v):
        payload = bytes([row]) + bytes(O_mat[row])
        ss1_send('k', payload)
        time.sleep(0.002)
    ss1_read(100)

def capture_one(x_bytes):
    target.flush()
    scope.arm()
    ss1_send('p', x_bytes)
    ret = scope.capture(poll_done=True)
    if ret: return None
    return scope.get_last_trace()

rng = np.random.default_rng(0xDEADBEEF)
O_A = (rng.integers(0, 16, size=(PARAM_v, PARAM_o))).astype(np.uint8)
O_B = (rng.integers(0, 16, size=(PARAM_v, PARAM_o))).astype(np.uint8)
np.save(f'{OUTDIR}/O_A.npy', O_A)
np.save(f'{OUTDIR}/O_B.npy', O_B)
print(f'[*] O_A[0,0]={O_A[0,0]}  O_B[0,0]={O_B[0,0]}', flush=True)

x_rng = np.random.default_rng(RNG_SEED)
x_seq = (x_rng.integers(0, 16, size=(N, PARAM_o))).astype(np.uint8)
np.save(f'{OUTDIR}/x_seq.npy', x_seq)

# ── Group A ──────────────────────────────────────────────────────────────────
print('[*] Loading O_A...', flush=True)
load_O(O_A)
time.sleep(0.05)

traces_A = []
print(f'[*] Capturing {N} traces (group A, masked, secret=O_A)...', flush=True)
for i in range(N):
    t = capture_one(bytearray(x_seq[i]))
    if t is not None: traces_A.append(t)
    if i % 200 == 0:
        print(f'  A {i}/{N}  ok={len(traces_A)}', flush=True)

# ── Group B ──────────────────────────────────────────────────────────────────
print('[*] Loading O_B...', flush=True)
load_O(O_B)
time.sleep(0.05)

traces_B = []
print(f'[*] Capturing {N} traces (group B, masked, secret=O_B)...', flush=True)
for i in range(N):
    t = capture_one(bytearray(x_seq[i]))
    if t is not None: traces_B.append(t)
    if i % 200 == 0:
        print(f'  B {i}/{N}  ok={len(traces_B)}', flush=True)

nA, nB = len(traces_A), len(traces_B)
print(f'[+] Collected A={nA}  B={nB}', flush=True)

A = np.array(traces_A, dtype=np.float32)
B = np.array(traces_B, dtype=np.float32)
np.save(f'{OUTDIR}/traces_A.npy', A)
np.save(f'{OUTDIR}/traces_B.npy', B)

# ── Fixed-vs-Fixed Welch t-test ───────────────────────────────────────────────
mA, mB = A.mean(0), B.mean(0)
vA, vB = A.var(0, ddof=1), B.var(0, ddof=1)
denom  = np.sqrt(vA/nA + vB/nB).clip(1e-12)
t      = (mA - mB) / denom
np.save(f'{OUTDIR}/t_stat_fvf.npy', t)

abs_t   = np.abs(t)
leaking = np.where(abs_t > 4.5)[0]
print(f'\n[+] MASKED Fixed-vs-Fixed TVLA:', flush=True)
print(f'    N per group         : {nA}/{nB}', flush=True)
print(f'    Leaking |t|>4.5    : {len(leaking)} / {SAMPLES}', flush=True)
print(f'    Max |t|             : {abs_t.max():.2f}', flush=True)
if len(leaking):
    top5 = np.argsort(abs_t)[-5:][::-1]
    print(f'    Top-5 samples       : {list(top5)}', flush=True)
    print(f'    Top-5 |t|           : {list(abs_t[top5].round(2))}', flush=True)
else:
    print('    → No leakage at first-order threshold (expected for masked impl)', flush=True)

# Compare with unmasked result for paper
print(f'\n    --- Comparison ---', flush=True)
print(f'    Unmasked max|t| : 195.66  (E0.2)', flush=True)
print(f'    Masked   max|t| : {abs_t.max():.2f}  (E2.1)', flush=True)

target.dis()
print('[+] DONE', flush=True)
