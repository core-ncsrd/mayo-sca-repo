# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""
MAYO-1 TVLA capture - SimpleSerial v1.1 (correct format)
SS1 format: cmd_char + hex_data + newline (NO length prefix)
Firmware: mayo_lm.c compiled with SS_VER_1_1
"""
import chipwhisperer as cw
import numpy as np
import os
import time

PARAM_o = 8
N = 200
SAMPLES = 5000
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(BASE, 'traces')
os.makedirs(OUTDIR, exist_ok=True)

FIXED_X = bytearray([i & 0x0F for i in range(PARAM_o)])

print('[*] Connecting...', flush=True)
scope = cw.scope()
target = cw.target(scope)   # SS1 default
scope.default_setup()
scope.adc.samples = SAMPLES
scope.gain.db = 45

# Reset target
scope.io.nrst = 'low'
time.sleep(0.1)
scope.io.nrst = 'high'
time.sleep(0.5)
print('[*] Target reset', flush=True)

def ss1_send(cmd_char, data_bytes):
    """SS1: cmd + hex(data) + newline. No length prefix."""
    line = cmd_char + data_bytes.hex() + '\n'
    target.write(line)

def ss1_read(timeout_ms=300):
    t0 = time.time()
    buf = ''
    while (time.time() - t0) < timeout_ms / 1000:
        buf += target.read()
        if '\n' in buf:
            break
        time.sleep(0.005)
    return buf.strip()

# Comms check
print('[*] Comms check...', flush=True)
target.flush()
ss1_send('p', FIXED_X)
r = ss1_read(500)
print(f'    p response: {repr(r)}', flush=True)

# Single-shot trigger test
print('[*] Single-shot trigger test...', flush=True)
target.flush()
scope.arm()
ss1_send('p', FIXED_X)
ret = scope.capture(poll_done=True)
r2 = ss1_read(200)
print(f'    triggered={not ret}  response={repr(r2)}', flush=True)
tr = np.array(scope.get_last_trace())
print(f'    trace: mean={tr.mean():.5f} std={tr.std():.5f}', flush=True)

if ret:
    print('[!] No trigger. Stopping.', flush=True)
    target.dis()
    import sys; sys.exit(1)

print(f'\n[*] Trigger OK. Starting TVLA ({N} fixed + {N} random)...', flush=True)

fixed_traces, rand_traces = [], []

for i in range(N):
    target.flush()
    scope.arm()
    ss1_send('p', FIXED_X)
    if not scope.capture(poll_done=True):
        fixed_traces.append(scope.get_last_trace())
    if i % 50 == 0:
        print(f'  fixed {i}/{N} ok={len(fixed_traces)}', flush=True)

for i in range(N):
    x = bytearray(b & 0x0F for b in os.urandom(PARAM_o))
    target.flush()
    scope.arm()
    ss1_send('p', x)
    if not scope.capture(poll_done=True):
        rand_traces.append(scope.get_last_trace())
    if i % 50 == 0:
        print(f'  rand  {i}/{N} ok={len(rand_traces)}', flush=True)

nf, nr = len(fixed_traces), len(rand_traces)
print(f'\n[+] Collected: fixed={nf} rand={nr}', flush=True)

if nf > 10 and nr > 10:
    A = np.array(fixed_traces, dtype=np.float32)
    B = np.array(rand_traces, dtype=np.float32)
    np.save(f'{OUTDIR}/fixed_traces.npy', A)
    np.save(f'{OUTDIR}/rand_traces.npy', B)
    m1, m2 = A.mean(0), B.mean(0)
    v1, v2 = A.var(0, ddof=1), B.var(0, ddof=1)
    denom = np.sqrt(v1/nf + v2/nr); denom[denom < 1e-12] = 1e-12
    t = (m1 - m2) / denom
    np.save(f'{OUTDIR}/t_stat.npy', t)
    abs_t = np.abs(t)
    leaking = np.where(abs_t > 4.5)[0]
    print(f'[+] TVLA: {len(leaking)} samples |t|>4.5, max |t|={abs_t.max():.2f}', flush=True)
    if len(leaking):
        top5 = np.argsort(abs_t)[-5:][::-1]
        print(f'    Top-5: samples={list(top5)} |t|={list(abs_t[top5].round(2))}', flush=True)
else:
    print('[!] Too few traces.', flush=True)

target.dis()
print('[+] DONE', flush=True)
