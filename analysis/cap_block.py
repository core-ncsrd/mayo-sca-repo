# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
import os
#!/usr/bin/env python3
"""Drift-robust BLOCK-INTERLEAVED fixed-vs-fixed TVLA for masked_shuffle.
Small alternating A/B blocks (infrequent, drainable key reloads) so A and B
sample the same slow-drift trajectory -> drift cancels in the A-B statistic.
Uses the proven campaign load_O (worked reliably at low reload frequency)."""
import chipwhisperer as cw
import numpy as np, os, time, json
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FW=f'{BASE}/firmware'; OUT=f'{BASE}/paper_assets'
LOG=open(f'{OUT}/block.log','a',buffering=1)
def log(*a): m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
PARAM_o,PARAM_v=8,78
HEXm=f'{FW}/mayo-lm-masked2-CW308_STM32F3.hex'
scope=cw.scope(); scope.default_setup(); target=cw.target(scope)
def reset(): scope.io.nrst='low'; time.sleep(0.1); scope.io.nrst='high'; time.sleep(0.4)
def ss(c,b): target.write(c+bytes(b).hex()+'\n')
def rd(ms=100):
    t0=time.time();buf=''
    while(time.time()-t0)<ms/1000:
        buf+=target.read()
        if '\n' in buf: break
        time.sleep(0.002)
    return buf
def loadO(O):
    target.flush()
    for r in range(PARAM_v): ss('k',bytes([r])+bytes(O[r])); time.sleep(0.0018); target.read()
    rd(120); target.flush()
def cap(x,S):
    target.flush(); scope.adc.samples=S; scope.arm(); ss('p',x)
    return None if scope.capture(poll_done=True) else scope.get_last_trace()

N=1600; BS=25; SAMPLES=24000
scope.clock.adc_src='clkgen_x1'; scope.adc.timeout=10; scope.gain.db=45
log('[flash] masked_shuffle'); cw.program_target(scope,cw.programmers.STM32FProgrammer,HEXm); reset()
rng=np.random.default_rng(0xDEADBEEF)
O_A=rng.integers(0,16,(PARAM_v,PARAM_o)).astype(np.uint8)
O_B=rng.integers(0,16,(PARAM_v,PARAM_o)).astype(np.uint8)
xr=np.random.default_rng(42); xseq=xr.integers(0,16,(N,PARAM_o)).astype(np.uint8)
A=[];B=[]; nblocks=N//BS; i=0
for b in range(nblocks):
    key=O_A if (b%2==0) else O_B; tag='A' if b%2==0 else 'B'
    loadO(key); time.sleep(0.03)
    got=0
    for k in range(BS):
        t=cap(bytes(xseq[i]),SAMPLES); i+=1
        if t is not None: (A if tag=='A' else B).append(t); got+=1
    if b%8==0: log(f'  block {b}/{nblocks} tag={tag} got={got} A={len(A)} B={len(B)}')
A=np.array(A,np.float32); B=np.array(B,np.float32)
od=f'{BASE}/traces_masked2_bi'; os.makedirs(od,exist_ok=True)
np.save(f'{od}/traces_A.npy',A); np.save(f'{od}/traces_B.npy',B)
nA,nB=A.shape[0],B.shape[0]
d=np.sqrt(A.var(0,ddof=1)/nA+B.var(0,ddof=1)/nB); d[d<1e-12]=1e-12
t=(A.mean(0)-B.mean(0))/d; at=np.abs(t); np.save(f'{OUT}/masked_shuffle_bi_t.npy',t)
# drift floor within a group (same secret)
def maxt(P,Q):
    dd=np.sqrt(P.var(0,ddof=1)/P.shape[0]+Q.var(0,ddof=1)/Q.shape[0]); dd[dd<1e-12]=1e-12
    return float(np.abs((P.mean(0)-Q.mean(0))/dd).max())
hA=nA//2; hB=nB//2
RES=dict(nA=int(nA),nB=int(nB),samples=int(SAMPLES),block_size=BS,
    max_t=float(at.max()),n_over_4p5=int((at>4.5).sum()),argmax=int(at.argmax()),
    pass45=bool(at.max()<4.5),
    driftfloor_A1A2=maxt(A[:hA],A[hA:]), driftfloor_B1B2=maxt(B[:hB],B[hB:]))
# TtD
ttd=[]
nmin=min(nA,nB)
for n in [50,100,200,300,500,700,nmin]:
    if n<=nmin:
        dd=np.sqrt(A[:n].var(0,ddof=1)/n+B[:n].var(0,ddof=1)/n); dd[dd<1e-12]=1e-12
        ttd.append((int(n),float(np.abs((A[:n].mean(0)-B[:n].mean(0))/dd).max())))
RES['ttd']=ttd
json.dump({'masked_shuffle_blockinterleaved':RES},open(f'{OUT}/block_results.json','w'),indent=2)
log('  BLOCK-INTERLEAVED masked_shuffle: max|t|=%.2f  #>4.5=%d/%d  PASS=%s  driftfloor A=%.2f B=%.2f'%(
    at.max(),RES['n_over_4p5'],SAMPLES,RES['pass45'],RES['driftfloor_A1A2'],RES['driftfloor_B1B2']))
log('  TtD:',ttd)
try: scope.dis()
except: pass
log('==== BI_DONE ===='); print('BI_DONE')
