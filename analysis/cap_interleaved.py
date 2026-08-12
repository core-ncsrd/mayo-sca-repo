# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
import os
#!/usr/bin/env python3
"""Drift-free INTERLEAVED fixed-vs-fixed TVLA for masked_shuffle (and a matched
unmasked control), reloading the secret O every trace by a per-trace coin so
slow drift affects both classes equally. Gold-standard TVLA acquisition."""
import chipwhisperer as cw
import numpy as np, os, time, json
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FW=f'{BASE}/firmware'; OUT=f'{BASE}/paper_assets'
LOG=open(f'{OUT}/interleave.log','a',buffering=1)
def log(*a): m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
PARAM_o,PARAM_v=8,78
HEX={'unmasked':f'{FW}/mayo-linearmap-CW308_STM32F3.hex',
     'masked_shuffle':f'{FW}/mayo-lm-masked2-CW308_STM32F3.hex'}
scope=cw.scope(); scope.default_setup(); target=cw.target(scope)
def reset(): scope.io.nrst='low'; time.sleep(0.1); scope.io.nrst='high'; time.sleep(0.4)
def flash(n): cw.program_target(scope,cw.programmers.STM32FProgrammer,HEX[n]); reset()
def ss(c,b): target.write(c+bytes(b).hex()+'\n')
def rd(ms=80):
    t0=time.time();buf=''
    while(time.time()-t0)<ms/1000:
        buf+=target.read()
        if '\n' in buf: break
        time.sleep(0.002)
    return buf
def loadO(O):
    for r in range(PARAM_v): ss('k',bytes([r])+bytes(O[r])); time.sleep(0.0012)
    rd(60)
def cap(x,S):
    target.flush(); scope.adc.samples=S; scope.arm(); ss('p',x)
    return None if scope.capture(poll_done=True) else scope.get_last_trace()

def run_interleaved(fw, N, S, seed_coin=7):
    scope.clock.adc_src='clkgen_x1'; scope.adc.timeout=12; scope.gain.db=45
    flash(fw)
    rng=np.random.default_rng(0xDEADBEEF)
    O_A=rng.integers(0,16,(PARAM_v,PARAM_o)).astype(np.uint8)
    O_B=rng.integers(0,16,(PARAM_v,PARAM_o)).astype(np.uint8)
    xr=np.random.default_rng(42); xseq=xr.integers(0,16,(N,PARAM_o)).astype(np.uint8)
    coin=np.random.default_rng(seed_coin).integers(0,2,N)
    A=[];B=[]; cur=None
    for i in range(N):
        want='A' if coin[i]==0 else 'B'
        if want!=cur:
            loadO(O_A if want=='A' else O_B); cur=want
        t=cap(bytes(xseq[i]),S)
        if t is not None: (A if want=='A' else B).append(t)
        if i%200==0: log(f'   {fw} {i}/{N} A={len(A)} B={len(B)}')
    A=np.array(A,np.float32); B=np.array(B,np.float32)
    nA,nB=A.shape[0],B.shape[0]
    d=np.sqrt(A.var(0,ddof=1)/nA+B.var(0,ddof=1)/nB); d[d<1e-12]=1e-12
    t=(A.mean(0)-B.mean(0))/d; at=np.abs(t)
    return dict(fw=fw,nA=int(nA),nB=int(nB),samples=int(S),max_t=float(at.max()),
                n_over=int((at>4.5).sum()),argmax=int(at.argmax()),pass45=bool(at.max()<4.5)), t, A, B

RES={}
log('==== INTERLEAVED masked_shuffle ====')
r,t,A,B=run_interleaved('masked_shuffle', N=1600, S=24000)
np.save(f'{OUT}/masked_shuffle_interleaved_t.npy',t)
od=f'{BASE}/traces_masked2_il'; os.makedirs(od,exist_ok=True)
np.save(f'{od}/traces_A.npy',A); np.save(f'{od}/traces_B.npy',B)
RES['masked_shuffle_interleaved']=r
log('   masked_shuffle interleaved: max|t|=%.2f  #>4.5=%d  PASS=%s'%(r['max_t'],r['n_over'],r['pass45']))
# TtD on interleaved (subsample increasing N by splitting)
nmin=min(A.shape[0],B.shape[0]); ttd=[]
for n in [50,100,200,300,500,700,nmin]:
    if n<=nmin:
        d=np.sqrt(A[:n].var(0,ddof=1)/n+B[:n].var(0,ddof=1)/n); d[d<1e-12]=1e-12
        ttd.append((int(n),float(np.abs((A[:n].mean(0)-B[:n].mean(0))/d).max())))
RES['masked_shuffle_interleaved']['ttd']=ttd
log('   TtD:',ttd)
json.dump(RES,open(f'{OUT}/interleave_results.json','w'),indent=2)
try: scope.dis()
except: pass
log('==== IL_DONE ===='); print('IL_DONE')
