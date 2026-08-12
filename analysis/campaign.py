#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""Hardware campaign: cycle counts + masked2 TVLA + unmasked CPA capture.
Writes incremental JSON + log; prints DONE marker at end."""
import chipwhisperer as cw
import numpy as np, os, time, json, sys

import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FW=f'{BASE}/firmware'
OUT=f'{BASE}/paper_assets'; os.makedirs(OUT,exist_ok=True)
LOG=open(f'{OUT}/campaign.log','a',buffering=1)
def log(*a):
    m=' '.join(str(x) for x in a); print(m,flush=True); LOG.write(m+'\n')
RES={}
def dump():
    with open(f'{OUT}/campaign_results.json','w') as f: json.dump(RES,f,indent=2)

PARAM_o,PARAM_v=8,78
HEX={'unmasked':f'{FW}/mayo-linearmap-CW308_STM32F3.hex',
     'masked_naive':f'{FW}/mayo-lm-masked-CW308_STM32F3.hex',
     'masked_shuffle':f'{FW}/mayo-lm-masked2-CW308_STM32F3.hex'}

scope=cw.scope(); scope.default_setup(); target=cw.target(scope)
def reset():
    scope.io.nrst='low'; time.sleep(0.1); scope.io.nrst='high'; time.sleep(0.4)
def flash(name):
    log(f'[flash] {name}')
    cw.program_target(scope, cw.programmers.STM32FProgrammer, HEX[name]); reset()
def ss_send(c,b): target.write(c+bytes(b).hex()+'\n')
def ss_read(ms=200):
    t0=time.time(); buf=''
    while (time.time()-t0)<ms/1000:
        buf+=target.read()
        if '\n' in buf: break
        time.sleep(0.002)
    return buf.strip()
def load_O(O):
    for r in range(PARAM_v):
        ss_send('k', bytes([r])+bytes(O[r])); time.sleep(0.0015)
    ss_read(80)
def cap(x, samples):
    target.flush(); scope.adc.samples=samples; scope.arm(); ss_send('p', x)
    ret=scope.capture(poll_done=True)
    return None if ret else scope.get_last_trace()

# =============== PHASE 1: cycle counts (clkgen_x1 => 1 sample = 1 cycle) ===============
log('==== PHASE 1: cycle counts ====')
scope.clock.adc_src='clkgen_x1'; scope.adc.samples=24400; scope.adc.timeout=6; scope.gain.db=25
cyc={}
xtest=bytes([1,2,3,4,5,6,7,8])
for name in ['unmasked','masked_naive','masked_shuffle']:
    try:
        flash(name); time.sleep(0.2)
        counts=[]
        for _ in range(12):
            target.flush(); scope.arm(); ss_send('p', xtest)
            scope.capture(poll_done=True)
            counts.append(int(scope.adc.trig_count))
            time.sleep(0.01)
        counts=[c for c in counts if c>0]
        med=int(np.median(counts)) if counts else -1
        cyc[name]=dict(median_cycles=med, min=int(min(counts)) if counts else -1,
                       max=int(max(counts)) if counts else -1, n=len(counts))
        log(f'  {name}: median {med} cycles  (all={counts})')
    except Exception as e:
        log(f'  {name} cycle-count ERROR: {e}'); cyc[name]={'error':str(e)}
RES['cycles']=cyc
if 'median_cycles' in cyc.get('unmasked',{}) and cyc['unmasked']['median_cycles']>0:
    u=cyc['unmasked']['median_cycles']
    for k in ['masked_naive','masked_shuffle']:
        if cyc.get(k,{}).get('median_cycles',-1)>0:
            RES.setdefault('overhead_factor',{})[k]=round(cyc[k]['median_cycles']/u,3)
    log('  overhead factors:', RES.get('overhead_factor'))
dump()

# =============== PHASE 2: masked_shuffle fixed-vs-fixed TVLA ===============
log('==== PHASE 2: masked_shuffle fvf TVLA ====')
try:
    N=800; SAMPLES=24000
    scope.clock.adc_src='clkgen_x1'; scope.adc.timeout=12; scope.gain.db=45
    flash('masked_shuffle')
    rng=np.random.default_rng(0xDEADBEEF)
    O_A=rng.integers(0,16,size=(PARAM_v,PARAM_o)).astype(np.uint8)
    O_B=rng.integers(0,16,size=(PARAM_v,PARAM_o)).astype(np.uint8)
    xr=np.random.default_rng(42); x_seq=xr.integers(0,16,size=(N,PARAM_o)).astype(np.uint8)
    od=f'{BASE}/traces_masked2'; os.makedirs(od,exist_ok=True)
    np.save(f'{od}/O_A.npy',O_A); np.save(f'{od}/O_B.npy',O_B); np.save(f'{od}/x_seq.npy',x_seq)
    def grab(O,tag):
        load_O(O); time.sleep(0.05); tr=[]
        for i in range(N):
            t=cap(bytes(x_seq[i]),SAMPLES)
            if t is not None: tr.append(t)
            if i%200==0: log(f'    {tag} {i}/{N} ok={len(tr)}')
        return np.array(tr,dtype=np.float32)
    A=grab(O_A,'A'); B=grab(O_B,'B')
    np.save(f'{od}/traces_A.npy',A); np.save(f'{od}/traces_B.npy',B)
    nA,nB=A.shape[0],B.shape[0]
    mA,mB=A.mean(0),B.mean(0); vA,vB=A.var(0,ddof=1),B.var(0,ddof=1)
    d=np.sqrt(vA/nA+vB/nB); d[d<1e-12]=1e-12; t=(mA-mB)/d
    np.save(f'{od}/t_stat_fvf.npy',t); at=np.abs(t)
    RES['masked_shuffle_tvla']=dict(nA=int(nA),nB=int(nB),samples=int(SAMPLES),
        max_t=float(at.max()),n_over_4p5=int((at>4.5).sum()),argmax=int(at.argmax()),
        passes_4p5_gate=bool(at.max()<4.5))
    log('  masked_shuffle TVLA: max|t|=%.2f  #>4.5=%d/%d  PASS=%s'%(
        at.max(),(at>4.5).sum(),SAMPLES,at.max()<4.5))
except Exception as e:
    import traceback; traceback.print_exc(file=LOG); log('  PHASE2 ERROR',e); RES['masked_shuffle_tvla']={'error':str(e)}
dump()

# =============== PHASE 3: unmasked CPA capture (fixed default key, random x) ===============
log('==== PHASE 3: unmasked CPA capture ====')
try:
    Ncpa=2500; SAMP=6000  # 6000 samples covers ~ first 6 rows at ADCx4 cadence
    scope.clock.adc_src='clkgen_x4'; scope.adc.timeout=6; scope.gain.db=25
    flash('unmasked')  # holds default key O[i]=(7i+3)&0xF
    xr=np.random.default_rng(1234); X=xr.integers(0,16,size=(Ncpa,PARAM_o)).astype(np.uint8)
    od=f'{BASE}/traces_cpa'; os.makedirs(od,exist_ok=True); np.save(f'{od}/x.npy',X)
    tr=[]
    for i in range(Ncpa):
        t=cap(bytes(X[i]),SAMP)
        if t is not None: tr.append(t)
        if i%500==0: log(f'    cpa {i}/{Ncpa} ok={len(tr)}')
    T=np.array(tr,dtype=np.float32); np.save(f'{od}/traces.npy',T)
    RES['cpa_capture']=dict(n=int(T.shape[0]),samples=int(SAMP),adc='x4')
    log('  CPA capture done:', T.shape)
except Exception as e:
    import traceback; traceback.print_exc(file=LOG); log('  PHASE3 ERROR',e); RES['cpa_capture']={'error':str(e)}
dump()

try: scope.dis()
except: pass
log('==== DONE ===='); dump()
print('CAMPAIGN_DONE')
