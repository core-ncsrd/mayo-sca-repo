# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
import os
#!/usr/bin/env python3
"""Drift test on masked TVLA + CPA on fresh capture."""
import numpy as np, json, os
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=f'{BASE}/paper_assets'
R={}
def welch(A,B):
    nA,nB=A.shape[0],B.shape[0]
    d=np.sqrt(A.var(0,ddof=1)/nA+B.var(0,ddof=1)/nB); d[d<1e-12]=1e-12
    return (A.mean(0)-B.mean(0))/d
def maxt(A,B): return float(np.abs(welch(A,B)).max()), int((np.abs(welch(A,B))>4.5).sum())

print('==== DRIFT TEST (masked TVLA: raw vs DC-removed vs linear-detrend) ====')
for tag,od in [('masked_naive','traces_masked'),('masked_shuffle','traces_masked2')]:
    try:
        A=np.load(f'{BASE}/{od}/traces_A.npy'); B=np.load(f'{BASE}/{od}/traces_B.npy')
        raw=maxt(A,B)
        # per-trace DC removal
        Ad=A-A.mean(1,keepdims=True); Bd=B-B.mean(1,keepdims=True)
        dc=maxt(Ad,Bd)
        # per-trace linear detrend
        n=A.shape[1]; xx=np.linspace(-1,1,n)
        def detr(M):
            b=(M*xx).sum(1)/ (xx@xx)
            return M-np.outer(b,xx)-M.mean(1,keepdims=True)
        Al=detr(A); Bl=detr(B); lin=maxt(Al,Bl)
        R[tag]=dict(raw_max_t=raw[0],raw_over=raw[1], dc_max_t=dc[0],dc_over=dc[1],
                    detrend_max_t=lin[0],detrend_over=lin[1], nA=int(A.shape[0]),nB=int(B.shape[0]))
        print(f'  {tag}: raw={raw[0]:.2f}({raw[1]})  DC-removed={dc[0]:.2f}({dc[1]})  detrend={lin[0]:.2f}({lin[1]})')
    except Exception as e:
        print('  ',tag,'ERR',e); R[tag]={'error':str(e)}

# ==== CPA on fresh capture ====
print('==== CPA (fresh traces_cpa) ====')
def mul_f(a,b):
    a=a.astype(np.uint16);b=b.astype(np.uint16)
    p=((a&1)*b)^((a&2)*b)^((a&4)*b)^((a&8)*b); top=p&0xF0
    return ((p^(top>>4)^(top>>3))&0x0F).astype(np.uint8)
HW=np.array([bin(i).count('1') for i in range(16)],float)
def hw(v): return HW[(v&0xF).astype(np.int64)]
try:
    T=np.load(f'{BASE}/traces_cpa/traces.npy'); X=np.load(f'{BASE}/traces_cpa/x.npy')
    n=min(T.shape[0],X.shape[0]); T=T[:n]; X=X[:n]; ns=T.shape[1]
    Okey=np.array([[(7*(r*8+j)+3)&0xF for j in range(8)] for r in range(78)],np.uint8)
    Tc=T-T.mean(0,keepdims=True); varT=(Tc**2).sum(0)
    # calibrate POI cadence over first 6 rows (48 products) using known key
    pidx=[];poim=[]
    for r in range(6):
        for j in range(8):
            pr=hw(mul_f(np.full(n,Okey[r,j],np.uint8),X[:,j])); pr=pr-pr.mean()
            corr=np.abs((Tc*pr[:,None]).sum(0)/np.sqrt((pr@pr)*varT+1e-12))
            pidx.append(r*8+j); poim.append(int(corr.argmax()))
    pidx=np.array(pidx);poim=np.array(poim)
    Am=np.vstack([np.ones_like(pidx),pidx]).T.astype(float)
    coef,_,_,_=np.linalg.lstsq(Am,poim.astype(float),rcond=None); base,stride=coef
    rms=float(np.sqrt(np.mean((Am@coef-poim)**2)))
    R['cadence']=dict(base=float(base),stride=float(stride),rms=rms)
    print(f'  cadence base={base:.1f} stride={stride:.2f} rms={rms:.1f}')
    def attack_row(r,win=45,partial=True):
        ranks=[];rec=[];pk=[]
        for j in range(8):
            c=int(round(base+stride*(r*8+j))); c0=max(0,c-win);c1=min(ns,c+win)
            seg=Tc[:,c0:c1]
            if partial:
                hx=hw(X[:,j]);hx=hx-hx.mean()
                beta=(seg*hx[:,None]).sum(0)/(hx@hx); seg=seg-np.outer(hx,beta)
            cand=np.zeros(16)
            for g in range(16):
                pr=hw(mul_f(np.full(n,g,np.uint8),X[:,j]));pr=pr-pr.mean()
                cand[g]=np.abs((seg*pr[:,None]).sum(0)/np.sqrt((pr@pr)*(seg**2).sum(0)+1e-12)).max()
            o=np.argsort(-cand); ranks.append(int(np.where(o==Okey[r,j])[0][0])+1); rec.append(int(o[0])); pk.append(float(cand[o[0]]))
        return ranks,rec,pk
    r0,rec0,pk0=attack_row(0)
    R['cpa_row0']=dict(n_traces=int(n),ranks=r0,recovered=rec0,true=[int(Okey[0,j]) for j in range(8)],
        exact_top1=int(sum(x==1 for x in r0)),mean_rank=float(np.mean(r0)),keyspace=int(np.prod(r0)),peak_corr=pk0)
    print(f'  row0 ranks={r0} top1={sum(x==1 for x in r0)}/8 mean={np.mean(r0):.2f} cand={int(np.prod(r0))}')
    # rows 0..5 (fit in 6000 samples)
    allr=[]
    for r in range(6):
        rr,_,_=attack_row(r); allr.append(rr)
    allr=np.array(allr)
    R['cpa_rows0_5']=dict(rows=6,mean_rank=float(allr.mean()),top1_frac=float((allr==1).mean()),
        top4_frac=float((allr<=4).mean()),per_row_ranks=allr.tolist(),
        per_row_candidates=[int(np.prod(allr[i])) for i in range(6)],
        log2_resid_6rows=float(np.log2(np.maximum(allr,1)).sum()),
        naive_log2_6rows=float(6*8*4))
    print('  rows0-5 meanrank=%.2f top1=%.2f top4=%.2f log2resid=%.1f (naive %d)'%(
        allr.mean(),(allr==1).mean(),(allr<=4).mean(),np.log2(np.maximum(allr,1)).sum(),6*8*4))
    # full-key extrapolation: mean per-row log2 residual * 78 rows
    per_row_log2=[float(np.log2(np.maximum(allr[i],1)).sum()) for i in range(6)]
    mean_row_log2=float(np.mean(per_row_log2))
    R['fullkey_extrap']=dict(mean_row_log2_resid=mean_row_log2,
        est_full_O_log2=mean_row_log2*78, naive_full_log2=float(78*8*4))
    print('  full-O extrap: ~2^%.1f (naive 2^%d)'%(mean_row_log2*78,78*8*4))
    # spectrum for figure (row0 nibble0)
    j=0;c=int(round(base+stride*0));c0=max(0,c-45);c1=min(ns,c+45);seg=Tc[:,c0:c1]
    hx=hw(X[:,0]);hx=hx-hx.mean();beta=(seg*hx[:,None]).sum(0)/(hx@hx);seg=seg-np.outer(hx,beta)
    spec=[float(np.abs((seg*(hw(mul_f(np.full(n,g,np.uint8),X[:,0]))-hw(mul_f(np.full(n,g,np.uint8),X[:,0])).mean())[:,None]).sum(0)/np.sqrt(((hw(mul_f(np.full(n,g,np.uint8),X[:,0]))-hw(mul_f(np.full(n,g,np.uint8),X[:,0])).mean())**2).sum()*(seg**2).sum(0)+1e-12)).max()) for g in range(16)]
    R['cpa_row0']['spectrum0']=spec
except Exception as e:
    import traceback;traceback.print_exc();R['cpa_row0']={'error':str(e)}

json.dump(R,open(f'{OUT}/analyze2_results.json','w'),indent=2)
print('WROTE analyze2_results.json')
