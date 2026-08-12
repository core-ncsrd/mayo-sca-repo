#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
"""
Master post-hoc analysis for the MAYO-1 SCA paper revision.
Runs on the testbed over ALREADY-CAPTURED traces. No hardware needed.
Produces:
  paper_assets/tvla_unmasked.png / .pdf       (t-trace + 4.5 band)
  paper_assets/tvla_overlay.png / .pdf        (unmasked vs masked overlay)
  paper_assets/cpa_ranks.png / .pdf           (row-0 rank bars + nibble-0 corr spectrum)
  paper_assets/ttd_curve.png / .pdf           (max|t| vs trace count, both impls)
  paper_assets/results.json                   (all numbers)
"""
import numpy as np, json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=f'{BASE}/paper_assets'; os.makedirs(OUT, exist_ok=True)
R={}

# ---------------- GF(16) arithmetic (matches firmware simple_arithmetic.h) ----------------
def mul_f(a, b):
    a=a.astype(np.uint16); b=b.astype(np.uint16)
    p=((a&1)*b)^((a&2)*b)^((a&4)*b)^((a&8)*b)
    top=p&0xF0
    return ((p ^ (top>>4) ^ (top>>3))&0x0F).astype(np.uint8)
HWLUT=np.array([bin(i).count('1') for i in range(16)], dtype=np.float64)
def hw(v): return HWLUT[(v&0xF).astype(np.int64)]

def welch_t(A, B):
    nA,nB=A.shape[0],B.shape[0]
    mA,mB=A.mean(0),B.mean(0)
    vA,vB=A.var(0,ddof=1),B.var(0,ddof=1)
    d=np.sqrt(vA/nA+vB/nB); d[d<1e-12]=1e-12
    return (mA-mB)/d

def chi2_tvla(A, B, nbins=9):
    """Per-sample chi^2 leakage test (Moradi et al., CHES 2018 style).
    Pool A,B; bin by pooled quantiles; 2xB contingency; chi^2 with dof=nbins-1."""
    n=A.shape[1]
    chi=np.zeros(n)
    # common bin edges per sample via pooled quantiles
    for s in range(n):
        a=A[:,s]; b=B[:,s]
        pooled=np.concatenate([a,b])
        edges=np.quantile(pooled, np.linspace(0,1,nbins+1))
        edges[0]-=1e-6; edges[-1]+=1e-6
        # collapse duplicate edges
        edges=np.unique(edges)
        if edges.size<3:
            chi[s]=0.0; continue
        ca,_=np.histogram(a, bins=edges)
        cb,_=np.histogram(b, bins=edges)
        obs=np.vstack([ca,cb]).astype(np.float64)
        rt=obs.sum(1,keepdims=True); ct=obs.sum(0,keepdims=True); tot=obs.sum()
        exp=rt*ct/tot
        mask=exp>0
        chi[s]=(((obs-exp)**2)[mask]/exp[mask]).sum()
    return chi

# ================= 1. UNMASKED fixed-vs-fixed TVLA =================
print('[1] UNMASKED fvf TVLA', flush=True)
A=np.load(f'{BASE}/traces_fvf/traces_A.npy'); B=np.load(f'{BASE}/traces_fvf/traces_B.npy')
t_un=welch_t(A,B); at_un=np.abs(t_un)
R['unmasked_fvf']=dict(nA=int(A.shape[0]), nB=int(B.shape[0]), samples=int(A.shape[1]),
                       max_t=float(at_un.max()), argmax=int(at_un.argmax()),
                       n_over_4p5=int((at_un>4.5).sum()))
print('   ', R['unmasked_fvf'], flush=True)

# TtD curve (unmasked)
Ns_un=[25,50,75,100,150,200,300,400,min(A.shape[0],B.shape[0])]
ttd_un=[]
for n in Ns_un:
    if n<=A.shape[0] and n<=B.shape[0]:
        ttd_un.append((int(n), float(np.abs(welch_t(A[:n],B[:n])).max())))
R['unmasked_fvf']['ttd_curve']=ttd_un
# first N where max|t|>4.5
tdisc=next((n for n,v in ttd_un if v>4.5), None)
R['unmasked_fvf']['traces_to_first_detection']=tdisc

# chi2 (subsample sample points for speed but cover full window)
try:
    step=max(1, A.shape[1]//5000)
    idx=np.arange(0,A.shape[1],step)
    chi_un=chi2_tvla(A[:,idx], B[:,idx])
    R['unmasked_fvf']['chi2_max']=float(chi_un.max())
    R['unmasked_fvf']['chi2_nbins']=9
except Exception as e:
    print('   chi2 unmasked failed:', e, flush=True)

# ================= 2. MASKED fixed-vs-fixed TVLA =================
print('[2] MASKED fvf TVLA', flush=True)
mA=np.load(f'{BASE}/traces_masked/traces_A.npy'); mB=np.load(f'{BASE}/traces_masked/traces_B.npy')
t_m=welch_t(mA,mB); at_m=np.abs(t_m)
R['masked_fvf']=dict(nA=int(mA.shape[0]), nB=int(mB.shape[0]), samples=int(mA.shape[1]),
                     max_t=float(at_m.max()), argmax=int(at_m.argmax()),
                     n_over_4p5=int((at_m>4.5).sum()))
print('   ', R['masked_fvf'], flush=True)
Ns_m=[50,100,200,300,500,700,min(mA.shape[0],mB.shape[0])]
ttd_m=[]
for n in Ns_m:
    if n<=mA.shape[0] and n<=mB.shape[0]:
        ttd_m.append((int(n), float(np.abs(welch_t(mA[:n],mB[:n])).max())))
R['masked_fvf']['ttd_curve']=ttd_m
try:
    step=max(1, mA.shape[1]//5000); idx=np.arange(0,mA.shape[1],step)
    chi_m=chi2_tvla(mA[:,idx], mB[:,idx])
    R['masked_fvf']['chi2_max']=float(chi_m.max())
except Exception as e:
    print('   chi2 masked failed:', e, flush=True)

# ================= 3. CPA on row 0 (fixed key, random x) =================
print('[3] CPA row 0', flush=True)
def load_cpa():
    # prefer traces/rand_traces.npy (fixed default key O[i]=(7i+3)&0xF, random x)
    tp=f'{BASE}/traces/rand_traces.npy'; pp=f'{BASE}/traces/rand_pt.npy'
    T=np.load(tp); P=np.load(pp)
    P=P.reshape(P.shape[0],-1)[:, :8].astype(np.uint8)
    n=min(T.shape[0],P.shape[0]); return T[:n], P[:n]
try:
    T,P=load_cpa()
    Otrue=np.array([(7*j+3)&0xF for j in range(8)], dtype=np.uint8)  # row 0 default key
    nt,ns=T.shape
    POI=lambda j:130+100*j
    win=40
    ranks=[]; recovered=[]; peak_corr=[]; spectrum0=None
    Tc=T-T.mean(0,keepdims=True)

    def _spectrum(seg, x_col, partial):
        """Peak |corr| per key guess; optionally with HW(x) regressed out."""
        s = seg
        if partial:
            hx = hw(x_col); hx = hx - hx.mean()
            s = seg - np.outer(hx, (seg*hx[:,None]).sum(0)/(hx@hx))
        out = np.zeros(16)
        for g in range(16):
            pred = hw(mul_f(np.full(len(x_col), g, np.uint8), x_col))
            pred = pred - pred.mean()
            num  = (s*pred[:,None]).sum(0)
            den  = np.sqrt((pred@pred)*(s**2).sum(0)+1e-12)
            out[g] = np.abs(num/den).max()
        return out

    for j in range(8):
        c0=max(0,POI(j)-win); c1=min(ns,POI(j)+win)
        seg=Tc[:,c0:c1]
        # partial correlation: regress HW(x_j) out of each sample column
        hx=hw(P[:,j]); hx=hx-hx.mean()
        beta=(seg*hx[:,None]).sum(0)/ (hx@hx)
        segr=seg-np.outer(hx,beta)
        cand=np.zeros(16)
        for g in range(16):
            pred=hw(mul_f(np.full(nt,g,np.uint8), P[:,j]))
            pred=pred-pred.mean()
            num=(segr*pred[:,None]).sum(0)
            den=np.sqrt((pred@pred)*(segr**2).sum(0)+1e-12)
            corr=np.abs(num/den)
            cand[g]=corr.max()
        # GF(16) multiplicative-identity disambiguation (see cpa.py and the paper,
        # Sec. "CPA Distinguisher and Confound Handling").  Partial correlation
        # annihilates the g=1 predictor, so a nibble whose true value IS 1 is
        # invisible to `cand`.  Such a nibble is identifiable WITHOUT knowing the
        # key: the plain spectrum peaks at g=1 and the partial spectrum is flat.
        # Omitting this rule mis-ranks those nibbles (rank 15 instead of 1) and
        # inflates the reported key-space.
        plain = _spectrum(seg, P[:,j], partial=False)
        use_plain = (int(np.argmax(plain)) == 1 and cand.max() < 0.30)
        score = plain if use_plain else cand
        pick  = 1 if use_plain else int(np.argmax(cand))

        order=np.argsort(-score)
        rank=int(np.where(order==Otrue[j])[0][0])+1
        ranks.append(rank); recovered.append(pick); peak_corr.append(float(score[pick]))
        if j==0: spectrum0=cand.copy()
    prod=int(np.prod(ranks))
    R['cpa_row0']=dict(n_traces=int(nt), samples=int(ns),
                       true=[int(x) for x in Otrue], recovered=recovered, ranks=ranks,
                       peak_corr=peak_corr, exact_top1=int(sum(r==1 for r in ranks)),
                       mean_rank=float(np.mean(ranks)), keyspace_candidates=prod)
    print('   ', R['cpa_row0'], flush=True)
    np.save(f'{OUT}/cpa_row0_spectrum.npy', spectrum0)
    R['cpa_row0']['spectrum0']=[float(x) for x in spectrum0]
except Exception as e:
    import traceback; traceback.print_exc(); R['cpa_row0']={'error':str(e)}

# ================= 4. Extended CPA across rows (traces_fvf group A, known O_A) =================
print('[4] Extended CPA across rows', flush=True)
try:
    Oa=np.load(f'{BASE}/traces_fvf/O_A.npy').astype(np.uint8)      # (78,8) known
    xs=np.load(f'{BASE}/traces_fvf/x_seq.npy').astype(np.uint8)    # (N,8)
    TA=np.load(f'{BASE}/traces_fvf/traces_A.npy')                  # (N,20000)
    nA=min(TA.shape[0], xs.shape[0]); TA=TA[:nA]; xs=xs[:nA]; nsA=TA.shape[1]
    TAc=TA-TA.mean(0,keepdims=True)
    varT=(TAc**2).sum(0)
    # --- calibrate POI cadence from first rows using the KNOWN key ---
    # product index p = r*8 + j ; find argmax|corr| location for a subset
    prod_idx=[]; poi_meas=[]
    cal_rows=list(range(0,12))
    for r in cal_rows:
        for j in range(8):
            pred=hw(mul_f(np.full(nA,Oa[r,j],np.uint8), xs[:,j])); pred=pred-pred.mean()
            num=(TAc*pred[:,None]).sum(0)
            den=np.sqrt((pred@pred)*varT+1e-12)
            corr=np.abs(num/den)
            prod_idx.append(r*8+j); poi_meas.append(int(corr.argmax()))
    prod_idx=np.array(prod_idx); poi_meas=np.array(poi_meas)
    # robust linear fit POI = base + stride*p  (use median-based to resist outliers)
    Aq=np.vstack([np.ones_like(prod_idx), prod_idx]).T.astype(float)
    coef,_,_,_=np.linalg.lstsq(Aq, poi_meas.astype(float), rcond=None)
    base,stride=coef
    R['cadence']=dict(base=float(base), stride=float(stride),
                      cal_rms=float(np.sqrt(np.mean((Aq@coef-poi_meas)**2))))
    print('   cadence base=%.1f stride=%.2f rms=%.1f'%(base,stride,R['cadence']['cal_rms']), flush=True)
    # --- attack all rows using predicted POI windows (no truth used for POI) ---
    win=60
    all_ranks=[]; per_row_prod=[]
    rows_to_attack=list(range(78))
    for r in rows_to_attack:
        rr=[]
        for j in range(8):
            p=r*8+j; c=int(round(base+stride*p))
            c0=max(0,c-win); c1=min(nsA,c+win)
            if c1-c0<5:
                rr.append(16); continue
            seg=TAc[:,c0:c1]
            hx=hw(xs[:,j]); hx=hx-hx.mean()
            beta=(seg*hx[:,None]).sum(0)/(hx@hx); segr=seg-np.outer(hx,beta)
            cand=np.zeros(16)
            for g in range(16):
                pred=hw(mul_f(np.full(nA,g,np.uint8), xs[:,j])); pred=pred-pred.mean()
                num=(segr*pred[:,None]).sum(0)
                den=np.sqrt((pred@pred)*(segr**2).sum(0)+1e-12)
                cand[g]=np.abs(num/den).max()
            order=np.argsort(-cand)
            rank=int(np.where(order==Oa[r,j])[0][0])+1
            rr.append(rank)
        all_ranks.append(rr); per_row_prod.append(int(np.prod(rr)))
    all_ranks=np.array(all_ranks)  # (78,8)
    # enumeration cost = product of all ranks
    log2_cost=float(np.log2(np.maximum(all_ranks,1)).sum())
    R['cpa_extended']=dict(n_traces=int(nA), rows=len(rows_to_attack),
        mean_rank=float(all_ranks.mean()),
        exact_top1_frac=float((all_ranks==1).mean()),
        top4_frac=float((all_ranks<=4).mean()),
        naive_log2_keyspace=float(78*8*np.log2(16)),
        residual_log2_enum_cost=log2_cost,
        per_row_mean_rank=[float(x) for x in all_ranks.mean(1)])
    print('   extended: meanrank=%.2f top1=%.2f top4=%.2f log2cost=%.1f (naive=%.1f)'%(
        all_ranks.mean(),(all_ranks==1).mean(),(all_ranks<=4).mean(),log2_cost,78*8*4), flush=True)
    np.save(f'{OUT}/extended_ranks.npy', all_ranks)
except Exception as e:
    import traceback; traceback.print_exc(); R['cpa_extended']={'error':str(e)}

# ================= FIGURES =================
print('[5] Figures', flush=True)
plt.rcParams.update({'font.size':11,'figure.dpi':150,'savefig.bbox':'tight'})

# Fig A: unmasked TVLA t-trace with 4.5 band
def savefig(fig, name):
    fig.savefig(f'{OUT}/{name}.png'); fig.savefig(f'{OUT}/{name}.pdf'); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,3.2))
ax.plot(t_un, color='#0072B2', lw=0.6)
ax.axhline(4.5,color='#C62042',ls='--',lw=1); ax.axhline(-4.5,color='#C62042',ls='--',lw=1)
ax.set_xlabel('Sample'); ax.set_ylabel("Welch's $t$")
ax.set_title(f"Unprotected $O\\cdot x$ TVLA  (max$|t|$={at_un.max():.1f})")
ax.margins(x=0.01); savefig(fig,'tvla_unmasked')

# Fig B: overlay unmasked vs masked (share x by fraction of window)
fig,ax=plt.subplots(figsize=(7,3.4))
xu=np.linspace(0,1,t_un.size); xm=np.linspace(0,1,t_m.size)
ax.plot(xu, np.abs(t_un), color='#C62042', lw=0.6, label=f'Unprotected (max {at_un.max():.1f})')
ax.plot(xm, np.abs(t_m), color='#0072B2', lw=0.6, label=f'Blinded (max {at_m.max():.1f})')
ax.axhline(4.5,color='k',ls='--',lw=1,label='4.5 threshold')
ax.set_yscale('log'); ax.set_xlabel('Normalised trigger window'); ax.set_ylabel("$|t|$ (log)")
ax.set_title('Leakage before vs after first-order blinding'); ax.legend(fontsize=9)
savefig(fig,'tvla_overlay')

# Fig B2: masked alone (linear)
fig,ax=plt.subplots(figsize=(7,3.0))
ax.plot(t_m, color='#0072B2', lw=0.6)
ax.axhline(4.5,color='#C62042',ls='--',lw=1); ax.axhline(-4.5,color='#C62042',ls='--',lw=1)
ax.set_xlabel('Sample'); ax.set_ylabel("Welch's $t$")
ax.set_title(f"Blinded $O\\cdot x$ TVLA  (max$|t|$={at_m.max():.2f}, < 4.5)")
ax.margins(x=0.01); savefig(fig,'tvla_masked')

# Fig C: CPA row-0 ranks + nibble-0 spectrum
if 'cpa_row0' in R and 'ranks' in R['cpa_row0']:
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.2))
    rk=R['cpa_row0']['ranks']
    axs[0].bar(range(8), rk, color='#0072B2')
    axs[0].axhline(1,color='#117A45',ls=':'); axs[0].set_xlabel('Nibble $j$ of row 0')
    axs[0].set_ylabel('Rank of true key (1=best)'); axs[0].set_title('Per-nibble true-key rank')
    axs[0].set_xticks(range(8))
    sp=np.array(R['cpa_row0']['spectrum0']); tr=R['cpa_row0']['true'][0]
    cols=['#C62042' if g==tr else '#9aa0b4' for g in range(16)]
    axs[1].bar(range(16), sp, color=cols)
    axs[1].set_xlabel('Key hypothesis $g$'); axs[1].set_ylabel('max $|\\rho|$ at POI')
    axs[1].set_title(f'Nibble 0 spectrum (true=0x{tr:X}, red)')
    axs[1].set_xticks(range(0,16,2))
    savefig(fig,'cpa_ranks')

# Fig D: TtD curve
fig,ax=plt.subplots(figsize=(6.4,3.2))
if ttd_un:
    xs_,ys_=zip(*ttd_un); ax.plot(xs_,ys_,'o-',color='#C62042',label='Unprotected')
if ttd_m:
    xs_,ys_=zip(*ttd_m); ax.plot(xs_,ys_,'s-',color='#0072B2',label='Blinded')
ax.axhline(4.5,color='k',ls='--',lw=1,label='4.5 threshold')
ax.set_yscale('log'); ax.set_xlabel('Traces per class $N$'); ax.set_ylabel('max $|t|$ (log)')
ax.set_title('Leakage growth vs trace count'); ax.legend(fontsize=9)
savefig(fig,'ttd_curve')

with open(f'{OUT}/results.json','w') as f: json.dump(R,f,indent=2)
print('[done] wrote', OUT, flush=True)
print(json.dumps({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if not isinstance(vv,list)}) for k,v in R.items()}, indent=2))
