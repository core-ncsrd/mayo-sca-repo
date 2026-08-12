# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NCSR "Demokritos" and AGH University of Krakow
import os
#!/usr/bin/env python3
"""Final publication figures using drift-free (block-interleaved) masked data."""
import numpy as np, json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
# Repository root: override with the MAYO_SCA_ROOT environment variable.
BASE = os.environ.get('MAYO_SCA_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=f'{BASE}/paper_assets'
plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.bbox':'tight','savefig.dpi':300})
def save(fig,n): fig.savefig(f'{OUT}/{n}.png'); fig.savefig(f'{OUT}/{n}.pdf'); plt.close(fig)
UN='#C62042'; MK='#0072B2'; GR='#117A45'

t_un=np.load(f'{BASE}/traces_fvf/t_stat_fvf.npy')          # unmasked fvf, 20000
t_mk=np.load(f'{OUT}/masked_shuffle_bi_t.npy')             # masked, drift-free, 24000
au,am=np.abs(t_un),np.abs(t_mk)
R=json.load(open(f'{OUT}/block_results.json'))['masked_shuffle_blockinterleaved']

# ---- Fig 1: two-panel TVLA (unmasked | masked) same 4.5 band ----
fig,ax=plt.subplots(1,2,figsize=(9,3.1),sharey=False)
ax[0].plot(t_un,color=UN,lw=0.5); ax[0].axhline(4.5,color='k',ls='--',lw=0.9); ax[0].axhline(-4.5,color='k',ls='--',lw=0.9)
ax[0].set_title(f'Unprotected  (max$|t|$={au.max():.0f})'); ax[0].set_xlabel('Sample'); ax[0].set_ylabel("Welch's $t$"); ax[0].margins(x=0.01)
ax[1].plot(t_mk,color=MK,lw=0.5); ax[1].axhline(4.5,color='k',ls='--',lw=0.9); ax[1].axhline(-4.5,color='k',ls='--',lw=0.9)
ax[1].set_title(f'Masked+shuffled  (max$|t|$={am.max():.2f} < 4.5)'); ax[1].set_xlabel('Sample'); ax[1].set_ylim(-6,6); ax[1].margins(x=0.01)
save(fig,'fig_tvla_pair')

# ---- Fig 2: overlay |t| log ----
fig,ax=plt.subplots(figsize=(7,3.2))
ax.plot(np.linspace(0,1,au.size),au,color=UN,lw=0.5,label=f'Unprotected (peak {au.max():.0f})')
ax.plot(np.linspace(0,1,am.size),am,color=MK,lw=0.6,label=f'Masked+shuffled (peak {am.max():.2f})')
ax.axhline(4.5,color='k',ls='--',lw=1,label='4.5 threshold')
ax.set_yscale('log'); ax.set_xlabel('Normalised trigger window'); ax.set_ylabel('$|t|$ (log scale)')
ax.set_title('Leakage before vs after the countermeasure'); ax.legend(fontsize=8,loc='upper right')
save(fig,'fig_tvla_overlay')

# ---- Fig 3: TtD curve (max|t| vs N), unmasked grows, masked flat ----
# unmasked ttd from results.json (analyze_all)
try: ttd_un=json.load(open(f'{OUT}/results.json'))['unmasked_fvf']['ttd_curve']
except Exception: ttd_un=[]
ttd_mk=R['ttd']
fig,ax=plt.subplots(figsize=(6.2,3.2))
if ttd_un:
    xu,yu=zip(*ttd_un); ax.plot(xu,yu,'o-',color=UN,label='Unprotected')
xm,ym=zip(*ttd_mk); ax.plot(xm,ym,'s-',color=MK,label='Masked+shuffled')
ax.axhline(4.5,color='k',ls='--',lw=1,label='4.5 threshold')
ax.set_yscale('log'); ax.set_xlabel('Traces per class $N$'); ax.set_ylabel('max $|t|$ (log)')
ax.set_title('Leakage growth vs trace count'); ax.legend(fontsize=8)
save(fig,'fig_ttd')

# chi2 for masked (drift-free) — quick
A=np.load(f'{BASE}/traces_masked2_bi/traces_A.npy'); B=np.load(f'{BASE}/traces_masked2_bi/traces_B.npy')
def chi2(A,B,nb=9,step=6):
    idx=np.arange(0,A.shape[1],step); chi=[]
    for s in idx:
        a=A[:,s];b=B[:,s]; e=np.quantile(np.concatenate([a,b]),np.linspace(0,1,nb+1))
        e=np.unique(e);
        if e.size<3: chi.append(0);continue
        e[0]-=1e-6;e[-1]+=1e-6
        ca,_=np.histogram(a,e);cb,_=np.histogram(b,e); ob=np.vstack([ca,cb]).astype(float)
        rt=ob.sum(1,keepdims=True);ct=ob.sum(0,keepdims=True);tot=ob.sum();ex=rt*ct/tot;m=ex>0
        chi.append((((ob-ex)**2)[m]/ex[m]).sum())
    return float(np.max(chi))
chi_mk=chi2(A,B)
summary=dict(unmasked_max_t=float(au.max()), masked_bi_max_t=float(am.max()),
             masked_bi_over=int((am>4.5).sum()), masked_chi2_max=chi_mk,
             driftfloor_A=R['driftfloor_A1A2'], driftfloor_B=R['driftfloor_B1B2'],
             reduction_factor=float(au.max()/am.max()))
json.dump(summary,open(f'{OUT}/final_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
print('FIGS: fig_tvla_pair fig_tvla_overlay fig_ttd  (+ cpa_ranks.png in traces/)')
