import csv, os, sys
import numpy as np
sys.path.insert(0,"/scratch/author/icbinb/experiments/H4-format-severity/code")
from run_h4 import parse_mutant
SC="/scratch/author/icbinb/data/proteingym/zero_shot_substitutions_scores"
res=[]
for fn in sorted(f for f in os.listdir(SC) if f.endswith(".csv")):
    recs=list(csv.DictReader(open(os.path.join(SC,fn))))
    ks=[];dm=[]
    for r in recs:
        p=parse_mutant(r["mutant"])
        if p is None: continue
        try: d=float(r["DMS_score"])
        except: continue
        ks.append(len(p)); dm.append(d)
    ks=np.array(ks,dtype=float); dm=np.array(dm,dtype=float)
    if len(ks)<300 or ks.max()<2: continue
    per={}
    for kv in np.unique(ks):
        idx=np.where(ks==kv)[0]
        if len(idx)>=50: per[kv]=dm[idx].std()
    if len(per)>=2:
        kk=np.array(sorted(per)); vv=np.array([per[k] for k in kk])
        if vv.std()>1e-12 and kk.std()>1e-12:
            res.append(float(np.corrcoef(kk,vv)[0,1]))
res=np.array(res)
print("assays analysed:", len(res))
print("corr(k, within-k DMS std): median %+.3f  mean %+.3f" % (np.median(res),res.mean()))
print("fraction NEGATIVE (dynamic range shrinks as k grows): %.2f" % (res<0).mean())
