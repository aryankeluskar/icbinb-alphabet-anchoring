import csv, os, sys, json
from collections import Counter, defaultdict
csv.field_size_limit(10**9)
D="/scratch/author/icbinb/data/proteingym/DMS_ProteinGym_substitutions/DMS_ProteinGym_substitutions"
ref="/scratch/author/icbinb/repos/ProteinGym/reference_files/DMS_substitutions.csv"
meta={r['DMS_id']:r for r in csv.DictReader(open(ref))}
per_assay={}; glob=Counter()
for fn in sorted(os.listdir(D)):
    if not fn.endswith('.csv'): continue
    aid=fn[:-4]; c=Counter()
    with open(os.path.join(D,fn)) as f:
        rd=csv.reader(f); hdr=next(rd); mi=hdr.index('mutant')
        for row in rd:
            if not row: continue
            k=row[mi].count(':')+1
            c[k]+=1; glob[k]+=1
    per_assay[aid]=c
json.dump({a:dict(c) for a,c in per_assay.items()}, open('/scratch/author/icbinb/data/k_distribution.json','w'), indent=1)

print("="*100); print("GLOBAL k DISTRIBUTION — ProteinGym v1.3 DMS substitutions (217 assays)"); print("="*100)
tot=sum(glob.values())
for k in sorted(glob): print("  k=%-3d %12d  (%5.2f%%)" % (k, glob[k], 100.0*glob[k]/tot))
print("  %-5s %12d" % ("TOTAL", tot))
multi=sum(v for k,v in glob.items() if k>=2)
print("\n  k=1     : %d (%.2f%%)" % (glob[1], 100.0*glob[1]/tot))
print("  k>=2    : %d (%.2f%%)   <-- MULTI-MUTANT" % (multi, 100.0*multi/tot))
na=sum(1 for a,c in per_assay.items() if any(k>=2 for k in c))
print("  assays with any k>=2: %d / %d" % (na, len(per_assay)))

print("\n"+"="*100); print("PER-ASSAY k>=2 BREAKDOWN (%d assays), sorted by n(k>=2) desc"%na); print("="*100)
print("%-46s %8s %8s %8s %8s %7s %7s %6s  %s"%("DMS_id","total","k=1","k=2","k=3","k=4","k=5+","maxk","Neff/L"))
rows=[(a,c) for a,c in per_assay.items() if any(k>=2 for k in c)]
rows.sort(key=lambda x:-sum(v for k,v in x[1].items() if k>=2))
for a,c in rows:
    t=sum(c.values()); m=meta.get(a,{})
    k5=sum(v for k,v in c.items() if k>=5)
    print("%-46s %8d %8d %8d %8d %7d %7d %6d  %s"%(a[:45],t,c.get(1,0),c.get(2,0),c.get(3,0),c.get(4,0),k5,max(c),m.get('MSA_Neff_L','?')[:7]))
