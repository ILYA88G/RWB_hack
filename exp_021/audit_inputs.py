"""Read-only inventory of inputs. Inspect checkpoint metadata without unpickling."""
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import improve_network as net
else:
    import improve_network as net


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def adapter_metadata(path):
    with zipfile.ZipFile(path) as z:
        member, = [n for n in z.namelist() if n.endswith("/data.pkl")]
        ops = [(op.name,arg) for op,arg,_ in pickletools.genops(z.read(member))
               if op.name not in {"BINPUT","LONG_BINPUT","MEMOIZE"}]
    def literal(key):
        i, = [i for i,(_,value) in enumerate(ops) if value == key]
        code,arg = ops[i+1]
        if code not in ("BINUNICODE","BININT","BININT2"):
            raise ValueError(f"Unsupported checkpoint metadata {key}")
        return arg
    return literal("base_checkpoint_sha256"), literal("train_origin")


def main():
    p = net.parser("input_manifest.csv")
    args = p.parse_args()
    if args.output.exists() and args.output.resolve() != net.EXP/"input_manifest.csv":
        raise ValueError("Refusing to overwrite unrelated file")
    root = net.ROOT
    model = root/"model_019b/refitted"
    files = {args.train,args.test,args.baseline, root/"requirements.txt",root/"docs/README.md"}
    for folder in ("submissions","reports","results_exp009","model_019b","src"):
        for f in (root/folder).rglob("*"):
            if f.is_file() and f.suffix in {".py",".csv",".npz",".npy",".cbm",".pt",".json",".txt"}:
                files.add(f)
    files.update(root.glob("*.py"))
    entries = []
    for f in sorted({f.resolve() for f in files}):
        name = f.relative_to(root) if f.is_relative_to(root) else f
        entries.append(dict(path=str(name),bytes=f.stat().st_size,sha256=digest(f)))
    pd.DataFrame(entries).to_csv(args.output,index=False)
    claimed, t = adapter_metadata(model/"model020a_adapter.pt")
    if claimed != digest(model/"model019a.pt"):
        raise ValueError("Adapter references a different TCN; refuse to use")
    series, panel = net.load_train(args.train)
    origin = series.index[-1]
    test = net.validate_grid(pd.read_parquet(args.test),panel.columns,origin)
    frames = {}
    audit = {}
    for name in ("network_008b","model_014a","model_015a","model_019b","model_020a"):
        f = pd.read_csv(root/f"submissions/submission_{name}.csv",dtype={"id":str})
        values = net.aligned(test,f)
        frames[name] = values
        total = np.bincount(test.h,weights=values)
        if name != "network_008b":
            np.testing.assert_allclose(total,audit["network_008b"]["totals"],rtol=1e-12)
        audit[name] = dict(totals=total.tolist(),minimum=values.min(),maximum=values.max(),
                           zero=int((values==0).sum()), id_order_matches=test.id.tolist()==f.id.tolist())
    for a,b in [("model_015a","model_019b"),("model_019b","model_020a")]:
        delta = np.abs(frames[a]-frames[b])
        print(a,b,"mean |change|",delta.mean(),"max",delta.max())
    print("SHA match",claimed,"adapter origin",t)
    print("final prefix: routes with mean336 <100:",int((panel.iloc[-336:].mean()<100).sum()))
    print("target zeros, <=50, max:",int((panel==0).sum().sum()),int((panel<=50).sum().sum()),panel.max().max())
    print("test time",test.timestamp.min(),test.timestamp.max(),"routes",panel.columns.min(),panel.columns.max())
    print("q0",net.q0_at(panel,origin),"base q",net.budgets(audit["model_020a"]["totals"],net.q0_at(panel,origin)))
    for o in [origin-pd.Timedelta(days=2),origin-pd.Timedelta(days=1),origin-8*net.STEP,origin]:
        _,width = net.last_halfhour(panel,o)
        print(o,"positive / inverted intervals; gap quantiles",int((width>0).sum()),
              int((width<0).sum()),np.quantile(np.maximum(-width,0),[0,.5,.9,1]))
    import sys
    sys.path.insert(0,str(root))
    import train_network_calibration_008b as old
    data,features = old.build_network(series.reset_index())
    forecast = old.predict_network(data,features,origin,"lgbm28")*1e6
    base = net.aligned(test,pd.read_csv(root/"submissions/submission_lgbm_006a.csv",dtype={"id":str}))
    base_sum = np.bincount(test.h,weights=base)
    new = net.proposed_totals(base_sum,forecast,"per_step",.8,.05)
    diff = np.max(np.abs(base*(new/base_sum)[test.h]-frames["network_008b"]))
    print("008b reproduction max absolute error",diff)
    if diff>1e-4:
        raise ValueError("Could not reproduce submitted 008b")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()

