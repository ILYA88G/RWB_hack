"""Compare route-level proxy OOF, choose on 3/4, check 5 and nested LOFO."""
from itertools import product
import json

import numpy as np
import pandas as pd

if __package__:
    from . import improve_network as net
else:
    import improve_network as net


def metric(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or not np.isfinite([y,p]).all() or y.sum() <= 0:
        raise ValueError("Invalid metric inputs")
    w = np.abs(y-p).sum()/y.sum()
    signed = p.sum()/y.sum()-1
    return dict(wape=w, signed_bias=signed, abs_bias=abs(signed), score=w+abs(signed))


def load_folds(panel, reports):
    ys, base6, base8, origins = [], [], [], []
    for fold in range(1,6):
        origin = panel.index[-1] - (pd.Timedelta(days=5-fold) if fold < 5 else 8*net.STEP)
        with np.load(reports/f"exp_005/ensemble_005b_fold_{fold}.npz") as a:
            assert pd.Timestamp(a["origin"].item()) == origin
            truth = panel.loc[origin+net.STEP:origin+8*net.STEP].to_numpy().T
            np.testing.assert_array_equal(truth, a["y_true"])
            old = a["lgbm_pred"].astype(float)
        if fold < 3:
            continue
        with np.load(reports/f"exp_006/oof_fold_{fold}.npz") as b:
            assert pd.Timestamp(b["origin"].item()) == origin
            new = b["new_pred"].astype(float)
            assert new.shape == old.shape == truth.shape == (1000,8)
        base = np.maximum(.5*old+.5*new,0)
        old_total = base.sum(axis=0)
        cached = 1e6*np.load(reports/f"exp_010/network_fold{fold}.npy")
        total = net.proposed_totals(old_total, cached, "per_step", .8, .05)
        base6.append(base)
        base8.append(base*(total/old_total))
        ys.append(truth)
        origins.append(origin)
    y, b6, b8 = map(np.stack, (ys,base6,base8))
    np.testing.assert_allclose(metric(y,b6)["score"], .3353359529200619, atol=1e-10, rtol=0)
    np.testing.assert_allclose(metric(y,b8)["score"], .33025948686895884, atol=1e-10, rtol=0)
    return origins, y, b6, b8


def choose(ys, base, predictions, indices):
    # Rank first by number of score wins, then worst fold change, then pooled score.
    def rank(key):
        p = predictions[key]
        changes = [metric(ys[i],p[i])["score"]-metric(ys[i],base[i])["score"] for i in indices]
        return (sum(d >= 0 for d in changes), max(changes),
                metric(ys[indices],p[indices])["score"], key[2], key[3], key)
    return min(predictions, key=rank)


def main():
    parser = net.parser("experiments.csv")
    parser.add_argument("--reports", type=net.Path, default=net.ROOT/"reports")
    args = parser.parse_args()
    if args.output.exists() and args.output.resolve() != net.EXP/"experiments.csv":
        raise ValueError("Refusing to overwrite unrelated file")
    series, panel = net.load_train(args.train)
    test = net.validate_grid(pd.read_parquet(args.test), panel.columns, series.index[-1])
    net.aligned(test,pd.read_csv(args.baseline,dtype={"id":str}))
    origins, y, base6, base = load_folds(panel,args.reports)
    models = [net.fit_predict(series,o) for o in origins]
    q0 = [net.q0_at(panel,o) for o in origins]
    rows, predictions = [], {}
    def record(kind, key, pred, ids, fold, group="all", mask=None):
        truth, guess = y[ids], pred[ids]
        if mask is not None:
            truth, guess = truth[:,mask], guess[:,mask]
        rows.append(dict(kind=kind, model=key[0], mode=key[1], alpha=key[2], cap=key[3],
                         seed=42, fold=fold, origin="|".join(str(origins[i]) for i in ids),
                         group=group, n=truth.size, **metric(truth,guess)))

    def full(kind,key,pred):
        for i in range(3):
            record(kind,key,pred,[i],i+3)
        record(kind,key,pred,[0,1,2],"pooled")
        record(kind,key,pred,[0,1],"dev")
    full("reference",("006a","",0,0),base6)
    full("reference",("008b","",0,0),base)
    grid = product(net.MODELS, ("global","per_step","smoothed"), (.15,.3,.5), (.02,.03,.05,.07))
    for key in grid:
        model, mode, alpha, cap = key
        corrected = []
        for i,b in enumerate(base):
            old = b.sum(axis=0)
            requested = net.proposed_totals(old,models[i][model],mode,alpha,cap)
            total,_ = net.safe_totals(old,requested,q0[i])
            corrected.append(b*(total/old))
        predictions[key] = np.stack(corrected)
        full("grid",key,predictions[key])
    key = choose(y,base,predictions,[0,1])
    selected = predictions[key]
    full("forward",key,selected)
    shape_key = (key[0], "shape", key[2], key[3])
    def shape_correct(b, forecast, initial):
        old = b.sum(axis=0)
        new, fallback = net.safe_totals(old, net.proposed_totals(old,forecast,*shape_key[1:]),initial)
        np.testing.assert_allclose(old.sum(),new.sum(),rtol=1e-12)
        return b*(new/old)
    shape_pred = np.stack([shape_correct(b, models[i][key[0]], q0[i]) for i,b in enumerate(base)])
    full("shape_ablation",shape_key,shape_pred)
    # Independent route matrices on two previously unused dates. 006a has no
    # caches here. Apply the archived 008b formula to the cached 005a shares.
    import sys
    sys.path.insert(0,str(net.ROOT))
    import train_network_calibration_008b as old_net
    # build_network groups by timestamp, so one already aggregated row per tick is sufficient.
    def legacy_prediction(o):
        legacy, cols = old_net.build_network(series.loc[:o].reset_index())
        return old_net.predict_network(legacy, cols, o, "lgbm28")
    earlier_gain = []
    for fold in (1,2):
        with np.load(args.reports/f"exp_005/ensemble_005b_fold_{fold}.npz") as a:
            o = pd.Timestamp(a["origin"].item())
            truth = a["y_true"].astype(float)
            proxy = a["lgbm_pred"].astype(float).clip(0)
        network = legacy_prediction(o)
        proxy,_ = old_net.corrected(proxy,network,.8,"per_step")
        early_forecast = net.fit_predict(series,o,[key[0]])[key[0]]
        adjusted = shape_correct(proxy,early_forecast,net.q0_at(panel,o))
        earlier_gain.append(metric(truth,proxy)["score"]-metric(truth,adjusted)["score"])
        for label,k,pr in [("005a_proxy",("005a+lgbm28","",0,0),proxy),
                           ("shape_external",shape_key,adjusted)]:
            rows.append(dict(kind=label,model=k[0],mode=k[1],alpha=k[2],cap=k[3],seed=42,
                             fold=fold,origin=str(o),group="all",n=truth.size,**metric(truth,pr)))
    cross = np.empty_like(base)
    for i in range(3):
        local = choose(y,base,predictions,[j for j in range(3) if j!=i])
        cross[i] = predictions[local][i]
        record("lofo",local,cross,[i],i+3)
    record("lofo",("mixed","",0,0),cross,[0,1,2],"pooled")

    for i,origin in enumerate(origins):
        volumes = panel.loc[:origin].iloc[-336:].mean()
        quartiles = pd.qcut(volumes.rank(method="first"),4,labels=False).to_numpy()
        for label,k,pred in [("reference",("008b","",0,0),base), ("forward",key,selected),
                             ("shape_ablation",shape_key,shape_pred)]:
            for q in range(4):
                record(label,k,pred,[i],i+3,f"volume_Q{q+1}",quartiles==q)
            for h in range(8):
                rows.append(dict(kind=label,model=k[0],mode=k[1],alpha=k[2],cap=k[3],
                                 seed=42,fold=i+3,origin=str(origin),group=f"h{h+1}",n=1000,
                                 **metric(y[i,:,h],pred[i,:,h])))
    # Prefer bias-neutral shape only with pooled gain and two external wins.
    if min(earlier_gain)>0 and metric(y,shape_pred)["score"] < metric(y,base)["score"]:
        key = shape_key
    rule = dict(zip(("model","mode","alpha","cap"),key))
    # Frozen-rule stress test, separate from rule selection. Rebuild a cheap
    # causal route proxy on six earlier Saturdays at the exact competition slot.
    batches = {"saturday_proxy":[], "saturday_021":[]}
    truths = []
    for weeks in range(6,0,-1):
        o = series.index[-1] - pd.Timedelta(weeks=weeks)
        t = panel.index.get_loc(o)
        history = panel.iloc[:t+1].to_numpy(dtype=float)
        route_profile = np.mean([history[t-d*48+1:t-d*48+9] for d in (1,2,3,7)],axis=0).T
        network = legacy_prediction(o)
        b,_ = old_net.corrected(route_profile,network,.8,"per_step")
        f = net.fit_predict(series,o,[key[0]])[key[0]]
        old = b.sum(axis=0)
        totals,_ = net.safe_totals(old,net.proposed_totals(old,f,*key[1:]),net.q0_at(panel,o))
        truth = panel.iloc[t+1:t+9].to_numpy().T
        truths.append(truth)
        for kind,pr in [("saturday_proxy",b),("saturday_021",b*(totals/old))]:
            batches[kind].append(pr)
            rows.append(dict(kind=kind,model="analogs+lgbm28" if kind.endswith("proxy") else key[0],
                             mode="" if kind.endswith("proxy") else key[1],seed=42,
                             alpha=0 if kind.endswith("proxy") else key[2],cap=0 if kind.endswith("proxy") else key[3],
                             fold=f"S-{weeks}",origin=str(o),group="all",n=truth.size,**metric(truth,pr)))
    for kind,pr in batches.items():
        rows.append(dict(kind=kind,seed=42,fold="pooled",group="all",n=np.size(truths),**metric(truths,pr)))
    table = pd.DataFrame(rows)
    table.to_csv(args.output,index=False)
    (args.output.parent/"rule.json").write_text(json.dumps(rule,indent=2))
    print(table.loc[(table.kind.isin(["reference","forward","lofo","shape_ablation"])) & (table.group=="all"),
                    ["kind","model","mode","alpha","cap","fold","wape","signed_bias","score"]].to_string(index=False))


if __name__ == "__main__":
    main()

