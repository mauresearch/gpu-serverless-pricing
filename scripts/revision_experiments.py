"""Additional experiments: dispatching with predicted request sizes, and
strategic manipulation of the Shapley allocation (Sybil accounts, delayed requests).

Usage
    python scripts/revision_experiments.py workpred --azure-functions data/invocations_per_function_md.anon.d01.csv --out results_rev/workpred
    python scripts/revision_experiments.py manipulation --azure-functions data/invocations_per_function_md.anon.d01.csv --out results_rev/manipulation
"""
import argparse
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_serverless_pricing_sim as sim  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_config(af):
    cfg = sim.Config()
    llm = os.path.join(ROOT, "data", "AzureLLMInferenceTrace_conv.csv")
    if os.path.exists(llm):
        cfg = replace(cfg, azure_llm_path=llm)
    if af:
        cfg = replace(cfg, workload="azure_functions", azure_functions_path=os.path.abspath(af))
    return cfg


def _run(args):
    label, variant, cfg = args
    res = sim.simulate(cfg)
    s = res["summary"]
    return dict(method=label, predictor=variant, seed=cfg.seed, **{k: s[k] for k in (
        "social_welfare_$", "provider_profit_$", "mix_adj_cost_per_1k_$", "sla_miss_incl_rejects_%",
        "p99_norm_latency", "rejection_%", "degraded_%", "avg_batch_size")})


def workpred(a):
    base = replace(base_config(a.azure_functions), core_samples=0, core_exact_max=0)
    variants = [("oracle", "oracle", 0.0), ("noisy 25%", "noisy", 0.25), ("noisy 50%", "noisy", 0.5),
                ("noisy 100%", "noisy", 1.0), ("mean only", "mean", 0.0)]
    jobs = [(lab, vname, replace(base, seed=s, work_pred=wp, work_pred_sigma=sg, **ov))
            for lab, ov in sim.MAIN_METHODS.items() for vname, wp, sg in variants
            for s in range(1, a.seeds + 1)]
    with Pool(a.workers) as pool:
        rows = pool.map(_run, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    g = df.groupby(["predictor", "method"])
    summ = g.mean(numeric_only=True).drop(columns="seed")
    ci = 1.96 * g.std(numeric_only=True).drop(columns="seed") / np.sqrt(a.seeds)
    summ = summ.join(ci.add_suffix(" (±95%CI)")).reset_index()
    with pd.ExcelWriter(os.path.join(a.out, "workpred.xlsx")) as xw:
        summ.to_excel(xw, sheet_name="Summary", index=False)
        df.to_excel(xw, sheet_name="PerSeed", index=False)
    print(summ[["predictor", "method", "social_welfare_$", "sla_miss_incl_rejects_%", "mix_adj_cost_per_1k_$"]]
          .round(2).to_string(index=False))


def fee_vector(times_by_user, O, bc, ip, ka):
    """Budget-balanced Shapley fees of one cell, as allocated by the framework."""
    g = sim.CostGame(times_by_user, ka, bc, ip)
    phi = g.shapley_closed_form()
    pos = {u: max(0.0, v) for u, v in phi.items()}
    tot = sum(pos.values())
    return {u: O * v / tot for u, v in pos.items()} if tot > 0 else {u: O / len(pos) for u in pos}, g


def _manip_seed(args):
    seed, cfg = args
    res = sim.simulate(replace(cfg, seed=seed, dump_cells=True, core_samples=0, core_exact_max=0))
    ka = cfg.keepalive_s
    p_ref, slo = res["p_ref"], res["slo_nominal"]
    syb, dly = [], []
    for (r, m), (tb, O, bc, ip) in res["cells"].items():
        if O <= 0 or len(tb) < 2:
            continue
        fee, g = fee_vector(tb, O, bc, ip, ka)
        k = O / g.cost(g.players)
        tmax = 10 ** 9
        for u, ts in tb.items():
            ts = np.sort(ts)
            # Sybil: split into k accounts, (a) alternating requests, (b) consecutive blocks in time
            for kacc in (2, 3):
                for mode in ("alternating", "time blocks"):
                    if ts.size < kacc:
                        continue
                    if mode == "alternating":
                        parts = [ts[i::kacc] for i in range(kacc)]
                    else:
                        parts = np.array_split(ts, kacc)
                    tb2 = {v: w for v, w in tb.items() if v != u}
                    for i, p in enumerate(parts):
                        tb2[tmax + i] = p
                    f2, _ = fee_vector(tb2, O, bc, ip, ka)
                    new = sum(f2[tmax + i] for i in range(kacc))
                    syb.append(dict(seed=seed, round=r, model=m, user=u, accounts=kacc, mode=mode,
                                    fee=fee[u], fee_split=new))
            # Delay: move requests that open a warm period of u to the next instant, within d seconds,
            # at which another user is warm
            others = [(s_, e_) for v in tb if v != u for s_, e_ in zip(*g.iv[v])]
            if not others:
                continue
            os_, oe_ = np.array([o[0] for o in others]), np.array([o[1] for o in others])
            st_u, en_u = g.iv[u]
            for d in (5.0, 15.0, 30.0, 60.0):
                ts2 = ts.copy()
                shifted = []
                for s0 in st_u:
                    if ((os_ <= s0) & (oe_ >= s0)).any():
                        continue
                    nxt = os_[(os_ > s0) & (os_ <= s0 + d)]
                    if nxt.size == 0:
                        continue
                    t1 = nxt.min()
                    mask = (ts2 >= s0) & (ts2 < t1)
                    shifted += list(t1 - ts2[mask])
                    ts2[mask] = t1
                if not shifted:
                    dly.append(dict(seed=seed, round=r, model=m, user=u, d=d, n_shifted=0, fee=fee[u],
                                    fee_delayed=fee[u], value_loss=0.0, overhead=O, overhead_delayed=O))
                    continue
                tb2 = dict(tb)
                tb2[u] = ts2
                g2 = sim.CostGame(tb2, ka, bc, ip)
                O2 = k * g2.cost(g2.players)
                f2, _ = fee_vector(tb2, O2, bc, ip, ka)
                slack = 0.8 * slo[m]
                late = sum(1 for dt in shifted if dt > slack)
                loss = late * (1 - cfg.late_value) * p_ref[m]
                dly.append(dict(seed=seed, round=r, model=m, user=u, d=d, n_shifted=len(shifted), fee=fee[u],
                                fee_delayed=f2[u], value_loss=loss, overhead=O, overhead_delayed=O2))
    return syb, dly


def manipulation(a):
    cfg = replace(base_config(a.azure_functions), **sim.MAIN_METHODS["Proposed"])
    with Pool(a.workers) as pool:
        out = pool.map(_manip_seed, [(s, cfg) for s in range(1, a.seeds + 1)], chunksize=1)
    syb = pd.DataFrame([r for o in out for r in o[0]])
    dly = pd.DataFrame([r for o in out for r in o[1]])
    syb["gain"] = syb["fee"] - syb["fee_split"]
    syb["gain_%"] = 100 * syb["gain"] / syb["fee"].where(syb["fee"] > 0)
    dly["fee_saving"] = dly["fee"] - dly["fee_delayed"]
    dly["net_gain"] = dly["fee_saving"] - dly["value_loss"]
    dly["fee_saving_pct"] = 100 * dly["fee_saving"] / dly["fee"].where(dly["fee"] > 0)
    dly["overhead_red_pct"] = 100 * (dly["overhead"] - dly["overhead_delayed"]) / dly["overhead"]
    s1 = syb.groupby(["accounts", "mode"]).agg(cases=("gain", "size"),
                                               profitable_pct=("gain", lambda x: 100 * (x > 1e-12).mean()),
                                               mean_change_pct=("gain_%", "mean"),
                                               max_gain_pct=("gain_%", "max")).reset_index()
    act = dly[dly["n_shifted"] > 0]
    s2 = dly.groupby("d").agg(user_rounds=("fee", "size"),
                              can_shift_pct=("n_shifted", lambda x: 100 * (x > 0).mean())).reset_index()
    s2b = act.groupby("d").agg(shifted_cases=("fee", "size"),
                               fee_saving_positive_pct=("fee_saving", lambda x: 100 * (x > 1e-12).mean()),
                               net_gain_positive_pct=("net_gain", lambda x: 100 * (x > 1e-12).mean()),
                               mean_fee_saving=("fee_saving", "mean"),
                               mean_value_loss=("value_loss", "mean"),
                               mean_net_gain=("net_gain", "mean"),
                               mean_fee_saving_pct=("fee_saving_pct", "mean"),
                               mean_overhead_reduction_pct=("overhead_red_pct", "mean")).reset_index()
    s2 = s2.merge(s2b, on="d", how="left")
    os.makedirs(a.out, exist_ok=True)
    with pd.ExcelWriter(os.path.join(a.out, "manipulation.xlsx")) as xw:
        s1.to_excel(xw, sheet_name="Sybil_Summary", index=False)
        s2.to_excel(xw, sheet_name="Delay_Summary", index=False)
        syb.to_excel(xw, sheet_name="Sybil_Cases", index=False)
        dly.to_excel(xw, sheet_name="Delay_Cases", index=False)
    print(s1.round(3).to_string(index=False))
    print(s2.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["workpred", "manipulation"])
    ap.add_argument("--azure-functions", default=os.path.join(ROOT, "data", "invocations_per_function_md.anon.d01.csv"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--out", default="results_rev")
    a = ap.parse_args()
    workpred(a) if a.what == "workpred" else manipulation(a)
