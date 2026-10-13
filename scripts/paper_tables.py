"""Tables 7-9 of the paper (Markdown) from the Excel outputs.

Usage: python scripts/paper_tables.py <results_root>
"""
import pandas as pd, sys
R = sys.argv[1] if len(sys.argv) > 1 else "results"
W = [("d01","Azure day 1"),("d05","Azure day 5"),("d10","Azure day 10"),("synthetic","Synthetic")]
X = lambda d, s: pd.read_excel(f"{R}/results_{d}/results.xlsx", s).set_index("method")
names = {"Industry (Fixed+Trigger)":"Industry","Industry + L3":"Industry + L3","Stackelberg-Prop [prior]":"Stackelberg-Prop","Stackelberg-Prop + L3":"Stackelberg-Prop + L3","Proposed":"Proposed"}
out = []
out.append("| Workload / method | $W$ (\\$) | $\\Pi$ (\\$) | $\\bar{C}^{\\mathrm{mix}}$ (\\$) | Miss (%) | P99 $R_j/S_j$ | $\\mathcal{F}$ | IR (%) | Core (%) |")
out.append("|:---|" + ":---:|"*8)
for d, wl in W:
    m = X(d, "Main_Summary")
    out.append(f"| {wl} | | | | | | | | |")
    for k, v in names.items():
        r = m.loc[k]
        out.append(f"| \\ \\ {v} | {r['social_welfare_$']:.0f}\u00a0±\u00a0{r['social_welfare_$ (±95%CI)']:.0f} | {r['provider_profit_$']:.0f} | {r['mix_adj_cost_per_1k_$']:.2f} | {r['sla_miss_incl_rejects_%']:.2f}\u00a0±\u00a0{r['sla_miss_incl_rejects_% (±95%CI)']:.2f} | {r['p99_norm_latency']:.2f} | {r['jain_cost_to_standalone']:.3f} | {r['ir_violation_%']:.2f} | {r['core_violation_%']:.2f} |")
print("\n".join(out)); print()
rules = [("Proposed","Shapley"),("w/o Shapley -> Equal","Equal"),("w/o Shapley -> Proportional","Proportional"),("w/o Shapley -> Trigger","Trigger")]
out = ["| Rule | $\\mathcal{F}$, D1 | $\\mathcal{F}$, D5 | $\\mathcal{F}$, D10 | $\\mathcal{F}$, Syn. | Core, D1 | Core, D5 | Core, D10 | Core, Syn. |", "|:---|" + ":---:|"*8]
A = {d: X(d, "Ablation_Summary") for d, _ in W}
for k, n in rules:
    out.append(f"| {n} | " + " | ".join(f"{A[d].loc[k,'jain_cost_to_standalone']:.3f}" for d,_ in W) + " | " + " | ".join(f"{A[d].loc[k,'core_violation_%']:.2f}" for d,_ in W) + " |")
print("\n".join(out)); print()
abl = [("w/o two-part -> Linear tariff","w/o two-part tariff"),("w/o capacity control","w/o capacity control"),("w/o fallback (admission only)","w/o fallback"),("w/o SLA-batch -> Fixed batch","Fixed batching"),("w/o SLA-batch -> No batch","No batching"),("w/o Stackelberg -> Fixed price","Fixed price")]
out = ["| Configuration | $\\Delta W$, D1 | $\\Delta W$, D5 | $\\Delta W$, D10 | $\\Delta W$, Syn. | Miss, D1 | Miss, D5 | Miss, D10 | Miss, Syn. |", "|:---|" + ":---:|"*8]
out.append("| Proposed | 0 | 0 | 0 | 0 | " + " | ".join(f"{A[d].loc['Proposed','sla_miss_incl_rejects_%']:.2f}" for d,_ in W) + " |")
for k, n in abl:
    dw = [100*(A[d].loc[k,'social_welfare_$']/A[d].loc['Proposed','social_welfare_$']-1) for d,_ in W]
    out.append(f"| {n} | " + " | ".join(f"{x:.1f}".replace("-","−") for x in dw) + " | " + " | ".join(f"{A[d].loc[k,'sla_miss_incl_rejects_%']:.2f}" for d,_ in W) + " |")
print("\n".join(out))
