"""Figures 3-6 of the paper from the Excel outputs.

Usage: python scripts/paper_figures.py <results_root> <output_dir>
where <results_root> contains results_d01/, results_d05/, results_d10/, results_synthetic/.
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":9,
                     "axes.linewidth":0.7,"xtick.major.width":0.6,"ytick.major.width":0.6,
                     "legend.frameon":False,"axes.spines.top":False,"axes.spines.right":False})
R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_figures"
os.makedirs(OUT, exist_ok=True)
X = lambda d, s: pd.read_excel(f"{R}/results_{d}/results.xlsx", s)
C = {"Industry":"#8c8c8c","Prior":"#3b6fb6","Proposed":"#c0392b","Equal":"#e08a3c","Trigger":"#5fb3c9"}
def label(fig, axes, y=-0.30):
    for k, ax in enumerate(axes):
        ax.text(0.5, y, "(%s)" % "abcdefghij"[k], transform=ax.transAxes, ha="center", va="top", fontsize=10)

# Fig. 3: pricing dynamics (day 1)
pr = X("d01", "Prices_Proposed"); dm = X("d01", "Demand_Proposed")
pfix = {"M-LLM-7B": 2*2.5/3600*(0.104+211*0.014)*1000, "L-LLM-70B": 2*31.6/3600*(0.084*1155/1020+211*0.028)*1000}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
for m, c, mk in (("M-LLM-7B", "#3b6fb6", "o"), ("L-LLM-70B", "#c0392b", "s")):
    ax[0].plot(pr["round"]+1, pr[f"{m} ($/1k req)"]/pfix[m], marker=mk, ms=3, lw=1, color=c, label=m)
ax[0].axhline(1, color="#666", lw=0.8, ls="--", label="Fixed price")
ax[0].set_xlabel("Round"); ax[0].set_ylabel(r"$p_{m,r}\,/\,p_m^{\mathrm{fix}}$"); ax[0].legend(fontsize=8)
ax[0].set_ylim(0.6, 1.2)
ax[1].plot(dm["round"]+1, dm["base"], color="#8c8c8c", lw=1, ls="--", label="Base demand")
ax[1].plot(dm["round"]+1, dm["planned"], color="#3b6fb6", lw=1, marker="o", ms=3, label="Best response")
ax[1].plot(dm["round"]+1, dm["realised"], color="#c0392b", lw=1, marker="^", ms=3, label="Realized")
ax[1].set_xlabel("Round"); ax[1].set_ylabel("Requests per round"); ax[1].legend(fontsize=8)
label(fig, ax, -0.28); fig.tight_layout(w_pad=2.5); fig.subplots_adjust(bottom=0.27)
fig.savefig(f"{OUT}/fig3_pricing.png", dpi=400)

# Fig. 4: per-user allocation fairness (day 1)
sheets = [("Shapley","PerUser_Proposed",C["Proposed"]),("Equal","PerUser_wo_Shapley_Equal",C["Equal"]),
          ("Proportional","PerUser_wo_Shapley_Proportional",C["Prior"]),("Trigger","PerUser_wo_Shapley_Trigger",C["Trigger"])]
pu = {n: X("d01", s) for n, s, _ in sheets}
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
for a, n in zip(ax[:2], ("Shapley", "Proportional")):
    d = pu[n][pu[n].requests > 0]
    sc = a.scatter(d.isolation_ratio, d.overhead_per_req*1000, c=np.log10(d.base_rate_per_min), cmap="viridis", s=12, edgecolor="k", lw=0.3)
    a.set_yscale("log"); a.set_xlabel("Isolation ratio"); a.set_ylabel("Overhead paid ($ per 1k req)")
    a.set_ylim(5e-3, 2e2)
cb = fig.colorbar(sc, ax=ax[1], pad=0.02, fraction=0.06); cb.set_label(r"$\log_{10}$ rate (req/min)", fontsize=8); cb.ax.tick_params(labelsize=7)
data = [pu[n].cost_to_standalone.dropna() for n, _, _ in sheets]
bp = ax[2].boxplot(data, widths=0.55, patch_artist=True, flierprops=dict(ms=2.5))
for p, (_, _, c) in zip(bp["boxes"], sheets): p.set_facecolor(c); p.set_alpha(0.75)
for md in bp["medians"]: md.set_color("k")
ax[2].axhline(1, color="k", ls="--", lw=0.8)
ax[2].set_yscale("log"); ax[2].set_xticks(range(1,5)); ax[2].set_xticklabels(["Shapley","Equal","Prop.","Trigger"], fontsize=8)
ax[2].set_ylabel("Overhead / stand-alone cost")
label(fig, ax, -0.36); fig.tight_layout(w_pad=1.2); fig.subplots_adjust(bottom=0.30)
fig.savefig(f"{OUT}/fig4_allocation.png", dpi=400)

# Fig. 5: sensitivity (day 1)
s = X("d01", "Sensitivity_Summary")
pars = [("load_scale","Load scale",True),("keepalive_s","Keep-alive window (s)",True),("cold_scale","Cold-start scale",False),("cs_weight",r"Weight $\omega$",False)]
mets = [("social_welfare_$","Social welfare ($)"),("sla_miss_incl_rejects_%","SLO miss (%)")]
fig, ax = plt.subplots(2, 4, figsize=(7.0, 4.5))
M5 = [("Industry (Fixed+Trigger)","Industry",C["Industry"],"s","--"),("Industry + L3","Industry + L3","#3a3a3a","D","-"),
      ("Stackelberg-Prop [prior]","Stackelberg-Prop",C["Prior"],"o","--"),("Stackelberg-Prop + L3","Stackelberg-Prop + L3","#6baed6","v","-"),
      ("Proposed","Proposed",C["Proposed"],"^","-")]
for i, (mt, ml) in enumerate(mets):
    for j, (p, pl, logx) in enumerate(pars):
        a = ax[i, j]
        for key, nm, col, mk, ls in M5:
            d = s[(s.param == p) & (s.method == key)].sort_values("value")
            a.errorbar(d.value, d[mt], yerr=d[f"{mt} (±95%CI)"], color=col, marker=mk, ms=3, lw=0.9, ls=ls, capsize=1.5, label=nm)
        vals = sorted(s[s.param == p].value.unique())
        if logx: a.set_xscale("log", base=2)
        a.set_xticks(vals); a.set_xticklabels([("%g" % v) for v in vals])
        a.set_xlabel(pl, fontsize=8.5)
        if j == 0: a.set_ylabel(ml)
        a.tick_params(labelsize=7.5)
h, l = ax[0,0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(h_pad=3.2, w_pad=1.0); fig.subplots_adjust(top=0.92, bottom=0.16, hspace=0.75)
label(fig, list(ax.flatten()), -0.36)
fig.savefig(f"{OUT}/fig5_sensitivity.png", dpi=400)

# Fig. 6: Shapley accuracy and scalability
acc = X("d01", "Shapley_Accuracy"); sc_ = X("d01", "Shapley_Scalability")
e = acc.groupby(["players","perms"])["rel_L1_error_%"].mean().unstack()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
for P, c in zip([16,32,64,128], ["#d4b9da","#c994c7","#df65b0","#980043"]):
    ax[0].plot(e.index, e[P], marker="o", ms=3, lw=1, color=c, label=f"Sampling, {P}")
ax[0].plot(e.index, e["closed-form"], marker="*", ms=6, lw=1.2, color="k", label="Theorem 1")
ax[0].set_xticks([4,6,8,10,12]); ax[0].set_xlabel("Users"); ax[0].set_ylabel(r"Relative $L_1$ error (%)"); ax[0].legend(fontsize=6.8)
t = acc.groupby("players")[["exact_time_s"]].mean()
t64 = acc[acc.perms.astype(str) == "64"].groupby("players")["mc_time_s"].mean()
tcf = acc[acc.perms.astype(str) == "closed-form"].groupby("players")["mc_time_s"].mean()
ax[1].semilogy(t.index, t.exact_time_s, marker="s", ms=3, lw=1, color="#3b6fb6", label="Enumeration")
ax[1].semilogy(t64.index, t64, marker="o", ms=3, lw=1, color="#df65b0", label="Sampling, 64")
ax[1].semilogy(tcf.index, tcf, marker="*", ms=6, lw=1.2, color="k", label="Theorem 1")
ax[1].set_xticks([4,6,8,10,12]); ax[1].set_xlabel("Users"); ax[1].set_ylabel("Time per round (s)"); ax[1].legend(fontsize=6.8)
ax[2].loglog(sc_.players, sc_.mc64_time_s, marker="o", ms=3, lw=1, color="#df65b0", label="Sampling, 64")
ax[2].loglog(sc_.players, sc_.closed_form_time_s, marker="*", ms=6, lw=1.2, color="k", label="Theorem 1")
ax[2].set_xlabel("Users"); ax[2].set_ylabel("Time per round (s)"); ax[2].legend(fontsize=6.8)
label(fig, ax, -0.30); fig.tight_layout(w_pad=1.2); fig.subplots_adjust(bottom=0.27)
fig.savefig(f"{OUT}/fig6_shapley.png", dpi=400)
