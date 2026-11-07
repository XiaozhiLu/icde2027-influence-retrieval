"""三张论文图 v2"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "pdf.fonttype": 42, "figure.dpi": 300,
})
OUT = r"E:\work_dsh\icde_src"
x = np.arange(4); lbl = ["1M", "10M", "50M", "100M"]

# ================= F1: coverage vs scale, all tau =================
base10 = [204.72, 203.07, 208.41, 204.33]
o10 = {0.7: [612.29, 647.10, 655.77, 662.28],
       0.8: [477.34, 502.38, 506.61, 510.68],
       0.9: [341.33, 355.27, 358.70, 358.41]}
base50 = [1002.51, 1002.64, 1020.64, 1016.20]
o50 = {0.7: [3011.70, 3136.18, 3204.50, 3229.45],
       0.8: [2350.31, 2438.55, 2491.81, 2504.38],
       0.9: [1680.38, 1730.74, 1763.37, 1770.34]}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for ax, b, o, k, ttl in [(axes[0], base10, o10, 10, "(a) $K=10$"),
                         (axes[1], base50, o50, 50, "(b) $K=50$")]:
    ax.plot(x, b, "o--", color="#999999", label="HNSW (no influence)")
    for tau, mk, c in [(0.7, "s-", "#1f4e79"), (0.8, "^-", "#2e75b6"), (0.9, "d-", "#9dc3e6")]:
        ax.plot(x, o[tau], mk, color=c, label=f"Ours ($\\tau$={tau})")
    ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_xlabel("Corpus size"); ax.set_ylabel("Avg. one-hop coverage")
    ax.set_title(ttl, fontsize=8); ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left", ncol=1)
fig.tight_layout(); fig.savefig(OUT + r"\FIG2_scale.pdf", bbox_inches="tight"); plt.close(fig)

# ================= F2: CiaoDVD trade-off, K=10 and K=50 =================
d = {
 10: dict(ours_r=[0.500,0.600,0.700,0.799,0.8965],
          ours_i=[522.51,469.90,413.44,337.88,224.29],
          grd_r =[0.500,0.600,0.700,0.799,0.8965],
          grd_i =[465.52,428.88,382.00,328.20,224.18]),
 50: dict(ours_r=[0.5037,0.6218,0.7008,0.8002,0.8989],
          ours_i=[1092.79,969.61,877.90,739.14,549.30],
          grd_r =[0.5004,0.6203,0.7002,0.7999,0.8987],
          grd_i =[708.47,660.95,643.17,561.30,491.19]),
}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
for ax, k, ttl in [(axes[0], 10, "(a) $K=10$"), (axes[1], 50, "(b) $K=50$")]:
    e = d[k]
    ax.plot(e["grd_r"], e["grd_i"], "o--", color="#999999", label="Greedy")
    ax.plot(e["ours_r"], e["ours_i"], "s-", color="#1f4e79", label="Ours")
    ax.fill_between(e["ours_r"], e["grd_i"], e["ours_i"], color="#1f4e79", alpha=0.10)
    ax.set_xlabel("Achieved Recall@K"); ax.set_ylabel("Avg. one-hop coverage")
    ax.set_title(ttl, fontsize=8); ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + r"\FIG3_ciao.pdf", bbox_inches="tight"); plt.close(fig)

# ================= F3: ablation, feasibility view =================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

# (a) incremental build-up
ax = axes[0]
names = ["$K$ + 1v1", "+OS", "+OS\n+2v2"]
cov = [354.65, 368.58, 370.66]
ax.bar(range(3), cov, color=["#9dc3e6", "#2e75b6", "#1f4e79"],
       width=0.55, edgecolor="black", linewidth=0.4)
for i, v in enumerate(cov):
    ax.text(i, v + 4, f"{v:.1f}", ha="center", fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel("Avg. one-hop coverage"); ax.set_ylim(330, 385)
ax.set_title("(a) Replacement module", fontsize=8)
ax.grid(axis="y", alpha=0.3, linewidth=0.4)

# (b) coverage vs recall, feasible region
ax = axes[1]
pts = [("Hybrid only", 0.594, 688.02, "#c9c9c9", "o"),
       ("Repl. only", 0.883, 370.66, "#9dc3e6", "s"),
       ("Full", 0.818, 436.25, "#1f4e79", "*")]
ax.axvspan(0.70, 1.0, color="#f2f7ee", zorder=0)
for tv, col in [(0.70, "#a9d18e"), (0.80, "#548235"), (0.90, "#375623")]:
    ax.axvline(tv, color=col, linestyle="--", linewidth=0.7, zorder=1)
ax.text(0.703, 330, "$\\tau$=0.7", fontsize=6, color="#548235", rotation=90, va="bottom")
ax.text(0.803, 330, "$\\tau$=0.8", fontsize=6, color="#375623", rotation=90, va="bottom")
ax.text(0.903, 330, "$\\tau$=0.9", fontsize=6, color="#1f3864", rotation=90, va="bottom")
for nm, r, c, col, mk in pts:
    ax.scatter(r, c, s=110 if mk == "*" else 45, marker=mk, color=col,
               edgecolor="black", linewidth=0.5, zorder=3)
    ax.annotate(nm, (r, c), textcoords="offset points", xytext=(6, 5), fontsize=6.5)
ax.set_xlabel("Achieved Recall@10"); ax.set_ylabel("Avg. one-hop coverage")
ax.set_xlim(0.55, 1.0); ax.set_ylim(320, 740)
ax.set_title("(b) Two-stage fusion", fontsize=8)
ax.grid(alpha=0.3, linewidth=0.4)

fig.tight_layout(); fig.savefig(OUT + r"\FIG4_ablation.pdf", bbox_inches="tight"); plt.close(fig)
print("ok")
