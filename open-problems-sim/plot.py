"""Draw figures.png from results.json."""
import json, os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(here, "results.json")))
fig, ax = plt.subplots(4, 3, figsize=(17, 19)); ax = ax.ravel()
B, O, G = "#2563eb", "#ea580c", "#6b7280"

R = r["riemann"]; s = np.linspace(0, 3, 200)
ax[0].hist(R["spacings"], bins=20, density=True, color=B, alpha=.7, label="first 150 zeros")
ax[0].plot(s, 32 / np.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / np.pi), color=O, label="random-matrix (GUE) prediction")
ax[0].plot(s, np.exp(-s), color=G, ls="--", label="uncorrelated points")
ax[0].set(title="1. Riemann: spacing between consecutive zeros", xlabel="normalised spacing"); ax[0].legend()

ax[1].semilogx(R["psi_x"], R["psi_err"], color=B)
ax[1].set(title="1. Riemann: prime-counting error (psi(x) - x) / sqrt(x)", xlabel="x")
ax[1].axhline(0, color=G, lw=.5)

P = r["p_vs_np"]; x = [d["ratio"] for d in P["sat"]]
ax[2].plot(x, [d["sat_fraction"] for d in P["sat"]], "o-", color=B, label="fraction satisfiable")
a2 = ax[2].twinx(); a2.plot(x, [d["median_branches"] for d in P["sat"]], "s-", color=O, label="median search branches")
ax[2].set(title=f"2. P vs NP: random 3-SAT, {P['sat_n']} variables", xlabel="clauses per variable")
ax[2].legend(loc="center right"); a2.legend(loc="upper right")

for run in r.get("navier_stokes", {}).get("runs", []):
    ax[3].plot(run["t"], run["enstrophy"], label=f"Re = {run['Re']:.0f}")
ax[3].set(title="3. Navier-Stokes: enstrophy of a 3D Taylor-Green vortex", xlabel="time"); ax[3].legend()

Y = r.get("yang_mills")
if Y:
    b = [d["beta"] for d in Y["data"]]
    ax[4].errorbar(b, [d["plaquette"] for d in Y["data"]], yerr=[d["plaquette_err"] for d in Y["data"]], fmt="o", color=B, label=f"Monte Carlo, {Y['L']}^4 lattice")
    bb = np.linspace(.3, 3.1, 100)
    ax[4].plot(bb[bb < 2], bb[bb < 2] / 4, color=O, label="strong-coupling expansion")
    ax[4].plot(bb[bb > 1.6], 1 - 3 / (4 * bb[bb > 1.6]), color=G, ls="--", label="weak-coupling expansion")
    ax[4].set(title="4. Yang-Mills: SU(2) average plaquette", xlabel="beta = 4 / g^2"); ax[4].legend()

for name, d in r["bsd"].items():
    ax[5].loglog(np.log(d["p"]), d["product"], label=f"{name}: slope {d['fitted_exponent']:.2f}")
ax[5].set(title="5. BSD: product of N_p / p over primes up to X", xlabel="log X"); ax[5].legend()

H = r["hodge"]["hypersurface_hodge_numbers"]; names = list(H)
ax[6].axis("off"); ax[6].set_title("6. Hodge: middle Hodge numbers of hypersurfaces")
txt = "\n".join(f"{n:<18} {H[n]}" for n in names)
txt += f"\n\nFermat quartic surface: its 48 lines span\nrank {r['hodge']['line_gram_rank']} = h^(1,1) = {r['hodge']['h11_of_K3']}"
ax[6].text(0, .95, txt, va="top", family="monospace", fontsize=10)

Gd = r["goldbach_twins"]
ax[7].plot(Gd["comet_n"], Gd["comet"], ".", ms=1.5, color=B)
ax[7].set(title="7. Goldbach: ways to write n as a sum of two primes", xlabel="even n")

t = Gd["twins"]
ax[8].loglog([d["X"] for d in t], [d["twin_pairs"] for d in t], "o", color=B, label="counted")
ax[8].loglog([d["X"] for d in t], [d["hardy_littlewood"] for d in t], color=O, label="Hardy-Littlewood prediction")
ax[8].set(title="8. Twin primes below X", xlabel="X"); ax[8].legend()

C = r["collatz"]
ax[9].plot(range(1, 10000), C["steps_first_10000"][1:], ".", ms=1.5, color=B)
ax[9].set(title="9. Collatz: steps to reach 1", xlabel="starting value")

e = np.array(C["steps_over_log_n_edges"])
ax[10].bar(e[:-1], C["steps_over_log_n_hist"], width=np.diff(e), color=B, alpha=.7, align="edge")
ax[10].axvline(C["steps_over_log_n_heuristic"], color=O, label="random-walk heuristic")
ax[10].set(title="9. Collatz: steps / ln(n), random 60-bit starts"); ax[10].legend()

A = r["abc"]
ax[11].semilogx(A["cs"], A["qualities"], ".", color=B, ms=4)
ax[11].axhline(1, color=G, lw=.5)
ax[11].set(title=f"10. abc: quality of all {len(A['cs'])} hits with c <= {A['C']:,}", xlabel="c", ylabel="log c / log rad(abc)")

for a in ax: a.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures.png"), dpi=110)
