"""Generate comparison plots showing stacking K demons mimics increasing τ.

Writes figures into ../images.
Run from anywhere:
  projectEnv/bin/python final-report/scripts/make_stacked_k_vs_tau.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Headless plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(ROOT, "code", "stacked_demons")
OUT_DIR = os.path.join(ROOT, "images")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)

from Demon import Demon, PhysParams  # noqa: E402
from Tape import Tape  # noqa: E402
from Simulation import StackedDemonSimulation  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    demon_n: int = 2
    N: int = 6000
    p0: float = 1.0
    sigma: float = 0.3
    omega: float = 0.8
    DeltaE: float = 1.0
    gamma: float = 1.0
    base_seed: int = 123
    repeats: int = 5


def _phi_for(K: int, tau: float, cfg: RunConfig) -> float:
    """Average φ over repeats for stability."""
    phys = PhysParams(sigma=cfg.sigma, omega=cfg.omega, DeltaE=cfg.DeltaE, gamma=cfg.gamma)

    phi_vals: list[float] = []
    for r in range(cfg.repeats):
        # Ensure full reproducibility across both tape init and Gillespie draws.
        np.random.seed(cfg.base_seed + 1000 * K + 37 * int(round(1000 * tau)) + r)

        demons = [Demon(n=cfg.demon_n, phys_params=phys, init_state="d0") for _ in range(K)]
        tape = Tape(N=cfg.N, p0=cfg.p0, seed=cfg.base_seed + r)
        sim = StackedDemonSimulation(demons=demons, tape=tape, tau=tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        phi_vals.append(float(stats["phi"]))

    return float(np.mean(phi_vals))


def main() -> None:
    cfg = RunConfig()

    tau0 = 1.0
    K_values = [1, 2, 3, 5, 8, 13]
    tau_eff = [K * tau0 for K in K_values]

    phi_stacked = [_phi_for(K=K, tau=tau0, cfg=cfg) for K in K_values]
    phi_single = [_phi_for(K=1, tau=t, cfg=cfg) for t in tau_eff]

    # Plot: φ vs effective interaction time
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(tau_eff, phi_single, "o-", label=r"Single demon: $K=1$, varying $\tau$")
    plt.plot(tau_eff, phi_stacked, "s-", label=rf"Stacked demons: varying $K$, fixed $\tau={tau0}$")
    plt.xlabel(r"Effective interaction time $\tau_{\mathrm{eff}} = K\tau$")
    plt.ylabel(r"Information current $\Phi$ (net $0\to 1$ flips per bit)")
    plt.title(
        r"Stacking $K$ demons vs increasing $\tau$" + "\n"
        + rf"$\sigma={cfg.sigma}$, $\omega={cfg.omega}$, $\gamma={cfg.gamma}$, $N={cfg.N}$, $p_0={cfg.p0}$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "stacked_k_vs_tau.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Companion plot: φ vs K directly
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(K_values, phi_stacked, "s-", color="tab:blue")
    plt.xlabel(r"Number of stacked demons $K$")
    plt.ylabel(r"Information current $\Phi$")
    plt.title(
        r"Increasing $K$ boosts $\Phi$ without changing tape speed" + "\n"
        + rf"Fixed per-demon $\tau={tau0}$, $\sigma={cfg.sigma}$, $\omega={cfg.omega}$"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path2 = os.path.join(OUT_DIR, "phi_vs_K_stacked.png")
    plt.savefig(out_path2, dpi=200)
    plt.close()

    print(f"Wrote {out_path}")
    print(f"Wrote {out_path2}")


if __name__ == "__main__":
    main()
