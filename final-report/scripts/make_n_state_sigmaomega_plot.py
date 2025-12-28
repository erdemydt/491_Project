"""Generate n-state plot with constant per-step (sigma, omega).

This matches the report discussion: preserve the *dimensionless* transition biases
(σ, ω) for each adjacent rung, instead of holding (T_h, T_c) fixed while shrinking ΔE.

Writes:
  ../images/n_state_phi_vs_n_sigmaomega.png

Run:
  ./projectEnv/bin/python final-report/scripts/make_n_state_sigmaomega_plot.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(ROOT, "code", "n_state")
OUT_DIR = os.path.join(ROOT, "images")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)

from Demon import Demon, PhysParams  # noqa: E402
from Tape import Tape  # noqa: E402
from Simulation import StackedDemonSimulation  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    N: int = 6000
    p0: float = 1.0
    tau: float = 10.0
    sigma: float = 0.2
    omega: float = 0.7
    DeltaE_per_step: float = 1.0
    gamma: float = 1.0
    seed: int = 123
    repeats: int = 4


def _phi_for_n(n: int, cfg: RunConfig) -> float:
    phi_vals: list[float] = []
    for r in range(cfg.repeats):
        np.random.seed(cfg.seed + 1000 * n + r)

        phys = PhysParams(
            sigma=cfg.sigma,
            omega=cfg.omega,
            DeltaE=cfg.DeltaE_per_step,
            gamma=cfg.gamma,
            delta_e_mode="per_state",
            preserve_mode="sigma_omega",
            demon_n=n,
        )
        demon = Demon(n=n, phys_params=phys, init_state="d0", energy_distribution="uniform")
        tape = Tape(N=cfg.N, p0=cfg.p0, seed=cfg.seed + r)
        sim = StackedDemonSimulation(demons=[demon], tape=tape, tau=cfg.tau)
        final_tape, _, _ = sim.run_full_simulation()
        stats = sim.compute_statistics(final_tape)
        phi_vals.append(float(stats["phi"]))

    return float(np.mean(phi_vals))


def main() -> None:
    cfg = RunConfig()
    n_values = list(range(2, 31))

    phi_values = []
    for i, n in enumerate(n_values):
        print(f"Progress: {i+1}/{len(n_values)} (n={n})", end="\r")
        phi_values.append(_phi_for_n(n, cfg))
    print()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(n_values, phi_values, "o-", color="tab:orange", linewidth=2, markersize=5)
    plt.xlabel(r"Number of demon states $n$")
    plt.ylabel(r"Information current $\Phi$")
    plt.title(
        r"$n$-state demon with per-rung biases held fixed" + "\n"
        + rf"Fixed $\sigma={cfg.sigma}$, $\omega={cfg.omega}$ per step; $\tau={cfg.tau}$, $N={cfg.N}$"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "n_state_phi_vs_n_sigmaomega.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
