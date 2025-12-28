"""Generate figures for the two-bit interacting demon section.

Creates:
  - ../images/two_bit_metrics_comparison.png
  - ../images/two_bit_thermo_comparison.png
  - ../images/two_bit_input_entropy_sweep.png

Run:
  ./projectEnv/bin/python final-report/scripts/make_two_bit_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(ROOT, "code", "two_bit_interacting")
OUT_DIR = os.path.join(ROOT, "images")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)

from Demon import TwoBitDemon, PhysParams  # noqa: E402
from Simulation import compare_demons, compare_demons_with_thermodynamics, sweep_p0_input_entropy_analysis  # noqa: E402


def save_metrics_comparison() -> None:
    phys = PhysParams(Th=1.6, Tc=1.0, DeltaE=1.0, gamma=1.0)
    tape_params = {"N": 10000, "p0": 1.0, "init_mode": "random"}
    tau = 10.0

    demon = TwoBitDemon(phys_params=phys, init_state="d")
    comp = compare_demons(
        tape_params=tape_params,
        phys_params=phys,
        tau=tau,
        seed=42,
        plot=False,
        two_bit_demon=demon,
        title="Default two-bit transitions",
    )

    two = comp["two_bit"]["stats"]
    one = comp["single_bit"]["stats"]

    labels = [r"$\Phi$", r"$\Delta S_B$", r"$\Delta\mathrm{corr}$", r"$\Delta I$ (pairs)"]
    two_vals = [
        two["phi"],
        two["changes"]["delta_entropy"],
        two["changes"]["delta_pair_correlation"],
        two["changes"]["delta_mutual_information"],
    ]
    one_vals = [
        one["phi"],
        one["changes"]["delta_entropy"],
        one["changes"]["delta_pair_correlation"],
        one["changes"]["delta_mutual_information"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(x - width / 2, two_vals, width, label="Two-bit demon", color="steelblue", alpha=0.85)
    plt.bar(x + width / 2, one_vals, width, label="Single-bit demon", color="coral", alpha=0.85)
    plt.xticks(x, labels)
    plt.axhline(0, color="gray", linewidth=1, alpha=0.6)
    plt.ylabel("Value")
    plt.title(r"Two-bit demon can reshape tape correlations" + "\n" + rf"$\tau={tau}$, $N={tape_params['N']}$, $T_h={phys.Th}$, $T_c={phys.Tc}$")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "two_bit_metrics_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def save_thermo_comparison() -> None:
    phys = PhysParams(Th=1.6, Tc=1.0, DeltaE=1.0, gamma=1.0)
    tape_params = {"N": 10000, "p0": 1.0, "init_mode": "random"}
    tau = 10.0

    comp = compare_demons_with_thermodynamics(
        tape_params=tape_params,
        phys_params=phys,
        tau=tau,
        seed=42,
        plot=False,
        title="Default two-bit transitions",
    )

    two_t = comp["two_bit"]["thermodynamics"]
    one_t = comp["single_bit"]["thermodynamics"]

    labels = [r"$Q_h$", r"$Q_c$", r"$\Delta S_{\mathrm{tot}}$"]
    two_vals = [
        two_t["energy"]["Q_h"],
        two_t["energy"]["Q_c"],
        two_t["entropy"]["S_total_production"],
    ]
    one_vals = [
        one_t["energy"]["Q_h"],
        one_t["energy"]["Q_c"],
        one_t["entropy"]["S_total_production"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(x - width / 2, two_vals, width, label="Two-bit demon", color="steelblue", alpha=0.85)
    plt.bar(x + width / 2, one_vals, width, label="Single-bit demon", color="coral", alpha=0.85)
    plt.xticks(x, labels)
    plt.axhline(0, color="gray", linewidth=1, alpha=0.6)
    plt.ylabel("Energy / entropy units")
    plt.title(r"Thermodynamic comparison" + "\n" + rf"$\tau={tau}$, $N={tape_params['N']}$")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "two_bit_thermo_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def save_input_entropy_sweep() -> None:
    phys = PhysParams(sigma=0.4, omega=0.6, DeltaE=1.0, gamma=1.0)

    results = sweep_p0_input_entropy_analysis(
        tape_size=4000,
        tau=2.0,
        phys_params=phys,
        seed_base=100,
        two_bit_demon=TwoBitDemon(phys_params=phys, init_state="d"),
        n_points=11,
        plot=False,
    )

    x_two = np.asarray(results["two_bit"]["input_entropy"], dtype=float)
    phi_two = np.asarray(results["two_bit"]["phi"], dtype=float)
    phi_one = np.asarray(results["single_bit"]["phi"], dtype=float)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(x_two, phi_two, "o-", color="steelblue", label="Two-bit")
    plt.plot(x_two, phi_one, "s--", color="coral", label="Single-bit")
    plt.axhline(0, color="gray", linewidth=1, alpha=0.6)
    plt.xlabel(r"Input tape entropy $S(\delta)\,N$")
    plt.ylabel(r"Information current $\Phi$")
    plt.title(r"$\Phi$ vs input entropy (sweeping $p_0$)" + "\n" + r"Highlights regimes where two-bit processing differs")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "two_bit_input_entropy_sweep.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    save_metrics_comparison()
    save_thermo_comparison()
    save_input_entropy_sweep()


if __name__ == "__main__":
    main()
