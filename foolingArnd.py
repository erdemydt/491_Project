import utilityFunctions as uf
import matplotlib.pyplot as plt

# Example usage using utility functions from utilityFunctions
gamma = 1.0
sigma, omega, _ = uf.temps_to_sigma_omega(Tc=1.6, Th=2.0, DeltaE=1.0, k_B=uf.kB)
print("sigma, omega, and epsilon from temps:", sigma, omega, _)
tau = 1.0
delta = 0.9  # incoming bit bias

det = uf.deterministic_solution(gamma, sigma, omega, tau, delta)
print("Deterministic Phi:", det["Phi"])
print("Outgoing bit distribution:", det["pB_out"])
print("Q_c->h per bit (in ΔE units):", uf.Q_c_to_h(det["Phi"], DeltaE=2.0))
print("Delta S_B per bit (nats):", uf.DeltaS_B(delta, det["Phi"]))
mc = uf.monte_carlo_tape(100000, gamma, sigma, omega, tau, delta, seed=42)
print("Monte Carlo Phi_emp:", mc["Phi_emp"])

conv = uf.convergence_study(gamma, sigma, omega, tau, delta)
print("Ns:", conv["N"])
print("Phi_emp:", conv["Phi_emp"])
print("Absolute errors:", conv["abs_err"])

# Optional: plot running convergence of Phi for one long run
plt.figure()
plt.plot(mc["running_phi"])
plt.axhline(det["Phi"], linestyle='--')
plt.xlabel("Number of bits processed")
plt.ylabel("Empirical Phi (running)")
plt.title("Convergence of Monte Carlo to Deterministic Phi")
plt.show()
    
