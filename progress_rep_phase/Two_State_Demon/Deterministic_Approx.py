import numpy as np
import Constants as const
from Utils import get_sigma_omega_from_T_H_C, get_epsilon_from_sigma_omega
from dataclasses import dataclass

@dataclass
class SimParams:
    sigma: float
    omega: float
    epsilon: float
    gamma: float
    DeltaE: float
    tau: float
    
STATE = {0: "0u", 1: "0d", 2: "1u", 3: "1d"}
def make_rates_split(params: "SimParams"):
    """
    Build separate generators for hot and cold channels.

    Off-diagonals (i->j) are the physical jump rates for that channel only.
    Diagonals are set so each channel matrix has row-sum zero, and
    R_total = R_hot + R_cold.

    Hot bath (vertical flips within each bit sector):
        d -> u :  gamma * (1 - sigma)
        u -> d :  gamma * (1 + sigma)

    Cold bath (cooperative diagonal only):
        0d -> 1u : 1 - omega
        1u -> 0d : 1 + omega

    Returns
    -------
    R_hot, R_cold, R_total, info
    """
    sigma = float(params.sigma)
    omega = float(params.omega)
    gamma = float(params.gamma)

    # --- validate ---
    if not (-1.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (-1,1); got {sigma}")
    if not (-1.0 < omega < 1.0):
        raise ValueError(f"omega must be in (-1,1); got {omega}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0; got {gamma}")

    # Hot rates (same in both bit sectors)
    rate_d_to_u = gamma * (1.0 - sigma)  # 0d->0u and 1d->1u
    rate_u_to_d = gamma * (1.0 + sigma)  # 0u->0d and 1u->1d

    R_hot  = np.zeros((4, 4), dtype=float)
    R_cold = np.zeros((4, 4), dtype=float)

    # ----- HOT: vertical flips -----
    # indices: 0:'0u', 1:'0d', 2:'1u', 3:'1d'
    R_hot[1, 0] = rate_u_to_d  # 0u -> 0d
    R_hot[0, 1] = rate_d_to_u  # 0d -> 0u
    R_hot[3, 2] = rate_u_to_d  # 1u -> 1d
    R_hot[2, 3] = rate_d_to_u  # 1d -> 1u

    # row-sum convention for the hot channel
    np.fill_diagonal(R_hot, -R_hot.sum(axis=1))

    # ----- COLD: cooperative diagonal only -----
    R_cold[1, 2] = 1.0 - omega  # 0d -> 1u
    R_cold[2, 1] = 1.0 + omega  # 1u -> 0d

    # row-sum convention for the cold channel
    np.fill_diagonal(R_cold, -R_cold.sum(axis=1))

    R_total = R_hot + R_cold

    info = {
        "sigma": sigma,
        "omega": omega,
        "gamma": gamma,
        "epsilon_from_sigma_omega": float((omega - sigma) / (1.0 - omega * sigma)),
        "state_index": STATE.copy(),
        "checks": {
            "rowsum_R_hot": R_hot.sum(axis=1).tolist(),
            "rowsum_R_cold": R_cold.sum(axis=1).tolist(),
            "rowsum_R_total": R_total.sum(axis=1).tolist(),
        },
        "rates": {
            "hot": {"d->u": rate_d_to_u, "u->d": rate_u_to_d},
            "cold": {"0d->1u": 1.0 - omega, "1u->0d": 1.0 + omega},
        },
    }
    return R_hot, R_cold, R_total, info
simparams = SimParams(
    *get_sigma_omega_from_T_H_C(const.T_H, const.T_C, const.k_B, const.DeltaE),
    epsilon=get_epsilon_from_sigma_omega(
        *get_sigma_omega_from_T_H_C(const.T_H, const.T_C, const.k_B, const.DeltaE)
    ),
    gamma=const.gamma,
    DeltaE=const.DeltaE,
    tau=1.0,
)

R_matrix, R_info = make_rates_split(simparams)