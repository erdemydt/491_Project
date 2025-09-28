# maxwell_manim.py
# ------------------------------------------------------------
# Maxwell's Demon — Bit-Tape Trajectory Animation with Manim
#
# What this does:
# - Simulates a single stochastic trajectory of the demon + bit tape.
# - Animates incoming bits moving past the demon, with color-coded flips:
#     * Bit 0  -> blue
#     * Bit 1  -> orange
#   If a flip occurs (0↔1), the bit's color switches during the interaction.
# - Shows demon state (u/d) and animates its changes.
# - Pulses an "energy packet" between reservoirs when a 0→1 (cold→hot) or 1→0 (hot→cold) happens.
# - Keeps live counters: processed bits, 0→1, 1→0, net, Q_c→h total, and running δ_out.
#
# Usage:
#   manim -pqh maxwell_manim.py MaxwellDemonBitTape
#   (or: -pql / -pqm / -pqh for quality levels)
#
# This file is self-contained and independent of the notebook cells.
# It assumes SciPy is available for expm.
# ------------------------------------------------------------

from __future__ import annotations
import datetime
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from scipy.linalg import expm

from manim import (
    Scene, VGroup, Mobject, Dot, Circle, Square, Line, Arrow, Text, DecimalNumber,
    Integer, always_redraw, ValueTracker, FadeIn, FadeOut, Transform, TransformMatchingShapes,
    ReplacementTransform, Create, Uncreate, AnimationGroup, LaggedStart, GrowArrow,
    UP, DOWN, LEFT, RIGHT, ORIGIN, PI,          
    UL, UR, DL, DR,                              
    BLUE, ORANGE, YELLOW, PURPLE, TEAL, RED, GREEN, GREY, WHITE, BLACK, LIGHT_GREY, GOLD, MAROON
)

# ----------------------------
# Simulation Core (physics)
# ----------------------------

STATE_IDX = {"0u": 0, "0d": 1, "1u": 2, "1d": 3}
IDX_TO_STATE = {v: k for k, v in STATE_IDX.items()}

def build_sigma(beta_h: float, DeltaE: float) -> float:
    return math.tanh(0.5 * beta_h * DeltaE)

def build_omega(beta_c: float, DeltaE: float) -> float:
    return math.tanh(0.5 * beta_c * DeltaE)

def intrinsic_rates(gamma: float, sigma: float) -> Tuple[float, float]:
    # Demon-only flips (hot bath): d<->u
    R_d_to_u = gamma * (1.0 - sigma)
    R_u_to_d = gamma * (1.0 + sigma)
    return R_d_to_u, R_u_to_d

def cooperative_rates(omega: float) -> Tuple[float, float]:
    # Cooperative flips (cold bath): 0d <-> 1u
    R_0d_to_1u = 1.0 - omega
    R_1u_to_0d = 1.0 + omega
    return R_0d_to_1u, R_1u_to_0d

def build_R(DeltaE: float, Th: float, Tc: float, gamma: float, kB: float = 1.0) -> Tuple[np.ndarray, Dict]:
    """
    Build the 4x4 rate matrix for joint states in order [0u, 0d, 1u, 1d].
    R[i, j] is rate from state j -> i. Columns sum to zero.
    """
    beta_h = 1.0 / (kB * Th)
    beta_c = 1.0 / (kB * Tc)
    sig = build_sigma(beta_h, DeltaE)
    om = build_omega(beta_c, DeltaE)

    R_d_to_u, R_u_to_d = intrinsic_rates(gamma, sig)
    R_0d_to_1u, R_1u_to_0d = cooperative_rates(om)

    R = np.zeros((4, 4), dtype=float)

    # Intrinsic: 0u <-> 0d
    R[STATE_IDX["0u"], STATE_IDX["0d"]] += R_d_to_u  # 0d -> 0u
    R[STATE_IDX["0d"], STATE_IDX["0u"]] += R_u_to_d  # 0u -> 0d

    # Intrinsic: 1u <-> 1d
    R[STATE_IDX["1u"], STATE_IDX["1d"]] += R_d_to_u  # 1d -> 1u
    R[STATE_IDX["1d"], STATE_IDX["1u"]] += R_u_to_d  # 1u -> 1d

    # Cooperative: 0d <-> 1u
    R[STATE_IDX["1u"], STATE_IDX["0d"]] += R_0d_to_1u  # 0d -> 1u
    R[STATE_IDX["0d"], STATE_IDX["1u"]] += R_1u_to_0d  # 1u -> 0d

    # Diagonals: column sum zero
    for j in range(4):
        R[j, j] = -np.sum(R[:, j]) + R[j, j]

    meta = {
        "beta_h": beta_h, "beta_c": beta_c,
        "sigma": sig, "omega": om,
        "R_d->u": R_d_to_u, "R_u->d": R_u_to_d,
        "R_0d->1u": R_0d_to_1u, "R_1u->0d": R_1u_to_0d
    }
    return R, meta

def compute_U(R: np.ndarray, tau: float) -> np.ndarray:
    """Matrix exponential U = exp(R * tau)"""
    U = expm(R * tau)
    return U

def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.clip(probs, 0.0, 1.0)
    s = probs.sum()
    if s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return rng.choice(len(probs), p=probs)

@dataclass
class SimParams:
    # Physics
    DeltaE: float = 2.0
    Th: float = 2.5
    Tc: float = 1.2
    gamma: float = 1.0
    tau: float = 1.0
    kB: float = 1.0

    # Tape (incoming)
    p0: float = 0.9  # prob of 0 on input tape
    # p1 = 1 - p0

    # Trajectory length
    N: int = 40

    # RNG
    seed: int = 10

    def derived(self) -> Dict:
        beta_h = 1.0 / (self.kB * self.Th)
        beta_c = 1.0 / (self.kB * self.Tc)
        eps = math.tanh(0.5 * (beta_c - beta_h) * self.DeltaE)  # ε
        delta = self.p0 - (1.0 - self.p0)                        # δ
        return {"epsilon": eps, "delta": delta}

# ----------------------------
# Manim Helper Mobjects
# ----------------------------

BIT_COLOR = {0: BLUE, 1: ORANGE}
# Joint state colors: bit-demon combinations
JOINT_STATE_COLOR = {
    "0u": BLUE,     # Bit 0, demon up - cool blue
    "0d": TEAL,     # Bit 0, demon down - deep teal  
    "1u": ORANGE,   # Bit 1, demon up - warm orange
    "1d": RED       # Bit 1, demon down - hot red
}
# Legacy demon colors for backward compatibility
DEMON_COLOR = {"u": YELLOW, "d": PURPLE}
RES_COLD_COLOR = TEAL
RES_HOT_COLOR = RED
ENERGY_COLOR_C2H = GOLD
ENERGY_COLOR_H2C = MAROON
ENERGY_COLOR_H2D = ORANGE  # Hot to demon
ENERGY_COLOR_D2H = PURPLE  # Demon to hot

def make_reservoir(label: str, temp_text: str, width=2.8, height=1.6, color=GREY):
    rect = Line(ORIGIN, RIGHT * width, stroke_color=color, stroke_width=6).add_tip()
    top = Line(LEFT * width/2, RIGHT * width/2, stroke_color=color)
    bot = top.copy().shift(DOWN * height)
    left = Line(ORIGIN, DOWN * height, stroke_color=color).move_to(LEFT * width/2 + DOWN * height/2)
    right = left.copy().shift(RIGHT * width)
    group = VGroup(top, bot, left, right)
    text = Text(label, weight="BOLD").scale(0.5).next_to(group, UP, buff=0.1)
    temp = Text(temp_text).scale(0.4).next_to(group, DOWN, buff=0.1)
    return VGroup(group, text, temp)

def make_demon(demon_state: str = "u"):
    # A circle with demon state (u/d) inside
    # Extract demon state if joint state is passed
    if len(demon_state) > 1:
        demon_state = demon_state[1]  # Extract demon part from joint state
    
    color = DEMON_COLOR.get(demon_state, YELLOW)
    
    circ = Circle(radius=0.4, color=color, fill_opacity=0.25, stroke_width=4)
    # Show only demon state
    letter = Text(demon_state, weight="BOLD").scale(0.6).set_color(color)
    
    # Add demon label
    demon_label = Text("DEMON", weight="BOLD").scale(0.3).next_to(circ, UP, buff=0.15)
    
    mob = VGroup(circ, letter, demon_label)
    return mob

def update_demon(mob: VGroup, demon_state: str):
    # Handle demon state updates (extract demon part if joint state passed)
    if len(demon_state) > 1:
        demon_state = demon_state[1]  # Extract demon part from joint state
    
    color = DEMON_COLOR.get(demon_state, YELLOW)
    
    circ, letter, label = mob
    circ.set_stroke(color)
    circ.set_fill(color, opacity=0.25)
    letter.become(Text(demon_state, weight="BOLD").scale(0.6).set_color(color))
    return mob

def make_bit(bit_val: int):
    sq = Square(side_length=0.5, color=BIT_COLOR[bit_val], fill_opacity=0.15, stroke_width=3)
    t = Text(str(bit_val), weight="BOLD").scale(0.4).set_color(BIT_COLOR[bit_val])
    return VGroup(sq, t)

def recolor_bit(bit_mob: VGroup, new_val: int):
    sq, t = bit_mob
    old_pos = t.get_center()  # Preserve position
    sq.set_stroke(BIT_COLOR[new_val]).set_fill(BIT_COLOR[new_val], opacity=0.15)
    new_text = Text(str(new_val), weight="BOLD").scale(0.4).set_color(BIT_COLOR[new_val])
    new_text.move_to(old_pos)  # Keep text in same position
    t.become(new_text)
    return bit_mob

def energy_arrow(start, end, direction: str = "c2d"):
    # Arrow showing heat flow direction between demon and reservoir
    color_map = {
        "c2d": RES_COLD_COLOR,   # Cold to demon
        "d2c": RES_COLD_COLOR,   # Demon to cold  
        "h2d": RES_HOT_COLOR,    # Hot to demon
        "d2h": RES_HOT_COLOR     # Demon to hot
    }
    color = color_map.get(direction, RES_COLD_COLOR)
    arrow = Arrow(start, end, stroke_width=4, stroke_color=color, 
                  tip_length=0.2, max_tip_length_to_length_ratio=0.15)
    return arrow

# ----------------------------
# Main Manim Scene
# ----------------------------

class MaxwellDemonBitTape(Scene):
    def construct(self):
        # ---------- Parameters ----------
        P = SimParams(
            DeltaE=1.5, Th=1.35, Tc=1.2, gamma=1.0, tau=0.8,
            p0=0.95, N=10, seed=datetime.datetime.now().microsecond
        )
        rng = np.random.default_rng(P.seed)
        deriv = P.derived()
        delta = deriv["delta"]
        epsilon = deriv["epsilon"]

        # Build physics operators
        R, meta = build_R(P.DeltaE, P.Th, P.Tc, P.gamma, P.kB)
        sigma = meta["sigma"]  # Extract sigma for display
        U = compute_U(R, P.tau)

        # ---------- Static Layout ----------
        # Positions
        x_left = -5.8
        x_right = 5.8
        y_center = 0
        y_tape = -1.2

        cold = make_reservoir("COLD", f"Tc = {P.Tc:.2f}", color=RES_COLD_COLOR).scale(0.9).to_edge(LEFT).shift(UP*1.2)
        hot = make_reservoir("HOT", f"Th = {P.Th:.2f}", color=RES_HOT_COLOR).scale(0.9).to_edge(RIGHT).shift(UP*1.2)

        demon = make_demon("u").move_to(ORIGIN + UP*0.2)

        # Labels (ε, δ, and σ) - positioned to fit screen
        info_text = VGroup(
            Text("Parameters", weight="BOLD").scale(0.4),
            Text(f"ΔE = {P.DeltaE:.2f}, γ = {P.gamma:.2f}, τ = {P.tau:.2f}").scale(0.3),
            Text(f"ε = {epsilon:.3f}  (temp contrast)").scale(0.3),
            Text(f"σ = {sigma:.3f}  (hot bias)").scale(0.3),
            Text(f"δ_in = {delta:.3f}  (tape bias)").scale(0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_corner(UL).shift(DOWN*0.2)

        # Scoreboard (live)
        vt_n = ValueTracker(0)
        vt_01 = ValueTracker(0)
        vt_10 = ValueTracker(0)
        vt_q = ValueTracker(0.0)
        vt_delta_out = ValueTracker(0.0)  # running (p0' - p1') over observed

        def make_score():
            n = Integer(int(vt_n.get_value())).set_color(WHITE).scale(0.5)
            c01 = Integer(int(vt_01.get_value())).set_color(BLUE).scale(0.5)
            c10 = Integer(int(vt_10.get_value())).set_color(ORANGE).scale(0.5)
            q = DecimalNumber(vt_q.get_value(), num_decimal_places=2).set_color(GOLD).scale(0.5)
            dout = DecimalNumber(vt_delta_out.get_value(), num_decimal_places=3).set_color(WHITE).scale(0.5)

            labels = VGroup(
                Text("Processed:").scale(0.45), n,
                Text("0→1:").scale(0.45).set_color(BLUE), c01,
                Text("1→0:").scale(0.45).set_color(ORANGE), c10,
                Text("Q_c→h total:").scale(0.45).set_color(GOLD), q,
                Text("δ_out:").scale(0.45), dout
            ).arrange_in_grid(rows=5, cols=2, buff=0.12, col_alignments="lr")
            # Set position relative to demon
            box = VGroup(labels).arrange( buff=0.15).next_to(demon, DOWN*1.2, buff=1.2)
            return box

        scoreboard = always_redraw(make_score)
        scoreboard.next_to(demon, DOWN*1.2, buff=1.2)
        # Position scoreboard above demon

        # Tape baseline
        baseline = Line([x_left, y_tape, 0], [x_right, y_tape, 0], color=LIGHT_GREY, stroke_width=2)

        # Energy connection points between demon and reservoirs
        cold_point = cold[0].get_center() + DOWN*0.3 + RIGHT*0.5
        hot_point = hot[0].get_center() + DOWN*0.3 + LEFT*0.5

        # Show static elements
        self.play(
            FadeIn(cold, shift=RIGHT*0.2),
            FadeIn(hot, shift=LEFT*0.2),
            FadeIn(demon, shift=DOWN*0.2),
            Create(baseline),
            FadeIn(info_text, shift=DOWN*0.2),
            FadeIn(scoreboard, shift=DOWN*0.2)
        )
        self.wait(0.1)

        # ---------- Simulation State ----------
        # Demon initial state
        current_demon_state = "u"  # Start with demon up
        current_joint_state = "0u"  # Track joint state for physics
        # Demon visual already shows 'u' from initialization

        # Counters
        n_proc = 0
        count_01 = 0  # 0 -> 1
        count_10 = 0  # 1 -> 0
        out_count_0 = 0
        out_count_1 = 0

        # ---------- Per-bit animation loop ----------
        # Timings
        t_move_in = 0.3
        t_interact = 0.3
        t_move_out = 0.3

        for k in range(P.N):
            # Sample incoming bit
            bit_in = 0 if rng.random() < P.p0 else 1

            # Build initial joint state from current demon state and incoming bit
            demon_only = current_joint_state[1]  # Extract just 'u' or 'd'
            initial_joint = f"{bit_in}{demon_only}"  # e.g., "0d" or "1u"
            idx0 = STATE_IDX[initial_joint]

            # Evolve one interval: final distribution is column of U
            probs_final = U[:, idx0]
            idx_tau = sample_categorical(probs_final, rng)
            final_joint_state = IDX_TO_STATE[idx_tau]  # e.g., "1u"
            bit_out = 1 if final_joint_state[0] == "1" else 0
            demon_out = final_joint_state[1]

            # Make a bit mobject at left, move to demon, interact, then move right
            bit_mob = make_bit(bit_in).move_to([x_left, y_tape, 0])
            self.play(FadeIn(bit_mob, shift=RIGHT*0.3), run_time=0.2)
            self.play(bit_mob.animate.move_to([0, y_tape, 0]), run_time=t_move_in)

            # Determine what type of transition occurred - ONLY ONE TYPE PER INTERACTION
            flip = (bit_out != bit_in)
            demon_changed = (demon_out != demon_only)
            
            energy_arrows = []
            demon_center = demon.get_center()
            bit_center = bit_mob.get_center()
            # Check for cooperative transition 0d <-> 1u (COLD bath only)
            if initial_joint == "0d" and final_joint_state == "1u":
                # 0d -> 1u: cooperative transition via COLD bath
                count_01 += 1
                arrow = energy_arrow(cold_point, demon_center, "c2d")
                arrow2 = energy_arrow(cold_point, bit_center, "d2c")
                arrow.set_color(RES_HOT_COLOR)
                arrow2.set_color(RES_HOT_COLOR)
                energy_arrows.append(arrow2)
                energy_arrows.append(arrow)
            elif initial_joint == "1u" and final_joint_state == "0d":
                # 1u -> 0d: cooperative transition via COLD bath  
                count_10 += 1
                arrow = energy_arrow(demon_center, cold_point, "d2c")
                arrow2 = energy_arrow(bit_center, cold_point, "c2d")
                arrow.set_color(RES_HOT_COLOR)
                arrow2.set_color(RES_HOT_COLOR)
                energy_arrows.append(arrow2)
                energy_arrows.append(arrow)
            elif initial_joint == "0u" and final_joint_state == "1d":
                # 0u -> 1d: intrinsic demon transition with bit flip (HOT bath AND COLD bath, energy taken from cold, dumped to hot, at once)
                count_01 += 1
                arrow_c2d = energy_arrow(cold_point, bit_center, "c2d")
                arrow_d2h = energy_arrow(demon_center, hot_point, "d2h")
                arrow_c2d.set_color(RES_COLD_COLOR)
                arrow_d2h.set_color(RES_HOT_COLOR)
                energy_arrows.append(arrow_c2d)
                energy_arrows.append(arrow_d2h)
            elif initial_joint == "1u" and final_joint_state == "0u":
                # 1u -> 0u: intrinsic demon transition with bit flip (HOT bath AND COLD bath, energy taken from cold and hot, at once)
                count_10 += 1
                arrow_c2d = energy_arrow(cold_point, bit_center, "c2d")
                arrow_d2h = energy_arrow(demon_center, hot_point, "d2h")
                arrow_c2d.set_color(RES_COLD_COLOR)
                arrow_d2h.set_color(RES_HOT_COLOR)
                energy_arrows.append(arrow_c2d)
                energy_arrows.append(arrow_d2h)
            elif initial_joint == "0d" and final_joint_state == "1d":
                # 0d -> 1d: intrinsic demon transition with bit flip (HOT bath AND COLD bath, energy taken from cold, dumped to hot, at once)
                count_01 += 1
                arrow_h2d = energy_arrow(hot_point, demon_center, "h2d")
                arrow_d2c = energy_arrow(bit_center, cold_point, "d2c")
                arrow_h2d.set_color(RES_HOT_COLOR)
                arrow_d2c.set_color(RES_COLD_COLOR)
                energy_arrows.append(arrow_h2d)
                energy_arrows.append(arrow_d2c)
            elif initial_joint == "1d" and final_joint_state == "0d":
                # 1d -> 0d: intrinsic demon transition with bit flip (HOT bath AND COLD bath, energy taken from hot and dumped to cold, at once)
                count_10 += 1
                arrow_h2d = energy_arrow(hot_point, demon_center, "h2d")
                arrow_d2c = energy_arrow(bit_center, cold_point, "d2c")
                arrow_h2d.set_color(RES_HOT_COLOR)
                arrow_d2c.set_color(RES_COLD_COLOR)
                energy_arrows.append(arrow_h2d)
                energy_arrows.append(arrow_d2c)
            elif initial_joint == "0u" and final_joint_state == "1u":
                # 0u -> 1u: intrinsic demon transition with bit flip (HOT bath AND COLD bath, energy taken from cold, dumped to hot, at once)
                count_01 += 1
                arrow_h2d = energy_arrow(hot_point, demon_center, "h2d")
                arrow_d2c = energy_arrow(bit_center, cold_point, "d2c")
                arrow_h2d.set_color(RES_HOT_COLOR)
                arrow_d2c.set_color(RES_COLD_COLOR)
                energy_arrows.append(arrow_h2d)
                energy_arrows.append(arrow_d2c)
            elif initial_joint == "1d" and final_joint_state == "0u":
                # 1d -> 0u: intrinsic demon transition with bit flip (HOT bath AND COLD bath, energy taken from hot, dumped to cold, at once)
                count_10 += 1
                arrow_h2d = energy_arrow(hot_point, demon_center, "h2d")
                arrow_d2c = energy_arrow(bit_center, cold_point, "d2c")
                arrow_h2d.set_color(RES_HOT_COLOR)
                arrow_d2c.set_color(RES_COLD_COLOR)
                energy_arrows.append(arrow_h2d)
                energy_arrows.append(arrow_d2c)

            elif demon_changed and not flip:
                # Intrinsic demon transition with same bit (HOT bath only)
                if demon_only == 'd' and demon_out == 'u':
                    # d -> u: energy from HOT bath
                    arrow = energy_arrow(hot_point, demon_center, "h2d")
                    arrow.set_color(RES_HOT_COLOR)
                    energy_arrows.append(arrow)
                elif demon_only == 'u' and demon_out == 'd':
                    # u -> d: energy to HOT bath
                    arrow = energy_arrow(demon_center, hot_point, "d2h")
                    arrow.set_color(RES_HOT_COLOR)
                    energy_arrows.append(arrow)
            
            # Animate transitions
            animations = []
            
            # Animate bit flip in-place if needed
            if flip:
                # Create new bit with flipped value at same position
                new_bit = make_bit(bit_out).move_to(bit_mob.get_center())
                animations.append(Transform(bit_mob, new_bit))
            
            # Add energy arrows
            if energy_arrows:
                for arrow in energy_arrows:
                    animations.append(GrowArrow(arrow))
            if demon_out != current_demon_state:
                # Create a fresh demon object with the new state for proper transformation
                new_demon = make_demon(demon_out).move_to(demon.get_center())
                animations.append(Transform(demon, new_demon))
                #self.play(Transform(demon, new_demon), run_time=0.25)
                
            if animations:
                self.wait(t_interact * 0.2)
                self.play(AnimationGroup(*animations), run_time=t_interact * 0.8)
                
                # Hold arrows briefly, then fade them
                if energy_arrows:
                    self.wait(0.2)
                    self.play(*[FadeOut(arrow) for arrow in energy_arrows], run_time=0.3)
            else:
                self.wait(t_interact * 0.2)

            # Update demon state if changed
            
            
            # Update state tracking
            current_demon_state = demon_out
            current_joint_state = f"{bit_out}{demon_out}"

            # Move bit out
            self.play(bit_mob.animate.move_to([x_right, y_tape, 0]), run_time=t_move_out)
            self.play(FadeOut(bit_mob), run_time=0.1)

            # Update counters
            n_proc += 1
            if bit_out == 0:
                out_count_0 += 1
            else:
                out_count_1 += 1

            vt_n.set_value(n_proc)
            vt_01.set_value(count_01)
            vt_10.set_value(count_10)

            # Q_c->h_total = (count_01 - count_10) * ΔE
            q_total = (count_01 - count_10) * P.DeltaE
            vt_q.set_value(q_total)

            # Running δ_out based on observed outgoing bits
            delta_out = (out_count_0 - out_count_1) / max(1, (out_count_0 + out_count_1))
            vt_delta_out.set_value(delta_out)

            self.wait(0.1)

        # End card

        end_text = Text("Simulation complete", weight="BOLD").scale(0.7).next_to(scoreboard, DOWN).set_color(WHITE)
        self.play(FadeIn(end_text, shift=UP*0.2), run_time=0.6)
        self.wait(0.6)
        self.play(FadeOut(end_text), run_time=0.4)