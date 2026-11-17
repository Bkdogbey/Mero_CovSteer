"""
Mean steering vs. covariance steering: a classroom demo
=======================================================

A vehicle must drive from the left to the goal on the right in 4 seconds.
Random disturbances push it around, so we don't know exactly where it is.
A wall with two gaps (slits) is in the way:

    * the UPPER slit is short and direct, but NARROW (0.5 wide)
    * the LOWER slit is WIDE (1.6), but a detour

Two planners, both required to stay collision-free with 99.9% probability:

    MEAN STEERING        plans only the average path. The uncertainty
                         is never controlled, so it keeps growing.
    COVARIANCE STEERING  plans the average path AND uses feedback to shrink
                         the uncertainty where it matters.
                         (Okamoto & Tsiotras, IEEE RA-L 2019)

Run:
    python covsteer_demo.py           # play the animation in a window
    python covsteer_demo.py --save    # also save covsteer_demo.gif

Needs: numpy, scipy, matplotlib, cvxpy (its built-in Clarabel solver is enough).

Simplification vs. the paper: the paper lets a mixed-integer solver choose the
slit AND the timing. Here we fix the timing (the wall is crossed between steps
8 and 12, which is what the full solver chose) and simply try both slits.
"""

import argparse

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from scipy.stats import norm

# =============================================================================
# 1. SETTINGS
# =============================================================================
DT, N = 0.2, 20                        # 20 steps of 0.2 s = 4 seconds

MU_START = np.array([-10.0, 0.1, 0.0, 0.0])           # [x, y, vx, vy]
SIGMA_START = np.diag([0.05, 0.05, 0.001, 0.001])     # start uncertainty
MU_GOAL = np.zeros(4)
SIGMA_GOAL = np.diag([0.01, 0.01, 0.001, 0.001])      # required final uncertainty

RISK = 1e-3                            # allowed chance of crossing each wall face
MARGIN = norm.ppf(1 - RISK)            # = 3.09 -> keep 3.09 sigma from every wall

OBSTACLES = [(-6, -4, 1.5, 5),         # (x_min, x_max, y_min, y_max)
             (-6, -4, -1.4, 1.0),
             (-6, -4, -5, -3)]
SLITS = {"upper": (1.0, 1.5),          # y-range of each gap
         "lower": (-3.0, -1.4)}
ENTER, EXIT = 8, 12                    # steps during which the wall is crossed

# =============================================================================
# 2. VEHICLE MODEL: we choose the acceleration u, noise w is added each step
#       x_{k+1} = A x_k + B u_k + D w_k
# =============================================================================
I2, Z2 = np.eye(2), np.zeros((2, 2))
A = np.block([[I2, DT * I2], [Z2, I2]])
B = np.vstack([0.5 * DT**2 * I2, DT * I2])
D = 0.01 * np.eye(4)

# All randomness in one vector xi = [z, w_0, ..., w_{N-1}], each part ~ N(0, I).
# The start is x_0 = MU_START + chol(SIGMA_START) z.
# Every random deviation below is (matrix) @ xi, so its covariance is
# (matrix) @ (matrix)^T. That keeps all the math linear.
N_XI = 4 + 4 * N
L0 = np.linalg.cholesky(SIGMA_START)


def noise_w(k):
    """Matrix that picks w_k out of xi."""
    S = np.zeros((4, N_XI))
    S[:, 4 + 4 * k: 8 + 4 * k] = np.eye(4)
    return S


# =============================================================================
# 3. THE PLANNER
# =============================================================================
def plan(slit, feedback):
    """Cheapest plan through the given slit, or None if it can't be done safely.

    Control:  u_k = v_k + K_k y_k
        v_k  planned acceleration         -> moves the AVERAGE position
        K_k  feedback gain                -> shrinks the SPREAD
        y_k  deviation the noise alone would cause (the paper's trick:
             it doesn't depend on K, so the problem stays convex)
    Mean steering is the same with K_k = 0.
    """
    v = cp.Variable((N, 2))
    K = [cp.Variable((2, 4)) for _ in range(N)] if feedback else [np.zeros((2, 4))] * N

    mean = [MU_START]                                    # average state
    Y = [np.hstack([L0, np.zeros((4, 4 * N))])]          # y_k = Y[k] @ xi
    T = [Y[0]]                                           # x_k - mean_k = T[k] @ xi
    for k in range(N):
        mean.append(A @ mean[k] + B @ v[k])
        Y.append(A @ Y[k] + D @ noise_w(k))
        T.append(A @ T[k] + B @ (K[k] @ Y[k]) + D @ noise_w(k))

    # Expected control energy = planned part + feedback part
    cost = cp.sum_squares(v) + sum(cp.sum_squares(K[k] @ Y[k]) for k in range(N))

    cons = [mean[N] == MU_GOAL]                          # arrive at the goal on average
    if feedback:                                         # ...with at most the goal spread
        cons.append(cp.sigma_max(np.diag(1 / np.sqrt(np.diag(SIGMA_GOAL))) @ T[N]) <= 1)

    def stay_inside(k, x_min, x_max, y_min, y_max):
        """Chance constraint: the average must sit MARGIN std's inside every face."""
        sx, sy = cp.norm(T[k][0, :]), cp.norm(T[k][1, :])   # std of x and y
        cons.extend([mean[k][0] + MARGIN * sx <= x_max, -mean[k][0] + MARGIN * sx <= -x_min,
                     mean[k][1] + MARGIN * sy <= y_max, -mean[k][1] + MARGIN * sy <= -y_min])

    y_lo, y_hi = SLITS[slit]
    for k in range(N + 1):
        if k <= ENTER:
            stay_inside(k, -12, -6, -5, 5)               # left of the wall
        if ENTER <= k <= EXIT:
            stay_inside(k, -12, 2, y_lo, y_hi)           # in line with the slit
        if k >= EXIT:
            stay_inside(k, -4, 2, -5, 5)                 # right of the wall

    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != "optimal":
        return None
    T_val = [t.value if isinstance(t, cp.Expression) else t for t in T]
    return dict(slit=slit, cost=prob.value, v=v.value,
                K=[k.value if feedback else k for k in K],
                mean=np.array([m if isinstance(m, np.ndarray) else m.value for m in mean]),
                cov=[t @ t.T for t in T_val])


def best_plan(feedback):
    """Try both slits, keep the cheaper safe one."""
    name = "covariance steering" if feedback else "mean steering"
    options = {}
    for slit in SLITS:
        p = plan(slit, feedback)
        print(f"  {name:20s} via {slit} slit: " + ("NOT POSSIBLE safely" if p is None
                                                  else f"cost J = {p['cost']:.1f}"))
        if p:
            options[slit] = p
    return min(options.values(), key=lambda p: p["cost"])


# =============================================================================
# 4. SIMULATE MANY VEHICLES (same random disturbances for both planners)
# =============================================================================
def simulate(p, n_vehicles, seed=0):
    """Positions, velocities and controls of n_vehicles under plan p."""
    rng = np.random.default_rng(seed)
    x = MU_START + rng.normal(size=(n_vehicles, 4)) @ L0.T
    y = x - MU_START
    xs, us = [x], []
    for k in range(N):
        u = p["v"][k] + y @ p["K"][k].T                  # u_k = v_k + K_k y_k
        w = rng.normal(size=(n_vehicles, 4))
        x = x @ A.T + u @ B.T + w @ D.T
        y = y @ A.T + w @ D.T
        xs.append(x)
        us.append(u)
    return np.array(xs), np.array(us)


def position_at(x_k, u_k, s):
    """Exact position s seconds after step k (acceleration held constant)."""
    return x_k[..., :2] + x_k[..., 2:] * s + 0.5 * u_k * s**2


def ellipse(center, cov2, n_sigma=3):
    """Points of the 3-sigma ellipse (holds ~99% of the vehicles)."""
    vals, vecs = np.linalg.eigh(cov2)
    t = np.linspace(0, 2 * np.pi, 80)
    pts = vecs @ (n_sigma * np.sqrt(np.maximum(vals, 0))[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    return center[0] + pts[0], center[1] + pts[1]


def hits_obstacle(pos):
    hit = np.zeros(len(pos), bool)
    for x0, x1, y0, y1 in OBSTACLES:
        hit |= (pos[:, 0] > x0) & (pos[:, 0] < x1) & (pos[:, 1] > y0) & (pos[:, 1] < y1)
    return hit


# =============================================================================
# 5. ANIMATION
# =============================================================================
# Colors of the slide deck
BLUE = "#2B5F9E"        # covariance steering
BURNT = "#A64B14"       # mean steering
ACCENT = "#FA6400"      # start, goal, crashes
NAVY = "#14213D"        # obstacles
GRAY = "0.45"           # vehicles
PAPER = "#FDFCF9"       # background
SUB = 8                       # animation frames per time step
HOLD = 45                     # frames to hold the final picture
N_VEHICLES = 300


def animate(plans, save):
    sims = [simulate(p, N_VEHICLES) for p in plans]
    saving = 1 - plans[1]["cost"] / plans[0]["cost"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=PAPER)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.80, bottom=0.10, wspace=0.08)
    caption = fig.text(0.5, 0.93, "", ha="center", va="center", fontsize=13, wrap=True)
    titles = ["MEAN STEERING\nplans the average path only",
              "COVARIANCE STEERING\nplans the average path + shrinks the uncertainty"]
    art = []
    for ax, p, title, color in zip(axes, plans, titles, (BURNT, BLUE)):
        ax.set_facecolor(PAPER)
        for x0, x1, y0, y1 in OBSTACLES:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=NAVY))
        ax.text(-5, 2.3, "narrow slit\n(0.5 wide)\n↓", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(-5, -3.6, "↑\nwide slit\n(1.6 wide)", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.plot(*ellipse(MU_START, SIGMA_START[:2, :2]), color=ACCENT, lw=2)
        ax.plot(*ellipse(MU_GOAL, SIGMA_GOAL[:2, :2]), color=ACCENT, lw=2)
        ax.text(-10, -0.95, "start", ha="center", color=ACCENT, fontsize=10)
        ax.text(0, -0.65, "goal", ha="center", color=ACCENT, fontsize=10)
        ax.plot(p["mean"][:, 0], p["mean"][:, 1], ":", color=color, lw=1, alpha=0.5)
        ax.set_xlim(-11.5, 1.2)
        ax.set_ylim(-4.2, 3.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight="bold")
        trail = [ax.plot([], [], color=color, lw=0.6, alpha=0.35)[0] for _ in range(N + 1)]
        dots = ax.scatter([], [], s=5, color=GRAY, zorder=3)
        crashed = ax.scatter([], [], s=30, marker="x", color=ACCENT, zorder=4)
        path, = ax.plot([], [], color=color, lw=2.5, zorder=5)
        ell, = ax.plot([], [], color=color, lw=2.5, zorder=5)
        info = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=10, family="monospace",
                       bbox=dict(fc="white", ec="0.7", alpha=0.9))
        art.append(dict(trail=trail, dots=dots, crashed=crashed, path=path, ell=ell, info=info,
                        ever_hit=np.zeros(N_VEHICLES, bool), path_pts=[]))

    def caption_for(t):
        if t < 0.6:
            return "Both start with the same uncertainty: the orange circle holds ~99% of the possible start positions."
        if t < 1.5:
            return "Left: nothing controls the spread, so the cloud keeps GROWING.   Right: feedback SHRINKS it before the wall."
        if t < 2.5:
            return ("Left: too wide for the narrow slit, so it must detour.   "
                    "Right: small enough to fit the short, narrow slit.")
        if t < 4.0:
            return "Both reach the goal on time. Only feedback also meets the goal's size (small orange circle)."
        return (f"Covariance steering: same safety, on target, with {saving:.0%} less control energy "
                f"(J = {plans[1]['cost']:.0f} vs {plans[0]['cost']:.0f}).")

    def update(frame):
        f = min(frame, N * SUB)
        k, s = min(f // SUB, N - 1), (f - min(f // SUB, N - 1) * SUB) * DT / SUB
        t = k * DT + s if frame <= N * SUB else 4.0 + 1e-9 * frame
        for (xs, us), p, a in zip(sims, plans, art):
            pos = position_at(xs[k], us[k], s)
            a["ever_hit"] |= hits_obstacle(pos)
            a["dots"].set_offsets(pos)
            a["crashed"].set_offsets(pos[a["ever_hit"]] if a["ever_hit"].any() else np.empty((0, 2)))
            # average path (exact) and its 3-sigma ellipse (covariance blended between steps)
            m = p["mean"]
            mean_pos = m[k, :2] + m[k, 2:] * s + 0.5 * p["v"][k] * s**2
            lam = s / DT
            cov = (1 - lam) * p["cov"][k][:2, :2] + lam * p["cov"][k + 1][:2, :2]
            a["ell"].set_data(*ellipse(mean_pos, cov))
            a["path_pts"].append(mean_pos)
            a["path"].set_data(*np.array(a["path_pts"]).T)
            for j in range(k + 1):
                a["trail"][j].set_data(*ellipse(m[j, :2], p["cov"][j][:2, :2]))
            half_width = 3 * np.sqrt(cov[1, 1])
            a["info"].set_text(f"time {min(t, 4.0):3.1f} s\n"
                               f"3-sigma half-width {half_width:4.2f}\n"
                               f"crashes {a['ever_hit'].sum():3d} / {N_VEHICLES}")
        if frame == N * SUB:                     # last step: show final ellipses
            for p, a in zip(plans, art):
                a["trail"][N].set_data(*ellipse(p["mean"][N, :2], p["cov"][N][:2, :2]))
        caption.set_text(caption_for(t))
        return []

    anim = FuncAnimation(fig, update, frames=N * SUB + 1 + HOLD, interval=60, repeat=True)
    if save:
        print("saving covsteer_demo.gif ...")
        anim.save("covsteer_demo.gif", writer=PillowWriter(fps=12), dpi=80)
        print("saved covsteer_demo.gif")
    plt.show()


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--save", action="store_true", help="also save covsteer_demo.gif")
    args = parser.parse_args()

    print("Planning (each planner tries both slits and keeps the cheaper safe one):")
    plans = [best_plan(feedback=False), best_plan(feedback=True)]
    print(f"\n  mean steering       -> {plans[0]['slit']} slit, J = {plans[0]['cost']:.1f}")
    print(f"  covariance steering -> {plans[1]['slit']} slit, J = {plans[1]['cost']:.1f}\n")
    animate(plans, args.save)
