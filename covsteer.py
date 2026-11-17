"""
Covariance steering for stochastic vehicle path planning.

Reproduction of K. Okamoto and P. Tsiotras, "Optimal Stochastic Vehicle Path
Planning Using Covariance Steering", IEEE RA-L 4(3), 2019.

Blocks, in the order the method is built up:
     1. Problem data
     2. Vehicle model                          Eq. 1
     3. Lifted dynamics                        Eq. 7
     4. Policy: mean and covariance            Theorem 1, Eqs. 12-16
     5. Open loop (K = 0)                      Sec. I
     6. Optimal covariance steering            Eqs. 17-19
     7. Chance constraints                     Eq. 20
     8. Obstacles: region selection            Eqs. 22-25
     9. Algorithm 1: mixed-integer solve
    10. Monte Carlo verification
    11. Plots
    12. main(): run everything

Run:
    python covsteer.py            # whole pipeline, figures pop up (about 3 min with MOSEK)
    python covsteer.py --sweep    # also redo the slit-width sweep (about 12 min)

Algorithm 1 needs MOSEK (license at ~/mosek/mosek.lic). Without it, the route
MOSEK chose is read from results/routes.json and the rest, which is convex once
the route is fixed, is solved with Clarabel.
results/ holds the two inputs we read: routes.json and width_sweep.json.

# ASSUMED = not given in the paper.  # RECONSTRUCTED = read off its figures.
"""

import argparse
import json
import time
import warnings
from functools import lru_cache
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import norm

RESULTS = Path(__file__).resolve().parent / "results"
warnings.filterwarnings("ignore", message="Argument sub")   # harmless MOSEK notice

# =============================================================================
# 1. PROBLEM DATA
# =============================================================================
dt, N = 0.2, 20          # 20 steps of 0.2 s = 4 s
nx, nu = 4, 2            # state [p_x, p_y, v_x, v_y], control [a_x, a_y]

mu_0 = np.array([-10.0, 0.1, 0.0, 0.0])           # start  x_0 ~ N(mu_0, Sigma_0)   ASSUMED
Sigma_0 = np.diag([0.05, 0.05, 0.001, 0.001])     # ASSUMED
mu_N = np.zeros(nx)                                # goal   x_N ~ N(mu_N, Sigma_N)   ASSUMED
Sigma_N = np.diag([0.01, 0.01, 0.001, 0.001])     # ASSUMED
V_MAX, K_MAX = 100.0, 10.0                         # bounds on V and K, from the paper

# =============================================================================
# 2. VEHICLE MODEL   x_{k+1} = A x_k + B u_k + D w_k,   w_k ~ N(0, I)       Eq. 1
# =============================================================================
I2, Z2 = np.eye(2), np.zeros((2, 2))
A = np.block([[I2, dt * I2],
              [Z2, I2]])               # position += dt * velocity
B = np.vstack([0.5 * dt**2 * I2,
               dt * I2])               # the control is the acceleration
D = 1e-2 * np.eye(nx)                  # noise scale, from the paper

# =============================================================================
# 3. LIFTED DYNAMICS   X = A_cal x_0 + B_cal U + D_cal W                     Eq. 7
# =============================================================================
# The whole trajectory X = [x_0; ...; x_N] is ONE linear function of the start,
# all controls U and all noise W, so we can optimize all controls at once.


def blk(k):
    """Rows of x_k inside the stacked trajectory X."""
    return slice(k * nx, (k + 1) * nx)


def lift():
    """Build A_cal, B_cal, D_cal by unrolling Eq. 1."""
    A_pow = [np.linalg.matrix_power(A, k) for k in range(N + 1)]
    A_cal = np.vstack(A_pow)
    B_cal = np.zeros(((N + 1) * nx, N * nu))
    D_cal = np.zeros(((N + 1) * nx, N * nx))
    for k in range(1, N + 1):          # x_k depends on u_j and w_j for j < k only
        for j in range(k):
            B_cal[blk(k), j * nu:(j + 1) * nu] = A_pow[k - 1 - j] @ B
            D_cal[blk(k), blk(j)] = A_pow[k - 1 - j] @ D
    return A_cal, B_cal, D_cal


A_cal, B_cal, D_cal = lift()

# =============================================================================
# 4. POLICY: MEAN AND COVARIANCE   (Theorem 1, Eqs. 12-16)
# =============================================================================
# Policy:  u_k = v_k + K_k y_k,   y_{k+1} = A y_k + D w_k,   y_0 = x_0 - mu_0
# y_k is the deviation the noise alone would cause. It does not depend on K, so
#     mean        X_bar   = A_cal mu_0 + B_cal V            (Eq. 15)  set by V only
#     covariance  Sigma_X = S S^T,  S = (I + B_cal K) L     (Eq. 16)  set by K only
# Both are LINEAR in the unknowns (V, K), so the problem stays convex.

# Square root of the noise-only covariance:  L L^T = A_cal Sigma_0 A_cal^T + D_cal D_cal^T
L = np.hstack([A_cal @ np.linalg.cholesky(Sigma_0), D_cal])


def policy(feedback=True):
    """Unknowns V, K and the mean and covariance they produce, as cvxpy expressions.
    feedback=False fixes K = 0 (open loop, "mean steering")."""
    V = cp.Variable(N * nu)
    if feedback:
        K = [cp.Variable((nu, nx)) for _ in range(N)]
        KL = cp.vstack([K[k] @ L[blk(k)] for k in range(N)])    # K times L, block by block
    else:
        K = None
        KL = np.zeros((N * nu, L.shape[1]))
    X_bar = A_cal @ mu_0 + B_cal @ V                             # Eq. 15
    S = L + B_cal @ KL                                           # Eq. 16
    return dict(V=V, K=K, KL=KL, X_bar=X_bar, S=S)


def solution(V, K=None, route=None):
    """The same formulas with numbers: mean and covariance at every step, and the cost."""
    if K is None:
        K = np.zeros((N, nu, nx))
    K = np.asarray(K)
    KL = np.vstack([K[k] @ L[blk(k)] for k in range(N)])
    S = L + B_cal @ KL
    sol = dict(V=V, K=K, route=route,
               mus=(A_cal @ mu_0 + B_cal @ V).reshape(N + 1, nx),
               Sigmas=np.array([S[blk(k)] @ S[blk(k)].T for k in range(N + 1)]),
               J_mean=float(V @ V),
               J_cov=float(np.sum(KL**2)))
    sol["J"] = sol["J_mean"] + sol["J_cov"]
    if route is not None:
        sol["slit"] = "upper" if route[2].any() else "lower"    # region R3 = upper slit
    return sol


# =============================================================================
# 5. OPEN LOOP (K = 0)
# =============================================================================
# With K = 0, S = L: the covariance is L L^T no matter what V is.
# The controls move the mean, but nothing can shrink the uncertainty.


def open_loop():
    """Least-energy controls V that bring the MEAN to the goal, with no feedback."""
    V = np.linalg.pinv(B_cal[blk(N)]) @ (mu_N - (A_cal @ mu_0)[blk(N)])
    return solution(V)


# =============================================================================
# 6. OPTIMAL COVARIANCE STEERING (closed loop)   Eqs. 17-19
# =============================================================================
# Cost (Eq. 17): expected control energy E[sum u_k^T u_k] = ||V||^2 + ||K L||_F^2
#     (Q = 0, R = I; ASSUMED, the paper does not give them)
# Terminal mean (Eq. 18a):        E_N X_bar = mu_N
# Terminal covariance (Eq. 19):   Sigma_xN <= Sigma_N  <=>  ||Sigma_N^{-1/2} S_N||_2 <= 1
Sigma_N_inv_half = np.diag(1 / np.sqrt(np.diag(Sigma_N)))


def build(feedback=True, eq19=True):
    """Policy (block 4) + cost + terminal constraints. eq19=False leaves Eq. 19 out."""
    P = policy(feedback)
    P["J"] = cp.sum_squares(P["V"]) + cp.sum_squares(P["KL"])            # Eq. 17
    P["cons"] = [P["X_bar"][blk(N)] == mu_N,                              # Eq. 18a
                 cp.abs(P["V"]) <= V_MAX]
    if feedback:
        P["cons"] += [cp.abs(K_k) <= K_MAX for K_k in P["K"]]
        if eq19:
            P["cons"].append(cp.sigma_max(Sigma_N_inv_half @ P["S"][blk(N)]) <= 1)  # Eq. 19
    return P


@lru_cache
def has_mosek():
    """True if MOSEK is installed and licensed."""
    try:
        cp.Problem(cp.Minimize(cp.square(cp.Variable() - 1))).solve(solver=cp.MOSEK)
        return True
    except Exception:
        return False


def solve_convex(feedback=True, geom=None, route=None):
    """Solve when the problem is convex: no obstacles, or obstacles with a fixed route.
    Uses the open-source Clarabel solver (MOSEK is only needed for the mixed-integer step)."""
    P = build(feedback)
    cons = P["cons"]
    if geom is not None:
        cons = cons + region_constraints(P, geom, route)
    prob = cp.Problem(cp.Minimize(P["J"]), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:          # Clarabel sometimes errors instead of "infeasible"
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    K = [K_k.value for K_k in P["K"]] if feedback else None
    return solution(P["V"].value, K, route)


# =============================================================================
# 7. CHANCE CONSTRAINTS   Eq. 20
# =============================================================================
# The noise is Gaussian (unbounded), so we ask each face a^T x <= b to hold
# with probability at least 1 - eps. For a Gaussian this is exact:
#     Pr(a^T x_k > b) <= eps   <=>   a^T mu_k + PHI_INV * std(a^T x_k) <= b
# i.e. keep the mean PHI_INV = 3.09 standard deviations away from the face.
# std(a^T x_k) = ||a^T S_k|| is a norm of something linear: a convex cone.
eps = 1e-3                       # allowed violation probability per face
PHI_INV = norm.ppf(1 - eps)      # = 3.09


def chance_margins(P):
    """margin[i, k] >= 3.09 * std of p_x (i = 0) or p_y (i = 1) at step k."""
    margin = cp.Variable((2, N + 1))
    cons = [margin[i] >= PHI_INV * cp.norm(P["S"][i::nx], 2, axis=1) for i in (0, 1)]
    return margin, cons


# =============================================================================
# 8. OBSTACLES: REGION SELECTION   Eqs. 22-25
# =============================================================================
# The free space is not convex (pass above OR below), so cover it with
# overlapping boxes R1..R4 (Eqs. 22-23). Binary M[r, k] = 1 means x_k and
# x_{k+1} are both in box r, so the whole segment is. One box per step (Eq. 24).
# "If M[r, k] = 1, the chance constraints of box r hold" is big-M (Eq. 25):
#     a^T mu + margin - b <= BIG_M * (1 - M[r, k])
WORKSPACE = (-12.0, 2.0, -5.0, 5.0)    # map edges (x_min, x_max, y_min, y_max)
BIG_M = 30.0                           # larger than any left-hand side on this map


def double_slit(top_gap=0.5):
    """Wall at x in [-6, -4] with two slits. RECONSTRUCTED from the paper's Figs. 2-4, 6."""
    top = 1.0 + top_gap
    return dict(top_gap=top_gap,
                obstacles=[(-6, -4, top, 5), (-6, -4, -1.4, 1.0), (-6, -4, -5, -3)],
                regions=[(-12, -6, -5, 5),       # R1: left of the wall
                         (-12, 2, -3, -1.4),     # R2: band through the lower slit (1.6 wide)
                         (-12, 2, 1.0, top),     # R3: band through the upper slit
                         (-4, 2, -5, 5)])        # R4: right of the wall


def faces(box):
    """The 4 faces of a box as (coordinate i, sign, b), meaning  sign * p_i <= b."""
    x_min, x_max, y_min, y_max = box
    return [(0, -1, -x_min), (0, 1, x_max), (1, -1, -y_min), (1, 1, y_max)]


def region_constraints(P, geom, M, forbid=()):
    """Chance constraints (block 7) inside the selected boxes.
    M is a boolean variable (solver picks the route) or a 0/1 array (route fixed)."""
    mean = [P["X_bar"][0::nx], P["X_bar"][1::nx]]      # mean p_x and p_y, k = 0..N
    margin, cons = chance_margins(P)

    for i, sign, b in faces(WORKSPACE):                # always stay on the map
        cons.append(sign * mean[i] + margin[i] <= b)

    if isinstance(M, cp.Variable):
        cons.append(cp.sum(M, axis=0) == 1)            # Eq. 24: one box per step
        cons += [M[r] == 0 for r in forbid]

    for r, box in enumerate(geom["regions"]):
        for i, sign, b in faces(box):
            for k in (slice(0, N), slice(1, N + 1)):   # both x_k and x_{k+1}
                cons.append(sign * mean[i][k] + margin[i][k] - b <= BIG_M * (1 - M[r]))  # Eq. 25
    return cons


# =============================================================================
# 9. ALGORITHM 1: MIXED-INTEGER SOLVE
# =============================================================================
# MOSEK solves mixed-integer SOCPs but not mixed-integer SDPs, and Eq. 19 is an
# SDP constraint. We use:   ||G||_2 <= 1   <=>   ||z^T G|| <= 1 for every unit z.
#   Lower bound: keep only a few directions z (a relaxation), solve the MISOCP.
#   Upper bound: fix the route it chose, solve with the exact Eq. 19 (convex).
#   If the bounds agree the route is certified optimal; else add the worst z, repeat.
MOSEK_OPTS = {"MSK_DPAR_MIO_MAX_TIME": 600.0, "MSK_DPAR_MIO_TOL_REL_GAP": 1e-6}

# Starting directions: the 4 axes and the 12 diagonals (e_i +- e_j) / sqrt(2)
CUT_DIRS = list(np.eye(nx))
for i in range(nx):
    for j in range(i + 1, nx):
        CUT_DIRS.append((np.eye(nx)[i] + np.eye(nx)[j]) / np.sqrt(2))
        CUT_DIRS.append((np.eye(nx)[i] - np.eye(nx)[j]) / np.sqrt(2))


def algorithm1(geom, feedback=True, forbid=(), tol=1e-4, max_iter=20):
    """Optimal route and policy through the regions of geom. Needs MOSEK. None if infeasible."""
    P = build(feedback, eq19=False)
    M = cp.Variable((len(geom["regions"]), N), boolean=True)
    cons = P["cons"] + region_constraints(P, geom, M, forbid)
    G = Sigma_N_inv_half @ P["S"][blk(N)]              # Eq. 19 is ||G||_2 <= 1
    cuts = list(CUT_DIRS) if feedback else []

    for _ in range(max_iter):
        relaxed = cp.Problem(cp.Minimize(P["J"]), cons + [cp.norm(z @ G) <= 1 for z in cuts])
        relaxed.solve(solver=cp.MOSEK, mosek_params=MOSEK_OPTS)
        if relaxed.status not in ("optimal", "optimal_inaccurate"):
            return None
        route = np.round(M.value).astype(int)
        if not feedback:                               # no Eq. 19: the MISOCP is exact
            return solution(P["V"].value, None, route)

        sol = solve_convex(True, geom, route)          # exact Eq. 19 on this route
        if sol is not None:
            sol["gap"] = (sol["J"] - relaxed.value) / sol["J"]
            if sol["gap"] <= tol:                      # bounds agree: certified optimal
                return sol
        cuts.append(np.linalg.svd(G.value)[0][:, 0])   # most violated direction
    raise RuntimeError("Algorithm 1: no optimality certificate after max_iter iterations")


def route_str(route):
    """Box chosen at each step, e.g. '11111111333344444444' = R1 x 8, R3 x 4, R4 x 8."""
    return "".join(str(r + 1) for r in route.argmax(axis=0))


def route_from_str(text):
    route = np.zeros((4, N), int)
    for k, c in enumerate(text):
        route[int(c) - 1, k] = 1
    return route


def plan(geom, feedback=True):
    """Run Algorithm 1 (MOSEK). Without MOSEK, reuse the saved route and solve the rest."""
    start = time.time()
    if has_mosek():
        sol = algorithm1(geom, feedback)
    else:
        routes = json.loads((RESULTS / "routes.json").read_text())
        name = f"slit{geom['top_gap']}_{'cov' if feedback else 'mean'}"
        sol = solve_convex(feedback, geom, route_from_str(routes[name]))
    sol["time"] = time.time() - start
    return sol


# =============================================================================
# 10. MONTE CARLO VERIFICATION
# =============================================================================
def simulate(sol, n=20_000, seed=0):
    """Drive n noisy vehicles with the policy u_k = v_k + K_k y_k. Returns states and controls."""
    rng = np.random.default_rng(seed)
    x = mu_0 + rng.standard_normal((n, nx)) @ np.linalg.cholesky(Sigma_0).T
    y = x - mu_0
    xs, us = [x], []
    for k in range(N):
        u = sol["V"][k * nu:(k + 1) * nu] + y @ sol["K"][k].T      # Eq. 12
        w = rng.standard_normal((n, nx))
        x = x @ A.T + u @ B.T + w @ D.T                            # Eq. 1
        y = y @ A.T + w @ D.T                                      # Eq. 13
        xs.append(x)
        us.append(u)
    return np.array(xs), np.array(us)


def collides(p, geom):
    """True where position p (n x 2) is inside an obstacle or off the map."""
    x, y = p[:, 0], p[:, 1]
    x_min, x_max, y_min, y_max = WORKSPACE
    hit = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
    for a0, a1, b0, b1 in geom["obstacles"]:
        hit |= (x > a0) & (x < a1) & (y > b0) & (y < b1)
    return hit


def verify(sol, geom, n=200_000):
    """Simulate n runs and compare with what the optimizer promised."""
    xs, us = simulate(sol, n, seed=8)

    # how often each face of the chosen boxes is crossed (should be about eps)
    worst_face = 0.0
    for k in range(N):
        box = geom["regions"][sol["route"][:, k].argmax()]
        for i, sign, b in faces(box):
            for x in (xs[k], xs[k + 1]):
                worst_face = max(worst_face, np.mean(sign * x[:, i] > b))

    # collisions at the time steps, and along the true path between them
    # (acceleration is constant between steps, so the path is a parabola)
    hit_steps = np.any([collides(x[:, :2], geom) for x in xs], axis=0)
    hit_path = hit_steps.copy()
    for k in range(N):
        for t in np.linspace(0, dt, 11)[1:-1]:
            p = xs[k, :, :2] + xs[k, :, 2:] * t + 0.5 * us[k] * t**2
            hit_path |= collides(p, geom)

    cov_excess = np.linalg.eigvalsh(sol["Sigmas"][-1] - Sigma_N).max()
    return {"route": sol["slit"] + " slit",
            "J, optimizer": f"{sol['J']:.2f}",
            "J, simulated": f"{np.sum(us**2, axis=(0, 2)).mean():.2f}",
            "terminal covariance excess (<= 0 is met)": f"{cov_excess:+.0e}",
            f"worst face violation (allowed {eps})": f"{worst_face:.4f}",
            "collisions at time steps": f"{hit_steps.mean():.4f}",
            "collisions along the path": f"{hit_path.mean():.4f}"}


def table(columns):
    """Markdown table from {column name: verify() result}."""
    names = list(columns)
    lines = ["| | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for row in columns[names[0]]:
        lines.append(f"| {row} | " + " | ".join(columns[c][row] for c in names) + " |")
    return "\n".join(lines)


# =============================================================================
# 11. PLOTS
# =============================================================================
# Colors of the slide deck
BLUE = "#2B5F9E"        # covariance steering
BURNT = "#A64B14"       # mean steering (K = 0)
ACCENT = "#FA6400"      # start and goal
NAVY = "#14213D"        # obstacles
GRAY = "#A0A0A0"        # simulated runs
PAPER = "#FDFCF9"       # figure background


def ellipse(mean, cov, n_std=3):
    """Points on the 3-std ellipse of a 2-D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 80)
    circle = np.vstack([np.cos(t), np.sin(t)])
    return mean[:, None] + n_std * vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * circle)


def draw(ax, sol=None, geom=None):
    """Obstacles (navy), 60 simulated runs (gray), 3-std ellipses and mean path
    (blue: covariance steering, burnt orange: mean steering), start and goal (orange)."""
    ax.set_facecolor(PAPER)
    if geom is not None:
        for x0, x1, y0, y1 in geom["obstacles"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=NAVY, lw=0))
        ax.set_xlim(-12, 1)
        ax.set_ylim(-5, 5)
    if sol is not None:
        color = BLUE if np.any(sol["K"]) else BURNT
        xs, _ = simulate(sol, n=60, seed=7)
        ax.plot(xs[:, :, 0], xs[:, :, 1], color=GRAY, lw=0.4)
        for mean, cov in zip(sol["mus"], sol["Sigmas"]):
            ax.plot(*ellipse(mean[:2], cov[:2, :2]), color=color, lw=0.8)
        ax.plot(sol["mus"][:, 0], sol["mus"][:, 1], ".-", color=color, lw=1.5)
    ax.plot(*ellipse(mu_0[:2], Sigma_0[:2, :2]), color=ACCENT, lw=2.5)
    ax.plot(*ellipse(mu_N[:2], Sigma_N[:2, :2]), color=ACCENT, lw=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def fig_panels(panels, geom=None, width=11.5):
    """Side-by-side plots from a list of (title, solution)."""
    height = 4.6 if geom else 2.8
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), layout="constrained",
                             squeeze=False, facecolor=PAPER)
    for ax, (title, sol) in zip(axes[0], panels):
        draw(ax, sol, geom)
        ax.set_title(title)
        if geom is None:
            ax.set_ylim(-1.5, 1.5)
    return fig


def fig_regions(geom):
    """The four boxes that cover the free space (paper Fig. 3)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained", facecolor=PAPER)
    for r, (ax, box) in enumerate(zip(axes.flat, geom["regions"])):
        x0, x1, y0, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=BLUE, alpha=0.14, lw=0))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=BLUE,
                               lw=1.5, ls="--"))
        draw(ax, geom=geom)
        ax.set_title(f"R{r + 1}")
    return fig


def fig_sweep(data=None):
    """Cost of going through the upper vs. the lower slit, against upper-slit width.
    Uses the saved sweep (results/width_sweep.json) unless data from sweep() is given."""
    if data is None:
        data = json.loads((RESULTS / "width_sweep.json").read_text())
    upper = [r for r in data["upper"] if r["feasible"]]
    infeasible = [r["width"] for r in data["upper"] if not r["feasible"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained", facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.plot([r["width"] for r in upper], [r["J"] for r in upper], "o-", color=BLUE,
            label="forced through upper slit")
    for r in upper:
        ax.annotate(r["pairs_in_slit"], (r["width"], r["J"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=BLUE)
    ax.axhline(data["lower"]["J"], color=ACCENT, ls="--", label="forced through lower slit")
    if infeasible:
        ax.axvspan(0.14, max(infeasible) + 0.0125, color="0.85", label="upper slit infeasible")
    ax.set_yscale("log")
    ax.set_xlabel("upper slit width")
    ax.set_ylabel("optimal cost J (log scale)")
    ax.set_title("numbers: time steps spent in the upper slit")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    return fig


def sweep(widths=(0.5, 0.45, 0.4, 0.35, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15)):
    """Force each route for a range of upper-slit widths (MOSEK). Returns the data for fig_sweep."""
    lower = algorithm1(double_slit(0.5), forbid=(2,))      # forbid R3; same for every width
    rows = []
    for w in widths:
        upper = algorithm1(double_slit(w), forbid=(1,))    # forbid R2
        row = dict(width=w, feasible=upper is not None)
        if upper is not None:
            row.update(J=upper["J"], J_mean=upper["J_mean"], J_cov=upper["J_cov"],
                       pairs_in_slit=int(upper["route"][2].sum()))
        rows.append(row)
        print(f"   width {w}: upper J = {row.get('J', 'infeasible')}")
    return dict(lower=dict(J=lower["J"], J_mean=lower["J_mean"], J_cov=lower["J_cov"]),
                upper=rows)


# =============================================================================
# 12. RUN THE PIPELINE
# =============================================================================
def show():
    """Display the figures made so far and keep going."""
    plt.show(block=False)
    plt.pause(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="redo the slit-width sweep")
    args = parser.parse_args()
    print("Solver:", "MOSEK" if has_mosek() else "Clarabel, with saved routes (no MOSEK)")

    print("\n1. No obstacles: open loop vs. optimal covariance steering")
    ol = open_loop()
    cl = solve_convex(feedback=True)
    for name, sol in [("open loop", ol), ("covariance steering", cl)]:
        print(f"   {name:20s} J = {sol['J']:.2f}   final variance {sol['Sigmas'][-1][0, 0]:.3f}"
              f"   (target {Sigma_N[0, 0]})")
    fig_panels([("open loop (K = 0)", ol), ("optimal covariance steering", cl)])
    fig_regions(double_slit(0.5))
    show()

    sols = {}
    for step, width in [(2, 0.5), (3, 0.2)]:
        print(f"\n{step}. Double slit, upper slit {width} wide")
        panels = []
        for feedback, name in [(False, "mean steering"), (True, "covariance steering")]:
            sol = plan(double_slit(width), feedback)
            sols[width, feedback] = sol
            print(f"   {name:20s} {sol['slit']} slit, J = {sol['J']:.2f}   ({sol['time']:.0f} s)")
            panels.append((f"{name}: {sol['slit']} slit, J = {sol['J']:.1f}", sol))
        fig_panels(panels, double_slit(width))
        show()

    fig_sweep(sweep() if args.sweep else None)

    print("\n4. Monte Carlo check, 200,000 runs per solution\n")
    columns = {}
    for (width, feedback), sol in sols.items():
        name = f"slit {width}, {'covariance' if feedback else 'mean'}"
        columns[name] = verify(sol, double_slit(width))
    print(table(columns))
    plt.show()


if __name__ == "__main__":
    main()
