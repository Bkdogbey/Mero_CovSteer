"""
=============================================================================
 Reproduction of:
   K. Okamoto and P. Tsiotras,
   "Optimal Stochastic Vehicle Path Planning Using Covariance Steering",
   IEEE Robotics and Automation Letters, vol. 4, no. 3, 2019.
=============================================================================

How to read this file
---------------------
The file follows the paper from top to bottom. Every equation we use is
labelled with the paper's number, e.g.  "# Eq. (7)".  Variable names copy
the paper's symbols as closely as Python allows:

    paper symbol        code name
    ------------        ---------
    x_k, u_k, w_k       x, u, w
    A_k, B_k, D_k       A, B, D          (time-invariant here)
    script-A            A_cal            (lifted / stacked matrices, Eq. 7)
    script-B            B_cal
    script-D            D_cal
    E_k                 E(k)             (picks x_k out of X)
    mu_0, Sigma_0       mu_0, Sigma_0    (initial distribution, Eq. 2)
    mu_N, Sigma_N       mu_N, Sigma_N    (target distribution, Eq. 3)
    epsilon             eps              (chance-constraint risk, Lemma 1)

Each "STEP" below is one small piece of the method. Run one at a time:

    ~/.conda/envs/covsteer/bin/python covsteer_repro.py --step 1

(Call the env's python directly: on this machine `conda run -n covsteer`
 silently picks up the pdstl env because pdstl is first on PATH.)

Tags used in comments:
    # ASSUMED        a value the paper does not give; we chose it
    # RECONSTRUCTED  geometry read off the paper's figures by eye
"""

import argparse
from pathlib import Path
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")                 # save figures to files, no pop-up window
import matplotlib.pyplot as plt
from scipy.stats import norm as std_normal

FIG_DIR = Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


# =============================================================================
# PROBLEM DATA   (paper Sec. IV-B, our slide 15)
# =============================================================================
dt = 0.2        # time step  Delta t
N = 20          # horizon: time steps k = 0, ..., N
eps = 1e-3      # allowed failure probability per constraint face

nx = 4          # state   x = [p_x, p_y, v_x, v_y]   (position, velocity)
nu = 2          # control u = [a_x, a_y]             (acceleration)
nw = 4          # noise   w ~ N(0, I)

# Initial distribution  x_0 ~ N(mu_0, Sigma_0)                        Eq. (2)
mu_0 = np.array([-10.0, 0.1, 0.0, 0.0])
Sigma_0 = np.diag([0.05, 0.05, 0.001, 0.001])

# Target distribution   x_N ~ N(mu_N, Sigma_N)                        Eq. (3)
mu_N = np.zeros(nx)
Sigma_N = np.diag([0.01, 0.01, 0.001, 0.001])

rng = np.random.default_rng(0)   # fixed seed so results are repeatable


# =============================================================================
# STEP 0 - Check which optimization solver is available
# =============================================================================
# The paper used MOSEK. MOSEK needs a (free, academic) license file at
# ~/mosek/mosek.lic. Until it exists we use CLARABEL, which is enough for every
# convex step. Only the mixed-integer step (Algorithm 1) truly needs MOSEK.

def pick_solver():
    test = cp.Problem(cp.Minimize(cp.square(cp.Variable() - 1)))
    try:
        test.solve(solver=cp.MOSEK)
        return "MOSEK"
    except Exception:
        return "CLARABEL"


def step0():
    solver = pick_solver()
    print("Solver in use:", solver)
    if solver != "MOSEK":
        print("  -> No MOSEK license yet. Fine for now; needed before Algorithm 1.")


# =============================================================================
# STEP 1 - The system and its "lifted" form          (paper Sec. II, Eq. 1, 7)
# =============================================================================
#
# 1a. One time step of the vehicle (a double integrator):
#
#         x_{k+1} = A x_k + B u_k + D w_k                             Eq. (1)
#
#     position_{k+1} = position_k + dt * velocity_k + (dt^2/2) * accel_k
#     velocity_{k+1} = velocity_k + dt * accel_k
#
# 1b. Stacking all N steps into one equation:
#
#         X = A_cal x_0 + B_cal U + D_cal W                           Eq. (7)
#
#     X = [x_0; x_1; ...; x_N]        (all states, one long vector)
#     U = [u_0; u_1; ...; u_{N-1}]    (all controls)
#     W = [w_0; w_1; ...; w_{N-1}]    (all noise samples)
#
#     Writing out x_k by applying Eq. (1) k times gives
#         x_k = A^k x_0 + sum_{j<k} A^{k-1-j} (B u_j + D w_j)
#     so block-row k of the stacked matrices is
#         A_cal[k]     = A^k
#         B_cal[k, j]  = A^{k-1-j} B    if j < k, else 0   (future u can't
#         D_cal[k, j]  = A^{k-1-j} D    if j < k, else 0    affect the past)
#
#     Why bother? With Eq. (7) the whole trajectory is a LINEAR function of
#     U, so later we can optimize all controls at once with a convex solver.

def double_integrator():
    I2 = np.eye(2)
    Z2 = np.zeros((2, 2))
    A = np.block([[I2, dt * I2],
                  [Z2, I2]])
    B = np.vstack([0.5 * dt**2 * I2,
                   dt * I2])
    D = 1e-2 * np.eye(nx)                 # noise scale from the paper
    return A, B, D


def lifted_matrices(A, B, D):
    """Build A_cal, B_cal, D_cal of Eq. (7)."""
    A_pow = [np.linalg.matrix_power(A, k) for k in range(N + 1)]   # A^0..A^N

    A_cal = np.vstack(A_pow)                          # size (N+1)nx x nx
    B_cal = np.zeros(((N + 1) * nx, N * nu))          # size (N+1)nx x N nu
    D_cal = np.zeros(((N + 1) * nx, N * nw))          # size (N+1)nx x N nw

    for k in range(1, N + 1):          # block-row  k  -> state x_k
        for j in range(k):             # block-col  j  -> input u_j / noise w_j
            r = slice(k * nx, (k + 1) * nx)
            B_cal[r, j * nu:(j + 1) * nu] = A_pow[k - 1 - j] @ B
            D_cal[r, j * nw:(j + 1) * nw] = A_pow[k - 1 - j] @ D
    return A_cal, B_cal, D_cal


def E(k):
    """E_k from Eq. (9)-(10):  x_k = E_k X  (selects the k-th state)."""
    E_k = np.zeros((nx, (N + 1) * nx))
    E_k[:, k * nx:(k + 1) * nx] = np.eye(nx)
    return E_k


def step1():
    A, B, D = double_integrator()
    A_cal, B_cal, D_cal = lifted_matrices(A, B, D)

    print("A =\n", A)
    print("B =\n", B)
    print("Sizes: A_cal", A_cal.shape, " B_cal", B_cal.shape,
          " D_cal", D_cal.shape)

    # --- Check: Eq. (7) must give the same trajectory as Eq. (1) ----------
    x_0 = rng.normal(size=nx)
    U = rng.normal(size=N * nu)
    W = rng.normal(size=N * nw)

    # (i) step-by-step with Eq. (1)
    x = x_0
    X_loop = [x]
    for k in range(N):
        u_k = U[k * nu:(k + 1) * nu]
        w_k = W[k * nw:(k + 1) * nw]
        x = A @ x + B @ u_k + D @ w_k
        X_loop.append(x)
    X_loop = np.concatenate(X_loop)

    # (ii) all at once with Eq. (7)
    X_lifted = A_cal @ x_0 + B_cal @ U + D_cal @ W

    err = np.max(np.abs(X_loop - X_lifted))
    print(f"Check Eq.(1) vs Eq.(7): max difference = {err:.1e}  (should be ~1e-15)")
    print("Check E(N) X == x_N:", np.allclose(E(N) @ X_lifted, X_loop[-nx:]))


# =============================================================================
# STEP 2 - Open loop: no feedback (K = 0)              (paper Sec. I, slide 5)
# =============================================================================
#
# Suppose the control is a fixed sequence, u_k = v_k (no feedback at all).
# Take the mean and covariance of Eq. (1):
#
#     mean:        mu_{k+1}    = A mu_k + B v_k
#     covariance:  Sigma_{k+1} = A Sigma_k A^T + D D^T
#
# (the noise w_k has zero mean, identity covariance, and is independent of x_k)
#
# Key observation: v_k appears ONLY in the mean. The covariance recursion has
# no control in it, so it just keeps growing (+ D D^T every step).  This is
# the "open-loop uncertainty" the paper criticises in Sec. I.
#
# Lifted form (stack Eq. (7) and take mean / covariance, with U = V):
#
#     X_bar   = A_cal mu_0 + B_cal V                                   Eq. (15)
#     Sigma_X = A_cal Sigma_0 A_cal^T + D_cal D_cal^T    (= Sigma_Y in Eq. 17)
#
# For a nominal control V we use the minimum-energy V that puts the mean
# exactly on the target:  E_N (A_cal mu_0 + B_cal V) = mu_N   (Eq. 18a).

def ellipse_xy(mean_2d, cov_2d, n_sigma=3, n_pts=60):
    """Points on the n-sigma ellipse of a 2-D Gaussian (for plotting)."""
    vals, vecs = np.linalg.eigh(cov_2d)
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.vstack([np.cos(t), np.sin(t)])
    return mean_2d[:, None] + n_sigma * vecs @ (np.sqrt(vals)[:, None] * circle)


def step2():
    A, B, D = double_integrator()
    A_cal, B_cal, D_cal = lifted_matrices(A, B, D)

    # --- Nominal control: least-norm V with  E_N (A_cal mu_0 + B_cal V) = mu_N
    M = E(N) @ B_cal
    V = np.linalg.pinv(M) @ (mu_N - E(N) @ A_cal @ mu_0)

    # --- (i) Recursion, one step at a time -----------------------------------
    mu, Sigma = mu_0.copy(), Sigma_0.copy()
    mus, Sigmas = [mu], [Sigma]
    for k in range(N):
        v_k = V[k * nu:(k + 1) * nu]
        mu = A @ mu + B @ v_k
        Sigma = A @ Sigma @ A.T + D @ D.T
        mus.append(mu)
        Sigmas.append(Sigma)

    # --- (ii) Lifted form, all at once ---------------------------------------
    X_bar = A_cal @ mu_0 + B_cal @ V
    Sigma_X = A_cal @ Sigma_0 @ A_cal.T + D_cal @ D_cal.T

    err_mu = max(np.abs(E(k) @ X_bar - mus[k]).max() for k in range(N + 1))
    err_S = max(np.abs(E(k) @ Sigma_X @ E(k).T - Sigmas[k]).max()
                for k in range(N + 1))
    print(f"Check recursion vs lifted:  mean err = {err_mu:.1e},"
          f"  covariance err = {err_S:.1e}")

    # --- (iii) Monte Carlo: actually simulate noisy trajectories -------------
    n_mc = 5000
    x = rng.multivariate_normal(mu_0, Sigma_0, size=n_mc)      # x_0 samples
    for k in range(N):
        w = rng.normal(size=(n_mc, nw))
        x = x @ A.T + V[k * nu:(k + 1) * nu] @ B.T + w @ D.T
    print(f"Monte Carlo ({n_mc} runs) terminal mean      = {x.mean(0).round(3)}")
    print(f"Formula                   terminal mean      = {mus[-1].round(3)}")
    print("Position variance at k=N  (MC / formula / target Sigma_N):",
          np.cov(x.T)[0, 0].round(4), "/", Sigmas[-1][0, 0].round(4),
          "/", Sigma_N[0, 0])
    print("-> Mean reaches the target, but the covariance is ~"
          f"{Sigmas[-1][0, 0] / Sigma_N[0, 0]:.0f}x too large. "
          "Open loop cannot fix that.")

    # --- Plot: mean path with 3-sigma position ellipses ----------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    P = np.array(mus)
    ax.plot(P[:, 0], P[:, 1], "b.-", label="mean $\\mu_k$")
    for k in range(N + 1):
        ex, ey = ellipse_xy(mus[k][:2], Sigmas[k][:2, :2])
        ax.plot(ex, ey, "b-", lw=0.6, alpha=0.6)
    ex, ey = ellipse_xy(mu_N[:2], Sigma_N[:2, :2])
    ax.plot(ex, ey, "m-", lw=2, label="target $3\\sigma$ ($\\Sigma_N$)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Open loop (K = 0): covariance only grows")
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.grid(alpha=0.3)
    out = FIG_DIR / "s05_open_loop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)


# =============================================================================
# STEP 3 - Covariance steering with feedback, no obstacles
#          (paper Sec. III-A: Theorem 1, Eqs. 12-19; slides 6, 8, 9, 10)
# =============================================================================
#
# 3a. Policy (Theorem 1, Eq. 12-13):
#
#         u_k = v_k + K_k y_k,      y_{k+1} = A y_k + D w_k,   y_0 = x_0 - mu_0
#
#     y_k is an AUXILIARY state: it is driven by the noise only, with no
#     control.  It is NOT x_k - mu_k once feedback is on (we check this below).
#     Stacked:  U = V + K Y,  Y = A_cal y_0 + D_cal W                 Eq. (14)
#     K is block-diagonal: block (k, k) = K_k, the last block column is zero.
#
# 3b. Mean and deviation (Eq. 15-16):
#
#         X_bar   = A_cal mu_0 + B_cal V
#         X_tilde = (I + B_cal K) Y
#
#     so  Sigma_X = (I + B_cal K) Sigma_Y (I + B_cal K)^T,
#         Sigma_Y = A_cal Sigma_0 A_cal^T + D_cal D_cal^T.
#     V moves only the mean, K moves only the covariance.
#
#     Square root of Sigma_Y: we use L = [A_cal Sigma_0^{1/2}, D_cal], which
#     satisfies L L^T = Sigma_Y exactly. Any factor with L L^T = Sigma_Y
#     gives the same norms in (19) and (20), so we don't need a matrix sqrt.
#
# 3c. Cost (Eq. 17):
#
#     J = X_bar^T Q X_bar + V^T R V + tr[((I+BK)^T Q (I+BK) + K^T R K) Sigma_Y]
#       = X_bar^T Q X_bar + V^T R V + ||Q^{1/2}(I+BK) L||_F^2 + ||R^{1/2} K L||_F^2
#       \________ mean part ________/ \___________ covariance part ___________/
#
# 3d. Terminal constraints:
#
#     E_N X_bar = mu_N                                                Eq. (18a)
#     || Sigma_N^{-1/2} E_N (I + B_cal K) L ||_2 <= 1                   Eq. (19)
#     The spectral norm being <= 1 is the same as
#     E_N Sigma_X E_N^T <= Sigma_N in the PSD order (the relaxed Eq. 18b).
#
# 3e. Bounds (Sec. IV-B):  -100 <= V <= 100,  -10 <= K <= 10.

# ASSUMED: the paper never gives Q_k, R_k. We use Q_k = 0 (no state penalty)
# and R_k = I (control energy only). Then the cost is the expected control
# energy, so a longer path or a tighter covariance both cost more.
Q_k = np.zeros((nx, nx))
R_k = np.eye(nu)
V_MAX, K_MAX = 100.0, 10.0


def build_cov_steering(feedback=True, terminal_cov="sdp"):
    """Variables, expressions and constraints of Problem 2 (no obstacles).

    Returns a dict, so later steps can add chance constraints on top of it.
    feedback=False gives the paper's 'mean steering' baseline (K = 0, and
    the terminal covariance constraint (19) is dropped).
    terminal_cov="sdp"  adds (19) directly as a spectral-norm constraint.
    terminal_cov="cuts" leaves (19) out and returns its matrix as "M_term";
                        Step 5 enforces it with cutting planes instead.
    """
    A, B, D = double_integrator()
    A_cal, B_cal, D_cal = lifted_matrices(A, B, D)
    nX = (N + 1) * nx

    L = np.hstack([A_cal @ np.linalg.cholesky(Sigma_0), D_cal])   # L L^T = Sigma_Y
    Q_bar = np.kron(np.eye(N + 1), Q_k)
    R_bar = np.kron(np.eye(N), R_k)

    V = cp.Variable(N * nu, name="V")
    if feedback:
        K_list = [cp.Variable((nu, nx), name=f"K{k}") for k in range(N)]
        Z = np.zeros((nu, nx))
        K = cp.bmat([[K_list[k] if j == k else Z for j in range(N + 1)]
                     for k in range(N)])
    else:
        K_list = []
        K = np.zeros((N * nu, nX))

    X_bar = A_cal @ mu_0 + B_cal @ V                                 # Eq. (15)
    IBK_L = (np.eye(nX) + B_cal @ K) @ L          # Sigma_X = IBK_L IBK_L^T

    J_mean = cp.quad_form(X_bar, Q_bar) + cp.quad_form(V, R_bar)
    J_cov = (cp.sum_squares(np.sqrt(Q_bar) @ IBK_L)        # Q, R diagonal here
             + cp.sum_squares(np.sqrt(R_bar) @ K @ L))      # Eq. (17)

    cons = [E(N) @ X_bar == mu_N,                                    # Eq. (18a)
            cp.abs(V) <= V_MAX]
    M_term = None
    if feedback:
        S_N_inv_half = np.diag(1 / np.sqrt(np.diag(Sigma_N)))      # diagonal
        M_term = S_N_inv_half @ E(N) @ IBK_L      # (19):  ||M_term||_2 <= 1
        if terminal_cov == "sdp":
            cons += [cp.sigma_max(M_term) <= 1]                     # Eq. (19)
        cons += [cp.abs(Kk) <= K_MAX for Kk in K_list]

    return dict(A=A, B=B, D=D, A_cal=A_cal, B_cal=B_cal, D_cal=D_cal, L=L,
                V=V, K=K, K_list=K_list, X_bar=X_bar, IBK_L=IBK_L,
                J_mean=J_mean, J_cov=J_cov, cons=cons, R_bar=R_bar,
                M_term=M_term)


def solution_stats(P):
    """Numeric mean / covariance trajectories of a solved problem."""
    K_val = P["K"].value if P["K_list"] else P["K"]
    IBK_L = (np.eye((N + 1) * nx) + P["B_cal"] @ K_val) @ P["L"]
    Sigma_X = IBK_L @ IBK_L.T
    X_bar = P["A_cal"] @ mu_0 + P["B_cal"] @ P["V"].value
    mus = [E(k) @ X_bar for k in range(N + 1)]
    Sigmas = [E(k) @ Sigma_X @ E(k).T for k in range(N + 1)]
    K_ks = [K_val[k * nu:(k + 1) * nu, k * nx:(k + 1) * nx] for k in range(N)]
    return mus, Sigmas, K_ks


def monte_carlo(P, n_mc=20000, seed=1):
    """Monte Carlo of a solved cvxpy problem dict P (see simulate_policy)."""
    _, _, K_ks = solution_stats(P)
    xs, ys, us, cost = simulate_policy(P["V"].value, K_ks, n_mc, seed)
    return xs, ys, cost


def simulate_policy(V, K_ks, n_mc=20000, seed=1):
    """Simulate the ACTUAL policy u_k = v_k + K_k y_k with noise.

    Returns state samples xs[k] (n_mc x nx), aux samples ys[k], controls
    us[k] (n_mc x nu), and the realized cost sum_k u_k^T R u_k of every run.
    """
    g = np.random.default_rng(seed)
    A, B, D = double_integrator()
    x = g.multivariate_normal(mu_0, Sigma_0, size=n_mc)
    y = x - mu_0                                                      # Eq. (13b)
    xs, ys, us, cost = [x], [y], [], np.zeros(n_mc)
    for k in range(N):
        u = V[k * nu:(k + 1) * nu] + y @ K_ks[k].T                     # Eq. (12)
        cost += np.einsum("ni,ij,nj->n", u, R_k, u)
        w = g.normal(size=(n_mc, nw))
        x = x @ A.T + u @ B.T + w @ D.T                               # Eq. (1)
        y = y @ A.T + w @ D.T                                         # Eq. (13a)
        xs.append(x)
        ys.append(y)
        us.append(u)
    return xs, ys, us, cost


def step3():
    results = {}
    for name, fb in [("mean steering (K=0)", False),
                     ("covariance steering", True)]:
        P = build_cov_steering(feedback=fb)
        prob = cp.Problem(cp.Minimize(P["J_mean"] + P["J_cov"]), P["cons"])
        prob.solve(solver=pick_solver())
        mus, Sigmas, K_ks = solution_stats(P)
        results[name] = (P, mus, Sigmas)

        print(f"\n=== {name} ===")
        print(f"status: {prob.status},  optimal J = {prob.value:.4f}"
              f"   (J_mean = {P['J_mean'].value:.4f},"
              f" J_cov = {float(P['J_cov'].value):.4f})")

        # --- Check 1: terminal constraints ----------------------------------
        print("terminal mean error ||mu_N - E_N X_bar|| =",
              f"{np.linalg.norm(mus[-1] - mu_N):.1e}")
        gap = np.linalg.eigvalsh(Sigma_N - Sigmas[-1])
        print("eig(Sigma_N - Sigma_x_N), all >= 0 means (19) holds:",
              gap.round(5))

        # --- Check 2: Monte Carlo with the real policy ----------------------
        xs, ys, cost = monte_carlo(P)
        print("MC  terminal pos. variance (x, y):",
              np.var(xs[-1][:, :2], axis=0).round(4),
              "  formula:", np.diag(Sigmas[-1])[:2].round(4),
              "  target:", np.diag(Sigma_N)[:2])
        print(f"MC  expected cost E[sum u'Ru] = {cost.mean():.4f}"
              f"   vs optimal J = {prob.value:.4f}")

        # --- Check 3: y_k is not x_k - mu_k under feedback ------------------
        if fb:
            k = N // 2
            diff = np.abs(ys[k] - (xs[k] - mus[k])).max()
            print(f"max |y_k - (x_k - mu_k)| at k={k}: {diff:.3f}"
                  "  (nonzero: y is the open-loop noise state, not the error)")
            print("bound check: max |K| =",
                  f"{max(np.abs(Kk).max() for Kk in K_ks):.2f}  (limit {K_MAX})")

    # --- Plot: side by side ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.6), sharey=True,
                             layout="constrained")
    for ax, (name, (P, mus, Sigmas)) in zip(axes, results.items()):
        Pm = np.array(mus)
        for k in range(N + 1):
            ex, ey = ellipse_xy(mus[k][:2], Sigmas[k][:2, :2])
            ax.plot(ex, ey, "b-", lw=0.6, alpha=0.6)
        ax.plot(Pm[:, 0], Pm[:, 1], "b.-", label="mean $\\mu_k$")
        ex, ey = ellipse_xy(mu_N[:2], Sigma_N[:2, :2])
        ax.plot(ex, ey, "m-", lw=2, label="target $3\\sigma$ ($\\Sigma_N$)")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y")
    axes[0].set_ylim(-1.5, 1.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("$3\\sigma$ position ellipses, no obstacles")
    out = FIG_DIR / "s08_cov_steering_free.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nSaved", out)


# =============================================================================
# STEP 4 - Chance constraints in ONE convex region     (Eq. 20; slide 11)
# =============================================================================
#
# A half-space  a^T x_k <= b  must hold with probability >= 1 - eps.
# a^T x_k is a scalar Gaussian with mean  a^T mu_k  and standard deviation
#     sigma_a = sqrt(a^T Sigma_k a) = || a^T E_k (I + B_cal K) L ||_2
# so Pr(a^T x_k > b) <= eps   <=>   a^T mu_k + Phi^{-1}(1-eps) sigma_a <= b.
#                                                                     Eq. (20)
# sigma_a is a NORM of something linear in K -> a second-order cone (SOC)
# constraint, still convex.  With eps = 1e-3, Phi^{-1}(1-eps) = 3.09.
#
# Demo (ASSUMED, not in the paper): a narrow "gate"  |p_y| <= 0.3  at steps
# k = 8..12.  The gate needs 3.09 sigma_y <= 0.3, i.e. sigma_y <= 0.097,
# but open loop has sigma_y ~ 0.24 there, so mean steering should fail.

PHI_INV = std_normal.ppf(1 - eps)          # Phi^{-1}(1 - eps) = 3.090
GATE_K = range(8, 13)
GATE_HALF = 0.3


def pos_std(P, k, i):
    """sigma of position coordinate i (0 = x, 1 = y) at step k, Eq. (20)."""
    return cp.norm(P["IBK_L"][k * nx + i, :])


def step4():
    print(f"Phi^-1(1 - eps) = {PHI_INV:.4f}")
    for name, fb in [("mean steering (K=0)", False),
                     ("covariance steering", True)]:
        P = build_cov_steering(feedback=fb)
        cons = list(P["cons"])
        for k in GATE_K:
            py = P["X_bar"][k * nx + 1]
            s = PHI_INV * pos_std(P, k, 1)
            cons += [py + s <= GATE_HALF,        # face  a = +e_y,  b = 0.3
                     -py + s <= GATE_HALF]       # face  a = -e_y,  b = 0.3
        prob = cp.Problem(cp.Minimize(P["J_mean"] + P["J_cov"]), cons)
        prob.solve(solver=pick_solver())
        print(f"\n=== {name} ===  status: {prob.status}")
        if prob.status != "optimal":
            print("  -> as expected: without feedback sigma_y cannot shrink "
                  "enough to fit the gate.")
            continue
        mus, Sigmas, K_ks = solution_stats(P)
        print(f"J = {prob.value:.4f}  (no gate: 94.0323)")

        # Monte Carlo: how often is each face of the gate crossed?
        n_mc = 400_000
        xs, _, _, _ = simulate_policy(P["V"].value, K_ks, n_mc=n_mc, seed=4)
        print(f" k   mu_y     3.09*sigma_y  slack(<=0)   MC Pr(y>0.3)  MC Pr(y<-0.3)"
              f"   (limit {eps:g} each)")
        for k in GATE_K:
            mu_y, sd_y = mus[k][1], np.sqrt(Sigmas[k][1, 1])
            slack = max(mu_y, -mu_y) + PHI_INV * sd_y - GATE_HALF
            up = np.mean(xs[k][:, 1] > GATE_HALF)
            lo = np.mean(xs[k][:, 1] < -GATE_HALF)
            print(f"{k:2d}  {mu_y:+.4f}   {PHI_INV * sd_y:.4f}      {slack:+.1e}"
                  f"     {up:.5f}       {lo:.5f}")
        se = np.sqrt(eps / n_mc)
        print(f"(MC standard error at p = 1e-3 is about {se:.1e})")

        # Plot
        fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
        x_lo = min(mus[k][0] for k in GATE_K)
        x_hi = max(mus[k][0] for k in GATE_K)
        for yb in (GATE_HALF, -GATE_HALF):
            ax.fill_between([x_lo, x_hi], yb, np.sign(yb) * 1.5,
                            color="red", alpha=0.8, lw=0)
        for i in range(60):
            ax.plot([x[i, 0] for x in xs], [x[i, 1] for x in xs],
                    color="0.5", lw=0.4)
        for k in range(N + 1):
            ex, ey = ellipse_xy(mus[k][:2], Sigmas[k][:2, :2])
            ax.plot(ex, ey, "b-", lw=0.6)
        P_ = np.array(mus)
        ax.plot(P_[:, 0], P_[:, 1], "b.-")
        ax.set_aspect("equal")
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Chance-constrained gate $|p_y|\\leq 0.3$ at k = 8..12 "
                     "(covariance steering)")
        out = FIG_DIR / "s11_chance_gate.png"
        fig.savefig(out, dpi=150)
        print("Saved", out)


# =============================================================================
# STEP 5 - Obstacles: Algorithm 1 on the double slit   (Sec. III-B, IV-B;
#          Eqs. 22-25; paper Figs. 2-3; slides 12, 13, 16)
# =============================================================================
#
# 5a. Free space = union of overlapping convex boxes R_r            Eq. (22-23)
#
# 5b. Binary M[r, k] = 1  means  x_k AND x_{k+1} are both in R_r (with
#     chance constraints).  Exactly one region per step pair:       Eq. (24)
#         sum_r M[r, k] = 1
#     Since both ends of every segment lie in the same convex box, the
#     straight segment between them is in that box too (Deits & Tedrake).
#     We use k = 0..N-1 (N pairs, so x_N is covered as well).
#
# 5c. "M[r,k] = 1 => chance constraints (25)" is written with big-M:
#         a^T mu_k + Phi^{-1}(1-eps) sigma_a(k) - b  <=  BIG_M (1 - M[r,k])
#     When M[r,k] = 0 the right side is huge and the constraint is off.
#     Every face here is axis aligned (a = +-e_x or +-e_y), so we only need
#     sigma_x(k), sigma_y(k): one SOC each, s[i,k] >= 3.09 * sigma_i(k).
#     The workspace walls are ALWAYS imposed, which bounds every left-hand
#     side by 14 + 14 = 28, so BIG_M = 30 is valid (never cuts off a solution).
#
# 5d. Solver issue: MOSEK solves mixed-integer SOCPs but NOT mixed-integer
#     SDPs, and (19) ||M_term||_2 <= 1 is an SDP constraint.  The paper went
#     through YALMIP, which runs its own branch & bound in that case.
#     We use an exact cutting-plane loop instead:
#         ||M||_2 <= 1   <=>   ||M^T z||_2 <= 1  for every unit z in R^4
#     Start with z = e_1..e_4 (4 SOCs), solve the MISOCP, and if the result
#     has ||M||_2 > 1 add z = top left singular vector and re-solve.
#     Each relaxed problem has FEWER constraints than the real one, so its
#     optimum is a lower bound; the first relaxed optimum that satisfies (19)
#     is therefore optimal for the real problem too.

WORKSPACE = (-12.0, 2.0, -5.0, 5.0)       # (x_min, x_max, y_min, y_max)
BIG_M = 30.0
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def double_slit(top_gap=0.5):
    """RECONSTRUCTED from paper Figs. 2, 3, 6 by eye.

    Wall at x in [-6, -4] with two slits:
        upper slit  y in [1.0, 1.0 + top_gap]   (0.5 wide in Fig. 2)
        lower slit  y in [-3.0, -1.4]           (1.6 wide)
    Fig. 6 puts the corner of the middle obstacle at (-6, -1.4).
    """
    top = 1.0 + top_gap
    obstacles = [(-6.0, -4.0, top, 5.0),       # upper block
                 (-6.0, -4.0, -1.4, 1.0),      # middle block
                 (-6.0, -4.0, -5.0, -3.0)]     # lower block
    regions = [(-12.0, -6.0, -5.0, 5.0),       # R1: everything left of wall
               (-12.0, 2.0, -3.0, -1.4),       # R2: band through lower slit
               (-12.0, 2.0, 1.0, top),         # R3: band through upper slit
               (-4.0, 2.0, -5.0, 5.0)]         # R4: everything right of wall
    return dict(obstacles=obstacles, regions=regions, top_gap=top_gap)


def box_faces(box):
    """Faces of a box as (coordinate i, sign, b):  sign * p_i <= b."""
    x0, x1, y0, y1 = box
    return [(0, -1, -x0), (0, +1, x1), (1, -1, -y0), (1, +1, y1)]


def region_constraints(P, geom, Mb, feedback, forbid=()):
    """Workspace + region-selection + big-M chance constraints (Eqs. 24-25).

    Mb is either a boolean cp.Variable (the real MI problem) or a fixed 0/1
    numpy array (route fixed -> the problem becomes convex).
    """
    X_bar, IBK_L = P["X_bar"], P["IBK_L"]
    pos = []
    for i in range(2):                       # position means, k = 0..N
        S = np.zeros((N + 1, (N + 1) * nx))
        S[np.arange(N + 1), np.arange(N + 1) * nx + i] = 1
        pos.append(S @ X_bar)

    cons = []
    if feedback:                             # s[i,k] >= 3.09 sigma_i(k)
        s = cp.Variable((2, N + 1), name="s")
        for i in range(2):
            rows = [k * nx + i for k in range(N + 1)]
            cons += [s[i] >= PHI_INV * cp.norm(IBK_L[rows, :], 2, axis=1)]
        sd = [s[0], s[1]]
    else:                                    # K = 0: sigma is a fixed number
        sd = [PHI_INV * np.linalg.norm(
            IBK_L[[k * nx + i for k in range(N + 1)]], axis=1) for i in range(2)]

    for i, sgn, b in box_faces(WORKSPACE):   # walls always on (BIG_M valid)
        cons += [sgn * pos[i] + sd[i] <= b]

    if isinstance(Mb, cp.Variable):
        cons += [cp.sum(Mb, axis=0) == 1]                            # Eq. (24)
        cons += [Mb[r] == 0 for r in forbid]
    for r, box in enumerate(geom["regions"]):
        if not isinstance(Mb, cp.Variable) and not Mb[r].any():
            continue                         # region unused: nothing to add
        for i, sgn, b in box_faces(box):
            for sl in (slice(0, N), slice(1, N + 1)):   # x_k and x_{k+1}
                cons += [sgn * pos[i][sl] + sd[i][sl] - b
                         <= BIG_M * (1 - Mb[r])]                     # Eq. (25)
    return cons


MOSEK_OPTS = {"MSK_DPAR_MIO_MAX_TIME": 600.0,
              "MSK_DPAR_MIO_TOL_REL_GAP": 1e-6}


def solve_algorithm1(geom, feedback=True, forbid=(), tol=1e-4, max_iter=20,
                     quiet=False):
    """Algorithm 1: mixed-integer covariance steering through regions.

    Cutting-plane loop with a bound certificate (see 5d):
      LB = optimum of the MISOCP with only finitely many cuts of (19)
      UB = optimum with the route fixed and the EXACT (19) (convex SDP)
    Stop when (UB - LB) / UB <= tol.  forbid = regions that may not be used.
    Returns a dict of numeric results, or None if infeasible.
    """
    import time
    t0, solver_time = time.time(), 0.0
    P = build_cov_steering(feedback=feedback, terminal_cov="cuts")
    NR = len(geom["regions"])
    Mb = cp.Variable((NR, N), boolean=True, name="M")
    cons = P["cons"] + region_constraints(P, geom, Mb, feedback, forbid)
    J = P["J_mean"] + P["J_cov"]

    zs = []                                   # initial cut directions
    if feedback:
        I4 = np.eye(nx)
        zs = [I4[i] for i in range(nx)]
        zs += [(I4[i] + sg * I4[j]) / np.sqrt(2)
               for i in range(nx) for j in range(i + 1, nx) for sg in (1, -1)]

    for it in range(1, max_iter + 1):
        cuts = [cp.norm(z @ P["M_term"]) <= 1 for z in zs]
        prob = cp.Problem(cp.Minimize(J), cons + cuts)
        prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_OPTS)
        solver_time += prob.solver_stats.solve_time or 0.0
        if prob.status not in ("optimal", "optimal_inaccurate"):
            if not quiet:
                print(f"   MISOCP status: {prob.status}")
            return None
        Mval = np.round(Mb.value).astype(int)
        LB = prob.value
        if not feedback:                      # no (19): MI problem is exact
            final, UB = (P, prob), LB
            break

        # exact problem with this route fixed -> upper bound
        P2 = build_cov_steering(feedback=True, terminal_cov="sdp")
        prob2 = cp.Problem(cp.Minimize(P2["J_mean"] + P2["J_cov"]),
                           P2["cons"] + region_constraints(P2, geom, Mval, True))
        prob2.solve(solver=cp.MOSEK)
        solver_time += prob2.solver_stats.solve_time or 0.0
        # route infeasible under the exact (19) -> no upper bound yet
        UB = prob2.value if prob2.status == "optimal" else np.inf
        gap = (UB - LB) / UB if np.isfinite(UB) else np.inf
        if not quiet:
            print(f"   iter {it}: {len(zs)} cuts, LB = {LB:.4f}, UB = {UB:.4f},"
                  f" rel. gap = {gap:.1e}")
        if gap <= tol:
            final = (P2, prob2)
            break
        K_val = P["K"].value                  # add the most violated direction
        M_num = (np.diag(1 / np.sqrt(np.diag(Sigma_N))) @ E(N)
                 @ (np.eye((N + 1) * nx) + P["B_cal"] @ K_val) @ P["L"])
        zs.append(np.linalg.svd(M_num)[0][:, 0])
    else:
        print("   WARNING: max_iter reached without certificate")
        final = (P2, prob2)

    Pf, probf = final
    mus, Sigmas, K_ks = solution_stats(Pf)
    used = [r for r in range(NR) if Mval[r].any()]
    route = "upper" if 2 in used else ("lower" if 1 in used else "?")
    return dict(feedback=feedback, top_gap=geom["top_gap"], status=probf.status,
                J=float(probf.value), J_mean=float(Pf["J_mean"].value),
                J_cov=float(Pf["J_cov"].value) if feedback else 0.0,
                V=Pf["V"].value.copy(), K_ks=K_ks, mus=mus, Sigmas=Sigmas,
                M=Mval, route=route, LB=float(LB), UB=float(UB),
                rel_gap=float((UB - LB) / UB), n_iter=it,
                solver_time=solver_time, wall_time=time.time() - t0)


def save_result(res, name):
    import pickle
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(res, f)


def load_result(name):
    import pickle
    with open(RESULTS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def draw_world(ax, geom, regions=False):
    from matplotlib.patches import Rectangle
    for (x0, x1, y0, y1) in geom["obstacles"]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="red", lw=0))
    ax.set_xlim(-12, 1)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def draw_solution(ax, geom, res, n_paths=60):
    draw_world(ax, geom)
    xs, _, _, _ = simulate_policy(res["V"], res["K_ks"], n_mc=n_paths, seed=7)
    for i in range(n_paths):
        ax.plot([x[i, 0] for x in xs], [x[i, 1] for x in xs],
                color="0.45", lw=0.4)
    for k in range(N + 1):
        ex, ey = ellipse_xy(res["mus"][k][:2], res["Sigmas"][k][:2, :2])
        ax.plot(ex, ey, "-", color="red" if k == 0 else "blue",
                lw=1.5 if k == 0 else 0.7)
    Pm = np.array(res["mus"])
    ax.plot(Pm[:, 0], Pm[:, 1], "b.-", lw=1.5)
    ex, ey = ellipse_xy(mu_N[:2], Sigma_N[:2, :2])
    ax.plot(ex, ey, "m-", lw=1.5)


def describe(res, name):
    print(f"-> {name}: route = {res['route']} slit, J = {res['J']:.3f} "
          f"(mean {res['J_mean']:.3f} + cov {res['J_cov']:.3f}), "
          f"solver {res['solver_time']:.1f}s, wall {res['wall_time']:.1f}s, "
          f"iterations {res['n_iter']}, certified rel. gap {res['rel_gap']:.1e}")
    print("   regions used per step pair (1-based):",
          "".join(str(int(np.argmax(res['M'][:, k])) + 1) for k in range(N)))


def step5():
    geom = double_slit(top_gap=0.5)

    # Figure: the four sub-regions (paper Fig. 3)
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for r, (ax, box) in enumerate(zip(axes.flat, geom["regions"])):
        x0, x1, y0, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               color="limegreen", lw=0))
        draw_world(ax, geom)
        ax.plot(*ellipse_xy(mu_0[:2], Sigma_0[:2, :2]), "r-")
        ax.plot(*ellipse_xy(mu_N[:2], Sigma_N[:2, :2]), "m-")
        ax.set_title(f"Sub-region {r + 1}")
    out = FIG_DIR / "s12_subregions.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)

    results = {}
    for name, fb in [("mean steering", False), ("covariance steering", True)]:
        print(f"\n=== {name} (upper slit width {geom['top_gap']}) ===")
        res = solve_algorithm1(geom, feedback=fb)
        describe(res, name)
        results[name] = res
        save_result(res, "orig_" + ("cov" if fb else "mean"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, (name, res), tag in zip(axes, results.items(), "ab"):
        draw_solution(ax, geom, res)
        ax.set_title(f"({tag}) {name}: {res['route']} slit, J = {res['J']:.1f}")
    out = FIG_DIR / "s16_double_slit.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


# =============================================================================
# STEP 6 - Narrower upper slit                 (paper Fig. 4; slide 17)
# =============================================================================
# Same setup, upper slit narrowed from 0.5 to 0.2.
# RECONSTRUCTED: 0.2 is read off Fig. 4(a) by eye; the paper gives no number.
# Paper's claim: covariance steering now prefers the LOWER slit, because
# shrinking the covariance enough for the upper slit costs more than the
# detour.  Step 7 checks that this is a cost decision, not infeasibility.

NARROW_GAP = 0.2


def step6():
    geom = double_slit(top_gap=NARROW_GAP)
    results = {}
    for name, fb in [("mean steering", False), ("covariance steering", True)]:
        print(f"\n=== {name} (upper slit width {NARROW_GAP}) ===")
        res = solve_algorithm1(geom, feedback=fb)
        describe(res, name)
        results[name] = res
        save_result(res, "narrow_" + ("cov" if fb else "mean"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    draw_world(axes[0], geom)
    axes[0].plot(*ellipse_xy(mu_0[:2], Sigma_0[:2, :2]), "r-", lw=1.5)
    axes[0].plot(*ellipse_xy(mu_N[:2], Sigma_N[:2, :2]), "m-", lw=1.5)
    axes[0].set_title(f"(a) problem setup: upper slit {NARROW_GAP} wide")
    res = results["covariance steering"]
    draw_solution(axes[1], geom, res)
    axes[1].set_title(f"(b) covariance steering: {res['route']} slit, "
                      f"J = {res['J']:.1f}")
    out = FIG_DIR / "s17_narrow_slit.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


# =============================================================================
# STEP 7 - Why does the route switch?  Width sweep        (extra; slide 17)
# =============================================================================
# For a range of upper-slit widths, solve covariance steering with the route
# FORCED (forbid region 2 -> upper only; forbid region 3 -> lower only).
# Algorithm 1 picks whichever is cheaper, so the crossing point of the two
# cost curves is where the planner switches routes.  The lower-route cost
# does not depend on the upper slit width (regions 1, 2, 4 are unchanged).

SWEEP_GAPS = [0.5, 0.45, 0.4, 0.35, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15]


def step7():
    import json
    lower = solve_algorithm1(double_slit(0.5), feedback=True, forbid=(2,),
                             quiet=True)
    print(f"lower route (any width): J = {lower['J']:.3f} "
          f"(mean {lower['J_mean']:.3f} + cov {lower['J_cov']:.3f})")
    rows = []
    for w in SWEEP_GAPS:
        r = solve_algorithm1(double_slit(w), feedback=True, forbid=(1,),
                             quiet=True)
        row = dict(width=w, feasible=r is not None,
                   J=r["J"] if r else None, J_mean=r["J_mean"] if r else None,
                   J_cov=r["J_cov"] if r else None,
                   min_sigma_y=(min(np.sqrt(S[1, 1]) for S in r["Sigmas"])
                                if r else None),
                   time=r["wall_time"] if r else None,
                   # step pairs spent in the upper-slit band (region 3)
                   pairs_in_slit=int(r["M"][2].sum()) if r else None,
                   peak_vx=float(max(m[2] for m in r["mus"])) if r else None)
        rows.append(row)
        if r:
            print(f"width {w:5.3f}: upper J = {r['J']:8.3f} (mean {r['J_mean']:.3f}"
                  f" + cov {r['J_cov']:.3f}), min sigma_y = {row['min_sigma_y']:.4f}"
                  f", {row['pairs_in_slit']} pairs in slit, peak v_x "
                  f"{row['peak_vx']:.1f}"
                  f"  -> {'upper' if r['J'] < lower['J'] else 'lower'} chosen"
                  f"   [{r['wall_time']:.0f}s]", flush=True)
        else:
            print(f"width {w:5.3f}: upper route infeasible -> lower chosen",
                  flush=True)
    out = dict(lower=dict(J=lower["J"], J_mean=lower["J_mean"],
                          J_cov=lower["J_cov"]), upper=rows)
    (RESULTS_DIR / "width_sweep.json").write_text(json.dumps(out, indent=1))
    plot_sweep()


def plot_sweep():
    """Draw figs/s17_width_sweep.png from results/width_sweep.json."""
    import json
    out = json.loads((RESULTS_DIR / "width_sweep.json").read_text())
    rows, lower = out["upper"], out["lower"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    feas = [r for r in rows if r["feasible"]]
    ax.plot([r["width"] for r in feas], [r["J"] for r in feas], "o-",
            color="tab:blue", label="forced through upper slit")
    for r in feas:
        ax.annotate(f"{r['pairs_in_slit']}", (r["width"], r["J"]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color="tab:blue")
    ax.axhline(lower["J"], color="tab:orange", ls="--",
               label="forced through lower slit")
    ax.set_yscale("log")
    inf_w = [r["width"] for r in rows if not r["feasible"]]
    if inf_w:
        ax.axvspan(min(SWEEP_GAPS) - 0.01, max(inf_w) + 0.0125, color="0.85",
                   label="upper slit infeasible")
    for w, lab in [(0.5, "Fig. 2"), (NARROW_GAP, "Fig. 4")]:
        ax.axvline(w, color="0.4", lw=0.8, ls=":")
        ax.text(w - 0.004, 0.03, lab, transform=ax.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="0.3")
    ax.set_xlabel("upper slit width")
    ax.set_ylabel("optimal cost J (log scale)")
    ax.set_title("numbers: segments $(x_k, x_{k+1})$ assigned to the "
                 "upper-slit region", fontsize=9)
    ax.set_xlim(min(SWEEP_GAPS) - 0.01, max(SWEEP_GAPS) + 0.01)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "s17_width_sweep.png", dpi=150)
    print("Saved", FIG_DIR / "s17_width_sweep.png")


# =============================================================================
# STEP 8 - Verification of all four solutions            (slide 18)
# =============================================================================
# For each saved solution:
#  * solver-side: terminal mean error, terminal covariance (19),
#    worst chance-constraint slack (25) on the selected regions
#  * Monte Carlo with the real policy (200k runs):
#      - worst per-face violation frequency of the selected regions vs eps
#      - collision rate at the sampled times k = 0..N
#      - collision rate along the CONTINUOUS path: between samples the
#        acceleration is held constant, so  p(t) = p_k + v_k t + u_k t^2/2
#        (checked at 10 points per interval)

N_MC_VERIFY = 200_000


def in_collision(p, geom):
    """True where position p (n x 2) is inside an obstacle or off the map."""
    x, y = p[:, 0], p[:, 1]
    x0, x1, y0, y1 = WORKSPACE
    hit = (x < x0) | (x > x1) | (y < y0) | (y > y1)
    for (a0, a1, b0, b1) in geom["obstacles"]:
        hit |= (x > a0) & (x < a1) & (y > b0) & (y < b1)
    return hit


def verify(res, geom, n_mc=N_MC_VERIFY):
    mus, Sigmas, M = res["mus"], res["Sigmas"], res["M"]
    out = dict(route=res["route"], J=res["J"], J_mean=res["J_mean"],
               J_cov=res["J_cov"], rel_gap=res["rel_gap"],
               solve_time=res["wall_time"])
    out["term_mean_err"] = float(np.linalg.norm(mus[-1] - mu_N))
    out["term_cov_lmax"] = float(np.linalg.eigvalsh(Sigmas[-1] - Sigma_N).max())

    # chance-constraint slack on the selected regions (<= 0 means satisfied)
    worst = -np.inf
    for k in range(N):
        r = int(np.argmax(M[:, k]))
        for kk in (k, k + 1):
            for i, sgn, b in box_faces(geom["regions"][r]):
                worst = max(worst, sgn * mus[kk][i]
                            + PHI_INV * np.sqrt(Sigmas[kk][i, i]) - b)
    out["chance_slack_max"] = float(worst)

    xs, _, us, cost = simulate_policy(res["V"], res["K_ks"], n_mc, seed=8)
    out["mc_cost"] = float(cost.mean())
    out["mc_term_var_xy"] = np.var(xs[-1][:, :2], axis=0).tolist()

    face_max = 0.0
    for k in range(N):
        r = int(np.argmax(M[:, k]))
        for kk in (k, k + 1):
            for i, sgn, b in box_faces(geom["regions"][r]):
                face_max = max(face_max, np.mean(sgn * xs[kk][:, i] > b))
    out["mc_face_viol_max"] = float(face_max)

    hit_k = np.array([in_collision(x[:, :2], geom) for x in xs])   # (N+1, n)
    out["mc_step_coll_max"] = float(hit_k.mean(axis=1).max())
    out["mc_traj_coll_sampled"] = float(hit_k.any(axis=0).mean())

    any_hit = hit_k.any(axis=0)
    for k in range(N):
        p, v, u = xs[k][:, :2], xs[k][:, 2:], us[k]
        for t in np.linspace(0, dt, 11)[1:-1]:
            any_hit |= in_collision(p + v * t + 0.5 * u * t**2, geom)
    out["mc_traj_coll_continuous"] = float(any_hit.mean())
    return out


def step8():
    import json
    cases = [("orig_mean", 0.5), ("orig_cov", 0.5),
             ("narrow_mean", NARROW_GAP), ("narrow_cov", NARROW_GAP)]
    table = {}
    for name, w in cases:
        table[name] = verify(load_result(name), double_slit(w))
        print(name, "done", flush=True)
    (RESULTS_DIR / "verification.json").write_text(json.dumps(table, indent=1))

    fmt = [("route", "route", "{}"),
           ("J", "optimal cost J", "{:.3f}"),
           ("J_mean", "  mean part", "{:.3f}"),
           ("J_cov", "  covariance part", "{:.3f}"),
           ("mc_cost", "MC average cost", "{:.3f}"),
           ("rel_gap", "certified optimality gap", "{:.0e}"),
           ("solve_time", "solve time [s]", "{:.1f}"),
           ("term_mean_err", "terminal mean error", "{:.1e}"),
           ("term_cov_lmax", "lambda_max(Sigma_xN - Sigma_N)  (<=0 ok)", "{:+.1e}"),
           ("chance_slack_max", "worst chance slack (<=0 ok)", "{:+.1e}"),
           ("mc_face_viol_max", f"MC worst per-face violation (eps={eps:g})", "{:.5f}"),
           ("mc_step_coll_max", "MC worst per-step collision rate", "{:.5f}"),
           ("mc_traj_coll_sampled", "MC trajectory collision (sampled k)", "{:.5f}"),
           ("mc_traj_coll_continuous", "MC trajectory collision (continuous)", "{:.5f}")]
    hdr = ("| metric | slit 0.5, mean | slit 0.5, cov | "
           f"slit {NARROW_GAP}, mean | slit {NARROW_GAP}, cov |")
    lines = [hdr, "|---|---|---|---|---|"]
    for key, label, f in fmt:
        lines.append(f"| {label} | " + " | ".join(
            f.format(table[n][key]) for n, _ in cases) + " |")
    md = "\n".join(lines)
    (RESULTS_DIR / "verification.md").write_text(
        f"Monte Carlo: {N_MC_VERIFY} runs per case, seed 8.\n\n" + md + "\n")
    print(md)


# =============================================================================
# RUN
# =============================================================================
STEPS = {0: step0, 1: step1, 2: step2, 3: step3, 4: step4, 5: step5,
         6: step6, 7: step7, 8: step8}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=max(STEPS))
    STEPS[parser.parse_args().step]()
