"""A simpler version of covsteer.py, following the same theory.

Run: python covsteer_demo_simple.py

Model -> lifted dynamics -> policy -> optimization -> simulation -> plots.
We solve for the mean controls and feedback gains using Clarabel. To keep
the code simple, region schedules come from results/routes.json rather than
running Algorithm 1. These schedules specify both the slit and crossing times;
we do not optimize those choices here. No MOSEK license is needed.
"""

import json
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import norm


# 1. PROBLEM AND VEHICLE MODEL (Eq. 1)
# State x = [px, py, vx, vy]; control u = [ax, ay].
dt, N = 0.2, 20
nx, nu = 4, 2
eps = 0.001                         # risk per face at each sampled state
z = norm.ppf(1 - eps)               # Gaussian safety multiplier, about 3.09

# ASSUMED distributions and weights: Q = 0, R = I (control energy).
mu_0 = np.array([-10.0, 0.1, 0.0, 0.0])
mu_N = np.zeros(nx)
Sigma_0 = np.diag([0.05, 0.05, 0.001, 0.001])
Sigma_N = np.diag([0.01, 0.01, 0.001, 0.001])

A = np.block([[np.eye(2), dt * np.eye(2)],
              [np.zeros((2, 2)), np.eye(2)]])
B = np.vstack([0.5 * dt**2 * np.eye(2), dt * np.eye(2)])
D = 0.01 * np.eye(nx)


# 2. LIFTED DYNAMICS (Eq. 7): X = A_stack x0 + B_stack U + D_stack W
def rows(k):
    """Rows of state x_k in the stacked trajectory X."""
    return slice(k * nx, (k + 1) * nx)


A_stack = np.vstack([np.linalg.matrix_power(A, k) for k in range(N + 1)])
B_stack = np.zeros(((N + 1) * nx, N * nu))
D_stack = np.zeros(((N + 1) * nx, N * nx))
for k in range(1, N + 1):
    for j in range(k):
        A_power = np.linalg.matrix_power(A, k - 1 - j)
        B_stack[rows(k), j * nu:(j + 1) * nu] = A_power @ B
        D_stack[rows(k), rows(j)] = A_power @ D

# Y = L xi, where xi is standard Gaussian; covariance(Y) = L L.T.
L = np.hstack([A_stack @ np.linalg.cholesky(Sigma_0), D_stack])


# 3. GEOMETRY AND CHANCE CONSTRAINTS (Eqs. 20, 22-25)
def geometry(width):
    """RECONSTRUCTED wall geometry; each box is (xmin, xmax, ymin, ymax)."""
    obstacles = [(-6, -4, 1 + width, 5), (-6, -4, -1.4, 1), (-6, -4, -5, -3)]
    regions = [(-12, -6, -5, 5),       # R1: left of wall
               (-12, 2, -3, -1.4),    # R2: lower slit
               (-12, 2, 1, 1 + width),  # R3: upper slit
               (-4, 2, -5, 5)]        # R4: right of wall
    return obstacles, regions


def inside_box(mean, S, k, box):
    """Pr(a.T x_k > b) <= eps becomes a.T mean_k + z * std <= b."""
    xmin, xmax, ymin, ymax = box
    px, py = mean[k * nx], mean[k * nx + 1]
    sx = cp.norm(S[k * nx, :])
    sy = cp.norm(S[k * nx + 1, :])
    return [px + z * sx <= xmax, -px + z * sx <= -xmin,
            py + z * sy <= ymax, -py + z * sy <= -ymin]


# 4. POLICY AND OPTIMIZATION (Theorem 1, Eqs. 12-19)
def solve(feedback, route, regions):
    """Optimize the controls for a fixed sequence of regions, or return None."""
    # u_k = v_k + K_k y_k; y_{k+1} = A y_k + D w_k.
    # y is the noise-only auxiliary state, not the actual tracking error.
    V = cp.Variable(N * nu)
    if feedback:
        K = [cp.Variable((nu, nx)) for k in range(N)]
        KL = cp.vstack([K[k] @ L[rows(k)] for k in range(N)])
    else:
        K = np.zeros((N, nu, nx))
        KL = np.zeros((N * nu, L.shape[1]))

    mean = A_stack @ mu_0 + B_stack @ V       # Eq. 15: V sets the mean
    S = L + B_stack @ KL                     # Eq. 16: covariance(X) = S S.T
    cost = cp.sum_squares(V) + cp.sum_squares(KL)  # Eq. 17: expected energy
    constraints = [mean[rows(N)] == mu_N, cp.abs(V) <= 100]
    if feedback:
        constraints += [cp.abs(gain) <= 10 for gain in K]
        # Eq. 19: final covariance <= Sigma_N in the positive-semidefinite order.
        inv_sqrt = np.diag(1 / np.sqrt(np.diag(Sigma_N)))
        constraints.append(cp.sigma_max(inv_sqrt @ S[rows(N)]) <= 1)
    # The mean-only baseline cannot control covariance, so it omits Eq. 19.

    # Each digit selects a region for BOTH endpoints of a time interval.
    # The route is fixed, so no binary variables or big-M constraints are needed.
    for k in range(N):
        box = regions[int(route[k]) - 1]
        constraints += inside_box(mean, S, k, box)
        constraints += inside_box(mean, S, k + 1, box)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL:
        return None

    gains = np.array([gain.value for gain in K]) if feedback else K
    S_value = S.value if feedback else S
    covariances = np.array([S_value[rows(k)] @ S_value[rows(k)].T
                            for k in range(N + 1)])
    return dict(V=V.value.reshape(N, nu), K=gains, J=float(problem.value),
                mean=mean.value.reshape(N + 1, nx), cov=covariances)


# 5. SIMULATION AND VERIFICATION
def simulate(plan, n=10_000, seed=1):
    """Run the actual policy with Gaussian initial states and process noise."""
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mu_0, Sigma_0, size=n)
    y = x - mu_0
    states, energy = [x], np.zeros(n)
    for k in range(N):
        u = plan['V'][k] + y @ plan['K'][k].T
        w = rng.standard_normal((n, nx))
        x = x @ A.T + u @ B.T + w @ D.T
        y = y @ A.T + w @ D.T
        states.append(x)
        energy += np.sum(u**2, axis=1)
    return np.array(states), energy


def report(plan, states, energy, route, regions, obstacles):
    mean_error = np.linalg.norm(plan['mean'][-1] - mu_N)
    cov_excess = np.linalg.eigvalsh(plan['cov'][-1] - Sigma_N).max()
    worst_slack = -np.inf
    for k in range(N):
        xmin, xmax, ymin, ymax = regions[int(route[k]) - 1]
        for j in (k, k + 1):
            px, py = plan['mean'][j, :2]
            sx, sy = np.sqrt(np.diag(plan['cov'][j])[:2])
            worst_slack = max(worst_slack, px + z * sx - xmax, -px + z * sx + xmin,
                              py + z * sy - ymax, -py + z * sy + ymin)
    hit = np.zeros(states.shape[1], dtype=bool)
    for xmin, xmax, ymin, ymax in obstacles:
        hit |= ((states[:, :, 0] > xmin) & (states[:, :, 0] < xmax)
                & (states[:, :, 1] > ymin) & (states[:, :, 1] < ymax)).any(axis=0)
    print(f"  J = {plan['J']:.3f}; Monte Carlo energy = {energy.mean():.3f}")
    print(f"  Terminal mean error: {mean_error:.1e}")
    print(f"  Terminal covariance excess: {cov_excess:+.1e} (<= 0 for feedback)")
    print(f"  Worst chance slack: {worst_slack:+.1e} (<= 0, up to solver tolerance)")
    print(f"  Sampled obstacle collisions: {hit.sum()} / {len(hit)}")


# 6. STATIC RESULTS: MEAN PATH, 3-SIGMA ELLIPSES, AND SAMPLE TRAJECTORIES
def ellipse(mean, cov):
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2 * np.pi, 60)
    circle = np.array([np.cos(angle), np.sin(angle)])
    points = mean[:, None] + 3 * vectors @ (np.sqrt(np.maximum(values, 0))[:, None] * circle)
    return points[0], points[1]


def plot(ax, plan, states, obstacles, title):
    for xmin, xmax, ymin, ymax in obstacles:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               color='tomato'))
    ax.plot(states[:, :40, 0], states[:, :40, 1], color='0.65', lw=0.4)
    ax.plot(plan['mean'][:, 0], plan['mean'][:, 1], 'b.-', label='mean path')
    for mean, cov in zip(plan['mean'], plan['cov']):
        ax.plot(*ellipse(mean[:2], cov[:2, :2]), color='tab:blue', lw=0.7)
    ax.plot(*ellipse(mu_N[:2], Sigma_N[:2, :2]), 'm-', label='target spread')
    ax.set(xlim=(-12, 2), ylim=(-5, 5), xlabel='x', ylabel='y',
           title=f"{title}\nJ = {plan['J']:.2f}", aspect='equal')
    ax.legend(loc='upper left', fontsize=8)


def main():
    path = Path(__file__).resolve().parent / 'results' / 'routes.json'
    routes = json.loads(path.read_text())
    print('Fresh convex solves using saved region schedules (slit AND timing).')
    print('Risk 0.001 is per face per sampled state, not a whole-trip guarantee.')
    print('Collision counts check sampled states only; finite samples do not prove zero risk.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row, width in enumerate((0.5, 0.2)):
        obstacles, regions = geometry(width)
        for col, feedback in enumerate((False, True)):
            name = 'cov' if feedback else 'mean'
            route = routes[f'slit{width}_{name}']
            title = f"{'Covariance' if feedback else 'Mean'} steering, slit width {width}"
            print(f'\n{title}: {"upper" if "3" in route else "lower"} route')
            plan = solve(feedback, route, regions)
            if plan is None:
                raise RuntimeError(f'No optimal solution for {title}.')
            states, energy = simulate(plan)
            report(plan, states, energy, route, regions, obstacles)
            plot(axes[row, col], plan, states, obstacles, title)
    plt.show()


if __name__ == '__main__':
    main()
