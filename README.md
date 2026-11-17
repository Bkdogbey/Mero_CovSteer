# Covariance Steering for Stochastic Vehicle Path Planning: A Reproduction

A Python reproduction of

> K. Okamoto and P. Tsiotras, "Optimal Stochastic Vehicle Path Planning Using Covariance Steering," *IEEE Robotics and Automation Letters*, 4(3):2276–2281, 2019. [doi:10.1109/LRA.2019.2901546](https://doi.org/10.1109/LRA.2019.2901546)

The method plans a vehicle's **mean path and its uncertainty** together. A feedback policy shrinks the state covariance where it matters, and chance constraints keep each obstacle face violated with probability at most ε. Obstacles are handled with mixed-integer region selection. The result: a planner that can take narrow passages a mean-only planner must avoid.

## Results

| Experiment | Planner | Route | Cost J |
|---|---|---|---|
| Upper slit 0.5 wide (paper Fig. 2) | mean steering (K = 0) | lower slit (detour) | 201.7 |
| | covariance steering | **upper slit** | **116.4** |
| Upper slit 0.2 wide (paper Fig. 4) | covariance steering | lower slit | 139.1 |

- Both qualitative results of the paper are reproduced. The covariance-steering solutions are certified optimal (relative gap ≤ 1e-5).
- A slit-width sweep (extra analysis) shows that the route switches between widths 0.25 and 0.225. The switch is driven by the *mean* cost of crossing the wall in fewer time steps.
- 200,000-run Monte Carlo checks:
  - per-face violation rates are about ε = 0.001;
  - terminal covariance targets are met exactly;
  - covariance steering has zero collisions at the sampled time steps.

  `python covsteer.py` prints the full table.

## Repository layout

| Path | Contents |
|---|---|
| [`covsteer.py`](covsteer.py) | The whole method as one pipeline (blocks 1–12), commented with the paper's equation numbers |
| [`notebooks/covsteer_pipeline.ipynb`](notebooks/covsteer_pipeline.ipynb) | Presentation notebook: runs the blocks of `covsteer.py` one by one, each with what it does |
| [`covsteer_demo.py`](covsteer_demo.py) | Classroom animation with captions: mean vs covariance steering through the double slit |
| [`covsteer_demo_simple.py`](covsteer_demo_simple.py) | Standalone simplified implementation: fixed region schedules, fresh convex solves, and static plots |
| `results/` | Inputs the code reads: routes chosen by Algorithm 1 (`routes.json`) and the slit-width sweep (`width_sweep.json`) |

## Setup

```bash
conda env create -f environment.yml
conda activate covsteer
python -m ipykernel install --user --name covsteer --display-name "Python (covsteer)"
```

Algorithm 1 (the mixed-integer step) needs [MOSEK](https://www.mosek.com/products/academic-licenses/), which offers free academic licenses. Put the license at `~/mosek/mosek.lic` (Windows: `C:\Users\<you>\mosek\mosek.lic`).
Without a license everything still runs: the route MOSEK chose is read from `results/routes.json`, and the rest of the problem, which is convex once the route is fixed, is solved with the open-source Clarabel solver. The costs come out the same.

## Usage

```bash
python covsteer.py            # five-stage walkthrough, static figures, and verification
python covsteer.py --saved-routes  # use saved region schedules; optimize controls with Clarabel
python covsteer.py --samples 10000 # shorter Monte Carlo check for rehearsal
python covsteer.py --sweep    # also re-run the slit-width sweep (licensed MOSEK)
python covsteer_demo.py       # animation with captions
python covsteer_demo_simple.py  # standalone simplified implementation with static plots
```

The walkthrough follows five stages: task, open-loop uncertainty, feedback,
obstacle planning, and verification. It prints a compact table of route, energy,
and terminal covariance results before the detailed Monte Carlo checks.
The width sweep runs only with `--sweep`.

The obstacle solve mode is printed at startup and for each solution. With MOSEK,
the code optimizes the region schedule and controls. With `--saved-routes`, or
when MOSEK is unavailable, it reuses the saved slit and crossing times and solves
the mean controls and feedback gains fresh. The obstacle-free example uses Clarabel.

The terminal covariance is an upper bound, imposed only for covariance steering.
Risk is specified per face at each sampled state. The between-step collision check
uses additional time samples and does not provide a continuous-time safety certificate.

| Block in `covsteer.py` | What it does | Paper |
|---|---|---|
| 1–3 | problem data, vehicle model, lifted dynamics | Eq. 1, 7 |
| 4 | policy: mean set by V, covariance set by K | Thm. 1, Eq. 12–16 |
| 5 | open loop (K = 0): covariance only grows | Sec. I |
| 6 | optimal covariance steering (closed loop) | Eq. 17–19 |
| 7 | chance constraints | Eq. 20 |
| 8 | obstacles: region selection | Eq. 22–25 |
| 9 | Algorithm 1: mixed-integer solve, certified optimal | Figs. 2–4 |
| 10 | Monte Carlo verification | — |
| 11–12 | plots; `main()` runs the pipeline | — |

Notebook: open `notebooks/covsteer_pipeline.ipynb` with the **Python (covsteer)** kernel and run all cells.

## Implementation notes

- **Mixed-integer SDP workaround.** MOSEK cannot solve mixed-integer semidefinite programs, and the terminal covariance constraint (Eq. 19) is semidefinite. We enforce it with second-order-cone cutting planes. A lower/upper-bound certificate proves optimality:
  - the relaxed problem gives a lower bound;
  - fixing the chosen route and solving the exact convex SDP gives an upper bound.

  The paper used YALMIP, which runs its own branch and bound for this case.
- **Assumed** (the paper does not report them): initial and target distributions μ₀, Σ₀, Σ_N, and cost weights Q = 0, R = I.
- **Reconstructed**: the obstacle and sub-region geometry was read off the paper's Figs. 2, 3, 4 and 6 by eye.


