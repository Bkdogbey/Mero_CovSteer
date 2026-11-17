# Presentation review

Reviewed November 17, 2026: all 27 slides, speaker notes, and PowerPoint-rendered slide images from `Optimal Covariance Control for Stochastic Vehicle Path Planning.pptx` on the Desktop. The original deck was opened read-only and was not edited.

The model-to-policy-to-constraints sequence is coherent, and the main costs agree with the local saved data and the previously checked implementation. Keep that structure. The most useful revisions are mathematical precision, solver transparency, and a clear transition into the live code.

## Corrections before presenting

| Slide | Issue | Suggested change |
|---|---|---|
| 13 | The displayed covariance cost places `Sigma_Y` after the closing trace bracket. This makes the notation read as a scalar multiplied by a matrix. | Put the entire product inside the trace: `tr((((I + B K)^T Qbar (I + B K)) + K^T Rbar K) Sigma_Y)`, using the lifted B. Keep the mean-cost terms outside. |
| 18 | The footer describes the full problem as second-order cones plus binaries. The terminal spectral-norm constraint generally requires a semidefinite cone. | Say: “Mixed-integer conic optimization: second-order chance constraints plus a semidefinite terminal constraint.” Then explain the implementation's cutting-plane workaround. |
| 3, 16, 20 | The general risk budget and the example's per-face threshold are easy to confuse. | Add: “This reproduction uses epsilon = 0.001 per region face at each sampled state. This is not a 0.001 whole-trip collision bound.” Risk allocation across faces, and separately across time, is needed for a specified joint guarantee. |
| 17 | Both endpoints in a convex region guarantee containment of their straight connecting segment. The simulated double-integrator path under held acceleration is curved, and endpoint chance constraints are probabilistic. | Say: “Consecutive states share a region with per-face chance margins. The straight segment between realized endpoints remains inside if both endpoints are inside. Actual between-step dynamics are checked at additional samples.” |
| 25 | “Monte Carlo confirms the safety and terminal constraints” overstates what finite samples establish. “Collision, continuous path” suggests an exhaustive continuous-time check. | Title: “Analytical checks and Monte Carlo support the planned constraints.” Row: “Collision, extra between-step samples.” Show terminal mean error, maximum covariance excess eigenvalue, and worst chance slack as analytical checks if space permits. |
| 26 | “The terminal covariance is met exactly” conflicts with the upper-bound formulation explained on slide 14. | Say: “The terminal covariance upper bound is satisfied within solver tolerance.” Describe Monte Carlo frequencies as consistent with the per-face threshold, rather than a proof of safety. |
| 6 | Open-loop covariance need not increase monotonically for an arbitrary linear system. Its recursion includes propagation by A, not just addition of D D^T. | Title: “Without feedback, planned controls cannot shape covariance.” Caption: “Sigma evolves as A Sigma A^T + D D^T, independently of v. In this double-integrator example, position uncertainty grows.” |
| 8 | “The covariance is quadratic in K, so the problem becomes convex” skips essential conditions. General quadratic expressions and covariance equalities need not be convex. | Say: “The deviation factor is affine in K. The quadratic cost is convex with positive-semidefinite weights; the chance constraints use norms; the terminal covariance equality is relaxed to an upper bound.” |
| 13 | “Shrinking Sigma raises the second term” is too general for the displayed cost with nonzero state weights Q. | Say: “The covariance cost includes state spread and feedback effort. Here Q = 0 and R = I, so it is feedback control energy.” Give the implementation form `J = ||V||^2 + ||KL||_F^2`. |

The model matrices on slide 19 and the lifting structure on slides 10–11 look consistent with the implementation. Slides 14–16 clearly establish the terminal bound and scalar Gaussian margin; keep those explanations and tighten the surrounding claims.

## Add an implementation and live-demo transition

Place a short slide after the optimization summary or immediately before the live demonstration:

**Implementation: what will run live**

- CVXPY expresses the objective and constraints.
- MOSEK chooses the region schedule using mixed-integer cone relaxations. Additional cuts enforce the terminal covariance condition.
- Clarabel solves the convex problems with exact terminal constraints for a fixed schedule.
- Saved-route mode reuses both the slit and crossing times, then optimizes V and K fresh. It does not independently rediscover the route or certify its global optimality.

Suggested command for the full configured solver path:

```powershell
& 'C:\Users\Bennett\anaconda3\envs\covsteer\python.exe' .\covsteer.py --samples 10000
```

Suggested saved-route rehearsal command:

```powershell
& 'C:\Users\Bennett\anaconda3\envs\covsteer\python.exe' .\covsteer.py --saved-routes --samples 10000
```

Show the printed solve mode. Explain `policy()`, `build()`, and `region_constraints()`, then compare the static paths and verification table. Use the covariance-bound failure of the mean-only baseline to explain why it omits that constraint.

## Results and presentation checks

- Slide 22's change from 201.7 to 116.4 is approximately 42.3% less control energy. Label it “42% less control energy” for precision.
- Slide 23's forced-upper-route cost of 218.4 agrees with `results/width_sweep.json`; the chosen lower route costs 139.1. The upper route remains feasible at width 0.2, so the route change is a cost decision in this reproduction.
- Slide 24's switch bracket, 0.25 to 0.225, agrees with the saved sweep. Treat the number of intervals assigned to the upper region as a schedule statistic, rather than an exact measurement of physical wall-crossing duration.
- The numeric Monte Carlo table is an archived 200,000-run check. If the live demo uses 10,000 runs, explain that frequencies will differ. Retain a clear label for the sample count and seed.
- The slide-25 runtimes should be labeled as a particular recorded full-solver run, with its mode and machine. Saved-route solves can be much faster and should not be presented as timings for mixed-integer route discovery.
- In slide-15 speaker notes, use “a covariance factor is affine in K.” The symmetric matrix square root of the controlled covariance is not generally affine.

## Layout and pacing

- The orange/navy palette, consistent headings, and progress indicator make the sequence easy to follow.
- The plot axis labels and table text on slides 20, 22–25 are small relative to the slide titles. Enlarge the figures or remove secondary details for projection. Slide 6's open-loop plot is particularly small.
- If presentation time is tight, move slides 10–11 (matrix construction details) and 24 (extra sweep) to backup. Keep the lifted equation, the meaning of V and K, the chance constraint, and the two main experiments in the main talk.
- Slide 13 has an unfinished card heading, “2 ·”. Replace it with “2 · Mean–covariance identity”.
- Slide 2: change “The mean steering alone is not enough” to “Mean steering alone is not enough”. Slide 8: replace “initial noise and error” with “initial deviation and process noise”.
- The title slide shows November 11, 2026; confirm that this is the intended presentation date.
- There are 27 actual slides, but the references footer reads 28. Renumber it.

## Sources checked

The paper's accepted preprint supports the policy, covariance relaxation, and Gaussian risk-allocation discussion: [Okamoto and Tsiotras, accepted RA-L preprint](https://dcsl.gatech.edu/publications/papers/ral19.pdf).

MOSEK's documentation explicitly excludes semidefinite constraints from its mixed-integer solver support: [MOSEK integer optimization documentation](https://docs.mosek.com/latest/pythonapi/tutorial-mio-shared.html).

Implementation-specific findings were checked against `covsteer.py`, `results/routes.json`, and `results/width_sweep.json`. Rendering and review did not change those files.
