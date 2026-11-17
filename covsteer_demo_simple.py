"""
Mean steering vs. covariance steering: a simple animation.

Both planners must get from the start to the goal in 4 s through a wall with
two slits, staying collision-free with probability 99.9%.
    Mean steering        plans only the average path; the spread keeps growing.
    Covariance steering  also uses feedback to shrink the spread before the wall.

The plans come from covsteer.py, using the routes in results/routes.json,
so this runs in a few seconds (no MOSEK needed).

Run:  python covsteer_demo_simple.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import covsteer as cs

N_CARS = 200

# 1. Plan both ways through the double slit (route fixed from the saved file)
geom = cs.double_slit(0.5)
routes = json.loads((cs.RESULTS / "routes.json").read_text())
mean_plan = cs.solve_convex(False, geom, cs.route_from_str(routes["slit0.5_mean"]))
cov_plan = cs.solve_convex(True, geom, cs.route_from_str(routes["slit0.5_cov"]))

# 2. Simulate the same number of vehicles under each plan
mean_xs, _ = cs.simulate(mean_plan, n=N_CARS, seed=1)    # shape (N+1, cars, 4)
cov_xs, _ = cs.simulate(cov_plan, n=N_CARS, seed=1)

# 3. Draw the two scenes
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=cs.PAPER)
panels = [(axes[0], "Mean steering", mean_plan, mean_xs, cs.BURNT),
          (axes[1], "Covariance steering", cov_plan, cov_xs, cs.BLUE)]

scenes = []
for ax, name, plan, xs, color in panels:
    cs.draw(ax, geom=geom)                               # obstacles, start and goal
    ax.set_title(f"{name}: {plan['slit']} slit, J = {plan['J']:.0f}")

    # the planned "tube": mean path and 3-sigma ellipse at every step (faint)
    ax.plot(plan["mus"][:, 0], plan["mus"][:, 1], ":", color=color)
    for mean, cov in zip(plan["mus"], plan["Sigmas"]):
        ax.plot(*cs.ellipse(mean[:2], cov[:2, :2]), color=color, lw=0.5, alpha=0.3)

    cars = ax.scatter([], [], s=6, color="0.35", zorder=3)
    ellipse, = ax.plot([], [], color=color, lw=2.5, zorder=4)
    label = ax.text(-3.7, -4.7, "", fontsize=10)              # bottom right, clear of the wall
    scenes.append(dict(plan=plan, xs=xs, cars=cars, ellipse=ellipse, label=label,
                       crashed=np.zeros(N_CARS, bool)))


# 4. Animate: one frame per time step
def update(k):
    for s in scenes:
        if k == 0:
            s["crashed"][:] = False                      # restart the count on each loop
        positions = s["xs"][k, :, :2]
        s["crashed"] |= cs.collides(positions, geom)

        s["cars"].set_offsets(positions)
        mean, cov = s["plan"]["mus"][k], s["plan"]["Sigmas"][k]
        s["ellipse"].set_data(*cs.ellipse(mean[:2], cov[:2, :2]))
        s["label"].set_text(f"t = {k * cs.dt:.1f} s\ncrashes: {s['crashed'].sum()} / {N_CARS}")
    return []


animation = FuncAnimation(fig, update, frames=cs.N + 1, interval=300, repeat_delay=1500)
plt.show()
