
"""
Heatmap generator around a point using steady-state heat diffusion.

Parameters:
  d  - distance from center to the goal (goal is outside the circle)
  r  - radius of the circle
  A  - 2D array where 0 = obstacle (blocks heat, reflects it), nonzero = free space

Boundary condition:
  Points on the circle boundary get heat = 1 / (distance_to_goal) * 100

Physics:
  - Heat diffuses inward from the circle boundary via Laplace's equation
  - Obstacle cells (A == 0) are treated as perfect reflectors:
      they carry no heat themselves, and their flux into neighbours is zeroed
      (Neumann / no-flux boundary), so heat flows around them
  - Solved iteratively with Gauss-Seidel until convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from testing import WoS_altered_g, load_ndarray
from helper import merge_domains_nd


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def generate_heatmap(
    d: float,
    r: float,
    A: np.ndarray,
    *,
    goal_angle_deg: float = 0.0,
    max_iters: int = 20_000,
    tol: float = 1e-5,
    plot: bool = True,
) -> np.ndarray:
    """
    Generate a steady-state heatmap inside a circle of radius r.

    Args:
        d             : Distance from the circle centre to the goal (goal is
                        outside the circle, so d > r is expected).
        r             : Radius of the circle (in array-index units).
        A             : 2-D numpy array.  Cells with value 0 are obstacles
                        (no-flux / reflective walls).  The array dimensions
                        define the spatial grid; the circle is centred in it.
        goal_angle_deg: Direction of the goal measured from the positive-x axis
                        (degrees, counter-clockwise). Default 0 → goal is to
                        the right of the centre.
        max_iters     : Maximum Gauss-Seidel iterations.
        tol           : Convergence tolerance (max absolute change per step).
        plot          : If True, display the heatmap with matplotlib.

    Returns:
        heat : 2-D numpy array of the same shape as A with heat values.
               Cells outside the circle and obstacle cells are NaN.
    """
    A = np.asarray(A, dtype=float)
    rows, cols = A.shape

    # Centre of the grid
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0

    # Goal position (outside the circle, at distance d from centre)
    angle_rad = np.radians(goal_angle_deg)
    goal_x = cx + d * np.cos(angle_rad)
    goal_y = cy - d * np.sin(angle_rad)  # y-axis flipped for image coords

    # ------------------------------------------------------------------
    # Build masks
    # ------------------------------------------------------------------
    ys, xs = np.mgrid[0:rows, 0:cols]
    dist_from_centre = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    inside_circle = dist_from_centre <= r             # True for interior + boundary
    obstacle_mask = (A == 0) & inside_circle          # obstacles inside circle
    free_mask     = inside_circle & ~obstacle_mask    # free cells inside circle

    # Boundary ring: cells whose distance from centre is within half a pixel of r
    on_boundary = np.abs(dist_from_centre - r) <= 0.5

    # ------------------------------------------------------------------
    # Boundary heat values:  heat = 1 / dist_to_goal * 100
    # ------------------------------------------------------------------
    dist_to_goal = np.sqrt((xs - goal_x) ** 2 + (ys - goal_y) ** 2)
    dist_to_goal = np.maximum(dist_to_goal, 1e-9)
    boundary_heat = 100000000.0 / dist_to_goal

    # ------------------------------------------------------------------
    # Initialise heat grid
    # ------------------------------------------------------------------
    heat = np.zeros((rows, cols))

    # Fix boundary values
    heat[on_boundary & free_mask] = boundary_heat[on_boundary & free_mask]

    # Interior free cells start at the mean boundary value as a warm start
    interior_free = free_mask & ~on_boundary
    if interior_free.any():
        heat[interior_free] = boundary_heat[on_boundary & free_mask].mean()

    # Obstacles and exterior → NaN (for display); solver ignores them
    heat[~inside_circle] = np.nan
    heat[obstacle_mask]   = np.nan

    # ------------------------------------------------------------------
    # Gauss-Seidel iteration (Laplace's equation on free interior cells)
    # ------------------------------------------------------------------
    solve_r, solve_c = np.where(interior_free)

    for iteration in tqdm(range(max_iters)):
        max_change = 0.0
        for i, j in zip(solve_r, solve_c):
            def neighbour(ni, nj):
                if not (0 <= ni < rows and 0 <= nj < cols):
                    return heat[i, j]          # reflect at grid edge
                if obstacle_mask[ni, nj] or not inside_circle[ni, nj]:
                    return heat[i, j]          # reflect at obstacle / exterior
                if np.isnan(heat[ni, nj]):
                    return heat[i, j]
                return heat[ni, nj]

            n_up    = neighbour(i - 1, j)
            n_down  = neighbour(i + 1, j)
            n_left  = neighbour(i, j - 1)
            n_right = neighbour(i, j + 1)

            new_val = (n_up + n_down + n_left + n_right) / 4.0
            change  = abs(new_val - heat[i, j])
            if change > max_change:
                max_change = change
            heat[i, j] = new_val

        if max_change < tol:
            print(f"Converged after {iteration + 1} iterations "
                  f"(max change = {max_change:.2e})")
            break
    else:
        print(f"Reached max iterations ({max_iters}); "
              f"last max change = {max_change:.2e}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if plot:
        _plot(heat, A, cx, cy, r, goal_x, goal_y, on_boundary, obstacle_mask,
              inside_circle)

    return heat


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot(heat, A, cx, cy, r, goal_x, goal_y,
          on_boundary, obstacle_mask, inside_circle):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             facecolor="#0d1117")

    cmap = LinearSegmentedColormap.from_list(
        "heat",
        ["#000033", "#0000ff", "#00ffff", "#ffff00", "#ff4400", "#ffffff"],
    )

    # --- Left panel: full heatmap ---
    ax = axes[0]
    ax.set_facecolor("#0d1117")

    display = heat.copy()
    im = ax.imshow(display, cmap=cmap, origin="upper", interpolation="bilinear")

    # Overlay obstacles in grey
    obs_display = np.zeros((*A.shape, 4))
    obs_display[obstacle_mask & inside_circle] = [0.4, 0.4, 0.4, 1.0]
    ax.imshow(obs_display, origin="upper")

    # Circle outline
    circle = plt.Circle((cx, cy), r, color="white", fill=False,
                         linewidth=1.5, linestyle="--", alpha=0.7)
    ax.add_patch(circle)

    # Goal marker
    ax.plot(goal_x, goal_y, marker="*", color="yellow",
            markersize=14, label="Goal", clip_on=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Heat", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Heatmap (steady-state diffusion)", color="white", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")

    # --- Right panel: boundary heat ring ---
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")

    bnd_display = np.full_like(heat, np.nan)
    bnd_display[on_boundary] = heat[on_boundary]

    ax2.imshow(bnd_display, cmap=cmap, origin="upper", interpolation="nearest")
    circle2 = plt.Circle((cx, cy), r, color="white", fill=False,
                          linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.add_patch(circle2)
    ax2.plot(goal_x, goal_y, marker="*", color="yellow",
             markersize=14, label="Goal", clip_on=False)

    ax2.set_title("Boundary heat ring  (1/dist_to_goal × 100)",
                  color="white", pad=10)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    plt.savefig("heatmap_output.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Plot saved to heatmap_output.png")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    A = load_ndarray('smaller_box_2d')

    '''r = 100.0          # circle radius in pixels
    d = 200.0          # distance from centre to goal (d > r → goal outside)
    goal_angle = 0.0 # goal direction in degrees from positive-x axis

    heat = generate_heatmap(d, r, A, goal_angle_deg=goal_angle,
                            max_iters=30_00, tol=1e-5, plot=True)

    interior_vals = heat[~np.isnan(heat)]
    print(f"\nHeat statistics inside the circle:")
    print(f"  min  = {interior_vals.min():.4f}")
    print(f"  max  = {interior_vals.max():.4f}")
    print(f"  mean = {interior_vals.mean():.4f}")'''

    min_corner_field = np.array([150,350])
    start = np.array([250,450])
    dim = len(start)
    new_min_corner = start - np.array([100]*dim)
    domain_size = 201
    min_corner = start
    domain = np.zeros(shape=[1]*dim)
    goal = np.array([250, 650])
    domain, min_corner = merge_domains_nd(domain, min_corner, new_min_corner, tuple([domain_size]*dim))
    WoS_altered_g(A, min_corner_field, goal, start, 100, 100000, domain, min_corner)
    cmap = LinearSegmentedColormap.from_list(
        "heat",
        ["#000033", "#0000ff", "#00ffff", "#ffff00", "#ff4400", "#ffffff"],
    )
    plt.imshow(domain, cmap=cmap)
    plt.savefig("heatmap_output_wos.png", dpi=150)
