import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from tqdm import tqdm
from helper import merge_domains_nd


def animate_heatmap_3d(
    heatmaps: list[np.ndarray],
    min_corners: list,
    paths: list,
    clip_percentile: float = 100,
    max_alpha: float = 0.3,
    threshold: float = 0.1,
    interval: int = 500,
    fps: int = 10,
    output_path: str = "forest_3d.gif",
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    max_shape = heatmaps[-1].shape

    def _set_axes():
        ax.set_xlim(0, max_shape[0])
        ax.set_ylim(0, max_shape[1])
        ax.set_zlim(0, max_shape[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    # --- helper: build scatter inputs for one frame ---
    def _prepare_frame(frame: int):
        hi, bye = merge_domains_nd(
            heatmaps[frame], min_corners[frame], min_corners[-1], heatmaps[-1].shape
        )

        data = np.clip(hi, 0, np.percentile(hi, clip_percentile))
        norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

        n = data.shape
        x, y, z = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij")

        v = norm.flatten()
        mask = v > threshold

        colors = cm.plasma(v[mask])
        colors[:, 3] = v[mask] * max_alpha

        xs = x.flatten()[mask]
        ys = y.flatten()[mask]
        zs = z.flatten()[mask]

        # shift path into the merged frame (same logic as your 2D version)
        path = np.array(paths[: frame + 1]).copy() - bye  # shape (N, 3)

        return xs, ys, zs, colors, path.T, data.min(), data.max()

    # --- seed frame 0 so axes are sized correctly from the start ---
    xs, ys, zs, colors, path, vmin, vmax = _prepare_frame(0)

    scatter = ax.scatter(xs, ys, zs, c=colors, marker="s", s=30, linewidths=0)
    (line,) = ax.plot(*path, color="red", linewidth=1.5)

    _set_axes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pbar = tqdm(total=len(heatmaps), desc="Rendering")

    def update(frame: int):
        ax.cla()  # 3-D scatter has no set_offsets3d that also updates colours,
                  # so clearing and redrawing is the cleanest approach

        xs, ys, zs, colors, path, vmin, vmax = _prepare_frame(frame)
        
        ax.scatter(xs, ys, zs, c=colors, marker="s", s=30, linewidths=0)
        ax.plot(*path, color="red", linewidth=1.5)
        _set_axes()  # restore limits after cla()
        ax.set_title(f"Frame {frame + 1} / {len(heatmaps)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        pbar.update(1)
        return []   # blit=False, so return value is ignored

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(heatmaps),
        interval=interval,
        blit=False,
    )

    ani.save(output_path, writer="pillow", fps=fps)
    pbar.close()
    plt.show()

def animate_2d(
    heatmaps: list[np.ndarray],
    min_corners: list,
    paths: list,
    idx: list,
    interval: int = 500,
    fps: int = 10,
    output_path: str = "2d.gif",):
    fig, ax = plt.subplots()

    # Initialise image with the final merged shape so the axes are sized correctly
    # from the start — this prevents imshow from locking in a stale extent on frame 0.
    initial_hi, _ = merge_domains_nd(
        heatmaps[0], min_corners[0], min_corners[-1], heatmaps[-1].shape
    )

    im = ax.imshow(
        initial_hi,
        cmap='inferno',
        animated=True,
        extent=[0, initial_hi.shape[1], initial_hi.shape[0], 0],  # [l, r, b, t]
        origin='upper',
    )
    plt.colorbar(im, ax=ax)
    scatter = ax.scatter([], [], c='white', s=10)
    line_object = ax.plot([], [], 'r-')[0]
    pbar = tqdm(total=len(heatmaps), desc='Rendering')
    valid = np.all((idx >= 0) & (idx < np.array(heatmaps[-1].shape)), axis=1)

    def update(frame):
        hi, bye = merge_domains_nd(
            heatmaps[frame], min_corners[frame], min_corners[-1], heatmaps[-1].shape
        )
        
        hi[idx[valid, 0], idx[valid, 1]] = np.nan
        hi[(hi == -1)] = np.nan
        domain_mask = np.zeros(hi.shape, dtype=bool)
        rel_a = np.round(np.array(min_corners[frame]) - bye).astype(int)
        rel_a = np.maximum(rel_a, 0)
        slices_a = tuple(slice(int(rel_a[d]), int(rel_a[d]) + heatmaps[frame].shape[d]) for d in range(hi.ndim))
        domain_mask[slices_a] = True

        hi[~domain_mask] = np.nan  # outside domain → nan

        

        # --- image ---
        im.set_data(hi)

        # Update the extent so every pixel in `hi` maps to exactly 1 data unit.
        # Without this the image is stretched/squashed whenever the merged shape grows.
        im.set_extent([0, hi.shape[1], hi.shape[0], 0])  # [left, right, bottom, top]

        # Keep axes limits in sync with the (possibly growing) image.
        ax.set_xlim(0, hi.shape[1])
        ax.set_ylim(hi.shape[0], 0)   # inverted: row 0 at top, matching origin='upper'

        ax.set_title(f'Frame {frame + 1} / {len(heatmaps)}')

        # --- path ---
        # Shift path coordinates into the merged array's local frame.
        # `bye` must be [row, col] order to match `paths`.
        path = np.array(paths[: frame + 2]).copy() - bye

        # imshow convention: x-axis = columns, y-axis = rows
        line_object.set_data(path[:, 1], path[:, 0])
        return [im, line_object]


    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(heatmaps),
        interval=interval,
        blit=False,
    )

    ani.save(output_path, writer='pillow', fps=fps, progress_callback=lambda i, n: pbar.update(1))
    pbar.close()
    plt.show()