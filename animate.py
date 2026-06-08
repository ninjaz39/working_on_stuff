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
    fps: int = 100,
    output_path: str = "2d.mp4",):

    import imageio

    fig, ax = plt.subplots()

    initial_hi, _ = merge_domains_nd(
        heatmaps[0], min_corners[0], min_corners[-1], heatmaps[-1].shape
    )

    im = ax.imshow(
        initial_hi,
        cmap='inferno',
        animated=True,
        extent=[0, initial_hi.shape[1], initial_hi.shape[0], 0],
        origin='upper',
    )
    plt.colorbar(im, ax=ax)
    scatter = ax.scatter([], [], c='white', s=10)
    line_object = ax.plot([], [], 'r-')[0]
    valid = np.all((idx >= 0) & (idx < np.array(heatmaps[-1].shape)), axis=1)

    def update(frame):
        hi = np.full(heatmaps[-1].shape, np.nan)

        rel_a = np.round(np.array(min_corners[frame]) - min_corners[-1]).astype(int)
        slices_a = tuple(slice(int(rel_a[d]), int(rel_a[d]) + heatmaps[frame].shape[d]) for d in range(hi.ndim))

        hi[slices_a] = heatmaps[frame]
        hi[idx[valid, 0], idx[valid, 1]] = np.nan
        hi[(hi == -1)] = np.nan

        im.set_data(hi)
        im.set_extent([0, hi.shape[1], hi.shape[0], 0])
        ax.set_xlim(0, hi.shape[1])
        ax.set_ylim(hi.shape[0], 0)
        ax.set_title(f'Frame {frame + 1} / {len(heatmaps)}')

        path = np.array(paths[: frame + 2]).copy() - min_corners[-1]
        line_object.set_data(path[:, 1], path[:, 0])

    # --- render frames manually with imageio ---
    pbar = tqdm(total=len(heatmaps), desc='Rendering')
    writer = imageio.get_writer(output_path, fps=fps)

    for i in range(len(heatmaps)):
        update(i)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:]  # ARGB → RGB
        writer.append_data(buf)
        pbar.update(1)

    writer.close()
    pbar.close()
    plt.show()