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

    initial_hi = heatmaps[0]
    '''merge_domains_nd(
        heatmaps[0], min_corners[0], min_corners[-1], heatmaps[-1].shape
    )'''

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

'''def animate_3d(
    heatmaps: list[np.ndarray],
    min_corners: list,
    paths: list,
    idx: list,
    fps: int = 100,
    output_path: str = "3d.mp4",
):
    """
    Animate 3D heatmaps with path overlay across 3 subplots side by side:
      - Left:   X-Y  (top-down, max-projected over Z)
      - Middle: X-Z  (side view, max-projected over Y)
      - Right:  Y-Z  (front view, max-projected over X)

    Parameters
    ----------
    heatmaps    : list of np.ndarray, shape (X, Y, Z)
    min_corners : list of array-like, length-3 offsets per frame + final
    paths       : list of [x, y, z] waypoints (one per frame + 1)
    idx         : array of shape (N, 3) — visited cell indices to mask as NaN
    fps         : frames per second for the output video
    output_path : where to save the .mp4
    """
    import imageio
    from tqdm import tqdm

    idx = np.array(idx)
    final_shape = heatmaps[-1].shape  # (X, Y, Z)
    valid = np.all((idx >= 0) & (idx < np.array(final_shape)), axis=1)

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(20, 6))
    fig.tight_layout(pad=3.0)

    # ── build merged volume for a given frame ──────────────────────────────
    def build_volume(frame: int, axis: int) -> np.ndarray:
        hi = np.full(final_shape, np.nan)
        rel = np.round(np.array(min_corners[frame]) - min_corners[-1]).astype(int)
        slices = tuple(
            slice(int(rel[d]), int(rel[d]) + heatmaps[frame].shape[d])
            for d in range(hi.ndim)
        )
        hi[slices] = heatmaps[frame]
        hi = np.nanmax(hi, axis=axis)
        if axis == 0:
            hi[idx[valid, 1], idx[valid, 2]] = np.nan
        elif axis == 1:
            hi[idx[valid, 0], idx[valid, 2]] = np.nan
        else:
            hi[idx[valid, 0], idx[valid, 1]] = np.nan
        hi[hi == -1] = np.nan
        return hi

    xy_proj = build_volume(0, axis=2)        # (X, Y) — collapse Z
    xz_proj = build_volume(0, axis=1).T      # (Z, X) — collapse Y, Z vertical
    yz_proj = build_volume(0, axis=0).T      # (Z, Y) — collapse X, Z vertical

    vmin = min(np.nanmin(h) for h in heatmaps)
    vmax = max(np.nanmax(h) for h in heatmaps)

    imshow_kwargs = dict(cmap="inferno", animated=True, aspect="auto",
                         vmin=vmin, vmax=vmax)

    im_xy = ax_xy.imshow(xy_proj, origin="upper",
                         extent=[0, xy_proj.shape[1], xy_proj.shape[0], 0],
                         **imshow_kwargs)
    im_xz = ax_xz.imshow(xz_proj, origin="lower",
                         extent=[0, xz_proj.shape[1], 0, xz_proj.shape[0]],
                         **imshow_kwargs)
    im_yz = ax_yz.imshow(yz_proj, origin="lower",
                         extent=[0, yz_proj.shape[1], 0, yz_proj.shape[0]],
                         **imshow_kwargs)

    for ax, im in [(ax_xy, im_xy), (ax_xz, im_xz), (ax_yz, im_yz)]:
        plt.colorbar(im, ax=ax)

    ax_xy.set(xlabel="Y", ylabel="X", title="X–Y  (top-down)")
    ax_xz.set(xlabel="X", ylabel="Z", title="X–Z  (side view)")
    ax_yz.set(xlabel="Y", ylabel="Z", title="Y–Z  (front view)")

    # Path lines and current-position dots
    line_xy = ax_xy.plot([], [], "r-", linewidth=1.2)[0]
    line_xz = ax_xz.plot([], [], "r-", linewidth=1.2)[0]
    line_yz = ax_yz.plot([], [], "r-", linewidth=1.2)[0]
    dot_xy  = ax_xy.scatter([], [], c="white", s=30, zorder=5)
    dot_xz  = ax_xz.scatter([], [], c="white", s=30, zorder=5)
    dot_yz  = ax_yz.scatter([], [], c="white", s=30, zorder=5)

    # ── per-frame update ───────────────────────────────────────────────────
    def update(frame: int):

        xy = build_volume(frame, axis=2)        # (X, Y)
        xz = build_volume(frame, axis=1).T      # (Z, X)
        yz = build_volume(frame, axis=0).T      # (Z, Y)

        im_xy.set_data(xy)
        im_xy.set_extent([0, xy.shape[1], xy.shape[0], 0])
        ax_xy.set_xlim(0, xy.shape[1])
        ax_xy.set_ylim(xy.shape[0], 0)

        im_xz.set_data(xz)
        im_xz.set_extent([0, xz.shape[1], 0, xz.shape[0]])
        ax_xz.set_xlim(0, xz.shape[1])
        ax_xz.set_ylim(0, xz.shape[0])

        im_yz.set_data(yz)
        im_yz.set_extent([0, yz.shape[1], 0, yz.shape[0]])
        ax_yz.set_xlim(0, yz.shape[1])
        ax_yz.set_ylim(0, yz.shape[0])

        fig.suptitle(f"Frame {frame + 1} / {len(heatmaps)}", fontsize=13)

        # Path trail shifted by global min_corner offset
        path = np.array(paths[: frame + 2]) - min_corners[-1]  # (N, 3)
        px, py, pz = path[:, 0], path[:, 1], path[:, 2]

        # X–Y: x is row (ylabel), y is col (xlabel) — matches imshow origin='upper'
        line_xy.set_data(py, px)
        dot_xy.set_offsets([[py[-1], px[-1]]])

        # X–Z: x along horizontal, z along vertical
        line_xz.set_data(px, pz)
        dot_xz.set_offsets([[px[-1], pz[-1]]])

        # Y–Z: y along horizontal, z along vertical
        line_yz.set_data(py, pz)
        dot_yz.set_offsets([[py[-1], pz[-1]]])

    # ── render to mp4 ─────────────────────────────────────────────────────
    pbar = tqdm(total=len(heatmaps), desc="Rendering 3D")
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
    plt.show()'''

def animate_3d(
    heatmaps: list[np.ndarray],
    min_corners: list,
    paths: list,
    idx: list,
    fps: int = 100,
    output_path: str = "3d_true.mp4",
    elev: float = 25,
    azim_start: float = -90,
    azim_range: float = 270,
    obstacle_color: str = "#444444",
    obstacle_alpha: float = 0.9,
):
    """
    Animate heatmaps as true 3D scatter/volume with a path overlay,
    and obstacles (from `idx`) explicitly rendered as a separate solid layer.

    Parameters
    ----------
    heatmaps       : list of np.ndarray, shape (X, Y, Z)
    min_corners    : list of array-like, length-3 offsets per frame + final
    paths          : list of [x, y, z] waypoints (one per frame + 1)
    idx            : array of shape (N, 3) — visited/obstacle cell indices
    fps            : frames per second for the output video
    output_path    : where to save the .mp4
    elev           : fixed camera elevation angle (degrees)
    azim_start     : starting azimuth angle (degrees)
    azim_range     : total degrees of pan over the full animation
    obstacle_color : color used to render obstacle voxels
    obstacle_alpha : opacity for obstacle voxels (kept high/constant so they
                     never fade out the way value-scaled heatmap voxels do)
    """
    import imageio
    from tqdm import tqdm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    idx = np.array(idx)
    final_shape = heatmaps[-1].shape  # (X, Y, Z)
    valid = np.all((idx >= 0) & (idx < np.array(final_shape)), axis=1)
    obstacle_idx = idx[valid]  # (M, 3) — coordinates of obstacle cells

    vmin = min(np.nanmin(h) for h in heatmaps)
    vmax = max(np.nanmax(h) for h in heatmaps)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def build_volume(frame: int) -> np.ndarray:
        """Place the frame's heatmap into the final-shape canvas (obstacles NOT masked here)."""
        vol = np.full(final_shape, np.nan)
        rel = np.round(np.array(min_corners[frame]) - min_corners[-1]).astype(int)
        slices = tuple(
            slice(int(rel[d]), int(rel[d]) + heatmaps[frame].shape[d])
            for d in range(vol.ndim)
        )
        vol[slices] = heatmaps[frame]
        vol[vol == -1] = np.nan
        return vol

    X, Y, Z = np.meshgrid(
        np.arange(final_shape[0]),
        np.arange(final_shape[1]),
        np.arange(final_shape[2]),
        indexing="ij",
    )

    all_vals = np.concatenate([h[~np.isnan(h)].ravel() for h in heatmaps])
    threshold = np.nanpercentile(all_vals, 60)

    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    pbar = tqdm(total=len(heatmaps), desc="Rendering 3D")
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in range(len(heatmaps)):
        ax.cla()

        vol = build_volume(frame)

        # Mask obstacle cells OUT of the heatmap layer so they don't double-render
        # with a faded heatmap color underneath the obstacle marker
        if obstacle_idx.size > 0:
            vol[obstacle_idx[:, 0], obstacle_idx[:, 1], obstacle_idx[:, 2]] = np.nan

        # ── Heatmap voxel layer ──────────────────────────────────────────
        mask = (~np.isnan(vol)) & (vol > threshold)
        xs, ys, zs, vs = X[mask], Y[mask], Z[mask], vol[mask]

        if xs.size > 0:
            colors = cmap(norm(vs))
            colors[:, 3] = 0.15 + 0.75 * (vs - vmin) / max(vmax - vmin, 1e-9)
            ax.scatter(
                xs, ys, zs,
                c=colors,
                s=18,
                marker="s",
                linewidths=0,
                depthshade=True,
                label="Heatmap",
            )

        # ── Obstacle layer (explicit, solid, distinct marker) ────────────
        if obstacle_idx.size > 0:
            ax.scatter(
                obstacle_idx[:, 0], obstacle_idx[:, 1], obstacle_idx[:, 2],
                c=obstacle_color,
                s=26,
                marker="X",          # visually distinct from heatmap squares
                alpha=obstacle_alpha,
                linewidths=0,
                depthshade=False,    # keep obstacles a flat, constant color
                zorder=9,
                label="Obstacle",
            )

        # ── Path trail ────────────────────────────────────────────────────
        path = np.array(paths[: frame + 2]) - min_corners[-1]
        if len(path) > 1:
            ax.plot(
                path[:, 0], path[:, 1], path[:, 2],
                color="red", linewidth=1.5, alpha=0.85, zorder=10,
            )
        ax.scatter(
            [path[-1, 0]], [path[-1, 1]], [path[-1, 2]],
            color="white", s=60, zorder=11, edgecolors="red", linewidths=1.2,
            label="Current position",
        )

        # ── Legend (only build once visually, but cheap enough to redo) ──
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = dict(zip(labels, handles))
            ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)

        # ── Camera pan ────────────────────────────────────────────────────
        t = frame / max(len(heatmaps) - 1, 1)
        azim = azim_start + t * azim_range
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(0, final_shape[0])
        ax.set_ylim(0, final_shape[1])
        ax.set_zlim(0, final_shape[2])
        fig.suptitle(f"Frame {frame + 1} / {len(heatmaps)}", fontsize=13)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, 1:]
        writer.append_data(buf)
        pbar.update(1)

    writer.close()
    pbar.close()
    plt.show()

def animate_4d(
    heatmaps: list[np.ndarray],
    min_corners: list,
    paths: list,
    idx: list,
    fps: int = 100,
    output_path: str = "4d.mp4",
):
    """
    Animate 4D heatmaps (X, Y, Z, T) with path overlay across 3 subplots:
      - Left:   X-Y  (top-down, max-projected over Z)
      - Middle: X-Z  (side view, max-projected over Y)
      - Right:  Y-Z  (front view, max-projected over X)

    At each animation frame, only the heatmap slice at the matching time index
    T is used before projecting down to 2D.

    Parameters
    ----------
    heatmaps    : list of np.ndarray, shape (X, Y, Z, T)
    min_corners : list of array-like, length-4 offsets per frame + final
                  (x_offset, y_offset, z_offset, t_offset)
    paths       : list of [x, y, z, t] waypoints (one per frame + 1)
    idx         : array of shape (N, 4) — visited cell indices to mask as NaN
    fps         : frames per second for the output video
    output_path : where to save the .mp4
    """
    import imageio
    from tqdm import tqdm

    idx = np.array(idx)
    final_shape = heatmaps[-1].shape  # (X, Y, Z, T)
    valid = np.all((idx >= 0) & (idx < np.array(final_shape)), axis=1)

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(20, 6))
    fig.tight_layout(pad=3.0)

    # ── build merged 4D volume for a given animation frame ─────────────────
    def build_volume(frame: int) -> np.ndarray:
        hi = np.full(final_shape, np.nan)
        rel = np.round(np.array(min_corners[frame]) - min_corners[-1]).astype(int)
        slices = tuple(
            slice(int(rel[d]), int(rel[d]) + heatmaps[frame].shape[d])
            for d in range(hi.ndim)
        )
        hi[slices] = heatmaps[frame]
        hi[idx[valid, 0], idx[valid, 1], idx[valid, 2], idx[valid, 3]] = np.nan
        hi[hi == -1] = np.nan
        return hi  # (X, Y, Z, T)

    # ── resolve the T index for the current animation frame ────────────────
    def get_t_index(frame: int) -> int:
        # path[frame] is [x, y, z, t]; subtract global t offset to get local index
        t_global = paths[frame][3]
        t_offset = min_corners[-1][3]
        return int(round(t_global - t_offset))

    # ── initial render ─────────────────────────────────────────────────────
    hi0 = build_volume(0)
    t0  = np.clip(get_t_index(0), 0, final_shape[3] - 1)
    vol0 = hi0[:, :, :, t0]  # (X, Y, Z)

    xy_proj = np.nanmax(vol0, axis=2)      # (X, Y)
    xz_proj = np.nanmax(vol0, axis=1).T   # (Z, X)
    yz_proj = np.nanmax(vol0, axis=0).T   # (Z, Y)

    vmin = min(np.nanmin(h) for h in heatmaps)
    vmax = max(np.nanmax(h) for h in heatmaps)

    imshow_kwargs = dict(cmap="inferno", animated=True, aspect="auto",
                         vmin=vmin, vmax=vmax)

    im_xy = ax_xy.imshow(xy_proj, origin="upper",
                         extent=[0, xy_proj.shape[1], xy_proj.shape[0], 0],
                         **imshow_kwargs)
    im_xz = ax_xz.imshow(xz_proj, origin="lower",
                         extent=[0, xz_proj.shape[1], 0, xz_proj.shape[0]],
                         **imshow_kwargs)
    im_yz = ax_yz.imshow(yz_proj, origin="lower",
                         extent=[0, yz_proj.shape[1], 0, yz_proj.shape[0]],
                         **imshow_kwargs)

    for ax, im in [(ax_xy, im_xy), (ax_xz, im_xz), (ax_yz, im_yz)]:
        plt.colorbar(im, ax=ax)

    ax_xy.set(xlabel="Y", ylabel="X", title="X–Y  (top-down)")
    ax_xz.set(xlabel="X", ylabel="Z", title="X–Z  (side view)")
    ax_yz.set(xlabel="Y", ylabel="Z", title="Y–Z  (front view)")

    line_xy = ax_xy.plot([], [], "r-", linewidth=1.2)[0]
    line_xz = ax_xz.plot([], [], "r-", linewidth=1.2)[0]
    line_yz = ax_yz.plot([], [], "r-", linewidth=1.2)[0]
    dot_xy  = ax_xy.scatter([], [], c="white", s=30, zorder=5)
    dot_xz  = ax_xz.scatter([], [], c="white", s=30, zorder=5)
    dot_yz  = ax_yz.scatter([], [], c="white", s=30, zorder=5)

    # ── per-frame update ───────────────────────────────────────────────────
    def update(frame: int):
        hi  = build_volume(frame)                                   # (X, Y, Z, T)
        t   = np.clip(get_t_index(frame), 0, final_shape[3] - 1)
        vol = hi[:, :, :, t]                                        # (X, Y, Z)

        xy = np.nanmax(vol, axis=2)      # (X, Y)
        xz = np.nanmax(vol, axis=1).T   # (Z, X)
        yz = np.nanmax(vol, axis=0).T   # (Z, Y)

        im_xy.set_data(xy)
        im_xy.set_extent([0, xy.shape[1], xy.shape[0], 0])
        ax_xy.set_xlim(0, xy.shape[1])
        ax_xy.set_ylim(xy.shape[0], 0)

        im_xz.set_data(xz)
        im_xz.set_extent([0, xz.shape[1], 0, xz.shape[0]])
        ax_xz.set_xlim(0, xz.shape[1])
        ax_xz.set_ylim(0, xz.shape[0])

        im_yz.set_data(yz)
        im_yz.set_extent([0, yz.shape[1], 0, yz.shape[0]])
        ax_yz.set_xlim(0, yz.shape[1])
        ax_yz.set_ylim(0, yz.shape[0])

        fig.suptitle(
            f"Frame {frame + 1} / {len(heatmaps)}  |  T = {t}",
            fontsize=13,
        )

        # Path trail — only points whose T matches current slice
        path_arr = np.array(paths[: frame + 2]) - min_corners[-1]  # (N, 4)
        t_offset = min_corners[-1][3]
        mask = np.abs(path_arr[:, 3]) < 0.5    # points at current T slice ± 0.5
        px, py, pz = path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]

        # Full trail (faint) + masked trail (solid) on each plot
        for line, dot, hx, hy, m_hx, m_hy in [
            (line_xy, dot_xy, py,  px,  py[mask],  px[mask]),
            (line_xz, dot_xz, px,  pz,  px[mask],  pz[mask]),
            (line_yz, dot_yz, py,  pz,  py[mask],  pz[mask]),
        ]:
            line.set_data(m_hx, m_hy)
            if len(m_hx):
                dot.set_offsets([[m_hx[-1], m_hy[-1]]])
            else:
                dot.set_offsets(np.empty((0, 2)))

    # ── render to mp4 ─────────────────────────────────────────────────────
    pbar = tqdm(total=len(heatmaps), desc="Rendering 4D")
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