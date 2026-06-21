import numpy as np
import matplotlib.pyplot as plt
def load_ndarray(name: str) -> np.ndarray:
    """Companion loader that reconstructs the array from the saved file."""
    with open(name, "r") as f:
        n     = int(f.readline().strip())
        shape = tuple(int(s) for s in f.readline().strip().split())
        flat  = list(map(float, f.readline().strip().split()))

    return np.array(flat).reshape(shape)
hi = (load_ndarray(r'C:\Users\chuen\Desktop\working_on_stuff\ref_data_1')).reshape(201,201)
idx = np.argwhere(hi <= 100)
#plot_heatmap_3d(solved_domain, temp-min_corner)
plt.imshow(hi, cmap='inferno')
plt.scatter(idx[:, 1], idx[:, 0], c='w')
plt.show()