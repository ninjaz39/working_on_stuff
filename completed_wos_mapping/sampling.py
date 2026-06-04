import numpy as np

def sample_sphere(DIM, r, center):
    """Sample a random point on the surface of a sphere."""
    vec = np.random.normal(0, 1, DIM)
    norm = np.sum(vec**2)**0.5
    vec = vec / norm  # Normalize to lie on the unit sphere
    return center + r * vec

def sample_ball(DIM, r, center):
    """Sample a random point inside a ball."""
    sample = np.array(np.random.normal(0, 1, DIM + 2))
    norm = np.sum(sample**2)**0.5
    sample = sample / norm  # Normalize to lie on the unit sphere
    sample = sample[:DIM]  # Take only the first DIM dimensions
    return center + sample * r

def sample_around_point(arr, center, sigma = 10):
    center = np.asarray(center)
    offset = np.random.randn(arr.ndim) * sigma
    idx = tuple(np.clip(np.round(center + offset).astype(int), 0, np.array(arr.shape) - 1))
    return idx