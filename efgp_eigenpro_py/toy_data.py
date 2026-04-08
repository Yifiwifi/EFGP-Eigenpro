import numpy as np


def generate_toy_data(n_samples=200, nu=1.0, sigma_eps=0.1, seed=42):
    rng = np.random.default_rng(seed)
    xi = rng.uniform(-0.5, 0.5, size=n_samples)
    eps = rng.normal(0.0, sigma_eps, size=n_samples)
    yi = np.cos(2.0 * np.pi * nu * xi) + eps
    return xi, yi


if __name__ == "__main__":
    x, y = generate_toy_data()
    print("x shape:", x.shape)
    print("y shape:", y.shape)
