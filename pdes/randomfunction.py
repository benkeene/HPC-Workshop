"""
An example of creating a random function generator using basis functions in PyTorch.
"""

import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
import matplotlib.pyplot as plt

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
torch.set_default_device(device)


class RandomFunctionGenerator(ABC):
    """Abstract base class for random function generators using basis functions."""

    @abstractmethod
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a random function evaluated at points x."""
        pass

    @abstractmethod
    def get_basis_functions(self) -> List[torch.Tensor]:
        """Get the basis functions used in the generator."""
        pass


class FourierRandomFunctionGenerator(RandomFunctionGenerator):
    def __init__(
        self,
        dim: int,
        n_modes: int,
        device: str,
        resolution: int = 100,
    ):
        self.dim = dim
        self.n_modes = n_modes
        self.resolution = resolution
        self.device = device

        # Store basis function generators instead of pre-computed basis functions
        self.basis_function_generators = []
        self.coefficients = torch.randn(
            n_modes * (2 if dim == 1 else 4), dtype=torch.float32
        )

        if dim == 1:
            for k in range(1, self.n_modes + 1):  # Start from 1 to avoid zero frequency
                scaling = self._1d_basis_scaling(k)
                self.basis_function_generators.append(
                    lambda x, k=k: torch.randn(1) * scaling * torch.sin(k * x)
                )
                self.basis_function_generators.append(
                    lambda x, k=k: torch.randn(1) * scaling * torch.cos(k * x)
                )
        elif dim == 2:
            for kx in range(1, self.n_modes + 1):
                for ky in range(1, self.n_modes + 1):
                    scaling = self._2d_basis_scaling(kx, ky)
                    self.basis_function_generators.append(
                        lambda X, Y, kx=kx, ky=ky: (
                            torch.randn(1)
                            * scaling
                            * torch.sin(kx * X)
                            * torch.sin(ky * Y)
                        ).to(self.device)
                    )
                    self.basis_function_generators.append(
                        lambda X, Y, kx=kx, ky=ky: (
                            torch.randn(1)
                            * scaling
                            * torch.cos(kx * X)
                            * torch.cos(ky * Y)
                        ).to(self.device)
                    )
                    self.basis_function_generators.append(
                        lambda X, Y, kx=kx, ky=ky: (
                            torch.randn(1)
                            * scaling
                            * torch.sin(kx * X)
                            * torch.cos(ky * Y)
                        ).to(self.device)
                    )
                    self.basis_function_generators.append(
                        lambda X, Y, kx=kx, ky=ky: (
                            torch.randn(1)
                            * scaling
                            * torch.cos(kx * X)
                            * torch.sin(ky * Y)
                        ).to(self.device)
                    )
        else:
            raise ValueError("Only 1D and 2D Fourier basis functions are supported.")

    def _1d_basis_scaling(self, k: int) -> float:
        """Scale the basis function for 1D Fourier series."""
        return 1.0 / (k**2)

    def _2d_basis_scaling(self, kx: int, ky: int) -> float:
        """Scale the basis function for 2D Fourier series."""
        return 1 / ((kx * ky) ** 2)

    def generate(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Generate a random function evaluated at points x (and y for 2D)."""

        if self.dim == 1:
            assert y is None, "y should not be provided for 1D functions"
            result = torch.zeros_like(x, dtype=torch.float32)
            for i, basis_gen in enumerate(self.basis_function_generators):
                result += basis_gen(x)
        elif self.dim == 2:
            assert y is not None, "y must be provided for 2D functions"
            # Ensure x and y are broadcastable
            if x.dim() == 1 and y.dim() == 1:
                X, Y = torch.meshgrid(x, y, indexing="ij")
            else:
                X, Y = x, y
            result = torch.zeros_like(X, dtype=torch.float32)
            for i, basis_gen in enumerate(self.basis_function_generators):
                result += basis_gen(X, Y)

        return result

    def get_basis_functions(self) -> List:
        """Get the basis function generators used in the generator."""
        return self.basis_function_generators


if __name__ == "__main__":
    n_modes = 4

    print("1D Example:")
    dim = 1
    generator_1d = FourierRandomFunctionGenerator(dim, n_modes, device=device)

    x = torch.linspace(0, 2 * torch.pi, 100)

    random_function_1d = generator_1d.generate(x)
    print(f"1D function shape: {random_function_1d.shape}")
    print(f"1D function values (first 5): {random_function_1d[:5]}")

    print("\n2D Example:")
    dim = 2
    generator_2d = FourierRandomFunctionGenerator(dim, n_modes, device=device)

    x = torch.linspace(0, 2 * torch.pi, 50)
    y = torch.linspace(0, 2 * torch.pi, 50)

    random_function_2d = generator_2d.generate(x, y)
    print(f"2D function shape: {random_function_2d.shape}")
    print(f"2D function value at (0,0): {random_function_2d[0, 0]}")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 5))

        # Plot 1D function
        ax1 = fig.add_subplot(131)
        x_1d = torch.linspace(0, 2 * torch.pi, 100)
        ax1.plot(x_1d.cpu().numpy(), random_function_1d.cpu().numpy())
        ax1.set_title("1D Random Fourier Function")
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.grid(True)

        ax2 = fig.add_subplot(132)
        im = ax2.imshow(
            random_function_2d.cpu().numpy(),
            extent=[0, 2 * torch.pi, 0, 2 * torch.pi],
            origin="lower",
            cmap="viridis",
        )
        ax2.set_title("2D Random Fourier Function (Heatmap)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.colorbar(im, ax=ax2)

        ax3 = fig.add_subplot(133, projection="3d")
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        Z_np = random_function_2d.cpu().numpy()

        surface = ax3.plot_surface(X_np, Y_np, Z_np, cmap="viridis", alpha=0.8)
        ax3.set_title("2D Random Fourier Function (3D Surface)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("f(x,y)")
        fig.colorbar(surface, ax=ax3, shrink=0.5)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")
