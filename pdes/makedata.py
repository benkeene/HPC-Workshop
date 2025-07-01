import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neuralop.losses.finite_diff import central_diff_2d
import os
import time
from datetime import datetime
from randomfunction import FourierRandomFunctionGenerator
import pprint


class HeatEquationDatasetGenerator:
    def __init__(self, **kwargs):
        """Initialize generator with default parameters."""
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

        # Default parameters
        self.params = {
            "n_samples": 100,
            "nx": 128,
            "ny": 128,
            "Lx": 2.0 * torch.pi,
            "Ly": 2.0 * torch.pi,
            "T": 0.5,
            "dt": 0.001,
            "alpha": 1,
            "save_evolution": True,
            "output_dir": "heat_data",
            "batch_size": 1000,
            "n_fourier_modes": 3,
            "dim": 2,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        self.params.update(kwargs)

        self.nt = int(self.params["T"] / self.params["dt"])
        self.dx = torch.tensor(
            self.params["Lx"] / (self.params["nx"] - 1), device=self.device
        )
        self.dy = torch.tensor(
            self.params["Ly"] / (self.params["ny"] - 1), device=self.device
        )
        self.dt_alpha = torch.tensor(
            self.params["dt"] * self.params["alpha"], device=self.device
        )

        # Create output directory
        os.makedirs(self.params["output_dir"], exist_ok=True)

        self._setup_grids()

        self._check_stability()

    def _setup_grids(self):
        """Set up spatial grids for computation."""
        x = torch.linspace(0, self.params["Lx"], self.params["nx"], device=self.device)
        y = torch.linspace(0, self.params["Ly"], self.params["ny"], device=self.device)

        self.X = x.repeat(self.params["ny"], 1).T
        self.Y = y.repeat(self.params["nx"], 1)

        assert self.X.shape == (self.params["nx"], self.params["ny"])
        assert self.Y.shape == (self.params["nx"], self.params["ny"])

    def _check_stability(self):
        """Check Von Neumann stability condition."""
        stability_param = self.params["dt"] / (
            self.params["alpha"] * (1 / (self.dx**2) + 1 / (self.dy**2))
        )
        if stability_param > 0.5:
            print(
                f"Warning: Von Neumann Condition violated: {stability_param:.4f} > 0.5"
            )

    def generate_initial_conditions(self):
        """Generate all initial conditions using Fourier random functions."""
        print(f"Generating all {self.params['n_samples']} initial conditions...")

        self.initial_conditions = torch.zeros(
            self.params["n_samples"],
            self.params["nx"],
            self.params["ny"],
            device=self.device,
        )

        generator = FourierRandomFunctionGenerator(
            dim=2, n_modes=self.params["n_fourier_modes"], device=self.device
        )

        for i in range(self.params["n_samples"]):
            u = generator.generate(self.X, self.Y).to(self.device)
            self.initial_conditions[i] = u

    def solve_heat_equation(self):
        """Solve heat equation using finite differences with periodic boundaries."""
        print(f"Processing all {self.params['n_samples']} samples simultaneously...")

        # Initialize solution
        u = self.initial_conditions.clone()

        # Setup evolution storage
        save_interval = 50
        n_snapshots = (self.nt // save_interval) + 1
        self.evolution = torch.zeros(
            n_snapshots,
            self.params["n_samples"],
            self.params["nx"],
            self.params["ny"],
            device=self.device,
        )

        # Store initial condition
        self.evolution[0] = u.clone()
        snapshot_idx = 1

        print("Starting vectorized time evolution...")

        # Time evolution loop
        for n in range(self.nt):
            if n % 100 == 0:
                print(f"Time step {n}/{self.nt}")

            # Save snapshots
            if n % save_interval == 0 and n > 0:
                self.evolution[snapshot_idx] = u.clone()
                snapshot_idx += 1

            # Compute Laplacian using central differences with periodic boundaries
            du_dx, du_dy = central_diff_2d(
                u, [self.dx, self.dy], fix_x_bnd=False, fix_y_bnd=False
            )
            d2u_dx2, _ = central_diff_2d(
                du_dx, [self.dx, self.dy], fix_x_bnd=False, fix_y_bnd=False
            )
            _, d2u_dy2 = central_diff_2d(
                du_dy, [self.dx, self.dy], fix_x_bnd=False, fix_y_bnd=False
            )

            # Update solution: u_new = u + dt * alpha * laplacian
            laplacian = d2u_dx2 + d2u_dy2
            u = u + self.dt_alpha * laplacian

        # Store final snapshot
        if snapshot_idx < n_snapshots:
            self.evolution[snapshot_idx] = u.clone()

        self.final_solutions = u

    def create_animation(self, sample_idx=0):
        """Create animation of heat evolution for a specific sample."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        plt.subplots_adjust(
            left=0.052, bottom=0.11, right=0.957, top=0.848, wspace=0.243, hspace=0.2
        )

        # Get evolution data for selected sample
        sample_evolution = self.evolution[:, sample_idx, :, :].cpu().numpy()
        global_min = sample_evolution.min()
        global_max = sample_evolution.max()

        # Create initial plots and colorbars (these will persist)
        levels = np.linspace(global_min, global_max, 20)
        initial_data = sample_evolution[0]

        # Initialize contour plot
        contour_plot = ax1.contourf(
            self.X.cpu(),
            self.Y.cpu(),
            initial_data,
            levels=levels,
            cmap="hot",
            vmin=global_min,
            vmax=global_max,
        )
        cbar1 = plt.colorbar(contour_plot, ax=ax1, shrink=0.8)
        cbar1.set_label("Temperature", rotation=270, labelpad=15)

        # Initialize heatmap
        heatmap_plot = ax2.imshow(
            initial_data.T,
            cmap="hot",
            origin="lower",
            extent=[0, self.params["Lx"], 0, self.params["Ly"]],
            vmin=global_min,
            vmax=global_max,
        )
        cbar2 = plt.colorbar(heatmap_plot, ax=ax2, shrink=0.8)
        cbar2.set_label("Temperature", rotation=270, labelpad=15)

        def animate(frame):
            current_data = sample_evolution[frame]
            current_min = current_data.min()
            current_max = current_data.max()

            ax1.clear()
            ax1.contourf(
                self.X.cpu(),
                self.Y.cpu(),
                current_data,
                levels=levels,
                cmap="hot",
                vmin=global_min,
                vmax=global_max,
            )
            ax1.set_title(
                f"Heat Distribution (Contour) - Step {frame * 50}\n"
                f"Max: {current_max:.4f}, Min: {current_min:.4f}"
            )
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")

            # Update heatmap data (more efficient than recreating)
            heatmap_plot.set_array(current_data.T)
            ax2.set_title(
                f"Heat Distribution (Heatmap) - Step {frame * 50}\n"
                f"Max: {current_max:.4f}, Min: {current_min:.4f}"
            )
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")

            return [heatmap_plot]  # Return artists that changed

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(sample_evolution),
            interval=30,
            repeat=True,
            blit=False,
        )

        # Save animation
        gif_path = f"{self.params['output_dir']}/heat_equation_evolution_{self.params['timestamp']}.gif"
        anim.save(gif_path, writer="pillow", fps=3)
        print(f"Animation saved to {gif_path}")
        plt.close()

    def create_visualization(self, sample_idx=0):
        """Create visualization comparing initial and final states."""

        initial = self.initial_conditions[sample_idx].cpu().numpy()
        final = self.final_solutions[sample_idx].cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Initial condition - contour
        im1 = axes[0, 0].contourf(
            self.X.cpu(), self.Y.cpu(), initial, levels=20, cmap="hot"
        )
        axes[0, 0].set_title("Initial Condition (Contour)")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0, 0])

        # Initial condition - heatmap
        im2 = axes[0, 1].imshow(
            initial.T,
            cmap="hot",
            origin="lower",
            extent=[0, self.params["Lx"], 0, self.params["Ly"]],
        )
        axes[0, 1].set_title("Initial Condition (Heatmap)")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[0, 1])

        # Final solution - contour
        im3 = axes[1, 0].contourf(
            self.X.cpu(), self.Y.cpu(), final, levels=20, cmap="hot"
        )
        axes[1, 0].set_title(f'Final Solution (t={self.params["T"]}) - Contour')
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        plt.colorbar(im3, ax=axes[1, 0])

        # Final solution - heatmap
        im4 = axes[1, 1].imshow(
            final.T,
            cmap="hot",
            origin="lower",
            extent=[0, self.params["Lx"], 0, self.params["Ly"]],
        )
        axes[1, 1].set_title(f'Final Solution (t={self.params["T"]}) - Heatmap')
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("y")
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()
        viz_path = f"{self.params['output_dir']}/sample_visualization_{self.params['timestamp']}.png"
        plt.savefig(viz_path, dpi=150)
        plt.close()
        print(f"Sample visualization saved to {viz_path}")

    def save_dataset(self):
        """Save dataset in NPZ format with metadata."""

        # Prepare dataset dictionary
        dataset = {
            "initial_conditions": self.initial_conditions,
            "final_solutions": self.final_solutions,
            "evolution": self.evolution,
            "params": self.params,
            "X": self.X,
            "Y": self.Y,
            "dx": self.dx,
            "dy": self.dy,
            "timestamp": self.params["timestamp"],
        }

        filename = (
            f"{self.params['output_dir']}/heat_dataset_{self.params['timestamp']}.pt"
        )
        torch.save(dataset, filename)
        print(f"Dataset saved to {filename}")

        return dataset

    def _save_metadata(self, initial_np, final_np, evolution_np):
        """Save metadata text file."""
        metadata_file = (
            f"{self.params['output_dir']}/metadata_{self.params['timestamp']}.txt"
        )

        with open(metadata_file, "w") as f:
            f.write("Heat Equation Dataset\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {self.params['timestamp']}\n")
            f.write(f"Number of samples: {self.params['n_samples']}\n")
            f.write(f"Grid size: {self.params['nx']} x {self.params['ny']}\n")
            f.write(
                f"Domain: [0, {self.params['Lx']:.2f}] x [0, {self.params['Ly']:.2f}]\n"
            )
            f.write(
                f"Time evolution: T = {self.params['T']}, dt = {self.params['dt']}\n"
            )
            f.write(f"Thermal diffusivity: alpha = {self.params['alpha']}\n")
            f.write(f"Fourier modes: {self.params['n_fourier_modes']}\n")
            f.write(f"\nData shapes:\n")
            f.write(f"  Initial conditions: {initial_np.shape}\n")
            f.write(f"  Final solutions: {final_np.shape}\n")
            f.write(f"  Evolution snapshots: {evolution_np.shape}\n")

        print(f"Metadata saved to {metadata_file}")

    def print_summary(self, dataset):
        """Print dataset summary and validate results."""

        print(f"\nDataset Summary:")
        print("=" * 50)
        pprint.pprint(self.params)
        print("Number of samples:", dataset["initial_conditions"].shape[0])
        print(
            "Grid size:",
            dataset["initial_conditions"].shape[1],
            "x",
            dataset["initial_conditions"].shape[2],
        )
        print(
            f"Initial temperature range: [{dataset['initial_conditions'].min():.4f}, "
            f"{dataset['initial_conditions'].max():.4f}]"
        )
        print(
            f"Final temperature range: [{dataset['final_solutions'].min():.4f}, "
            f"{dataset['final_solutions'].max():.4f}]"
        )
        print(
            f"Average temperature decay: "
            f"{dataset['initial_conditions'].max() - dataset['final_solutions'].max():.4f}"
        )


def test_dataset():
    params = {
        "n_samples": 100,
        "nx": 128,
        "ny": 128,
        "Lx": 2.0 * torch.pi,
        "Ly": 2.0 * torch.pi,
        "T": 0.5,
        "dt": 0.001,
        "alpha": 1,
        "save_evolution": True,
        "output_dir": "heat_data",
        "batch_size": 1000,
        "n_fourier_modes": 4,
        "dim": 2,
    }

    generator = HeatEquationDatasetGenerator(**params)

    print("Generating heat equation dataset...")
    print("=" * 50)

    start_time = time.time()

    generator.generate_initial_conditions()

    assert generator.initial_conditions[0].max() == 0.05644312500953674

    generator.solve_heat_equation()

    assert generator.final_solutions[0].max() == 0.0025205323472619057

    print("\n\n--- Asserts passed ---")

    dataset = generator.save_dataset()

    generator.create_animation(sample_idx=0)

    generator.create_visualization(sample_idx=0)

    generator.print_summary(dataset)

    generator._save_metadata(
        dataset["initial_conditions"].cpu().numpy(),
        dataset["final_solutions"].cpu().numpy(),
        dataset["evolution"].cpu().numpy(),
    )

    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # test_dataset()
    generator = HeatEquationDatasetGenerator()

    generator.generate_initial_conditions()

    generator.solve_heat_equation()

    dataset = generator.save_dataset()

    timestamp = generator.create_animation(sample_idx=0)

    generator.create_visualization(sample_idx=0)

    generator.print_summary(dataset)

    generator._save_metadata(
        dataset["initial_conditions"],
        dataset["final_solutions"],
        dataset["evolution"],
    )

    print("Done.")
