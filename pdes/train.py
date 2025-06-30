import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO
import neuralop
import os
import glob
from datetime import datetime
import time

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

def load_dataset(data_path=None, data_dir="heat_data"):
    if data_path is None:
        pattern = os.path.join(data_dir, "heat_dataset_*.pt")
        data_files = glob.glob(pattern)
        data_path = max(data_files, key=os.path.getmtime)
        print(f"Loading most recent dataset: {data_path}")
    else:
        print(f"Loading specified dataset: {data_path}")
    
    dataset = torch.load(data_path, map_location=device, weights_only=False)

    assert len(dataset['initial_conditions']) == len(dataset['final_solutions']) == dataset['params']['n_samples'], \
        "Mismatch in number of samples between initial conditions and final solutions."

    print(f"Loaded dataset with {dataset['params']['n_samples']} samples")
    print(f"Grid size: {dataset['params']['nx']} x {dataset['params']['ny']}")
    print(f"Data shape: {dataset['initial_conditions'].shape}")
    print(f"Temperature range - Initial: [{dataset['initial_conditions'].min():.4f}, {dataset['initial_conditions'].max():.4f}]")
    print(f"Temperature range - Final: [{dataset['final_solutions'].min():.4f}, {dataset['final_solutions'].max():.4f}]")

    return dataset
    


def load_heat_data(data_path=None, data_dir="heat_data"):
    """
    Load heat equation data from saved .pt file
    """
    if data_path is None:
        # Find the most recent dataset file
        pattern = os.path.join(data_dir, "heat_dataset_*.pt")
        data_files = glob.glob(pattern)
        data_path = max(data_files, key=os.path.getmtime)
        print(f"Loading most recent dataset: {data_path}")
    else:
        print(f"Loading specified dataset: {data_path}")

    # Load the .pt file
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    # Extract tensors and convert to CPU if needed
    initial_conditions = data["initial_conditions"].cpu()  # Shape: (n_samples, nx, ny)
    final_solutions = data["final_solutions"].cpu()  # Shape: (n_samples, nx, ny)

    # Convert to float tensors
    X_data = initial_conditions.float()
    y_data = final_solutions.float()

    # Create metadata dictionary
    metadata = {
        "n_samples": initial_conditions.shape[0],
        "nx": initial_conditions.shape[1],
        "ny": initial_conditions.shape[2],
        "parameters": data["parameters"] if "parameters" in data else {},
        "X_grid": data["X_grid"] if "X_grid" in data else None,
        "Y_grid": data["Y_grid"] if "Y_grid" in data else None,
        "dx": data["dx"].item() if "dx" in data else None,
        "dy": data["dy"].item() if "dy" in data else None,
        "timestamp": data["timestamp"] if "timestamp" in data else None,
    }

    print(f"Loaded dataset with {metadata['n_samples']} samples")
    print(f"Grid size: {metadata['nx']} x {metadata['ny']}")
    print(f"Data shape: {X_data.shape}")
    print(f"Temperature range - Initial: [{X_data.min():.4f}, {X_data.max():.4f}]")
    print(f"Temperature range - Final: [{y_data.min():.4f}, {y_data.max():.4f}]")

    return X_data, y_data, metadata


def create_data_loaders(X_data, y_data, train_split=0.8, batch_size=16, shuffle=True):
    """
    Create train and test data loaders from loaded data
    """
    n_samples = X_data.shape[0]
    n_train = int(n_samples * train_split)

    # Shuffle indices if requested
    if shuffle:
        indices = torch.randperm(n_samples)
        X_data = X_data[indices]
        y_data = y_data[indices]

    # Split data
    X_train = X_data[:n_train]
    X_test = X_data[n_train:]
    y_train = y_data[:n_train]
    y_test = y_data[n_train:]

    # Add channel dimension for FNO (batch, channels, height, width)
    X_train = X_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {y_train.shape}")

    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, (X_test, y_test)


def create_fno_model(grid_size=(64, 64), model_config=None):
    if model_config is None:
        model_config = {
            "n_modes": (16, 16),  # Fourier modes in each dimension
            "hidden_channels": 64,  # Hidden layer width
            "in_channels": 1,  # Input channels (scalar field)
            "out_channels": 1,  # Output channels (scalar field)
            "lifting_channels": 128,  # Lifting layer width
            "projection_channels": 128,  # Projection layer width
            "n_layers": 4,  # Number of FNO layers
        }

    model = FNO(**model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created FNO model with {n_params:,} parameters")
    print(f"Model configuration: {model_config}")

    return model


def train_model(model, train_loader, test_loader, training_config):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config["learning_rate"]
    )
    criterion = nn.MSELoss()

    # Training history
    history = {"train_loss": [], "test_loss": []}

    print(f"Training FNO model on {device}...")
    print(f"Starting training for {training_config['epochs']} epochs...")

    best_test_loss = float("inf")

    for epoch in range(training_config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()

        test_loss /= len(test_loader)

        # Store history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"{dataset['params']['output_dir']}/fno_model_{training_config['timestamp']}.pth")

        # Log progress
        if epoch % training_config["log_interval"] == 0:
            print(
                f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}"
            )

        # Reset to training mode for next epoch
        model.train()

    print(f"\nTraining completed! Best test loss: {best_test_loss:.6f}")
    print(f"Best model saved as 'heat_data/fno_model_{training_config['timestamp']}.pth'")

    return model, history


def visualize_results(model, test_data, dataset=None, n_samples=4):
    """
    Visualize model predictions vs true solutions
    """
    X_test, y_test = test_data

    model.eval()
    with torch.no_grad():
        # Get predictions for first few test samples
        X_sample = X_test[:n_samples].to(device)
        y_pred = model(X_sample).cpu()
        y_true = y_test[:n_samples].cpu()
        X_sample = X_sample.cpu()

    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        # Initial condition
        im0 = axes[0, i].imshow(X_sample[i, 0], cmap="viridis", origin="lower")
        axes[0, i].set_title(f"Initial Condition {i+1}")
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        # True solution
        im1 = axes[1, i].imshow(y_true[i, 0], cmap="viridis", origin="lower")
        axes[1, i].set_title(f"True Solution {i+1}")
        axes[1, i].axis("off")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Predicted solution
        im2 = axes[2, i].imshow(y_pred[i, 0], cmap="viridis", origin="lower")
        axes[2, i].set_title(f"FNO Prediction {i+1}")
        axes[2, i].axis("off")
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save figure
    plt.savefig(f"heat_data/fno_results_{dataset['params']['timestamp']}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Calculate error metrics
    mse = torch.mean((y_pred - y_true) ** 2)
    mae = torch.mean(torch.abs(y_pred - y_true))
    relative_error = torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_true) + 1e-8))

    print(f"\nTest Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Relative Error: {relative_error:.4f}")

    return mse.item(), mae.item(), relative_error.item()


def plot_training_history(history, dataset=None):
    """Plot training history"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Loss curves
    epochs = range(len(history["train_loss"]))
    ax.semilogy(epochs, history["train_loss"], label="Train Loss", color="blue")
    ax.semilogy(epochs, history["test_loss"], label="Test Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training and Test Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"{dataset['params']['output_dir']}/training_history_{dataset['params']['timestamp']}.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Main training script"""
    config = {
        "data_path": None,  # Will auto-find latest dataset
        "data_dir": "heat_data",
        "train_split": 0.8,
        "batch_size": 100,
        "model_config": {
            "n_modes": (16, 16),
            "hidden_channels": 64,
            "in_channels": 1,
            "out_channels": 1,
            "lifting_channels": 128,
            "projection_channels": 128,
            "n_layers": 4,
        },
        "training_config": {
            "epochs": 500,
            "learning_rate": 1e-3,
            "log_interval": 10,
        },
    }

    print("=" * 60)
    print("FNO Training on Heat Equation Data")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    X_data, y_data, metadata = load_heat_data(
        data_path=config["data_path"], data_dir=config["data_dir"]
    )

    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, test_loader, test_data = create_data_loaders(
        X_data,
        y_data,
        train_split=config["train_split"],
        batch_size=config["batch_size"],
    )

    # Create model
    print("\n3. Creating FNO model...")
    grid_size = (metadata["nx"], metadata["ny"])
    model = create_fno_model(grid_size, config["model_config"])

    # Train model
    print("\n4. Training model...")
    start_time = time.time()
    model, history = train_model(
        model, train_loader, test_loader, config["training_config"]
    )
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # Load best model
    print("\n5. Loading best model and evaluating...")
    torch.serialization.add_safe_globals([torch._C._nn.gelu])
    torch.serialization.add_safe_globals(
        [neuralop.layers.spectral_convolution.SpectralConv]
    )
    model.load_state_dict(torch.load("best_fno_model.pth"))

    # Visualize results
    print("\n6. Visualizing results...")
    mse, mae, rel_error = visualize_results(model, test_data, metadata)

    # Plot training history
    print("\n7. Plotting training history...")
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final MSE: {mse:.6f}")
    print(f"Final MAE: {mae:.6f}")
    print(f"Final Relative Error: {rel_error:.4f}")
    print("Best model saved as 'best_fno_model.pth'")
    print("=" * 60)


if __name__ == "__main__":
    # main()

    config = {
        "data_path": None,  # Will auto-find latest dataset
        "data_dir": "heat_data",
        "train_split": 0.8,
        "batch_size": 100,
        "model_config": {
            "n_modes": (16, 16),
            "hidden_channels": 64,
            "in_channels": 1,
            "out_channels": 1,
            "lifting_channels": 128,
            "projection_channels": 128,
            "n_layers": 4,
        },
        "training_config": {
            "epochs": 500,
            "learning_rate": 1e-3,
            "log_interval": 10,
        },
    }

    # Load dataset
    dataset = load_dataset()

    config['training_config']['timestamp'] = dataset['params']['timestamp']

    # Create data loaders
    train_loader, test_loader, test_data = create_data_loaders(
        X_data=dataset['initial_conditions'],
        y_data=dataset['final_solutions'],
        train_split=config['train_split'],
        batch_size=config['batch_size'],
    )

    # Create FNO model
    grid_size = (dataset['params']['nx'], dataset['params']['ny'])
    model = create_fno_model(grid_size, config['model_config'])

    # Train model
    start_time = time.time()
    model, history = train_model(
        model, train_loader, test_loader, config['training_config']
    )
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    #Visualize results
    mse, mae, rel_error = visualize_results(model, test_data, dataset=dataset)

    # Plot training history
    plot_training_history(history, dataset=dataset)
