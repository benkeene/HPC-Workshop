import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import os

# --- EDIT FROM HERE --------------------------------------------


# Target function to approximate
def target_function(x):
    return torch.sin(x) * torch.exp(-x / 5)


# Training parameters
learning_rate = 0.001  # How fast the model learns
epochs = 1000  # Number of training iterations
batch_size = 32  # Number of samples per gradient update
hidden_size = 64  # Size of hidden layer in the neural network
training_domain = (-5, 5)  # Range of x values for training data
n_samples = 50  # Number of training samples
sampling_strategy = "random"  # 'uniform' or 'random'


# Network architecture -- simple fully connected neural network
class SimpleNet(nn.Module):
    def __init__(self, hidden_size=64):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# --- EDIT TO HERE --------------------------------------------

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
num_snapshots = 31

# Generate training data using PyTorch
if sampling_strategy == "uniform":
    x_train = torch.linspace(training_domain[0], training_domain[1], n_samples)
elif sampling_strategy == "random":
    x_train = torch.empty(n_samples).uniform_(training_domain[0], training_domain[1])
else:
    raise ValueError("Invalid sampling strategy. Use 'uniform' or 'random'.")
y_train = target_function(x_train)

# Reshape tensors for network input
x_tensor = x_train.unsqueeze(1)
y_tensor = y_train.unsqueeze(1)

# Create model, loss, and optimizer
model = SimpleNet(hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

x_test = torch.linspace(-5, 5, 200)
x_test_tensor = x_test.unsqueeze(1)
true_values = target_function(x_test)

# Store snapshots
snapshots = []
snapshot_epochs = torch.linspace(0, epochs - 1, num_snapshots, dtype=torch.int).tolist()

# Training loop with snapshots
for epoch in range(epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # Save snapshot
    if epoch in snapshot_epochs:
        model.eval()
        with torch.no_grad():
            predictions = model(x_test_tensor).squeeze()
            snapshots.append(
                {
                    "epoch": epoch,
                    "predictions": predictions.clone(),
                    "loss": loss.item(),
                }
            )
        model.train()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

y_min = torch.min(
    torch.stack([snapshot["predictions"] for snapshot in snapshots])
).item()
y_max = torch.max(
    torch.stack([snapshot["predictions"] for snapshot in snapshots])
).item()
y_min = min(y_min, true_values.min().item())
y_max = max(y_max, true_values.max().item())
print(f"Y-axis limits: {y_min:.3f} to {y_max:.3f}")

# Create parameter text for display
param_text = f"""Parameters:
Learning Rate: {learning_rate}
Epochs: {epochs}
Batch Size: {batch_size}
Hidden Size: {hidden_size}
Training Domain: {training_domain}
Samples: {n_samples}
Sampling: {sampling_strategy}"""

# Create animation
fig, ax = plt.subplots(figsize=(12, 8))


def animate(frame):
    ax.clear()

    snapshot = snapshots[frame]
    epoch = snapshot["epoch"]
    predictions = snapshot["predictions"]
    loss = snapshot["loss"]

    ax.plot(x_test, true_values, "b-", label="True Function", linewidth=3, alpha=0.5)
    ax.plot(x_test, predictions, "r--", label="Neural Network", linewidth=2)

    # Training data points
    ax.scatter(x_train, y_train, alpha=0.7, color="green", s=20, label="Training Data")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"Neural Network Learning Progress\nEpoch: {epoch+1}, Loss: {loss:.6f}",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(y_min - 0.1, y_max * 1.3)

    # Add parameter text box
    ax.text(
        0.02,
        0.98,
        param_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


# Animation
anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=800, repeat=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gif_path = f"results/anim_{timestamp}.gif"
anim.save(gif_path, writer="pillow", fps=3)

# Final test evaluation
model.eval()
with torch.no_grad():
    final_predictions = model(x_test_tensor).squeeze()
    final_mse = torch.mean((final_predictions - true_values) ** 2)
    print(f"\nFinal MSE on test data: {final_mse:.6f}")

# Plot final
plt.figure(figsize=(12, 8))
plt.plot(x_test, true_values, "b-", label="True Function", linewidth=3)
plt.plot(x_test, final_predictions, "r--", label="Final Neural Network", linewidth=2)
plt.scatter(
    x_train[:50], y_train[:50], alpha=0.3, color="gray", s=20, label="Training Data"
)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("Final Function Approximation", fontsize=14)
plt.legend(fontsize=11)
plt.xlim(-5, 5)
plt.ylim(y_min - 0.1, y_max + 0.1)
plt.grid(True, alpha=0.3)

plt.gca().text(
    0.02,
    0.98,
    param_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

img_path = f"results/final_{timestamp}.png"
plt.savefig(img_path)
