import numpy as np
import seaborn as sns
import torch
from torch import nn

from nnscaling import aesthetics


def plot_decision_boundary(
    model: nn.Module,  # PyTorch model
    final_state_dict: dict,  # Final state dict of the model
    X: torch.Tensor,  # Input data
    y: torch.Tensor,  # Label vector
    ax,
    labels: list[int] = [0, 1],  # Labels
) -> None:
    # Get Seaborn's tab colors
    orange = aesthetics.SeabornColors.orange
    blue = aesthetics.SeabornColors.blue

    # Initialization
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    x_in = torch.tensor(x_in, dtype=torch.float32).to(next(model.parameters()).device)

    # Plot data points
    sns.scatterplot(
        x=X[y == labels[0], 0].cpu().numpy(),
        y=X[y == labels[0], 1].cpu().numpy(),
        ax=ax,
        color=orange,
        marker="o",
        s=50,
    )
    sns.scatterplot(
        x=X[y == labels[1], 0].cpu().numpy(),
        y=X[y == labels[1], 1].cpu().numpy(),
        ax=ax,
        color=blue,
        marker="o",
        s=50,
    )

    # Load final model state and set to eval mode
    model.load_state_dict(final_state_dict)
    model.eval()

    # Get predictions on grid points
    with torch.no_grad():
        out = model(x_in)
        y_pred = torch.argmax(out, dim=1).cpu().numpy().reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(
        xx,
        yy,
        y_pred,
        colors=[orange, blue],
        alpha=0.3,
        levels=np.linspace(labels[0], labels[1], 3),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
