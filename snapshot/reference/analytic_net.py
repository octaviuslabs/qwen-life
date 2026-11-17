"""Analytically weighted PyTorch implementation of one Life generation.

Every parameter is assigned directly. There is no dataset, loss, optimizer, or
gradient step.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from life import Boundary, Grid


class AnalyticLifeNet(nn.Module):
    """A tiny exact neural circuit for binary Conway states.

    The first convolution computes

        score = live_neighbors + 0.5 * center

    Life outputs one exactly when that score is 2.5, 3.0, or 3.5. Four ReLU
    hinges form a trapezoid that is exactly one on those attainable scores and
    zero on every other attainable score.
    """

    def __init__(self, boundary: Boundary = Boundary.DEAD) -> None:
        super().__init__()
        self.boundary = Boundary(boundary)
        self.neighborhood = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        self.hinges = nn.Conv2d(1, 4, kernel_size=1, bias=True)
        self.combine = nn.Conv2d(4, 1, kernel_size=1, bias=False)

        with torch.no_grad():
            spatial = torch.ones((1, 1, 3, 3), dtype=torch.float32)
            spatial[0, 0, 1, 1] = 0.5
            self.neighborhood.weight.copy_(spatial)

            self.hinges.weight.fill_(1.0)
            self.hinges.bias.copy_(torch.tensor([-2.0, -2.5, -3.5, -4.0]))

            self.combine.weight.copy_(
                torch.tensor([2.0, -2.0, -2.0, 2.0]).reshape(1, 4, 1, 1)
            )

        self.requires_grad_(False)

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 4 or state.shape[1] != 1:
            raise ValueError("expected state shape [batch, 1, height, width]")
        if self.boundary is Boundary.TOROIDAL:
            padded = F.pad(state, (1, 1, 1, 1), mode="circular")
        else:
            padded = F.pad(state, (1, 1, 1, 1), mode="constant", value=0.0)

        # Use the programmed convolution weights as explicit patch dot-products.  Some
        # optimized Conv2d CPU kernels use Winograd-style transforms; on a large batch those
        # can leave ~1e-6 cancellation residue even though every value in this circuit is an
        # exactly representable integer or half-integer.  The explicit sums preserve the same
        # neural weights and parameter count while keeping the advertised bit-exact contract
        # independent of the selected convolution backend.
        batch, _, height, width = state.shape
        patches = F.unfold(padded, kernel_size=3)
        spatial = self.neighborhood.weight.reshape(1, 9, 1)
        score = (patches * spatial).sum(dim=1, keepdim=True)

        hinge_weight = self.hinges.weight.reshape(1, 4, 1)
        hinge_bias = self.hinges.bias.reshape(1, 4, 1)
        hinges = F.relu(score * hinge_weight + hinge_bias)
        combine_weight = self.combine.weight.reshape(1, 4, 1)
        output = (hinges * combine_weight).sum(dim=1)
        return output.reshape(batch, 1, height, width)


def tensor_from_grid(grid: Grid, *, device: torch.device | str | None = None) -> Tensor:
    return torch.tensor(
        grid.cells, dtype=torch.float32, device=device
    ).reshape(1, 1, grid.height, grid.width)


def grid_from_tensor(state: Tensor) -> Grid:
    if state.ndim != 4 or state.shape[0] != 1 or state.shape[1] != 1:
        raise ValueError("expected one state with shape [1, 1, height, width]")
    height, width = state.shape[-2:]
    flat = state.detach().to(device="cpu").reshape(-1)
    rounded = torch.round(flat)
    if not torch.equal(flat, rounded):
        raise ValueError("network state is not exactly binary")
    return Grid(width, height, tuple(int(value) for value in rounded.tolist()))


def analytic_step(model: AnalyticLifeNet, grid: Grid) -> Grid:
    parameter = next(model.parameters())
    state = tensor_from_grid(grid, device=parameter.device)
    return grid_from_tensor(model(state))
