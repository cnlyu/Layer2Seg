"""Joint time/energy loss used by both stages."""

from torch import nn


class ScaledL1(nn.Module):
    def forward(self, outputs, targets):
        if outputs.shape != targets.shape or outputs.ndim != 2 or outputs.shape[1] % 2:
            raise ValueError("Scaled L1 requires matching [batch, even number of outputs] tensors")
        middle = outputs.shape[1] // 2
        time_scale = targets[:, :middle].abs().mean() + 1e-5
        energy_scale = targets[:, middle:].abs().mean() + 1e-5
        return ((outputs[:, :middle] - targets[:, :middle]).abs().mean() / time_scale
                + (outputs[:, middle:] - targets[:, middle:]).abs().mean() / energy_scale)
