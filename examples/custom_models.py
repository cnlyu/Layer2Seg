"""Example factory for the optional profiling workflow."""

from torch import nn


def small_cnn(channels=8):
    return nn.Sequential(
        nn.Conv2d(3, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10),
    )
