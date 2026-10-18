"""Optional NVIDIA power readers. Public readings are converted from mW to W."""

from contextlib import contextmanager
import math
from pathlib import Path

import torch


def watts(milliwatts):
    value = float(milliwatts) / 1000.
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid power sensor reading: {milliwatts} mW")
    return value


class TorchPower:
    def __init__(self, index):
        if not callable(getattr(torch.cuda, "power_draw", None)):
            raise ValueError("Desktop power collection requires PyTorch >=2.1 with torch.cuda.power_draw(); "
                             "install compatible CUDA-enabled torch and torchvision builds")
        self.device = torch.device(f"cuda:{index}")
        self.metadata = {"backend": "torch", "api": "torch.cuda.power_draw",
                         "scope": "GPU and associated circuitry", "device": str(self.device), "units": "W",
                         "sampling": "Sensor-reported power queried at each executed layer boundary"}
        self.read()

    def read(self):
        return watts(torch.cuda.power_draw(self.device))

    def close(self):
        pass


class JetsonPower:
    def __init__(self, rail, interval):
        try:
            from jtop import jtop
        except ImportError as error:
            raise ValueError("Jetson power collection needs jetson-stats and its running jtop service") from error
        self.monitor = jtop(interval=interval)
        try:
            self.monitor.start()
            if not self.monitor.ok():
                raise ValueError("The jtop service did not provide power data")
            available = self.monitor.power.get("rail", {})
            if rail == "auto":
                rail = next((name for name in ["VDD_GPU_SOC", "VDD_GPU", "POM_5V_GPU", "VDD_CPU_GPU_CV"]
                             if name in available), None)
            if rail != "total" and rail not in available:
                raise ValueError(f"Select --power_rail from {', '.join(available)}, or total")
            self.rail = rail
            self.metadata = {"backend": "jtop", "scope": f"Jetson rail: {rail}", "rail": rail,
                             "interval_seconds": self.monitor.interval, "units": "W",
                             "sampling": "Latest sensor-reported power queried at each executed layer boundary"}
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        if not self.monitor.ok(spin=True):
            raise ValueError("The jtop power sensor stopped")
        data = self.monitor.power
        reading = data["tot"] if self.rail == "total" else data["rail"][self.rail]
        return watts(reading["power"])

    def close(self):
        self.monitor.close()


@contextmanager
def open_power(backend, index, rail="auto", interval=.01):
    if backend == "auto":
        backend = "jtop" if Path("/etc/nv_tegra_release").exists() else "torch"
    reader = JetsonPower(rail, interval) if backend == "jtop" else TorchPower(index)
    try:
        yield reader
    finally:
        reader.close()
