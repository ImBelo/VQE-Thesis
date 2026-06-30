import pennylane as qml
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

@dataclass
class NoiseConfig:
    """Configuration for noise model"""
    model: str                   # "none", "depolarizing", "amplitude_damping", "phase_damping", "thermal_relaxation", "mixed"

    strength: Union[float, str]  # 0.0 to 1.0, or "low", "medium", etc.
    qubits: Optional[List[int]] = None
    t1: float = 50.0
    t2: float = 70.0
    tg: float = 1.0


