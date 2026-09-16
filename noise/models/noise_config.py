from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class NoiseConfig:
    """Configuration for a noise model."""

    model: str
    strength: Union[float, str]
    qubits: Optional[List[int]] = None
    t1: float = 150.0
    t2: float = 130.0
    tg: float = 0.3  # gate time in microseconds
