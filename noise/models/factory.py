from __future__ import annotations

from typing import Callable, ClassVar, Dict, Optional, Union

import pennylane as qml

from .base import BaseNoiseModel
from .generic import GenericNoiseModel
from .noise_config import NoiseConfig
from .thermal_relaxation import ThermalNoiseModel


class NoiseFactory:
    """Registry-driven factory that routes configs to noise models."""

    NOISE_LEVELS: ClassVar[Dict[str, float]] = {
        "none": 0.0,
        "low": 0.001,
        "medium": 0.01,
        "high": 0.05,
        "very_high": 0.1,
    }

    _REGISTRY: ClassVar[Dict[str, Callable[[NoiseConfig, float], BaseNoiseModel]]] = {
        "depolarizing": lambda c, s: GenericNoiseModel(
            qml.DepolarizingChannel, s, c.qubits
        ),
        "amplitude_damping": lambda c, s: GenericNoiseModel(
            qml.AmplitudeDamping, s, c.qubits
        ),
        "phase_damping": lambda c, s: GenericNoiseModel(
            qml.PhaseDamping, s, c.qubits
        ),
        "thermal_relaxation": lambda c, s: ThermalNoiseModel(
            c.t1, c.t2, c.tg, c.qubits
        ),
    }

    @classmethod
    def create(
        cls,
        config: Union[dict, NoiseConfig, None],
    ) -> Optional[BaseNoiseModel]:
        if not config:
            return None

        cfg: NoiseConfig = (
            config if isinstance(config, NoiseConfig) else NoiseConfig(**config)
        )

        if cfg.model == "none":
            return None

        strength: float = (
            cls.NOISE_LEVELS.get(cfg.strength, cfg.strength)   # type: ignore[arg-type]
            if isinstance(cfg.strength, str)
            else cfg.strength
        )

        if cfg.model not in cls._REGISTRY:
            raise ValueError(f"Unknown noise model profile: {cfg.model}")

        return cls._REGISTRY[cfg.model](cfg, strength)
