# noise_models/factory.py
import pennylane as qml
from typing import Union, Optional

# Import the config from your main project folder
from .noise_config import NoiseConfig

# Import your separated local classes
from .base import BaseNoiseModel
from .generic import GenericNoiseModel
from .thermal_relaxation import ThermalNoiseModel

class NoiseFactory:
    """Registry-driven factory that dynamically routes error profiles."""
    
    NOISE_LEVELS = {
        "none": 0.0, "low": 0.001, "medium": 0.01, "high": 0.05, "very_high": 0.1
    }
    
    _REGISTRY = {
        "depolarizing": lambda c, s: GenericNoiseModel(qml.DepolarizingChannel, s, c.qubits),
        "amplitude_damping": lambda c, s: GenericNoiseModel(qml.AmplitudeDamping, s, c.qubits),
        "phase_damping": lambda c, s: GenericNoiseModel(qml.PhaseDamping, s, c.qubits),
        "thermal_relaxation": lambda c, s: ThermalNoiseModel(c.t1, c.t2, c.tg, c.qubits)
    }
    
    @classmethod
    def create(cls, config: Union[dict, NoiseConfig]) -> Optional[BaseNoiseModel]:
        if not config:
            return None
            
        cfg = config if isinstance(config, NoiseConfig) else NoiseConfig(**config)
        
        if cfg.model == "none":
            return None
            
        strength = cls.NOISE_LEVELS.get(cfg.strength, cfg.strength) if isinstance(cfg.strength, str) else cfg.strength
        
        if cfg.model not in cls._REGISTRY:
            raise ValueError(f"Unknown noise model profile: {cfg.model}")
            
        return cls._REGISTRY[cfg.model](cfg, strength)
