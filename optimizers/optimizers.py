from __future__ import annotations

from typing import Any

from loguru import logger

from .adam import Adam
from .baseoptimizer import BaseVQEOptimizer
from .cobyla import Cobyla


class OptimizerFactory:
    """Factory for creating unified VQE optimizers."""

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseVQEOptimizer:
        opt_type: str = config.get("type", "adam").lower()

        if opt_type == "cobyla":
            return Cobyla(config=config)

        elif opt_type == "adam":
            return Adam(config=config)

        else:
            logger.warning(
                "⚠ Unknown optimizer %r, defaulting to Adam", opt_type
            )
            return Adam(config=config)
