from loguru import logger

from optimizers.baseoptimizer import BaseVQEOptimizer
from .cobyla import Cobyla
from .adam import Adam
class OptimizerFactory:
    """Factory for creating unified VQE optimizers."""
    
    @classmethod
    def create(cls, config: dict) -> BaseVQEOptimizer:
        opt_type = config.get("type", "adam").lower()
        
        if opt_type == "cobyla":
            return Cobyla(config=config)
            
        elif opt_type == "adam":
            return Adam(config=config)
            
        else:
            logger.warning(f"⚠ Unknown optimizer '{opt_type}', defaulting to Adam")
            return Adam(config=config)
