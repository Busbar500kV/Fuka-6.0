"""
Fuka-6.0 core package.

Exports the main simulation primitives so experiments can do:
    from core import SubstrateConfig, PlasticityConfig, RunConfig, run_simulation
"""

from .substrate import Substrate, SubstrateConfig
from .plasticity import PlasticityConfig, update_g, compute_fitness
from .metrics import SlotConfig
from .run import RunConfig, run_simulation

__all__ = [
    "Substrate",
    "SubstrateConfig",
    "PlasticityConfig",
    "update_g",
    "compute_fitness",
    "SlotConfig",
    "RunConfig",
    "run_simulation",
]