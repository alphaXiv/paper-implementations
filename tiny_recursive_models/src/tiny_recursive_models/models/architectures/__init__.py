"""Model architecture implementations."""

# Make architecture modules easily importable
from tiny_recursive_models.models.architectures import trm
from tiny_recursive_models.models.architectures import hrm
from tiny_recursive_models.models.architectures import trm_singlez
from tiny_recursive_models.models.architectures import trm_hier6
from tiny_recursive_models.models.architectures import transformers_baseline

__all__ = [
    "trm",
    "hrm",
    "trm_singlez",
    "trm_hier6",
    "transformers_baseline",
]
