"""Model architecture implementations."""

# Make architecture modules easily importable
from trm.models.architectures import trm
from trm.models.architectures import hrm
from trm.models.architectures import trm_singlez
from trm.models.architectures import trm_hier6
from trm.models.architectures import transformers_baseline

__all__ = [
    "trm",
    "hrm",
    "trm_singlez",
    "trm_hier6",
    "transformers_baseline",
]
