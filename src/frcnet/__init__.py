"""FRCNet public interfaces."""

from frcnet.data import ALLOWED_COHORT_NAMES, BatchInput, validate_batch_input
from frcnet.models import FRCNetModel, ModelOutput
from frcnet.training import LossBreakdown, compute_total_loss, run_train_step
from frcnet.utils import RuntimeSpec, completion_score, content_entropy, move_batch_to_device, resolve_runtime

__all__ = [
    "__version__",
    "ALLOWED_COHORT_NAMES",
    "BatchInput",
    "FRCNetModel",
    "LossBreakdown",
    "ModelOutput",
    "RuntimeSpec",
    "completion_score",
    "compute_total_loss",
    "content_entropy",
    "move_batch_to_device",
    "resolve_runtime",
    "run_train_step",
    "validate_batch_input",
]

__version__ = "0.1.0"
