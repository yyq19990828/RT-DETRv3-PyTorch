# Weight Conversion Module
# Provides tools for converting model weights between PaddlePaddle and PyTorch formats

__version__ = "1.0.0"

from .converter import WeightConverter
from .models import (
    BatchConversionResult,
    BatchConversionSummary,
    CheckpointFile,
    ConversionConfig,
    ConversionSession,
    ConversionStatistics,
    DtypeConversion,
    Parameter,
    ParameterMapping,
    ShapeMismatch,
)

__all__ = [
    "WeightConverter",
    "BatchConversionResult",
    "BatchConversionSummary",
    "CheckpointFile",
    "Parameter",
    "ParameterMapping",
    "ConversionSession",
    "ConversionConfig",
    "ConversionStatistics",
    "ShapeMismatch",
    "DtypeConversion",
]


def configure_logging(level="INFO"):
    """Configure logging for weight conversion module

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging

    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )


def get_logger(name):
    """Get logger instance for weight conversion

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger instance
    """
    import logging

    return logging.getLogger(name)
