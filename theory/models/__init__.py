"""theory.models — Part A rate-distortion solver for joint SCT × TurboQuant."""
from .rate_distortion import (RateDistortion, ByteModel,
                              build_weight_bytes_fn, build_sct_dL_fn)

__all__ = ["RateDistortion", "ByteModel", "build_weight_bytes_fn", "build_sct_dL_fn"]
