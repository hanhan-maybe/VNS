"""Causal offline NVC detection for the frozen SPARC338 DSD cohort."""

__version__ = "1.0.0"

from .subject_nvc_validation import validate_subject_nvc

__all__ = ["validate_subject_nvc"]
