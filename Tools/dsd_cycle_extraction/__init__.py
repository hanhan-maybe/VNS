"""Independent stable micturition-cycle extraction for the SPARC338 DSD cohort."""

from .config import SUBJECTS

__all__ = ["SUBJECTS", "extract_subject_cycles"]


def __getattr__(name):
    # Keep lightweight modules such as cycle_qc importable in both the legacy
    # top-level test layout and the package CLI layout.  The full subject
    # pipeline is imported only when its public entry point is requested.
    if name == "extract_subject_cycles":
        from .subject_pipeline import extract_subject_cycles
        return extract_subject_cycles
    raise AttributeError(name)
