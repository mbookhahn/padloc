"""PadLoc — training-free floor-plan localization by sparse geometric matching."""
from .config import PadLocConfig
from .pipeline import ChunkResult, PadLoc
from .reference import Reference
from .windows import Frame

__all__ = ["PadLocConfig", "PadLoc", "ChunkResult", "Reference", "Frame"]
__version__ = "1.0.0"
