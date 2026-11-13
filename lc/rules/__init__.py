"""Rule-based solver. Importing this package registers every family."""
from . import fam_access, fam_calendar, fam_movement, fam_quantity  # noqa: F401
from .core import REGISTRY, solve  # noqa: F401
