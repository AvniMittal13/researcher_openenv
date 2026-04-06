"""Root-level models re-export for OpenEnv compatibility."""

try:
    from researcher1.env.models import ResearchAction, ResearchObservation, ResearchState
except ModuleNotFoundError:
    from env.models import ResearchAction, ResearchObservation, ResearchState

__all__ = ["ResearchAction", "ResearchObservation", "ResearchState"]
