class PipelineError(Exception):
    """Base class for all pipeline exceptions."""
    pass

class GenerationError(PipelineError):
    """Exception raised when structure generation fails."""
    pass

class ValidationError(PipelineError):
    """Exception raised when structure validation fails."""
    pass

class ExplorationError(PipelineError):
    """Exception raised when exploration fails."""
    pass
