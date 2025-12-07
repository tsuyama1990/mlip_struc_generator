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

class PhysicsViolationError(PipelineError):
    """Exception raised when physical constraints are violated (e.g. atom overlap, explosion)."""
    pass

class TimeoutError(PipelineError):
    """Exception raised when an operation exceeds its time limit."""
    pass

class ConvergenceError(PipelineError):
    """Exception raised when a calculation fails to converge (e.g. SCF)."""
    pass

class ConfigurationError(PipelineError):
    """Exception raised when configuration is invalid or missing resources."""
    pass
