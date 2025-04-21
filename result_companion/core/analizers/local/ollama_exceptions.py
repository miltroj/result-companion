class OllamaServerNotRunning(Exception):
    """Exception raised when Ollama server is not running."""


class OllamaNotInstalled(Exception):
    """Exception raised when Ollama is not installed."""


class OllamaModelNotAvailable(Exception):
    """Exception raised when the required Ollama model is not available."""
