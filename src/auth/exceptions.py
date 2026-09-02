class AuthenticationError(Exception):
    """Base exception for authentication failures."""
    pass


class InvalidTokenError(AuthenticationError):
    """Raised when JWT validation fails."""
    pass


class MissingTokenError(AuthenticationError):
    """Raised when no token is provided."""
    pass
