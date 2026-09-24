class AuthenticationError(Exception):
    """Base exception for authentication failures."""

    pass


class InvalidTokenError(AuthenticationError):
    """Raised when JWT validation fails."""

    pass


class MissingTokenError(AuthenticationError):
    """Raised when no token is provided."""

    pass


class AuthenticationProviderError(AuthenticationError):
    """Raised when the authentication provider is unavailable."""

    pass
