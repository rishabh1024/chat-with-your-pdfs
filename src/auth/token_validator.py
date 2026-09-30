import logging
from functools import cache

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

from core.settings import load_environment_variables

from .exceptions import AuthenticationProviderError, InvalidTokenError

settings = load_environment_variables()
logger = logging.getLogger(__name__)


supabase_auth_url = settings.supabase.auth_url.unicode_string().rstrip("/")


@cache
def get_jwks_client() -> PyJWKClient:
    jwks_url = f"{supabase_auth_url}/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_url)


def validate_jw_token(token: str) -> dict:
    """
    Validate Supabase JWT and return decoded claims.

    Raises InvalidTokenError for any validation failure.
    """

    jwks_client = get_jwks_client()
    token_issuer = f"{supabase_auth_url}/auth/v1"

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256", "HS256"],
            audience="authenticated",
            issuer=token_issuer,
            options={"require": ["sub", "exp", "iat"]},
        )
    except PyJWKClientConnectionError as client_connection_error:
        logger.warning(
            "auth.token.validation.failed reason=jwks_client_not_reachable error_type=%s",
            type(client_connection_error).__name__,
        )
        raise AuthenticationProviderError(
            "Authentication Provider is Unavailable. Token Validation Failed."
        ) from client_connection_error
    except PyJWKClientError as jwk_client_error:
        logger.warning(
            "auth.token.validation.failed reason=jwks_validation_failed error_type=%s",
            type(jwk_client_error).__name__,
        )
        raise AuthenticationProviderError(
            "Authentication provider is unavailable. Token validation failed."
        ) from jwk_client_error
    except jwt.ExpiredSignatureError as err:
        logger.debug("auth.token.rejected reason=expired")
        raise InvalidTokenError("Token has expired. User should reauthenticate.") from err
    except jwt.InvalidTokenError as e:
        logger.warning(
            "auth.token.rejected reason=invalid or malformed error_type=%s",
            type(e).__name__,
        )
        raise InvalidTokenError("Token validation failed.The token is invalid or malformed.") from e
