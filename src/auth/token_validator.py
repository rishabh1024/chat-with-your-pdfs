import logging
from functools import cache

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError
from src.core.settings import load_environment_variables

from .exceptions import InvalidTokenError

settings = load_environment_variables()
logger = logging.getLogger(__name__)


@cache
def get_jwks_client() -> tuple[PyJWKClient, str]:
    # HttpUrl stringifies with a trailing slash; strip it so issuer/JWKS URLs match Supabase.
    supabase_auth_url = settings.supabase.auth_url.unicode_string()
    jwks_url = f"{supabase_auth_url}/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_url), supabase_auth_url


def validate_jw_token(token: str) -> dict:
    """
    Validate Supabase JWT and return decoded claims.

    Raises InvalidTokenError for any validation failure.
    """
    jwks_client, supabase_auth_url = get_jwks_client()
    token_issuer = f"{supabase_auth_url}/auth/v1"
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "RS256", "HS256"],
            audience="authenticated",
            issuer=token_issuer,
        )
        return claims
    except PyJWKClientError as jwk_client_error:
        logger.warning(
            "auth.token.rejected reason=jwks_lookup_failed error_type=%s",
            type(jwk_client_error).__name__,
        )
        raise InvalidTokenError(
            "Token Validation Failed. Unable to sign with the key"
        ) from jwk_client_error
    except jwt.ExpiredSignatureError as err:
        logger.debug("auth.token.rejected reason=expired")
        raise InvalidTokenError("Token has expired") from err
    except jwt.InvalidTokenError as e:
        logger.warning(
            "auth.token.rejected reason=invalid error_type=%s",
            type(e).__name__,
        )
        raise InvalidTokenError(f"Token validation failed: {e}") from e
