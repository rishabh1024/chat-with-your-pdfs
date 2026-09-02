import os
from functools import cache

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError

from .exceptions import InvalidTokenError


@cache
def get_jwks_client() -> PyJWKClient:
    supabase_url = os.environ.get("SUPABASE_AUTH_URL")

    if not supabase_url:
        raise ValueError("SUPABASE_URL must be set")

    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_url)

def validate_jw_token(token: str) -> dict:
    """
    Validate Supabase JWT and return decoded claims.

    Raises InvalidTokenError for any validation failure.
    """
    jwks_client = get_jwks_client()

    supabase_url = os.environ.get("SUPABASE_AUTH_URL")
    token_issuer = f"{supabase_url}/auth/v1"

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, signing_key.key,
            algorithms=["ES256", "RS256", "HS256"],
            audience="authenticated",
            issuer=token_issuer
        )
        return claims
    except PyJWKClientError as jwk_client_error:
      raise InvalidTokenError("Token Validation Failed. Unable to sign with the key") from jwk_client_error
    except jwt.ExpiredSignatureError as err:
        raise InvalidTokenError("Token has expired") from err
    except jwt.InvalidTokenError as e:
        raise InvalidTokenError(f"Token validation failed: {e}") from e