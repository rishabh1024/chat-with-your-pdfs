from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.auth.models import AuthenticatedUser

from .exceptions import AuthenticationProviderError, InvalidTokenError
from .token_validator import validate_jw_token

bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_authenticated_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> AuthenticatedUser:

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_credentials",
                "message": "Authorization credentials are required.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = validate_jw_token(credentials.credentials)

        subject = claims.get("sub")

        if not isinstance(subject, str):
            raise InvalidTokenError("Token subject is invalid.")

        try:
            authenticated_user_id = UUID(subject)
        except ValueError as error:
            raise InvalidTokenError("Token subject is invalid.") from error

        return AuthenticatedUser(user_id=authenticated_user_id, claims=claims)
    except AuthenticationProviderError as authentication_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": (
                    "The authentication provider is unavailable. "
                    "Please try again shortly."
                )
            },
        ) from authentication_error
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "User could not be authenticated. Token is invalid or expired."
            },
        ) from e
