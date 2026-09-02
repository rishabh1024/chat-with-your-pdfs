from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.auth.models import AuthenticatedUser

from .exceptions import InvalidTokenError
from .token_validator import validate_jw_token

bearer_scheme = HTTPBearer()

async def get_current_authenticated_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
) -> AuthenticatedUser:
    try:
        claims = validate_jw_token(credentials.credentials)
        return AuthenticatedUser(user_id=UUID(claims["sub"]), claims=claims)
    except InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
