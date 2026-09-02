from dataclasses import dataclass
from uuid import UUID

"""
Added claims as dictionary field for future extensibility.
We can add fields like email, role etc. based on the
requirement in future.
"""
@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: UUID
    claims: dict