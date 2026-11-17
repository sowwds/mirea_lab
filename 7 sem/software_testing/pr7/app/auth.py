"""Учебные пользователи и случайные непрозрачные bearer-токены."""

from hmac import compare_digest
from secrets import token_urlsafe

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

bearer = HTTPBearer(auto_error=False)
USERS = {"admin": ("admin123", "admin"), "viewer": ("viewer123", "viewer")}


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


def login(body: LoginIn, request: Request):
    """Выдать токен после проверки учебных реквизитов."""
    user = USERS.get(body.username)
    if user is None or not compare_digest(body.password, user[0]):
        raise HTTPException(401, "Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    token = token_urlsafe(32)
    request.app.state.tokens[token] = user[1]
    return {"access_token": token, "token_type": "bearer"}


def get_role(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    """Вернуть роль по токену или отказать с кодом 401."""
    role = request.app.state.tokens.get(credentials.credentials) if credentials else None
    if role is None:
        raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})
    return role


def require_admin(role=Depends(get_role)):
    """Разрешить запись только администратору."""
    if role != "admin":
        raise HTTPException(403, "Admin role required")
    return role
