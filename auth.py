import hashlib
import os
import secrets

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse


router = APIRouter()

LOGIN_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>모꼬지 LLM 로그인</title>
</head>
<body>
  <main>
    <h1>모꼬지 LLM</h1>
    {error}
    <form method="post" action="/login">
      <label>아이디 <input name="username" autocomplete="username" required></label>
      <label>비밀번호 <input type="password" name="password" autocomplete="current-password" required></label>
      <button type="submit">로그인</button>
    </form>
  </main>
</body>
</html>
"""


def get_session_secret() -> str:
    secret = os.getenv("MOKKOJI_SESSION_SECRET")
    if not secret:
        raise RuntimeError("MOKKOJI_SESSION_SECRET is not configured")
    return secret


def verify_credentials(username: str, password: str) -> bool:
    expected_username = os.getenv("MOKKOJI_AUTH_USERNAME", "")
    stored = os.getenv("MOKKOJI_AUTH_PASSWORD_HASH", "")

    try:
        salt_hex, expected_hash = stored.split(":", maxsplit=1)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False

    actual_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    ).hex()
    return secrets.compare_digest(username, expected_username) and secrets.compare_digest(
        actual_hash, expected_hash
    )


def require_auth(request: Request) -> None:
    if request.session.get("authenticated") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다.",
        )


@router.get("/login", response_class=HTMLResponse)
def login_page() -> str:
    return LOGIN_PAGE.format(error="")


@router.post("/login")
def login(request: Request, username: str = Form(), password: str = Form()):
    if not verify_credentials(username, password):
        return HTMLResponse(
            LOGIN_PAGE.format(error="<p>아이디 또는 비밀번호가 올바르지 않습니다.</p>"),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    request.session.clear()
    request.session["authenticated"] = True
    return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/logout")
def logout(request: Request, _: None = Depends(require_auth)):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
