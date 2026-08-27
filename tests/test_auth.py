import hashlib

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from auth import require_auth, router


def password_hash(password: str) -> str:
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    ).hex()
    return f"{salt.hex()}:{digest}"


def create_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MOKKOJI_AUTH_USERNAME", "mokkoji")
    monkeypatch.setenv("MOKKOJI_AUTH_PASSWORD_HASH", password_hash("correct-password"))

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret", https_only=True)
    app.include_router(router)

    @app.get("/protected", dependencies=[Depends(require_auth)])
    def protected() -> dict[str, bool]:
        return {"authenticated": True}

    return TestClient(app, base_url="https://testserver")


def test_login_protects_route_and_logout_clears_session(monkeypatch) -> None:
    client = create_client(monkeypatch)

    assert client.get("/protected").status_code == 401
    assert (
        client.post(
            "/login",
            data={"username": "mokkoji", "password": "wrong-password"},
        ).status_code
        == 401
    )

    response = client.post(
        "/login",
        data={"username": "mokkoji", "password": "correct-password"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert client.get("/protected").json() == {"authenticated": True}

    assert client.post("/logout", follow_redirects=False).status_code == 303
    assert client.get("/protected").status_code == 401
