from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from auth import hash_password, require_auth, router


def create_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MOKKOJI_AUTH_USERNAME", "mokkoji")
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    monkeypatch.setenv(
        "MOKKOJI_AUTH_PASSWORD_HASH", hash_password("correct-password", salt)
    )

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
