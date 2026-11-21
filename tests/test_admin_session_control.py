import time
import pytest
from app import create_app
from app.config import Config


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["ADMIN_PASSWORD"] = "testpassword123"
    Config.ADMIN_PASSWORD = "testpassword123"
    with app.test_client() as client:
        yield client


def test_unauthenticated_admin_access_redirects(client):
    res = client.get("/admin", follow_redirects=False)
    assert res.status_code == 302
    assert "/admin/login" in res.headers["Location"]


def test_unauthenticated_admin_api_returns_401(client):
    res = client.get("/api/admin/status")
    assert res.status_code == 401
    data = res.get_json()
    assert data.get("session_expired") is True


def test_admin_login_success_and_session_setup(client):
    res = client.post("/admin/login", data={"password": "testpassword123"}, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["Location"] == "/admin"

    # Check authenticated access to /admin
    res_admin = client.get("/admin")
    assert res_admin.status_code == 200

    # Check session endpoint
    res_sess = client.get("/api/admin/session")
    assert res_sess.status_code == 200
    sess_data = res_sess.get_json()
    assert sess_data["authenticated"] is True
    assert sess_data["remaining_seconds"] > 0
    assert sess_data["timeout_minutes"] == 20


def test_admin_inactivity_timeout_page(client):
    # Log in
    client.post("/admin/login", data={"password": "testpassword123"})

    # Simulate 21 minutes of inactivity (1260 seconds ago)
    with client.session_transaction() as sess:
        sess["last_activity"] = time.time() - 1260

    # Accessing /admin after 21 minutes must redirect to login with expired=1
    res = client.get("/admin", follow_redirects=False)
    assert res.status_code == 302
    assert "expired=1" in res.headers["Location"]

    # Verify session is cleared
    with client.session_transaction() as sess:
        assert sess.get("admin_authenticated") is None


def test_admin_inactivity_timeout_api(client):
    # Log in
    client.post("/admin/login", data={"password": "testpassword123"})

    # Simulate 21 minutes of inactivity
    with client.session_transaction() as sess:
        sess["last_activity"] = time.time() - 1260

    # API request after timeout must return 401 with session_expired: True
    res = client.get("/api/admin/session")
    assert res.status_code == 401
    data = res.get_json()
    assert data.get("session_expired") is True


def test_admin_explicit_logout(client):
    # Log in
    client.post("/admin/login", data={"password": "testpassword123"})
    
    # Explicit logout
    res = client.get("/admin/logout", follow_redirects=False)
    assert res.status_code == 302
    assert "logged_out=1" in res.headers["Location"]

    # Subsequent access is unauthorized
    res_admin = client.get("/admin", follow_redirects=False)
    assert res_admin.status_code == 302
    assert "/admin/login" in res_admin.headers["Location"]
