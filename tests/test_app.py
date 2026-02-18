import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    with app.test_client() as client:
        yield client


def _register(client, username="testuser", password="testpass"):
    return client.post(
        "/register",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def _login(client, username="testuser", password="testpass"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


# ---------- Auth tests ----------

def test_login_page_renders(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Login" in resp.data


def test_register_page_renders(client):
    resp = client.get("/register")
    assert resp.status_code == 200
    assert b"Register" in resp.data


def test_register_and_login(client):
    resp = _register(client)
    assert b"Registration successful" in resp.data

    resp = _login(client)
    assert b"Welcome" in resp.data


def test_login_invalid_credentials(client):
    _register(client)
    resp = _login(client, password="wrong")
    assert b"Invalid username or password" in resp.data


def test_logout(client):
    _register(client)
    _login(client)
    resp = client.get("/logout", follow_redirects=True)
    assert b"logged out" in resp.data


# ---------- Access control ----------

def test_dashboard_requires_login(client):
    resp = client.get("/dashboard", follow_redirects=True)
    assert b"Please log in" in resp.data


def test_predict_requires_login(client):
    resp = client.get("/predict", follow_redirects=True)
    assert b"Please log in" in resp.data


# ---------- Prediction ----------

def test_predict_page_renders(client):
    _register(client)
    _login(client)
    resp = client.get("/predict")
    assert resp.status_code == 200
    assert b"Prediction" in resp.data


def test_predict_returns_result(client):
    _register(client)
    _login(client)
    resp = client.post(
        "/predict",
        data={"date": "2025-01-15", "time": "14:00", "num_houses": "1000"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"kWh" in resp.data
    assert b"Prediction Result" in resp.data


def test_predict_missing_fields(client):
    _register(client)
    _login(client)
    resp = client.post(
        "/predict",
        data={"date": "", "time": ""},
        follow_redirects=True,
    )
    assert b"required" in resp.data


def test_predict_default_houses(client):
    _register(client)
    _login(client)
    resp = client.post(
        "/predict",
        data={"date": "2025-06-01", "time": "08:00", "num_houses": ""},
        follow_redirects=True,
    )
    assert b"5567" in resp.data


def test_index_redirects_to_login(client):
    resp = client.get("/")
    assert resp.status_code == 302
