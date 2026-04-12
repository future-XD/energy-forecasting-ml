import os
import functools

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///energy.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)


class ModelConfig(db.Model):
    """Stores the AI model API connection settings."""
    id = db.Column(db.Integer, primary_key=True)
    api_url = db.Column(db.String(500), nullable=True)
    api_key = db.Column(db.String(200), nullable=True)

    @staticmethod
    def get():
        return ModelConfig.query.first()


def _init_db():
    """Create tables and seed default rows."""
    db.create_all()
    if not ModelConfig.query.first():
        db.session.add(ModelConfig(api_url=None, api_key=None))
        db.session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def login_required(view):
    """Redirect anonymous users to the login page."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        db.session.add(User(
            username=username,
            password_hash=generate_password_hash(password),
        ))
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session["user"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Application routes
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    config = ModelConfig.get()
    return render_template("dashboard.html", config=config)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    coming_soon = False

    if request.method == "POST":
        date_str = request.form.get("date", "").strip()
        time_str = request.form.get("time", "").strip()

        if not date_str or not time_str:
            flash("Date and time are required.", "danger")
            return redirect(url_for("predict"))

        # The prediction model is under active development.
        # Once the AI model API is ready, predictions will be served here.
        coming_soon = True

    return render_template("predict.html", coming_soon=coming_soon)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    config = ModelConfig.get()

    if request.method == "POST":
        api_url = request.form.get("api_url", "").strip()
        api_key = request.form.get("api_key", "").strip()

        config.api_url = api_url or None
        config.api_key = api_key or None
        db.session.commit()
        flash("Settings saved successfully.", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", config=config)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        _init_db()
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
