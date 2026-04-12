import io
import os
import functools
from datetime import timedelta
from urllib.parse import urlparse

import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session
)
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:////tmp/energy.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Feature 11: session lifetime for "Remember me"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

# Limit upload size to 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

db = SQLAlchemy(app)

# Feature 9: rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


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
        """Return the singleton ModelConfig row, or None if not yet seeded."""
        return ModelConfig.query.first()


def _init_db():
    """Create tables and seed default rows."""
    db.create_all()
    if not ModelConfig.query.first():
        try:
            db.session.add(ModelConfig(api_url=None, api_key=None))
            db.session.commit()
        except db.exc.IntegrityError:
            db.session.rollback()


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


def admin_required(view):
    """Restrict a view to the 'admin' user."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("user") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)
    return wrapped


# Feature 10: password strength validation
def validate_password(password):
    """Return an error message if the password is too weak, else None."""
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not any(c.isalpha() for c in password):
        return "Password must contain at least one letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one digit."
    return None


# Feature 4 / 5 / 6: CSV processing
# Sample hourly averages (kWh/hh) representative of the London Smart Meter dataset,
# shown on the dashboard before any file is uploaded.
SAMPLE_HOURLY_KWH = [
    0.18, 0.15, 0.13, 0.12, 0.12, 0.14,
    0.18, 0.28, 0.32, 0.27, 0.24, 0.23,
    0.24, 0.22, 0.21, 0.22, 0.24, 0.30,
    0.38, 0.42, 0.40, 0.35, 0.29, 0.22,
]


def process_csv(file_stream):
    """
    Parse an uploaded CSV and return (stats, chart_labels, chart_data, peak_info).

    Supports the London Smart Meter format (columns: LCLid, tstp, energy(kWh/hh))
    as well as any CSV with a datetime-like column and a numeric energy column.
    """
    try:
        df = pd.read_csv(file_stream)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV must have at least two columns.")

    # Detect datetime column
    dt_col = None
    for col in df.columns:
        lower = col.lower()
        if any(kw in lower for kw in ("tstp", "time", "date", "timestamp", "datetime")):
            dt_col = col
            break
    if dt_col is None:
        dt_col = df.columns[0]

    # Detect energy column
    energy_col = None
    for col in df.columns:
        if col == dt_col:
            continue
        lower = col.lower()
        if any(kw in lower for kw in ("energy", "kwh", "consumption", "usage", "power", "wh")):
            energy_col = col
            break
    if energy_col is None:
        # Fall back to first numeric column that isn't the datetime column
        for col in df.columns:
            if col != dt_col and pd.api.types.is_numeric_dtype(df[col]):
                energy_col = col
                break
    if energy_col is None:
        # Last resort: second column
        energy_col = df.columns[1] if df.columns[1] != dt_col else df.columns[0]

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    df = df.dropna(subset=[energy_col])

    if df.empty:
        raise ValueError("No valid rows found after parsing datetime and energy columns.")

    stats = {
        "rows": len(df),
        "dt_col": dt_col,
        "energy_col": energy_col,
        "min_kwh": round(float(df[energy_col].min()), 4),
        "max_kwh": round(float(df[energy_col].max()), 4),
        "avg_kwh": round(float(df[energy_col].mean()), 4),
        "total_kwh": round(float(df[energy_col].sum()), 2),
    }

    # Hourly aggregation (average kWh per hour-of-day)
    df["_hour"] = df[dt_col].dt.hour
    hourly = df.groupby("_hour")[energy_col].mean()
    chart_labels = [f"{h:02d}:00" for h in range(24)]
    chart_data = [round(float(hourly.get(h, 0.0)), 4) for h in range(24)]

    # Feature 6: top-3 peak hours
    top_hours = hourly.nlargest(3)
    peak_info = [
        {
            "hour": f"{int(h):02d}:00–{int(h) + 1:02d}:00",
            "avg_kwh": round(float(v), 4),
        }
        for h, v in top_hours.items()
    ]

    return stats, chart_labels, chart_data, peak_info


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

        # Feature 10: enforce password strength
        pw_error = validate_password(password)
        if pw_error:
            flash(pw_error, "danger")
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
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            # Feature 11: remember me
            if request.form.get("remember"):
                session.permanent = True
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
    chart_labels = [f"{h:02d}:00" for h in range(24)]
    return render_template(
        "dashboard.html",
        config=config,
        chart_labels=chart_labels,
        chart_data=SAMPLE_HOURLY_KWH,
    )


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

        if api_url:
            parsed = urlparse(api_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                flash("Please enter a valid URL (must start with http:// or https://).", "danger")
                return redirect(url_for("settings"))

        config.api_url = api_url or None
        config.api_key = api_key or None
        db.session.commit()
        flash("Settings saved successfully.", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", config=config)


# Feature 7: user profile / password change
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_pw = request.form.get("current_password", "").strip()
        new_pw = request.form.get("new_password", "").strip()
        confirm_pw = request.form.get("confirm_password", "").strip()

        user = User.query.filter_by(username=session["user"]).first()

        if not user:
            session.pop("user", None)
            flash("User account not found. Please log in again.", "danger")
            return redirect(url_for("login"))

        if not check_password_hash(user.password_hash, current_pw):
            flash("Current password is incorrect.", "danger")
            return redirect(url_for("profile"))

        if new_pw != confirm_pw:
            flash("New passwords do not match.", "danger")
            return redirect(url_for("profile"))

        pw_error = validate_password(new_pw)
        if pw_error:
            flash(pw_error, "danger")
            return redirect(url_for("profile"))

        user.password_hash = generate_password_hash(new_pw)
        db.session.commit()
        flash("Password updated successfully.", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html")


# Feature 8: admin panel
@app.route("/admin")
@login_required
@admin_required
def admin():
    users = User.query.order_by(User.id).all()
    return render_template("admin.html", users=users)


# Feature 4 / 5 / 6: CSV upload, chart, peak-hour detection
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    stats = chart_labels = chart_data = peak_info = error = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("Please select a CSV file to upload.", "danger")
            return redirect(url_for("upload"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only .csv files are supported.", "danger")
            return redirect(url_for("upload"))

        try:
            stream = io.StringIO(file.stream.read().decode("utf-8", errors="replace"))
            stats, chart_labels, chart_data, peak_info = process_csv(stream)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "upload.html",
        stats=stats,
        chart_labels=chart_labels,
        chart_data=chart_data,
        peak_info=peak_info,
        error=error,
    )


# ---------------------------------------------------------------------------
# Initialise the database (runs on every cold start / import, including Vercel)
with app.app_context():
    _init_db()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
