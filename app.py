import os
import functools
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ---------------------------------------------------------------------------
# In-memory user store (replace with a database in production)
# ---------------------------------------------------------------------------
users: dict[str, str] = {}


def login_required(view):
    """Redirect anonymous users to the login page."""
    @functools.wraps(view)
    def wrapped(**kwargs):
        if "user" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view(**kwargs)
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

        if username in users:
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        users[username] = generate_password_hash(password)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        stored_hash = users.get(username)
        if stored_hash and check_password_hash(stored_hash, password):
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
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None

    if request.method == "POST":
        date_str = request.form.get("date", "").strip()
        time_str = request.form.get("time", "").strip()
        num_houses = request.form.get("num_houses", "").strip()

        if not date_str or not time_str:
            flash("Date and time are required.", "danger")
            return redirect(url_for("predict"))

        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            flash("Invalid date or time format.", "danger")
            return redirect(url_for("predict"))

        # --- placeholder prediction logic --------------------------------
        # Replace this block with actual model inference once trained.
        hour = dt.hour
        month = dt.month
        houses = int(num_houses) if num_houses else 5567

        base_load = 1500.0  # kWh baseline
        hour_factor = 1.0 + 0.3 * abs(hour - 12) / 12.0
        season_factor = 1.0 + 0.2 * (1 if month in (12, 1, 2) else -0.1)
        predicted_kwh = round(base_load * hour_factor * season_factor * (houses / 5567), 2)
        # -----------------------------------------------------------------

        prediction = {
            "datetime": dt.strftime("%A, %B %d, %Y at %H:%M"),
            "houses": houses,
            "kwh": predicted_kwh,
        }

    return render_template("predict.html", prediction=prediction)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
