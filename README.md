# energy-forecasting-ml

A web application for forecasting aggregate electricity usage based on the
London Smart Meter dataset (5,400+ households, half-hourly readings).

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

| Path | Description |
|------|-------------|
| `app.py` | Flask application (routes, auth, prediction) |
| `templates/` | Jinja2 HTML templates |
| `static/style.css` | Stylesheet |
| `tests/test_app.py` | Automated tests |
| `requirements.txt` | Python dependencies |
