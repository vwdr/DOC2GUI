import json
from pathlib import Path
from typing import Dict

from flask import Flask, redirect, render_template, request, url_for

from webapp.forms import FORM_INDEX, FORMS


BASE_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = BASE_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))


def _serialize_submission(form_id: str, payload: Dict[str, str]) -> None:
    path = SUBMISSIONS_DIR / f"{form_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.route("/")
def index():
    return render_template("index.html", forms=FORMS)


@app.route("/form/<form_id>")
def render_form(form_id: str):
    form = FORM_INDEX.get(form_id)
    if not form:
        return redirect(url_for("index"))
    return render_template("form.html", form=form)


@app.route("/submit/<form_id>", methods=["POST"])
def submit_form(form_id: str):
    form = FORM_INDEX.get(form_id)
    if not form:
        return redirect(url_for("index"))
    payload = {}
    for field in form.fields:
        if field.field_type == "checkbox":
            payload[field.field_id] = "true" if request.form.get(field.field_id) else "false"
        elif field.field_type == "file":
            file = request.files.get(field.field_id)
            payload[field.field_id] = file.filename if file else ""
        else:
            payload[field.field_id] = request.form.get(field.field_id, "")
    _serialize_submission(form_id, payload)
    return render_template("submitted.html", form=form, payload=payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
