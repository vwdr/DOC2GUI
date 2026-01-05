from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = BASE_DIR / "data" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _draw_form(pdf_path: Path, title: str, url: str, sections):
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    width, height = LETTER
    y = height - inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, y, title)
    y -= 0.4 * inch

    c.setFont("Helvetica", 10)
    c.drawString(inch, y, f"Target Form URL: {url}")
    y -= 0.3 * inch
    c.drawString(inch, y, "Use the provided user JSON to fill fields exactly.")
    y -= 0.5 * inch

    for section in sections:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(inch, y, section["title"])
        y -= 0.3 * inch
        c.setFont("Helvetica", 11)
        for field in section["fields"]:
            c.drawString(inch, y, f"{field}:")
            c.line(3 * inch, y - 2, width - inch, y - 2)
            y -= 0.25 * inch
            if y < inch:
                c.showPage()
                y = height - inch
    c.showPage()
    c.save()


def _draw_resume(pdf_path: Path) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    width, height = LETTER
    y = height - inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, y, "Sample Resume")
    y -= 0.4 * inch
    c.setFont("Helvetica", 11)
    c.drawString(inch, y, "Name: Jordan Example")
    y -= 0.25 * inch
    c.drawString(inch, y, "Experience: 3 years of data operations")
    y -= 0.25 * inch
    c.drawString(inch, y, "Skills: Python, SQL, Customer Support")
    c.showPage()
    c.save()


def main():
    _draw_form(
        PDF_DIR / "insurance_form.pdf",
        "Local Insurance Claim - Instructions",
        "http://127.0.0.1:8000/form/insurance",
        [
            {
                "title": "CLAIM DETAILS",
                "fields": [
                    "Full Name",
                    "Policy ID",
                    "Claim Amount (USD)",
                    "Incident Date",
                    "Incident Type",
                    "Agree to local test terms",
                    "Incident Notes",
                ],
            }
        ],
    )

    _draw_form(
        PDF_DIR / "employment_form.pdf",
        "Local Employment Application - Instructions",
        "http://127.0.0.1:8000/form/employment",
        [
            {
                "title": "APPLICANT INFO",
                "fields": [
                    "Full Name",
                    "Email",
                    "Phone",
                    "Position Applied For",
                    "Available Start Date",
                    "Eligible to work locally",
                    "Resume (PDF)",
                    "Cover Letter",
                ],
            }
        ],
    )

    _draw_form(
        PDF_DIR / "medical_form.pdf",
        "Local Medical Intake - Instructions",
        "http://127.0.0.1:8000/form/medical",
        [
            {
                "title": "PATIENT DETAILS",
                "fields": [
                    "Patient Name",
                    "Date of Birth",
                    "Insurance Provider",
                    "Symptoms Summary",
                    "Visit Type",
                    "Patient is fasting",
                    "Emergency Contact",
                ],
            }
        ],
    )

    _draw_resume(DOCS_DIR / "resume.pdf")

    print(f"Generated PDFs in {PDF_DIR}")


if __name__ == "__main__":
    main()
