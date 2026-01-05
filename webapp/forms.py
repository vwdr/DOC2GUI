from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FieldOption:
    label: str
    value: str


@dataclass
class FieldDef:
    field_id: str
    label: str
    field_type: str  # text, number, date, select, checkbox, textarea, file
    required: bool = False
    options: Optional[List[FieldOption]] = None


@dataclass
class FormDef:
    form_id: str
    title: str
    description: str
    fields: List[FieldDef]


FORMS: List[FormDef] = [
    FormDef(
        form_id="insurance",
        title="Local Insurance Claim",
        description="Demo claim intake for local testing only.",
        fields=[
            FieldDef("full_name", "Full Name", "text", True),
            FieldDef("policy_id", "Policy ID", "text", True),
            FieldDef("claim_amount", "Claim Amount (USD)", "number", True),
            FieldDef("incident_date", "Incident Date", "date", True),
            FieldDef(
                "incident_type",
                "Incident Type",
                "select",
                True,
                options=[
                    FieldOption("Accident", "accident"),
                    FieldOption("Theft", "theft"),
                    FieldOption("Other", "other"),
                ],
            ),
            FieldDef("agree_terms", "I agree to the local test terms", "checkbox", True),
            FieldDef("notes", "Incident Notes", "textarea", False),
        ],
    ),
    FormDef(
        form_id="employment",
        title="Local Employment Application",
        description="Demo employment form for local testing only.",
        fields=[
            FieldDef("full_name", "Full Name", "text", True),
            FieldDef("email", "Email", "text", True),
            FieldDef("phone", "Phone", "text", True),
            FieldDef(
                "position",
                "Position Applied For",
                "select",
                True,
                options=[
                    FieldOption("Data Analyst", "data_analyst"),
                    FieldOption("Support Engineer", "support_engineer"),
                    FieldOption("Operations Associate", "ops_associate"),
                ],
            ),
            FieldDef("start_date", "Available Start Date", "date", True),
            FieldDef("eligible_work", "Eligible to work locally", "checkbox", True),
            FieldDef("resume", "Resume (PDF)", "file", False),
            FieldDef("cover_letter", "Cover Letter", "textarea", False),
        ],
    ),
    FormDef(
        form_id="medical",
        title="Local Medical Intake",
        description="Demo medical intake for local testing only.",
        fields=[
            FieldDef("patient_name", "Patient Name", "text", True),
            FieldDef("dob", "Date of Birth", "date", True),
            FieldDef("insurance_provider", "Insurance Provider", "text", False),
            FieldDef("symptoms", "Symptoms Summary", "textarea", True),
            FieldDef(
                "visit_type",
                "Visit Type",
                "select",
                True,
                options=[
                    FieldOption("Checkup", "checkup"),
                    FieldOption("Follow-up", "follow_up"),
                    FieldOption("Urgent", "urgent"),
                ],
            ),
            FieldDef("fasting", "Patient is fasting", "checkbox", False),
            FieldDef("emergency_contact", "Emergency Contact", "text", True),
        ],
    ),
]


FORM_INDEX = {form.form_id: form for form in FORMS}
