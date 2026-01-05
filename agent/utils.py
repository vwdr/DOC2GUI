from typing import List

from agent.planner import FieldInfo


def extract_fields(page) -> List[FieldInfo]:
    field_data = page.eval_on_selector_all(
        "[data-field]",
        """
        (elements) => elements.map(el => {
            const fieldId = el.getAttribute('data-field');
            const tag = el.tagName.toLowerCase();
            const type = el.getAttribute('type') || tag;
            const labelEl = document.querySelector(`label[for='${el.id}']`);
            const label = labelEl ? labelEl.textContent.trim() : fieldId;
            let options = [];
            if (tag === 'select') {
                options = Array.from(el.options).map(opt => opt.value).filter(Boolean);
            }
            return { fieldId, label, type, options, tag };
        })
        """,
    )

    fields = []
    for item in field_data:
        if item["fieldId"] == "submit":
            continue
        field_type = item["type"]
        if item["tag"] == "textarea":
            field_type = "textarea"
        if field_type == "checkbox":
            field_type = "checkbox"
        elif field_type == "file":
            field_type = "file"
        elif field_type == "select":
            field_type = "select"
        elif field_type in {"text", "number", "date"}:
            field_type = field_type
        else:
            field_type = "text"
        fields.append(
            FieldInfo(
                field_id=item["fieldId"],
                label=item["label"],
                field_type=field_type,
                selector=f"[data-field='{item['fieldId']}']",
                options=item.get("options") or None,
            )
        )
    return fields
