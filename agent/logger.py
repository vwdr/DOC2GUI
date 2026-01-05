import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ActionRecord:
    step: int
    action_type: str
    selector: str
    value: Optional[str]
    status: str
    evidence: List[str]


class RunLogger:
    def __init__(self, base_dir: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir = self.run_dir / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.action_log: List[ActionRecord] = []
        self.grounding: Dict[int, List[str]] = {}

    def save_screenshot(self, page, step: int) -> str:
        path = self.screenshot_dir / f"step_{step:02d}.png"
        page.screenshot(path=str(path), full_page=True)
        return str(path)

    def log_action(self, record: ActionRecord) -> None:
        self.action_log.append(record)

    def log_grounding(self, step: int, chunk_ids: List[str]) -> None:
        self.grounding[step] = chunk_ids

    def save(self) -> None:
        actions_path = self.run_dir / "actions.json"
        actions_path.write_text(json.dumps([asdict(a) for a in self.action_log], indent=2), encoding="utf-8")
        grounding_path = self.run_dir / "grounding.json"
        grounding_path.write_text(json.dumps(self.grounding, indent=2), encoding="utf-8")

    def save_audit_log(self, chunks: Dict[str, str]) -> None:
        lines = []
        for record in self.action_log:
            lines.append(f"Step {record.step}: {record.action_type} {record.selector} -> {record.value}")
            for chunk_id in record.evidence:
                chunk_text = chunks.get(chunk_id, "")
                lines.append(f"  - Evidence {chunk_id}: {chunk_text}")
        (self.run_dir / "audit_log.txt").write_text("\n".join(lines), encoding="utf-8")
