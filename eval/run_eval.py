import json
import time
from pathlib import Path
from typing import Dict, List

from agent.agent_loop import run_agent


TEST_CASES = [
    {
        "name": "insurance",
        "pdf": "data/pdfs/insurance_form.pdf",
        "user_data": "data/user_data/insurance.json",
        "form_url": "http://127.0.0.1:8000/form/insurance",
        "expected": "data/expected/insurance.json",
    },
    {
        "name": "employment",
        "pdf": "data/pdfs/employment_form.pdf",
        "user_data": "data/user_data/employment.json",
        "form_url": "http://127.0.0.1:8000/form/employment",
        "expected": "data/expected/employment.json",
    },
    {
        "name": "medical",
        "pdf": "data/pdfs/medical_form.pdf",
        "user_data": "data/user_data/medical.json",
        "form_url": "http://127.0.0.1:8000/form/medical",
        "expected": "data/expected/medical.json",
    },
]


def _load_json(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare(expected: Dict[str, str], actual: Dict[str, str]) -> Dict[str, float]:
    total = len(expected)
    correct = 0
    for key, value in expected.items():
        if str(actual.get(key, "")) == str(value):
            correct += 1
    return {
        "total_fields": total,
        "correct_fields": correct,
        "accuracy": correct / total if total else 0.0,
    }


def main() -> None:
    reports: List[Dict[str, float]] = []
    for case in TEST_CASES:
        start = time.time()
        args = type("Args", (), {})()
        args.pdf = case["pdf"]
        args.user_data = case["user_data"]
        args.form_url = case["form_url"]
        args.docs = ["data/docs/policy.txt"]
        args.index_dir = f"data/index/{case['name']}"
        args.model_path = "models/layout_cnn.pt"
        args.embed_model = "sentence-transformers/all-MiniLM-L6-v2"
        args.llm_provider = "transformers"
        args.llm_model = "google/flan-t5-base"
        args.top_k = 6
        args.headless = True
        args.rebuild_index = True
        args.run_dir = "runs"

        run_dir = run_agent(args)
        duration = time.time() - start

        submission_path = Path("webapp/submissions") / f"{case['name']}.json"
        expected_path = Path(case["expected"])
        if not submission_path.exists():
            print(f"Missing submission for {case['name']}")
            continue

        expected = _load_json(expected_path)
        actual = _load_json(submission_path)
        score = _compare(expected, actual)

        actions_path = Path(run_dir) / "actions.json"
        actions = json.loads(actions_path.read_text(encoding="utf-8"))
        retries = sum(1 for action in actions if action["status"].startswith("error"))
        grounded = sum(1 for action in actions if action.get("evidence"))
        groundedness = grounded / len(actions) if actions else 0.0

        reports.append(
            {
                "case": case["name"],
                "duration_sec": round(duration, 2),
                "accuracy": round(score["accuracy"], 3),
                "correct_fields": score["correct_fields"],
                "total_fields": score["total_fields"],
                "retries": retries,
                "groundedness": round(groundedness, 3),
                "run_dir": str(run_dir),
            }
        )

    out_path = Path("runs") / "eval_report.json"
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
