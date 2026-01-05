from typing import List

from agent.logger import ActionRecord, RunLogger
from agent.planner import ActionStep


def execute_actions(page, actions: List[ActionStep], logger: RunLogger) -> None:
    for idx, action in enumerate(actions, start=1):
        status = "ok"
        try:
            if action.action_type == "fill":
                page.fill(action.selector, action.value or "")
            elif action.action_type == "select":
                page.select_option(action.selector, action.value or "")
            elif action.action_type == "check":
                if str(action.value).lower() in {"true", "1", "yes"}:
                    page.check(action.selector)
                else:
                    page.uncheck(action.selector)
            elif action.action_type == "upload":
                if action.value:
                    page.set_input_files(action.selector, action.value)
            elif action.action_type == "submit":
                page.click(action.selector)
                page.wait_for_load_state("networkidle")
            else:
                status = "skipped"
        except Exception as exc:
            status = f"error: {exc}"
        logger.save_screenshot(page, idx)
        logger.log_action(
            ActionRecord(
                step=idx,
                action_type=action.action_type,
                selector=action.selector,
                value=action.value,
                status=status,
                evidence=action.evidence,
            )
        )
        logger.log_grounding(idx, action.evidence)
