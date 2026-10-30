"""Rebuild the readable experiment backlog from canonical Backlog.md tasks.

Update task notes/status through tools/backlog.exe first, then run this script.
No canonical task files are hand-edited. The root backlog.md is a derived report.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DISPLAY_ZONE = ZoneInfo("America/Los_Angeles")


def local_timestamp(value: str) -> str:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    local = instant.astimezone(DISPLAY_ZONE)
    return f"{local:%Y-%m-%d %H:%M:%S %Z} (America/Los_Angeles, {local:%z})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(ROOT / "tools" / "backlog.exe"))
    args = parser.parse_args()
    binary = Path(args.binary).resolve()

    def cli(*arguments: str) -> str:
        return subprocess.run([str(binary), *arguments], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8").stdout

    plan_path = ROOT / "experiment_plan.json"
    if plan_path.exists():
        from experiment_plan_backlog_view import description
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_task = plan.get("backlog_plan_task")
        if plan_task:
            rendered = description(plan)
            current = json.loads(cli("task", "view", plan_task, "--json"))["task"]
            if current.get("description", "").strip() != rendered.strip():
                cli("task", "edit", plan_task, "--description", rendered)

    cli("board", "export", "backlog.md", "--force")
    target = ROOT / "backlog.md"
    report = target.read_text(encoding="utf-8").rstrip()
    report = re.sub(r"(?m)^Generated on: .+$", f"Generated on: {local_timestamp(datetime.now(timezone.utc).isoformat())}", report)
    report += "\n\nThis board and the experiment notes below are generated from the official [Backlog.md](https://github.com/MrLesk/Backlog.md) CLI's local task files. Recorded results describe completed checks; pending work is explicitly identified. Displayed timestamps use America/Los_Angeles; canonical task metadata stores UTC.\n\n"
    summaries = json.loads(cli("task", "list", "--json"))["tasks"]
    summaries.sort(key=lambda task: tuple(int(part) for part in task["id"].rsplit("-", 1)[-1].split(".")))
    for summary in summaries:
        task = json.loads(cli("task", "view", summary["id"], "--json"))["task"]
        task_path = (ROOT / task["path"]).resolve().as_posix()
        report += f"## [{task['id']} — {task['title']}](<{task_path}>)\n\n"
        report += f"Status: **{task['status']}**. Verified acceptance criteria: {task['acceptanceCriteriaCompleted']}/{task['acceptanceCriteriaCount']}.\n\n"
        if task.get("updatedAt"):
            report += f"Updated: {local_timestamp(task['updatedAt'])}.\n\n"
        if task.get("description"):
            report += task["description"].rstrip() + "\n\n"
        if task.get("implementationNotes"):
            report += task["implementationNotes"].rstrip() + "\n\n"
        if task.get("implementationPlan"):
            report += "Current plan:\n\n" + task["implementationPlan"].rstrip() + "\n\n"
        if task.get("acceptanceCriteria"):
            report += "Acceptance criteria:\n\n"
            for criterion in task["acceptanceCriteria"]:
                mark = "x" if criterion["checked"] else " "
                report += f"- [{mark}] {criterion['text']}\n"
            report += "\n"
    target.write_text(report.rstrip() + "\n", encoding="utf-8")
    print(f"Refreshed experiment board and notes: {target}")


if __name__ == "__main__":
    main()
