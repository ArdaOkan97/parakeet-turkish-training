from datetime import datetime
from pathlib import Path
from subprocess import run
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[1]
output = root / "outputs/parakeet_turkish"
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
notes = (
    f"{stamp} (America/Los_Angeles): The final auditor now validates the actual exported transcription CLI smoke against the first fixed Turkish development source, WAV hash, exact/normalized paired benchmark transcript, Turkish dev-only fitted temperature, word surfaces and all raw/fitted/conditional word confidences. "
    "CPU regression now passes 77,141 checks over real original baseline900 evidence and a synthetic CLI-shape fixture; all14 deliberate corruptions are rejected. The synthetic fixture is not actual exported-model inference. "
    "Actual final-report/CLI integration remains pending the completion worker, and ASR-8 remains open. The auditor reports a clearly labeled report SHA at audit time plus a stable audited evidence-payload SHA excluding only status/verification, because the worker subsequently updates those administrative fields. "
    "No GPU inference or active training/helper mutations were performed for this audit preparation."
)
run([str(output / "tools/backlog.exe"), "task", "edit", "ASR-8", "--append-notes", notes], cwd=output, check=True)
run([str(root / "work/training-env/Scripts/python.exe"), str(output / "refresh_backlog.py")], cwd=root, check=True)
