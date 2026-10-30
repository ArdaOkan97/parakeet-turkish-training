import sqlite3
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "outputs/parakeet_turkish/trackio/parakeet-turkish.db"
db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    quoted = '"' + name.replace('"', '""') + '"'
    columns = db.execute(f"PRAGMA table_info({quoted})").fetchall()
    print(name, [col[1] for col in columns])
    print("rows", db.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    if any(col[1] == "step" for col in columns):
        print("max_step", db.execute(f"SELECT MAX(step) FROM {quoted}").fetchone()[0])
db.close()
