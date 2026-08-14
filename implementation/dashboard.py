from __future__ import annotations

import sys
from pathlib import Path

dashboard_dir = Path(__file__).resolve().parent / "dashboard"
sys.path.insert(0, str(dashboard_dir))

from main import app  # noqa: E402


server = app.server


if __name__ == "__main__":
    app.run(debug=False)
