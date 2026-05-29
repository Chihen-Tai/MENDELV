"""Print installed MACE calculator factories without creating a calculator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mendel.mlip import diagnose_mace_calculators  # noqa: E402


def main() -> int:
    report = diagnose_mace_calculators()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
