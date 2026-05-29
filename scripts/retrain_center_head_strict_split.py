"""CLI: retrain atom-center head on a leakage-resistant split dataset.

This trains only the atom reaction-center classifier. It is not MLIP training.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_center_head import _build_parser, main as _train_main  # noqa: E402,I001


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.description = "Strict-split retraining for the Phase 8.11 atom center head. Not MLIP."
    parser.set_defaults(
        data=_ROOT / "data" / "reactions.center_validated.template_split.json",
        output=_ROOT / "models" / "atom_center_head_template_split.pt",
        report=_ROOT / "reports" / "atom_center_training_template_split_report.json",
    )
    args = parser.parse_args(argv)
    if args.output.resolve() == (_ROOT / "models" / "atom_center_head.pt").resolve():
        print("ERROR: refusing to overwrite models/atom_center_head.pt", file=sys.stderr)
        return 1
    delegated = []
    for key, value in vars(args).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                delegated.append(flag)
        else:
            delegated.extend([flag, str(value)])
    return _train_main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
