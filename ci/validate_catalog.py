"""Static model catalog validator (issue #102).

CI gate: exits 1 if any catalog/models/*.json fails the schema, so drift
fails loudly. Uses the packaged pydantic schema (single source of truth).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "catalog" / "models"


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if not target.is_dir():
        print(f"catalog dir not found: {target}")
        return 1
    from fusion_model_hub.catalog.schema import validate_dir

    try:
        entries = validate_dir(target)
    except ValueError as e:
        print(f"FM-CATALOG ERROR: {e}")
        return 1
    names = [e.name for e in entries]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        print(f"FM-CATALOG ERROR: duplicate model names: {dupes}")
        return 1
    print(f"FM-CATALOG OK: {len(entries)} entries valid in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
