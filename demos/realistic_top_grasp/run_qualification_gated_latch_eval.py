from __future__ import annotations

import sys

from run_contact_only_eval import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "qualification_gated_latch_eval"])
    raise SystemExit(main())
