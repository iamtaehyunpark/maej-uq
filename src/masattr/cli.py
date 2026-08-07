"""Entry point: dispatch to one experiment module.

Each subcommand is a self-contained ``runs/*.py`` with its own argparse, so
``masattr e1 --help`` shows exactly that experiment's arguments and nothing
else. This file only routes.

Manifest order (Part C §7)::

    masattr freeze                 # serialize prompts / type rules / type map + hashes
    masattr load --assert          # 126 AG / 58 HC / 4092 steps / 3+3 flagged
    masattr typecheck              # rules vs HC parsed types, ≥90% gate
    masattr retype --splitter ...  # gate the plan/delegate splitter on HC, apply to AG
    masattr judge --subsets alg hc --judge hf:<id>      [× readout × policy × GT]
    masattr e0 --scores ...        # field sanity + threshold stability; fixes the primary rule
    masattr e1 --scores ... --folds ... --decision ...   # primary vs baselines
    masattr baselines --generators openai:gpt-4o judge:hf:<id> --impl repo --repo-path ...
    masattr e2/e3/e4/e5/e6/e7      # ablations
    masattr e9 --e1-results ...    # stratification, no new runs

E8 (success-control) is gated behind an explicit owner decision — Part D — and
is not implemented.
"""

from __future__ import annotations

import sys
from typing import Sequence

from . import __version__
from .runs import COMMANDS, ORDER


def usage() -> str:
    return "\n".join(
        [
            f"masattr {__version__} — MAS attribution harness",
            "",
            "usage: masattr <command> [args]",
            "",
            "commands (manifest order):",
            *(f"  {c}" for c in ORDER),
            "  baselines",
            "  freeze        serialize prompts and type rules to specs/, then hash every spec",
            "",
            "`masattr <command> --help` for a command's arguments.",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "freeze":
        from . import specs

        hashes = specs.freeze()
        print("froze specs/:")
        for k, v in hashes.items():
            print(f"  {k}: {v}")
        return 0

    mod = COMMANDS.get(cmd)
    if mod is None:
        print(f"unknown command {cmd!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    return mod.main(rest)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
