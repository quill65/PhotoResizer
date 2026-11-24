"""Module entry point: launches GUI by default, or CLI when arguments are supplied."""

from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if not args or args == ["--gui"]:
        from photo_resizer.gui import main as gui_main

        return gui_main()

    from photo_resizer.cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
