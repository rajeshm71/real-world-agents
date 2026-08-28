"""CLI entry point for the sample_repo demo project.

Ties together `greet` and `math_helpers`; used only as a target for
the agent's codebase Q&A demo.
"""

from __future__ import annotations

import argparse

from greet import farewell, greet
from math_helpers import add, multiply


def main() -> int:
    parser = argparse.ArgumentParser(prog="sample-repo")
    parser.add_argument("name")
    parser.add_argument("--add", type=float, nargs=2, metavar=("A", "B"))
    parser.add_argument("--multiply", type=float, nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    print(greet(args.name))
    if args.add:
        print(f"{args.add[0]} + {args.add[1]} = {add(*args.add)}")
    if args.multiply:
        print(f"{args.multiply[0]} * {args.multiply[1]} = {multiply(*args.multiply)}")
    print(farewell(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
