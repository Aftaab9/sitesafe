"""Cross-platform task runner for machines without `make` (e.g. Windows).

    python tasks.py                 list targets
    python tasks.py test            run one target
    python tasks.py train EPOCHS=30

It reads the Makefile, so the two can never disagree.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def parse():
    targets, order, cur = {}, [], None
    for line in (ROOT / "Makefile").read_text().splitlines():
        m = re.match(r"^([a-z-]+):\s*([a-z -]*?)\s*(?:##\s*(.*))?$", line)
        if m and not line.startswith("\t"):
            cur = m.group(1)
            targets[cur] = {"deps": m.group(2).split() if not m.group(3) else [], "help": m.group(3) or "", "cmds": []}
            order.append(cur)
        elif line.startswith("\t") and cur:
            targets[cur]["cmds"].append(line.strip())
    return targets, order


def expand(cmd: str, var: dict) -> str:
    cmd = cmd.replace("$(PY)", f'"{sys.executable}"')
    cmd = re.sub(r"\$\(or \$\((\w+)\),([^)]*)\)", lambda m: var.get(m.group(1), m.group(2)), cmd)
    return cmd


def run(name: str, targets: dict, var: dict):
    t = targets[name]
    for d in t["deps"]:
        run(d, targets, var)
    for c in t["cmds"]:
        c = expand(c, var)
        print(f"\n> {c}", flush=True)
        if subprocess.call(c, shell=True, cwd=ROOT) != 0:
            sys.exit(f"step failed: {c}")


if __name__ == "__main__":
    targets, order = parse()
    args = [a for a in sys.argv[1:] if "=" not in a]
    var = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if not args:
        for n in order:
            print(f"  {n:<14} {targets[n]['help']}")
        sys.exit(0)
    for a in args:
        if a not in targets:
            sys.exit(f"unknown target: {a}")
        run(a, targets, var)
