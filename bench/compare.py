#!/usr/bin/env python3
"""Check the Zig engine against the 2023 Python solver and time all three tiers.

For every turn of a game, solver.py (via py_reference.py) and the Zig GADDAG
generator each list every legal play; the lists must be identical, scores
included. Timings for the Zig GADDAG generator and the Zig port of the
Python algorithm come from `bingo bench`. Prints a Markdown table.

solver.py's rules are solitaire without blanks, so the Zig side runs with
`--players 1 --blanks 0`.

    python3 bench/compare.py bench/game.txt [--turns 0,5,10]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINGO = os.path.join(ROOT, 'zig-out', 'bin', 'bingo')
PYREF = os.path.join(ROOT, 'bench', 'py_reference.py')


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f'{" ".join(cmd)} failed:\n{p.stderr}')
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('game', help='game file')
    ap.add_argument('--turns', help='comma-separated turns to check (default: all)')
    ap.add_argument('--reps', type=int, default=5, help='Zig timing repetitions')
    args = ap.parse_args()
    game = os.path.abspath(args.game)
    rules = ['--players', '1', '--blanks', '0']

    bench = run([BINGO, 'bench', '--game', game, '--engine', 'csp', '--reps', str(args.reps)] + rules).stdout
    zig = {}
    for line in bench.splitlines():
        f = line.split()
        if f and f[0].isdigit():
            zig[int(f[0])] = (int(f[1]), float(f[2]), float(f[4]))
    turns = sorted(zig) if not args.turns else [int(x) for x in args.turns.split(',')]

    print('| turn | plays | Python (s) | Zig port of Python (ms) | Zig GADDAG (ms) | Python / GADDAG | match |')
    print('|---:|---:|---:|---:|---:|---:|:---:|')
    tot_py = tot_csp = tot_gad = 0.0
    for turn in turns:
        py = run([sys.executable, PYREF, game, '--turns', str(turn)])
        py_secs = float(re.search(r'seconds=([\d.]+)', py.stderr).group(1))
        gz = run([BINGO, 'moves', '--game', game, '--turns', str(turn)] + rules)
        same = sorted(py.stdout.splitlines()) == sorted(gz.stdout.splitlines())
        n, gad_ms, csp_ms = zig[turn]
        tot_py += py_secs
        tot_csp += csp_ms
        tot_gad += gad_ms
        ratio = f'{py_secs * 1e3 / gad_ms:,.0f}x' if gad_ms > 0 else 'n/a'
        print(f'| {turn} | {n:,} | {py_secs:.2f} | {csp_ms:.1f} | {gad_ms:.2f} | {ratio} | {"yes" if same else "NO"} |',
              flush=True)
        if not same:
            sys.exit(f'move lists differ at turn {turn}')
    print(f'| all | | {tot_py:.1f} | {tot_csp:.0f} | {tot_gad:.1f} | {tot_py * 1e3 / tot_gad:,.0f}x | |')


if __name__ == '__main__':
    main()
