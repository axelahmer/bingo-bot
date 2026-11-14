#!/usr/bin/env python3
"""Reference move generator built on the original 2023 solver.py (unchanged).

Replays a position, then enumerates every legal play the same way
ScrabbleBoard.sample_play_by_score does (get_csps -> solve_csp ->
calculate_score), and prints the plays in a canonical form that the Zig
engine can be diffed against:

    <new tiles sorted by square, as RR.CC.L> <TAB> <score>

Single-tile plays that form words in both directions are generated twice by
solver.py (once per direction); they collapse to one line here.
"""
import argparse
import contextlib
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_solver():
    # solver.py loads sowpods.txt (and writes corpus.pkl) relative to the
    # working directory at import time.
    prev = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    try:
        import solver
    finally:
        os.chdir(prev)
    return solver


def read_turns(path):
    """One turn per line: ROW COL across|down WORD, or pass (None)."""
    turns = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if line == 'pass':
                turns.append(None)
                continue
            row, col, direction, word = line.split()
            turns.append((int(row), int(col), direction, word.upper()))
    return turns


def placement(board, row, col, word, direction):
    tiles = []
    for i, letter in enumerate(word):
        pos = (row + i, col) if direction == 'down' else (row, col + i)
        if pos not in board.tiles:
            tiles.append((pos[0], pos[1], letter))
    tiles.sort()
    return ' '.join(f'{r:02d}.{c:02d}.{l}' for r, c, l in tiles)


def all_moves(board, board_cls):
    moves = {}
    raw = 0
    for row, col, direction, length, constraints in board.get_csps(min_new_letters=1):
        for word in board_cls.solve_csp((row, col, direction, length, constraints)):
            score, score_norm = board.calculate_score(row, col, word, direction)
            if score_norm > 0:
                raw += 1
                key = placement(board, row, col, word, direction)
                prev = moves.setdefault(key, score)
                assert prev == score, (key, prev, score)
    return moves, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('game', nargs='?', help='game file: one turn per line, ROW COL across|down WORD or pass')
    ap.add_argument('--turns', type=int, help='use only the first TURNS turns')
    ap.add_argument('--count', action='store_true', help='print only counts and timing')
    args = ap.parse_args()

    solver = load_solver()
    board = solver.ScrabbleBoard()
    turns = read_turns(args.game) if args.game else []
    if args.turns is not None:
        turns = turns[:args.turns]
    with contextlib.redirect_stdout(io.StringIO()):
        for turn in turns:
            if turn is not None:
                row, col, direction, word = turn
                board.place_word(row, col, word, direction)

    t0 = time.perf_counter()
    moves, raw = all_moves(board, solver.ScrabbleBoard)
    dt = time.perf_counter() - t0

    if not args.count:
        out = sys.stdout
        for key in sorted(moves):
            out.write(f'{key}\t{moves[key]}\n')
    print(f'python: turns={len(turns)} moves={len(moves)} raw={raw} '
          f'seconds={dt:.3f}', file=sys.stderr)


if __name__ == '__main__':
    main()
