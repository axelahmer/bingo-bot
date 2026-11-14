# bingo-bot

How high can a game of Scrabble score when nobody is trying to stop you?

This repo searches for very high-scoring games of a solitaire Scrabble in
which the whole bag is your rack: each turn you may play any tiles still in
the bag, up to seven, and the game ends when the bag is empty or no play
exists. The goal is the largest total score.

`solver.py` is the 2023 Python experiment. It is kept unchanged as the
reference implementation and the performance baseline. `src/` is the Zig
engine that replaces it.

## Build and run

Needs Zig 0.17.1 (`pip install ziglang==0.17.1` also provides it).

    zig build -Doptimize=ReleaseFast
    zig build test

The first command that needs the lexicon builds `sowpods.gaddag` from
`sowpods.txt` (about 1.6 s) and reuses it afterwards.

    zig-out/bin/bingo rollout --games 1000 --seed 1 --out best.txt   # 2023 policy, all cores
    zig-out/bin/bingo show --plays best.txt                          # print a game
    zig-out/bin/bingo moves --plays bench/game.txt --ply 10          # every legal play at ply 10
    zig-out/bin/bingo bench --plays bench/game.txt --engine csp      # timing per ply
    zig-out/bin/bingo verify --games 4                               # GADDAG vs the CSP port

A play file has one play per line, `ROW COL across|down WORD`, with blanks
written as lower-case letters. Both the Zig engine and
`bench/py_reference.py` read this format.

## Rules model

- Standard board, letter values and tile distribution, SOWPODS word list.
- A play puts 1 to 7 tiles from the bag in one line, forms only valid
  words, and touches the board (the first play covers the centre). Placing
  exactly 7 tiles scores a 50-point bonus.
- Blanks are off by default, as in `solver.py`. `--blanks 2` gives the full
  100-tile set.
- There are no racks, no opponent and no end-of-game rack adjustment. Any
  real two-player game, read as one sequence of plays, is also a legal
  solitaire game with at least the same total (end-of-game rack
  adjustments never add to the combined score), so with `--blanks 2` the
  solitaire maximum is an upper bound on the combined score of a real game.

## Engine

**Lexicon** (`src/lexicon.zig`). A GADDAG (Gordon 1994): every word is stored
once for each split point, as the reversed prefix, a separator, then the
suffix, so generation can start mid-word and grow both ways. It is
minimised by hash-consing identical child arrays: 267,751 words become
1,081,889 node records (8.3 MiB). Each record is one `u64`: a 27-bit child mask, an
accept bit, a 14-bit *reach* mask and a 22-bit index of the first child.
The child for symbol `s` is `nodes[base + popcount(mask & ((1 << s) - 1))]`.
The reach mask records which numbers of further letters can complete a
word from this node.

**Board** (`src/board.zig`). The board is stored in both orientations so
every line is a contiguous 15-byte slice. Each empty square keeps a 26-bit
cross-check mask and the face value of the perpendicular word through it.
A play updates only the squares at the ends of the runs it touches (at most
16), using the lexicon.

**Move generation** (`src/movegen.zig`). For every legal play, the walk
starts at its *pivot*: the rightmost square of the main word that holds a
tile or has a perpendicular word. From the pivot it reads the GADDAG
leftwards, then crosses the separator and fills the open squares to the
right. Each play has exactly one pivot, so each play is produced once.
At each empty square the letters to try are
`node.children & square.cross_check & bag.non_empty`, and a child is
entered only when its reach mask fits the free squares and the tiles left.
The running score is one `u64` of additive fields (letter points, cross
words, tiles, counts of double- and triple-word squares), so a tile is one
add and a play is scored in O(1) when it is recorded. Plays go to a
comptime visitor that can count, collect or sample them.

The usual rule, inherited from Appel and Jacobson's left parts, starts
each play at its leftmost anchor. With a 7-tile rack that is cheap. With
the whole bag as the rack, it enumerates every left part in open space
before meeting the tiles and cross-checks to the right, and in the early
and middle game it ran slower than the brute-force port of `solver.py`.
Pivoting on the rightmost constrained square meets every constraint before
the open squares. That change made mid-game generation 6 to 8 times faster.

**Sampling** (`src/search.zig`). The 2023 policy samples a play with
probability proportional to `exp(points_per_tile / T)`. Plays with the same
score and tile count share a weight, so the sampler counts plays per
(score, tiles) bucket, keeps one uniform representative per bucket by
reservoir sampling, and picks a bucket with the Gumbel-max trick. The draw
is exact, needs no `exp` or `log` per play, and cannot overflow.

**CSP port** (`src/csp.zig`). A direct Zig port of the `solver.py`
algorithm (halo, cross-checks by scanning the word list, one constraint
problem per slot, filter the whole word list). It serves as an independent
oracle for tests and as the middle benchmark tier, which separates the gain
from changing language from the gain from changing algorithm.

## Correctness

- `bench/compare.py` replays `bench/game.txt` and, at every ply, checks that
  `solver.py` and the Zig engine list exactly the same plays with the same
  scores.
- `bingo verify` plays random games with the full word list and compares the
  GADDAG generator with the CSP port at every position. It also re-scores
  every play from scratch and checks the incremental cross-checks against a
  full recompute. 4 games without blanks: 9,489,611 plays over 90 positions.
  2 games with two blanks: 72,697,862 plays over 39 positions. All matched.
- `zig build test` runs the same differential test on a 2,116-word
  dictionary, with and without blanks, plus checks on the lexicon, play
  parsing, scoring and the sampler's distribution.

## Benchmarks

Measured on a 4-core Intel Xeon at 2.1 GHz (cloud VM, timings vary by
about 10%), Zig 0.17.1 ReleaseFast, Python 3.13 with numpy 2.5.

Listing every legal play at selected plies of `bench/game.txt`. Python is
`solver.py` itself; Zig times are the best of 5 runs. Every row was checked
to give the same plays and scores in Python and Zig. The full table comes
from `python3 bench/compare.py bench/game.txt`.

| ply | legal plays | Python | Zig port of Python | Zig GADDAG |
|---:|---:|---:|---:|---:|
| 0 | 891,584 | 8.97 s | 72.7 ms | 47.0 ms |
| 2 | 613,386 | 11.18 s | 70.1 ms | 33.2 ms |
| 4 | 289,154 | 8.32 s | 53.1 ms | 17.5 ms |
| 6 | 109,500 | 8.15 s | 41.0 ms | 7.9 ms |
| 8 | 81,235 | 7.67 s | 37.3 ms | 6.8 ms |
| 10 | 32,345 | 6.98 s | 33.0 ms | 3.5 ms |
| 14 | 7,044 | 6.95 s | 26.1 ms | 0.86 ms |
| 18 | 121 | 6.11 s | 18.0 ms | 0.03 ms |
| 24 | 26 | 6.00 s | 16.4 ms | 0.02 ms |
| all 26 positions | 3,396,542 | 195.8 s | 881 ms | 205 ms |

Over the whole game, the same algorithm in Zig is 222 times faster than
Python, and the GADDAG generator is a further 4.3 times faster (956 times
in total). The GADDAG's cost follows the number of legal plays (about
16.6 million plays per second here), while the word-list filter costs
nearly the same at every ply, so the gap grows from about 190 times at the
first play to over 200,000 times at the end.

Full games with the 2023 policy (`bingo rollout --games 64`): 4.95 games
per second on one thread and 20.5 on four. In `solver.py` the same game
spends about 3 minutes in move generation alone.

## Notes on the 2023 code

Kept: the solitaire framing, points per tile as the greedy signal (tiles
are the budget), the annealed softmax rollout, and the anchor and
cross-check ideas. Scoring was right: every score matches.

Fixed in the port, still present in `solver.py`:

- `np.exp(scores / (temp + 0.001))` can overflow near the end of a game.
  With one tile left, T + 0.001 is about 0.0316, so a one-tile play scoring
  23 points gives `exp(727) = inf`, then NaN probabilities.
- When tiles remain but no play exists (it happens in a few percent of
  rollouts), `np.random.choice` gets an empty list and raises.
- A one-tile play that forms words both ways is generated once per
  direction, so it is sampled with twice its weight.
- Blanks are ignored, the `corpus.pkl` cache never notices a changed word
  list, and runs are not seeded.

## Next steps

- **Bounded generation.** Greedy play, beam search and top-K policies only
  need the best plays. An upper bound on what a partial play can still
  score (from the remaining squares' premiums and the highest letter values
  left) would let the walk skip most of the tree.
- **Incremental generation.** A play changes about a quarter of the 30
  lines. Caching each line's plays and regenerating only the changed lines
  (plus a bag check on the cached plays) should cut rollout cost again.
- **Search.** Nested Monte Carlo Search (Cazenave 2009) and Nested Rollout
  Policy Adaptation (Rosin 2011) set records on single-player puzzles such
  as Morpion Solitaire and fit this problem directly. Rollouts already
  scale linearly with threads.
- **An upper bound** on the total score for the tile set, so results can be
  reported with an optimality gap.
