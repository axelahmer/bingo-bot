# bingo-bot

How high can a game of Scrabble score when nobody is trying to stop you?

This repo searches for the highest-scoring Scrabble game: two players (or
one, in solitaire) who cooperate, play under the full rules, and get to
choose the draws. Earlier hand-built records appeared in *Word Ways*: Ralph
Beaman's cooperative game of 4,142 points (1974), Jeff Grant's 4,207 with
99 tiles (1979), and Grant's solitaire game of 4,454 using all 100 tiles
(1981), later raised to 4,470.

Best found so far (SOWPODS, full rules, two blanks): 3,693 in solitaire and
3,635 for two cooperating players, both in `results/` (see Search). The
best single play found is OXYPHENBUTAZONE for 1,619.

`solver.py` is the 2023 Python experiment. It is kept unchanged as the
reference implementation and the performance baseline. `src/` is the Zig
engine that replaces it.

## Build and run

Needs Zig 0.17.1 (`pip install ziglang==0.17.1` also provides it).

    zig build -Doptimize=ReleaseFast
    zig build test

The first command that needs the lexicon builds `sowpods.gaddag` from
`sowpods.txt` (about 1.6 s) and reuses it afterwards.

    zig-out/bin/bingo search --players 1 --hooks strict --per-signature 8 --seeds 2 --top 16 --out best.txt
    zig-out/bin/bingo beam --width 64 --out best.txt                 # beam search, no plans
    zig-out/bin/bingo plans --lane row0 --top 20                     # candidate 15-letter plays
    zig-out/bin/bingo rollout --games 1000 --seed 1 --out best.txt   # 2023 policy, all cores
    zig-out/bin/bingo check --game best.txt                          # replay with real racks and draws
    zig-out/bin/bingo show --game best.txt --turns 10                # the board after 10 turns
    zig-out/bin/bingo moves --game best.txt --turns 10               # every legal play there
    zig-out/bin/bingo bench --game bench/game-full.txt               # timing per turn
    zig-out/bin/bingo verify --games 4                               # GADDAG vs the CSP port

A game file has one turn per line: `ROW COL across|down WORD` (blanks
written as lower-case letters) or `pass`. Players alternate. Rule options:
`--players 1|2` (default 2), `--blanks N` (default 2), `--end wespa|naspa`
(default wespa). `solver.py`'s rules are `--players 1 --blanks 0`.

### Running on many cores

A beam search splits each layer's positions across `--threads` threads
(default: all cores). Between layers it sorts and merges serially, and the
first layer is one position, so on 4 cores one beam runs 3.6 times faster
than on one (about 4% of the work is serial). By Amdahl's law that is about
8 times on 12 cores.

Record attempts are many independent searches (plan sets times seeds), so
it is better to run several at once. `--jobs J` runs J searches at a time
and splits the threads between them; with one thread per search there are
no barriers at all. On 4 cores, four seeds took 55 s as four 1-thread jobs
and 60 s one after another with 4 threads each, and the gap grows with the
core count. Each search adds only a few MB (a width-256 run peaks at 72 MB,
most of it the lexicon and word sets, which the jobs share). Results depend
only on the plans, seed and search options, never on `--jobs` or
`--threads`.

On a 12-core machine:

    # many seeds of one plan set: one thread per search
    zig-out/bin/bingo beam --players 1 --width 128 --hooks strict --letter-cost 27 \
        --plan row0:OXYPHENBUTAZONE:0,1,3,4,7,11,14 --plan row14:SESQUISULPHIDES:0,3,4,7,9,11,14 \
        --seed 1 --seeds 48 --jobs 12 --out best.txt
    # many plan pairs, a few seeds each
    zig-out/bin/bingo search --players 1 --width 128 --hooks strict --letter-cost 27 \
        --top 48 --seeds 4 --jobs 12 --out best.txt
    # one very wide beam: all threads in one search
    zig-out/bin/bingo beam --players 1 --width 2048 --hooks strict --letter-cost 27 --plan ... --plan ...

The beam is chaotic, so many seeds at a moderate width are a better use of
cores than one very wide beam: width 512 did worse than 256 on the best
pair.

## Rules

- Standard board, letter values and 100-tile set (two blanks), SOWPODS.
- Each player draws 7 tiles and refills after every play while the bag
  lasts. A play places 1 to 7 tiles from the mover's rack in one line,
  forms only valid words, and touches the board (the first covers the
  centre and uses at least two tiles). All 7 at once scores 50 more.
- A player may pass (exchanging is never useful with chosen draws, so it
  is not offered). Six consecutive scoreless turns end the game, and each
  player then loses the value of their own rack.
- The game also ends when a player empties their rack with the bag empty.
  Under WESPA rules (the default, and the rules that go with SOWPODS) they
  gain the other player's rack value and the other player loses it. Under
  NASPA rules they gain twice that value.

The draws are chosen rather than random, since we want the best possible
game. A rack is then described by its size alone, and its contents are
decided when its tiles are played. The only constraint is that a play fits
the mover's rack size, which follows the refill rule. Any sequence of
turns that respects it is a real game: at every moment, give each player
the next tiles of that player's own future plays, plus at the end the tiles
they are left holding. `bingo check` builds exactly those racks and draws,
replays the game against them, and prints the full record. One consequence,
which matches the *Word Ways* history: a two-player game can never use all
100 tiles, because when one player goes out the other still holds at least
one.

With chosen draws and a shared goal, the two-player game is still a search
for one decision maker, who plays both sides and picks every draw. It
differs from solitaire only in these ways:

- The partner always holds up to 7 tiles, so the bag empties 7 tiles
  sooner and racks shrink earlier at the end. With racks described by size,
  the partner's rack never stops the mover from using any unplayed letter.
- The game ends when either player goes out, leaving at least one tile
  unplayed (99 at most).
- The end adjustment cancels out of the combined score under WESPA rules
  (the player going out gains exactly what the partner loses). Under NASPA
  rules it adds twice the value of the partner's last tiles.

Every two-player game, replayed by one player, is also a legal solitaire
game with one or more tiles still to play. So solitaire is the slightly
easier problem: under WESPA rules its best score is at least the
cooperative best, less the value of any leftover tiles the solitaire
player then cannot play.

## Search

### Where the points are

The whole bag holds 187 face points, and a premium square counts only for
the play that covers it. The big lever is the three triple-word squares on
each edge row and column: a 15-letter word across all three is worth 27
times its letters. The finishing play has to place those three squares
itself, and since a play holds at most 7 tiles, at least 8 of the word's
letters must already be on the board in exactly the right squares while the
triple-word squares stay empty. SOWPODS has 5,757 fifteen-letter words;
OXYPHENBUTAZONE finished with 7 tiles scores (41 + 3 + 10) x 27 + 50 = 1,508
on an otherwise empty lane. Two such plays, on rows 0 and 14, are worth more
than any whole game the rollouts find.

Score-driven search never sets these up. It spends triple-word squares one
at a time: a small beam built PAMPHLETEERINGS across row 0 in three plays
(TSADDIQIM down through (0,7), then PAMPHLET, then the extension) for
122 + 110 + 128 points, where one play could have been worth over 1,000.

### Two baselines

`bingo beam` (`src/beam.zig`) is a beam search over the number of tiles
placed. Layer t keeps the best positions with t tiles on the board, so
positions compete on score after spending the same number of tiles: the
per-tile normalisation of the 2023 solver applied to whole positions. Each
position keeps its best plays of every size, a play of k tiles feeds layer
t + k, and transpositions are merged by a Zobrist hash. Ties are broken by
a seeded hash, so each run is reproducible and different seeds give
different searches.

`bingo plans` and `bingo search` add lane plans (`src/plan.zig`). A plan
names a lane, a 15-letter word and the 7 squares its finish will fill.
While it is open the search keeps the finish squares (and the squares just
inside them) empty, lets each setup square take only its letter, holds the
finishing tiles back, keeps scarce setup letters for their squares, and
ranks positions by score plus a bonus per filled setup square. Once every
setup square is filled the finish is offered as a play; a plan that leaves
no legal play is dropped. `bingo search` pairs plans for rows 0 and 14 whose
words fit the natural tiles together, prefers finishes whose setup letters
are common, and beam-searches each pair with a few seeds.

Results under the full rules (two blanks):

| method | solitaire | two players | time |
|---|---:|---:|---:|
| best of 400 rollouts (2023 policy) | 1,820 | 1,814 | 23 s |
| beam, width 256, no plans | 1,826 | | 48 s |
| plan search: 16 pairs, 2 seeds each, width 64 | 3,336 | 3,251 | about 7 min |
| best pair, hooks and letter cost (below) | 3,693 | 3,635 | 47 s, 86 s |

The first three rows come from the code as it was before the changes
below; the last row is reproduced by

    P="--plan row0:OXYPHENBUTAZONE:0,1,3,4,7,11,14 --plan row14:SESQUISULPHIDES:0,3,4,7,9,11,14"
    zig-out/bin/bingo beam --players 1 --width 128 --seed 2 --hooks strict --letter-cost 27 $P --out best.txt
    zig-out/bin/bingo beam --players 2 --width 256 --seed 2 --hooks strict --letter-cost 27 $P --out best.txt

The two best games are in `results/` and replay under `bingo check`. The
solitaire game uses all 100 tiles; MATHEMATICIZING (1,346) and
XYLOTYPOGRAPHIC (1,157) across rows 0 and 14 are 75% of its score, and the
search learned to play both finishes last. The two-player game plays 99
tiles (a two-player game cannot play all 100).

### Hooks, guards and diversity

Three changes made the plans pay off.

- **Hooks** (`--hooks strict`). The squares beside a finish square are no
  longer kept empty. Each finishing tile's cross word grows into the board
  while it stays a word. The empty square at the open end of the run beside
  a finish square accepts only letters that keep "finish letter + run" a
  word (on a top or left lane; "run + finish letter" on a bottom or right
  lane), and plays along the run are checked the same way. The hook points
  a position already has count in its ranking, so the beam builds the hooks
  it can see. `--hooks prefix` also allows prefixes that are not yet words,
  with a ranking penalty for each; it gave the same results.
- **Guards.** Unfilled setup squares get the same rule, scoring nothing, so
  that their letter can always still be placed. Before guards, a blank Q
  above a setup O ended a plan, because no word ends in QO. A setup square
  on the middle row has runs on both sides, so its guard asks only for a
  substring of some word. The GADDAG answers that directly: a path from the
  root spelling the string backwards exists only for substrings of words. A
  plan is also dropped as soon as one of its setup letters fits neither the
  across nor the down cross-check.
- **Diversity** (`--per-signature K`). A layer's best positions tend to
  share one line of setup, and when it fails they all fail. This keeps at
  most K positions per layer with the same filled setup squares.

A bug fix came first. On the empty board the generator walks only across
plays and reports each one again as its mirror image down. That assumes
the constraints are symmetric, and plan restrictions are not, so it emitted
first plays that broke them. With restrictions both directions are now
walked. That alone turned the 3,336 game's settings into a 2,340 game: the
beam is chaotic, so settings have to be compared over many seeds.

Solitaire, width 64, 8 seeds per setting. "Both" counts the games where
both planned x27 plays were made:

| plans | setting | both | best game | best play |
|---|---|---:|---:|---:|
| MATHEMATICIZING + XYLOTYPOGRAPHIC | hooks off | 0 | 3,069 | 1,157 |
| | strict hooks | 1 | 3,422 | 1,439 |
| | strict, `--approach 10` | 2 | 3,422 | 1,430 |
| | strict, `--per-signature 8` | 6 | 3,414 | 1,442 |
| OXYPHENBUTAZONE + SESQUISULPHIDES | strict hooks | 8 | 3,624 | 1,610 |
| | strict, `--per-signature 8` | 8 | 3,641 | 1,589 |

OXYPHENBUTAZONE + SESQUISULPHIDES has the best estimate of any pair whose
words fit the natural tiles together (1,508 + 1,157), and with hooks the
search lands both plays from every seed tried. `bingo search` over the 24
best pairs (strict hooks, `--per-signature 8`, 2 seeds each) found nothing
better: the next best pairs reached 3,531.

The best two games came from one more change, `--letter-cost 27`. When a
play spends more of a letter than the plans' setups can spare, it is
charged that letter's value times 27, the points a blank on the setup
square would cost the x27 word. (Refusing such plays outright, with
`Options.guard`, made results worse.) In the 3,668 game YODELLER had taken the last
natural L, so SESQUISULPHIDES needed a blank; with the cost, YODELLER took
the blank and SESQUISULPHIDES scored 1,187 instead of 1,166. Across seeds
the cost is not uniformly better (3,601 to 3,693 at width 128), but it
gave both best games.

These ranking terms run for every generated play, billions per search, so
the visitor that keeps each position's best plays computes them in order of
cost. Score, setup bonus and hook change come first. Then, if even the
largest possible distance gain could not lift the play past the worst one
kept for its size, the play is dropped before the loops over its tiles.
That halved the time per play with the letter cost on (27.5 s to 13.9 s for
a width-64 search) and keeps exactly the same plays.

| game | total | OXYPHENBUTAZONE | SESQUISULPHIDES | other plays |
|---|---:|---:|---:|---:|
| `results/solitaire-3693.txt` | 3,693 | 1,580 | 1,187 | 926 (27 plays) |
| `results/two-player-3635.txt` | 3,635 (2,094 + 1,541) | 1,589 | 1,187 | 859 (24 plays) |

Both replay under `bingo check`. In the solitaire game the first play,
OWLLIKE down from (1,7), later became the hook BOWLLIKE (51 points through
the triple-word square at (0,7)). `results/two-player-play-1619.txt` (a
3,594 game) holds the best single play found: OXYPHENBUTAZONE for 1,619,
with hooks XI, BOWLLIKE, ZO and ESTRANGE.

The middle row is implemented too (`row7` and `col7` lanes, and `--mid K`
in `bingo search` adds a x9 plan across row 7) but does not work yet. Until
its finish is played, a row-7 plan is a wall across the middle of the
board, crossable only at its 8 setup squares, each fixed to one letter.
Runs with one explored a tenth as many plays and never finished it.

### Published records

Whole games. All were built by hand with chosen draws, except Sudbery's,
which was played out with random draws.

| score | tiles | players | word list | source |
|---:|---:|---|---|---|
| 3,108 | | 2, cooperative | | Darryl Francis, *Word Ways* 1972 |
| 4,047 | | 2, cooperative | general dictionaries, BENZOXYCAMPHORS | Ron Jerome, *Word Ways* 1974 |
| 4,142 | | 2, cooperative | general dictionaries, BENZOXYCAMPHORS | Ralph Beaman, *Word Ways* 1974 |
| 4,207 | 99 | 2, cooperative | general dictionaries, BENZOXYCAMPHORS | Jeff Grant, *Word Ways* 1979 |
| 4,454 | 100 | solitaire | general dictionaries, BENZOXYCAMPHORS | Jeff Grant, *Word Ways* 1981 |
| 4,470 | 100 | solitaire | as above, adjusted | Jeff Grant, *Word Ways* 2009 |
| 3,986 | 99 | 2, cooperative | OSPD and Webster's Tenth Collegiate | Stephen Root and Nathan Hedt, *Word Ways* 2003 |
| 3,674 | | 2 (3,658 by one) | North American and British lists | Clare Sudbery, 2004 (PixiePit) |

Single plays:

| score | word list | method | source |
|---:|---|---|---|
| 2,037 | general dictionaries (SESQUIOXIDIZING) | by hand | Josefa Heifetz, *Word Ways* 1974 |
| 1,778 | OWL and the Long Words List | by hand | Dan Stock, in the rec.games.board.scrabble FAQ |
| 1,784 | NASPA list | by hand | "Jodi and Eric", NASPA wiki |
| 1,786 | NWL23 | SAT solver, proved best for NWL23 | Bob Lucassen, Woogles blog 2024 |

The record plays are all OXYPHENBUTAZONE across an edge with seven hooks.
Dan Stock's: (54 x 27 = 1,458) + 50 + OPACIFYING 63 + BRAINWASHING 63 +
EJACULATING 63 + ZARFS 27 + PREINTERVIEWED 25 + AMELIORATIVE 17 + YELKS 12.
Every word of that play is in our SOWPODS list. Lucassen's 1,786 needs
OPACIFICATIONS, which is not.

So the published targets for SOWPODS are a 3,674-point game, played with
random draws, and a single play of about 1,780. Nothing published seems to
search for the whole-game maximum by computer: the game records are hand
constructions, the SAT solver addressed one play, and a 2024 Wolfram
Community post searched randomly for all-bingo games.

Against those targets: the 3,693 solitaire game passes the 3,674 game, but
that was a two-player game, and our best two-player game (3,635) is 39
points short of it. Our best single play (1,619) is about 160 points short
of the record plays; the gap is almost all hooks.

What limits the search now:

- **Hooks.** Hooks are found, not designed: the best plays carry 70 to 110
  points of them (BOWLLIKE through (0,7) in the best game). The record
  single plays carry 270 or more, from seven long hooks packed into one
  dense block, which needs a constraint solver to design.
- **The middle row.** No x9 play is made. The two free triple-word squares,
  (7,0) and (7,14), need a row-7 plan that does not split the board.
- **The rest of the game.** Outside the two big plays, about 920 points come
  from about 27 plays.
- **Letters.** A guard that refuses plays leaving a plan short of letters
  (`Options.guard`) made results worse: it also forbids spending a natural
  letter that a free blank could later replace.

## Engine

**Lexicon** (`src/lexicon.zig`). A GADDAG (Gordon 1994): every word is
stored once for each split point, as the reversed prefix, a separator, then
the suffix, so generation can start mid-word and grow both ways. It is
minimised by hash-consing identical child arrays: 267,751 words become
1,081,889 node records (8.3 MiB). Each record is one `u64`: a 27-bit child
mask, an accept bit, a 15-bit *reach* mask and a 21-bit index of the first
child. The child for symbol `s` is
`nodes[base + popcount(mask & ((1 << s) - 1))]`. Bit `d` of the reach mask
says some word can stop exactly `d` letters further on; with the
generator's need mask shifted into the same position, "can any word through
this node still fit?" is a single AND.

**Board** (`src/board.zig`). The board is stored in both orientations so
every line is a contiguous 15-byte slice. Each empty square keeps a 26-bit
cross-check mask and the face value of the perpendicular word through it.
A play updates only the squares at the ends of the runs it touches (at most
16), using the lexicon.

**Game** (`src/game.zig`). Turns, rack sizes, the bag, passing, the end of
the game and its rack adjustment, plus `realize`, which rebuilds real racks
and draws for a finished game and replays it with them.

**Move generation** (`src/movegen.zig`). For every legal play, the walk
starts at its *pivot*: the rightmost square of the main word that holds a
tile or has a perpendicular word. From the pivot it reads the GADDAG
leftwards, then crosses the separator and fills the open squares to the
right. Each play has exactly one pivot, so each play is produced once. The
usual rule, inherited from Appel and Jacobson's left parts, starts each play
at its leftmost anchor; with a 7-tile rack that is cheap, but when any
unplayed tile can be drawn it enumerates every left part in open space
before meeting the constraints on the right, and it ran slower than the
brute-force port of `solver.py`. Pivoting on the rightmost constrained square
made mid-game generation 6 to 8 times faster.

The rest is a series of smaller savings, each kept only if it measured
faster:

- At each empty square the letters to try are
  `node.children & square.cross_check & supply.non_empty`, and a child is
  entered only if its reach fits the free squares and the tiles left, read
  from per-pivot tables of need masks.
- The running score is one `u64` of additive fields (letter points, cross
  words, tiles, counts of double- and triple-word squares), and each line
  precomputes the delta for every letter on every empty square, so placing
  a tile is one load and one add. A play is scored in O(1) when recorded.
- Blanks are not walked. The walk chooses letters and counts how many of
  them the natural tiles cannot cover; when a play is recorded, every legal
  choice of blank tiles is reported with the points each blank loses
  subtracted. Walking blank choices repeats the same GADDAG paths for every
  subset of positions. This made blank-heavy positions about 6 times faster.
- The generator is specialised at compile time on whether any blank is
  left, so the solitaire and late-game paths carry no blank logic.
- On the empty board, every down play mirrors an across play with the same
  score (the board is symmetric about its diagonal), so only across is
  walked.
- The last square a word can reach records its plays without touching the
  tile counts, and tile availability is kept as a bitmask updated bit by bit
  (reading the counts as a vector after a one-byte store stalled on store
  forwarding and was slower despite fewer instructions).

Plays go to a comptime visitor that can count, collect or sample them, so
nothing is stored unless a visitor keeps it.

**Sampling** (`src/search.zig`). The 2023 policy samples a play with
probability proportional to `exp(points_per_tile / T)`. Plays with the same
score and tile count share a weight, so the sampler counts plays per
(score, tiles) bucket, keeps one uniform representative per bucket by
reservoir sampling, and picks a bucket with the Gumbel-max trick. The
reservoir skips ahead (after keeping play `n`, the next kept play is
`floor(n / U) + 1`), so a play costs an increment and a compare. The draw is
exact, needs no `exp` or `log` per play, and cannot overflow.

**CSP port** (`src/csp.zig`). A direct Zig port of the `solver.py`
algorithm (halo, cross-checks by scanning the word list, one constraint
problem per slot, filter the whole word list), extended to blanks by trying
every subset of new tiles. It is an independent oracle for tests and the
middle benchmark tier, which separates the gain from changing language
from the gain from changing algorithm.

## Correctness

- `bench/compare.py` replays `bench/game.txt` and, at every turn, checks that
  `solver.py` and the Zig engine list exactly the same plays with the same
  scores.
- `bingo verify` plays random games on SOWPODS and compares the GADDAG
  generator with the CSP port at every position. It also re-scores every
  play from scratch, checks the incremental cross-checks against a full
  recompute, and replays every finished game with real racks. All matched:

  | rules | positions | plays matched |
  |---|---:|---:|
  | solitaire, no blanks | 56 | 7,461,221 |
  | solitaire, two blanks, NASPA | 44 | 63,436,005 |
  | two players, two blanks, WESPA | 56 | 91,017,043 |
  | two players, two blanks, NASPA | 36 | 60,164,589 |

- `zig build test` runs the same differential test on a 2,116-word
  dictionary under several rule sets, plus checks on the lexicon, play
  parsing, scoring, the end-of-game rules, the blank-as-needed mode and the
  sampler's distribution.

## Benchmarks

Measured on a 4-core Intel Xeon at 2.1 GHz (cloud VM, timings vary by
about 5 to 10%), Zig 0.17.1 ReleaseFast, Python 3.13 with numpy 2.5.

Listing every legal play at each turn of `bench/game.txt` under
`solver.py`'s rules. Python is `solver.py` itself; Zig times are the best of
5 runs. Every row gives the same plays and scores in Python and Zig. The
full table comes from `python3 bench/compare.py bench/game.txt`.

| turn | legal plays | Python | Zig port of Python | Zig GADDAG |
|---:|---:|---:|---:|---:|
| 0 | 891,584 | 9.13 s | 80.6 ms | 20.0 ms |
| 2 | 613,386 | 11.67 s | 77.7 ms | 28.9 ms |
| 4 | 289,154 | 9.95 s | 75.5 ms | 20.3 ms |
| 6 | 109,500 | 7.59 s | 32.6 ms | 8.6 ms |
| 8 | 81,235 | 8.52 s | 27.8 ms | 4.5 ms |
| 10 | 32,345 | 7.07 s | 27.2 ms | 2.2 ms |
| 14 | 7,044 | 8.55 s | 34.0 ms | 0.82 ms |
| 18 | 121 | 7.42 s | 18.1 ms | 0.04 ms |
| 24 | 26 | 7.20 s | 0.6 ms | 0.02 ms |
| all 25 positions | 3,396,542 | 209.7 s | 840 ms | 168 ms |

Over the whole game the same algorithm in Zig is 250 times faster than
Python, and the GADDAG generator is a further 5 times faster (1,248 times in
total). The word-list filter costs about the same at every turn while the
GADDAG's cost follows the number of legal plays, so the gap grows from about
460 times at the first play to over 300,000 times at the end.

Under the full rules, `bench/game-full.txt` (two players, two blanks) has
23.6 million legal plays over its 14 turns, 21.6 million of them on the
empty board, where every word can use either blank on any of its tiles.
Listing them all takes about 180 ms.

Full games with the 2023 policy (`bingo rollout`), games per second:

| rules | 1 thread | 4 threads |
|---|---:|---:|
| full rules (two players, two blanks) | 4.0 | 15.7 |
| full rules, `--blank-use needed` | 4.4 | 16.5 |
| `solver.py`'s rules | 6.9 | 25.5 |

In `solver.py` a game spends about 3.5 minutes in move generation alone.

## Notes on the 2023 code

Kept: the cooperative framing, points per tile as the greedy signal (tiles
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
- Blanks, racks and the end of the game are not modelled, the `corpus.pkl`
  cache never notices a changed word list, and runs are not seeded.

## Next steps

- **Design the hooks.** The record plays get 270 points or more from seven
  long hooks packed into one dense block. Finding such a block is a small
  crossword-construction problem (a SAT or constraint search over the
  squares beside the lane, as Lucassen did for one play). Its letters can
  then become setup squares of the plan, which the beam already knows how
  to fill.
- **A third big play that does not split the board.** Row 7 between (7,0)
  and (7,14) gives a x9 play worth 300 to 500 points, but a row-7 plan
  walls off the two halves until it is played. Plans that open late, after
  the edge setups are done, may avoid that.
- **Remote setup squares.** Most failed setups leave a square far from the
  play (the Y at (0,2), the L and O at (14,2) and (14,3)). A distance
  potential (`--approach`) helped on one pair and hurt on another.
- **Better play around the plans.** Nested Monte Carlo Search (Cazenave
  2009) or Nested Rollout Policy Adaptation (Rosin 2011), which set records
  on single-player puzzles such as Morpion Solitaire, run over the beam's
  candidate plays, or a learned evaluation, for the 900 points that are not
  the big plays.
- **Cache move lists.** Every rollout from the empty board regenerates the
  same 21.6 million first plays, and nested search revisits positions
  constantly. A play changes about a quarter of the 30 lines, so caching
  each line's plays and regenerating only the changed lines should cut
  rollout cost again.
- **Bounded generation.** The beam keeps only the best few plays of each
  size from each position. An upper bound on what a partial play can still
  score would let the walk skip most of the tree.
- **An upper bound** on the total score for the tile set, so results can be
  reported with an optimality gap.
