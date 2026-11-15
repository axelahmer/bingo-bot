# bingo-bot

How high can a game of Scrabble score when nobody is trying to stop you?

This repo searches for the highest-scoring Scrabble game: two players (or
one, in solitaire) who cooperate, play under the full rules, and get to
choose the draws. Earlier hand-built records appeared in *Word Ways*: Ralph
Beaman's cooperative game of 4,142 points (1974), Jeff Grant's 4,207 with
99 tiles (1979), and Grant's solitaire game of 4,454 using all 100 tiles
(1981), later raised to 4,470.

Best found so far (SOWPODS, full rules, two blanks):
- **Our own design:** 3,864 in solitaire and 3,844 for two cooperating
  players. The best single play found is OXYPHENBUTAZONE for 1,619.
- **Hedt's board:** 4,049 in solitaire, using all 100 tiles. This is
  Nathan Hedt's 2014 record board (4,044 from 99 tiles, Collins words),
  played in an order our search found.
- **The record:** Hedt's own order gives 4,054 with all 100 tiles. This is
  the published solution of Al Zimmermann's 2014 contest Alphabet City
  Light, and is in `results/hedt-4054.txt`.

All of these games are in `results/` and replay under `bingo check`. Every
record we know of, by type and word list, is in [RECORDS.md](RECORDS.md).

`solver.py` is the 2023 Python experiment. It is kept unchanged as the
reference implementation and the performance baseline. `src/` is the Zig
engine that replaces it.

## Build and run

Needs Zig 0.17.1 (`pip install ziglang==0.17.1` also provides it).

    zig build -Doptimize=ReleaseFast
    zig build test

The first command that needs the lexicon builds `sowpods.gaddag` from
`sowpods.txt` (about 1.6 s) and reuses it afterwards.

    zig-out/bin/bingo search --players 1 --hooks prefix --per-signature 8 --seeds 2 --top 16 --out best.txt
    zig-out/bin/bingo beam --width 64 --out best.txt                 # beam search, no plans
    zig-out/bin/bingo plans --lane row0 --top 20                     # candidate 15-letter plays
    zig-out/bin/bingo rollout --games 1000 --seed 1 --out best.txt   # 2023 policy, all cores
    zig-out/bin/bingo check --game best.txt                          # replay with real racks and draws
    zig-out/bin/bingo html --game best.txt --out best.html           # a page that replays it on a board
    zig-out/bin/bingo gif --game best.txt --out best.gif             # the same as an animated GIF
    zig-out/bin/bingo improve --game best.txt --plan ... --jobs 12   # large-neighbourhood search from a game
    zig-out/bin/bingo trace --game best.txt --plan ...               # when and why each plan was lost
    zig-out/bin/bingo show --game best.txt --turns 10                # the board after 10 turns
    zig-out/bin/bingo moves --game best.txt --turns 10               # every legal play there
    zig-out/bin/bingo bench --game bench/game-full.txt               # timing per turn
    zig-out/bin/bingo verify --games 4                               # GADDAG vs the CSP port

Word lists: `sowpods.txt` (267,751 words) is the default. Collins Scrabble
Words 2024 (280,887 words, 13,759 more than ours and 623 fewer) is not in
the repository, since the Collins lists are copyrighted. To fetch it:

    curl -L -o CSW24.txt "https://raw.githubusercontent.com/echo544/scrabble-game/main/CSW24.txt"
    tr -d '\r' < CSW24.txt | awk '{print toupper($1)}' | grep -E '^[A-Z]{2,15}$' | sort -u > csw24.txt

then pass `--words csw24.txt --cache csw24.gaddag` to any command.

A game file has one turn per line, or several separated by commas. Each
turn is one of:
- `ROW COL across|down WORD`, counting from 0;
- standard notation: `8B WORD` plays across from row 8, column B, and
  `B7 WORD` plays down;
- `pass`.

Blanks are written as lower-case letters, so a published move list can be
pasted as it is. Players alternate. Rule options:
`--players 1|2` (default 2), `--blanks N` (default 2), `--end wespa|naspa`
(default wespa). `solver.py`'s rules are `--players 1 --blanks 0`.

`bingo html` writes one self-contained page that plays games out on a
board, one move at a time: arrow keys step, space plays and pauses, and
the move list jumps to any turn. Pass `--game` once per game (up to 8) to
choose between them on the page. It needs no network: the page carries
the moves, scores and words.

`bingo gif` writes the same replay as a looping animated GIF (540 by 626
pixels, about 15 KB a turn) for sharing. The GIF shows the board after
each turn with the new tiles outlined, and the word, score and running
total below. `--title` sets the line above the board, and `--delay` the
time per turn in milliseconds (default 1000). Big plays and the final
position stay up longer. The letters are Inter, rasterized once into
`src/glyphs.zig` by `tools/glyphs.py`, so nothing is needed at run time.

### Records

[RECORDS.md](RECORDS.md) lists the best published games and plays by type
(one player, two players, single play) and word list (Collins, North
American, general dictionaries), with ours beside them and a log of every
saved search. It is generated from `records.csv`, so edit that and run

    python3 tools/records.py

After a search saves a game, log it with

    python3 tools/records.py add results/new-game.txt

which replays the game with `bingo check` (so the score is checked, not
copied), takes the settings, word list and search time from the file's
header, adds the machine and commit, and regenerates the page. `--play`
logs the game's best single play instead, and `--note TEXT` adds a note.

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

    # Hedt's three words, many seeds: one thread per search
    P="--plan row0:MICROTECHNIQUES:0,2,7,9,11,12,14 --plan row7:UNPROGRESSIVELY:0,1,10,11,12,13,14 --plan row14:OXYPHENBUTAZONE:0,1,2,3,7,11,14"
    zig-out/bin/bingo beam --players 1 --width 128 --hooks prefix --per-signature 4 --letter-cost 27 \
        $P --seed 1 --seeds 48 --jobs 12 --out best.txt
    # many plan sets, a few seeds each
    zig-out/bin/bingo search --players 1 --width 128 --hooks prefix --per-signature 4 --letter-cost 27 \
        --mid 2 --top 24 --seeds 4 --jobs 12 --out best.txt
    # large-neighbourhood search from the best game so far, for hours
    H="--plan col7:UNPROGRESSIVELY:0,1,10,11,12,13,14 --plan col0:MICROTECHNIQUES:0,2,7,9,11,12,14 --plan col14:OXYPHENBUTAZONE:0,1,2,3,7,11,14"
    zig-out/bin/bingo improve --players 1 --game results/hedt-4054.txt --hooks prefix \
        --goal-bonus 0 --mid-bonus 0 --width 256 $H --rounds 500 --jobs 12 \
        --free-plays 6 --radius 3 --seed 1 --out better.txt

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
| a middle spine and two edges (below) | 3,828 | 3,816 | 90 s, 45 s |

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
  with a ranking penalty for each. It first seemed to make no difference,
  but strict hooks rule out the record games' lines: Hedt builds MIFF as
  MIF first, and QAT as QA. With today's code prefix hooks are better on
  our own plans too (see "Designed hooks, board targets and Hedt's
  board").
- **Guards.** Unfilled setup squares get the same rule, scoring nothing, so
  that their letter can always still be placed. Before guards, a blank Q
  above a setup O ended a plan, because no word ends in QO. A setup square
  on the middle row has runs on both sides, so its guard asks only for a
  substring of some word. The GADDAG answers that directly: a path from the
  root spelling the string backwards exists only for substrings of words. A
  plan is also dropped when one of its setup letters can never be laid:
  it fits neither cross-check, and neither run beside it can still grow
  into a word.
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

### A middle spine, as in the record games

Two record games have full move lists: Jeff Grant's 4,454 (solitaire, *Word
Ways* 1981) and Nathan Hedt's 4,044 (Collins words, 2014; see below). Both
have the same shape. The first play goes through the centre, and by the
fourth play the middle row (or column) holds a 15-letter word scored x9:
Grant's JACKPUDDINGHOOD around PUDDING for 504, Hedt's UNPROGRESSIVELY
around PROGRESS for 302. That word is a spine. Both edge setups are built
off it, and the two x27 plays come last.

Our first middle plans were finished last like the edge plans, so until
the end the middle row was a wall, crossable only at its setup squares,
each fixed to one letter. Runs with one explored a tenth as many plays and
never finished it. Now a middle plan has the record shape. Its setup is an
8-letter word through the centre (PROGRESS), and its finish takes both ends
(UN and IVELY). It earns its own small bonus per setup square
(`--mid-bonus`, default 20; eight squares are worth less than the 302-point
finish), so the beam plays the finish as soon as it can.

Hedt's three words in our orientation (`--plan row0:MICROTECHNIQUES:0,2,7,9,11,12,14
--plan row7:UNPROGRESSIVELY:0,1,10,11,12,13,14 --plan
row14:OXYPHENBUTAZONE:0,1,2,3,7,11,14`), solitaire, width 64, strict hooks,
letter cost 27, 8 seeds per setting:

| setting | seeds above 3,700 | best game |
|---|---:|---:|
| `--mid-bonus 66` (a third of the edge bonus, as before) | 0 | 2,648 |
| `--mid-bonus 0` | 0 | 3,618 |
| `--mid-bonus 20` | 1 | 3,791 |
| `--mid-bonus 20 --per-signature 8` | 4 | 3,816 |
| `--mid-bonus 20 --per-signature 4` | 8 | 3,824 |

At width 128 the best solitaire game is 3,828 (`results/solitaire-3828.txt`).
For two players at width 64 all eight seeds scored 3,756 to 3,816
(`results/two-player-3816.txt`, 1,922 + 1,894). In that game UNPROGRESSIVELY
came at turn 16 for 302, MICROTECHNIQUES scored 1,246 and OXYPHENBUTAZONE
1,520; the other 24 plays scored 748.

`bingo search --mid K` adds the K best middle words of this shape that fit
with each pair. It now also tries each pair both ways round (hooks run off
the first letters of a top-row word and onto the last letters of a
bottom-row word) and allows 4-letter setup runs on the edges by default
(`--max-run 4`), which MICROTECHNIQUES needs for ROTE.

### Designed hooks, board targets and Hedt's board

**Why two x27 plays and one x9.** There are eight triple-word squares, and
a premium counts only for the play that covers it, so each one multiplies
exactly one play. A 15-letter word on an edge row takes three of them
(x27), and one on the middle row takes the other two (x9; the centre was
covered by the first play). Two opposite edges and the middle row use all
eight, and no other split comes close: a second word on an adjacent edge
shares a corner and gets x9 at best. A x27 play is worth 1,200 to 1,600
points against about 30 for a typical play, so a game without all three
needs about 1,500 points from elsewhere. Every record game has this
structure.

`--drop-penalty V` (default 2000 for `search`) is the pruning that follows
from this. A line that loses one of its planned words drops V points per
lost word in the ranking. It survives only when the beam has nothing better,
so the search never ends empty-handed. A hard prune ended with no game at
all in our tests, once every line had lost something.

**Designed hooks.** A plan can name hooks as tails, words laid beside a
finish square so the finishing tile makes a chosen cross word:

    --plan row0:MICROTECHNIQUES:0,2,7,9,11,12,14:0=IFF,7=LAWLIKE

This sets up MIFF and CLAWLIKE down the board. LAWLIKE, Hedt's first play,
ends on the centre square. Tail squares behave like setup squares:
- they take only their letter;
- they earn a share of the hook's points in the ranking (`--tail-bonus`, in
  percent);
- the finish waits until they are full;
- the square past a tail's end stays empty;
- a tile placed beside an empty tail square must make a whole word with
  its letter.

With Hedt's hooks as tails our search reproduced his MICROTECHNIQUES for
exactly 1,325 and his OXYPHENBUTAZONE for exactly 1,586, though not both in
one game. The top edge needs a connector word across row 1, as his
ALIENAGE does: it supplies the second letters of four hooks and three
setup squares at once.

**Board targets.** `--target ROW,COL,across|down,WORD` fixes a word on the
board. `--target-game F` fixes every tile of game F's final board and keeps
every other square empty until they are all filled. The search then
replays that board in the best order it can find, and places any tiles
left over.

Re-sequencing Hedt's board this way gave **4,049 with all 100 tiles**
(`results/solitaire-4049-hedt-board.txt`). That beats his published 4,044,
but only after two changes:
- strict hooks forbid his line (MIF, QA);
- the setup bonus had to be set to 0 (`--goal-bonus 0 --mid-bonus 0`). The
  default 200 per setup square made the beam fill setup squares first and
  lose his strong opening (LAWLIKE 86, PROGRESS 62, UNPROGRESSIVELY 302 by
  turn 3). It scored 3,862.

His own order scores 5 more: IF then FA makes 35 where our IFF makes 30.
His published 4,054 (`results/hedt-4054.txt`, from the contest's final
report) is the 4,044 game with the leftover O played as KO for 10, which is
the most that O can score anywhere.

**Learning targets: large-neighbourhood search.** `bingo improve --game G`
treats the best game so far as the target and searches around it. Each
search in a round:
1. frees the tiles of 1 to `--free-plays` random plays (never a planned
   finish) and the empty squares within `--radius` of them;
2. keeps every other tile as a target;
3. starts from the game's position before its first freed play, and
   beam-searches the rest.

Rounds run `--jobs` searches at once, and any better game becomes the next
target, so the targets are learned from the search's own results. From
Hedt's 4,054 it found nothing better in 12 rounds of 4 searches (width 256,
radius 2). That is a job for many cores and hours.

**Results with prefix hooks.** Our own three plans in our orientation,
letter cost 27, `--per-signature 4`, 8 seeds:

| setting | scores | best |
|---|---|---:|
| solitaire, width 64, strict hooks | 3,125 to 3,822, most near 3,800 | 3,822 |
| solitaire, width 64, prefix hooks | 3,189 to 3,854, most near 3,840 | 3,854 |
| solitaire, width 128, prefix hooks | 3,584 to 3,864 | **3,864** |
| two players, width 64, prefix hooks | 2,720 to 3,844 | **3,844** |

The two bold games are `results/solitaire-3864.txt` and
`results/two-player-3844.txt`.

**Bugs found on the way:**
- *Plans dropped too early.* A plan was dropped as soon as a setup letter
  failed both cross-checks now, even when the run beside the square could
  still grow into a word.
- *Finishes took other plans' tiles.* A finish took any natural tile in the
  pool, including ones reserved for another plan's finish. The other plan
  then needed a blank that nothing had kept back.
- *Finished games dropped.* A layer stopped reading candidates once it was
  full, so a game that went out with a lower rank was never recorded.
- *Candidate words cut short.* `search` stopped collecting a word's
  alternative finish squares once 80 words had been seen.

`bingo trace --game G --plan ...` replays a game against plans and reports
when each one stopped being reachable, and why.

### Learned weights: Nested Rollout Policy Adaptation

**Symmetry.** Only the transpose (rows and columns swapped) maps words to
words; rotations and flips reverse them. Plans fix one orientation, and
the beam merges positions that different orders of the same plays reach
(a Zobrist hash of the board), so symmetry costs nothing.

**Branch and bound.** Exact search over whole games is out of reach: about
30 turns, each with up to millions of plays. It is plausible for fixed
sub-problems with strong bounds. The main one is the best order of play
for a given final board, since the score of a play depends only on which
squares are already filled.

**NRPA.** `bingo nrpa` is Nested Rollout Policy Adaptation (Rosin 2011),
the method behind several records in single-player puzzles such as
Morpion Solitaire. A rollout plays a whole game, choosing each play with
probability proportional to exp(weight + bias × score). A level-n search
runs `--iterations` level-(n-1) searches and after each one moves the
weights towards the best game so far. The weights are the learned targets:
they come to name the plays, and their order, that good games share. Plays
are restricted exactly as in the beam search, so `--plan`, `--target` and
`--target-game` steer it the same way, and `--jobs` runs independent
searches on separate cores.

On Hedt's board (`--target-game results/hedt-4054.txt`, prefix hooks, no
setup bonus), a rollout with its adaptation takes about 1.5 ms. Level 2
(10,000 rollouts) takes 35 s per job, and level 3 about an hour. Best of 4
jobs at level 2:

| `--bias` | best |
|---|---:|
| 0.005 | 3,172 |
| 0.02 | 4,011 |
| 0.05 | 4,007 |
| 0.1 | 3,165 |

Level 3 with bias 0.03 found **4,054**, exactly Hedt's score, in three of
four jobs within four top-level iterations (40,000 rollouts each). It was
given only his final board, not his order of play. The beam search stops
at 4,049 on the same board. Since three independent searches agree with
his own, 4,054 looks optimal or nearly so for his board, and beating it
probably needs a new board rather than a better order.

A position offers NRPA the same short list as the beam: the best
`--per-size` plays of each size by the beam's ranking key, with `--bias`
weighing that key. That keeps rollouts fast when no target board fences
the search, and it also helps on a target board. At level 2 (bias 0.02,
best of 4 jobs) the default 8 per size reached 4,052, against 3,996 with
32 per size and 4,011 with every play. Each new best game is written to
`--out` as soon as it is found.

### Published records

The published records, with their word lists, methods and sources, are in
[RECORDS.md](RECORDS.md). The notes below are about how they were made.

The record plays are all OXYPHENBUTAZONE across an edge with seven hooks.
Dan Stock's: (54 x 27 = 1,458) + 50 + OPACIFYING 63 + BRAINWASHING 63 +
EJACULATING 63 + ZARFS 27 + PREINTERVIEWED 25 + AMELIORATIVE 17 + YELKS 12.

Computers have been used on single plays (Kyle Cronan's exhaustive program,
Lucassen's SAT solver) and once on whole games: Al Zimmermann's 2014
programming contest "Alphabet City Light" asked for the highest combined
score. Hedt won it with 4,037 (4,032 from 99 tiles) and later reached
4,054 with all 100 tiles; the runner-up scored 4,026 with TWL words. The
contest's final report lists his 4,054 move by move, and DejMar's 4,044
listing is the same game without its next-to-last play. The other game records were built and then tuned by hand: the
Facebook record went 3,953, 3,962, 3,966, 3,969, 3,983, 3,986, 3,995
between 2009 and 2012.

`results/hedt-4044.txt` is Hedt's game as DejMar transcribed it in the
Scrabulizer thread. It replays under `bingo check --players 1` with both
our SOWPODS list and CSW24: 4,044 points from 99 tiles, every score as
published (UNPROGRESSIVELY 302, MICROTECHNIQUES 1,325, OXYPHENBUTAZONE
1,586). It is a one-rack game. As a two-player game its order is
impossible: when OXYPHENBUTAZONE is played only 12 tiles are unplayed, so
the bag is empty and whoever plays it holds exactly those 7 tiles and goes
out, yet OB has to follow it.

Where that leaves us:
- **Solitaire.** Our own design reaches 3,864, which is 180 short of Hedt's
  4,044. Our order of his board scores 4,049 with all 100 tiles: above his
  published game, but 5 below the 4,054 that his own order and the last O
  give.
- **Two players.** Our 3,844 beats the 3,674 SOWPODS game. The Facebook
  game (4,012 together, with TWL06 words, nearly all of which are in
  Collins) is still 168 points ahead.
- **Single play.** Our best, 1,619, is about 170 points short of the
  Collins records. The gap is hooks.

What limits the search now:

- **Hooks.** Tails let us name hooks, but choosing them is still by hand.
  In the 3,816 game the edge finishes carry 35 and 12 points of hooks; in
  Hedt's game 114 (MIFF, CLAWLIKE, NE, QAT, UG) and 78 (JAMB, DITZ, DE). His
  hooks also need connector words (ALIENAGE, IMMANTLED) that a free beam
  rarely finds. The record single plays carry 270 or more, from seven long
  hooks packed into one dense block, which needs a constraint solver to
  design.
- **Timing of the spine.** Our searches play the x9 word at turns 16 to 25;
  the record games play it by turn 4, and their first play is already a
  hook for an edge word (Hedt's LAWLIKE, later CLAWLIKE).
- **The rest of the game.** Outside the three big plays our games score
  about 750 points, Hedt's 831.
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

- **Design the hooks.** Tails and targets can now carry a design to the
  beam. What is missing is choosing it: for each finish square, a hook
  word and a connector word across row 1 (or 13) that satisfies every
  column at once. This is a small crossword-construction problem (a SAT or
  constraint search over the squares beside the lane, as Lucassen did for
  one play).
- **Run `improve` and `nrpa` for hours.** On 12 cores, `--jobs 12` runs 12
  neighbourhoods per round or 12 independent NRPA searches. Wider
  neighbourhoods (`--free-plays 6 --radius 3`) can change the shape of the
  board. NRPA at level 3 or 4 with plans but no target board searches new
  boards, not just new orders.
- **Exact ordering of a fixed board.** The best order of play for a given
  final board is a longest-path problem over the subsets of its tiles. With
  a good upper bound, branch and bound could prove whether 4,054 is optimal
  for Hedt's board.
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
