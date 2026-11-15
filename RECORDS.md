# Records

The highest-scoring Scrabble games and single plays we know of, by type and
word list, with this repo's best (bingo-bot) beside them. Every game marked
*replays* is in `results/` and passes `bingo check` under the full rules.

Generated from `records.csv` by `tools/records.py` on 2026-11-15. To log a new
game, run `python3 tools/records.py add results/GAME.txt`; to correct a
published record, edit `records.csv` and run `python3 tools/records.py`.

## Best known

The best published result with moves, and ours.

| | Collins words (SOWPODS, CSW) | North American words (OSPD, OWL, TWL, NWL) | General dictionaries |
|---|---|---|---|
| One player | 4,054 (Nathan Hedt, 2014)<br>ours 4,054 | none known | 4,470 (Jeff Grant, 2009) |
| Two players | 3,674 (Clare Sudbery, 2002)<br>ours **3,844** | 4,012 (RyanM561 and TenzingHillary, 2012) | 4,207 (Jeff Grant, 1979) |
| Single plays | 1,787 (DejMar, 2016)<br>ours 1,619 | 1,786 (Bob Lucassen, 2024) | 2,044 (Arian Smit, 2008) |

## Whole games, one player

One player plays every rack (solitaire) and chooses every draw.

### Collins words (SOWPODS, CSW)

| Score | Tiles | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---:|---|---|---|---|---|---|---|
| 4,054 | 100 | CSW12 | Nathan Hedt | 2014 | program | [replays](results/hedt-4054.txt) | [AZsPCS final report](https://azspcs.com/Contest/AlphabetCityLight/FinalReport) | the published solution of Al Zimmermann's contest Alphabet City Light, all 100 tiles |
| 4,054 | 100 | SOWPODS | **bingo-bot** | 2026 | NRPA over Hedt's final board as targets | [replays](results/solitaire-4054-nrpa-hedt-board.txt) | [this repo](#our-searches) | Hedt's board in a different order from his; ties his 4,054 |
| 4,044 | 99 | CSW12 | Nathan Hedt | 2014 | program | [replays](results/hedt-4044.txt) | [Scrabulizer comments](https://www.scrabulizer.com/blog/post/3) | DejMar's listing of the same game without its next-to-last play (KO), so one O is left over |

### General dictionaries

| Score | Tiles | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---:|---|---|---|---|---|---|---|
| 4,470 | 100 | Webster's 2nd and 3rd, the OED | Jeff Grant | 2009 | by hand | published | [Word Ways](https://digitalcommons.butler.edu/wordways/vol42/iss4/11) | revises his 4,454 |
| 4,454 | 100 | Webster's 2nd and 3rd, the OED | Jeff Grant | 1981 | by hand | published | [Word Ways](https://digitalcommons.butler.edu/wordways/vol14/iss1/3) | FREQUENTATIVELY, JACKPUDDINGHOOD, BENZOXYCAMPHORS |

## Whole games, two players

Two players take turns and cooperate; the score is both totals together. Draws are chosen unless noted.

### Collins words (SOWPODS, CSW)

| Score | Tiles | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---:|---|---|---|---|---|---|---|
| 3,844 | 99 | SOWPODS | **bingo-bot** | 2026 | beam search with lane plans | [replays](results/two-player-3844.txt) | [this repo](#our-searches) |  |
| 3,674 |  | PixiePit list (British and North American) | Clare Sudbery | 2002 | by hand | published | [PixiePit Hall of Fame](https://www.thepixiepit.co.uk/scrabble/highscores2.html) | played out with random draws; 3,658 by one player |

### North American words (OSPD, OWL, TWL, NWL)

| Score | Tiles | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---:|---|---|---|---|---|---|---|
| 4,012 | 99 | TWL06 (Facebook Scrabble) | RyanM561 and TenzingHillary | 2012 | by hand | published | [Scrabulizer comments](https://www.scrabulizer.com/blog/post/3) | played out on Facebook; 3,995 by one player |
| 3,986 | 99 | OSPD and Webster's Tenth (all in TWL06) | Stephen Root and Nathan Hedt | 2003 | by hand | published | [Word Ways](https://digitalcommons.butler.edu/wordways/vol36/iss2/19) | MICROTECHNIQUES, OVERIDENTIFYING, OXYPHENBUTAZONE |

### General dictionaries

| Score | Tiles | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---:|---|---|---|---|---|---|---|
| 4,207 | 99 | general dictionaries | Jeff Grant | 1979 | by hand | published | [Word Ways](https://digitalcommons.butler.edu/wordways/vol12/iss3/3) | revises Beaman's 4,142 |
| 4,142 |  | general dictionaries | Ralph Beaman | 1974 | by hand | published | Word Ways, November 1974 | BENZOXYCAMPHORS |
| 4,047 |  | general dictionaries | Ron Jerome | 1974 | by hand | published | [Word Ways](https://digitalcommons.butler.edu/wordways/vol7/iss3/11) | BENZOXYCAMPHORS |
| 3,108 |  | general dictionaries | Darryl Francis | 1972 | by hand | published | Word Ways, 1972 |  |

## Single plays

One move, on a board built for it.

### Collins words (SOWPODS, CSW)

| Score | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---|---|---|---|---|---|---|
| 1,791 | SOWPODS | Ben Woo | 2011 | by hand | not published | [Scrabulizer comments](https://www.scrabulizer.com/blog/post/3) | word scores only, and they do not add up: OECUMENICAL through a triple-word square is worth 51, not 66 |
| 1,787 | CSW15 | DejMar | 2016 | by hand | published | [Scrabulizer comments](https://www.scrabulizer.com/blog/post/3) |  |
| 1,785 | SOWPODS | Arian Smit | 2008 | by hand | published | [Scrabulizer](https://www.scrabulizer.com/blog/post/3) | several boards reach it |
| 1,619 | SOWPODS | **bingo-bot** | 2026 | beam search with lane plans | [replays](results/two-player-play-1619.txt) | [this repo](#our-searches) | OXYPHENBUTAZONE, in a 3,594-point two-player game |

### North American words (OSPD, OWL, TWL, NWL)

| Score | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---|---|---|---|---|---|---|
| 1,786 | NWL23 | Bob Lucassen | 2024 | SAT solver | published | [Woogles blog](https://blog.woogles.io/posts/2024-05-27-new-theoretical-highest-play-discovered) | proved best for NWL23; needs OPACIFICATIONS, not in Collins |
| 1,784 | NASPA list | Jodi and Eric |  | by hand | published | NASPA wiki |  |
| 1,783 | TWL06 | Kyle Cronan | 2011 | exhaustive program | published | [Scrabulizer comments](https://www.scrabulizer.com/blog/post/3) | found nothing higher for TWL06 |
| 1,780 | TWL06 | Arian Smit | 2008 | by hand | published | [Scrabulizer](https://www.scrabulizer.com/blog/post/3) |  |
| 1,778 | OWL and the Long Words List | Dan Stock |  | by hand | published | rec.games.board.scrabble FAQ | seven hooks under OXYPHENBUTAZONE; every word is also in SOWPODS |

### General dictionaries

| Score | Word list | By | Year | Method | Moves | Source | Notes |
|---:|---|---|---|---|---|---|---|
| 2,044 | SOWPODS plus SESQUIOXIDIZING | Arian Smit | 2008 | by hand | published | [Scrabulizer](https://www.scrabulizer.com/blog/post/3) | SESQUIOXIDIZING is in no official list |
| 2,037 | general dictionaries | Josefa Heifetz | 1974 | by hand | published | [Word Ways](https://digitalcommons.butler.edu/wordways/vol7/iss1/2) | SESQUIOXIDIZING |

## Our searches

Every result saved in `results/`, with the search that found it. Time is wall
clock for the beam that found the game; a search runs many beams.

| Date | Score | Type | Word list | Plans | Search settings | Time | Machine | Commit | Game | Notes |
|---|---:|---|---|---|---|---:|---|---|---|---|
| 2026-11-15 | 4,054 | one player | SOWPODS | UNPROGRESSIVELY + MICROTECHNIQUES + OXYPHENBUTAZONE | nrpa level 3, 100 iterations, alpha 1, bias 0.02, 8 per size, 4 jobs, goal bonus 0, hooks prefix, seed 5, mid bonus 0, target game hedt-4054.txt, target bonus 30 | 69 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | f43de69 | [solitaire-4054-nrpa-hedt-board.txt](results/solitaire-4054-nrpa-hedt-board.txt) | Hedt's board in a different order from his; ties his 4,054 |
| 2026-11-15 | 4,049 | one player | SOWPODS | UNPROGRESSIVELY + MICROTECHNIQUES + OXYPHENBUTAZONE | beam width 512, 8 per size, blank keep 0, goal bonus 0, hooks prefix, seed 1, mid bonus 0, drop penalty 2000, target game hedt-4044.txt, target bonus 30 | 2 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | 3345cae | [solitaire-4049-hedt-board.txt](results/solitaire-4049-hedt-board.txt) | Hedt's board in our order of play; his own order scores 4,054 |
| 2026-11-15 | 3,864 | one player | SOWPODS | MICROTECHNIQUES + UNPROGRESSIVELY + OXYPHENBUTAZONE | beam width 128, 8 per size, blank keep 0, goal bonus 200, hooks prefix, seed 8, per signature 4, letter cost 27, mid bonus 20 | 2 min | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | 3345cae | [solitaire-3864.txt](results/solitaire-3864.txt) |  |
| 2026-11-15 | 3,844 | two players | SOWPODS | MICROTECHNIQUES + UNPROGRESSIVELY + OXYPHENBUTAZONE | beam width 64, 8 per size, blank keep 0, goal bonus 200, hooks prefix, seed 5, per signature 4, letter cost 27, mid bonus 20 | 87 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | 3345cae | [two-player-3844.txt](results/two-player-3844.txt) |  |
| 2026-11-15 | 3,828 | one player | SOWPODS | MICROTECHNIQUES + UNPROGRESSIVELY + OXYPHENBUTAZONE | beam width 128, 8 per size, blank keep 0, goal bonus 200, hooks strict, seed 1, per signature 4, letter cost 27, mid bonus 20 | 90 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | d8e58bc | [solitaire-3828.txt](results/solitaire-3828.txt) |  |
| 2026-11-15 | 3,816 | two players | SOWPODS | MICROTECHNIQUES + UNPROGRESSIVELY + OXYPHENBUTAZONE | beam width 64, 8 per size, blank keep 0, goal bonus 200, hooks strict, seed 4, per signature 4, letter cost 27, mid bonus 20 | 45 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | d8e58bc | [two-player-3816.txt](results/two-player-3816.txt) |  |
| 2026-11-15 | 3,693 | one player | SOWPODS | OXYPHENBUTAZONE + SESQUISULPHIDES | beam width 128, 8 per size, blank keep 0, hooks strict, seed 2, letter cost 27 | 47 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | ca8352b | [solitaire-3693.txt](results/solitaire-3693.txt) |  |
| 2026-11-15 | 3,635 | two players | SOWPODS | OXYPHENBUTAZONE + SESQUISULPHIDES | beam width 256, 8 per size, blank keep 0, hooks strict, seed 2, letter cost 27 | 86 s | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | ca8352b | [two-player-3635.txt](results/two-player-3635.txt) |  |
| 2026-11-15 | 1,619 | single play | SOWPODS | OXYPHENBUTAZONE + SESQUISULPHIDES | beam width 64, 8 per size, blank keep 0, hooks strict, seed 5 |  | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | ca8352b | [two-player-play-1619.txt](results/two-player-play-1619.txt) | OXYPHENBUTAZONE, in a 3,594-point two-player game |
| 2026-11-14 | 3,336 | one player | SOWPODS | MATHEMATICIZING + XYLOTYPOGRAPHIC | beam width 64, goal bonus 200, seed 1 |  | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | dffa628 | [solitaire-3336.txt](results/solitaire-3336.txt) | plan search over 16 pairs, 2 seeds each, about 7 min in all; since a later move generator fix these settings give 2,340 |
| 2026-11-14 | 3,251 | two players | SOWPODS | OVEREMPHASIZING + CYCLOHEXYLAMINE | beam width 64, goal bonus 200, seed 2 |  | 4 threads, Intel Xeon @ 2.10GHz (cloud container) | dffa628 | [two-player-3251.txt](results/two-player-3251.txt) | plan search over 16 pairs, 2 seeds each, about 7 min in all |
