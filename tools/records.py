#!/usr/bin/env python3
"""The records table: records.csv holds the data, RECORDS.md is generated from it.

    python3 tools/records.py                  regenerate RECORDS.md
    python3 tools/records.py add GAME...      replay each saved game with `bingo check`,
                                              log it in records.csv, regenerate RECORDS.md

Options for add:
    --play            log the game's highest single play instead of the game
    --note TEXT       a note for the row
    --machine TEXT    describe the machine (default: thread count and CPU model)

The score, tiles and player count come from the replay, and the search settings,
word list and search time from the header comment that `bingo search` and
`bingo beam` write. Adding a file that is already logged replaces its row.
"""
import csv
import datetime
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "records.csv")
MD = os.path.join(ROOT, "RECORDS.md")
OURS = "bingo-bot"

FIELDS = ["kind", "players", "score", "tiles", "words", "family", "who", "date", "method",
          "moves", "source", "url", "file", "notes", "settings", "seconds", "machine", "commit"]

# (kind, players, heading, what it means)
SECTIONS = [
    ("game", "1", "Whole games, one player",
     "One player plays every rack (solitaire) and chooses every draw."),
    ("game", "2", "Whole games, two players",
     "Two players take turns and cooperate; the score is both totals together. "
     "Draws are chosen unless noted."),
    ("play", "", "Single plays",
     "One move, on a board built for it."),
]
FAMILIES = [
    ("Collins", "Collins words (SOWPODS, CSW)"),
    ("North American", "North American words (OSPD, OWL, TWL, NWL)"),
    ("General", "General dictionaries"),
]


def load():
    with open(CSV, newline="") as f:
        return list(csv.DictReader(f))


def save(rows):
    order = {(k, p): i for i, (k, p, _, _) in enumerate(SECTIONS)}
    fam = {f: i for i, (f, _) in enumerate(FAMILIES)}
    rows.sort(key=lambda r: (order[(r["kind"], r["players"])], fam[r["family"]], -int(r["score"])))
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def family_of(words):
    w = words.lower()
    if any(x in w for x in ("sowpods", "csw", "collins")):
        return "Collins"
    if any(x in w for x in ("twl", "owl", "nwl", "ospd", "naspa")):
        return "North American"
    return "General"


def plans_of(settings):
    return " + ".join(re.findall(r"(?:row|col)\d+:([A-Za-z]+):", settings))


def search_of(settings):
    s = re.sub(r"^search:\s*", "", settings)
    s = re.sub(r",?\s*(?:plans?\s+)?(?:and\s+)?(?:row|col)\d+:[A-Za-z]+:[\d,]+", "", s)
    return s.strip(" ,")


def fmt(n):
    return f"{int(n):,}"


def who(r):
    return f"**{r['who']}**" if r["who"] == OURS else r["who"]


def moves(r):
    if r["file"]:
        return f"[replays]({r['file']})"
    return {"published": "published", "none": "not published"}.get(r["moves"], r["moves"])


def source(r):
    if r["who"] == OURS:
        return "[this repo](#our-searches)"
    return f"[{r['source']}]({r['url']})" if r["url"] else r["source"]


def kind_label(r):
    return "single play" if r["kind"] == "play" else ("one player" if r["players"] == "1" else "two players")


def summary(rows):
    out = ["| | " + " | ".join(label for _, label in FAMILIES) + " |",
           "|---|" + "---|" * len(FAMILIES)]
    for kind, players, heading, _ in SECTIONS:
        cells = []
        for fam, _ in FAMILIES:
            sel = [r for r in rows if r["kind"] == kind and r["players"] == players and r["family"] == fam]
            theirs = max((r for r in sel if r["who"] != OURS and r["moves"] != "none"),
                         key=lambda r: int(r["score"]), default=None)
            ours = max((r for r in sel if r["who"] == OURS), key=lambda r: int(r["score"]), default=None)
            parts = []
            if theirs:
                parts.append(f"{fmt(theirs['score'])} ({theirs['who']}, {theirs['date'][:4]})")
            if ours:
                beats = theirs is None or int(ours["score"]) > int(theirs["score"])
                parts.append(f"ours **{fmt(ours['score'])}**" if beats else f"ours {fmt(ours['score'])}")
            cells.append("<br>".join(parts) or "none known")
        out.append(f"| {heading.replace('Whole games, ', '').capitalize()} | " + " | ".join(cells) + " |")
    return out


def render(rows):
    today = datetime.date.today().isoformat()
    lines = [
        "# Records",
        "",
        "The highest-scoring Scrabble games and single plays we know of, by type and",
        "word list, with this repo's best (bingo-bot) beside them. Every game marked",
        "*replays* is in `results/` and passes `bingo check` under the full rules.",
        "",
        f"Generated from `records.csv` by `tools/records.py` on {today}. To log a new",
        "game, run `python3 tools/records.py add results/GAME.txt`; to correct a",
        "published record, edit `records.csv` and run `python3 tools/records.py`.",
        "",
        "## Best known",
        "",
        "The best published result with moves, and ours.",
        "",
    ]
    lines += summary(rows)
    lines.append("")
    for kind, players, heading, about in SECTIONS:
        lines += [f"## {heading}", "", about, ""]
        for fam, label in FAMILIES:
            sel = [r for r in rows if r["kind"] == kind and r["players"] == players and r["family"] == fam]
            # Only our best goes in these tables; the log below keeps every run.
            ours = [r for r in sel if r["who"] == OURS]
            best = max(ours, key=lambda r: int(r["score"])) if ours else None
            sel = [r for r in sel if r["who"] != OURS or r is best]
            if not sel:
                continue
            sel.sort(key=lambda r: -int(r["score"]))
            lines += [f"### {label}", ""]
            if kind == "game":
                lines += ["| Score | Tiles | Word list | By | Year | Method | Moves | Source | Notes |",
                          "|---:|---:|---|---|---|---|---|---|---|"]
                for r in sel:
                    lines.append(f"| {fmt(r['score'])} | {r['tiles']} | {r['words']} | {who(r)} | "
                                 f"{r['date'][:4]} | {r['method']} | {moves(r)} | {source(r)} | {r['notes']} |")
            else:
                lines += ["| Score | Word list | By | Year | Method | Moves | Source | Notes |",
                          "|---:|---|---|---|---|---|---|---|"]
                for r in sel:
                    lines.append(f"| {fmt(r['score'])} | {r['words']} | {who(r)} | {r['date'][:4]} | "
                                 f"{r['method']} | {moves(r)} | {source(r)} | {r['notes']} |")
            lines.append("")
    ours = sorted((r for r in rows if r["who"] == OURS),
                  key=lambda r: (r["date"], int(r["score"])), reverse=True)
    lines += ["## Our searches", "",
              "Every result saved in `results/`, with the search that found it. Time is wall",
              "clock for the beam that found the game; a search runs many beams.", "",
              "| Date | Score | Type | Word list | Plans | Search settings | Time | Machine | Commit | Game | Notes |",
              "|---|---:|---|---|---|---|---:|---|---|---|---|"]
    for r in ours:
        t = ""
        if r["seconds"]:
            s = float(r["seconds"])
            t = f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min"
        lines.append(f"| {r['date']} | {fmt(r['score'])} | {kind_label(r)} | {r['words']} | "
                     f"{plans_of(r['settings'])} | {search_of(r['settings'])} | {t} | {r['machine']} | "
                     f"{r['commit']} | [{os.path.basename(r['file'])}]({r['file']}) | {r['notes']} |")
    lines.append("")
    with open(MD, "w") as f:
        f.write("\n".join(lines))


def machine():
    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1]
                    break
    except OSError:
        pass
    model = re.sub(r"\((R|TM)\)|\b(CPU|Processor)\b", "", model)
    model = " ".join(model.split())
    return f"{os.cpu_count()} threads, {model}" if model else f"{os.cpu_count()} threads"


def add(path, play=False, note="", machine_text=None):
    text = open(path).read()
    header = " ".join(l[1:].strip() for l in text.splitlines() if l.startswith("#"))
    words = re.search(r"\bwords (\S+?)(?:;|\s|$)", header)
    words = words.group(1) if words else "sowpods.txt"
    seconds = re.search(r"\bsearch ([\d.]+) s\b", header)
    settings = re.match(r"(.*?):\s*[\d,]+ points", header)
    settings = settings.group(1) if settings else ""

    bingo = os.path.join(ROOT, "zig-out", "bin", "bingo")
    cache = os.path.splitext(words)[0] + ".gaddag"
    out = subprocess.run([bingo, "check", "--game", os.path.abspath(path), "--words", words,
                          "--cache", cache], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"bingo check failed for {path}:\n{out.stderr}")
    lines = out.stdout.splitlines()
    players = "1" if lines[0].startswith("1 player") else "2"
    last = lines[-1]
    total = int(re.search(r"(?:score|=) (\d+)", last).group(1))
    racks = re.search(r"racks (\d+)/(\d+)", last)
    tiles = 100 - int(racks.group(1)) - int(racks.group(2))
    best, word = 0, ""
    for l in lines:
        m = re.match(r"\s*\d+\s+p\d\s+\S+\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(?:across|down)\s+(\S+)", l)
        if m and int(m.group(1)) > best:
            best, word = int(m.group(1)), m.group(2).upper()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    label = os.path.splitext(words)[0]
    label = "SOWPODS" if label == "sowpods" else label.upper()
    game = "one-player" if players == "1" else "two-player"
    if play:
        note = "; ".join(x for x in (f"{word}, in a {total:,}-point {game} game", note) if x)
    row = {
        "kind": "play" if play else "game",
        "players": "" if play else players,
        "score": str(best if play else total),
        "tiles": "" if play else str(tiles),
        "words": label,
        "family": family_of(label),
        "who": OURS,
        "date": datetime.date.today().isoformat(),
        "method": "beam search with lane plans",
        "moves": "replays",
        "file": os.path.relpath(os.path.abspath(path), ROOT),
        "notes": note,
        "settings": settings,
        "seconds": seconds.group(1) if seconds else "",
        "machine": machine_text or machine(),
        "commit": commit,
    }
    rows = [r for r in load() if not (r["file"] == row["file"] and r["kind"] == row["kind"])]
    rows.append(row)
    save(rows)
    print(f"logged {kind_label(row)} {int(row['score']):,} ({label}) from {row['file']}")
    if row["file"].startswith(".."):
        print(f"note: {path} is outside the repo; keep games in results/ so the links work")


def main(args):
    if args[:1] == ["add"]:
        args = args[1:]
        opts = {"--note": "", "--machine": None}
        for k in opts:
            if k in args:
                i = args.index(k)
                opts[k] = args[i + 1]
                del args[i:i + 2]
        play = "--play" in args
        files = [a for a in args if not a.startswith("--")]
        if not files:
            sys.exit(__doc__)
        for f in files:
            add(f, play=play, note=opts["--note"], machine_text=opts["--machine"])
    elif args:
        sys.exit(__doc__)
    render(load())
    print(f"wrote {os.path.relpath(MD)}")


if __name__ == "__main__":
    main(sys.argv[1:])
