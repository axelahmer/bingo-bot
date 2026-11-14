const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const csp = @import("csp.zig");
const game = @import("game.zig");
const search = @import("search.zig");

const Lexicon = lexicon.Lexicon;
const Move = board.Move;
const Position = board.Position;
const Game = game.Game;
const Turn = game.Turn;

const usage =
    \\usage: bingo <command> [options]
    \\
    \\commands:
    \\  rollout   play games with the 2023 annealed-softmax policy
    \\  check     replay a game under the full rules with real racks and draws
    \\  show      print the position after a game's turns
    \\  moves     list every legal play for the player to move
    \\  bench     time move generation at each turn of a game
    \\  verify    check the GADDAG generator against the CSP port on random games
    \\  lexicon   build (or load) the GADDAG and report its size
    \\
    \\rules (the defaults are a standard two-player game):
    \\  --players N     1 (solitaire) or 2
    \\  --blanks N      blanks in the bag (default 2; solver.py used 0)
    \\  --end R         end-of-game adjustment: wespa (default) or naspa
    \\
    \\options:
    \\  --words PATH    word list (default sowpods.txt)
    \\  --cache PATH    GADDAG cache file (default sowpods.gaddag, "none" to skip)
    \\  --game PATH     a game: one turn per line, "ROW COL across|down WORD" or "pass"
    \\  --turns K       use only the first K turns of --game
    \\  --blank-use U   any (default) or needed: generate blank plays only where
    \\                  no natural tile is left
    \\  --engine E      gaddag (default) or csp, for moves and bench
    \\  --games N       games for rollout and verify
    \\  --threads N     rollout worker threads (default: all cores)
    \\  --seed S        random seed
    \\  --reps N        timing repetitions per position for bench
    \\  --out PATH      write the best rollout game to PATH
    \\  --count         moves: print only the count and timing
    \\
;

const Options = struct {
    command: []const u8 = "",
    words: []const u8 = "sowpods.txt",
    cache: []const u8 = "sowpods.gaddag",
    game: ?[]const u8 = null,
    turns: ?usize = null,
    rules: rules.Rules = .{},
    blank_use: movegen.BlankMode = .any,
    engine: []const u8 = "gaddag",
    games: usize = 0,
    threads: usize = 0,
    seed: u64 = 1,
    reps: usize = 5,
    out: ?[]const u8 = null,
    count_only: bool = false,
};

fn parseArgs(args: []const [:0]const u8) !Options {
    var o = Options{};
    if (args.len < 2) return error.Usage;
    o.command = args[1];
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--count")) {
            o.count_only = true;
            continue;
        }
        if (i + 1 >= args.len) return error.Usage;
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, a, "--words")) {
            o.words = v;
        } else if (std.mem.eql(u8, a, "--cache")) {
            o.cache = v;
        } else if (std.mem.eql(u8, a, "--game")) {
            o.game = v;
        } else if (std.mem.eql(u8, a, "--turns")) {
            o.turns = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--players")) {
            o.rules.players = try std.fmt.parseInt(u8, v, 10);
            if (o.rules.players != 1 and o.rules.players != 2) return error.Usage;
        } else if (std.mem.eql(u8, a, "--blanks")) {
            o.rules.blanks = try std.fmt.parseInt(u8, v, 10);
        } else if (std.mem.eql(u8, a, "--end")) {
            o.rules.end_rule = std.meta.stringToEnum(rules.EndRule, v) orelse return error.Usage;
        } else if (std.mem.eql(u8, a, "--blank-use")) {
            o.blank_use = if (std.mem.eql(u8, v, "any")) .any else if (std.mem.eql(u8, v, "needed")) .as_needed else return error.Usage;
        } else if (std.mem.eql(u8, a, "--engine")) {
            o.engine = v;
        } else if (std.mem.eql(u8, a, "--games")) {
            o.games = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--threads")) {
            o.threads = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--seed")) {
            o.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, a, "--reps")) {
            o.reps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--out")) {
            o.out = v;
        } else {
            std.debug.print("unknown option {s}\n", .{a});
            return error.Usage;
        }
    }
    return o;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const opts = parseArgs(args) catch {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    };

    var out_buf: [1 << 16]u8 = undefined;
    var out_file: Io.File.Writer = .init(.stdout(), io, &out_buf);
    const out = &out_file.interface;
    defer out.flush() catch {};

    const cache: ?[]const u8 = if (std.mem.eql(u8, opts.cache, "none")) null else opts.cache;
    const t0 = Io.Timestamp.now(io, .awake);
    var lex = try Lexicon.loadOrBuild(gpa, io, opts.words, cache);
    defer lex.deinit(gpa);
    const lex_secs = secondsSince(io, t0);

    const cmd = opts.command;
    if (std.mem.eql(u8, cmd, "lexicon")) {
        try out.print("words {d}, nodes {d} ({d:.1} MiB), ready in {d:.2} s\n", .{
            lex.word_count, lex.nodes.len, @as(f64, @floatFromInt(lex.nodes.len * 8)) / (1 << 20), lex_secs,
        });
    } else if (std.mem.eql(u8, cmd, "show")) {
        var turns: std.ArrayList(Turn) = .empty;
        defer turns.deinit(gpa);
        const g = try loadGame(gpa, io, &lex, opts, &turns);
        try g.pos.writeBoard(out);
        try writeSummary(out, &g);
    } else if (std.mem.eql(u8, cmd, "check")) {
        try cmdCheck(gpa, io, &lex, opts, out);
    } else if (std.mem.eql(u8, cmd, "moves")) {
        try cmdMoves(gpa, io, &lex, opts, out);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        try cmdBench(gpa, io, &lex, opts, out);
    } else if (std.mem.eql(u8, cmd, "rollout")) {
        try cmdRollout(gpa, io, &lex, opts, out);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        try cmdVerify(gpa, io, &lex, opts, out);
    } else {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
}

fn secondsSince(io: Io, t0: Io.Timestamp) f64 {
    const d = t0.untilNow(io, .awake);
    return @as(f64, @floatFromInt(d.toNanoseconds())) / 1e9;
}

fn supplyFor(g: *const Game, opts: Options) movegen.Supply {
    var s = g.supply();
    s.blanks = opts.blank_use;
    return s;
}

/// Replay `--game` (all of it, or its first `--turns` turns) from the start
/// under the full rules, appending each turn to `turns`.
fn loadGame(gpa: Allocator, io: Io, lex: *const Lexicon, opts: Options, turns: *std.ArrayList(Turn)) !Game {
    const r = opts.rules;
    var g = Game.init(r);
    const path = opts.game orelse return g;
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var line_no: usize = 0;
    while (lines.next()) |raw| {
        line_no += 1;
        if (opts.turns) |k| if (turns.items.len >= k) break;
        const line = std.mem.trim(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len == 0) continue;
        if (g.over()) {
            std.debug.print("{s}:{d}: the game is already over\n", .{ path, line_no });
            return error.TurnAfterEnd;
        }
        if (std.mem.eql(u8, line, "pass")) {
            g.pass();
            try turns.append(gpa, .pass);
            continue;
        }
        var tok = std.mem.tokenizeAny(u8, line, " \t");
        const row = std.fmt.parseInt(usize, tok.next() orelse return badLine(path, line_no), 10) catch return badLine(path, line_no);
        const col = std.fmt.parseInt(usize, tok.next() orelse return badLine(path, line_no), 10) catch return badLine(path, line_no);
        const dir_s = tok.next() orelse return badLine(path, line_no);
        const word = tok.next() orelse return badLine(path, line_no);
        const d: u1 = if (std.mem.eql(u8, dir_s, "across")) 0 else if (std.mem.eql(u8, dir_s, "down")) 1 else return badLine(path, line_no);
        const m = g.pos.parsePlay(lex, r, row, col, d, word) catch |err| {
            std.debug.print("{s}:{d}: illegal play {s}: {t}\n", .{ path, line_no, line, err });
            return err;
        };
        if (m.placed > g.rack[g.turn]) {
            std.debug.print("{s}:{d}: {s} needs {d} tiles but the rack holds {d}\n", .{ path, line_no, line, m.placed, g.rack[g.turn] });
            return error.RackTooSmall;
        }
        g.play(lex, &m);
        try turns.append(gpa, .{ .play = m });
    }
    return g;
}

fn badLine(path: []const u8, line_no: usize) error{BadGameFile} {
    std.debug.print("{s}:{d}: expected ROW COL across|down WORD, or pass\n", .{ path, line_no });
    return error.BadGameFile;
}

/// Write turns in the game-file format, replaying them from `start`.
fn writeTurns(w: *Io.Writer, lex: *const Lexicon, start: *const Game, turns: []const Turn, with_scores: bool) !void {
    var g = start.*;
    for (turns) |*t| {
        if (with_scores) {
            const s: i32 = switch (t.*) {
                .play => |m| m.score,
                .pass => 0,
            };
            try w.print("{d:>4}  p{d}  ", .{ @as(u32, @intCast(s)), @as(u8, g.turn) + 1 });
        }
        switch (t.*) {
            .play => |*m| {
                try g.pos.writePlay(m, w);
                g.play(lex, m);
            },
            .pass => {
                try w.writeAll("pass");
                g.pass();
            },
        }
        try w.writeByte('\n');
    }
}

fn writeSummary(w: *Io.Writer, g: *const Game) !void {
    if (g.rules.players == 1) {
        try w.print("score {d}", .{g.score[0]});
    } else {
        try w.print("scores {d} + {d} = {d}", .{ g.score[0], g.score[1], g.total() });
    }
    try w.print("; bag {d}, racks {d}/{d}; ", .{ g.bag, g.rack[0], g.rack[1] });
    switch (g.end) {
        .none => try w.print("player {d} to move\n", .{@as(u8, g.turn) + 1}),
        .went_out => try w.print("over: player {d} went out (adjustment {d}, {d})\n", .{ @as(u8, g.turn) + 1, g.adjust[0], g.adjust[1] }),
        .scoreless => try w.print("over: {d} scoreless turns (adjustment {d}, {d})\n", .{ g.rules.scoreless_limit, g.adjust[0], g.adjust[1] }),
    }
}

fn writeRack(w: *Io.Writer, counts: *const [27]u8) !void {
    var n: usize = 0;
    for (0..26) |l| {
        for (0..counts[l]) |_| {
            try w.writeByte(@as(u8, @intCast(l)) + 'A');
            n += 1;
        }
    }
    for (0..counts[rules.BLANK_SLOT]) |_| {
        try w.writeByte('?');
        n += 1;
    }
    for (n..7) |_| try w.writeByte(' ');
}

const Keyed = struct {
    key: [8]u16,
    score: i32,
    idx: u32,

    fn lessThan(_: void, a: Keyed, b: Keyed) bool {
        return switch (std.mem.order(u16, &a.key, &b.key)) {
            .lt => true,
            .gt => false,
            .eq => a.score < b.score,
        };
    }
};

/// Sort plays by placement and drop repeats. Repeats with different scores
/// mean a scoring bug, so they are an error.
fn canonicalize(gpa: Allocator, moves: []const Move) ![]Keyed {
    const keyed = try gpa.alloc(Keyed, moves.len);
    for (moves, 0..) |*m, i| keyed[i] = .{ .key = m.key(), .score = m.score, .idx = @intCast(i) };
    std.sort.pdq(Keyed, keyed, {}, Keyed.lessThan);
    var n: usize = 0;
    for (keyed) |k| {
        if (n > 0 and std.mem.eql(u16, &keyed[n - 1].key, &k.key)) {
            if (keyed[n - 1].score != k.score) {
                gpa.free(keyed);
                return error.InconsistentScores;
            }
            continue;
        }
        keyed[n] = k;
        n += 1;
    }
    return gpa.realloc(keyed, n);
}

/// Collect every play from one engine; returns the generation time in seconds.
fn allMoves(
    gpa: Allocator,
    io: Io,
    lex: *const Lexicon,
    words: ?*const csp.WordSets,
    g: *const Game,
    sup: movegen.Supply,
    use_csp: bool,
    collector: *movegen.Collector,
) !f64 {
    const t0 = Io.Timestamp.now(io, .awake);
    if (use_csp) {
        const halo = csp.computeHalo(words.?, &g.pos);
        const t1 = Io.Timestamp.now(io, .awake);
        try csp.generate(movegen.Collector, collector, gpa, words.?, &g.pos, &halo, g.rules, sup);
        if (collector.failed) return error.OutOfMemory;
        return secondsSince(io, t1);
    }
    movegen.generate(movegen.Collector, collector, &g.pos, lex, g.rules, sup);
    if (collector.failed) return error.OutOfMemory;
    return secondsSince(io, t0);
}

fn loadWordSets(gpa: Allocator, io: Io, path: []const u8) !csp.WordSets {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(text);
    return csp.WordSets.init(gpa, text);
}

fn cmdMoves(gpa: Allocator, io: Io, lex: *const Lexicon, opts: Options, out: *Io.Writer) !void {
    var turns: std.ArrayList(Turn) = .empty;
    defer turns.deinit(gpa);
    const g = try loadGame(gpa, io, lex, opts, &turns);
    const use_csp = std.mem.eql(u8, opts.engine, "csp");
    var words: ?csp.WordSets = if (use_csp) try loadWordSets(gpa, io, opts.words) else null;
    defer if (words) |*w| w.deinit(gpa);

    var col = movegen.Collector{ .gpa = gpa };
    defer col.deinit();
    const secs = try allMoves(gpa, io, lex, if (words) |*w| w else null, &g, supplyFor(&g, opts), use_csp, &col);
    const keyed = try canonicalize(gpa, col.moves.items);
    defer gpa.free(keyed);
    if (!opts.count_only) {
        for (keyed) |k| try col.moves.items[k.idx].writeCanonical(out);
    }
    std.debug.print("{s}: turns={d} moves={d} raw={d} seconds={d:.4}\n", .{
        opts.engine, turns.items.len, keyed.len, col.moves.items.len, secs,
    });
}

fn cmdBench(gpa: Allocator, io: Io, lex: *const Lexicon, opts: Options, out: *Io.Writer) !void {
    var turns: std.ArrayList(Turn) = .empty;
    defer turns.deinit(gpa);
    _ = try loadGame(gpa, io, lex, opts, &turns);
    const with_csp = std.mem.eql(u8, opts.engine, "csp");
    var words: ?csp.WordSets = if (with_csp) try loadWordSets(gpa, io, opts.words) else null;
    defer if (words) |*w| w.deinit(gpa);

    try out.print("{s:>4} {s:>9} {s:>12} {s:>12}{s}\n", .{
        "turn", "moves", "gaddag_ms", "Mmoves/s", if (with_csp) "       csp_ms  csp/gaddag" else "",
    });
    var g = Game.init(opts.rules);
    var total_secs: f64 = 0;
    var total_moves: u64 = 0;
    for (0..turns.items.len + 1) |k| {
        if (g.over()) break;
        const sup = supplyFor(&g, opts);
        var best: f64 = std.math.inf(f64);
        var counter = movegen.Counter{};
        for (0..@max(opts.reps, 1)) |_| {
            counter = .{};
            const t0 = Io.Timestamp.now(io, .awake);
            movegen.generate(movegen.Counter, &counter, &g.pos, lex, g.rules, sup);
            best = @min(best, secondsSince(io, t0));
        }
        total_secs += best;
        total_moves += counter.count;
        try out.print("{d:>4} {d:>9} {d:>12.3} {d:>12.2}", .{
            k, counter.count, best * 1e3, @as(f64, @floatFromInt(counter.count)) / best / 1e6,
        });
        if (words) |*w| {
            const halo = csp.computeHalo(w, &g.pos);
            var c2 = movegen.Counter{};
            const t0 = Io.Timestamp.now(io, .awake);
            try csp.generate(movegen.Counter, &c2, gpa, w, &g.pos, &halo, g.rules, sup);
            const s = secondsSince(io, t0);
            try out.print(" {d:>12.1} {d:>11.0}", .{ s * 1e3, s / best });
        }
        try out.writeByte('\n');
        try out.flush();
        if (k == turns.items.len) break;
        switch (turns.items[k]) {
            .play => |*m| g.play(lex, m),
            .pass => g.pass(),
        }
    }
    try out.print("total: {d} moves in {d:.3} ms ({d:.2} M moves/s)\n", .{
        total_moves, total_secs * 1e3, @as(f64, @floatFromInt(total_moves)) / total_secs / 1e6,
    });
}

fn cmdCheck(gpa: Allocator, io: Io, lex: *const Lexicon, opts: Options, out: *Io.Writer) !void {
    var turns: std.ArrayList(Turn) = .empty;
    defer turns.deinit(gpa);
    _ = try loadGame(gpa, io, lex, opts, &turns);
    const realized = try gpa.create(game.Realized);
    defer gpa.destroy(realized);
    const r = opts.rules;
    const g = try game.realize(lex, r, turns.items, realized);

    try out.print("{d} player{s}, {d} blanks, {s} end; {d} turns, every play taken from a real rack\n", .{
        r.players, if (r.players == 1) "" else "s", r.blanks, @tagName(r.end_rule), turns.items.len,
    });
    try out.writeAll("opening racks:");
    for (0..r.players) |p| {
        try out.print("  p{d} ", .{p + 1});
        try writeRack(out, &realized.opening[p]);
    }
    try out.writeAll("\n\nturn  who  rack     score  play                        draws\n");
    var replay = Game.init(r);
    for (turns.items, 0..) |*t, k| {
        try out.print("{d:>4}  p{d}   ", .{ k + 1, @as(u8, replay.turn) + 1 });
        try writeRack(out, &realized.rack_before[k]);
        var buf: [64]u8 = undefined;
        var w = Io.Writer.fixed(&buf);
        switch (t.*) {
            .play => |*m| {
                try replay.pos.writePlay(m, &w);
                try out.print("  {d:>5}  {s:<26}  ", .{ @as(u32, @intCast(m.score)), w.buffered() });
                replay.play(lex, m);
            },
            .pass => {
                try out.print("  {d:>5}  {s:<26}  ", .{ 0, "pass" });
                replay.pass();
            },
        }
        try writeRack(out, &realized.drawn[k]);
        try out.writeByte('\n');
    }
    try out.writeAll("\nfinal racks:");
    for (0..r.players) |p| {
        try out.print("  p{d} ", .{p + 1});
        try writeRack(out, &realized.final[p]);
    }
    try out.writeByte('\n');
    try writeSummary(out, &g);
}

const Stats = struct {
    games: u64 = 0,
    score_sum: f64 = 0,
    score_sq: f64 = 0,
    turns_sum: f64 = 0,
    left_sum: f64 = 0,
    considered: u64 = 0,
    /// Games that ended on scoreless turns with tiles unplayed.
    stuck: u64 = 0,
    min: i32 = std.math.maxInt(i32),
    best: search.Playout = .{ .game = undefined },

    fn add(s: *Stats, p: *const search.Playout) void {
        s.games += 1;
        const x: f64 = @floatFromInt(p.game.total());
        s.score_sum += x;
        s.score_sq += x * x;
        s.turns_sum += @floatFromInt(p.n);
        s.left_sum += @floatFromInt(p.game.pos.pool_total);
        s.considered += p.considered;
        s.stuck += @intFromBool(p.game.end == .scoreless);
        s.min = @min(s.min, p.game.total());
        if (s.games == 1 or p.game.total() > s.best.game.total()) s.best = p.*;
    }

    fn merge(s: *Stats, o: *const Stats) void {
        if (o.games == 0) return;
        if (s.games == 0 or o.best.game.total() > s.best.game.total()) s.best = o.best;
        s.games += o.games;
        s.score_sum += o.score_sum;
        s.score_sq += o.score_sq;
        s.turns_sum += o.turns_sum;
        s.left_sum += o.left_sum;
        s.considered += o.considered;
        s.stuck += o.stuck;
        s.min = @min(s.min, o.min);
    }
};

const Shared = struct {
    lex: *const Lexicon,
    policy: search.SoftmaxPolicy,
    start: Game,
    games: usize,
    next: std.atomic.Value(usize) = .init(0),
    seed: u64,
};

fn rolloutWorker(sh: *Shared, stats: *Stats) void {
    const playout = std.heap.page_allocator.create(search.Playout) catch return;
    defer std.heap.page_allocator.destroy(playout);
    var prng = std.Random.DefaultPrng.init(0);
    var sampler = search.SoftmaxSampler.init(std.heap.page_allocator, &prng, sh.start.rules) catch {
        std.debug.print("out of memory\n", .{});
        return;
    };
    defer sampler.deinit(std.heap.page_allocator);
    while (true) {
        const g = sh.next.fetchAdd(1, .monotonic);
        if (g >= sh.games) break;
        // One stream per game, so results do not depend on the thread count.
        prng = std.Random.DefaultPrng.init(sh.seed *% 0x9E3779B97F4A7C15 +% g);
        search.softmaxRollout(&sh.start, sh.lex, &sampler, sh.policy, playout);
        stats.add(playout);
    }
}

fn cmdRollout(gpa: Allocator, io: Io, lex: *const Lexicon, opts: Options, out: *Io.Writer) !void {
    var prefix: std.ArrayList(Turn) = .empty;
    defer prefix.deinit(gpa);
    var sh = Shared{
        .lex = lex,
        .policy = .{ .blank_use = opts.blank_use },
        .start = try loadGame(gpa, io, lex, opts, &prefix),
        .games = if (opts.games == 0) 8 else opts.games,
        .seed = opts.seed,
    };
    const nthreads = @max(1, if (opts.threads != 0) opts.threads else std.Thread.getCpuCount() catch 1);
    const stats = try gpa.alloc(Stats, nthreads);
    defer gpa.free(stats);
    for (stats) |*s| s.* = .{};
    const threads = try gpa.alloc(std.Thread, nthreads);
    defer gpa.free(threads);

    const t0 = Io.Timestamp.now(io, .awake);
    for (threads, stats) |*t, *s| t.* = try std.Thread.spawn(.{}, rolloutWorker, .{ &sh, s });
    for (threads) |t| t.join();
    const secs = secondsSince(io, t0);

    const all = try gpa.create(Stats);
    defer gpa.destroy(all);
    all.* = .{};
    for (stats) |*s| all.merge(s);
    const n: f64 = @floatFromInt(all.games);
    const mean = all.score_sum / n;
    const sd = @sqrt(@max(0, all.score_sq / n - mean * mean));
    try out.print(
        \\games {d} on {d} threads in {d:.2} s ({d:.2} games/s, {d:.1} M plays generated/s)
        \\total score mean {d:.1} sd {d:.1} min {d} max {d}; turns {d:.1}
        \\ended on scoreless turns: {d} games ({d:.1}%); {d:.2} tiles unplayed per game on average
        \\
    , .{
        all.games,         nthreads,                                             secs,
        n / secs,          @as(f64, @floatFromInt(all.considered)) / secs / 1e6, mean,
        sd,                all.min,                                              all.best.game.total(),
        all.turns_sum / n, all.stuck,                                            @as(f64, @floatFromInt(all.stuck)) * 100 / n,
        all.left_sum / n,
    });

    const best = &all.best;
    try writeTurns(out, lex, &sh.start, best.turns[0..best.n], true);
    try best.game.pos.writeBoard(out);
    try writeSummary(out, &best.game);
    if (opts.out) |p| {
        var file = try Io.Dir.cwd().createFile(io, p, .{});
        defer file.close(io);
        var buf: [4096]u8 = undefined;
        var fw = file.writer(io, &buf);
        const r = opts.rules;
        try fw.interface.print("# best of {d} rollouts, seed {d}: {d} points ({d} players, {d} blanks, {s})\n", .{
            all.games, opts.seed, best.game.total(), r.players, r.blanks, @tagName(r.end_rule),
        });
        try writeTurns(&fw.interface, lex, &Game.init(r), prefix.items, false);
        try writeTurns(&fw.interface, lex, &sh.start, best.turns[0..best.n], false);
        try fw.interface.flush();
    }
}

fn cmdVerify(gpa: Allocator, io: Io, lex: *const Lexicon, opts: Options, out: *Io.Writer) !void {
    var words = try loadWordSets(gpa, io, opts.words);
    defer words.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rng = prng.random();
    const games = if (opts.games == 0) 2 else opts.games;
    var positions: usize = 0;
    var compared: usize = 0;
    var gad = movegen.Collector{ .gpa = gpa };
    defer gad.deinit();
    var ref = movegen.Collector{ .gpa = gpa };
    defer ref.deinit();
    var turns: std.ArrayList(Turn) = .empty;
    defer turns.deinit(gpa);
    const realized = try gpa.create(game.Realized);
    defer gpa.destroy(realized);

    for (0..games) |gi| {
        var g = Game.init(opts.rules);
        turns.clearRetainingCapacity();
        while (!g.over()) {
            const sup = supplyFor(&g, opts);
            gad.moves.clearRetainingCapacity();
            ref.moves.clearRetainingCapacity();
            _ = try allMoves(gpa, io, lex, &words, &g, sup, false, &gad);
            if (opts.blank_use == .any) {
                _ = try allMoves(gpa, io, lex, &words, &g, sup, true, &ref);
                const a = try canonicalize(gpa, gad.moves.items);
                defer gpa.free(a);
                const b = try canonicalize(gpa, ref.moves.items);
                defer gpa.free(b);
                if (a.len != gad.moves.items.len) {
                    try out.print("game {d} turn {d}: GADDAG produced duplicate plays\n", .{ gi, g.turns });
                    return error.VerifyFailed;
                }
                try compareSets(out, &g.pos, a, b, gad.moves.items, ref.moves.items, gi);
                compared += a.len;
            }
            for (gad.moves.items) |*m| {
                if (g.pos.scoreMove(m, g.rules) != m.score or m.placed > sup.max_tiles) {
                    try out.print("game {d} turn {d}: bad play (score {d}, rescored {d}): ", .{ gi, g.turns, m.score, g.pos.scoreMove(m, g.rules) });
                    try m.writeCanonical(out);
                    return error.VerifyFailed;
                }
            }
            positions += 1;
            if (gad.moves.items.len == 0) {
                g.pass();
                try turns.append(gpa, .pass);
                continue;
            }
            // Alternate between a random play and the top-scoring one so the
            // games cover both cluttered and premium-heavy boards.
            const pick = if (rng.boolean())
                gad.moves.items[rng.uintLessThan(usize, gad.moves.items.len)]
            else blk: {
                var best = gad.moves.items[0];
                for (gad.moves.items) |m| if (m.score > best.score) {
                    best = m;
                };
                break :blk best;
            };
            g.play(lex, &pick);
            try turns.append(gpa, .{ .play = pick });

            var fresh = g.pos;
            fresh.recomputeCrossChecks(lex);
            for (0..2) |d| for (0..rules.NN) |i| {
                if (g.pos.cells[d][i] != 0) continue;
                if (fresh.xmask[d][i] != g.pos.xmask[d][i] or fresh.xsum[d][i] != g.pos.xsum[d][i]) {
                    try out.print("game {d} turn {d}: stale cross-check at dir {d} index {d}\n", .{ gi, g.turns, d, i });
                    return error.VerifyFailed;
                }
            };
        }
        // The finished game must replay with real racks and give the same scores.
        const again = try game.realize(lex, opts.rules, turns.items, realized);
        if (again.score[0] != g.score[0] or again.score[1] != g.score[1]) return error.VerifyFailed;
        try out.print("game {d}: {d} turns, ", .{ gi, g.turns });
        try writeSummary(out, &g);
        try out.flush();
    }
    if (opts.blank_use == .any) {
        try out.print("ok: {d} positions, {d} plays matched, every game replayed with real racks\n", .{ positions, compared });
    } else {
        try out.print("ok: {d} positions rescored, every game replayed with real racks\n", .{positions});
    }
}

fn compareSets(out: *Io.Writer, pos: *const Position, a: []const Keyed, b: []const Keyed, am: []const Move, bm: []const Move, g: usize) !void {
    var i: usize = 0;
    var j: usize = 0;
    var bad: usize = 0;
    while (i < a.len or j < b.len) {
        const ord: std.math.Order = if (i == a.len) .gt else if (j == b.len) .lt else std.mem.order(u16, &a[i].key, &b[j].key);
        switch (ord) {
            .eq => {
                if (a[i].score != b[j].score) {
                    try out.print("score differs (gaddag {d}, csp {d}): ", .{ a[i].score, b[j].score });
                    try am[a[i].idx].writeCanonical(out);
                    bad += 1;
                }
                i += 1;
                j += 1;
            },
            .lt => {
                try out.writeAll("only gaddag: ");
                try am[a[i].idx].writeCanonical(out);
                bad += 1;
                i += 1;
            },
            .gt => {
                try out.writeAll("only csp:    ");
                try bm[b[j].idx].writeCanonical(out);
                bad += 1;
                j += 1;
            },
        }
        if (bad >= 20) break;
    }
    if (bad > 0) {
        try out.print("game {d}: generators disagree ({d} vs {d} plays)\n", .{ g, a.len, b.len });
        try pos.writeBoard(out);
        return error.VerifyFailed;
    }
}

test {
    _ = rules;
    _ = lexicon;
    _ = board;
    _ = movegen;
    _ = csp;
    _ = game;
    _ = search;
    _ = @import("tests.zig");
}
