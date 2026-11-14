const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const csp = @import("csp.zig");
const search = @import("search.zig");

const Lexicon = lexicon.Lexicon;
const Move = board.Move;
const Position = board.Position;

const usage =
    \\usage: bingo <command> [options]
    \\
    \\commands:
    \\  moves     list every legal play in a position, one per line
    \\  show      print a position
    \\  bench     time move generation at each ply of a game
    \\  rollout   play games with the 2023 annealed-softmax policy
    \\  verify    check the GADDAG generator against the CSP port on random games
    \\  lexicon   build (or load) the GADDAG and report its size
    \\
    \\options:
    \\  --words PATH    word list (default sowpods.txt)
    \\  --cache PATH    GADDAG cache file (default sowpods.gaddag, "none" to skip)
    \\  --plays PATH    position as a play file: "ROW COL across|down WORD" per line
    \\  --ply K         use only the first K plays of --plays
    \\  --engine E      gaddag (default) or csp, for `moves` and `bench`
    \\  --blanks N      blanks in the bag (default 0, as in the Python reference)
    \\  --games N       games for rollout and verify
    \\  --threads N     rollout worker threads (default: all cores)
    \\  --seed S        random seed
    \\  --reps N        timing repetitions per position for bench
    \\  --out PATH      write the best rollout game as a play file
    \\  --count         moves: print only the count and timing
    \\
;

const Options = struct {
    command: []const u8 = "",
    words: []const u8 = "sowpods.txt",
    cache: []const u8 = "sowpods.gaddag",
    plays: ?[]const u8 = null,
    ply: ?usize = null,
    engine: []const u8 = "gaddag",
    blanks: u8 = 0,
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
        } else if (std.mem.eql(u8, a, "--plays")) {
            o.plays = v;
        } else if (std.mem.eql(u8, a, "--ply")) {
            o.ply = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--engine")) {
            o.engine = v;
        } else if (std.mem.eql(u8, a, "--blanks")) {
            o.blanks = try std.fmt.parseInt(u8, v, 10);
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

    const r = rules.Rules{ .blanks = opts.blanks };
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
        const pos = try loadPosition(gpa, io, &lex, r, opts.plays, opts.ply, null);
        try pos.writeBoard(out);
    } else if (std.mem.eql(u8, cmd, "moves")) {
        try cmdMoves(gpa, io, &lex, r, opts, out);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        try cmdBench(gpa, io, &lex, r, opts, out);
    } else if (std.mem.eql(u8, cmd, "rollout")) {
        try cmdRollout(gpa, io, &lex, r, opts, out);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        try cmdVerify(gpa, io, &lex, r, opts, out);
    } else {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
}

fn secondsSince(io: Io, t0: Io.Timestamp) f64 {
    const d = t0.untilNow(io, .awake);
    return @as(f64, @floatFromInt(d.toNanoseconds())) / 1e9;
}

/// Replay a play file onto an empty board. When `moves_out` is given, every
/// replayed play is appended to it.
fn loadPosition(
    gpa: Allocator,
    io: Io,
    lex: *const Lexicon,
    r: rules.Rules,
    path: ?[]const u8,
    limit: ?usize,
    moves_out: ?*std.ArrayList(Move),
) !Position {
    var pos = Position.init(r);
    const p = path orelse return pos;
    const text = try Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var line_no: usize = 0;
    var applied: usize = 0;
    while (lines.next()) |raw| {
        line_no += 1;
        if (limit) |k| if (applied >= k) break;
        const line = std.mem.trim(u8, raw[0 .. std.mem.indexOfScalar(u8, raw, '#') orelse raw.len], " \t\r");
        if (line.len == 0) continue;
        var tok = std.mem.tokenizeAny(u8, line, " \t");
        const row = std.fmt.parseInt(usize, tok.next() orelse return badLine(p, line_no), 10) catch return badLine(p, line_no);
        const col = std.fmt.parseInt(usize, tok.next() orelse return badLine(p, line_no), 10) catch return badLine(p, line_no);
        const dir_s = tok.next() orelse return badLine(p, line_no);
        const word = tok.next() orelse return badLine(p, line_no);
        const d: u1 = if (std.mem.eql(u8, dir_s, "across")) 0 else if (std.mem.eql(u8, dir_s, "down")) 1 else return badLine(p, line_no);
        const m = pos.parsePlay(lex, r, row, col, d, word) catch |err| {
            std.debug.print("{s}:{d}: illegal play {s}: {t}\n", .{ p, line_no, line, err });
            return err;
        };
        pos.apply(lex, &m);
        if (moves_out) |list| try list.append(gpa, m);
        applied += 1;
    }
    return pos;
}

fn badLine(path: []const u8, line_no: usize) error{BadPlayFile} {
    std.debug.print("{s}:{d}: expected ROW COL across|down WORD\n", .{ path, line_no });
    return error.BadPlayFile;
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
    pos: *const Position,
    r: rules.Rules,
    use_csp: bool,
    collector: *movegen.Collector,
) !f64 {
    const t0 = Io.Timestamp.now(io, .awake);
    if (use_csp) {
        const halo = csp.computeHalo(words.?, pos);
        const t1 = Io.Timestamp.now(io, .awake);
        try csp.generate(movegen.Collector, collector, gpa, words.?, pos, &halo, r);
        if (collector.failed) return error.OutOfMemory;
        return secondsSince(io, t1);
    }
    movegen.generate(movegen.Collector, collector, pos, lex, r);
    if (collector.failed) return error.OutOfMemory;
    return secondsSince(io, t0);
}

fn loadWordSets(gpa: Allocator, io: Io, path: []const u8) !csp.WordSets {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(text);
    return csp.WordSets.init(gpa, text);
}

fn cmdMoves(gpa: Allocator, io: Io, lex: *const Lexicon, r: rules.Rules, opts: Options, out: *Io.Writer) !void {
    const pos = try loadPosition(gpa, io, lex, r, opts.plays, opts.ply, null);
    const use_csp = std.mem.eql(u8, opts.engine, "csp");
    var words: ?csp.WordSets = if (use_csp) try loadWordSets(gpa, io, opts.words) else null;
    defer if (words) |*w| w.deinit(gpa);

    var col = movegen.Collector{ .gpa = gpa };
    defer col.deinit();
    const secs = try allMoves(gpa, io, lex, if (words) |*w| w else null, &pos, r, use_csp, &col);
    const keyed = try canonicalize(gpa, col.moves.items);
    defer gpa.free(keyed);
    if (!opts.count_only) {
        for (keyed) |k| try col.moves.items[k.idx].writeCanonical(out);
    }
    std.debug.print("{s}: plies={d} moves={d} raw={d} seconds={d:.4}\n", .{
        opts.engine, pos.plies, keyed.len, col.moves.items.len, secs,
    });
}

fn cmdBench(gpa: Allocator, io: Io, lex: *const Lexicon, r: rules.Rules, opts: Options, out: *Io.Writer) !void {
    var game: std.ArrayList(Move) = .empty;
    defer game.deinit(gpa);
    _ = try loadPosition(gpa, io, lex, r, opts.plays, opts.ply, &game);
    const with_csp = std.mem.eql(u8, opts.engine, "csp");
    var words: ?csp.WordSets = if (with_csp) try loadWordSets(gpa, io, opts.words) else null;
    defer if (words) |*w| w.deinit(gpa);

    try out.print("{s:>4} {s:>8} {s:>12} {s:>12}{s}\n", .{
        "ply", "moves", "gaddag_ms", "Mmoves/s", if (with_csp) "       csp_ms  csp/gaddag" else "",
    });
    var pos = Position.init(r);
    var total_secs: f64 = 0;
    var total_moves: u64 = 0;
    for (0..game.items.len + 1) |ply| {
        var best: f64 = std.math.inf(f64);
        var counter = movegen.Counter{};
        for (0..@max(opts.reps, 1)) |_| {
            counter = .{};
            const t0 = Io.Timestamp.now(io, .awake);
            movegen.generate(movegen.Counter, &counter, &pos, lex, r);
            best = @min(best, secondsSince(io, t0));
        }
        total_secs += best;
        total_moves += counter.count;
        try out.print("{d:>4} {d:>8} {d:>12.3} {d:>12.2}", .{
            ply, counter.count, best * 1e3, @as(f64, @floatFromInt(counter.count)) / best / 1e6,
        });
        if (words) |*w| {
            const halo = csp.computeHalo(w, &pos);
            var c2 = movegen.Counter{};
            const t0 = Io.Timestamp.now(io, .awake);
            try csp.generate(movegen.Counter, &c2, gpa, w, &pos, &halo, r);
            const s = secondsSince(io, t0);
            try out.print(" {d:>12.1} {d:>11.0}", .{ s * 1e3, s / best });
        }
        try out.writeByte('\n');
        try out.flush();
        if (ply < game.items.len) pos.apply(lex, &game.items[ply]);
    }
    try out.print("total: {d} moves in {d:.3} ms ({d:.2} M moves/s)\n", .{
        total_moves, total_secs * 1e3, @as(f64, @floatFromInt(total_moves)) / total_secs / 1e6,
    });
}

const Stats = struct {
    games: u64 = 0,
    score_sum: f64 = 0,
    score_sq: f64 = 0,
    plies_sum: f64 = 0,
    left_sum: f64 = 0,
    considered: u64 = 0,
    /// Games that ended with tiles in the bag and no legal play.
    stuck: u64 = 0,
    min: i32 = std.math.maxInt(i32),
    best: search.Game = .{},

    fn add(s: *Stats, g: *const search.Game) void {
        s.games += 1;
        const x: f64 = @floatFromInt(g.score);
        s.score_sum += x;
        s.score_sq += x * x;
        s.plies_sum += @floatFromInt(g.plies);
        s.left_sum += @floatFromInt(g.tiles_left);
        s.considered += g.considered;
        s.stuck += @intFromBool(g.tiles_left > 0);
        s.min = @min(s.min, g.score);
        if (s.games == 1 or g.score > s.best.score) s.best = g.*;
    }

    fn merge(s: *Stats, o: *const Stats) void {
        if (o.games == 0) return;
        if (s.games == 0 or o.best.score > s.best.score) s.best = o.best;
        s.games += o.games;
        s.score_sum += o.score_sum;
        s.score_sq += o.score_sq;
        s.plies_sum += o.plies_sum;
        s.left_sum += o.left_sum;
        s.considered += o.considered;
        s.stuck += o.stuck;
        s.min = @min(s.min, o.min);
    }
};

const Shared = struct {
    lex: *const Lexicon,
    rules: rules.Rules,
    start: Position,
    games: usize,
    next: std.atomic.Value(usize) = .init(0),
    seed: u64,
};

fn rolloutWorker(sh: *Shared, stats: *Stats) void {
    var game: search.Game = .{};
    var prng = std.Random.DefaultPrng.init(0);
    var sampler = search.SoftmaxSampler.init(std.heap.page_allocator, &prng, sh.rules) catch {
        std.debug.print("out of memory\n", .{});
        return;
    };
    defer sampler.deinit(std.heap.page_allocator);
    while (true) {
        const g = sh.next.fetchAdd(1, .monotonic);
        if (g >= sh.games) break;
        // One stream per game, so results do not depend on the thread count.
        prng = std.Random.DefaultPrng.init(sh.seed *% 0x9E3779B97F4A7C15 +% g);
        search.softmaxRollout(&sh.start, sh.lex, sh.rules, &sampler, .{}, &game);
        stats.add(&game);
    }
}

fn cmdRollout(gpa: Allocator, io: Io, lex: *const Lexicon, r: rules.Rules, opts: Options, out: *Io.Writer) !void {
    var sh = Shared{
        .lex = lex,
        .rules = r,
        .start = try loadPosition(gpa, io, lex, r, opts.plays, opts.ply, null),
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

    var all = Stats{};
    for (stats) |*s| all.merge(s);
    const n: f64 = @floatFromInt(all.games);
    const mean = all.score_sum / n;
    const sd = @sqrt(@max(0, all.score_sq / n - mean * mean));
    try out.print(
        \\games {d} on {d} threads in {d:.2} s ({d:.2} games/s, {d:.1} M plays generated/s)
        \\score mean {d:.1} sd {d:.1} min {d} max {d}; plies {d:.1}
        \\stuck with tiles left: {d} games ({d:.1}%), {d:.2} tiles per game on average
        \\
    , .{
        all.games,         nthreads,                                             secs,
        n / secs,          @as(f64, @floatFromInt(all.considered)) / secs / 1e6, mean,
        sd,                all.min,                                              all.best.score,
        all.plies_sum / n, all.stuck,                                            @as(f64, @floatFromInt(all.stuck)) * 100 / n,
        all.left_sum / n,
    });

    // Replay the best game to print it and, if asked, save it.
    var pos = sh.start;
    var file_buf: [4096]u8 = undefined;
    var file: ?Io.File = null;
    var fw: Io.File.Writer = undefined;
    if (opts.out) |p| {
        file = try Io.Dir.cwd().createFile(io, p, .{});
        fw = file.?.writer(io, &file_buf);
        try fw.interface.print("# best of {d} rollouts, seed {d}, blanks {d}: {d} points\n", .{
            all.games, opts.seed, r.blanks, all.best.score,
        });
    }
    defer if (file) |f| f.close(io);
    for (all.best.moves[0..all.best.plies]) |*m| {
        try out.print("{d:>4}  ", .{@as(u32, @intCast(m.score))});
        try pos.writePlay(m, out);
        try out.writeByte('\n');
        if (file != null) {
            try pos.writePlay(m, &fw.interface);
            try fw.interface.writeByte('\n');
        }
        pos.apply(lex, m);
    }
    if (file != null) try fw.interface.flush();
    try pos.writeBoard(out);
}

fn cmdVerify(gpa: Allocator, io: Io, lex: *const Lexicon, r: rules.Rules, opts: Options, out: *Io.Writer) !void {
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

    for (0..games) |g| {
        var pos = Position.init(r);
        while (true) {
            gad.moves.clearRetainingCapacity();
            ref.moves.clearRetainingCapacity();
            _ = try allMoves(gpa, io, lex, &words, &pos, r, false, &gad);
            _ = try allMoves(gpa, io, lex, &words, &pos, r, true, &ref);
            const a = try canonicalize(gpa, gad.moves.items);
            defer gpa.free(a);
            const b = try canonicalize(gpa, ref.moves.items);
            defer gpa.free(b);
            if (a.len != gad.moves.items.len) {
                try out.print("game {d} ply {d}: GADDAG produced duplicate plays\n", .{ g, pos.plies });
                return error.VerifyFailed;
            }
            try compareSets(out, &pos, a, b, gad.moves.items, ref.moves.items, g);
            for (gad.moves.items) |*m| {
                if (pos.scoreMove(m, r) != m.score) {
                    try out.print("game {d} ply {d}: score {d} but rescoring gives {d}: ", .{ g, pos.plies, m.score, pos.scoreMove(m, r) });
                    try m.writeCanonical(out);
                    return error.VerifyFailed;
                }
            }
            positions += 1;
            compared += a.len;
            if (gad.moves.items.len == 0 or pos.bag_total == 0) break;

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
            pos.apply(lex, &pick);

            var fresh = pos;
            fresh.recomputeCrossChecks(lex);
            for (0..2) |d| for (0..rules.NN) |i| {
                if (pos.cells[d][i] != 0) continue;
                if (fresh.xmask[d][i] != pos.xmask[d][i] or fresh.xsum[d][i] != pos.xsum[d][i]) {
                    try out.print("game {d} ply {d}: stale cross-check at dir {d} index {d}\n", .{ g, pos.plies, d, i });
                    return error.VerifyFailed;
                }
            };
        }
        try out.print("game {d}: {d} plies, score {d}, {d} tiles left\n", .{ g, pos.plies, pos.score, pos.bag_total });
        try out.flush();
    }
    try out.print("ok: {d} positions, {d} plays matched\n", .{ positions, compared });
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
        try out.print("game {d} ply {d}: generators disagree ({d} vs {d} plays)\n", .{ g, pos.plies, a.len, b.len });
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
    _ = search;
    _ = @import("tests.zig");
}
