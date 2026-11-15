//! Nested Rollout Policy Adaptation (Rosin 2011) over whole games, the
//! method behind records in single-player puzzles such as Morpion
//! Solitaire. A rollout plays a game to the end, choosing each play at
//! random with probability proportional to exp(weight of the play + bias *
//! its score). A level-n search runs N level-(n-1) searches and after each
//! one moves the weights towards the best game found so far: the chosen
//! plays gain, every play that competed with them loses in proportion to
//! its probability. The weights are learned targets. They come to name the
//! plays, and so the order, that the best games share.
//!
//! Plays are restricted exactly as in the beam search (plans with their
//! reserves, board targets, hooks and guards), so the same options steer
//! both. As in the beam, a position offers only its best `per_size` plays
//! of each size by the beam's ranking key (score plus what the play does
//! for the plans), and the bias weighs that key.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const game = @import("game.zig");
const planmod = @import("plan.zig");
const beam = @import("beam.zig");

const Allocator = std.mem.Allocator;
const Move = board.Move;
const Game = game.Game;
const Turn = game.Turn;

pub const Options = struct {
    level: u8 = 3,
    /// Searches per level.
    iterations: usize = 100,
    /// Learning rate of the adaptation step.
    alpha: f64 = 1.0,
    /// Weight per point of a play's score in the rollout policy.
    bias: f64 = 0.0,
    seed: u64 = 1,
    /// Plans, targets, hooks and the rest, as for the beam search (its
    /// width and ranking options are not used).
    steer: beam.Options = .{},
};

const Policy = std.AutoHashMapUnmanaged(u64, f64);

/// A play's identity: the tiles it places (square and letter, blank or not).
fn code(m: *const Move) u64 {
    var h: u64 = 0x9E3779B97F4A7C15;
    var it = m.new_mask;
    while (it != 0) : (it &= it - 1) {
        const i: usize = @ctz(it);
        var x: u64 = (@as(u64, m.row(i) * rules.N + m.col(i)) << 8) | m.tiles[i];
        x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
        x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
        h ^= x ^ (x >> 31);
    }
    return h;
}

const Search = struct {
    gpa: Allocator,
    lex: *const lexicon.Lexicon,
    start: *const Game,
    opts: Options,
    rng: std.Random,
    moves: std.ArrayList(Move) = .empty,
    /// The beam's ranking key of each play in `moves`.
    keys: std.ArrayList(i32) = .empty,
    weights: std.ArrayList(f64) = .empty,
    top: beam.TopBySize = .{ .k = 8 },
    deltas: std.ArrayList(struct { code: u64, d: f64 }) = .empty,
    rollouts: u64 = 0,

    fn deinit(s: *Search) void {
        s.moves.deinit(s.gpa);
        s.keys.deinit(s.gpa);
        s.weights.deinit(s.gpa);
        s.deltas.deinit(s.gpa);
    }

    /// The plays allowed in `g` (into s.moves). Plans that leave no play
    /// are given up, as in the beam search; none at all means a pass.
    fn legal(s: *Search, g: *const Game, dropped: *u8) !void {
        const o = &s.opts.steer;
        const ctx = planmod.Context{
            .hooks = o.hooks,
            .affixes = o.affixes,
            .goal_bonus = o.goal_bonus,
            .mid_bonus = o.mid_bonus,
            .approach = o.approach,
            .tail_bonus = o.tail_bonus,
            .later_bonus = o.later_bonus,
            .targets = o.targets,
            .target_bonus = o.target_bonus,
            .exclusive = o.exclusive,
            .free = o.free,
            .lex = s.lex,
        };
        while (true) {
            s.moves.clearRetainingCapacity();
            s.keys.clearRetainingCapacity();
            var sup = g.supply();
            sup.blanks = o.blank_use;
            const act = planmod.activate(o.plans, &g.pos, dropped.*, ctx);
            for (0..27) |t| sup.counts[t] -|= act.reserve[t];
            const steer = act.live != 0 or o.targets.len != 0;
            if (steer) sup.restrict = &act.restrict;
            s.top.k = @min(@max(o.per_size, 1), beam.TopBySize.max_k);
            s.top.reset();
            s.top.act = if (steer) &act else null;
            s.top.pos = &g.pos;
            s.top.guard = o.guard;
            s.top.pending_penalty = o.pending_penalty;
            s.top.letter_cost = o.letter_cost;
            movegen.generate(beam.TopBySize, &s.top, &g.pos, s.lex, g.rules, sup);
            for (1..rules.N + 1) |size| {
                for (s.top.moves[size][0..s.top.n[size]], s.top.keys[size][0..s.top.n[size]]) |m, key| {
                    try s.moves.append(s.gpa, m);
                    try s.keys.append(s.gpa, key);
                }
            }
            for (o.plans, 0..) |*p, i| {
                if ((act.live >> @intCast(i)) & 1 == 0) continue;
                var avail: [27]u8 = undefined;
                for (0..27) |t| avail[t] = @min(g.pos.pool[t] -| (act.reserve[t] - p.reserve[t]), p.reserve[t] +| act.spare[t]);
                if (p.finishMove(s.lex, g.rules, &g.pos, &avail)) |m| {
                    if (m.placed <= g.rack[g.turn]) {
                        try s.moves.append(s.gpa, m);
                        try s.keys.append(s.gpa, m.score);
                    }
                }
            }
            if (s.moves.items.len != 0 or act.live == 0) return;
            dropped.* |= act.live;
        }
    }

    /// Softmax weights of s.moves under `pol` (into s.weights); returns their sum.
    fn probabilities(s: *Search, pol: *const Policy) !f64 {
        s.weights.clearRetainingCapacity();
        var top: f64 = -std.math.inf(f64);
        for (s.moves.items, s.keys.items) |*m, key| {
            const w = (pol.get(code(m)) orelse 0) + s.opts.bias * @as(f64, @floatFromInt(key));
            try s.weights.append(s.gpa, w);
            top = @max(top, w);
        }
        var z: f64 = 0;
        for (s.weights.items) |*w| {
            w.* = @exp(w.* - top);
            z += w.*;
        }
        return z;
    }

    /// Play one game from the start under `pol`; returns its total.
    fn rollout(s: *Search, pol: *const Policy, turns: *std.ArrayList(Turn)) !i32 {
        var g = s.start.*;
        var dropped: u8 = 0;
        turns.clearRetainingCapacity();
        while (!g.over()) {
            try s.legal(&g, &dropped);
            if (s.moves.items.len == 0) {
                g.pass();
                try turns.append(s.gpa, .pass);
                continue;
            }
            const z = try s.probabilities(pol);
            var x = s.rng.float(f64) * z;
            var pick: usize = s.moves.items.len - 1;
            for (s.weights.items, 0..) |w, i| {
                if (x < w) {
                    pick = i;
                    break;
                }
                x -= w;
            }
            const m = s.moves.items[pick];
            g.play(s.lex, &m);
            try turns.append(s.gpa, .{ .play = m });
        }
        s.rollouts += 1;
        return g.total();
    }

    /// Move `pol` towards the game `best`.
    fn adapt(s: *Search, pol: *Policy, best: []const Turn) !void {
        var g = s.start.*;
        var dropped: u8 = 0;
        s.deltas.clearRetainingCapacity();
        for (best) |t| {
            try s.legal(&g, &dropped);
            switch (t) {
                .pass => g.pass(),
                .play => |m| {
                    const z = try s.probabilities(pol);
                    for (s.moves.items, s.weights.items) |*mv, w| {
                        try s.deltas.append(s.gpa, .{ .code = code(mv), .d = -s.opts.alpha * w / z });
                    }
                    try s.deltas.append(s.gpa, .{ .code = code(&m), .d = s.opts.alpha });
                    g.play(s.lex, &m);
                },
            }
        }
        for (s.deltas.items) |d| {
            const gop = try pol.getOrPut(s.gpa, d.code);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += d.d;
        }
    }

    fn nested(s: *Search, level: u8, pol: *Policy, best_turns: *std.ArrayList(Turn), report: ?*Report) !i32 {
        var best: i32 = std.math.minInt(i32);
        var tmp: std.ArrayList(Turn) = .empty;
        defer tmp.deinit(s.gpa);
        for (0..s.opts.iterations) |it| {
            var score: i32 = undefined;
            if (level == 1) {
                score = try s.rollout(pol, &tmp);
            } else {
                var child = try pol.clone(s.gpa);
                defer child.deinit(s.gpa);
                score = try s.nested(level - 1, &child, &tmp, null);
            }
            if (score >= best) {
                best = score;
                best_turns.clearRetainingCapacity();
                try best_turns.appendSlice(s.gpa, tmp.items);
            }
            try s.adapt(pol, best_turns.items);
            if (report) |r| try r.top(s, it + 1, best, best_turns.items);
        }
        return best;
    }
};

/// Progress of one job's top-level search.
const Report = struct {
    io: std.Io,
    out: *std.Io.Writer,
    job: usize,
    iterations: usize,
    shared: *Shared,

    fn top(r: *Report, s: *Search, it: usize, best: i32, turns: []const Turn) !void {
        const sh = r.shared;
        sh.mutex.lockUncancelable(r.io);
        defer sh.mutex.unlock(r.io);
        if (best > sh.best) {
            sh.best = best;
            sh.best_turns.clearRetainingCapacity();
            try sh.best_turns.appendSlice(sh.gpa, turns);
            if (sh.saver) |sv| try sv.save(sv.ctx, sh.best_turns.items, best);
        }
        try r.out.print("job {d:>2} iteration {d:>3}/{d}: best {d:>5}, overall {d:>5}; {d} rollouts\n", .{
            r.job, it, r.iterations, @as(u32, @intCast(@max(0, best))), @as(u32, @intCast(@max(0, sh.best))), s.rollouts,
        });
        try r.out.flush();
    }
};

/// Called with each new best game (under the lock), e.g. to write it out.
pub const Saver = struct {
    ctx: *anyopaque,
    save: *const fn (ctx: *anyopaque, turns: []const Turn, total: i32) anyerror!void,
};

/// The best game over all jobs.
pub const Shared = struct {
    mutex: std.Io.Mutex = .init,
    gpa: Allocator,
    best: i32 = std.math.minInt(i32),
    best_turns: std.ArrayList(Turn) = .empty,
    failed: bool = false,
    saver: ?Saver = null,

    pub fn deinit(sh: *Shared) void {
        sh.best_turns.deinit(sh.gpa);
    }
};

/// One NRPA search from `start`, reporting each top-level iteration.
pub fn run(gpa: Allocator, lex: *const lexicon.Lexicon, start: *const Game, opts: Options, io: std.Io, out: *std.Io.Writer, shared: *Shared, job: usize) void {
    var prng = std.Random.DefaultPrng.init(opts.seed);
    var s = Search{ .gpa = gpa, .lex = lex, .start = start, .opts = opts, .rng = prng.random() };
    defer s.deinit();
    var pol: Policy = .empty;
    defer pol.deinit(gpa);
    var best: std.ArrayList(Turn) = .empty;
    defer best.deinit(gpa);
    var report = Report{ .io = io, .out = out, .job = job, .iterations = opts.iterations, .shared = shared };
    _ = s.nested(@max(1, opts.level), &pol, &best, &report) catch {
        shared.mutex.lockUncancelable(io);
        defer shared.mutex.unlock(io);
        shared.failed = true;
    };
}

test "a play's code depends on its tiles, not their order" {
    var a = Move{ .dir = 0, .line = 7, .start = 6, .end = 7, .placed = 2, .new_mask = 0b11 << 6 };
    a.tiles[6] = rules.cellOf(0, false);
    a.tiles[7] = rules.cellOf(19, false);
    var b = a;
    try std.testing.expectEqual(code(&a), code(&b));
    b.tiles[7] = rules.cellOf(19, true);
    try std.testing.expect(code(&a) != code(&b));
}
