//! Beam search over the number of tiles placed.
//!
//! The tile supply is the budget, so positions are compared with others that
//! have spent the same number of tiles: layer t holds the best `width`
//! positions with exactly t tiles on the board (the per-tile normalisation of
//! the 2023 solver, applied to whole positions). Expanding a position keeps
//! its best few plays of each size; a play of k tiles sends a child to layer
//! t + k. Layers are closed in order, so a layer is final once every layer
//! that can reach it has been expanded. Positions reached by different
//! orders of the same plays are merged (Zobrist hash of the board, plus whose
//! turn it is and the rack sizes).

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const game = @import("game.zig");
const planmod = @import("plan.zig");
const affix = @import("affix.zig");

const Allocator = std.mem.Allocator;
const Move = board.Move;
const Game = game.Game;
const N = rules.N;

pub const Options = struct {
    /// Positions kept per layer.
    width: usize = 64,
    /// Plays of each size (1..7 tiles) kept from each position.
    per_size: usize = 8,
    /// Ranking bonus per blank still unplayed. Blanks enable plays later, and
    /// a plain score comparison does not see that.
    blank_keep: i32 = 0,
    threads: usize = 1,
    blank_use: movegen.BlankMode = .any,
    /// Big plays to set up (see plan.zig). Until a plan's finish is played,
    /// its squares are restricted and its finishing tiles held back.
    plans: []const planmod.Plan = &.{},
    /// Ranking bonus per filled setup square of an unfinished edge plan
    /// (middle plans get a third).
    goal_bonus: i32 = 100,
    /// Ranking bonus per filled setup square of a middle plan.
    mid_bonus: i32 = 20,
    /// Whether finishes may earn hooks, and how hooks are grown.
    hooks: planmod.HookMode = .off,
    /// Word, prefix and suffix sets; required when `hooks` is not `.off`.
    affixes: ?*const affix.Affixes = null,
    /// Ranking penalty per hook that is not yet a word (it blocks a finish).
    pending_penalty: i32 = 30,
    /// Ranking penalty per square of distance from an unfilled setup square
    /// to the nearest tile (beyond adjacent).
    approach: i32 = 0,
    /// Skip plays that would leave a live plan short of letters.
    guard: bool = false,
    /// Ranking cost per point of letter value for each tile a play takes
    /// from a live plan's setup letters (0: none). With a x27 plan, 27
    /// prices the blank the setup would then need.
    letter_cost: i32 = 0,
    /// At most this many positions per layer with the same plan progress
    /// (which setup squares are filled, which plans were given up); 0 for
    /// no limit. Keeps alternative lines of setup alive.
    per_signature: usize = 0,
    /// Ranking bonus for each tail (see plan.Tail), in percent of its hook's
    /// points, shared among its squares.
    tail_bonus: i32 = 100,
    /// Ranking penalty per plan a line has lost (given up, or no longer
    /// reachable). The records need all their big plays, so with a large
    /// penalty such lines survive only when no line with every plan does.
    drop_penalty: i32 = 0,
    /// Setup and tail bonuses of the plans after the first live one, in
    /// percent (see plan.Context.later_bonus).
    later_bonus: i32 = 100,
    /// Board targets (see plan.Target) and the bonus for each one filled.
    targets: []const planmod.Target = &.{},
    target_bonus: i32 = 0,
    /// Keep other squares empty until every target is filled, except those
    /// marked in `free`.
    exclusive: bool = false,
    free: ?*const [rules.NN]bool = null,
    /// Tie-break order between equally ranked positions; different seeds
    /// give different (each reproducible) searches.
    seed: u64 = 0,
};

/// Keeps the best `k` plays of each size, ranked by score plus what the
/// play does for the open plans: a bonus for each setup square it fills,
/// and the change in hook points (less a penalty for each hook it leaves
/// blocking a finish).
pub const TopBySize = struct {
    pub const max_k = 32;
    k: usize,
    /// The open plans of the position being expanded, if any.
    act: ?*const planmod.Active = null,
    pos: ?*const board.Position = null,
    /// Skip plays that would starve a live plan.
    guard: bool = false,
    letter_cost: i32 = 0,
    pending_penalty: i32 = 0,
    n: [N + 1]u8 = @splat(0),
    worst: [N + 1]i32 = @splat(std.math.minInt(i32)),
    worst_at: [N + 1]u8 = @splat(0),
    keys: [N + 1][max_k]i32 = undefined,
    moves: [N + 1][max_k]Move = undefined,
    seen: u64 = 0,

    pub fn reset(self: *TopBySize) void {
        self.n = @splat(0);
        self.worst = @splat(std.math.minInt(i32));
        self.seen = 0;
    }

    pub fn visit(self: *TopBySize, c: *const movegen.Candidate) void {
        self.seen += 1;
        const s = c.placed;
        var key = c.score;
        if (self.act) |a| {
            key += a.goalValue(c.dir, c.line, c.new_mask);
            if (a.n_hooks != 0 or a.n_guards != 0) {
                const d = a.hookDelta(self.pos.?, c) orelse return;
                key += d.value - self.pending_penalty * d.pending;
            }
            // What is left can add at most `max_approach` (the letter cost
            // only subtracts), so most plays stop here, before the loops
            // over their tiles.
            if (self.n[s] == self.k and key + a.max_approach <= self.worst[s]) return;
            if (self.guard and !a.affordable(c.dir, c.line, c.new_mask, c.tiles, c.blanks)) return;
            key += a.approachValue(c.dir, c.line, c.new_mask);
            if (self.letter_cost != 0) key -= a.letterCost(self.letter_cost, c.dir, c.line, c.new_mask, c.tiles, c.blanks);
        }
        if (self.n[s] < self.k) {
            self.moves[s][self.n[s]] = c.toMove();
            self.keys[s][self.n[s]] = key;
            self.n[s] += 1;
            if (self.n[s] == self.k) self.findWorst(s);
        } else if (key > self.worst[s]) {
            self.moves[s][self.worst_at[s]] = c.toMove();
            self.keys[s][self.worst_at[s]] = key;
            self.findWorst(s);
        }
    }

    fn findWorst(self: *TopBySize, s: usize) void {
        var w: i32 = std.math.maxInt(i32);
        var at: u8 = 0;
        for (self.keys[s][0..self.n[s]], 0..) |k, i| {
            if (k < w) {
                w = k;
                at = @intCast(i);
            }
        }
        self.worst[s] = w;
        self.worst_at[s] = at;
    }
};

/// One step of a game's history; histories share prefixes.
const Hist = struct {
    parent: u32,
    turn: game.Turn,
};
const no_hist = std.math.maxInt(u32);

const Entry = struct {
    g: Game,
    hist: u32,
    board_hash: u64,
    /// Plans given up in this line of play.
    dropped: u8 = 0,
};

const Cand = struct {
    rank: i64,
    board_hash: u64,
    parent_layer: u8,
    parent: u32,
    /// Forced passes to replay before the play (no play was possible).
    passes: u8,
    dropped: u8,
    move: Move,

    /// Best first; ties broken by board (mixed with the seed) and move, so
    /// results do not depend on which thread expanded what.
    fn before(seed: u64, a: Cand, b: Cand) bool {
        if (a.rank != b.rank) return a.rank > b.rank;
        const ha = tileHash(@intCast(seed & 0xffff), 0) ^ a.board_hash;
        const hb = tileHash(@intCast(seed & 0xffff), 0) ^ b.board_hash;
        if (ha != hb) return tileHash(0, 0) *% ha < tileHash(0, 0) *% hb;
        return std.mem.order(u16, &a.move.key(), &b.move.key()) == .lt;
    }
};

pub const Result = struct {
    best: ?Game,
    turns: []game.Turn,
    expanded: u64,
    plays_seen: u64,
    seconds: f64,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.turns);
    }
};

/// Hash contribution of one tile; XOR over the board's tiles is order-free.
fn tileHash(sq: usize, cell: u8) u64 {
    var x: u64 = (@as(u64, sq) << 8) | cell;
    x +%= 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

fn moveHash(m: *const Move) u64 {
    var h: u64 = 0;
    var it = m.new_mask;
    while (it != 0) : (it &= it - 1) {
        const i: usize = @ctz(it);
        h ^= tileHash(m.row(i) * N + m.col(i), m.tiles[i]);
    }
    return h;
}

fn boardHash(pos: *const board.Position) u64 {
    var h: u64 = 0;
    for (pos.cells[0], 0..) |c, sq| {
        if (c != 0) h ^= tileHash(sq, c);
    }
    return h;
}

/// The full state key: the board plus what the board does not show.
fn stateKey(board_hash: u64, g: *const Game, dropped: u8) u64 {
    const extra = (@as(u64, dropped) << 32) | (@as(u64, g.turn) << 24) | (@as(u64, g.rack[0]) << 16) | (@as(u64, g.rack[1]) << 8) | g.scoreless;
    return board_hash ^ tileHash(4096, 0) *% (extra + 1);
}

fn blanksIn(m: *const Move) i32 {
    var n: i32 = 0;
    var it = m.new_mask;
    while (it != 0) : (it &= it - 1) n += @intFromBool(rules.isBlank(m.tiles[@ctz(it)]));
    return n;
}

const Shared = struct {
    lex: *const lexicon.Lexicon,
    opts: Options,
    layer: []const Entry,
    layer_index: u8,
    next: std.atomic.Value(usize) = .init(0),
};

const Worker = struct {
    gpa: Allocator,
    top: TopBySize,
    out: std.ArrayList(Cand) = .empty,
    /// Positions that turned out to be finished (no play, game over).
    done: std.ArrayList(struct { entry: u32, passes: u8, g: Game }) = .empty,
    plays_seen: u64 = 0,
    failed: bool = false,

    fn run(w: *Worker, sh: *Shared) void {
        while (true) {
            const i = sh.next.fetchAdd(1, .monotonic);
            if (i >= sh.layer.len) break;
            w.expand(sh, @intCast(i)) catch {
                w.failed = true;
                return;
            };
        }
    }

    fn expand(w: *Worker, sh: *Shared, idx: u32) !void {
        const e = &sh.layer[idx];
        var g = e.g;
        var passes: u8 = 0;
        var act: planmod.Active = undefined;
        var finishes: [planmod.max_plans]Move = undefined;
        var fin_plan: [planmod.max_plans]u8 = undefined;
        var n_fin: usize = 0;
        var dropped = e.dropped;
        const ctx = planmod.Context{
            .hooks = sh.opts.hooks,
            .affixes = sh.opts.affixes,
            .goal_bonus = sh.opts.goal_bonus,
            .mid_bonus = sh.opts.mid_bonus,
            .approach = sh.opts.approach,
            .tail_bonus = sh.opts.tail_bonus,
            .later_bonus = sh.opts.later_bonus,
            .targets = sh.opts.targets,
            .target_bonus = sh.opts.target_bonus,
            .exclusive = sh.opts.exclusive,
            .free = sh.opts.free,
            .lex = sh.lex,
        };
        var undone: u8 = 0;
        for (sh.opts.plans, 0..) |*p, i| {
            if (!p.completed(&g.pos)) undone |= @as(u8, 1) << @intCast(i);
        }
        while (true) {
            // Live plans narrow the supply and the squares.
            var sup = g.supply();
            sup.blanks = sh.opts.blank_use;
            act = planmod.activate(sh.opts.plans, &g.pos, dropped, ctx);
            for (0..27) |t| sup.counts[t] -|= act.reserve[t];
            n_fin = 0;
            for (sh.opts.plans, 0..) |*p, i| {
                if ((act.live >> @intCast(i)) & 1 == 0) continue;
                // The finish may take its own reserve and the spare tiles,
                // never what the other live plans still need.
                var avail: [27]u8 = undefined;
                for (0..27) |t| avail[t] = @min(g.pos.pool[t] -| (act.reserve[t] - p.reserve[t]), p.reserve[t] +| act.spare[t]);
                if (p.finishMove(sh.lex, g.rules, &g.pos, &avail)) |m| {
                    if (m.placed <= g.rack[g.turn]) {
                        finishes[n_fin] = m;
                        fin_plan[n_fin] = @intCast(i);
                        n_fin += 1;
                    }
                }
            }
            const steer = act.live != 0 or sh.opts.targets.len != 0;
            if (steer) sup.restrict = &act.restrict;
            w.top.reset();
            w.top.act = if (steer) &act else null;
            w.top.pos = &g.pos;
            w.top.guard = sh.opts.guard;
            w.top.pending_penalty = sh.opts.pending_penalty;
            w.top.letter_cost = sh.opts.letter_cost;
            movegen.generate(TopBySize, &w.top, &g.pos, sh.lex, g.rules, sup);
            w.plays_seen += w.top.seen;
            var kept: usize = 0;
            for (w.top.n) |k| kept += k;
            if (kept != 0 or n_fin != 0) break;
            // Stuck under the plans: give them up rather than pass.
            if (act.live != 0) {
                dropped |= act.live;
                continue;
            }
            g.pass();
            passes += 1;
            if (g.over()) {
                try w.done.append(w.gpa, .{ .entry = idx, .passes = passes, .g = g });
                return;
            }
        }
        const total: i64 = g.total();
        const blanks_left: i32 = g.pos.pool[rules.BLANK_SLOT];
        // What the open plans promise: their setup bonus and hook points,
        // less the price of the plans this line has lost.
        const pen = sh.opts.pending_penalty;
        const lost: i64 = @popCount(undone & ~act.live);
        const promise: i64 = act.bonus_now + act.approach_now + act.hook_value - pen * act.hook_pending - sh.opts.drop_penalty * lost;
        for (1..N + 1) |s| {
            for (w.top.moves[s][0..w.top.n[s]], w.top.keys[s][0..w.top.n[s]]) |*m, key| {
                const keep = sh.opts.blank_keep * (blanks_left - blanksIn(m));
                try w.out.append(w.gpa, .{
                    .rank = total + key + keep + promise,
                    .board_hash = e.board_hash ^ moveHash(m),
                    .parent_layer = sh.layer_index,
                    .parent = idx,
                    .passes = passes,
                    .dropped = dropped,
                    .move = m.*,
                });
            }
        }
        // A finish turns its plan's promise into its real score.
        for (finishes[0..n_fin], fin_plan[0..n_fin]) |*m, pi| {
            const keep = sh.opts.blank_keep * (blanks_left - blanksIn(m));
            const own: i64 = act.own_bonus[pi] + act.plan_hook_value[pi] - pen * act.plan_pending[pi];
            try w.out.append(w.gpa, .{
                .rank = total + m.score + keep + promise - own,
                .board_hash = e.board_hash ^ moveHash(m),
                .parent_layer = sh.layer_index,
                .parent = idx,
                .passes = passes,
                .dropped = dropped,
                .move = m.*,
            });
        }
    }
};

const Search = struct {
    gpa: Allocator,
    lex: *const lexicon.Lexicon,
    opts: Options,
    hist: std.ArrayList(Hist) = .empty,
    layers: [101]std.ArrayList(Entry) = @splat(.empty),
    cands: [101 + N]std.ArrayList(Cand) = @splat(.empty),
    best: ?Game = null,
    best_hist: u32 = no_hist,
    expanded: u64 = 0,
    plays_seen: u64 = 0,

    fn deinit(s: *Search) void {
        s.hist.deinit(s.gpa);
        for (&s.layers) |*l| l.deinit(s.gpa);
        for (&s.cands) |*c| c.deinit(s.gpa);
    }

    fn addHist(s: *Search, parent: u32, turn: game.Turn) !u32 {
        try s.hist.append(s.gpa, .{ .parent = parent, .turn = turn });
        return @intCast(s.hist.items.len - 1);
    }

    fn finished(s: *Search, g: *const Game, hist: u32) void {
        const better = if (s.best) |b|
            g.total() > b.total() or (g.total() == b.total() and boardHash(&g.pos) < boardHash(&b.pos))
        else
            true;
        if (better) {
            s.best = g.*;
            s.best_hist = hist;
        }
    }

    /// Turn layer t's candidates into its positions: best first, one per state.
    fn close(s: *Search, t: usize) !void {
        const cands = s.cands[t].items;
        std.sort.pdq(Cand, cands, s.opts.seed, Cand.before);
        var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer seen.deinit(s.gpa);
        var sigs: std.AutoHashMapUnmanaged(u64, u32) = .empty;
        defer sigs.deinit(s.gpa);
        for (cands) |*c| {
            const parent = &s.layers[c.parent_layer].items[c.parent];
            if (s.layers[t].items.len >= s.opts.width) {
                // The layer is full; only a play that ends the game (by
                // going out) still counts.
                if (c.passes != 0 or parent.g.bag != 0 or c.move.placed != parent.g.rack[parent.g.turn]) continue;
            }
            var g = parent.g;
            var h = parent.hist;
            for (0..c.passes) |_| {
                g.pass();
                h = try s.addHist(h, .pass);
            }
            g.play(s.lex, &c.move);
            const key = stateKey(c.board_hash, &g, c.dropped);
            if (seen.contains(key)) continue;
            if (g.over()) {
                try seen.put(s.gpa, key, {});
                s.finished(&g, try s.addHist(h, .{ .play = c.move }));
                continue;
            }
            if (s.layers[t].items.len >= s.opts.width) continue;
            if (s.opts.per_signature != 0 and s.opts.plans.len != 0) {
                var sig: u64 = c.dropped;
                for (s.opts.plans, 0..) |*p, i| {
                    sig = sig *% 0x9E3779B97F4A7C15 +% (@as(u64, p.filled(&g.pos)) << 1 | @intFromBool(p.done(&g.pos))) +% i;
                    sig = sig *% 0x9E3779B97F4A7C15 +% p.tailMask(&g.pos);
                }
                const sg = try sigs.getOrPut(s.gpa, sig);
                if (!sg.found_existing) sg.value_ptr.* = 0;
                if (sg.value_ptr.* >= s.opts.per_signature) continue;
                sg.value_ptr.* += 1;
            }
            try seen.put(s.gpa, key, {});
            h = try s.addHist(h, .{ .play = c.move });
            try s.layers[t].append(s.gpa, .{ .g = g, .hist = h, .board_hash = c.board_hash, .dropped = c.dropped });
        }
        s.cands[t].clearAndFree(s.gpa);
    }

    fn expand(s: *Search, t: usize) !void {
        const layer = s.layers[t].items;
        if (layer.len == 0) return;
        var sh = Shared{ .lex = s.lex, .opts = s.opts, .layer = layer, .layer_index = @intCast(t) };
        const nthreads = @max(1, @min(s.opts.threads, layer.len));
        const workers = try s.gpa.alloc(Worker, nthreads);
        defer s.gpa.free(workers);
        for (workers) |*w| w.* = .{ .gpa = s.gpa, .top = .{ .k = @min(s.opts.per_size, TopBySize.max_k) } };
        defer for (workers) |*w| {
            w.out.deinit(s.gpa);
            w.done.deinit(s.gpa);
        };
        if (nthreads == 1) {
            workers[0].run(&sh);
        } else {
            const threads = try s.gpa.alloc(std.Thread, nthreads);
            defer s.gpa.free(threads);
            for (threads, workers) |*th, *w| th.* = try std.Thread.spawn(.{}, Worker.run, .{ w, &sh });
            for (threads) |th| th.join();
        }
        for (workers) |*w| {
            if (w.failed) return error.OutOfMemory;
            s.plays_seen += w.plays_seen;
            for (w.out.items) |c| try s.cands[t + c.move.placed].append(s.gpa, c);
            for (w.done.items) |*d| {
                var h = layer[d.entry].hist;
                for (0..d.passes) |_| h = try s.addHist(h, .pass);
                s.finished(&d.g, h);
            }
        }
        s.expanded += layer.len;
    }
};

/// Search from `start` and return the best finished game found.
pub fn search(
    gpa: Allocator,
    lex: *const lexicon.Lexicon,
    start: *const Game,
    opts: Options,
    progress: ?*std.Io.Writer,
    io: std.Io,
) !Result {
    var s = Search{ .gpa = gpa, .lex = lex, .opts = opts };
    defer s.deinit();
    const t0 = std.Io.Timestamp.now(io, .awake);
    const first: usize = start.pos.tiles_on_board;
    const root = try s.addHist(no_hist, .pass); // placeholder, skipped when rebuilding
    try s.layers[first].append(gpa, .{ .g = start.*, .hist = root, .board_hash = boardHash(&start.pos) });
    const last: usize = start.pos.tiles_on_board + start.pos.pool_total;
    for (first..last + 1) |t| {
        if (t > first) try s.close(t);
        try s.expand(t);
        // Layers more than a rack behind can no longer receive or give children.
        if (t >= first + N) s.layers[t - N].clearAndFree(gpa);
        if (progress) |w| {
            if (s.layers[t].items.len > 0) {
                // Scores as unsigned: a signed width-padded integer prints with a '+'.
                const top: u32 = @intCast(@max(0, s.layers[t].items[0].g.total()));
                const fin: u32 = @intCast(@max(0, if (s.best) |b| b.total() else 0));
                const secs = secondsSince(io, t0);
                try w.print("tiles {d:>3}: {d:>4} positions, best {d:>5}; finished best {d:>5}; {d:.1} s\n", .{
                    t, s.layers[t].items.len, top, fin, secs,
                });
                try w.flush();
            }
        }
    }

    var turns: std.ArrayList(game.Turn) = .empty;
    errdefer turns.deinit(gpa);
    var h = s.best_hist;
    while (h != no_hist and h != root) : (h = s.hist.items[h].parent) {
        try turns.append(gpa, s.hist.items[h].turn);
    }
    std.mem.reverse(game.Turn, turns.items);
    return .{
        .best = s.best,
        .turns = try turns.toOwnedSlice(gpa),
        .expanded = s.expanded,
        .plays_seen = s.plays_seen,
        .seconds = secondsSince(io, t0),
    };
}

fn secondsSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    const d = t0.untilNow(io, .awake);
    return @as(f64, @floatFromInt(d.toNanoseconds())) / 1e9;
}

test "TopBySize keeps the best plays of each size" {
    var top = TopBySize{ .k = 2 };
    top.reset();
    const tiles: [N]u8 = @splat(0);
    const scores = [_]i32{ 5, 9, 3, 7, 11 };
    for (scores) |sc| {
        const c = movegen.Candidate{ .score = sc, .placed = 2, .dir = 0, .line = 0, .start = 0, .end = 1, .new_mask = 3, .tiles = &tiles };
        top.visit(&c);
    }
    try std.testing.expectEqual(@as(u8, 2), top.n[2]);
    var got = [2]i32{ top.moves[2][0].score, top.moves[2][1].score };
    std.mem.sort(i32, &got, {}, std.sort.desc(i32));
    try std.testing.expectEqual([2]i32{ 11, 9 }, got);
}
