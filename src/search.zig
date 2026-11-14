//! Search drivers. For now only the 2023 policy: at every turn pick a play
//! with probability proportional to exp(points_per_tile / T), where T cools
//! linearly as the bag empties. Metaheuristics (beam search, NMCS, NRPA)
//! will sit on top of the same generator and visitor interface.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");

/// Every play places at least one tile, so a game cannot be longer.
pub const max_plies = 100;

pub const SoftmaxPolicy = struct {
    /// T = temp_scale * tiles_left / initial_tiles (solver.py used 3).
    temp_scale: f64 = 3.0,
    /// Added to T, as in solver.py.
    temp_eps: f64 = 0.001,
};

/// Draws one play from softmax(points_per_tile / T), exactly, in one
/// streaming pass and without an exp() or log() per play.
///
/// Plays with the same (score, tiles placed) have the same weight, so they
/// are grouped: each bucket keeps a count and one representative chosen by
/// reservoir sampling (uniform over the bucket). At the end, the Gumbel-max
/// trick picks a bucket from weights count * exp(x / T):
///     argmax_k  x_k / T + ln(count_k) + G_k,   G_k ~ Gumbel(0, 1).
/// P(play) = P(its bucket) * 1/count = exp(x / T) / Z, as in solver.py, and
/// small temperatures cannot overflow.
pub const SoftmaxSampler = struct {
    prng: *std.Random.DefaultPrng,
    stride: usize,
    counts: []u32,
    reps: []board.Move,
    touched: []u32,
    n_touched: usize = 0,
    /// Plays seen since the last pick.
    seen: u64 = 0,
    inv_temp: f64 = 1,
    /// Plays scoring at least `score_cap` skip the buckets and get their
    /// own Gumbel key; the best of them competes at pick time.
    over_key: f64 = -std.math.inf(f64),
    over_move: board.Move = .{},

    pub const score_cap = 4096;

    pub fn init(gpa: std.mem.Allocator, prng: *std.Random.DefaultPrng, r: rules.Rules) !SoftmaxSampler {
        const stride = @as(usize, r.rack) + 1;
        const counts = try gpa.alloc(u32, score_cap * stride);
        errdefer gpa.free(counts);
        @memset(counts, 0);
        const reps = try gpa.alloc(board.Move, score_cap * stride);
        errdefer gpa.free(reps);
        const touched = try gpa.alloc(u32, score_cap * stride);
        return .{ .prng = prng, .stride = stride, .counts = counts, .reps = reps, .touched = touched };
    }

    pub fn deinit(self: *SoftmaxSampler, gpa: std.mem.Allocator) void {
        gpa.free(self.counts);
        gpa.free(self.reps);
        gpa.free(self.touched);
    }

    inline fn gumbel(self: *SoftmaxSampler) f64 {
        return -@log(self.prng.random().floatExp(f64));
    }

    pub fn visit(self: *SoftmaxSampler, c: *const movegen.Candidate) void {
        self.seen += 1;
        const s: usize = @intCast(c.score);
        if (s >= score_cap) {
            const per_tile = @as(f64, @floatFromInt(c.score)) / @as(f64, @floatFromInt(c.placed));
            const key = per_tile * self.inv_temp + self.gumbel();
            if (key > self.over_key) {
                self.over_key = key;
                self.over_move = c.toMove();
            }
            return;
        }
        const b = s * self.stride + c.placed;
        const n = self.counts[b] + 1;
        self.counts[b] = n;
        if (n == 1) {
            self.touched[self.n_touched] = @intCast(b);
            self.n_touched += 1;
            self.reps[b] = c.toMove();
        } else if ((@as(u128, self.prng.next()) * n) >> 64 == 0) {
            // Keep this play with probability 1/n.
            self.reps[b] = c.toMove();
        }
    }

    /// Choose among the plays seen since the last call and reset.
    pub fn pick(self: *SoftmaxSampler, out: *board.Move) bool {
        var best = self.over_key;
        var found = best != -std.math.inf(f64);
        if (found) out.* = self.over_move;
        for (self.touched[0..self.n_touched]) |b| {
            const score: f64 = @floatFromInt(b / self.stride);
            const placed: f64 = @floatFromInt(b % self.stride);
            const key = score / placed * self.inv_temp + @log(@as(f64, @floatFromInt(self.counts[b]))) + self.gumbel();
            if (key > best) {
                best = key;
                out.* = self.reps[b];
                found = true;
            }
            self.counts[b] = 0;
        }
        self.n_touched = 0;
        self.seen = 0;
        self.over_key = -std.math.inf(f64);
        return found;
    }
};

pub const Game = struct {
    score: i32 = 0,
    plies: u16 = 0,
    tiles_left: u8 = 0,
    /// Plays generated over the whole game (a measure of work done).
    considered: u64 = 0,
    moves: [max_plies]board.Move = undefined,
};

/// Play from `start` until the bag is empty or no play exists.
pub fn softmaxRollout(
    start: *const board.Position,
    lex: *const lexicon.Lexicon,
    r: rules.Rules,
    sampler: *SoftmaxSampler,
    policy: SoftmaxPolicy,
    game: *Game,
) void {
    var pos = start.*;
    game.plies = 0;
    game.considered = 0;
    while (pos.bag_total > 0 and game.plies < max_plies) {
        const t = policy.temp_scale * @as(f64, @floatFromInt(pos.bag_total)) /
            @as(f64, @floatFromInt(pos.initial_total));
        sampler.inv_temp = 1.0 / (t + policy.temp_eps);
        movegen.generate(SoftmaxSampler, sampler, &pos, lex, r);
        game.considered += sampler.seen;
        const m = &game.moves[game.plies];
        if (!sampler.pick(m)) break;
        game.plies += 1;
        pos.apply(lex, m);
    }
    game.score = pos.score;
    game.tiles_left = pos.bag_total;
}

test "softmax sampler matches the target distribution" {
    // Three plays: (score 2, 1 tile), (score 2, 1 tile), (score 6, 2 tiles).
    // Per-tile values 2, 2, 3 at T = 1 give weights e^2, e^2, e^3.
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    var s = try SoftmaxSampler.init(gpa, &prng, .{});
    defer s.deinit(gpa);
    const tiles: [rules.N]u8 = @splat(0);
    const plays = [_]movegen.Candidate{
        .{ .score = 2, .placed = 1, .dir = 0, .line = 0, .start = 0, .end = 1, .new_mask = 1, .tiles = &tiles },
        .{ .score = 2, .placed = 1, .dir = 0, .line = 1, .start = 0, .end = 1, .new_mask = 1, .tiles = &tiles },
        .{ .score = 6, .placed = 2, .dir = 0, .line = 2, .start = 0, .end = 1, .new_mask = 3, .tiles = &tiles },
    };
    var hits = [3]u32{ 0, 0, 0 };
    const trials = 200_000;
    for (0..trials) |_| {
        s.inv_temp = 1;
        for (&plays) |*c| s.visit(c);
        var m: board.Move = .{};
        try std.testing.expect(s.pick(&m));
        hits[m.line] += 1;
    }
    const e2 = @exp(2.0);
    const e3 = @exp(3.0);
    const z = 2 * e2 + e3;
    const want = [3]f64{ e2 / z, e2 / z, e3 / z };
    for (hits, want) |h, p| {
        const got = @as(f64, @floatFromInt(h)) / trials;
        try std.testing.expectApproxEqAbs(p, got, 0.005);
    }
}
