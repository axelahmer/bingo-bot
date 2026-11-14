//! GADDAG move generation (Gordon 1994) adapted to the bag-as-rack setting.
//!
//! Every play is walked from one pivot square: the rightmost square of its
//! main word that is constrained, meaning it holds a tile or has a
//! perpendicular word through it. From the pivot we walk the GADDAG
//! leftwards, then (after the separator) rightwards. Everything right of the
//! pivot is open board, so all constraints are met before the walk enters
//! unconstrained squares. Each play has exactly one pivot, so it is produced
//! once. (Gordon's leftmost-anchor rule enumerates open space first, which
//! is cheap with a 7-tile rack but wasteful when the whole bag is the rack.)
//!
//! At each empty square the letters worth trying are one AND away:
//!
//!     candidates = node.children & square.cross_check & bag.non_empty
//!
//! and a child is entered only if its reach mask says some word through it
//! fits the free squares and the tiles left to place. Scores are
//! accumulated during the walk, so a play is scored in O(1) when recorded,
//! and plays go to a comptime visitor that can count, collect, or sample
//! them without storing every play. A one-tile play that forms words both
//! ways is reported only by the across pass.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");

const N = rules.N;
const Move = board.Move;
const Position = board.Position;

/// A play as the generator sees it while recording. `tiles` points into the
/// generator's scratch buffer, so call `toMove` to keep it.
pub const Candidate = struct {
    score: i32,
    placed: u8,
    dir: u1,
    line: u8,
    start: u8,
    end: u8,
    new_mask: u16,
    tiles: *const [N]u8,

    pub fn toMove(c: *const Candidate) Move {
        var m = Move{
            .score = c.score,
            .dir = c.dir,
            .line = c.line,
            .start = c.start,
            .end = c.end,
            .placed = c.placed,
            .new_mask = c.new_mask,
        };
        var it = c.new_mask;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            m.tiles[i] = c.tiles[i];
        }
        return m;
    }
};

/// Generate every legal play in `pos`, calling `visitor.visit(*const Candidate)`.
pub fn generate(comptime V: type, visitor: *V, pos: *const Position, lex: *const lexicon.Lexicon, r: rules.Rules) void {
    std.debug.assert(r.rack <= N);
    var g = Generator(V){
        .visitor = visitor,
        .nodes = lex.nodes.ptr,
        .rack = r.rack,
        .max_place = @min(r.rack, pos.bag_total),
        .bag = pos.bag,
        .avail = 0,
    };
    for (0..26) |l| {
        if (pos.bag[l] != 0) g.avail |= @as(u32, 1) << @intCast(l);
    }
    g.bingo = @splat(0);
    g.bingo[r.rack] = r.bingo;
    if (g.max_place == 0) return;
    const root = lex.root();
    const first_play = pos.tiles_on_board == 0;
    for (0..2) |d| {
        for (0..N) |line| {
            if (first_play and line != rules.CENTER) continue;
            g.loadLine(pos, @intCast(d), line, first_play);
            var it = g.pivots;
            while (it != 0) : (it &= it - 1) {
                const p: usize = @ctz(it);
                g.setPivot(p);
                const c = g.cells[p];
                if (c == 0) {
                    g.fill(true, p, root, 0, 0);
                } else {
                    const ch = g.child(root, rules.letterOf(c));
                    if (lexicon.reach(ch) & g.needLeft(p, g.max_place) != 0) {
                        g.left(p, ch, @intCast(rules.cellValue(c)));
                    }
                }
            }
        }
    }
}

/// Running score of a partial play, packed into one register. Every field
/// only ever grows by addition, so placing a tile is a single add:
///   bits  0..15  main word letter points (after letter premiums)
///   bits 16..31  total of the perpendicular words formed
///   bits 32..39  tiles placed
///   bits 40..47  double-word squares covered
///   bits 48..55  triple-word squares covered
const Acc = u64;
const CROSS_SHIFT = 16;
const PLACED_SHIFT = 32;
const DW_SHIFT = 40;
const TW_SHIFT = 48;
const pow3 = [8]i32{ 1, 3, 9, 27, 81, 243, 729, 2187 };

inline fn placedOf(acc: Acc) usize {
    return @intCast((acc >> PLACED_SHIFT) & 0xff);
}

fn Generator(comptime V: type) type {
    return struct {
        const Self = @This();

        visitor: *V,
        nodes: [*]const u64,
        rack: u8,
        max_place: u8,
        bingo: [N + 1]i32 = undefined,
        bag: [27]u8,
        /// Bit l set while the bag still holds a natural tile l.
        avail: u32,

        // The current line.
        dir: u1 = 0,
        line: u8 = 0,
        /// Squares that hold a tile or have a perpendicular word.
        constrained: u16 = 0,
        /// Squares that can be the rightmost constrained square of a play.
        pivots: u16 = 0,
        cells: [N]u8 = undefined,
        xm: [N]u32 = undefined,
        /// Per-square pieces of a tile's score delta.
        lmul: [N]u64 = undefined,
        wmul: [N]u64 = undefined,
        xsum: [N]u64 = undefined,
        /// All ones where a perpendicular word passes through, else zero.
        xflag: [N]u64 = undefined,
        /// One placed tile plus the square's word premium, as an Acc delta.
        dbase: [N]u64 = undefined,
        /// Bit e set when a word may end at e (next square empty or edge).
        end_ok: u16 = 0,
        /// Bit s set when a word may start at s (previous square empty or edge).
        start_ok: u16 = 0,
        /// lfit[i][b]: distances d for which squares i-d .. i-1 hold at most b empties.
        lfit: [N][N + 1]u16 = undefined,
        /// rfit[i][b]: distances d for which squares i+1 .. i+d hold at most b empties.
        rfit: [N][N + 1]u16 = undefined,

        // The current pivot.
        pivot: usize = 0,
        pivot_right_free: bool = false,
        /// Last square the rightward walk may fill (the square before the
        /// next constrained one).
        hi: usize = 0,
        /// Usable word starts at or left of the pivot, bit-reversed: bit (15 - s).
        lstart_rev: u16 = 0,
        /// Usable word ends from the pivot to `hi`.
        rend: u16 = 0,

        tiles: [N]u8 = @splat(0),
        new_mask: u16 = 0,

        fn loadLine(self: *Self, pos: *const Position, d: u1, line: usize, first_play: bool) void {
            const base = line * N;
            self.dir = d;
            self.line = @intCast(line);
            self.cells = pos.cells[d][base..][0..N].*;
            var empty: u16 = 0;
            var cross: u16 = 0;
            for (0..N) |i| {
                const xm = pos.xmask[d][base + i];
                const wm = rules.premiums.word[d][base + i];
                self.xm[i] = xm;
                self.lmul[i] = rules.premiums.letter[d][base + i];
                self.wmul[i] = wm;
                self.xsum[i] = pos.xsum[d][base + i];
                self.xflag[i] = if (xm & board.HAS_CROSS != 0) ~@as(u64, 0) else 0;
                self.dbase[i] = (@as(u64, 1) << PLACED_SHIFT) | switch (wm) {
                    2 => @as(u64, 1) << DW_SHIFT,
                    3 => @as(u64, 1) << TW_SHIFT,
                    else => 0,
                };
                const bit = @as(u16, 1) << @intCast(i);
                if (self.cells[i] == 0) {
                    empty |= bit;
                    if (xm & board.HAS_CROSS != 0) cross |= bit;
                }
            }
            const occ = ~empty & ((1 << N) - 1);
            if (first_play) {
                // The centre square stands in for a constraint.
                self.constrained = 1 << rules.CENTER;
                self.pivots = 1 << rules.CENTER;
            } else {
                self.constrained = occ | cross;
                // A pivot's right neighbour cannot be a tile: the tile would
                // be a constrained square further right in the same word.
                self.pivots = self.constrained & ~(occ >> 1);
            }
            if (self.pivots == 0) return;
            const edge: u16 = 1 << (N - 1);
            self.end_ok = (empty >> 1) | edge;
            self.start_ok = (empty << 1 | 1) & ((1 << N) - 1);
            for (0..N) |i| {
                var dl: usize = 0;
                var cl: usize = 0;
                var dr: usize = 0;
                var cr: usize = 0;
                for (0..N + 1) |b| {
                    while (dl < i and cl + ((empty >> @intCast(i - dl - 1)) & 1) <= b) {
                        cl += (empty >> @intCast(i - dl - 1)) & 1;
                        dl += 1;
                    }
                    while (i + dr + 1 < N and cr + ((empty >> @intCast(i + dr + 1)) & 1) <= b) {
                        cr += (empty >> @intCast(i + dr + 1)) & 1;
                        dr += 1;
                    }
                    self.lfit[i][b] = @intCast((@as(u32, 2) << @intCast(dl)) - 1);
                    self.rfit[i][b] = @intCast((@as(u32, 2) << @intCast(dr)) - 1);
                }
            }
        }

        fn setPivot(self: *Self, p: usize) void {
            self.pivot = p;
            self.pivot_right_free = p + 1 == N or self.cells[p + 1] == 0;
            const upto: u16 = @intCast((@as(u32, 2) << @intCast(p)) - 1);
            const above = self.constrained & ~upto;
            self.hi = if (above == 0) N - 1 else @as(usize, @ctz(above)) - 1;
            const thru_hi: u16 = @intCast((@as(u32, 2) << @intCast(self.hi)) - 1);
            self.lstart_rev = @bitReverse(self.start_ok & upto);
            self.rend = self.end_ok & thru_hi & ~(upto >> 1);
        }

        /// Word lengths still to come, going left from i, that end at a usable start.
        inline fn needLeft(self: *const Self, i: usize, budget: usize) u16 {
            return (self.lstart_rev >> @intCast(15 - i)) & self.lfit[i][budget];
        }

        /// Word lengths still to come, going right from i, that end at a usable end.
        inline fn needRight(self: *const Self, i: usize, budget: usize) u16 {
            return (self.rend >> @intCast(i)) & self.rfit[i][budget];
        }

        /// Child of `n` for `sym`, or the dead record 0. Branch-free: the
        /// lexicon keeps a spare record past the end so the load is safe.
        inline fn child(self: *const Self, n: u64, sym: u5) u64 {
            const m: u32 = @truncate(n & ((1 << 27) - 1));
            const bit = @as(u32, 1) << sym;
            const rec = self.nodes[@as(usize, @intCast(n >> lexicon.BASE_SHIFT)) + @popCount(m & (bit - 1))];
            return if (m & bit != 0) rec else 0;
        }

        inline fn take(self: *Self, l: u5) void {
            self.bag[l] -= 1;
            self.avail &= ~(@as(u32, @intFromBool(self.bag[l] == 0)) << l);
        }

        inline fn give(self: *Self, l: u5) void {
            self.bag[l] += 1;
            self.avail |= @as(u32, 1) << l;
        }

        /// Acc delta for placing a tile worth `value` at square i.
        inline fn delta(self: *const Self, i: usize, value: u64) u64 {
            const lv = value * self.lmul[i];
            const cross = ((self.xsum[i] + lv) * self.wmul[i]) & self.xflag[i];
            return self.dbase[i] + lv + (cross << CROSS_SHIFT);
        }

        /// Try each letter on the empty square i and continue the walk.
        inline fn fill(self: *Self, comptime leftward: bool, i: usize, node: u64, acc: Acc, start: usize) void {
            const placed = placedOf(acc);
            if (placed >= self.max_place) return;
            const allowed = lexicon.letterMask(node) & self.xm[i];
            const blanks = self.bag[rules.BLANK_SLOT] != 0;
            const cand = if (blanks) allowed else allowed & self.avail;
            if (cand == 0) return;
            const budget = self.max_place - placed - 1;
            const need = if (leftward) self.needLeft(i, budget) else self.needRight(i, budget);

            // First pass, branch-free: which children can still finish a word?
            var kids: [26]u64 = undefined;
            var pass: u32 = 0;
            const m: u32 = @truncate(node & ((1 << 27) - 1));
            const base: usize = @intCast(node >> lexicon.BASE_SHIFT);
            var it = cand;
            while (it != 0) : (it &= it - 1) {
                const l: u5 = @intCast(@ctz(it));
                const ch = self.nodes[base + @popCount(m & ((@as(u32, 1) << l) - 1))];
                kids[l] = ch;
                pass |= @as(u32, @intFromBool(lexicon.reach(ch) & need != 0)) << l;
            }
            if (pass == 0) return;

            const bit = @as(u16, 1) << @intCast(i);
            self.new_mask |= bit;
            it = pass & self.avail;
            while (it != 0) : (it &= it - 1) {
                const l: u5 = @intCast(@ctz(it));
                self.take(l);
                self.tiles[i] = rules.cellOf(l, false);
                const a = acc + self.delta(i, rules.letter_values[l]);
                if (leftward) self.left(i, kids[l], a) else self.right(i, kids[l], a, start);
                self.give(l);
            }
            if (blanks) {
                self.bag[rules.BLANK_SLOT] -= 1;
                const a = acc + self.delta(i, 0);
                it = pass;
                while (it != 0) : (it &= it - 1) {
                    const l: u5 = @intCast(@ctz(it));
                    self.tiles[i] = rules.cellOf(l, true);
                    if (leftward) self.left(i, kids[l], a) else self.right(i, kids[l], a, start);
                }
                self.bag[rules.BLANK_SLOT] += 1;
            }
            self.tiles[i] = 0;
            self.new_mask &= ~bit;
        }

        /// Square i (at or left of the pivot) holds the walk's latest letter.
        fn left(self: *Self, i: usize, node: u64, acc: Acc) void {
            if (i > 0 and self.cells[i - 1] != 0) {
                // A tile to the left must join the word.
                const c = self.cells[i - 1];
                const ch = self.child(node, rules.letterOf(c));
                if (lexicon.reach(ch) & self.needLeft(i - 1, self.max_place - placedOf(acc)) == 0) return;
                self.left(i - 1, ch, acc + @as(u64, @intCast(rules.cellValue(c))));
                return;
            }
            // The word may start at i and end at the pivot ...
            if (self.pivot_right_free and lexicon.accepts(node) and placedOf(acc) != 0) {
                self.record(i, self.pivot, acc);
            }
            // ... or start at i and run on into the open squares right of the pivot ...
            if (self.pivot < self.hi) {
                const sep = self.child(node, lexicon.SEP);
                if (lexicon.reach(sep) & self.needRight(self.pivot, self.max_place - placedOf(acc)) != 0) {
                    self.fill(false, self.pivot + 1, sep, acc, i);
                }
            }
            // ... or keep growing left.
            if (i > 0) self.fill(true, i - 1, node, acc, 0);
        }

        /// Square i (an open square right of the pivot) holds the walk's latest letter.
        fn right(self: *Self, i: usize, node: u64, acc: Acc, start: usize) void {
            if (lexicon.accepts(node) and (self.end_ok >> @intCast(i)) & 1 != 0) self.record(start, i, acc);
            if (i < self.hi) self.fill(false, i + 1, node, acc, start);
        }

        inline fn record(self: *Self, start: usize, end: usize, acc: Acc) void {
            const placed = placedOf(acc);
            // A lone tile with an across neighbour is reported by the across pass.
            if (self.dir == 1 and placed == 1 and self.xm[@ctz(self.new_mask)] & board.HAS_CROSS != 0) return;
            const main: i32 = @intCast(acc & 0xffff);
            const cross: i32 = @intCast((acc >> CROSS_SHIFT) & 0xffff);
            const dw: u5 = @intCast((acc >> DW_SHIFT) & 0xff);
            const tw: usize = @intCast((acc >> TW_SHIFT) & 0xff);
            const cand = Candidate{
                .score = (main << dw) * pow3[tw] + cross + self.bingo[placed],
                .placed = @intCast(placed),
                .dir = self.dir,
                .line = self.line,
                .start = @intCast(start),
                .end = @intCast(end),
                .new_mask = self.new_mask,
                .tiles = &self.tiles,
            };
            self.visitor.visit(&cand);
        }
    };
}

/// Counts plays and sums their scores (a cheap checksum for benchmarks).
pub const Counter = struct {
    count: u64 = 0,
    score_sum: u64 = 0,

    pub inline fn visit(self: *Counter, c: *const Candidate) void {
        self.count += 1;
        self.score_sum += @intCast(c.score);
    }
};

/// Collects every play. Allocation failure is recorded and later plays dropped.
pub const Collector = struct {
    gpa: std.mem.Allocator,
    moves: std.ArrayList(Move) = .empty,
    failed: bool = false,

    pub fn visit(self: *Collector, c: *const Candidate) void {
        self.moves.append(self.gpa, c.toMove()) catch {
            self.failed = true;
        };
    }

    pub fn deinit(self: *Collector) void {
        self.moves.deinit(self.gpa);
    }
};

/// Keeps the highest-scoring play.
pub const Best = struct {
    best: Move = .{},
    found: bool = false,

    pub inline fn visit(self: *Best, c: *const Candidate) void {
        if (!self.found or c.score > self.best.score) {
            self.best = c.toMove();
            self.found = true;
        }
    }
};
