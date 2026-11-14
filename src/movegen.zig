//! GADDAG move generation (Gordon 1994) for plays drawn from a large supply
//! of tiles (with chosen draws, any unplayed tile can be on the rack).
//!
//! Every play is walked from one pivot square: the rightmost square of its
//! main word that is constrained, meaning it holds a tile or has a
//! perpendicular word through it. From the pivot we walk the GADDAG
//! leftwards, then (after the separator) rightwards. Everything right of the
//! pivot is open board, so all constraints are met before the walk enters
//! unconstrained squares. Each play has exactly one pivot, so it is produced
//! once. (The usual leftmost-anchor rule enumerates open space first, which
//! is cheap with a 7-tile rack but wasteful with this many tiles to choose
//! from.)
//!
//! At each empty square the letters worth trying are one AND away:
//!
//!     candidates = node.children & square.cross_check & supply.non_empty
//!
//! and a child is entered only if its reach mask says some word through it
//! fits the free squares and the tiles left to place. Scores are
//! accumulated during the walk, so a play is scored in O(1) when recorded,
//! and plays go to a comptime visitor that can count, collect, or sample
//! them without storing every play. A one-tile play that forms words both
//! ways is reported only by the across pass.
//!
//! Blanks are not walked. The walk chooses letters only, and counts how
//! many of them the natural tiles cannot cover. When a play is recorded,
//! each legal way of making some of its tiles blanks is reported, with the
//! points each blank loses subtracted. Walking blank choices directly
//! repeats the same GADDAG paths for every subset of blank positions.

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
    /// Line positions of new tiles that are blanks; `tiles` holds natural cells.
    blanks: u16 = 0,
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
            m.tiles[i] = c.tiles[i] | (if ((c.blanks >> @intCast(i)) & 1 != 0) rules.BLANK_FLAG else 0);
        }
        return m;
    }
};

pub const BlankMode = enum {
    /// Every legal way of using blanks: the full move set.
    any,
    /// Blanks only where no natural tile is left, and no more than needed.
    as_needed,
};

/// The tiles a play may be made from.
pub const Supply = struct {
    /// Tiles by letter; index 26 counts blanks.
    counts: [27]u8,
    /// Most tiles one play may place (the mover's rack size).
    max_tiles: u8,
    blanks: BlankMode = .any,
};

/// Generate every legal play in `pos` from `sup`, calling
/// `visitor.visit(*const Candidate)` once per play.
pub fn generate(
    comptime V: type,
    visitor: *V,
    pos: *const Position,
    lex: *const lexicon.Lexicon,
    r: rules.Rules,
    sup: Supply,
) void {
    if (sup.counts[rules.BLANK_SLOT] != 0) {
        run(V, true, visitor, pos, lex, r, sup);
    } else {
        run(V, false, visitor, pos, lex, r, sup);
    }
}

fn run(
    comptime V: type,
    comptime with_blanks: bool,
    visitor: *V,
    pos: *const Position,
    lex: *const lexicon.Lexicon,
    r: rules.Rules,
    sup: Supply,
) void {
    std.debug.assert(r.rack <= N);
    const max_place = @min(r.rack, sup.max_tiles);
    if (max_place == 0) return;
    var g = Generator(V, with_blanks){
        .visitor = visitor,
        .nodes = lex.nodes.ptr,
        .rack = r.rack,
        .max_place = max_place,
        .blanks = sup.counts[rules.BLANK_SLOT],
        .only_needed = sup.blanks == .as_needed,
    };
    for (0..26) |l| {
        g.left[l] = @intCast(sup.counts[l]);
        if (sup.counts[l] != 0) g.avail |= @as(u32, 1) << @intCast(l);
    }
    g.bingo[r.rack] = r.bingo;
    const root = lex.root();
    // The board is symmetric about its diagonal, so on the empty board every
    // down play mirrors an across play with the same score: walk across
    // only and report each play in both directions.
    const first_play = pos.tiles_on_board == 0;
    g.mirror = first_play;
    for (0..@as(usize, if (first_play) 1 else 2)) |d| {
        for (0..N) |line| {
            if (first_play and line != rules.CENTER) continue;
            g.loadLine(pos, @intCast(d), line, first_play);
            var it = g.pivots;
            while (it != 0) : (it &= it - 1) {
                const p: usize = @ctz(it);
                g.setPivot(p);
                const c = g.cells[p];
                if (c == 0) {
                    g.fillLeft(p, root, 0);
                } else {
                    // The pivot ends a run of tiles; afterLeft reads the rest.
                    g.afterLeft(p, g.child(root, rules.letterOf(c)), @intCast(rules.cellValue(c)));
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

/// Index pairs (j1 < j2), ordered by j2 then j1, so the pairs among the
/// first k items are exactly the first k(k-1)/2 entries.
const pairs: [N * (N - 1) / 2][2]u8 = blk: {
    var out: [N * (N - 1) / 2][2]u8 = undefined;
    var n: usize = 0;
    for (1..N) |j2| {
        for (0..j2) |j1| {
            out[n] = .{ j1, j2 };
            n += 1;
        }
    }
    break :blk out;
};

inline fn placedOf(acc: Acc) usize {
    return @intCast((acc >> PLACED_SHIFT) & 0xff);
}

fn Generator(comptime V: type, comptime with_blanks: bool) type {
    return struct {
        const Self = @This();

        visitor: *V,
        nodes: [*]const u64,
        rack: u8,
        max_place: u8,
        bingo: [N + 1]i32 = @splat(0),
        /// Natural tiles of each letter not used by the current partial play;
        /// negative when blanks stand in.
        left: [26]i8 = @splat(0),
        /// Bit l set while left[l] > 0. Kept up to date bit by bit: reading
        /// `left` as a vector right after a one-byte store would stall on
        /// store forwarding.
        avail: u32 = 0,
        blanks: u8,
        /// Tiles in the current partial play that must be blanks.
        deficit: u8 = 0,
        only_needed: bool,

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
        /// dtab[i][l]: the whole Acc delta for a natural tile l on square i.
        dtab: [N][26]u64 = undefined,
        /// Bit e set when a word may end at e (next square empty or edge).
        end_ok: u16 = 0,
        /// Bit s set when a word may start at s (previous square empty or edge).
        start_ok: u16 = 0,
        /// lfit[i][b]: distances d for which squares i-d .. i-1 hold at most b empties.
        lfit: [N][N + 1]u16 = undefined,
        /// rfit[i][b]: distances d for which squares i+1 .. i+d hold at most b empties.
        rfit: [N][N + 1]u16 = undefined,
        /// For the current pivot, needl[i][b] / needr[i][b]: the distances to
        /// a usable word start / end going left / right from square i with b
        /// tiles still to place (see needLeft / needRight).
        needl: [N][N + 1]u16 = undefined,
        needr: [N][N + 1]u16 = undefined,

        // The current pivot.
        pivot: usize = 0,
        pivot_right_free: bool = false,
        /// Last square the rightward walk may fill (the square before the
        /// next constrained one).
        hi: usize = 0,

        tiles: [N]u8 = @splat(0),
        new_mask: u16 = 0,
        /// Report each play again as its mirror image (empty board only).
        mirror: bool = false,

        /// Hand a play to the visitor (and its mirror image, on the empty board).
        inline fn emit(self: *Self, cand: *Candidate) void {
            self.visitor.visit(cand);
            if (self.mirror) {
                cand.dir = 1;
                self.visitor.visit(cand);
                cand.dir = 0;
            }
        }

        fn loadLine(self: *Self, pos: *const Position, d: u1, line: usize, first_play: bool) void {
            const base = line * N;
            self.dir = d;
            self.line = @intCast(line);
            self.cells = pos.cells[d][base..][0..N].*;
            var empty: u16 = 0;
            var cross: u16 = 0;
            for (0..N) |i| {
                const xm = pos.xmask[d][base + i];
                self.xm[i] = xm;
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
            for (0..N) |i| {
                const wm = rules.premiums.word[d][base + i];
                self.lmul[i] = rules.premiums.letter[d][base + i];
                self.wmul[i] = wm;
                self.xsum[i] = pos.xsum[d][base + i];
                self.xflag[i] = if (self.xm[i] & board.HAS_CROSS != 0) ~@as(u64, 0) else 0;
                self.dbase[i] = (@as(u64, 1) << PLACED_SHIFT) | switch (wm) {
                    2 => @as(u64, 1) << DW_SHIFT,
                    3 => @as(u64, 1) << TW_SHIFT,
                    else => 0,
                };
            }
            var e = empty;
            while (e != 0) : (e &= e - 1) {
                const i: usize = @ctz(e);
                for (0..26) |l| self.dtab[i][l] = self.delta(i, rules.letter_values[l]);
            }
            const edge: u16 = 1 << (N - 1);
            self.end_ok = (empty >> 1) | edge;
            self.start_ok = (empty << 1 | 1) & ((1 << N) - 1);
            // Only budgets up to max_place are ever asked for.
            for (0..N) |i| {
                var dl: usize = 0;
                var cl: usize = 0;
                var dr: usize = 0;
                var cr: usize = 0;
                for (0..@as(usize, self.max_place) + 1) |b| {
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
            const lstart_rev = @bitReverse(self.start_ok & upto);
            const rend = self.end_ok & thru_hi & ~(upto >> 1);
            for (0..p + 1) |i| {
                for (0..@as(usize, self.max_place) + 1) |b| {
                    self.needl[i][b] = (lstart_rev >> @intCast(15 - i)) & self.lfit[i][b];
                }
            }
            for (p..self.hi + 1) |i| {
                for (0..@as(usize, self.max_place) + 1) |b| {
                    self.needr[i][b] = (rend >> @intCast(i)) & self.rfit[i][b];
                }
            }
        }

        /// Distances d (going left from i) to a usable word start that the
        /// remaining `budget` of tiles can reach.
        inline fn needLeft(self: *const Self, i: usize, budget: usize) u16 {
            return self.needl[i][budget];
        }

        /// Distances d (going right from i) to a usable word end that the
        /// remaining `budget` of tiles can reach.
        inline fn needRight(self: *const Self, i: usize, budget: usize) u16 {
            return self.needr[i][budget];
        }

        /// Child of `n` for `sym`, or the dead record 0. Branch-free: the
        /// lexicon keeps a spare record past the end so the load is safe.
        inline fn child(self: *const Self, n: u64, sym: u5) u64 {
            const m: u32 = @truncate(n & ((1 << 27) - 1));
            const bit = @as(u32, 1) << sym;
            const rec = self.nodes[@as(usize, @intCast(n >> lexicon.BASE_SHIFT)) + @popCount(m & (bit - 1))];
            return if (m & bit != 0) rec else 0;
        }

        inline fn use(self: *Self, l: u5) void {
            const v = self.left[l] - 1;
            self.left[l] = v;
            self.avail &= ~(@as(u32, @intFromBool(v <= 0)) << l);
            if (with_blanks) self.deficit += @intFromBool(v < 0);
        }

        inline fn unuse(self: *Self, l: u5) void {
            const v = self.left[l];
            if (with_blanks) self.deficit -= @intFromBool(v < 0);
            self.left[l] = v + 1;
            self.avail |= @as(u32, @intFromBool(v + 1 > 0)) << l;
        }

        /// Letters worth trying on empty square i under GADDAG node `node`.
        inline fn candidates(self: *const Self, node: u64, i: usize) u32 {
            const allowed = lexicon.letterMask(node) & self.xm[i];
            if (with_blanks and self.deficit < self.blanks) return allowed;
            return allowed & self.avail;
        }

        /// Load the children of `node` for `cand` into `kids` and return the
        /// letters whose child can still finish a word within `need`.
        /// Branch-free apart from the loop itself.
        inline fn screen(self: *const Self, node: u64, cand: u32, need: u16, kids: *[26]u64) u32 {
            const m: u32 = @truncate(node & ((1 << 27) - 1));
            const base: usize = @intCast(node >> lexicon.BASE_SHIFT);
            const want = lexicon.needBits(need);
            var pass: u32 = 0;
            var it = cand;
            while (it != 0) : (it &= it - 1) {
                const l: u5 = @intCast(@ctz(it));
                const ch = self.nodes[base + @popCount(m & ((@as(u32, 1) << l) - 1))];
                kids[l] = ch;
                pass |= @as(u32, @intFromBool(ch & want != 0)) << l;
            }
            return pass;
        }

        /// Acc delta for placing a tile worth `value` at square i.
        inline fn delta(self: *const Self, i: usize, value: u64) u64 {
            const lv = value * self.lmul[i];
            const cross = ((self.xsum[i] + lv) * self.wmul[i]) & self.xflag[i];
            return self.dbase[i] + lv + (cross << CROSS_SHIFT);
        }

        /// Place each worthwhile letter on the empty square i (at or left of
        /// the pivot) and continue the walk. The caller has checked that a
        /// tile can still be placed.
        fn fillLeft(self: *Self, i: usize, node: u64, acc: Acc) void {
            const cand = self.candidates(node, i);
            if (cand == 0) return;
            const need = self.needLeft(i, self.max_place - placedOf(acc) - 1);
            var kids: [26]u64 = undefined;
            const pass = self.screen(node, cand, need, &kids);
            if (pass == 0) return;
            const bit = @as(u16, 1) << @intCast(i);
            self.new_mask |= bit;
            var it = pass;
            while (it != 0) : (it &= it - 1) {
                const l: u5 = @intCast(@ctz(it));
                self.use(l);
                self.tiles[i] = rules.cellOf(l, false);
                self.afterLeft(i, kids[l], acc + self.dtab[i][l]);
                self.unuse(l);
            }
            self.tiles[i] = 0;
            self.new_mask &= ~bit;
        }

        /// Square i (at or left of the pivot) holds the walk's latest letter.
        inline fn afterLeft(self: *Self, at: usize, at_node: u64, at_acc: Acc) void {
            var i = at;
            var node = at_node;
            var acc = at_acc;
            // A run of tiles to the left must join the word.
            while (i > 0 and self.cells[i - 1] != 0) {
                i -= 1;
                node = self.child(node, rules.letterOf(self.cells[i]));
                acc += @intCast(rules.cellValue(self.cells[i]));
            }
            const budget = self.max_place - placedOf(acc);
            const need = self.needLeft(i, budget);
            if (node & lexicon.needBits(need) == 0) return;
            // The word may start at i and end at the pivot ...
            if (self.pivot_right_free and lexicon.accepts(node) and placedOf(acc) != 0) {
                self.record(i, self.pivot, acc);
            }
            // ... or start at i and run on into the open squares right of the pivot ...
            if (self.pivot < self.hi) {
                const sep = self.child(node, lexicon.SEP);
                if (sep & lexicon.needBits(self.needRight(self.pivot, budget)) != 0) {
                    self.fillRight(self.pivot + 1, sep, acc, i);
                }
            }
            // ... or start further left.
            if (node & lexicon.needBits(need & ~@as(u16, 1)) != 0) self.fillLeft(i - 1, node, acc);
        }

        /// Place each worthwhile letter on the open square i (right of the
        /// pivot), record the words that end there, and continue right.
        fn fillRight(self: *Self, i: usize, node: u64, acc: Acc, start: usize) void {
            const cand = self.candidates(node, i);
            if (cand == 0) return;
            const need = self.needRight(i, self.max_place - placedOf(acc) - 1);
            var kids: [26]u64 = undefined;
            const pass = self.screen(node, cand, need, &kids);
            if (pass == 0) return;
            const ends_here = need & 1 != 0;
            const onward = need & ~@as(u16, 1);
            const onward_bits = lexicon.needBits(onward);
            const bit = @as(u16, 1) << @intCast(i);
            self.new_mask |= bit;
            var it = pass;
            if (onward == 0) {
                // No word can go past this square, so `need` is just "ends
                // here" and every passing child is a finished play. Only the
                // blank expansion still looks at the letter counts.
                while (it != 0) : (it &= it - 1) {
                    const l: u5 = @intCast(@ctz(it));
                    self.tiles[i] = rules.cellOf(l, false);
                    if (with_blanks) {
                        self.left[l] -= 1;
                        self.deficit += @intFromBool(self.left[l] < 0);
                    }
                    self.record(start, i, acc + self.dtab[i][l]);
                    if (with_blanks) {
                        self.deficit -= @intFromBool(self.left[l] < 0);
                        self.left[l] += 1;
                    }
                }
            } else {
                while (it != 0) : (it &= it - 1) {
                    const l: u5 = @intCast(@ctz(it));
                    const ch = kids[l];
                    self.use(l);
                    self.tiles[i] = rules.cellOf(l, false);
                    const a = acc + self.dtab[i][l];
                    if (ends_here and lexicon.accepts(ch)) self.record(start, i, a);
                    if (ch & onward_bits != 0) self.fillRight(i + 1, ch, a, start);
                    self.unuse(l);
                }
            }
            self.tiles[i] = 0;
            self.new_mask &= ~bit;
        }

        inline fn record(self: *Self, start: usize, end: usize, acc: Acc) void {
            const placed = placedOf(acc);
            // A lone tile with an across neighbour is reported by the across pass.
            if (self.dir == 1 and placed == 1 and self.xm[@ctz(self.new_mask)] & board.HAS_CROSS != 0) return;
            const main: i32 = @intCast(acc & 0xffff);
            const cross: i32 = @intCast((acc >> CROSS_SHIFT) & 0xffff);
            const dw: u5 = @intCast((acc >> DW_SHIFT) & 0xff);
            const tw: usize = @intCast((acc >> TW_SHIFT) & 0xff);
            const word_mult = (@as(i32, 1) << dw) * pow3[tw];
            var cand = Candidate{
                .score = main * word_mult + cross + self.bingo[placed],
                .placed = @intCast(placed),
                .dir = self.dir,
                .line = self.line,
                .start = @intCast(start),
                .end = @intCast(end),
                .new_mask = self.new_mask,
                .tiles = &self.tiles,
            };
            if (with_blanks) self.expandBlanks(&cand, word_mult) else self.emit(&cand);
        }

        /// Report every legal way of making some of the play's tiles blanks.
        /// Tiles whose letter the natural supply cannot cover must be blanks;
        /// with `.any`, other tiles may be blanks too, up to the blanks there are.
        fn expandBlanks(self: *Self, cand: *Candidate, word_mult: i32) void {
            var at: [N]u8 = undefined;
            var letter: [N]u5 = undefined;
            // Points lost if that tile is a blank: its letter points in the
            // main word, and in its perpendicular word if it makes one.
            var loss: [N]i32 = undefined;
            var k: usize = 0;
            var it = self.new_mask;
            while (it != 0) : (it &= it - 1) {
                const i: usize = @ctz(it);
                const l = rules.letterOf(self.tiles[i]);
                const lv = @as(i32, rules.letter_values[l]) * @as(i32, @intCast(self.lmul[i]));
                var lost = lv * word_mult;
                if (self.xflag[i] != 0) lost += lv * @as(i32, @intCast(self.wmul[i]));
                at[k] = @intCast(i);
                letter[k] = l;
                loss[k] = lost;
                k += 1;
            }
            // Letters needing a blank, with repeats.
            var short: [N]u5 = undefined;
            var ns: usize = 0;
            if (self.deficit != 0) {
                var seen: u32 = 0;
                for (letter[0..k]) |l| {
                    if ((seen >> l) & 1 != 0) continue;
                    seen |= @as(u32, 1) << l;
                    var d: i32 = -@as(i32, self.left[l]);
                    while (d > 0) : (d -= 1) {
                        short[ns] = l;
                        ns += 1;
                    }
                }
            }
            const base_score = cand.score;
            const most = @min(self.blanks, k);
            if (most > 2) return self.expandMany(cand, at[0..k], letter[0..k], loss[0..k], short[0..ns], most);
            // At most two blanks: the common case, unrolled.
            if (ns == 0) self.emit(cand);
            if (most >= 1 and ns <= 1 and !(self.only_needed and ns != 1)) {
                for (0..k) |j| {
                    if (ns == 1 and letter[j] != short[0]) continue;
                    cand.blanks = @as(u16, 1) << @intCast(at[j]);
                    cand.score = base_score - loss[j];
                    self.emit(cand);
                }
            }
            if (most >= 2 and !(self.only_needed and ns != 2)) {
                // All pairs of the k tiles: the first k(k-1)/2 entries of `pairs`.
                const all = pairs[0 .. k * (k - 1) / 2];
                switch (ns) {
                    0 => for (all) |pr| self.visitTwoBlanks(cand, base_score, at[pr[0]], at[pr[1]], loss[pr[0]] + loss[pr[1]]),
                    1 => for (all) |pr| {
                        if (letter[pr[0]] != short[0] and letter[pr[1]] != short[0]) continue;
                        self.visitTwoBlanks(cand, base_score, at[pr[0]], at[pr[1]], loss[pr[0]] + loss[pr[1]]);
                    },
                    else => for (all) |pr| {
                        const a = letter[pr[0]];
                        const b = letter[pr[1]];
                        if (!((a == short[0] and b == short[1]) or (a == short[1] and b == short[0]))) continue;
                        self.visitTwoBlanks(cand, base_score, at[pr[0]], at[pr[1]], loss[pr[0]] + loss[pr[1]]);
                    },
                }
            }
            cand.score = base_score;
            cand.blanks = 0;
        }

        inline fn visitTwoBlanks(self: *Self, cand: *Candidate, base_score: i32, a: u8, b: u8, lost: i32) void {
            cand.blanks = (@as(u16, 1) << @intCast(a)) | (@as(u16, 1) << @intCast(b));
            cand.score = base_score - lost;
            self.emit(cand);
        }

        /// The general case for tile sets with more than two blanks.
        fn expandMany(self: *Self, cand: *Candidate, at: []const u8, letter: []const u5, loss: []const i32, short: []const u5, most: usize) void {
            const base_score = cand.score;
            const k = at.len;
            var mask: u32 = 0;
            while (mask < (@as(u32, 1) << @intCast(k))) : (mask += 1) {
                const size = @popCount(mask);
                if (size > most or size < short.len) continue;
                if (self.only_needed and size != short.len) continue;
                // Every letter in `short` needs its own blank of that letter.
                var covered = true;
                var used_by_short: u32 = 0;
                for (short) |l| {
                    var found = false;
                    for (0..k) |j| {
                        const b = @as(u32, 1) << @intCast(j);
                        if (mask & b != 0 and used_by_short & b == 0 and letter[j] == l) {
                            used_by_short |= b;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        covered = false;
                        break;
                    }
                }
                if (!covered) continue;
                var score = base_score;
                var blanks: u16 = 0;
                for (0..k) |j| {
                    if ((mask >> @intCast(j)) & 1 != 0) {
                        blanks |= @as(u16, 1) << @intCast(at[j]);
                        score -= loss[j];
                    }
                }
                cand.score = score;
                cand.blanks = blanks;
                self.emit(cand);
            }
            cand.score = base_score;
            cand.blanks = 0;
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
