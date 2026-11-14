//! A direct port of the 2023 Python move generator (solver.py). It has two
//! jobs: an independent oracle for testing the GADDAG generator, and a
//! benchmark tier that separates the speedup from changing language from the
//! speedup from changing algorithm.
//!
//! The Python structure is kept: a "halo" of empty squares next to tiles,
//! cross-check sets found by scanning every word of the right length, one
//! constraint problem per (start square, direction, length), each solved by
//! filtering the whole word list of that length, then scored square by
//! square. Python sets of letters become 26-bit masks; nothing else is tuned.
//!
//! Two deliberate differences, both no-ops when blanks are off (as in the
//! Python reference): blanks are supported by trying every way of using
//! them for the new tiles, and a cross word counts when it exists rather than
//! when its face value is positive.

const std = @import("std");
const rules = @import("rules.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const Allocator = std.mem.Allocator;

const N = rules.N;
const NN = rules.NN;
const Position = board.Position;
const ALL = board.ALL_LETTERS;

/// Python's WORD_SETS: the lexicon split by word length.
pub const WordSets = struct {
    /// Words of length n, concatenated, as letter codes 0..25.
    by_len: [N + 1][]u8,

    pub fn init(gpa: Allocator, text: []const u8) !WordSets {
        // Like Python: upper-case every word and keep each once.
        const upper = try gpa.alloc(u8, text.len);
        defer gpa.free(upper);
        _ = std.ascii.upperString(upper, text);
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(gpa);
        var lists: [N + 1]std.ArrayList(u8) = @splat(.empty);
        errdefer for (&lists) |*l| l.deinit(gpa);
        var tok = std.mem.tokenizeAny(u8, upper, " \t\r\n");
        outer: while (tok.next()) |w| {
            if (w.len < 2 or w.len > N) continue;
            for (w) |ch| if (!std.ascii.isAlphabetic(ch)) continue :outer;
            const gop = try seen.getOrPut(gpa, w);
            if (gop.found_existing) continue;
            for (w) |ch| try lists[w.len].append(gpa, ch - 'A');
        }
        var ws: WordSets = undefined;
        for (&lists, 0..) |*l, n| ws.by_len[n] = try l.toOwnedSlice(gpa);
        return ws;
    }

    pub fn deinit(self: *WordSets, gpa: Allocator) void {
        for (self.by_len) |s| gpa.free(s);
        self.* = undefined;
    }
};

/// Python's halo plus halo_letters_across / halo_letters_down.
pub const Halo = struct {
    in_halo: [NN]bool,
    /// [orientation][square]: letters allowed by the across (0) or down (1)
    /// word through the square.
    letters: [2][NN]u32,
};

pub fn computeHalo(ws: *const WordSets, pos: *const Position) Halo {
    var h: Halo = .{ .in_halo = @splat(false), .letters = @splat(@splat(ALL)) };
    if (pos.tiles_on_board == 0) {
        h.in_halo[rules.CENTER * N + rules.CENTER] = true;
        return h;
    }
    for (0..N) |r| {
        for (0..N) |c| {
            if (pos.at(r, c) != 0) continue;
            const near = (r > 0 and pos.at(r - 1, c) != 0) or (r + 1 < N and pos.at(r + 1, c) != 0) or
                (c > 0 and pos.at(r, c - 1) != 0) or (c + 1 < N and pos.at(r, c + 1) != 0);
            if (!near) continue;
            h.in_halo[r * N + c] = true;
            h.letters[0][r * N + c] = crossLetters(ws, pos, r, c, 0);
            h.letters[1][r * N + c] = crossLetters(ws, pos, r, c, 1);
        }
    }
    return h;
}

/// get_cross_letters: letters that complete the word through (r, c) along
/// `orient` (0 = across, 1 = down), found by scanning the word list.
fn crossLetters(ws: *const WordSets, pos: *const Position, r: usize, c: usize, orient: u1) u32 {
    const line = pos.cells[orient][(if (orient == 0) r else c) * N ..][0..N];
    const i = if (orient == 0) c else r;
    var a = i;
    while (a > 0 and line[a - 1] != 0) a -= 1;
    var b = i;
    while (b + 1 < N and line[b + 1] != 0) b += 1;
    if (a == i and b == i) return ALL;
    const left = line[a..i];
    const right = line[i + 1 .. b + 1];
    const len = b + 1 - a;
    const words = ws.by_len[len];
    var out: u32 = 0;
    var k: usize = 0;
    while (k < words.len) : (k += len) {
        const w = words[k..][0..len];
        if (matches(w[0..left.len], left) and matches(w[left.len + 1 ..], right)) {
            out |= @as(u32, 1) << @intCast(w[left.len]);
        }
    }
    return out;
}

fn matches(w: []const u8, cells: []const u8) bool {
    for (w, cells) |l, c| {
        if (l != rules.letterOf(c)) return false;
    }
    return true;
}

/// One constraint problem: a slot of `len` squares from (row, col).
pub const Csp = struct {
    row: u8,
    col: u8,
    dir: u1,
    len: u8,
    cons: [N]u32,
};

/// get_csps. A slot may hold at most `max_new` empty squares.
pub fn collectCsps(gpa: Allocator, pos: *const Position, halo: *const Halo, max_new: usize, out: *std.ArrayList(Csp)) !void {
    for (0..2) |dd| {
        const d: u1 = @intCast(dd);
        for (0..N) |row| {
            for (0..N) |col| {
                if (d == 0 and col > 0 and pos.at(row, col - 1) != 0) continue;
                if (d == 1 and row > 0 and pos.at(row - 1, col) != 0) continue;
                var p = Csp{ .row = @intCast(row), .col = @intCast(col), .dir = d, .len = 0, .cons = @splat(ALL) };
                var new_count: usize = 0;
                var halo_hit = false;
                var x = row;
                var y = col;
                var i: usize = 0;
                while (x < N and y < N) {
                    const cell = pos.at(x, y);
                    if (cell != 0) {
                        p.cons[i] = @as(u32, 1) << rules.letterOf(cell);
                    } else if (halo.in_halo[x * N + y]) {
                        halo_hit = true;
                        p.cons[i] = halo.letters[1 - d][x * N + y];
                    }
                    if (cell == 0) new_count += 1;
                    if (d == 0) y += 1 else x += 1;
                    i += 1;
                    const next_is_tile = x < N and y < N and pos.at(x, y) != 0;
                    if (!next_is_tile and halo_hit and new_count >= 1 and new_count <= max_new and i > 1) {
                        p.len = @intCast(i);
                        try out.append(gpa, p);
                    }
                }
            }
        }
    }
}

/// solve_csp + calculate_score for every problem; plays go to `visitor`.
pub fn generate(
    comptime V: type,
    visitor: *V,
    gpa: Allocator,
    ws: *const WordSets,
    pos: *const Position,
    halo: *const Halo,
    r: rules.Rules,
    sup: movegen.Supply,
) !void {
    var csps: std.ArrayList(Csp) = .empty;
    defer csps.deinit(gpa);
    try collectCsps(gpa, pos, halo, @min(r.rack, sup.max_tiles), &csps);
    for (csps.items) |*p| {
        const n: usize = p.len;
        const words = ws.by_len[n];
        var k: usize = 0;
        next: while (k < words.len) : (k += n) {
            const w = words[k..][0..n];
            for (w, 0..) |l, i| {
                if ((p.cons[i] >> @intCast(l)) & 1 == 0) continue :next;
            }
            emitScored(V, visitor, pos, p, w, r, &sup.counts);
        }
    }
}

/// calculate_score, generalised to try each way of covering new tiles with blanks.
fn emitScored(comptime V: type, visitor: *V, pos: *const Position, p: *const Csp, w: []const u8, r: rules.Rules, supply: *const [27]u8) void {
    const d = p.dir;
    const line: usize = if (d == 0) p.row else p.col;
    const start: usize = if (d == 0) p.col else p.row;
    var new_idx: [N]u8 = undefined;
    var k: usize = 0;
    for (0..w.len) |i| {
        if (pos.cells[d][line * N + start + i] == 0) {
            new_idx[k] = @intCast(i);
            k += 1;
        }
    }
    const blanks = supply[rules.BLANK_SLOT];
    const subsets = @as(u32, 1) << @intCast(k);
    var s: u32 = 0;
    while (s < subsets) : (s += 1) {
        if (@popCount(s) > blanks) continue;
        // Natural tiles must be in the supply.
        var left = supply.*;
        var ok = true;
        for (0..k) |j| {
            if ((s >> @intCast(j)) & 1 != 0) continue;
            const l = w[new_idx[j]];
            if (left[l] == 0) {
                ok = false;
                break;
            }
            left[l] -= 1;
        }
        if (!ok) continue;

        var tiles: [N]u8 = @splat(0);
        var new_mask: u16 = 0;
        for (0..k) |j| {
            const i = start + new_idx[j];
            tiles[i] = rules.cellOf(@intCast(w[new_idx[j]]), (s >> @intCast(j)) & 1 != 0);
            new_mask |= @as(u16, 1) << @intCast(i);
        }

        var main: i32 = 0;
        var wmul: i32 = 1;
        var cross_total: i32 = 0;
        for (0..w.len) |i| {
            const li = start + i;
            const idx = line * N + li;
            const existing = pos.cells[d][idx];
            if (existing != 0) {
                main += rules.cellValue(existing);
                continue;
            }
            const lm: i32 = rules.premiums.letter[d][idx];
            const pw: i32 = rules.premiums.word[d][idx];
            wmul *= pw;
            const ps = rules.cellValue(tiles[li]) * lm;
            main += ps;
            // get_cross_word_value along the perpendicular.
            const run = pos.cells[1 - d][li * N ..][0..N];
            var a = line;
            while (a > 0 and run[a - 1] != 0) a -= 1;
            var b = line;
            while (b + 1 < N and run[b + 1] != 0) b += 1;
            if (a == line and b == line) continue;
            var cs: i32 = 0;
            for (a..b + 1) |q| {
                if (q != line) cs += rules.cellValue(run[q]);
            }
            cross_total += (cs + ps) * pw;
        }
        const bingo: i32 = if (k == r.rack) r.bingo else 0;
        const cand = movegen.Candidate{
            .score = main * wmul + cross_total + bingo,
            .placed = @intCast(k),
            .dir = d,
            .line = @intCast(line),
            .start = @intCast(start),
            .end = @intCast(start + w.len - 1),
            .new_mask = new_mask,
            .tiles = &tiles,
        };
        visitor.visit(&cand);
    }
}
