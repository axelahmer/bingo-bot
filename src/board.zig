//! Board state: the board stored in both orientations (so every line, across
//! or down, is a contiguous 15-byte slice), per-square cross-check sets, and
//! the pool of unplayed tiles (the bag plus every rack). Turns, racks and
//! the end of the game live in game.zig.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const Lexicon = lexicon.Lexicon;
const Io = std.Io;

const N = rules.N;
const NN = rules.NN;

/// Set in `xmask` when a perpendicular word passes through the square.
pub const HAS_CROSS: u32 = 1 << 26;
pub const ALL_LETTERS: u32 = lexicon.LETTERS;

pub const Move = struct {
    score: i32 = 0,
    /// 0 = across (line is a row), 1 = down (line is a column).
    dir: u1 = 0,
    line: u8 = 0,
    /// Extent of the main word along the line, inclusive.
    start: u8 = 0,
    end: u8 = 0,
    placed: u8 = 0,
    /// Bit i set when the tile at line position i is new.
    new_mask: u16 = 0,
    /// Cells of the new tiles, by line position. Other entries are 0.
    tiles: [N]u8 = @splat(0),

    pub fn row(m: *const Move, i: usize) usize {
        return if (m.dir == 0) m.line else i;
    }

    pub fn col(m: *const Move, i: usize) usize {
        return if (m.dir == 0) i else m.line;
    }

    /// Sortable identity of a move: its new tiles as (square << 6 | cell),
    /// in board order. Two moves with the same key place the same tiles.
    pub fn key(m: *const Move) [8]u16 {
        var k: [8]u16 = @splat(0xffff);
        var n: usize = 0;
        var it = m.new_mask;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            const sq = m.row(i) * N + m.col(i);
            k[n] = @intCast((sq << 6) | m.tiles[i]);
            n += 1;
        }
        return k;
    }

    /// "RR.CC.L RR.CC.L ...<TAB>score", the format bench/py_reference.py prints.
    pub fn writeCanonical(m: *const Move, w: *Io.Writer) !void {
        var it = m.new_mask;
        var first = true;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            if (!first) try w.writeByte(' ');
            first = false;
            try w.print("{d:0>2}.{d:0>2}.{c}", .{ m.row(i), m.col(i), cellChar(m.tiles[i]) });
        }
        try w.print("\t{d}\n", .{m.score});
    }
};

pub fn cellChar(c: u8) u8 {
    if (c == 0) return '.';
    const ch = @as(u8, rules.letterOf(c)) + 'A';
    return if (rules.isBlank(c)) std.ascii.toLower(ch) else ch;
}

pub const PlayError = error{
    OffBoard,
    BadWord,
    LetterMismatch,
    NotDelimited,
    NoNewTiles,
    TooManyTiles,
    NotInBag,
    NotConnected,
    NotAWord,
    BadCrossWord,
};

pub const Position = struct {
    /// [dir][line * N + i]; cells[1] is the transpose of cells[0].
    cells: [2][NN]u8,
    /// Letters allowed on an empty square for a play in direction `dir`,
    /// given the perpendicular word through it, plus the HAS_CROSS flag.
    xmask: [2][NN]u32,
    /// Face value of the tiles in that perpendicular word.
    xsum: [2][NN]u16,
    /// Unplayed tiles (bag and racks) by letter; index 26 counts blanks.
    pool: [27]u8,
    pool_total: u8,
    initial_total: u8,
    tiles_on_board: u8,
    score: i32,
    plies: u16,

    pub fn init(r: rules.Rules) Position {
        var p: Position = undefined;
        p.cells = @splat(@splat(0));
        p.xmask = @splat(@splat(ALL_LETTERS));
        p.xsum = @splat(@splat(0));
        p.pool = rules.standard_counts;
        p.pool[rules.BLANK_SLOT] = r.blanks;
        var total: u32 = 0;
        for (p.pool) |n| total += n;
        p.pool_total = @intCast(total);
        p.initial_total = p.pool_total;
        p.tiles_on_board = 0;
        p.score = 0;
        p.plies = 0;
        return p;
    }

    pub inline fn at(self: *const Position, r: usize, c: usize) u8 {
        return self.cells[0][r * N + c];
    }

    /// Anchor squares of one line: empty squares next to a tile, or the
    /// centre on an empty board. Every legal play covers at least one.
    pub fn anchors(self: *const Position, d: u1, line: usize) u16 {
        if (self.tiles_on_board == 0) {
            return if (line == rules.CENTER) 1 << rules.CENTER else 0;
        }
        const row = self.cells[d][line * N ..][0..N];
        const xm = self.xmask[d][line * N ..][0..N];
        var occ: u16 = 0;
        var cross: u16 = 0;
        for (0..N) |i| {
            if (row[i] != 0) occ |= @as(u16, 1) << @intCast(i);
            if (xm[i] & HAS_CROSS != 0) cross |= @as(u16, 1) << @intCast(i);
        }
        const full: u16 = (1 << N) - 1;
        return ~occ & full & ((occ << 1) | (occ >> 1) | cross);
    }

    pub fn apply(self: *Position, lex: *const Lexicon, m: *const Move) void {
        const d = m.dir;
        const p = 1 - d;
        const line: usize = m.line;
        var it = m.new_mask;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            const c = m.tiles[i];
            self.cells[d][line * N + i] = c;
            self.cells[p][i * N + line] = c;
            self.pool[if (rules.isBlank(c)) rules.BLANK_SLOT else rules.letterOf(c)] -= 1;
        }
        self.pool_total -= m.placed;
        self.tiles_on_board += m.placed;
        self.score += m.score;
        self.plies += 1;

        // Squares just past either end of the main word: their perpendicular
        // plays now see this word as a cross word.
        if (m.start > 0) self.refreshCross(lex, p, m.start - 1, line);
        if (m.end + 1 < N) self.refreshCross(lex, p, m.end + 1, line);
        // Squares just past either end of each new tile's perpendicular run.
        it = m.new_mask;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            const run = self.cells[p][i * N ..][0..N];
            var a = line;
            while (a > 0 and run[a - 1] != 0) a -= 1;
            var b = line;
            while (b + 1 < N and run[b + 1] != 0) b += 1;
            if (a > 0) self.refreshCross(lex, d, a - 1, i);
            if (b + 1 < N) self.refreshCross(lex, d, b + 1, i);
        }
    }

    /// Recompute xmask/xsum for the square at (line, i) in direction `d`.
    fn refreshCross(self: *Position, lex: *const Lexicon, d: u1, line: usize, i: usize) void {
        const idx = line * N + i;
        if (self.cells[d][idx] != 0) return;
        const run = self.cells[1 - d][i * N ..][0..N];
        var a = line;
        while (a > 0 and run[a - 1] != 0) a -= 1;
        var b = line;
        while (b + 1 < N and run[b + 1] != 0) b += 1;
        if (a == line and b == line) {
            self.xmask[d][idx] = ALL_LETTERS;
            self.xsum[d][idx] = 0;
            return;
        }
        var sum: i32 = 0;
        for (run[a..line]) |c| sum += rules.cellValue(c);
        for (run[line + 1 .. b + 1]) |c| sum += rules.cellValue(c);
        self.xmask[d][idx] = lex.crossMask(run[a..line], run[line + 1 .. b + 1]) | HAS_CROSS;
        self.xsum[d][idx] = @intCast(sum);
    }

    /// Rebuild every cross-check from scratch (used to test `apply`).
    pub fn recomputeCrossChecks(self: *Position, lex: *const Lexicon) void {
        for (0..2) |d| {
            for (0..N) |line| {
                for (0..N) |i| {
                    self.xmask[d][line * N + i] = ALL_LETTERS;
                    self.xsum[d][line * N + i] = 0;
                    self.refreshCross(lex, @intCast(d), line, i);
                }
            }
        }
    }

    /// Score a play from the board alone, without the cached cross sums.
    pub fn scoreMove(self: *const Position, m: *const Move, r: rules.Rules) i32 {
        const d = m.dir;
        const p = 1 - d;
        const line: usize = m.line;
        var main: i32 = 0;
        var wmul: i32 = 1;
        var cross: i32 = 0;
        for (m.start..@as(usize, m.end) + 1) |i| {
            const idx = line * N + i;
            if (m.new_mask & (@as(u16, 1) << @intCast(i)) == 0) {
                main += rules.cellValue(self.cells[d][idx]);
                continue;
            }
            const lv = rules.cellValue(m.tiles[i]) * rules.premiums.letter[d][idx];
            const wm: i32 = rules.premiums.word[d][idx];
            main += lv;
            wmul *= wm;
            const run = self.cells[p][i * N ..][0..N];
            var a = line;
            while (a > 0 and run[a - 1] != 0) a -= 1;
            var b = line;
            while (b + 1 < N and run[b + 1] != 0) b += 1;
            if (a == line and b == line) continue;
            var s: i32 = lv;
            for (a..b + 1) |k| {
                if (k != line) s += rules.cellValue(run[k]);
            }
            cross += s * wm;
        }
        const bingo: i32 = if (m.placed == r.rack) r.bingo else 0;
        return main * wmul + cross + bingo;
    }

    /// Build and fully validate a play given as text, e.g. (7, 4, 0, "JOKER").
    /// Lowercase letters are blanks.
    pub fn parsePlay(
        self: *const Position,
        lex: *const Lexicon,
        r: rules.Rules,
        row: usize,
        col: usize,
        d: u1,
        word: []const u8,
    ) PlayError!Move {
        const line = if (d == 0) row else col;
        const start = if (d == 0) col else row;
        if (line >= N or word.len < 2 or start + word.len > N) return error.OffBoard;
        var m = Move{
            .dir = d,
            .line = @intCast(line),
            .start = @intCast(start),
            .end = @intCast(start + word.len - 1),
        };
        var pool = self.pool;
        var cells: [N]u8 = undefined;
        for (word, 0..) |ch, k| {
            if (!std.ascii.isAlphabetic(ch)) return error.BadWord;
            const i = start + k;
            const l: u5 = @intCast(std.ascii.toUpper(ch) - 'A');
            const existing = self.cells[d][line * N + i];
            if (existing != 0) {
                if (rules.letterOf(existing) != l) return error.LetterMismatch;
                cells[k] = existing;
                continue;
            }
            const blank = std.ascii.isLower(ch);
            const slot: usize = if (blank) rules.BLANK_SLOT else l;
            if (pool[slot] == 0) return error.NotInBag;
            pool[slot] -= 1;
            m.tiles[i] = rules.cellOf(l, blank);
            m.new_mask |= @as(u16, 1) << @intCast(i);
            m.placed += 1;
            cells[k] = m.tiles[i];
        }
        if (m.placed == 0) return error.NoNewTiles;
        if (m.placed > r.rack) return error.TooManyTiles;
        if (start > 0 and self.cells[d][line * N + start - 1] != 0) return error.NotDelimited;
        if (m.end + 1 < N and self.cells[d][line * N + m.end + 1] != 0) return error.NotDelimited;
        if (m.new_mask & self.anchors(d, line) == 0) return error.NotConnected;
        if (!lex.containsCells(cells[0..word.len])) return error.NotAWord;
        var it = m.new_mask;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            const run = self.cells[1 - d][i * N ..][0..N];
            var a = line;
            while (a > 0 and run[a - 1] != 0) a -= 1;
            var b = line;
            while (b + 1 < N and run[b + 1] != 0) b += 1;
            if (a == line and b == line) continue;
            var buf: [N]u8 = undefined;
            @memcpy(buf[0 .. b + 1 - a], run[a .. b + 1]);
            buf[line - a] = m.tiles[i];
            if (!lex.containsCells(buf[0 .. b + 1 - a])) return error.BadCrossWord;
        }
        m.score = self.scoreMove(&m, r);
        return m;
    }

    /// "ROW COL across|down WORD", the play-file format both engines read.
    pub fn writePlay(self: *const Position, m: *const Move, w: *Io.Writer) !void {
        try w.print("{d} {d} {s} ", .{
            m.row(m.start),
            m.col(m.start),
            if (m.dir == 0) "across" else "down",
        });
        for (m.start..@as(usize, m.end) + 1) |i| {
            const c = if (m.new_mask & (@as(u16, 1) << @intCast(i)) != 0)
                m.tiles[i]
            else
                self.cells[m.dir][@as(usize, m.line) * N + i];
            try w.writeByte(cellChar(c));
        }
    }

    pub fn writeBoard(self: *const Position, w: *Io.Writer) !void {
        try w.writeAll("  ");
        for (0..N) |c| try w.print("{d:>3}", .{c});
        try w.writeByte('\n');
        for (0..N) |r| {
            try w.print("{d:>2}  ", .{r});
            for (0..N) |c| {
                const cell = self.at(r, c);
                const ch: u8 = if (cell != 0) cellChar(cell) else switch (rules.premiums.word[0][r * N + c]) {
                    3 => '=',
                    2 => '-',
                    else => switch (rules.premiums.letter[0][r * N + c]) {
                        3 => '"',
                        2 => '\'',
                        else => '.',
                    },
                };
                try w.print("  {c}", .{ch});
            }
            try w.writeByte('\n');
        }
        try w.print("tiles on board {d}, unplayed {d}\n", .{ self.tiles_on_board, self.pool_total });
    }
};
