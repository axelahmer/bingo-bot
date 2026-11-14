//! Board geometry, tile set, premium squares and tile encoding.

const std = @import("std");

pub const N = 15;
pub const NN = N * N;
pub const CENTER = N / 2;

/// Letters are 0..25. The bag has one extra slot for blanks.
pub const BLANK_SLOT = 26;

/// Board cell encoding: 0 = empty, otherwise (letter + 1), with BLANK_FLAG set
/// when the tile is a blank standing in for that letter.
pub const EMPTY: u8 = 0;
pub const BLANK_FLAG: u8 = 0x20;

pub inline fn cellOf(letter: u5, blank: bool) u8 {
    return (@as(u8, letter) + 1) | (if (blank) BLANK_FLAG else 0);
}

pub inline fn letterOf(cell: u8) u5 {
    return @intCast((cell & 0x1f) - 1);
}

pub inline fn isBlank(cell: u8) bool {
    return cell & BLANK_FLAG != 0;
}

/// Face value of a tile on the board (blanks are worth nothing).
pub inline fn cellValue(cell: u8) i32 {
    return if (isBlank(cell)) 0 else letter_values[letterOf(cell)];
}

pub const letter_values = [26]u8{
    1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, //  A..M
    1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, // N..Z
};

/// Standard English distribution; the last entry is the blank count.
pub const standard_counts = [27]u8{
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, //  A..M
    6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, // N..Z
    2,
};

/// How the end-of-game rack adjustment works when a player goes out.
pub const EndRule = enum {
    /// WESPA (the rules that go with Collins/SOWPODS): the player who goes
    /// out gains the value of the opponent's rack, and the opponent loses it.
    wespa,
    /// NASPA: the player who goes out gains twice the opponent's rack value.
    naspa,
};

/// The rules of a game. The defaults are a standard two-player game; the
/// 2023 Python reference is `.{ .players = 1, .blanks = 0 }`.
pub const Rules = struct {
    /// Players taking turns. 1 is solitaire.
    players: u8 = 2,
    /// Blanks in the bag (the standard set has 2).
    blanks: u8 = 2,
    /// Rack size: tiles drawn at the start and refilled after each play.
    rack: u8 = 7,
    /// Bonus for placing all `rack` tiles in one play.
    bingo: i32 = 50,
    /// This many consecutive scoreless turns end the game.
    scoreless_limit: u8 = 6,
    end_rule: EndRule = .wespa,
};

/// Total face value of a set of tile counts (blanks are worth nothing).
pub fn countsValue(counts: *const [27]u8) i32 {
    var v: i32 = 0;
    for (counts[0..26], letter_values) |n, lv| v += @as(i32, n) * lv;
    return v;
}

/// Standard premium layout. '.' plain, 'd' double letter, 't' triple letter,
/// 'D' double word, 'T' triple word.
const layout = [N]*const [N]u8{
    "T..d...T...d..T",
    ".D...t...t...D.",
    "..D...d.d...D..",
    "d..D...d...D..d",
    "....D.....D....",
    ".t...t...t...t.",
    "..d...d.d...d..",
    "T..d...D...d..T",
    "..d...d.d...d..",
    ".t...t...t...t.",
    "....D.....D....",
    "d..D...d...D..d",
    "..D...d.d...D..",
    ".D...t...t...D.",
    "T..d...T...d..T",
};

pub const Premiums = struct {
    /// Indexed [dir][line * N + i] so a line is contiguous in either direction.
    letter: [2][NN]u8,
    word: [2][NN]u8,
};

pub const premiums: Premiums = blk: {
    @setEvalBranchQuota(10_000);
    var p: Premiums = undefined;
    for (0..N) |r| {
        for (0..N) |c| {
            const ch = layout[r][c];
            const lm: u8 = switch (ch) {
                'd' => 2,
                't' => 3,
                else => 1,
            };
            const wm: u8 = switch (ch) {
                'D' => 2,
                'T' => 3,
                else => 1,
            };
            p.letter[0][r * N + c] = lm;
            p.word[0][r * N + c] = wm;
            p.letter[1][c * N + r] = lm;
            p.word[1][c * N + r] = wm;
        }
    }
    break :blk p;
};

test "premium layout matches the standard board" {
    var dl: usize = 0;
    var tl: usize = 0;
    var dw: usize = 0;
    var tw: usize = 0;
    for (0..NN) |i| {
        switch (premiums.letter[0][i]) {
            2 => dl += 1,
            3 => tl += 1,
            else => {},
        }
        switch (premiums.word[0][i]) {
            2 => dw += 1,
            3 => tw += 1,
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 24), dl);
    try std.testing.expectEqual(@as(usize, 12), tl);
    try std.testing.expectEqual(@as(usize, 17), dw);
    try std.testing.expectEqual(@as(usize, 8), tw);
    // The Python reference lists (1, 5) as a triple letter and (0, 3) as a double letter.
    try std.testing.expectEqual(@as(u8, 3), premiums.letter[0][1 * N + 5]);
    try std.testing.expectEqual(@as(u8, 2), premiums.letter[0][0 * N + 3]);
    var total: usize = 0;
    for (standard_counts[0..26]) |n| total += n;
    try std.testing.expectEqual(@as(usize, 98), total);
}
