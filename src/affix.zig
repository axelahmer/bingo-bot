//! Exact sets of the word list's words, prefixes and suffixes, for building
//! hooks. A hook word runs from a plan's finish square into the board: on a
//! top or left lane the finish letter comes first and the word grows
//! downwards or rightwards, so until it is finished it must be a prefix of
//! some word; on a bottom or right lane it must be a suffix.
//!
//! A string of up to 25 letters is packed into a u128 as (letter + 1) in
//! 5-bit fields, first letter lowest, so its key can be put together from
//! pieces: `key(a ++ b) == key(a) | key(b) << (5 * a.len)`.

const std = @import("std");

pub const Key = u128;

pub fn key(letters: []const u5) Key {
    var k: Key = 0;
    for (letters, 0..) |l, i| k |= (@as(Key, l) + 1) << @intCast(5 * i);
    return k;
}

/// Key of one letter at position i.
pub inline fn letterAt(l: u5, i: usize) Key {
    return (@as(Key, l) + 1) << @intCast(5 * i);
}

pub const Affixes = struct {
    words: Set = .empty,
    prefixes: Set = .empty,
    suffixes: Set = .empty,

    const Set = std.AutoHashMapUnmanaged(Key, void);

    /// Build from a word list (one word per line, letters only).
    pub fn build(gpa: std.mem.Allocator, text: []const u8) !Affixes {
        var a = Affixes{};
        errdefer a.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
        var letters: [25]u5 = undefined;
        while (it.next()) |w| {
            if (w.len > letters.len) continue;
            var ok = true;
            for (w, 0..) |ch, i| {
                if (!std.ascii.isAlphabetic(ch)) {
                    ok = false;
                    break;
                }
                letters[i] = @intCast(std.ascii.toUpper(ch) - 'A');
            }
            if (!ok) continue;
            const s = letters[0..w.len];
            try a.words.put(gpa, key(s), {});
            for (1..s.len + 1) |n| {
                try a.prefixes.put(gpa, key(s[0..n]), {});
                try a.suffixes.put(gpa, key(s[s.len - n ..]), {});
            }
        }
        return a;
    }

    pub fn deinit(a: *Affixes, gpa: std.mem.Allocator) void {
        a.words.deinit(gpa);
        a.prefixes.deinit(gpa);
        a.suffixes.deinit(gpa);
    }

    pub fn isWord(a: *const Affixes, k: Key) bool {
        return a.words.contains(k);
    }

    pub fn isPrefix(a: *const Affixes, k: Key) bool {
        return a.prefixes.contains(k);
    }

    pub fn isSuffix(a: *const Affixes, k: Key) bool {
        return a.suffixes.contains(k);
    }
};

test "keys compose and the sets answer" {
    const gpa = std.testing.allocator;
    var a = try Affixes.build(gpa, "BRAIN\nBRAINWASH\nELKS\n");
    defer a.deinit(gpa);
    const b = [_]u5{ 1, 17, 0 }; // BRA
    try std.testing.expect(a.isPrefix(key(&b)));
    try std.testing.expect(!a.isWord(key(&b)));
    const ks = [_]u5{ 10, 18 }; // KS
    try std.testing.expect(a.isSuffix(key(&ks)));
    try std.testing.expect(!a.isPrefix(key(&ks)));
    // key(BR ++ A) == key(BR) | A at position 2.
    try std.testing.expectEqual(key(&b), key(b[0..2]) | letterAt(0, 2));
}
