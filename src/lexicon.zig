//! GADDAG lexicon (Gordon 1994) stored as a minimized automaton of bitmask
//! nodes. Every word w = u ++ v with u non-empty is stored as rev(u) SEP v,
//! and the whole word reversed is stored as rev(w), so move generation can
//! start at an anchor square and grow in both directions.
//!
//! Each node is one u64 record:
//!   bits 0..25   letters that have a child
//!   bit  26      separator child
//!   bit  27      accepting
//!   bits 28..41  reach: bit d set when some path of exactly d more letters
//!                ends where a word may stop (an accepting node, or before the
//!                separator, a node with a separator child); bit 13 means
//!                "13 or more"
//!   bits 42..63  index of the first child record
//! Children sit contiguously in symbol order, so the child for symbol s is
//! nodes[base + popcount(mask & ((1 << s) - 1))]. Child arrays with identical
//! contents are stored once, which gives the minimal automaton. The record 0
//! doubles as the dead state: no children and not accepting.
//!
//! The reach mask lets the move generator drop a branch as soon as no word
//! through it can fit the free squares and the tiles left to place. Without
//! it the leftward walk visits almost every reversed substring of the
//! lexicon, because the bag-as-rack setting offers nearly every letter.

const std = @import("std");
const rules = @import("rules.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const SEP: u5 = 26;
pub const LETTERS: u32 = (1 << 26) - 1;
const SYMBOLS: u64 = (1 << 27) - 1;
pub const ACCEPT: u64 = 1 << 27;
pub const REACH_SHIFT = 28;
pub const REACH_BITS = 14;
pub const BASE_SHIFT = 42;
pub const MAX_RECORDS = 1 << (64 - BASE_SHIFT);
pub const Node = u64;

const file_magic: u64 = 0x3130_4741_4444_4147; // "GADDAG01" little-endian
const format_version: u64 = 3;

pub inline fn letterMask(n: Node) u32 {
    return @as(u32, @truncate(n)) & LETTERS;
}

pub inline fn accepts(n: Node) bool {
    return n & ACCEPT != 0;
}

/// Reach mask widened to 15 bits: the saturated bit 13 also covers 14.
pub inline fn reach(n: Node) u16 {
    const r: u16 = @intCast((n >> REACH_SHIFT) & ((1 << REACH_BITS) - 1));
    return r | ((r & (1 << 13)) << 1);
}

inline fn base(n: Node) usize {
    return @intCast(n >> BASE_SHIFT);
}

pub const Lexicon = struct {
    nodes: []const u64,
    word_count: u32,
    words_hash: u64 = 0,
    /// Allocation backing `nodes` (it may carry a file header in front).
    storage: []u64,

    pub fn deinit(self: *Lexicon, gpa: Allocator) void {
        gpa.free(self.storage);
        self.* = undefined;
    }

    pub inline fn root(self: *const Lexicon) Node {
        return self.nodes[0];
    }

    pub inline fn child(self: *const Lexicon, n: Node, sym: u5) Node {
        const m: u32 = @truncate(n & SYMBOLS);
        const bit = @as(u32, 1) << sym;
        if (m & bit == 0) return 0;
        return self.nodes[base(n) + @popCount(m & (bit - 1))];
    }

    /// Caller guarantees that `n` has a child for `sym`.
    pub inline fn childUnchecked(self: *const Lexicon, n: Node, sym: u5) Node {
        const m: u32 = @truncate(n & SYMBOLS);
        const bit = @as(u32, 1) << sym;
        return self.nodes[base(n) + @popCount(m & (bit - 1))];
    }

    /// True if `word` (letters 0..25) is in the lexicon.
    pub fn contains(self: *const Lexicon, word: []const u5) bool {
        if (word.len == 0) return false;
        var n = self.root();
        var i = word.len;
        while (i > 0) {
            i -= 1;
            n = self.child(n, word[i]);
            if (n == 0) return false;
        }
        return accepts(n);
    }

    /// Like `contains`, but for board cells (blanks count as their letter).
    pub fn containsCells(self: *const Lexicon, cells: []const u8) bool {
        if (cells.len == 0) return false;
        var n = self.root();
        var i = cells.len;
        while (i > 0) {
            i -= 1;
            n = self.child(n, rules.letterOf(cells[i]));
            if (n == 0) return false;
        }
        return accepts(n);
    }

    /// Set of letters L such that before ++ L ++ after is a word. Both
    /// arguments are board cells and at least one of them is non-empty.
    pub fn crossMask(self: *const Lexicon, before: []const u8, after: []const u8) u32 {
        var out: u32 = 0;
        var n = self.root();
        if (before.len > 0) {
            // Path: rev(before) SEP L after.
            var i = before.len;
            while (i > 0) {
                i -= 1;
                n = self.child(n, rules.letterOf(before[i]));
                if (n == 0) return 0;
            }
            n = self.child(n, SEP);
            if (n == 0) return 0;
            var it = letterMask(n);
            while (it != 0) : (it &= it - 1) {
                const l: u5 = @intCast(@ctz(it));
                var m = self.childUnchecked(n, l);
                for (after) |c| {
                    m = self.child(m, rules.letterOf(c));
                    if (m == 0) break;
                }
                if (accepts(m)) out |= @as(u32, 1) << l;
            }
        } else {
            // Path: rev(after) L, the whole word reversed.
            var i = after.len;
            while (i > 0) {
                i -= 1;
                n = self.child(n, rules.letterOf(after[i]));
                if (n == 0) return 0;
            }
            var it = letterMask(n);
            while (it != 0) : (it &= it - 1) {
                const l: u5 = @intCast(@ctz(it));
                if (accepts(self.childUnchecked(n, l))) out |= @as(u32, 1) << l;
            }
        }
        return out;
    }

    pub fn save(self: *const Lexicon, io: Io, path: []const u8) !void {
        const header = [4]u64{ file_magic, format_version, self.words_hash, self.word_count };
        var file = try Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &buf);
        try fw.interface.writeAll(std.mem.sliceAsBytes(&header));
        try fw.interface.writeAll(std.mem.sliceAsBytes(self.nodes));
        try fw.interface.flush();
    }

    pub fn load(gpa: Allocator, io: Io, path: []const u8, expect_hash: u64) !Lexicon {
        const bytes = try Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .unlimited, .of(u64), null);
        errdefer gpa.free(bytes);
        if (bytes.len < 40 or bytes.len % 8 != 0) return error.BadLexiconFile;
        const words: []u64 = @alignCast(std.mem.bytesAsSlice(u64, bytes));
        if (words[0] != file_magic or words[1] != format_version) return error.BadLexiconFile;
        if (words[2] != expect_hash) return error.StaleLexiconFile;
        return .{
            .nodes = words[4..],
            .word_count = @intCast(words[3]),
            .words_hash = words[2],
            .storage = words,
        };
    }

    /// Reads the word list, then loads the cached automaton if it was built
    /// from the same list, else builds it and refreshes the cache.
    pub fn loadOrBuild(gpa: Allocator, io: Io, words_path: []const u8, cache_path: ?[]const u8) !Lexicon {
        const text = try Io.Dir.cwd().readFileAlloc(io, words_path, gpa, .unlimited);
        defer gpa.free(text);
        const h = std.hash.Wyhash.hash(format_version, text);
        if (cache_path) |cp| {
            if (load(gpa, io, cp, h)) |lex| return lex else |_| {}
        }
        var lex = try build(gpa, text);
        lex.words_hash = h;
        if (cache_path) |cp| lex.save(io, cp) catch |err| {
            std.log.warn("could not write lexicon cache {s}: {t}", .{ cp, err });
        };
        return lex;
    }
};

/// Parses a word list (one word per line, letters only, any case). Words
/// shorter than 2 or longer than the board are skipped.
pub fn build(gpa: Allocator, text: []const u8) !Lexicon {
    // Collect words as letter codes.
    var letters: std.ArrayList(u8) = .empty;
    defer letters.deinit(gpa);
    var spans: std.ArrayList([2]u32) = .empty;
    defer spans.deinit(gpa);
    var tok = std.mem.tokenizeAny(u8, text, " \t\r\n");
    outer: while (tok.next()) |w| {
        if (w.len < 2 or w.len > rules.N) continue;
        for (w) |ch| if (!std.ascii.isAlphabetic(ch)) continue :outer;
        const off: u32 = @intCast(letters.items.len);
        for (w) |ch| try letters.append(gpa, std.ascii.toUpper(ch) - 'A');
        try spans.append(gpa, .{ off, @intCast(w.len) });
    }

    // Emit every GADDAG string into one buffer.
    var total: usize = 0;
    for (spans.items) |s| total += @as(usize, s[1]) * s[1] + s[1] - 1;
    const buf = try gpa.alloc(u8, total);
    defer gpa.free(buf);
    var refs: std.ArrayList(u64) = .empty;
    defer refs.deinit(gpa);
    var pos: usize = 0;
    for (spans.items) |s| {
        const w = letters.items[s[0]..][0..s[1]];
        const n = w.len;
        for (1..n + 1) |i| {
            const start = pos;
            var k = i;
            while (k > 0) {
                k -= 1;
                buf[pos] = w[k];
                pos += 1;
            }
            if (i < n) {
                buf[pos] = SEP;
                pos += 1;
                @memcpy(buf[pos..][0 .. n - i], w[i..]);
                pos += n - i;
            }
            try refs.append(gpa, (@as(u64, start) << 8) | (pos - start));
        }
    }
    std.debug.assert(pos == total);

    const Sort = struct {
        buf: []const u8,
        fn str(ctx: @This(), r: u64) []const u8 {
            return ctx.buf[@intCast(r >> 8)..][0..@intCast(r & 0xff)];
        }
        fn lessThan(ctx: @This(), a: u64, b: u64) bool {
            return std.mem.order(u8, ctx.str(a), ctx.str(b)) == .lt;
        }
    };
    const sctx = Sort{ .buf = buf };
    std.sort.pdq(u64, refs.items, sctx, Sort.lessThan);

    // Drop duplicates (repeated words in the input).
    var uniq: usize = 0;
    for (refs.items) |r| {
        if (uniq > 0 and std.mem.eql(u8, sctx.str(refs.items[uniq - 1]), sctx.str(r))) continue;
        refs.items[uniq] = r;
        uniq += 1;
    }
    var word_count: u32 = 0;
    for (refs.items[0..uniq]) |r| {
        if (std.mem.indexOfScalar(u8, sctx.str(r), SEP) == null) word_count += 1;
    }

    var b = Builder{ .gpa = gpa, .buf = buf, .refs = refs.items[0..uniq] };
    defer b.map.deinit(gpa);
    errdefer b.nodes.deinit(gpa);
    // Slot 0 holds the root record; child arrays follow.
    try b.nodes.append(gpa, 0);
    const root = if (uniq == 0) 0 else try b.node(0, uniq, 0);
    b.nodes.items[0] = root;
    // A spare record past the end lets child lookups load before testing
    // the child bit (see movegen), without reading out of bounds.
    try b.nodes.append(gpa, 0);
    const storage = try b.nodes.toOwnedSlice(gpa);
    return .{ .nodes = storage, .word_count = word_count, .storage = storage };
}

const Span = struct { off: u32, len: u32 };

const SpanCtx = struct {
    nodes: *const std.ArrayList(u64),
    pub fn hash(ctx: SpanCtx, s: Span) u64 {
        return hashKids(ctx.nodes.items[s.off..][0..s.len]);
    }
    pub fn eql(ctx: SpanCtx, a: Span, b: Span) bool {
        const items = ctx.nodes.items;
        return std.mem.eql(u64, items[a.off..][0..a.len], items[b.off..][0..b.len]);
    }
};

const KidsAdapter = struct {
    nodes: *const std.ArrayList(u64),
    pub fn hash(_: KidsAdapter, kids: []const u64) u64 {
        return hashKids(kids);
    }
    pub fn eql(ctx: KidsAdapter, kids: []const u64, s: Span) bool {
        return std.mem.eql(u64, kids, ctx.nodes.items[s.off..][0..s.len]);
    }
};

fn hashKids(kids: []const u64) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(kids));
}

const Builder = struct {
    gpa: Allocator,
    buf: []const u8,
    refs: []const u64,
    nodes: std.ArrayList(u64) = .empty,
    map: std.HashMapUnmanaged(Span, void, SpanCtx, 80) = .empty,

    fn str(self: *const Builder, r: u64) []const u8 {
        return self.buf[@intCast(r >> 8)..][0..@intCast(r & 0xff)];
    }

    /// Strings refs[lo..hi] share their first `depth` symbols. Returns the
    /// record for the state reached by that prefix.
    fn node(self: *Builder, lo_in: usize, hi: usize, depth: usize) (Allocator.Error || error{LexiconTooLarge})!u64 {
        var lo = lo_in;
        var rec: u64 = 0;
        if (self.str(self.refs[lo]).len == depth) {
            rec |= ACCEPT;
            lo += 1;
        }
        var kids: [27]u64 = undefined;
        var nk: usize = 0;
        // A word may stop here if this node accepts or, before the
        // separator, if the separator follows.
        var r: u32 = @intFromBool(rec & ACCEPT != 0);
        while (lo < hi) {
            const sym = self.str(self.refs[lo])[depth];
            var j = lo + 1;
            while (j < hi and self.str(self.refs[j])[depth] == sym) j += 1;
            const kid = try self.node(lo, j, depth + 1);
            kids[nk] = kid;
            nk += 1;
            rec |= @as(u64, 1) << @intCast(sym);
            if (sym == SEP) r |= 1 else r |= @as(u32, @intCast((kid >> REACH_SHIFT) & ((1 << REACH_BITS) - 1))) << 1;
            lo = j;
        }
        // Saturate: bit 13 stands for every distance of 13 or more.
        if (r >> 13 != 0) r = (r & ((1 << 13) - 1)) | (1 << 13);
        rec |= @as(u64, r) << REACH_SHIFT;
        if (nk > 0) {
            const off = try self.intern(kids[0..nk]);
            if (off >= MAX_RECORDS) return error.LexiconTooLarge;
            rec |= @as(u64, off) << BASE_SHIFT;
        }
        return rec;
    }

    fn intern(self: *Builder, kids: []const u64) Allocator.Error!u32 {
        try self.nodes.ensureUnusedCapacity(self.gpa, kids.len);
        const gop = try self.map.getOrPutContextAdapted(
            self.gpa,
            kids,
            KidsAdapter{ .nodes = &self.nodes },
            SpanCtx{ .nodes = &self.nodes },
        );
        if (gop.found_existing) return gop.key_ptr.off;
        const off: u32 = @intCast(self.nodes.items.len);
        self.nodes.appendSliceAssumeCapacity(kids);
        gop.key_ptr.* = .{ .off = off, .len = @intCast(kids.len) };
        return off;
    }
};

pub fn wordToLetters(word: []const u8, out: []u5) []u5 {
    for (word, 0..) |ch, i| out[i] = @intCast(std.ascii.toUpper(ch) - 'A');
    return out[0..word.len];
}

test "small lexicon: membership and cross masks" {
    const gpa = std.testing.allocator;
    var lex = try build(gpa, "CAT\nCATS\nAT\nSCAT\nACT\nTA\nat\n");
    defer lex.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 6), lex.word_count);
    var tmp: [15]u5 = undefined;
    for ([_][]const u8{ "CAT", "CATS", "AT", "SCAT", "ACT", "TA" }) |w| {
        try std.testing.expect(lex.contains(wordToLetters(w, &tmp)));
    }
    for ([_][]const u8{ "CA", "CATSS", "T", "SCA", "TAC", "S" }) |w| {
        try std.testing.expect(!lex.contains(wordToLetters(w, &tmp)));
    }
    const C = rules.cellOf(2, false);
    const A = rules.cellOf(0, false);
    const T = rules.cellOf(19, false);
    // C?T -> A ; ?AT -> C ; CA? -> T ; ?T -> A ; A? -> T
    try std.testing.expectEqual(@as(u32, 1 << 0), lex.crossMask(&.{C}, &.{T}));
    try std.testing.expectEqual(@as(u32, 1 << 2), lex.crossMask(&.{}, &.{ A, T }));
    try std.testing.expectEqual(@as(u32, 1 << 19), lex.crossMask(&.{ C, A }, &.{}));
    try std.testing.expectEqual(@as(u32, 1 << 0), lex.crossMask(&.{}, &.{T}));
    try std.testing.expectEqual(@as(u32, 1 << 19), lex.crossMask(&.{A}, &.{}));
    // CAT? -> S ; ?CAT -> S
    try std.testing.expectEqual(@as(u32, 1 << 18), lex.crossMask(&.{ C, A, T }, &.{}));
    try std.testing.expectEqual(@as(u32, 1 << 18), lex.crossMask(&.{}, &.{ C, A, T }));
}
