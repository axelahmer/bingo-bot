//! Plans for the biggest plays on the board: a 15-letter word across a
//! whole edge lane (row 0, row 14, column 0 or column 14), which covers three
//! triple-word squares and is worth 27 times its letters.
//!
//! Premiums count only for tiles placed in that play, so the finishing play
//! must place the three triple-word squares itself, and with at most 7 tiles
//! per play at least 8 of the word's letters have to be on the board already.
//! A plan names the word and the squares the finish will fill (`finish`);
//! the other lane squares are setup squares, to be filled with the word's
//! letters by earlier plays. Until the finish is played, the search keeps
//! the finish squares (and the squares just inside them) empty, lets setup
//! squares take only their letter, and holds the finishing tiles back.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");

const N = rules.N;
const NN = rules.NN;
const Lexicon = lexicon.Lexicon;

pub const Lane = enum {
    row0,
    row14,
    col0,
    col14,

    fn dir(l: Lane) u1 {
        return switch (l) {
            .row0, .row14 => 0,
            .col0, .col14 => 1,
        };
    }

    fn line(l: Lane) usize {
        return switch (l) {
            .row0, .col0 => 0,
            .row14, .col14 => N - 1,
        };
    }
};

pub const Plan = struct {
    lane: Lane,
    word: [N]u5,
    /// Lane positions filled by the finishing play.
    finish: u16,
    /// Finishing tiles to hold back, by tile kind (index 26: blanks).
    reserve: [27]u8,
    /// How many of the finish's letters of each kind are blanks.
    blank_for: [26]u8,

    /// Board square of lane position i.
    pub fn square(p: *const Plan, i: usize) usize {
        const l = p.lane.line();
        return if (p.lane.dir() == 0) l * N + i else i * N + l;
    }

    /// The square one step from lane position i towards the middle.
    fn inner(p: *const Plan, i: usize) usize {
        const l = p.lane.line();
        const in = if (l == 0) 1 else l - 1;
        return if (p.lane.dir() == 0) in * N + i else i * N + in;
    }

    pub fn setup(p: *const Plan) u16 {
        return ~p.finish & ((1 << N) - 1);
    }

    /// The finish has been played (its squares are covered).
    pub fn done(p: *const Plan, pos: *const board.Position) bool {
        return pos.cells[0][p.square(@ctz(p.finish))] != 0;
    }

    /// Setup squares already filled.
    pub fn filled(p: *const Plan, pos: *const board.Position) u16 {
        var m: u16 = 0;
        var it = p.setup();
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            if (pos.cells[0][p.square(i)] != 0) m |= @as(u16, 1) << @intCast(i);
        }
        return m;
    }

    /// Narrow `restrict` (allowed letters per square) for this plan.
    pub fn restrictInto(p: *const Plan, restrict: *[NN]u32) void {
        for (0..N) |i| {
            if ((p.finish >> @intCast(i)) & 1 != 0) {
                restrict[p.square(i)] = 0;
                restrict[p.inner(i)] = 0;
            } else {
                restrict[p.square(i)] &= @as(u32, 1) << p.word[i];
            }
        }
    }

    /// For the generator's visitors: per (direction, line), the line
    /// positions that are empty setup squares of this plan.
    pub fn goalMasks(p: *const Plan, pos: *const board.Position, out: *[2][N]u16) void {
        var it = p.setup() & ~p.filled(pos);
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            const sq = p.square(i);
            const r = sq / N;
            const c = sq % N;
            out[0][r] |= @as(u16, 1) << @intCast(c);
            out[1][c] |= @as(u16, 1) << @intCast(r);
        }
    }

    /// The finishing play, if every setup square is filled and it is legal now.
    pub fn finishMove(p: *const Plan, lex: *const Lexicon, r: rules.Rules, pos: *const board.Position) ?board.Move {
        if (p.done(pos) or p.filled(pos) != p.setup()) return null;
        var text: [N]u8 = undefined;
        var naturals = pos.pool;
        for (0..N) |i| {
            const l = p.word[i];
            const ch: u8 = @as(u8, l) + 'A';
            if ((p.finish >> @intCast(i)) & 1 != 0 and naturals[l] == 0) {
                text[i] = std.ascii.toLower(ch);
            } else {
                text[i] = ch;
                if ((p.finish >> @intCast(i)) & 1 != 0) naturals[l] -= 1;
            }
        }
        const l = p.lane.line();
        const row: usize = if (p.lane.dir() == 0) l else 0;
        const col: usize = if (p.lane.dir() == 0) 0 else l;
        return pos.parsePlay(lex, r, row, col, p.lane.dir(), &text) catch null;
    }

    /// Score of the finish on an empty lane with no cross words: every letter
    /// at face value (blanks for letters the tile set lacks), the new tiles'
    /// letter premiums, times 27, plus the bonus for 7 tiles.
    pub fn estimate(p: *const Plan, r: rules.Rules) i32 {
        var sum: i32 = 0;
        for (0..N) |i| {
            const v: i32 = rules.letter_values[p.word[i]];
            const fin = (p.finish >> @intCast(i)) & 1 != 0;
            const lm: i32 = if (fin) rules.premiums.letter[0][p.square(i)] else 1;
            sum += v * lm;
        }
        // Blanks on finish squares score nothing (ignoring which square they take).
        for (0..26) |l| sum -= @as(i32, p.blank_for[l]) * rules.letter_values[l];
        const tiles = @popCount(p.finish);
        return sum * 27 + (if (tiles == r.rack) r.bingo else 0);
    }
};

pub const MakeError = error{ BadPlan, TilesUnavailable };

/// Build a plan, or fail if the tile set cannot supply the word.
pub fn make(lane: Lane, word_text: []const u8, finish: u16, r: rules.Rules) MakeError!Plan {
    if (word_text.len != N) return error.BadPlan;
    var p = Plan{ .lane = lane, .word = undefined, .finish = finish, .reserve = @splat(0), .blank_for = @splat(0) };
    for (word_text, 0..) |ch, i| {
        if (!std.ascii.isAlphabetic(ch)) return error.BadPlan;
        p.word[i] = @intCast(std.ascii.toUpper(ch) - 'A');
    }
    const tw: u16 = (1 << 0) | (1 << 7) | (1 << 14);
    if (finish & tw != tw or @popCount(finish) > r.rack) return error.BadPlan;
    // Whole word against the tile set: setup letters must come from natural
    // tiles or blanks too.
    var set = rules.standard_counts;
    set[rules.BLANK_SLOT] = r.blanks;
    var need: [26]u8 = @splat(0);
    for (p.word) |l| need[l] += 1;
    var short: u32 = 0;
    for (0..26) |l| short += need[l] -| set[l];
    if (short > set[rules.BLANK_SLOT]) return error.TilesUnavailable;
    // Reserve the finishing tiles: naturals first, blanks for the rest. Setup
    // letters take naturals before the finish does.
    var finish_need: [26]u8 = @splat(0);
    var it = finish;
    while (it != 0) : (it &= it - 1) finish_need[p.word[@ctz(it)]] += 1;
    for (0..26) |l| {
        const setup_need = need[l] - finish_need[l];
        const left = set[l] -| setup_need;
        const nat = @min(finish_need[l], left);
        p.reserve[l] = nat;
        p.blank_for[l] = finish_need[l] - nat;
        p.reserve[rules.BLANK_SLOT] += finish_need[l] - nat;
    }
    return p;
}

/// What the open plans demand of one position.
pub const Active = struct {
    /// Plans still worth pursuing (finish not played, setup still suppliable).
    live: u8 = 0,
    restrict: [NN]u32 = @splat(lexicon.LETTERS),
    /// Per (direction, line): positions that are empty setup squares.
    goal: [2][N]u16 = @splat(@splat(0)),
    /// Tiles held back for finishes.
    reserve: [27]u8 = @splat(0),
    /// Setup squares already filled, over the live plans.
    filled: u32 = 0,
    /// Tiles of each kind (index 26: blanks) that plays may spend away from
    /// the setup squares without starving a live plan.
    spare: [27]u8 = @splat(N),
    /// Kinds whose spare count one play could exceed (fewer than 7 spare).
    tight: u32 = 0,

    /// Whether a play leaves every live plan enough tiles. Tiles placed on
    /// setup squares count against the setup, not the spare.
    pub fn affordable(a: *const Active, dir: u1, line: u8, new_mask: u16, tiles: *const [N]u8, blanks: u16) bool {
        var off = new_mask & ~a.goal[dir][line];
        var used: u32 = 0;
        var count: [27]u8 = undefined;
        while (off != 0) : (off &= off - 1) {
            const i: usize = @ctz(off);
            const kind: u5 = if ((blanks >> @intCast(i)) & 1 != 0) rules.BLANK_SLOT else rules.letterOf(tiles[i]);
            const bit = @as(u32, 1) << kind;
            if (a.tight & bit == 0) continue;
            if (used & bit == 0) {
                used |= bit;
                count[kind] = 0;
            }
            count[kind] += 1;
            if (count[kind] > a.spare[kind]) return false;
        }
        return true;
    }
};

/// Work out which plans are live in `pos` and what they demand. A plan is
/// dropped when its remaining setup letters can no longer be supplied. A
/// letter with no natural tiles to spare beyond the open setup squares is
/// allowed only on those squares.
pub fn activate(plans: []const Plan, pos: *const board.Position, dropped: u8) Active {
    var a = Active{};
    var live: u8 = 0;
    for (plans, 0..) |*p, i| {
        if (!p.done(pos)) live |= @as(u8, 1) << @intCast(i);
    }
    live &= ~dropped;
    while (true) {
        var reserve: [27]u8 = @splat(0);
        var need: [26]u8 = @splat(0);
        for (plans, 0..) |*p, i| {
            if ((live >> @intCast(i)) & 1 == 0) continue;
            for (0..27) |t| reserve[t] += p.reserve[t];
            var it = p.setup() & ~p.filled(pos);
            while (it != 0) : (it &= it - 1) need[p.word[@ctz(it)]] += 1;
        }
        var short: u32 = 0;
        for (0..26) |l| short += need[l] -| (pos.pool[l] -| reserve[l]);
        const blanks_free = pos.pool[rules.BLANK_SLOT] -| reserve[rules.BLANK_SLOT];
        if (short <= blanks_free) {
            a.reserve = reserve;
            for (0..26) |l| {
                if (need[l] == 0) continue;
                const free = pos.pool[l] -| reserve[l];
                a.spare[l] = free -| need[l];
                if (a.spare[l] < 7) a.tight |= @as(u32, 1) << @intCast(l);
            }
            if (short > 0) {
                a.spare[rules.BLANK_SLOT] = @intCast(blanks_free - short);
                if (a.spare[rules.BLANK_SLOT] < 7) a.tight |= @as(u32, 1) << rules.BLANK_SLOT;
            }
            // Scarce setup letters may go only where a setup square wants them.
            for (0..26) |l| {
                if (need[l] == 0 or (pos.pool[l] -| reserve[l]) > need[l]) continue;
                const bit = @as(u32, 1) << @intCast(l);
                for (&a.restrict) |*m| m.* &= ~bit;
                for (plans, 0..) |*p, i| {
                    if ((live >> @intCast(i)) & 1 == 0) continue;
                    var it = p.setup() & ~p.filled(pos);
                    while (it != 0) : (it &= it - 1) {
                        const j: usize = @ctz(it);
                        if (p.word[j] == l) a.restrict[p.square(j)] |= bit;
                    }
                }
            }
            break;
        }
        // Drop the last live plan and try again.
        live &= ~(@as(u8, 1) << @intCast(7 - @clz(live)));
        if (live == 0) break;
    }
    a.live = live;
    for (plans, 0..) |*p, i| {
        if ((live >> @intCast(i)) & 1 == 0) continue;
        p.restrictInto(&a.restrict);
        p.goalMasks(pos, &a.goal);
        a.filled += @popCount(p.filled(pos));
    }
    return a;
}

/// Recompute the reserves of several plans together, so they do not count
/// on the same tiles: setup letters of every plan take natural tiles first,
/// then each finish takes the naturals left, and blanks cover the rest.
pub fn reserveJointly(plans: []Plan, r: rules.Rules) MakeError!void {
    var left = rules.standard_counts;
    left[rules.BLANK_SLOT] = r.blanks;
    for (plans) |*p| {
        var it = p.setup();
        while (it != 0) : (it &= it - 1) {
            const l = p.word[@ctz(it)];
            if (left[l] > 0) {
                left[l] -= 1;
            } else if (left[rules.BLANK_SLOT] > 0) {
                left[rules.BLANK_SLOT] -= 1;
            } else return error.TilesUnavailable;
        }
    }
    for (plans) |*p| {
        p.reserve = @splat(0);
        p.blank_for = @splat(0);
        var it = p.finish;
        while (it != 0) : (it &= it - 1) {
            const l = p.word[@ctz(it)];
            if (left[l] > 0) {
                left[l] -= 1;
                p.reserve[l] += 1;
            } else if (left[rules.BLANK_SLOT] > 0) {
                left[rules.BLANK_SLOT] -= 1;
                p.reserve[rules.BLANK_SLOT] += 1;
                p.blank_for[l] += 1;
            } else return error.TilesUnavailable;
        }
    }
}

/// Parse "row0:WORD:0,1,2,4,7,11,14".
pub fn parse(text: []const u8, r: rules.Rules) !Plan {
    var it = std.mem.splitScalar(u8, text, ':');
    const lane = std.meta.stringToEnum(Lane, it.next() orelse return error.BadPlan) orelse return error.BadPlan;
    const word = it.next() orelse return error.BadPlan;
    var finish: u16 = 0;
    var nums = std.mem.splitScalar(u8, it.next() orelse return error.BadPlan, ',');
    while (nums.next()) |n| {
        const v = try std.fmt.parseInt(u4, n, 10);
        finish |= @as(u16, 1) << v;
    }
    return make(lane, word, finish, r);
}

pub fn write(p: *const Plan, w: *std.Io.Writer) !void {
    try w.print("{s}:", .{@tagName(p.lane)});
    for (p.word) |l| try w.writeByte(@as(u8, l) + 'A');
    try w.writeByte(':');
    var first = true;
    var it = p.finish;
    while (it != 0) : (it &= it - 1) {
        if (!first) try w.writeByte(',');
        first = false;
        try w.print("{d}", .{@ctz(it)});
    }
}

/// True if every run of setup letters (between finish squares) of length
/// two or more is a word, so the setup can be laid in that lane at all.
pub fn runsAreWords(lex: *const Lexicon, word: *const [N]u5, finish: u16) bool {
    var i: usize = 0;
    while (i < N) {
        if ((finish >> @intCast(i)) & 1 != 0) {
            i += 1;
            continue;
        }
        var j = i;
        while (j < N and (finish >> @intCast(j)) & 1 == 0) j += 1;
        if (j - i >= 2 and !lex.contains(word[i..j])) return false;
        i = j;
    }
    return true;
}

pub const Scored = struct {
    plan: Plan,
    estimate: i32,
    /// Longest run of setup letters (shorter runs are easier to lay).
    longest_run: u8,
    /// How hard the setup letters are to bring to the lane: 1 per common
    /// letter, 2 for letters with 2 or 3 tiles, 4 for single-tile letters.
    difficulty: u8,

    fn better(_: void, a: Scored, b: Scored) bool {
        return a.estimate > b.estimate;
    }
};

/// Every plan for `lane` whose setup runs are words and whose longest run
/// is at most `max_run`, best estimate first.
pub fn enumerate(gpa: std.mem.Allocator, lex: *const Lexicon, words_text: []const u8, lane: Lane, max_run: usize, r: rules.Rules) ![]Scored {
    var out: std.ArrayList(Scored) = .empty;
    errdefer out.deinit(gpa);
    var tok = std.mem.tokenizeAny(u8, words_text, " \t\r\n");
    const others = [_]u4{ 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 };
    const tw: u16 = (1 << 0) | (1 << 7) | (1 << 14);
    while (tok.next()) |w| {
        if (w.len != N) continue;
        var letters: [N]u5 = undefined;
        for (w, 0..) |ch, i| letters[i] = @intCast(std.ascii.toUpper(ch) - 'A');
        // Choose 4 more finish squares (7 tiles: a bingo).
        for (0..others.len) |a| for (a + 1..others.len) |b| for (b + 1..others.len) |c| for (c + 1..others.len) |d| {
            const finish = tw | (@as(u16, 1) << others[a]) | (@as(u16, 1) << others[b]) | (@as(u16, 1) << others[c]) | (@as(u16, 1) << others[d]);
            const run = longestRun(finish);
            if (run > max_run) continue;
            if (!runsAreWords(lex, &letters, finish)) continue;
            const p = make(lane, w, finish, r) catch continue;
            try out.append(gpa, .{ .plan = p, .estimate = p.estimate(r), .longest_run = @intCast(run), .difficulty = difficulty(&p) });
        };
    }
    std.sort.pdq(Scored, out.items, {}, Scored.better);
    return out.toOwnedSlice(gpa);
}

fn difficulty(p: *const Plan) u8 {
    var d: u8 = 0;
    var it = p.setup();
    while (it != 0) : (it &= it - 1) {
        const n = rules.standard_counts[p.word[@ctz(it)]];
        d += if (n == 1) 4 else if (n <= 3) 2 else 1;
    }
    return d;
}

/// True if the two plans' words fit the natural tiles together (no blanks).
pub fn fitNaturally(a: *const Plan, b: *const Plan) bool {
    var need: [26]u8 = @splat(0);
    for (a.word) |l| need[l] += 1;
    for (b.word) |l| need[l] += 1;
    for (0..26) |l| {
        if (need[l] > rules.standard_counts[l]) return false;
    }
    return true;
}

fn longestRun(finish: u16) usize {
    var best: usize = 0;
    var run: usize = 0;
    for (0..N) |i| {
        if ((finish >> @intCast(i)) & 1 != 0) {
            run = 0;
        } else {
            run += 1;
            best = @max(best, run);
        }
    }
    return best;
}

test "a plan's reserve, restrictions and estimate" {
    const r = rules.Rules{};
    const p = try parse("row0:OXYPHENBUTAZONE:0,1,2,4,7,11,14", r);
    // Finish letters O X Y H B Z E, all available as natural tiles.
    try std.testing.expectEqual(@as(u8, 1), p.reserve[23]); // X
    try std.testing.expectEqual(@as(u8, 1), p.reserve[25]); // Z
    try std.testing.expectEqual(@as(u8, 0), p.reserve[rules.BLANK_SLOT]);
    var restrict: [NN]u32 = @splat(lexicon.LETTERS);
    p.restrictInto(&restrict);
    try std.testing.expectEqual(@as(u32, 0), restrict[0]); // (0,0) kept free
    try std.testing.expectEqual(@as(u32, 0), restrict[N + 0]); // (1,0) too
    try std.testing.expectEqual(@as(u32, 1 << 15), restrict[3]); // (0,3) only P
    // Face 41, Z on the double letter at (0,11): (41 + 10) * 27 + 50.
    try std.testing.expectEqual(@as(i32, 51 * 27 + 50), p.estimate(r));
}
