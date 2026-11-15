//! Plans for the biggest plays on the board: a 15-letter word across a whole
//! lane. An edge lane (row 0, row 14, column 0 or column 14) has three
//! triple-word squares, so a word across it is worth 27 times its letters.
//! The middle row and column have two, at their ends; the centre between
//! them is covered by the first play, so a word across is worth 9 times.
//!
//! Premiums count only for tiles placed in that play, so the finishing play
//! must place the triple-word squares itself, and with at most 7 tiles per
//! play at least 8 of the word's letters have to be on the board already.
//! A plan names the word and the squares the finish will fill (`finish`);
//! the other lane squares are setup squares, to be filled with the word's
//! letters by earlier plays. Until the finish is played, the search keeps
//! the finish squares empty, lets setup squares take only their letter, and
//! holds the finishing tiles back.
//!
//! Each finishing tile also makes a cross word with whatever lies beside it
//! off the lane: a hook. A hook through a triple-word square is tripled, so
//! the record single plays add 270 or more points this way. With hooks off,
//! the squares beside the finish squares are kept empty. With hooks on, an
//! edge lane's hooks grow into the board: the square at the open end of
//! each hook accepts only letters that keep the finish letter and the tiles
//! beside it a word (`.strict`), or a prefix or suffix of one (`.prefix`).
//! Middle lanes keep both sides empty.
//!
//! A plan can also name hooks outright: a tail is a word laid beside a
//! finish square, across the lane, so that the finishing tile makes a
//! chosen cross word with it (Hedt's CLAWLIKE: the tail LAWLIKE, his first
//! play, waits for the C of MICROTECHNIQUES on a triple-word square). Tail
//! squares are targets like setup squares: they take only their letter, earn
//! a ranking bonus when filled, and the finish waits until they are.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");
const affix = @import("affix.zig");

const N = rules.N;
const NN = rules.NN;
const Lexicon = lexicon.Lexicon;
const Key = affix.Key;
const none: u8 = 0xff;

pub const Lane = enum {
    row0,
    row14,
    col0,
    col14,
    row7,
    col7,

    pub fn dir(l: Lane) u1 {
        return switch (l) {
            .row0, .row14, .row7 => 0,
            .col0, .col14, .col7 => 1,
        };
    }

    pub fn line(l: Lane) usize {
        return switch (l) {
            .row0, .col0 => 0,
            .row14, .col14 => N - 1,
            .row7, .col7 => rules.CENTER,
        };
    }

    pub fn middle(l: Lane) bool {
        return l.line() == rules.CENTER;
    }

    /// Lane positions the finish must cover: the triple-word squares.
    pub fn required(l: Lane) u16 {
        const ends: u16 = (1 << 0) | (1 << (N - 1));
        return if (l.middle()) ends else ends | (1 << rules.CENTER);
    }

    /// Lane positions the finish cannot cover: the centre of a middle lane,
    /// which the first play takes.
    pub fn forbidden(l: Lane) u16 {
        return if (l.middle()) 1 << rules.CENTER else 0;
    }
};

pub const HookMode = enum {
    /// Keep the squares beside finish squares empty.
    off,
    /// Hooks must be words after every play.
    strict,
    /// Hooks must stay prefixes (top and left lanes) or suffixes (bottom
    /// and right lanes) of words; one that is not yet a word blocks the
    /// finish until it is extended.
    prefix,
};

/// How plans are pursued, shared by every position of a search.
pub const Context = struct {
    hooks: HookMode = .off,
    /// Required unless `hooks` is `.off`.
    affixes: ?*const affix.Affixes = null,
    /// For guarding middle-lane setup squares (substring checks).
    lex: ?*const Lexicon = null,
    /// Ranking bonus per filled setup square of an edge plan.
    goal_bonus: i32 = 0,
    /// The same for a middle plan. The record games finish their middle
    /// word early (an 8-letter core through the centre, then both ends for
    /// x9) and build both edges off it, so this is kept small enough that
    /// the finish pays at once instead of waiting for the end.
    mid_bonus: i32 = 20,
    /// Ranking penalty per square of distance between an unfilled setup
    /// square and the nearest tile, beyond the first, so that the board
    /// grows towards setup squares far from the play.
    approach: i32 = 0,
    /// Ranking bonus for a tail, in percent of its hook's points, shared
    /// among its squares.
    tail_bonus: i32 = 0,
    /// Bonuses of the plans after the first live one, in percent: below
    /// 100 the plans are built one at a time, in the order given.
    later_bonus: i32 = 100,
    /// Board targets: each empty one takes only its letter and earns
    /// `target_bonus` when filled.
    targets: []const Target = &.{},
    target_bonus: i32 = 0,
    /// Keep every other square empty until all targets are filled (to
    /// build a given board again, then place what is left), except the
    /// squares marked in `free`.
    exclusive: bool = false,
    free: ?*const [NN]bool = null,
};

pub const max_tails = 6;

/// A letter wanted on one square, apart from any plan: a word of a
/// designed template, or a tile of a strong board to build again.
pub const Target = struct {
    sq: u8,
    letter: u5,
};

/// Tiles to lay beside a finish square, across the lane, before the finish:
/// the finishing tile then makes the hook word `letter ++ tail` (top and
/// left lanes) or `tail ++ letter` (bottom and right lanes).
pub const Tail = struct {
    /// Lane position of the finish square.
    at: u8,
    len: u8,
    /// In board order along the hook line, top to bottom or left to right.
    letters: [N]u5 = undefined,
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
    /// How many of the setup letters of each kind have no natural tile left
    /// and must be blanks.
    blank_setup: [26]u8 = @splat(0),
    /// Designed hooks (edge lanes only).
    tails: [max_tails]Tail = undefined,
    n_tails: u8 = 0,

    /// Board square of lane position i.
    pub fn square(p: *const Plan, i: usize) usize {
        const l = p.lane.line();
        return if (p.lane.dir() == 0) l * N + i else i * N + l;
    }

    /// Board square of letter j of tail t.
    pub fn tailSquare(p: *const Plan, t: *const Tail, j: usize) usize {
        const k = if (p.lane.line() == 0) 1 + j else N - 1 - @as(usize, t.len) + j;
        return if (p.lane.dir() == 0) k * N + t.at else @as(usize, t.at) * N + k;
    }

    /// The square just past the far end of tail t, which must stay empty
    /// until the finish, or null at the board's edge.
    pub fn tailBeyond(p: *const Plan, t: *const Tail) ?usize {
        const k: isize = if (p.lane.line() == 0) @as(isize, t.len) + 1 else @as(isize, N) - 2 - @as(isize, t.len);
        if (k < 0 or k >= N) return null;
        const ku: usize = @intCast(k);
        return if (p.lane.dir() == 0) ku * N + t.at else @as(usize, t.at) * N + ku;
    }

    /// The tail at lane position i, if any.
    pub fn tailAt(p: *const Plan, i: usize) ?*const Tail {
        for (p.tails[0..p.n_tails]) |*t| {
            if (t.at == i) return t;
        }
        return null;
    }

    /// Every tail square is filled.
    pub fn tailsFilled(p: *const Plan, pos: *const board.Position) bool {
        for (p.tails[0..p.n_tails]) |*t| {
            for (0..t.len) |j| {
                if (pos.cells[0][p.tailSquare(t, j)] == 0) return false;
            }
        }
        return true;
    }

    /// Tail squares filled, one bit per square in tail order (for signatures).
    pub fn tailMask(p: *const Plan, pos: *const board.Position) u64 {
        var m: u64 = 0;
        var b: u6 = 0;
        for (p.tails[0..p.n_tails]) |*t| {
            for (0..t.len) |j| {
                if (pos.cells[0][p.tailSquare(t, j)] != 0) m |= @as(u64, 1) << b;
                b +%= 1;
            }
        }
        return m;
    }

    /// Points tail t adds to the finish (natural tiles assumed).
    pub fn tailValue(p: *const Plan, t: *const Tail) i32 {
        const sq = p.square(t.at);
        var v: i32 = @as(i32, rules.letter_values[p.word[t.at]]) * rules.premiums.letter[0][sq];
        for (t.letters[0..t.len]) |l| v += rules.letter_values[l];
        return v * rules.premiums.word[0][sq];
    }

    /// The hook word tail t makes with its finish letter.
    pub fn hookWord(p: *const Plan, t: *const Tail, out: *[N + 1]u5) []const u5 {
        const n: usize = t.len;
        if (p.lane.line() == 0) {
            out[0] = p.word[t.at];
            @memcpy(out[1 .. n + 1], t.letters[0..n]);
        } else {
            @memcpy(out[0..n], t.letters[0..n]);
            out[n] = p.word[t.at];
        }
        return out[0 .. n + 1];
    }

    /// The squares beside lane position i, off the lane: one for an edge
    /// lane (towards the middle), two for a middle lane.
    fn sides(p: *const Plan, i: usize, out: *[2]usize) []const usize {
        const l = p.lane.line();
        var n: usize = 0;
        if (l > 0) {
            out[n] = if (p.lane.dir() == 0) (l - 1) * N + i else i * N + (l - 1);
            n += 1;
        }
        if (l + 1 < N) {
            out[n] = if (p.lane.dir() == 0) (l + 1) * N + i else i * N + (l + 1);
            n += 1;
        }
        return out[0..n];
    }

    pub fn setup(p: *const Plan) u16 {
        return ~p.finish & ((1 << N) - 1);
    }

    /// The finish has been played (its squares are covered).
    pub fn done(p: *const Plan, pos: *const board.Position) bool {
        return pos.cells[0][p.square(@ctz(p.finish))] != 0;
    }

    /// The lane spells the plan's word (the finish landed as planned, or the
    /// same word was laid some other way).
    pub fn completed(p: *const Plan, pos: *const board.Position) bool {
        for (0..N) |i| {
            const c = pos.cells[0][p.square(i)];
            if (c == 0 or rules.letterOf(c) != p.word[i]) return false;
        }
        return true;
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

    /// Product of the word premiums under the finish (27 on an edge, 9 in
    /// the middle).
    pub fn multiplier(p: *const Plan) i32 {
        var m: i32 = 1;
        var it = p.finish;
        while (it != 0) : (it &= it - 1) m *= rules.premiums.word[0][p.square(@ctz(it))];
        return m;
    }

    /// Ranking bonus per setup square, scaled by the plan's multiplier.
    pub fn bonus(p: *const Plan, goal_bonus: i32) i32 {
        return @divTrunc(goal_bonus * p.multiplier(), 27);
    }

    /// Narrow `restrict` (allowed letters per square) for this plan. With
    /// `open_edges`, an edge lane leaves the squares beside its finish
    /// squares to the hook rules instead of keeping them empty; with
    /// `open_middle`, a middle lane leaves them to its guards.
    pub fn restrictInto(p: *const Plan, restrict: *[NN]u32, open_edges: bool, open_middle: bool) void {
        for (0..N) |i| {
            if ((p.finish >> @intCast(i)) & 1 != 0) {
                restrict[p.square(i)] = 0;
                if (p.tailAt(i) != null) continue;
                if (if (p.lane.middle()) open_middle else open_edges) continue;
                var buf: [2]usize = undefined;
                for (p.sides(i, &buf)) |sq| restrict[sq] = 0;
            } else {
                restrict[p.square(i)] &= @as(u32, 1) << p.word[i];
            }
        }
        // A tail takes only its letters, and nothing may join its far end.
        for (p.tails[0..p.n_tails]) |*t| {
            for (0..t.len) |j| restrict[p.tailSquare(t, j)] &= @as(u32, 1) << t.letters[j];
            if (p.tailBeyond(t)) |sq| restrict[sq] = 0;
        }
    }

    /// The finishing play, if every setup and tail square is filled and it
    /// is legal now. `avail` is what the finish may take, by tile kind: it
    /// uses a natural tile where it can and a blank where it must.
    pub fn finishMove(p: *const Plan, lex: *const Lexicon, r: rules.Rules, pos: *const board.Position, avail: *const [27]u8) ?board.Move {
        if (p.done(pos) or p.filled(pos) != p.setup() or !p.tailsFilled(pos)) return null;
        var text: [N]u8 = undefined;
        var left = avail.*;
        for (0..N) |i| {
            const l = p.word[i];
            const ch: u8 = @as(u8, l) + 'A';
            text[i] = ch;
            if ((p.finish >> @intCast(i)) & 1 == 0) continue;
            if (left[l] > 0) {
                left[l] -= 1;
            } else if (left[rules.BLANK_SLOT] > 0) {
                left[rules.BLANK_SLOT] -= 1;
                text[i] = std.ascii.toLower(ch);
            } else return null;
        }
        const l = p.lane.line();
        const row: usize = if (p.lane.dir() == 0) l else 0;
        const col: usize = if (p.lane.dir() == 0) 0 else l;
        return pos.parsePlay(lex, r, row, col, p.lane.dir(), &text) catch null;
    }

    /// Score of the finish on an empty lane with no cross words: every letter
    /// at face value (blanks for letters the tile set lacks), the new tiles'
    /// letter premiums, times the word premiums, plus the bonus for 7 tiles.
    pub fn estimate(p: *const Plan, r: rules.Rules) i32 {
        var sum: i32 = 0;
        for (0..N) |i| {
            const v: i32 = rules.letter_values[p.word[i]];
            const fin = (p.finish >> @intCast(i)) & 1 != 0;
            const lm: i32 = if (fin) rules.premiums.letter[0][p.square(i)] else 1;
            sum += v * lm;
        }
        // Blanks score nothing (ignoring which square they take).
        for (0..26) |l| sum -= @as(i32, p.blank_for[l] + p.blank_setup[l]) * rules.letter_values[l];
        const tiles = @popCount(p.finish);
        var hooks: i32 = 0;
        for (p.tails[0..p.n_tails]) |*t| hooks += p.tailValue(t);
        return sum * p.multiplier() + (if (tiles == r.rack) r.bingo else 0) + hooks;
    }

    /// Add a tail at finish square `at`; false if the square is not a finish
    /// square, already has one, the lane is a middle lane, or it is too long.
    pub fn addTail(p: *Plan, at: usize, letters: []const u5) bool {
        if (p.lane.middle() or p.n_tails == max_tails) return false;
        if (at >= N or (p.finish >> @intCast(at)) & 1 == 0 or p.tailAt(at) != null) return false;
        if (letters.len == 0 or letters.len > N - 1) return false;
        var t = Tail{ .at = @intCast(at), .len = @intCast(letters.len) };
        @memcpy(t.letters[0..letters.len], letters);
        p.tails[p.n_tails] = t;
        p.n_tails += 1;
        return true;
    }

    /// Every hook word, and every tail of two or more letters, is a word.
    pub fn tailsAreWords(p: *const Plan, lex: *const Lexicon) bool {
        for (p.tails[0..p.n_tails]) |*t| {
            var buf: [N + 1]u5 = undefined;
            if (!lex.contains(p.hookWord(t, &buf))) return false;
            if (t.len >= 2 and !lex.contains(t.letters[0..t.len])) return false;
        }
        return true;
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
    const req = lane.required();
    if (finish & req != req or finish & lane.forbidden() != 0 or @popCount(finish) > r.rack) return error.BadPlan;
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
        p.blank_setup[l] = setup_need -| set[l];
        const left = set[l] -| setup_need;
        const nat = @min(finish_need[l], left);
        p.reserve[l] = nat;
        p.blank_for[l] = finish_need[l] - nat;
        p.reserve[rules.BLANK_SLOT] += finish_need[l] - nat;
    }
    return p;
}

/// One hook of an open plan: the cross word its finishing tile will make.
/// The hook line runs across the lane through the finish square; the tail
/// is the run of tiles beside the finish square on that line, and the
/// joining square is the empty square at the tail's far end.
pub const Hook = struct {
    plan: u8,
    /// Guards an unfilled setup square instead: its letter is not placed by
    /// the finish, and the hook only has to stay buildable (it scores nothing).
    guard: bool = false,
    /// The hook line.
    dir: u1,
    line: u8,
    /// The finish letter comes first (top or left lane) or last.
    first: bool,
    letter: u5,
    tail_len: u8 = 0,
    /// Line index of the joining square, or `none`.
    join: u8 = none,
    /// The hook word with a letter x on the joining square is
    /// key_pre | letterAt(x, slot) | key_post.
    key_pre: Key = 0,
    key_post: Key = 0,
    slot: u8 = 0,
    /// The word could still grow past the run beyond the joining square.
    open: bool = false,
    /// Letters the joining square may take.
    allowed: u32 = 0,
    /// Face value of the tail's tiles, and of the run beyond the joining square.
    tail_value: i32 = 0,
    beyond_value: i32 = 0,
    /// The finishing tile's points with its letter premium, and the finish
    /// square's word premium.
    finish_points: i32,
    wm: i32,
    /// Points the hook adds to the finish now (0 unless the tail makes a word).
    value: i32 = 0,
    /// The tail is not empty and does not make a word: the finish is blocked.
    pending: bool = false,
    /// Pending, and no letter can extend it: the plan cannot finish.
    dead: bool = false,
};

fn computeHook(p: *const Plan, plan_index: usize, i: usize, pos: *const board.Position, ctx: Context, guard: bool) Hook {
    const aff = ctx.affixes.?;
    const hdir: u1 = 1 - p.lane.dir();
    const cells = pos.cells[hdir][i * N ..][0..N];
    const first = p.lane.line() == 0;
    const sq = p.square(i);
    const L = p.word[i];
    var h = Hook{
        .plan = @intCast(plan_index),
        .guard = guard,
        .dir = hdir,
        .line = @intCast(i),
        .first = first,
        .letter = L,
        .finish_points = @as(i32, rules.letter_values[L]) * rules.premiums.letter[0][sq],
        .wm = rules.premiums.word[0][sq],
    };
    var text: [N]u5 = undefined;
    var cur: Key = 0;
    if (first) {
        text[0] = L;
        var n: usize = 1;
        var k: usize = 1;
        while (k < N and cells[k] != 0) : (k += 1) {
            text[n] = rules.letterOf(cells[k]);
            h.tail_value += rules.cellValue(cells[k]);
            n += 1;
        }
        h.tail_len = @intCast(n - 1);
        cur = affix.key(text[0..n]);
        if (k < N) {
            h.join = @intCast(k);
            h.slot = @intCast(n);
            h.key_pre = cur;
            var b = k + 1;
            var m: usize = 0;
            var post: [N]u5 = undefined;
            while (b < N and cells[b] != 0) : (b += 1) {
                post[m] = rules.letterOf(cells[b]);
                h.beyond_value += rules.cellValue(cells[b]);
                m += 1;
            }
            h.key_post = affix.key(post[0..m]) << @intCast(5 * (n + 1));
            h.open = b < N;
        }
    } else {
        // The tail ends next to the finish square, at index N - 2.
        var t0: usize = N - 1;
        while (t0 > 0 and cells[t0 - 1] != 0) t0 -= 1;
        var n: usize = 0;
        for (cells[t0 .. N - 1]) |c| {
            text[n] = rules.letterOf(c);
            h.tail_value += rules.cellValue(c);
            n += 1;
        }
        text[n] = L;
        n += 1;
        h.tail_len = @intCast(n - 1);
        cur = affix.key(text[0..n]);
        if (t0 > 0) {
            const k = t0 - 1;
            h.join = @intCast(k);
            var b0 = k;
            while (b0 > 0 and cells[b0 - 1] != 0) b0 -= 1;
            var pre: [N]u5 = undefined;
            for (cells[b0..k], 0..) |c, j| {
                pre[j] = rules.letterOf(c);
                h.beyond_value += rules.cellValue(c);
            }
            h.slot = @intCast(k - b0);
            h.key_pre = affix.key(pre[0 .. k - b0]);
            h.key_post = cur << @intCast(5 * (h.slot + 1));
            h.open = b0 > 0;
        }
    }
    if (h.tail_len > 0) {
        if (aff.isWord(cur)) {
            h.value = (h.tail_value + h.finish_points) * h.wm;
        } else {
            h.pending = true;
        }
    }
    if (h.join != none) {
        // A guarded setup letter can be laid by a play along the hook line
        // that also extends the run, so its hook need only stay a prefix.
        const mode: HookMode = if (guard) .prefix else ctx.hooks;
        for (0..26) |x| {
            const kx = h.key_pre | affix.letterAt(@intCast(x), h.slot) | h.key_post;
            const ok = switch (mode) {
                .off => true,
                .strict => aff.isWord(kx),
                .prefix => if (!h.open) aff.isWord(kx) else if (first) aff.isPrefix(kx) else aff.isSuffix(kx),
            };
            if (ok) h.allowed |= @as(u32, 1) << @intCast(x);
        }
    }
    h.dead = h.pending and h.allowed == 0;
    return h;
}

/// Guards an empty target square: a setup or finish square of a middle
/// lane, or a tail square. The line through it, across the lane or the
/// hook, can grow both ways, so the tiles beside the square, with its
/// letter between them, must stay a substring of some word: in the GADDAG,
/// a path from the root spelling the string backwards.
pub const Guard = struct {
    /// The run must be a whole word (tails), not just part of one.
    strict: bool = false,
    dir: u1,
    line: u8,
    letter: u5,
    /// Line index of the lane square.
    at: u8,
    /// The run through the lane square (its letter included) and its extent.
    text: [N]u5,
    lo: u8,
    hi: u8,
};

fn isSubstring(lex: *const Lexicon, letters: []const u5) bool {
    var n = lex.root();
    var i = letters.len;
    while (i > 0) {
        i -= 1;
        n = lex.child(n, letters[i]);
        if (n == 0) return false;
    }
    return true;
}

/// The run of tiles on `cells` through index `at` (taken as holding `l`).
fn runThrough(cells: *const [N]u8, at: usize, l: u5, text: *[N]u5) [2]usize {
    var lo = at;
    while (lo > 0 and cells[lo - 1] != 0) lo -= 1;
    var hi = at;
    while (hi + 1 < N and cells[hi + 1] != 0) hi += 1;
    for (lo..hi + 1) |k| text[k - lo] = if (k == at) l else rules.letterOf(cells[k]);
    return .{ lo, hi };
}

pub const max_plans = 8;
pub const max_hooks = 64;
pub const max_guards = 64;

/// Change in hook value (and in blocking hooks) that one play makes.
pub const Delta = struct {
    value: i32 = 0,
    pending: i32 = 0,
};

/// What the open plans demand of one position.
pub const Active = struct {
    /// Plans still worth pursuing (finish not played, setup still suppliable).
    live: u8 = 0,
    restrict: [NN]u32 = @splat(lexicon.LETTERS),
    n_plans: u8 = 0,
    /// Per (direction, line): positions that are empty target squares
    /// (setup or tail squares) of live plans.
    goal_any: [2][N]u16 = @splat(@splat(0)),
    /// Ranking bonus for filling each empty target square (summed over the
    /// plans that want it).
    sq_bonus: [NN]i32 = @splat(0),
    /// Bonus earned by the target squares already filled, by plan, and in all.
    own_bonus: [max_plans]i32 = @splat(0),
    bonus_now: i32 = 0,
    /// Distance penalty now (negative), and how much a tile on each empty
    /// square would reduce it.
    approach_now: i32 = 0,
    approach_gain: [NN]i32 = @splat(0),
    /// The largest of `approach_gain`, a bound for early rejection.
    max_approach: i32 = 0,
    has_approach: bool = false,
    /// Tiles held back for finishes.
    reserve: [27]u8 = @splat(0),
    /// Setup squares already filled, over the live plans.
    filled: u32 = 0,
    /// Tiles of each kind (index 26: blanks) that plays may spend away from
    /// the setup squares without starving a live plan.
    spare: [27]u8 = @splat(N),
    /// Kinds whose spare count one play could exceed (fewer than 7 spare).
    tight: u32 = 0,

    mode: HookMode = .off,
    aff: ?*const affix.Affixes = null,
    hooks: [max_hooks]Hook = undefined,
    n_hooks: u8 = 0,
    /// Per (direction, line): positions that are joining squares of hooks
    /// running across that line.
    join_mask: [2][N]u16 = @splat(@splat(0)),
    /// Hook index by board square of its joining square.
    join_hook: [NN]u8 = @splat(none),
    /// Per (direction, line): the hook whose finish letter comes before the
    /// line's first square, and the one whose letter comes after its last.
    line_hook: [2][N][2]u8 = @splat(@splat(.{ none, none })),
    guards: [max_guards]Guard = undefined,
    n_guards: u8 = 0,
    lex: ?*const Lexicon = null,
    /// Per (direction, line): the guards on that line, one bit each.
    line_guards: [2][N]u64 = @splat(@splat(0)),
    /// Hook points now, and hooks blocking a finish, over the live plans.
    hook_value: i32 = 0,
    hook_pending: i32 = 0,
    plan_hook_value: [max_plans]i32 = @splat(0),
    plan_pending: [max_plans]i32 = @splat(0),

    /// Ranking bonus for the target squares a play fills.
    pub fn goalValue(a: *const Active, dir: u1, line: u8, new_mask: u16) i32 {
        var it = new_mask & a.goal_any[dir][line];
        var v: i32 = 0;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            v += a.sq_bonus[if (dir == 0) @as(usize, line) * N + i else i * N + line];
        }
        return v;
    }

    fn addGoal(a: *Active, sq: usize, bonus: i32) void {
        a.sq_bonus[sq] += bonus;
        a.goal_any[0][sq / N] |= @as(u16, 1) << @intCast(sq % N);
        a.goal_any[1][sq % N] |= @as(u16, 1) << @intCast(sq / N);
    }

    /// Reduction in the distance penalty from a play's new tiles (the best
    /// single tile's, a slight underestimate).
    pub fn approachValue(a: *const Active, dir: u1, line: u8, new_mask: u16) i32 {
        if (!a.has_approach) return 0;
        var best: i32 = 0;
        var it = new_mask;
        while (it != 0) : (it &= it - 1) {
            const i: usize = @ctz(it);
            const sq = if (dir == 0) @as(usize, line) * N + i else i * N + line;
            best = @max(best, a.approach_gain[sq]);
        }
        return best;
    }

    /// Whether a play leaves every live plan enough tiles. Tiles placed on
    /// setup squares count against the setup, not the spare.
    pub fn affordable(a: *const Active, dir: u1, line: u8, new_mask: u16, tiles: *const [N]u8, blanks: u16) bool {
        var off = new_mask & ~a.goal_any[dir][line];
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

    /// Guard the empty square `sq`, which must take letter `l`, along the
    /// line in direction `gdir` through it: restrict the squares just beyond
    /// both ends of the run through it.
    fn addGuard(a: *Active, lex: *const Lexicon, gdir: u1, sq: usize, l: u5, pos: *const board.Position, strict: bool) void {
        if (a.n_guards == max_guards) return;
        const line = if (gdir == 0) sq / N else sq % N;
        const at = if (gdir == 0) sq % N else sq / N;
        var g = Guard{ .strict = strict, .dir = gdir, .line = @intCast(line), .letter = l, .at = @intCast(at), .text = undefined, .lo = 0, .hi = 0 };
        const cells = pos.cells[gdir][line * N ..][0..N];
        const ext = runThrough(cells, at, g.letter, &g.text);
        g.lo = @intCast(ext[0]);
        g.hi = @intCast(ext[1]);
        a.line_guards[gdir][line] |= @as(u64, 1) << @intCast(a.n_guards);
        a.guards[a.n_guards] = g;
        a.n_guards += 1;
        // A tile just beyond either end joins the run (and any run past it).
        for ([2]bool{ true, false }) |low| {
            if (low and ext[0] == 0) continue;
            if (!low and ext[1] == N - 1) continue;
            const j = if (low) ext[0] - 1 else ext[1] + 1;
            const jsq = if (gdir == 0) line * N + j else j * N + line;
            var tmp = cells.*;
            var allowed: u32 = 0;
            for (0..26) |x| {
                tmp[j] = rules.cellOf(@intCast(x), false);
                var text: [N]u5 = undefined;
                const e = runThrough(&tmp, at, g.letter, &text);
                const run = text[0 .. e[1] - e[0] + 1];
                if (if (strict) lex.contains(run) else isSubstring(lex, run)) allowed |= @as(u32, 1) << @intCast(x);
            }
            a.restrict[jsq] &= allowed;
        }
    }

    /// Ranking cost of the setup letters a play takes away from the live
    /// plans: each natural tile beyond the spare forces a blank into a
    /// setup square (losing that letter's points in the plan's word), and
    /// each blank beyond the spare leaves a setup square unsuppliable.
    /// `w` is the cost per point of letter value.
    pub fn letterCost(a: *const Active, w: i32, dir: u1, line: u8, new_mask: u16, tiles: *const [N]u8, blanks: u16) i32 {
        if (a.tight == 0) return 0;
        var off = new_mask & ~a.goal_any[dir][line];
        var used: u32 = 0;
        var count: [27]u8 = undefined;
        var cost: i32 = 0;
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
            if (count[kind] > a.spare[kind]) cost += w * (if (kind == rules.BLANK_SLOT) 15 else rules.letter_values[kind]);
        }
        return cost;
    }

    fn addHook(a: *Active, h: Hook) void {
        if (a.n_hooks == max_hooks) return;
        const idx = a.n_hooks;
        a.hooks[idx] = h;
        a.n_hooks += 1;
        a.line_hook[h.dir][h.line][if (h.first) 0 else 1] = idx;
        if (h.guard) {
            if (h.join != none) a.restrict[if (h.dir == 0) @as(usize, h.line) * N + h.join else @as(usize, h.join) * N + h.line] &= h.allowed;
            return;
        }
        a.hook_value += h.value;
        a.plan_hook_value[h.plan] += h.value;
        if (h.pending) {
            a.hook_pending += 1;
            a.plan_pending[h.plan] += 1;
        }
        if (h.join != none) {
            const sq = if (h.dir == 0) @as(usize, h.line) * N + h.join else @as(usize, h.join) * N + h.line;
            a.restrict[sq] &= h.allowed;
            if (a.join_hook[sq] == none) {
                a.join_hook[sq] = idx;
                a.join_mask[1 - h.dir][h.join] |= @as(u16, 1) << @intCast(h.line);
            }
        }
    }

    /// How a play changes the hooks, or null if it would leave a hook that
    /// can never become a word. Plays that put one tile on a joining square
    /// are already limited to good letters by `restrict`; plays along a
    /// hook line can extend a tail by several tiles and are checked here.
    pub fn hookDelta(a: *const Active, pos: *const board.Position, c: *const movegen.Candidate) ?Delta {
        var d = Delta{};
        var gm = a.line_guards[c.dir][c.line];
        while (gm != 0) : (gm &= gm - 1) {
            const g = &a.guards[@ctz(gm)];
            if (c.end + 1 == g.at or c.start == g.at + 1) {
                if (!a.guardOk(pos, c, g)) return null;
            }
        }
        if (a.n_hooks == 0) return d;
        const lh = a.line_hook[c.dir][c.line];
        if (lh[0] != none and c.start == 1) {
            if (!a.lineHook(&d, pos, c, &a.hooks[lh[0]])) return null;
        }
        if (lh[1] != none and c.end == N - 2) {
            if (!a.lineHook(&d, pos, c, &a.hooks[lh[1]])) return null;
        }
        var it = c.new_mask & a.join_mask[c.dir][c.line];
        while (it != 0) : (it &= it - 1) {
            const j: usize = @ctz(it);
            const sq = if (c.dir == 0) @as(usize, c.line) * N + j else j * N + c.line;
            const h = &a.hooks[a.join_hook[sq]];
            const x = rules.letterOf(c.tiles[j]);
            const blank = (c.blanks >> @intCast(j)) & 1 != 0;
            const k = h.key_pre | affix.letterAt(x, h.slot) | h.key_post;
            const word = a.aff.?.isWord(k);
            const xv: i32 = if (blank) 0 else rules.letter_values[x];
            const v: i32 = if (word) (h.tail_value + h.beyond_value + xv + h.finish_points) * h.wm else 0;
            d.value += v - h.value;
            d.pending += @as(i32, @intFromBool(!word)) - @intFromBool(h.pending);
        }
        return d;
    }

    /// A play along a guard's line that ends next to its square.
    fn guardOk(a: *const Active, pos: *const board.Position, c: *const movegen.Candidate, g: *const Guard) bool {
        var cells = pos.cells[c.dir][@as(usize, c.line) * N ..][0..N].*;
        var it = c.new_mask;
        while (it != 0) : (it &= it - 1) {
            const j: usize = @ctz(it);
            cells[j] = c.tiles[j];
        }
        var text: [N]u5 = undefined;
        const ext = runThrough(&cells, g.at, g.letter, &text);
        const run = text[0 .. ext[1] - ext[0] + 1];
        return if (g.strict) a.lex.?.contains(run) else isSubstring(a.lex.?, run);
    }

    /// A play along hook h's line whose word touches the finish square.
    fn lineHook(a: *const Active, d: *Delta, pos: *const board.Position, c: *const movegen.Candidate, h: *const Hook) bool {
        const aff = a.aff.?;
        var letters: [N + 1]u5 = undefined;
        var n: usize = 0;
        var sum: i32 = 0;
        if (h.first) {
            letters[0] = h.letter;
            n = 1;
        }
        const cells = pos.cells[c.dir][@as(usize, c.line) * N ..][0..N];
        for (c.start..@as(usize, c.end) + 1) |j| {
            if (cells[j] != 0) {
                letters[n] = rules.letterOf(cells[j]);
                sum += rules.cellValue(cells[j]);
            } else {
                const x = rules.letterOf(c.tiles[j]);
                letters[n] = x;
                if ((c.blanks >> @intCast(j)) & 1 == 0) sum += rules.letter_values[x];
            }
            n += 1;
        }
        if (!h.first) {
            letters[n] = h.letter;
            n += 1;
        }
        const k = affix.key(letters[0..n]);
        const word = aff.isWord(k);
        if (!word) {
            if (a.mode != .prefix and !h.guard) return false;
            const open = if (h.first) @as(usize, c.end) + 1 < N else c.start > 0;
            if (!open) return false;
            if (!(if (h.first) aff.isPrefix(k) else aff.isSuffix(k))) return false;
        }
        if (h.guard) return true;
        const v: i32 = if (word) (sum + h.finish_points) * h.wm else 0;
        d.value += v - h.value;
        d.pending += @as(i32, @intFromBool(!word)) - @intFromBool(h.pending);
        return true;
    }
};

/// Whether letter `l` can still be laid on the empty square `sq`. A play
/// across or down puts it there with the perpendicular run through the
/// square as its cross word. That run must make a word with `l` now (the
/// cross-check), or it must be able to grow first: one of its end squares
/// is empty and not held empty by a plan, and the run with `l` is part of
/// some word. The square is lost only when neither direction can work.
fn placeable(pos: *const board.Position, lex: ?*const Lexicon, sq: usize, l: u5, held: *const [NN]bool) bool {
    const r = sq / N;
    const c = sq % N;
    const bit = @as(u32, 1) << l;
    for (0..2) |d| {
        if (pos.xmask[d][if (d == 0) sq else c * N + r] & bit != 0) return true;
        const lx = lex orelse continue;
        // The cross word of a play in direction d runs the other way.
        const row_run = d == 1;
        const line = if (row_run) r else c;
        const at = if (row_run) c else r;
        const cells = pos.cells[if (row_run) 0 else 1][line * N ..][0..N];
        var text: [N]u5 = undefined;
        const ext = runThrough(cells, at, l, &text);
        var grows = false;
        for ([2]isize{ @as(isize, @intCast(ext[0])) - 1, @as(isize, @intCast(ext[1])) + 1 }) |e| {
            if (e < 0 or e >= N) continue;
            const eu: usize = @intCast(e);
            if (!held[if (row_run) line * N + eu else eu * N + line]) grows = true;
        }
        if (grows and isSubstring(lx, text[0 .. ext[1] - ext[0] + 1])) return true;
    }
    return false;
}

/// Every empty setup and tail square of `p` can still take its letter.
fn targetsPlaceable(p: *const Plan, pos: *const board.Position, lex: ?*const Lexicon, held: *const [NN]bool) bool {
    var it = p.setup() & ~p.filled(pos);
    while (it != 0) : (it &= it - 1) {
        const i: usize = @ctz(it);
        if (!placeable(pos, lex, p.square(i), p.word[i], held)) return false;
    }
    for (p.tails[0..p.n_tails]) |*t| {
        for (0..t.len) |j| {
            const sq = p.tailSquare(t, j);
            if (pos.cells[0][sq] == 0 and !placeable(pos, lex, sq, t.letters[j], held)) return false;
        }
    }
    return true;
}

/// Work out which plans are live in `pos` and what they demand. A plan is
/// dropped when one of its target squares can no longer take its letter,
/// when its remaining letters can no longer be supplied, or when one of its
/// hooks can never become a word. A letter with no natural tiles to spare
/// beyond the open target squares is allowed only on those squares.
pub fn activate(plans: []const Plan, pos: *const board.Position, dropped: u8, ctx: Context) Active {
    std.debug.assert(plans.len <= max_plans);
    var a = Active{ .n_plans = @intCast(plans.len), .mode = ctx.hooks, .aff = ctx.affixes };
    var live: u8 = 0;
    for (plans, 0..) |*p, i| {
        if (!p.done(pos)) live |= @as(u8, 1) << @intCast(i);
    }
    live &= ~dropped;
    // Squares the live plans keep empty until their finishes.
    var held: [NN]bool = @splat(false);
    for (plans, 0..) |*p, pi| {
        if ((live >> @intCast(pi)) & 1 == 0) continue;
        var it = p.finish;
        while (it != 0) : (it &= it - 1) held[p.square(@ctz(it))] = true;
        for (p.tails[0..p.n_tails]) |*t| {
            if (p.tailBeyond(t)) |sq| held[sq] = true;
        }
    }
    for (plans, 0..) |*p, pi| {
        if ((live >> @intCast(pi)) & 1 == 0) continue;
        if (!targetsPlaceable(p, pos, ctx.lex, &held)) live &= ~(@as(u8, 1) << @intCast(pi));
    }

    // Hooks of finish squares without tails (when hooks are on) and guards
    // of unfilled setup squares (whenever the affix sets are there), for
    // edge lanes.
    const hooks_on = ctx.hooks != .off and ctx.affixes != null;
    var hk: [max_plans][N]Hook = undefined;
    var nh: [max_plans]u8 = @splat(0);
    if (ctx.affixes != null) {
        for (plans, 0..) |*p, pi| {
            if ((live >> @intCast(pi)) & 1 == 0 or p.lane.middle()) continue;
            var tailed: u16 = 0;
            for (p.tails[0..p.n_tails]) |*t| tailed |= @as(u16, 1) << @intCast(t.at);
            const squares = (if (hooks_on) p.finish & ~tailed else 0) | (p.setup() & ~p.filled(pos));
            var it = squares;
            while (it != 0) : (it &= it - 1) {
                const i: usize = @ctz(it);
                const h = computeHook(p, pi, i, pos, ctx, (p.finish >> @intCast(i)) & 1 == 0);
                hk[pi][nh[pi]] = h;
                nh[pi] += 1;
                if (h.dead) live &= ~(@as(u8, 1) << @intCast(pi));
            }
        }
    }

    while (live != 0) {
        var reserve: [27]u8 = @splat(0);
        var need: [26]u8 = @splat(0);
        for (plans, 0..) |*p, i| {
            if ((live >> @intCast(i)) & 1 == 0) continue;
            for (0..27) |t| reserve[t] += p.reserve[t];
            var it = p.setup() & ~p.filled(pos);
            while (it != 0) : (it &= it - 1) need[p.word[@ctz(it)]] += 1;
            for (p.tails[0..p.n_tails]) |*t| {
                for (0..t.len) |j| {
                    if (pos.cells[0][p.tailSquare(t, j)] == 0) need[t.letters[j]] += 1;
                }
            }
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
            // Scarce letters may go only where a target square wants them.
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
                    for (p.tails[0..p.n_tails]) |*t| {
                        for (0..t.len) |j| {
                            if (t.letters[j] == l) a.restrict[p.tailSquare(t, j)] |= bit;
                        }
                    }
                }
            }
            break;
        }
        // Drop the last live plan and try again.
        live &= ~(@as(u8, 1) << @intCast(7 - @clz(live)));
    }
    a.live = live;
    const first_live: usize = if (live == 0) 0 else @ctz(live);
    for (plans, 0..) |*p, pi| {
        if ((live >> @intCast(pi)) & 1 == 0) continue;
        p.restrictInto(&a.restrict, hooks_on, ctx.lex != null);
        const scale: i32 = if (pi == first_live) 100 else ctx.later_bonus;
        const b = @divTrunc((if (p.lane.middle()) ctx.mid_bonus else p.bonus(ctx.goal_bonus)) * scale, 100);
        var it = p.setup();
        while (it != 0) : (it &= it - 1) {
            const sq = p.square(@ctz(it));
            if (pos.cells[0][sq] != 0) {
                a.own_bonus[pi] += b;
                a.filled += 1;
            } else a.addGoal(sq, b);
        }
        for (p.tails[0..p.n_tails]) |*t| {
            const tb = @divTrunc(p.tailValue(t) * ctx.tail_bonus * scale, 100 * 100 * @as(i32, t.len));
            for (0..t.len) |j| {
                const sq = p.tailSquare(t, j);
                if (pos.cells[0][sq] != 0) a.own_bonus[pi] += tb else a.addGoal(sq, tb);
            }
        }
        a.bonus_now += a.own_bonus[pi];
        for (hk[pi][0..nh[pi]]) |h| a.addHook(h);
    }
    if (ctx.lex) |lex| {
        a.lex = lex;
        for (plans, 0..) |*p, pi| {
            if ((live >> @intCast(pi)) & 1 == 0) continue;
            if (p.lane.middle()) {
                // Setup squares and finish squares alike: the finish's cross
                // words must still be possible when it is played.
                var it = (p.setup() & ~p.filled(pos)) | p.finish;
                while (it != 0) : (it &= it - 1) {
                    const i: usize = @ctz(it);
                    a.addGuard(lex, 1 - p.lane.dir(), p.square(i), p.word[i], pos, false);
                }
            }
            // Tail squares, across their hook line.
            for (p.tails[0..p.n_tails]) |*t| {
                for (0..t.len) |j| {
                    const sq = p.tailSquare(t, j);
                    if (pos.cells[0][sq] == 0) a.addGuard(lex, p.lane.dir(), sq, t.letters[j], pos, true);
                }
            }
        }
    }
    var open_targets: usize = 0;
    var wanted: [NN]bool = @splat(false);
    for (ctx.targets) |t| {
        wanted[t.sq] = true;
        if (pos.cells[0][t.sq] != 0) {
            a.bonus_now += ctx.target_bonus;
            continue;
        }
        open_targets += 1;
        a.restrict[t.sq] &= @as(u32, 1) << t.letter;
        a.addGoal(t.sq, ctx.target_bonus);
    }
    if (ctx.exclusive and open_targets != 0) {
        for (0..NN) |sq| {
            if (!wanted[sq] and !(if (ctx.free) |f| f[sq] else false)) a.restrict[sq] = 0;
        }
    }
    if (ctx.approach > 0 and live != 0) computeApproach(&a, plans, pos, ctx.approach);
    return a;
}

/// Why plan `i` is not live in `pos` (for tracing a game against plans).
pub fn diagnose(plans: []const Plan, pos: *const board.Position, dropped: u8, ctx: Context, i: usize, w: *std.Io.Writer) !void {
    const p = &plans[i];
    if (p.done(pos)) return w.writeAll(if (p.completed(pos)) "finished" else "a finish square was covered");
    if ((dropped >> @intCast(i)) & 1 != 0) return w.writeAll("given up earlier");
    var held: [NN]bool = @splat(false);
    for (plans) |*q| {
        var it = q.finish;
        while (it != 0) : (it &= it - 1) held[q.square(@ctz(it))] = true;
        for (q.tails[0..q.n_tails]) |*t| {
            if (q.tailBeyond(t)) |sq| held[sq] = true;
        }
    }
    var it = p.setup() & ~p.filled(pos);
    while (it != 0) : (it &= it - 1) {
        const k: usize = @ctz(it);
        const sq = p.square(k);
        if (!placeable(pos, ctx.lex, sq, p.word[k], &held)) return w.print("setup square ({d},{d}) can no longer take {c}", .{ sq / N, sq % N, @as(u8, p.word[k]) + 'A' });
    }
    for (p.tails[0..p.n_tails]) |*t| {
        for (0..t.len) |j| {
            const sq = p.tailSquare(t, j);
            if (pos.cells[0][sq] == 0 and !placeable(pos, ctx.lex, sq, t.letters[j], &held))
                return w.print("tail square ({d},{d}) can no longer take {c}", .{ sq / N, sq % N, @as(u8, t.letters[j]) + 'A' });
        }
    }
    if (ctx.affixes != null and !p.lane.middle()) {
        var tailed: u16 = 0;
        for (p.tails[0..p.n_tails]) |*t| tailed |= @as(u16, 1) << @intCast(t.at);
        const hooks_on = ctx.hooks != .off;
        var sq_it = (if (hooks_on) p.finish & ~tailed else 0) | (p.setup() & ~p.filled(pos));
        while (sq_it != 0) : (sq_it &= sq_it - 1) {
            const k: usize = @ctz(sq_it);
            const h = computeHook(p, i, k, pos, ctx, (p.finish >> @intCast(k)) & 1 == 0);
            if (h.dead) return w.print("the hook at lane square {d} can never be a word", .{k});
        }
    }
    return w.writeAll("short of tiles, together with the other plans");
}

fn manhattan(a: usize, b: usize) i32 {
    const ar: i32 = @intCast(a / N);
    const ac: i32 = @intCast(a % N);
    const br: i32 = @intCast(b / N);
    const bc: i32 = @intCast(b % N);
    return @intCast(@abs(ar - br) + @abs(ac - bc));
}

/// Distance of each unfilled setup square to the nearest tile (to the
/// centre on an empty board), and what a tile on each empty square saves.
fn computeApproach(a: *Active, plans: []const Plan, pos: *const board.Position, w: i32) void {
    var targets: [64]usize = undefined;
    var excess: [64]i32 = undefined;
    var nt: usize = 0;
    var tiles: [NN]u8 = undefined;
    var n_tiles: usize = 0;
    for (0..NN) |sq| {
        if (pos.cells[0][sq] != 0) {
            tiles[n_tiles] = @intCast(sq);
            n_tiles += 1;
        }
    }
    const centre = rules.CENTER * N + rules.CENTER;
    var empty_targets: [64]usize = undefined;
    var ne: usize = 0;
    for (plans, 0..) |*p, pi| {
        if ((a.live >> @intCast(pi)) & 1 == 0) continue;
        var it = p.setup() & ~p.filled(pos);
        while (it != 0) : (it &= it - 1) {
            if (ne == empty_targets.len) break;
            empty_targets[ne] = p.square(@ctz(it));
            ne += 1;
        }
        for (p.tails[0..p.n_tails]) |*t| {
            for (0..t.len) |j| {
                const sq = p.tailSquare(t, j);
                if (ne == empty_targets.len or pos.cells[0][sq] != 0) continue;
                empty_targets[ne] = sq;
                ne += 1;
            }
        }
    }
    for (empty_targets[0..ne]) |s| {
        var d: i32 = if (n_tiles == 0) manhattan(s, centre) + 1 else std.math.maxInt(i32);
        for (tiles[0..n_tiles]) |t| d = @min(d, manhattan(s, t));
        if (d <= 1) continue;
        targets[nt] = s;
        excess[nt] = d - 1;
        nt += 1;
    }
    if (nt == 0) return;
    a.has_approach = true;
    for (excess[0..nt]) |e| a.approach_now -= w * e;
    for (0..NN) |q| {
        if (pos.cells[0][q] != 0) continue;
        var g: i32 = 0;
        for (targets[0..nt], excess[0..nt]) |s, e| g += @max(0, e - (manhattan(q, s) - 1));
        a.approach_gain[q] = w * g;
        a.max_approach = @max(a.max_approach, w * g);
    }
}

/// Recompute the reserves of several plans together, so they do not count
/// on the same tiles: setup letters of every plan take natural tiles first,
/// then each finish takes the naturals left, and blanks cover the rest.
pub fn reserveJointly(plans: []Plan, r: rules.Rules) MakeError!void {
    var left = rules.standard_counts;
    left[rules.BLANK_SLOT] = r.blanks;
    for (plans) |*p| {
        p.blank_setup = @splat(0);
        var it = p.setup();
        while (it != 0) : (it &= it - 1) {
            const l = p.word[@ctz(it)];
            if (left[l] > 0) {
                left[l] -= 1;
            } else if (left[rules.BLANK_SLOT] > 0) {
                left[rules.BLANK_SLOT] -= 1;
                p.blank_setup[l] += 1;
            } else return error.TilesUnavailable;
        }
        for (p.tails[0..p.n_tails]) |*t| {
            for (t.letters[0..t.len]) |l| {
                if (left[l] > 0) {
                    left[l] -= 1;
                } else if (left[rules.BLANK_SLOT] > 0) {
                    left[rules.BLANK_SLOT] -= 1;
                } else return error.TilesUnavailable;
            }
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

/// Blanks the plan's letters use (setup and finish).
pub fn blanksUsed(p: *const Plan) u32 {
    var n: u32 = 0;
    for (0..26) |l| n += p.blank_for[l] + p.blank_setup[l];
    return n;
}

/// Parse "row0:WORD:0,1,2,4,7,11,14", optionally followed by tails as
/// ":0=IFF,7=LAWLIKE" (finish square = tail letters, in board order).
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
    var p = try make(lane, word, finish, r);
    if (it.next()) |tails| {
        var ts = std.mem.splitScalar(u8, tails, ',');
        while (ts.next()) |spec| {
            const eq = std.mem.indexOfScalar(u8, spec, '=') orelse return error.BadPlan;
            const at = try std.fmt.parseInt(u4, spec[0..eq], 10);
            const letters_text = spec[eq + 1 ..];
            if (letters_text.len == 0 or letters_text.len > N - 1) return error.BadPlan;
            var letters: [N]u5 = undefined;
            for (letters_text, 0..) |ch, i| {
                if (!std.ascii.isAlphabetic(ch)) return error.BadPlan;
                letters[i] = @intCast(std.ascii.toUpper(ch) - 'A');
            }
            if (!p.addTail(at, letters[0..letters_text.len])) return error.BadPlan;
        }
    }
    if (it.next() != null) return error.BadPlan;
    return p;
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
    for (p.tails[0..p.n_tails], 0..) |*t, k| {
        try w.print("{s}{d}=", .{ if (k == 0) ":" else ",", t.at });
        for (t.letters[0..t.len]) |l| try w.writeByte(@as(u8, l) + 'A');
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
/// is at most `max_run`, best estimate first. Finishes place 7 tiles.
pub fn enumerate(gpa: std.mem.Allocator, lex: *const Lexicon, words_text: []const u8, lane: Lane, max_run: usize, r: rules.Rules) ![]Scored {
    var out: std.ArrayList(Scored) = .empty;
    errdefer out.deinit(gpa);
    const req = lane.required();
    var others_buf: [N]u4 = undefined;
    var n_others: usize = 0;
    for (0..N) |i| {
        if (((req | lane.forbidden()) >> @intCast(i)) & 1 == 0) {
            others_buf[n_others] = @intCast(i);
            n_others += 1;
        }
    }
    const e = Enum{
        .gpa = gpa,
        .lex = lex,
        .out = &out,
        .lane = lane,
        .req = req,
        .others = others_buf[0..n_others],
        .extra = r.rack - @popCount(req),
        .max_run = max_run,
        .r = r,
    };
    var tok = std.mem.tokenizeAny(u8, words_text, " \t\r\n");
    while (tok.next()) |w| {
        if (w.len != N) continue;
        var letters: [N]u5 = undefined;
        for (w, 0..) |ch, i| letters[i] = @intCast(std.ascii.toUpper(ch) - 'A');
        try e.choose(w, &letters, req, 0, 0);
    }
    std.sort.pdq(Scored, out.items, {}, Scored.better);
    return out.toOwnedSlice(gpa);
}

const Enum = struct {
    gpa: std.mem.Allocator,
    lex: *const Lexicon,
    out: *std.ArrayList(Scored),
    lane: Lane,
    req: u16,
    others: []const u4,
    extra: usize,
    max_run: usize,
    r: rules.Rules,

    /// Add `extra - depth` more finish squares from others[from..].
    fn choose(e: *const Enum, w: []const u8, letters: *const [N]u5, finish: u16, from: usize, depth: usize) !void {
        if (depth == e.extra) {
            const run = longestRun(finish);
            if (run > e.max_run) return;
            if (!runsAreWords(e.lex, letters, finish)) return;
            const p = make(e.lane, w, finish, e.r) catch return;
            try e.out.append(e.gpa, .{ .plan = p, .estimate = p.estimate(e.r), .longest_run = @intCast(run), .difficulty = difficulty(&p) });
            return;
        }
        for (from..e.others.len) |k| {
            try e.choose(w, letters, finish | (@as(u16, 1) << e.others[k]), k + 1, depth + 1);
        }
    }
};

fn difficulty(p: *const Plan) u8 {
    var d: u8 = 0;
    var it = p.setup();
    while (it != 0) : (it &= it - 1) {
        const n = rules.standard_counts[p.word[@ctz(it)]];
        d += if (n == 1) 4 else if (n <= 3) 2 else 1;
    }
    return d;
}

/// True if the plans' words and tails fit the tile set together with at
/// most `blanks` blanks standing in.
pub fn fitTogether(plans: []const Plan, blanks: u32) bool {
    var need: [26]u32 = @splat(0);
    for (plans) |*p| {
        for (p.word) |l| need[l] += 1;
        for (p.tails[0..p.n_tails]) |*t| {
            for (t.letters[0..t.len]) |l| need[l] += 1;
        }
    }
    var short: u32 = 0;
    for (0..26) |l| short += need[l] -| rules.standard_counts[l];
    return short <= blanks;
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
    p.restrictInto(&restrict, false, false);
    try std.testing.expectEqual(@as(u32, 0), restrict[0]); // (0,0) kept free
    try std.testing.expectEqual(@as(u32, 0), restrict[N + 0]); // (1,0) too
    try std.testing.expectEqual(@as(u32, 1 << 15), restrict[3]); // (0,3) only P
    // Face 41, Z on the double letter at (0,11): (41 + 10) * 27 + 50.
    try std.testing.expectEqual(@as(i32, 51 * 27 + 50), p.estimate(r));
    try std.testing.expectEqual(@as(i32, 27), p.multiplier());
}

test "a middle plan: x9, both sides kept free, centre is setup" {
    const r = rules.Rules{};
    try std.testing.expectError(error.BadPlan, parse("row7:OXYPHENBUTAZONE:0,1,2,4,7,11,14", r));
    const p = try parse("row7:OXYPHENBUTAZONE:0,1,2,4,6,11,14", r);
    try std.testing.expectEqual(@as(i32, 9), p.multiplier());
    var restrict: [NN]u32 = @splat(lexicon.LETTERS);
    p.restrictInto(&restrict, true, false);
    try std.testing.expectEqual(@as(u32, 0), restrict[6 * N + 0]);
    try std.testing.expectEqual(@as(u32, 0), restrict[8 * N + 0]);
    try std.testing.expectEqual(@as(u32, 1 << 1), restrict[7 * N + 7]); // centre: B
    // Face 41 plus Z doubled: 51 * 9 + 50.
    try std.testing.expectEqual(@as(i32, 51 * 9 + 50), p.estimate(r));
}

test "hooks limit the squares beside finish squares" {
    const gpa = std.testing.allocator;
    var aff = try affix.Affixes.build(gpa, "OH\nOX\nON\nBRA\nBRAIN\n");
    defer aff.deinit(gpa);
    const r = rules.Rules{};
    const plans = [_]Plan{try parse("row0:OXYPHENBUTAZONE:0,1,3,4,7,11,14", r)};
    const pos = board.Position.init(r);
    const bit = struct {
        fn f(ch: u8) u32 {
            return @as(u32, 1) << @intCast(ch - 'A');
        }
    }.f;
    // Under the O at (0,0): only letters that make a word with it.
    const strict = activate(&plans, &pos, 0, .{ .hooks = .strict, .affixes = &aff });
    try std.testing.expectEqual(bit('H') | bit('N') | bit('X'), strict.restrict[1 * N + 0]);
    // Under the B at (0,7) no two-letter word starts with B, but BR starts BRA.
    try std.testing.expectEqual(@as(u32, 0), strict.restrict[1 * N + 7]);
    const prefix = activate(&plans, &pos, 0, .{ .hooks = .prefix, .affixes = &aff });
    try std.testing.expectEqual(bit('R'), prefix.restrict[1 * N + 7]);
    // With hooks off the squares beside finish squares stay empty.
    const off = activate(&plans, &pos, 0, .{ .affixes = &aff });
    try std.testing.expectEqual(@as(u32, 0), off.restrict[1 * N + 0]);
}

test "tails: squares, restrictions, hook value and the plan text" {
    const r = rules.Rules{};
    const p = try parse("row0:MICROTECHNIQUES:0,2,7,9,11,12,14:0=IFF,7=LAWLIKE", r);
    try std.testing.expectEqual(@as(u8, 2), p.n_tails);
    // LAWLIKE runs down column 7 from row 1; its E lands on the centre.
    const t = p.tailAt(7).?;
    try std.testing.expectEqual(@as(usize, 1 * N + 7), p.tailSquare(t, 0));
    try std.testing.expectEqual(@as(usize, 7 * N + 7), p.tailSquare(t, 6));
    try std.testing.expectEqual(@as(?usize, 8 * N + 7), p.tailBeyond(t));
    // CLAWLIKE through the triple word: (3 + 1+1+4+1+1+5+1) * 3.
    try std.testing.expectEqual(@as(i32, 17 * 3), p.tailValue(t));
    var restrict: [NN]u32 = @splat(lexicon.LETTERS);
    p.restrictInto(&restrict, false, false);
    try std.testing.expectEqual(@as(u32, 1 << ('L' - 'A')), restrict[1 * N + 7]);
    try std.testing.expectEqual(@as(u32, 0), restrict[8 * N + 7]); // nothing joins the far end
    try std.testing.expectEqual(@as(u32, 0), restrict[1 * N + 2]); // no tail at 2: kept empty
    // A bottom lane's tail ends next to the finish square.
    const b = try parse("row14:OXYPHENBUTAZONE:0,1,2,3,7,11,14:7=JAM", r);
    const bt = b.tailAt(7).?;
    try std.testing.expectEqual(@as(usize, 11 * N + 7), b.tailSquare(bt, 0));
    try std.testing.expectEqual(@as(usize, 13 * N + 7), b.tailSquare(bt, 2));
    try std.testing.expectEqual(@as(?usize, 10 * N + 7), b.tailBeyond(bt));
    // Written back as parsed.
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try write(&p, &w);
    try std.testing.expectEqualStrings("row0:MICROTECHNIQUES:0,2,7,9,11,12,14:0=IFF,7=LAWLIKE", w.buffered());
    // Tails only at finish squares, and not on middle lanes.
    try std.testing.expectError(error.BadPlan, parse("row0:MICROTECHNIQUES:0,2,7,9,11,12,14:1=AB", r));
    try std.testing.expectError(error.BadPlan, parse("row7:UNPROGRESSIVELY:0,1,10,11,12,13,14:0=AB", r));
}

test "a target square stays reachable while the run beside it can still grow" {
    const gpa = std.testing.allocator;
    var lex = try lexicon.build(gpa, "TAB\nBAT\nAT\nTA\nKABS\n");
    defer lex.deinit(gpa);
    var pos = board.Position.init(.{});
    // AB across at (7,7); A above and T below the target square (7,6).
    for ([_][3]u8{ .{ 7, 7, 'A' }, .{ 7, 8, 'B' }, .{ 6, 6, 'A' }, .{ 8, 6, 'T' } }) |t| {
        const c = rules.cellOf(@intCast(t[2] - 'A'), false);
        pos.cells[0][@as(usize, t[0]) * N + t[1]] = c;
        pos.cells[1][@as(usize, t[1]) * N + t[0]] = c;
        pos.tiles_on_board += 1;
    }
    pos.recomputeCrossChecks(&lex);
    var held: [NN]bool = @splat(false);
    const sq = 7 * N + 6;
    // T makes TAB across: fine now.
    try std.testing.expect(placeable(&pos, &lex, sq, 'T' - 'A', &held));
    // X: XAB and AXT are parts of no word, however the runs grow.
    try std.testing.expect(!placeable(&pos, &lex, sq, 'X' - 'A', &held));
    // K: KAB is not a word yet, but KABS is, while the row can still grow.
    try std.testing.expect(placeable(&pos, &lex, sq, 'K' - 'A', &held));
    held[7 * N + 5] = true;
    held[7 * N + 9] = true;
    try std.testing.expect(!placeable(&pos, &lex, sq, 'K' - 'A', &held));
}
