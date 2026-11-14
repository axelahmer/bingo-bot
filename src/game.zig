//! The rules of a whole game on top of board.zig: players taking turns,
//! racks, the bag, passing, and the end of the game with its rack
//! adjustment.
//!
//! Draws are chosen, not random: the engine looks for the best possible
//! game, so the luck of the draw is ours to pick. A rack is then described
//! by its size alone, and its contents are fixed only when its tiles are
//! played. Every play must fit the mover's rack size, which follows the
//! real refill rule (7 tiles while the bag lasts). Under that one
//! constraint, any sequence of turns is a real game: at every moment, give
//! each player the next tiles of that player's own future plays, plus at the
//! end the tiles they are left holding. `realize` builds exactly those
//! racks and draws and replays the game against them.

const std = @import("std");
const rules = @import("rules.zig");
const lexicon = @import("lexicon.zig");
const board = @import("board.zig");
const movegen = @import("movegen.zig");

const Lexicon = lexicon.Lexicon;
const Move = board.Move;

pub const End = enum { none, went_out, scoreless };

pub const Turn = union(enum) {
    play: Move,
    pass,
};

/// Every play places a tile and at most six scoreless turns follow one
/// another, so no game is longer than this.
pub const max_turns = 2 * 100 + 8;

/// Tile kinds from cheapest to dearest: blanks, then letters by value.
pub const cheapest_first: [27]u5 = blk: {
    var order: [27]u5 = undefined;
    order[0] = rules.BLANK_SLOT;
    var n: usize = 1;
    for (1..11) |v| {
        for (0..26) |l| {
            if (rules.letter_values[l] == v) {
                order[n] = l;
                n += 1;
            }
        }
    }
    break :blk order;
};

pub const Game = struct {
    pos: board.Position,
    rules: rules.Rules,
    /// Tiles on each player's rack.
    rack: [2]u8,
    /// Tiles not yet drawn.
    bag: u8,
    score: [2]i32,
    /// The player to move (always 0 in solitaire).
    turn: u1,
    turns: u16,
    /// Consecutive turns that scored nothing (passes and zero-point plays).
    scoreless: u8,
    end: End,
    /// End-of-game rack adjustment, already included in `score`.
    adjust: [2]i32,

    pub fn init(r: rules.Rules) Game {
        std.debug.assert(r.players == 1 or r.players == 2);
        var g = Game{
            .pos = .init(r),
            .rules = r,
            .rack = .{ 0, 0 },
            .bag = 0,
            .score = .{ 0, 0 },
            .turn = 0,
            .turns = 0,
            .scoreless = 0,
            .end = .none,
            .adjust = .{ 0, 0 },
        };
        g.bag = g.pos.pool_total;
        for (0..r.players) |p| g.draw(@intCast(p));
        return g;
    }

    pub fn over(g: *const Game) bool {
        return g.end != .none;
    }

    /// Combined score of all players.
    pub fn total(g: *const Game) i32 {
        return g.score[0] + g.score[1];
    }

    fn draw(g: *Game, p: u1) void {
        const d = @min(g.rules.rack - g.rack[p], g.bag);
        g.rack[p] += d;
        g.bag -= d;
    }

    /// What the player to move may play from: any unplayed tile (the draws
    /// are ours), but no more tiles than their rack holds.
    pub fn supply(g: *const Game) movegen.Supply {
        return .{ .counts = g.pos.pool, .max_tiles = g.rack[g.turn] };
    }

    pub fn play(g: *Game, lex: *const Lexicon, m: *const Move) void {
        std.debug.assert(!g.over() and m.placed <= g.rack[g.turn]);
        g.pos.apply(lex, m);
        g.score[g.turn] += m.score;
        g.rack[g.turn] -= m.placed;
        g.draw(g.turn);
        g.turns += 1;
        g.scoreless = if (m.score == 0) g.scoreless + 1 else 0;
        if (g.bag == 0 and g.rack[g.turn] == 0) return g.finish(.went_out);
        if (g.scoreless >= g.rules.scoreless_limit) return g.finish(.scoreless);
        g.next();
    }

    pub fn pass(g: *Game) void {
        std.debug.assert(!g.over());
        g.turns += 1;
        g.scoreless += 1;
        if (g.scoreless >= g.rules.scoreless_limit) return g.finish(.scoreless);
        g.next();
    }

    fn next(g: *Game) void {
        if (g.rules.players == 2) g.turn ^= 1;
    }

    fn finish(g: *Game, how: End) void {
        g.end = how;
        const held = g.heldTiles();
        switch (how) {
            .went_out => if (g.rules.players == 2) {
                const other = g.turn ^ 1;
                const v = rules.countsValue(&held[other]);
                switch (g.rules.end_rule) {
                    .wespa => {
                        g.adjust[g.turn] = v;
                        g.adjust[other] = -v;
                    },
                    .naspa => g.adjust[g.turn] = 2 * v,
                }
            },
            .scoreless => for (0..g.rules.players) |p| {
                g.adjust[p] = -rules.countsValue(&held[p]);
            },
            .none => unreachable,
        }
        g.score[0] += g.adjust[0];
        g.score[1] += g.adjust[1];
    }

    /// The tiles each player holds. After going out, the other player holds
    /// everything unplayed. Otherwise racks hold the cheapest unplayed tiles,
    /// player 0 first: the draws were ours to choose, and the split between
    /// racks does not change the combined score.
    pub fn heldTiles(g: *const Game) [2][27]u8 {
        var out: [2][27]u8 = @splat(@splat(0));
        if (g.end == .went_out) {
            if (g.rules.players == 2) out[g.turn ^ 1] = g.pos.pool;
            return out;
        }
        var left = g.pos.pool;
        for (0..g.rules.players) |p| {
            var need = g.rack[p];
            for (cheapest_first) |slot| {
                const t = @min(need, left[slot]);
                out[p][slot] += t;
                left[slot] -= t;
                need -= t;
            }
        }
        return out;
    }
};

/// Racks and draws of a game replayed with real tiles.
pub const Realized = struct {
    /// Each player's rack after the opening draw.
    opening: [2][27]u8,
    /// The mover's rack before each turn, and what they drew after it.
    rack_before: [max_turns][27]u8,
    drawn: [max_turns][27]u8,
    /// What each player holds at the end.
    final: [2][27]u8,
};

pub const RealizeError = error{ TooManyTurns, TurnAfterEnd, RackTooSmall, DrawFailed, NotOnRack, FinalRacksDiffer };

fn slotOf(cell: u8) u5 {
    return if (rules.isBlank(cell)) rules.BLANK_SLOT else rules.letterOf(cell);
}

/// Replay `turns` (plays already legal on the board, e.g. from parsePlay or
/// the generator) under the full rules, then rebuild real racks and draws
/// for them and replay the game again with those racks, checking that every
/// play comes off the mover's rack and every draw comes out of the bag.
/// Returns the final game state.
pub fn realize(lex: *const Lexicon, r: rules.Rules, turns: []const Turn, out: *Realized) RealizeError!Game {
    if (turns.len > max_turns) return error.TooManyTurns;

    // 1. Under the rack-size rule.
    var g = Game.init(r);
    for (turns) |t| {
        if (g.over()) return error.TurnAfterEnd;
        switch (t) {
            .pass => g.pass(),
            .play => |m| {
                if (m.placed > g.rack[g.turn]) return error.RackTooSmall;
                g.play(lex, &m);
            },
        }
    }

    // 2. Each player's tiles in the order they need them: the tiles of
    // their plays, then what they hold at the end.
    var seq: [2][100]u5 = undefined;
    var len: [2]usize = .{ 0, 0 };
    var mover: u1 = 0;
    for (turns) |t| {
        switch (t) {
            .play => |m| {
                var it = m.new_mask;
                while (it != 0) : (it &= it - 1) {
                    seq[mover][len[mover]] = slotOf(m.tiles[@ctz(it)]);
                    len[mover] += 1;
                }
            },
            .pass => {},
        }
        if (r.players == 2) mover ^= 1;
    }
    const held = g.heldTiles();
    for (0..r.players) |p| {
        for (0..27) |s| {
            for (0..held[p][s]) |_| {
                seq[p][len[p]] = @intCast(s);
                len[p] += 1;
            }
        }
    }

    // 3. Replay with real racks: each draw takes the next tiles the drawer needs.
    const Replay = struct {
        bag: [27]u8,
        bag_total: usize,
        rack: [2][27]u8 = @splat(@splat(0)),
        size: [2]usize = .{ 0, 0 },
        next: [2]usize = .{ 0, 0 },
        seq: *const [2][100]u5,
        len: [2]usize,
        rack_size: usize,

        fn draw(st: *@This(), p: usize, into: ?*[27]u8) RealizeError!void {
            const d = @min(st.rack_size - st.size[p], st.bag_total);
            for (0..d) |_| {
                if (st.next[p] >= st.len[p]) return error.DrawFailed;
                const t = st.seq[p][st.next[p]];
                st.next[p] += 1;
                if (st.bag[t] == 0) return error.DrawFailed;
                st.bag[t] -= 1;
                st.bag_total -= 1;
                st.rack[p][t] += 1;
                st.size[p] += 1;
                if (into) |x| x[t] += 1;
            }
        }
    };
    const start = board.Position.init(r);
    var rp = Replay{ .bag = start.pool, .bag_total = start.pool_total, .seq = &seq, .len = len, .rack_size = r.rack };
    for (0..r.players) |p| try rp.draw(p, null);
    out.opening = rp.rack;
    mover = 0;
    for (turns, 0..) |t, k| {
        out.rack_before[k] = rp.rack[mover];
        out.drawn[k] = @splat(0);
        switch (t) {
            .play => |m| {
                var it = m.new_mask;
                while (it != 0) : (it &= it - 1) {
                    const sl = slotOf(m.tiles[@ctz(it)]);
                    if (rp.rack[mover][sl] == 0) return error.NotOnRack;
                    rp.rack[mover][sl] -= 1;
                    rp.size[mover] -= 1;
                }
                try rp.draw(mover, &out.drawn[k]);
            },
            .pass => {},
        }
        if (r.players == 2) mover ^= 1;
    }
    out.final = rp.rack;
    for (0..r.players) |p| {
        if (!std.mem.eql(u8, &rp.rack[p], &held[p])) return error.FinalRacksDiffer;
    }
    return g;
}

test "cheapest_first orders tiles by value" {
    try std.testing.expectEqual(@as(u5, rules.BLANK_SLOT), cheapest_first[0]);
    var last: i32 = 0;
    for (cheapest_first[1..]) |l| {
        try std.testing.expect(rules.letter_values[l] >= last);
        last = rules.letter_values[l];
    }
}

test "opening racks and the bag" {
    const g = Game.init(.{});
    try std.testing.expectEqual(@as(u8, 7), g.rack[0]);
    try std.testing.expectEqual(@as(u8, 7), g.rack[1]);
    try std.testing.expectEqual(@as(u8, 86), g.bag);
    const solo = Game.init(.{ .players = 1 });
    try std.testing.expectEqual(@as(u8, 93), solo.bag);
    try std.testing.expectEqual(@as(u8, 0), solo.rack[1]);
}

test "six scoreless turns end the game and cost each player their rack" {
    var g = Game.init(.{});
    for (0..5) |_| g.pass();
    try std.testing.expect(!g.over());
    g.pass();
    try std.testing.expectEqual(End.scoreless, g.end);
    // Each rack holds the cheapest tiles: two blanks and five 1-pointers,
    // then seven 1-pointers.
    try std.testing.expectEqual(@as(i32, -5), g.score[0]);
    try std.testing.expectEqual(@as(i32, -7), g.score[1]);
}

test "going out: WESPA moves the rack value across, NASPA doubles it" {
    for ([_]rules.EndRule{ .wespa, .naspa }) |rule| {
        var g = Game.init(.{ .end_rule = rule });
        // As if player 1 has just gone out with the bag empty, leaving Q and Z.
        g.pos.pool = @splat(0);
        g.pos.pool[16] = 1;
        g.pos.pool[25] = 1;
        g.pos.pool_total = 2;
        g.bag = 0;
        g.rack = .{ 0, 2 };
        g.finish(.went_out);
        switch (rule) {
            .wespa => try std.testing.expectEqual([2]i32{ 20, -20 }, g.score),
            .naspa => try std.testing.expectEqual([2]i32{ 40, 0 }, g.score),
        }
    }
}
