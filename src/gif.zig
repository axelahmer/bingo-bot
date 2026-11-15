//! An animated GIF of a game, for sharing: the board after each turn with
//! the new tiles marked, and a line saying what was played. Everything is
//! drawn here, with the letters from glyphs.zig, so it needs nothing at run
//! time. Colours follow the HTML viewer's light theme.

const std = @import("std");
const rules = @import("rules.zig");
const glyphs = @import("glyphs.zig");

const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const N = rules.N;
const NN = rules.NN;

const cell = 32;
const gap = 2;
const margin = 14;
const board_px = N * cell + (N + 1) * gap;
const title_h = 30;
const info_h = 56;
pub const width = board_px + 2 * margin;
pub const height = margin + title_h + board_px + info_h + margin;
const board_x = margin;
const board_y = margin + title_h;

/// Base colours (palette indices 0..15).
const Color = enum(u8) { bg, grid, sq, dl, tl, dw, tw, tile, hi, accent, ink, muted, tile_ink, blank_ink, star };
const base_rgb = [_]u24{
    0xffffff, 0xd4dbe0, 0xc9d6d0, 0xa5c9e6, 0x336fb3, 0xeaa9a2, 0xc64a3f, 0xf1e2bf,
    0xfff1d2, 0xdf9414, 0x1c2227, 0x5d6973, 0x2a2117, 0x8a6a35, 0x3b2a2a,
};

/// Ink on a background, in 16 steps of coverage (palette 16 + 16 * ramp + step).
const Ramp = enum(u8) { tile_ink, tile_ink_hi, blank_ink, blank_ink_hi, ink, muted, accent, star };
const ramp_pairs = [_][2]Color{
    .{ .tile_ink, .tile }, .{ .tile_ink, .hi },  .{ .blank_ink, .tile }, .{ .blank_ink, .hi },
    .{ .ink, .bg },        .{ .muted, .bg },     .{ .accent, .bg },      .{ .star, .dw },
};

fn blend(fg: u24, bg: u24, a: u32) u24 {
    var out: u24 = 0;
    for (0..3) |k| {
        const s: u5 = @intCast(8 * k);
        const f: u32 = (fg >> s) & 0xff;
        const b: u32 = (bg >> s) & 0xff;
        out |= @as(u24, @intCast((f * a + b * (15 - a) + 7) / 15)) << s;
    }
    return out;
}

fn palette() [256]u24 {
    var p: [256]u24 = @splat(0);
    for (base_rgb, 0..) |c, i| p[i] = c;
    for (ramp_pairs, 0..) |pr, r| {
        for (0..16) |a| p[16 + 16 * r + a] = blend(base_rgb[@intFromEnum(pr[0])], base_rgb[@intFromEnum(pr[1])], @intCast(a));
    }
    return p;
}

/// One frame of the animation, in palette indices.
pub const Canvas = struct {
    px: [width * height]u8,

    pub fn clear(c: *Canvas) void {
        @memset(&c.px, @intFromEnum(Color.bg));
    }

    fn fill(c: *Canvas, x: usize, y: usize, w: usize, h: usize, color: Color) void {
        for (y..y + h) |yy| @memset(c.px[yy * width + x ..][0..w], @intFromEnum(color));
    }

    fn glyph(c: *Canvas, g: *const glyphs.Glyph, pen_x: isize, base_y: isize, ramp: Ramp) void {
        for (0..g.h) |j| {
            for (0..g.w) |i| {
                const ch = g.px[j * @as(usize, g.w) + i];
                const a: u8 = if (ch <= '9') ch - '0' else ch - 'a' + 10;
                if (a == 0) continue;
                const x = pen_x + g.x + @as(isize, @intCast(i));
                const y = base_y + g.y + @as(isize, @intCast(j));
                if (x < 0 or y < 0 or x >= width or y >= height) continue;
                c.px[@as(usize, @intCast(y)) * width + @as(usize, @intCast(x))] = 16 + 16 * @intFromEnum(ramp) + a;
            }
        }
    }

    /// Draw UTF-8 text with its pen starting at x on the baseline; returns
    /// the pen position after it.
    fn text(c: *Canvas, font: *const glyphs.Font, s: []const u8, x: isize, base_y: isize, ramp: Ramp) isize {
        var pen = x;
        var it = (std.unicode.Utf8View.init(s) catch return pen).iterator();
        while (it.nextCodepoint()) |cp| {
            const g = font.get(cp) orelse font.get('?').?;
            c.glyph(g, pen, base_y, ramp);
            pen += g.adv;
        }
        return pen;
    }
};

fn measure(font: *const glyphs.Font, s: []const u8) isize {
    var w: isize = 0;
    var it = (std.unicode.Utf8View.init(s) catch return 0).iterator();
    while (it.nextCodepoint()) |cp| w += (font.get(cp) orelse font.get('?').?).adv;
    return w;
}

/// What a frame says under the board.
pub const Caption = struct {
    /// First line: a word (bold), its score (accent), and a note (muted)
    /// right-aligned.
    word: []const u8 = "",
    score: []const u8 = "",
    note: []const u8 = "",
    /// Second line (muted).
    detail: []const u8 = "",
};

/// Draw the board (cells as in board.Position, `new` marks the tiles just
/// placed), the title above it, and the caption below.
pub fn draw(c: *Canvas, title: []const u8, cells: *const [NN]u8, new: *const [NN]bool, cap: Caption) void {
    c.clear();
    _ = c.text(&glyphs.bold, title, margin, margin + 18, .ink);
    c.fill(board_x, board_y, board_px, board_px, .grid);
    for (0..N) |r| {
        for (0..N) |col| {
            const sq = r * N + col;
            const x = board_x + gap + col * (cell + gap);
            const y = board_y + gap + r * (cell + gap);
            const v = cells[sq];
            if (v == 0) {
                const color: Color = switch (rules.premiums.word[0][sq]) {
                    3 => .tw,
                    2 => .dw,
                    else => switch (rules.premiums.letter[0][sq]) {
                        3 => .tl,
                        2 => .dl,
                        else => .sq,
                    },
                };
                c.fill(x, y, cell, cell, color);
                if (sq == rules.CENTER * N + rules.CENTER) {
                    const g = glyphs.tile.get(0x2605).?;
                    c.glyph(g, @as(isize, @intCast(x + (cell - g.w) / 2)) - g.x, @as(isize, @intCast(y + 22)), .star);
                }
                continue;
            }
            const fresh = new[sq];
            c.fill(x, y, cell, cell, if (fresh) .hi else .tile);
            if (fresh) {
                c.fill(x, y, cell, 2, .accent);
                c.fill(x, y + cell - 2, cell, 2, .accent);
                c.fill(x, y, 2, cell, .accent);
                c.fill(x + cell - 2, y, 2, cell, .accent);
            }
            const blank = rules.isBlank(v);
            const l = rules.letterOf(v);
            const g = glyphs.tile.get('A' + @as(u21, l)).?;
            const ramp: Ramp = if (blank) (if (fresh) .blank_ink_hi else .blank_ink) else (if (fresh) .tile_ink_hi else .tile_ink);
            c.glyph(g, @as(isize, @intCast(x)) + @divTrunc(@as(isize, cell) - g.w, 2) - g.x - 1, @as(isize, @intCast(y + 23)), ramp);
            if (!blank) {
                var vb: [2]u8 = undefined;
                const vs = std.fmt.bufPrint(&vb, "{d}", .{rules.letter_values[l]}) catch unreachable;
                var pen: isize = @as(isize, @intCast(x + cell - 3));
                for (vs) |_| pen -= glyphs.value.get('0').?.adv;
                for (vs) |ch| {
                    const vg = glyphs.value.get(ch).?;
                    c.glyph(vg, pen, @as(isize, @intCast(y + cell - 4)), if (fresh) .tile_ink_hi else .tile_ink);
                    pen += vg.adv;
                }
            }
        }
    }
    const y1: isize = board_y + board_px + 24;
    var pen = c.text(&glyphs.bold, cap.word, margin, y1, .ink);
    if (cap.score.len != 0) _ = c.text(&glyphs.bold, cap.score, pen + 10, y1, .accent);
    pen = width - margin - measure(&glyphs.text, cap.note);
    _ = c.text(&glyphs.text, cap.note, pen, y1, .muted);
    _ = c.text(&glyphs.text, cap.detail, margin, y1 + 22, .muted);
}

/// GIF LZW (8-bit codes, 12-bit table), as the GIF89a spec and the gifenc
/// encoder do it: the code width grows when the next code no longer fits.
const Lzw = struct {
    keys: [1 << 14]u32,
    vals: [1 << 14]u16,
    out: *std.ArrayList(u8),
    gpa: Allocator,
    acc: u64 = 0,
    nbits: u6 = 0,

    const empty: u32 = 0xffff_ffff;
    const mask = (1 << 14) - 1;

    fn reset(z: *Lzw) void {
        @memset(&z.keys, empty);
    }

    fn slot(key: u32) usize {
        return @as(usize, (key *% 2654435761) >> 18) & mask;
    }

    fn find(z: *const Lzw, key: u32) ?u16 {
        var h = slot(key);
        while (z.keys[h] != empty) : (h = (h + 1) & mask) {
            if (z.keys[h] == key) return z.vals[h];
        }
        return null;
    }

    fn put(z: *Lzw, key: u32, val: u16) void {
        var h = slot(key);
        while (z.keys[h] != empty) h = (h + 1) & mask;
        z.keys[h] = key;
        z.vals[h] = val;
    }

    fn emit(z: *Lzw, code: u16, size: u6) !void {
        z.acc |= @as(u64, code) << z.nbits;
        z.nbits += size;
        while (z.nbits >= 8) {
            try z.out.append(z.gpa, @truncate(z.acc));
            z.acc >>= 8;
            z.nbits -= 8;
        }
    }

    fn encode(z: *Lzw, pixels: []const u8) !void {
        const clear: u16 = 256;
        const stop: u16 = 257;
        z.reset();
        var size: u6 = 9;
        var next: u16 = 258;
        try z.emit(clear, size);
        var prefix: u16 = pixels[0];
        for (pixels[1..]) |p| {
            const key = (@as(u32, prefix) << 8) | p;
            if (z.find(key)) |code| {
                prefix = code;
                continue;
            }
            try z.emit(prefix, size);
            if (next < 4096) {
                if (next == (@as(u16, 1) << @intCast(size))) size += 1;
                z.put(key, next);
                next += 1;
            } else {
                try z.emit(clear, size);
                z.reset();
                size = 9;
                next = 258;
            }
            prefix = p;
        }
        try z.emit(prefix, size);
        try z.emit(stop, size);
        if (z.nbits > 0) try z.out.append(z.gpa, @truncate(z.acc));
        z.acc = 0;
        z.nbits = 0;
    }
};

fn writeU16(w: *Writer, v: usize) !void {
    try w.writeByte(@intCast(v & 0xff));
    try w.writeByte(@intCast(v >> 8));
}

/// Write the frames as a looping GIF. Each frame after the first stores
/// only the rectangle that changed; `delays` are in hundredths of a second.
pub fn write(gpa: Allocator, w: *Writer, frames: []const *const Canvas, delays: []const u16) !void {
    try w.writeAll("GIF89a");
    try writeU16(w, width);
    try writeU16(w, height);
    try w.writeAll(&.{ 0xf7, 0, 0 });
    for (palette()) |c| try w.writeAll(&.{ @intCast(c >> 16), @intCast((c >> 8) & 0xff), @intCast(c & 0xff) });
    try w.writeAll(&.{ 0x21, 0xff, 0x0b });
    try w.writeAll("NETSCAPE2.0");
    try w.writeAll(&.{ 3, 1, 0, 0, 0 });

    const z = try gpa.create(Lzw);
    defer gpa.destroy(z);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(gpa);
    var rect: std.ArrayList(u8) = .empty;
    defer rect.deinit(gpa);
    for (frames, delays, 0..) |f, d, k| {
        // The rectangle that differs from the frame before.
        var x0: usize = 0;
        var y0: usize = 0;
        var x1: usize = width - 1;
        var y1: usize = height - 1;
        if (k > 0) {
            const prev = frames[k - 1];
            x0 = width;
            y0 = height;
            x1 = 0;
            y1 = 0;
            for (0..height) |y| {
                for (0..width) |x| {
                    if (f.px[y * width + x] == prev.px[y * width + x]) continue;
                    x0 = @min(x0, x);
                    x1 = @max(x1, x);
                    y0 = @min(y0, y);
                    y1 = @max(y1, y);
                }
            }
            if (x0 > x1) {
                x0 = 0;
                y0 = 0;
                x1 = 0;
                y1 = 0;
            }
        }
        // Graphic control: keep the previous frame underneath, then wait.
        try w.writeAll(&.{ 0x21, 0xf9, 4, 0x04 });
        try writeU16(w, d);
        try w.writeAll(&.{ 0, 0 });
        try w.writeByte(0x2c);
        try writeU16(w, x0);
        try writeU16(w, y0);
        try writeU16(w, x1 - x0 + 1);
        try writeU16(w, y1 - y0 + 1);
        try w.writeByte(0);
        rect.clearRetainingCapacity();
        for (y0..y1 + 1) |y| try rect.appendSlice(gpa, f.px[y * width + x0 .. y * width + x1 + 1]);
        data.clearRetainingCapacity();
        z.* = .{ .keys = undefined, .vals = undefined, .out = &data, .gpa = gpa };
        try z.encode(rect.items);
        try w.writeByte(8);
        var i: usize = 0;
        while (i < data.items.len) : (i += 255) {
            const n = @min(255, data.items.len - i);
            try w.writeByte(@intCast(n));
            try w.writeAll(data.items[i .. i + n]);
        }
        try w.writeByte(0);
    }
    try w.writeByte(0x3b);
}

test "LZW round trip through a reference decoder" {
    const gpa = std.testing.allocator;
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(gpa);
    var pixels: [20000]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    // Long runs and some noise, so the table fills and clears.
    for (&pixels, 0..) |*p, i| p.* = if (i % 97 < 60) @intCast(i / 400 % 7) else prng.random().int(u8);
    const z = try gpa.create(Lzw);
    defer gpa.destroy(z);
    z.* = .{ .keys = undefined, .vals = undefined, .out = &data, .gpa = gpa };
    try z.encode(&pixels);

    // Decode (a plain GIF LZW decoder).
    var prefix: [4096]u16 = undefined;
    var suffix: [4096]u8 = undefined;
    var lens: [4096]u16 = undefined;
    for (0..256) |c| {
        suffix[c] = @intCast(c);
        lens[c] = 1;
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var bitpos: usize = 0;
    var size: u6 = 9;
    var next: u16 = 258;
    var old: ?u16 = null;
    while (true) {
        var code: u16 = 0;
        for (0..size) |b| {
            const bit = (data.items[(bitpos + b) / 8] >> @intCast((bitpos + b) % 8)) & 1;
            code |= @as(u16, bit) << @intCast(b);
        }
        bitpos += size;
        if (code == 256) {
            size = 9;
            next = 258;
            old = null;
            continue;
        }
        if (code == 257) break;
        var buf: [4096]u8 = undefined;
        var first: u8 = undefined;
        if (code < next) {
            var n = lens[code];
            var c = code;
            while (true) {
                n -= 1;
                buf[n] = suffix[c];
                if (c < 256) break;
                c = prefix[c];
            }
            first = buf[0];
            try out.appendSlice(gpa, buf[0..lens[code]]);
        } else {
            // code == next: the previous string plus its own first letter.
            const o = old.?;
            var n = lens[o];
            var c = o;
            while (true) {
                n -= 1;
                buf[n] = suffix[c];
                if (c < 256) break;
                c = prefix[c];
            }
            first = buf[0];
            buf[lens[o]] = first;
            try out.appendSlice(gpa, buf[0 .. lens[o] + 1]);
        }
        if (old) |o| {
            if (next < 4096) {
                prefix[next] = o;
                suffix[next] = first;
                lens[next] = lens[o] + 1;
                next += 1;
                if (next == (@as(u16, 1) << @intCast(size)) and size < 12) size += 1;
            }
        }
        old = code;
    }
    try std.testing.expectEqualSlices(u8, &pixels, out.items);
}
