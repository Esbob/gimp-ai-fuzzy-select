"""The setup's wizard images: a butterfly like a paper kite (pale cells with a peach tint
near the body, black veins and spots, a black border with white spots) with a dashed selection
outline hugging it and a pointer clicking it: an object of many colours that one click selects
(where Fuzzy Select would need many). Large: the panel of the first and last page (164:314),
the butterfly sitting on a leaf; small: the square at the top right of the other pages,
transparent. Several sizes each, for the display scaling. Drawn at 4x and scaled down.

  python wizard_images.py [out folder, default windows/] [leaf | flowers | rock]

Needs NumPy and OpenCV (e.g. the plug-in's virtual environment); takes about a minute."""
import os
import sys
import numpy as np, cv2

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BACKGROUND = sys.argv[2] if len(sys.argv) > 2 else "leaf"
K = 4
BLACK = (40, 28, 38)          # BGR
PALE = (238, 228, 240)
PEACH = (190, 205, 245)
WHITE = (250, 248, 252)


def spline(pts, n=400):
    """A closed Catmull-Rom curve through pts."""
    pts = np.asarray(pts, float)
    out = []
    for i in range(len(pts)):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % len(pts)], pts[(i + 2) % len(pts)]
        for t in np.linspace(0, 1, n // len(pts), endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


FORE = [(0.03, -0.10), (0.22, -0.40), (0.52, -0.62), (0.86, -0.72), (1.0, -0.64), (0.99, -0.42),
        (0.86, -0.14), (0.60, 0.01), (0.30, 0.05), (0.06, 0.03)]
HIND = [(0.05, 0.02), (0.34, 0.04), (0.60, 0.14), (0.71, 0.34), (0.64, 0.57), (0.45, 0.71),
        (0.22, 0.69), (0.09, 0.48), (0.03, 0.22)]
# per wing: outline, the base the veins start at, the part of the outline that is the outer
# margin (as fractions of the outline), veins, dots in every other cell
WINGS = [(FORE, (0.06, -0.04), (0.08, 0.72), 7), (HIND, (0.07, 0.10), (0.22, 0.92), 6)]


def wing_layer(H, W, cx, cy, u, side, detail):
    """One side's wings drawn into an image, and their mask."""
    img = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    to_px = lambda p: np.array([cx + side * p[0] * u, cy + p[1] * u])
    border = 0.075 * u
    vein = max(2, int(0.02 * u * detail))
    for outline, base, (t0, t1), nveins in WINGS:
        nveins = max(4, int(round(nveins * detail)))
        curve = np.array([to_px(p) for p in spline(outline)])
        poly = curve.astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
        inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * border) | 1,) * 2))
        b = to_px(base)
        # pale cells, peach near the body
        yy, xx = np.mgrid[0:H, 0:W]
        t = np.clip(np.hypot(xx - b[0], yy - b[1]) / (0.75 * u), 0, 1)[..., None]
        cells = (np.array(PEACH) * (1 - t) + np.array(PALE) * t).astype(np.uint8)
        layer = np.zeros_like(img)
        layer[:] = BLACK
        layer[inner > 127] = cells[inner > 127]
        # veins from the base to the margin, white spots in the border between them, black
        # dots in every other cell
        ends = [curve[int(len(curve) * (t0 + (t1 - t0) * k / (nveins - 1)))] for k in range(nveins)]
        for e in ends:
            cv2.line(layer, tuple(b.astype(int)), tuple(e.astype(int)), BLACK, vein, cv2.LINE_AA)
        for k in range(nveins - 1):
            mid = curve[int(len(curve) * (t0 + (t1 - t0) * (k + 0.5) / (nveins - 1)))]
            d = (b - mid) / np.linalg.norm(b - mid)
            spot = mid + d * border * 0.55
            r = int(border * 0.33)
            cv2.ellipse(layer, tuple(spot.astype(int)), (r, int(r * 1.3)),
                        float(np.degrees(np.arctan2(d[1], d[0]))), 0, 360, WHITE, -1, cv2.LINE_AA)
            if k % 2 == 1:
                dot = b + (mid - b) * 0.62
                cv2.circle(layer, tuple(dot.astype(int)), max(2, int(0.035 * u)), BLACK, -1, cv2.LINE_AA)
        # a cross vein closing the cell near the base
        cv2.polylines(layer, [np.array([b + (e - b) * 0.38 for e in ends], np.int32)], False, BLACK,
                      vein, cv2.LINE_AA)
        img[m > 127] = layer[m > 127]
        mask |= m
    return img, mask


def butterfly(H, W, cx, cy, u, detail=1.0):
    """The butterfly (BGR image, mask without the antennae, the antennae drawn separately)."""
    left, ml = wing_layer(H, W, cx, cy, u, -1, detail)
    right, mr = wing_layer(H, W, cx, cy, u, 1, detail)
    img = np.where(mr[..., None] > 127, right, left)
    mask = ml | mr
    # body: head, thorax, a striped abdomen
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (int(cx), int(cy - 0.02 * u)), (int(0.055 * u), int(0.16 * u)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(body, (int(cx), int(cy + 0.30 * u)), (int(0.045 * u), int(0.30 * u)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.circle(body, (int(cx), int(cy - 0.20 * u)), int(0.055 * u), 255, -1, cv2.LINE_AA)
    img[body > 127] = BLACK
    for k in range(5):
        y = int(cy + (0.10 + 0.085 * k) * u)
        cv2.line(img, (int(cx - 0.04 * u), y), (int(cx + 0.04 * u), y), WHITE, max(2, int(0.015 * u)), cv2.LINE_AA)
    cv2.circle(img, (int(cx - 0.022 * u), int(cy - 0.215 * u)), max(2, int(0.014 * u)), WHITE, -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx + 0.022 * u), int(cy - 0.215 * u)), max(2, int(0.014 * u)), WHITE, -1, cv2.LINE_AA)
    mask |= body
    return img, mask


def antennae(img, cx, cy, u, color, alpha=None):
    for side in (-1, 1):
        t = np.linspace(0, 1, 40)
        xs = cx + side * (0.02 + 0.30 * t ** 1.4) * u
        ys = cy - (0.24 + 0.50 * t - 0.06 * t * t) * u
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, max(2, int(0.016 * u)), cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), max(2, int(0.03 * u)), color, -1, cv2.LINE_AA)


def ants(img, mask, grow, thick, dash):
    """Marching ants around mask grown by `grow` px."""
    g = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        pts = c[:, 0, :]
        for i in range(len(pts) - 1):
            col = (255, 255, 255, 255) if (i // dash) % 2 else (25, 25, 25, 255)
            cv2.line(img, tuple(map(int, pts[i])), tuple(map(int, pts[i + 1])), col[:img.shape[2]], thick, cv2.LINE_AA)


def pointer(img, x, y, s):
    p = np.array([[0, 0], [0, 17], [4.5, 13], [7.5, 20], [10.5, 18.8], [7.6, 12], [13, 12]], float)
    p = (p * s + [x, y]).astype(np.int32)
    cv2.fillPoly(img, [p], (255, 255, 255, 255)[:img.shape[2]], cv2.LINE_AA)
    cv2.polylines(img, [p], True, (20, 20, 20, 255)[:img.shape[2]], max(2, int(s * 0.9)), cv2.LINE_AA)


def sparkle(img, x, y, s, color):
    pts = [[x + (s if i % 2 == 0 else s * 0.28) * np.sin(np.pi / 4 * i),
            y - (s if i % 2 == 0 else s * 0.28) * np.cos(np.pi / 4 * i)] for i in range(8)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def large(w, h):
    W, H = w * K, h * K
    rng = np.random.default_rng(3)
    # pink rock: a gradient with soft blotches (made also for the other backgrounds: its random
    # numbers come first, so they stay as they were chosen)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (yy / H)[..., None]
    bg = np.array([170, 140, 205], np.float32) * (1 - t) + np.array([125, 95, 160], np.float32) * t
    noise = cv2.resize(rng.normal(0, 1, (h // 12, w // 12)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (H // 4, W // 4)).astype(np.float32), (0, 0), 1.2)
    fine = cv2.resize(fine, (W, H), interpolation=cv2.INTER_LINEAR)
    bg = bg + (14 * noise + 9 * fine)[..., None]
    img = np.clip(bg, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), K * 0.6)  # (a photo's softness behind the butterfly)
    if BACKGROUND == "flowers":
        img = meadow(W, H, rng)
    elif BACKGROUND == "leaf":
        img = leaf_close(W, H, rng)
    cx, cy, u = W * 0.5, H * 0.44, W * 0.40
    bimg, bmask = butterfly(H, W, cx, cy, u)
    if BACKGROUND == "leaf":  # its shadow on the leaf, a little below and to the right
        sh = np.roll(np.roll(bmask, int(0.05 * u), 0), int(0.03 * u), 1)
        sh = cv2.GaussianBlur(sh.astype(np.float32) / 255, (0, 0), 0.05 * u)[..., None]
        img = (img * (1 - 0.45 * sh)).astype(np.uint8)
    img[bmask > 127] = bimg[bmask > 127]
    antennae(img, cx, cy, u, BLACK)
    ants(img, bmask, int(0.05 * u), max(3, int(0.022 * u)), int(0.10 * u) or 1)
    pointer(img, cx + 0.50 * u, cy - 0.28 * u, 0.028 * u)
    return img


FLOWERS = [((205, 170, 245), (60, 200, 250)), ((235, 235, 245), (40, 190, 245)),  # pink, white
           ((215, 140, 190), (70, 215, 250)), ((120, 150, 245), (60, 90, 170)),    # lilac, coral
           ((80, 215, 250), (40, 120, 200)), ((250, 230, 235), (60, 170, 240))]    # yellow, white


def meadow(W, H, rng):
    """Flowers and leaves seen from above, slightly out of focus (the butterfly is sharp)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = (yy / H)[..., None]
    img = (np.array([70, 120, 60], np.float32) * (1 - t) + np.array([45, 85, 40], np.float32) * t)
    img = np.clip(img + 10 * cv2.resize(rng.normal(0, 1, (H // 40, W // 40)).astype(np.float32), (W, H),
                                        interpolation=cv2.INTER_CUBIC)[..., None], 0, 255).astype(np.uint8)
    for _ in range(90):  # leaves
        x, y = rng.uniform(-0.1, 1.1) * W, rng.uniform(-0.05, 1.05) * H
        L = rng.uniform(0.06, 0.16) * W
        g = rng.uniform(0.7, 1.25)
        cv2.ellipse(img, (int(x), int(y)), (int(L), int(L * 0.38)), rng.uniform(0, 180), 0, 360,
                    tuple(int(c * g) for c in (70, 150, 75)), -1, cv2.LINE_AA)
    for _ in range(26):  # flowers: petals around a centre
        x, y = rng.uniform(-0.05, 1.05) * W, rng.uniform(-0.03, 1.03) * H
        r = rng.uniform(0.05, 0.11) * W
        petal, centre = FLOWERS[rng.integers(len(FLOWERS))]
        n = int(rng.integers(5, 11))
        a0 = rng.uniform(0, 2 * np.pi)
        for k in range(n):
            a = a0 + 2 * np.pi * k / n
            px, py = x + 0.55 * r * np.cos(a), y + 0.55 * r * np.sin(a)
            shade = rng.uniform(0.88, 1.0)
            cv2.ellipse(img, (int(px), int(py)), (int(0.5 * r), int(0.24 * r * (8 / n) ** 0.5)),
                        float(np.degrees(a)), 0, 360, tuple(int(c * shade) for c in petal), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), int(0.24 * r), centre, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), int(0.24 * r), tuple(int(c * 0.7) for c in centre), max(2, int(0.04 * r)),
                   cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), K * 1.6)
    # a little darker in the middle, behind the butterfly, so its outline stands out
    v = np.exp(-(((xx - 0.5 * W) / (0.55 * W)) ** 2 + ((yy - 0.44 * H) / (0.25 * H)) ** 2))[..., None]
    return np.clip(img * (1 - 0.25 * v), 0, 255).astype(np.uint8)


def leaf(W, H, rng):
    """A big leaf seen from above, its tip near the top, filling the panel; dark ground beyond."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ground = np.array([30, 45, 35], np.float32) + 12 * cv2.resize(
        rng.normal(0, 1, (H // 30, W // 30)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
    img = np.clip(ground, 0, 255).astype(np.uint8)
    # the midrib: a gentle curve from below the bottom to the tip
    mid = lambda y: W * (0.47 + 0.05 * np.sin(np.pi * y / H))
    tip_y = 0.04 * H
    # outline: half-width along the midrib, widest a little below the middle
    ys = np.linspace(tip_y, 1.15 * H, 300)
    s = (ys - tip_y) / (1.15 * H - tip_y)
    half = W * 0.53 * np.sin(np.pi * np.clip(s, 0, 1) ** 0.75) ** 0.9
    outline = np.concatenate([np.stack([mid(ys) - half, ys], 1), np.stack([mid(ys) + half, ys], 1)[::-1]])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [outline.astype(np.int32)], 255, cv2.LINE_AA)
    # green, lighter along the middle, darker towards the edges where the leaf curves away
    across = np.abs(xx - mid(yy)) / np.maximum(np.interp(yy, ys, half), 1)
    shade = (1.08 - 0.42 * np.clip(across, 0, 1) ** 1.6)[..., None]
    texture = 1 + 0.05 * cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), K * 2)[..., None]
    green = np.clip(np.array([60, 150, 85], np.float32) * shade * texture, 0, 255).astype(np.uint8)
    img[m > 127] = green[m > 127]
    f = img.astype(np.float32)
    vein, groove = (120, 205, 170), (30, 80, 40)
    px1 = lambda r: max(1, int(r * W))

    def paint(draw, color, alpha):
        """Blend what draw(layer) draws (in 255 on a single channel) over f."""
        nonlocal f
        layer = np.zeros((H, W), np.uint8)
        draw(layer)
        a = (layer.astype(np.float32) / 255 * alpha)[..., None]
        f = f * (1 - a) + np.array(color, np.float32) * a

    # the finest veins: a net of small, roughly polygonal cells over the whole surface (the
    # edges of Voronoi cells around random points), in two sizes
    def cells(size, thick):
        def draw(layer):
            sub = cv2.Subdiv2D((0, 0, W, H))
            n = int(W * H / (size * W) ** 2)
            for x, y in zip(rng.uniform(1, W - 2, n), rng.uniform(1, H - 2, n)):
                sub.insert((float(x), float(y)))
            facets, _ = sub.getVoronoiFacetList([])
            cv2.polylines(layer, [np.round(fc).astype(np.int32) for fc in facets], True, 255, thick, cv2.LINE_AA)
        return draw
    paint(cells(0.035, px1(0.0014)), vein, 0.30)
    paint(cells(0.013, px1(0.0008)), vein, 0.16)
    # side veins: from the midrib out and up towards the edge, alternating sides
    sides = {-1: [], 1: []}
    for y0 in np.linspace(0.12 * H, 1.1 * H, 11):
        for side in (-1, 1):
            y0s = y0 + (0.025 * H if side > 0 else 0)
            hw = np.interp(y0s, ys, half)
            # (on past the edge, which cuts it off: near the stem the leaf is narrower where
            # a vein starts than where it ends)
            t = np.linspace(0, 1.6, 80)
            bend = 0.03 * hw * np.sin(np.pi * t / 1.6) * rng.uniform(-1, 1)
            px = mid(y0s) + side * hw * 0.92 * t
            py = y0s - hw * 0.55 * t ** 1.4 + bend
            sides[side].append(np.stack([px, py], 1))
    # small veins across between neighbouring side veins, slightly bowed
    def cross(layer):
        for vs in sides.values():
            for a_, b_ in zip(vs, vs[1:]):
                for i in range(8, 76, 9):
                    p, q = a_[i], b_[min(i + rng.integers(-3, 4), 79)]
                    tt = np.linspace(0, 1, 30)[:, None]
                    normal = np.array([-(q - p)[1], (q - p)[0]]) / (np.linalg.norm(q - p) + 1e-6)
                    wave = (0.010 * W * rng.uniform(0.5, 1) * np.sin(np.pi * tt * rng.uniform(1, 2.5)
                                                                     + rng.uniform(0, 3)))
                    pts = p + (q - p) * tt + normal * wave
                    cv2.polylines(layer, [pts.astype(np.int32)], False, 255, px1(0.0016), cv2.LINE_AA)
    paint(cross, vein, 0.30)
    # each vein lies in a slight groove: a soft darker line along it, then the vein itself
    off = np.array([0.0015 * W, 0.0025 * W])
    paint(lambda l: [cv2.polylines(l, [(v + off).astype(np.int32)], False, 255, px1(0.011), cv2.LINE_AA)
                     for vs in sides.values() for v in vs], groove, 0.22)
    paint(lambda l: [cv2.polylines(l, [v.astype(np.int32)], False, 255, px1(0.0055), cv2.LINE_AA)
                     for vs in sides.values() for v in vs], vein, 1.0)
    my = np.linspace(tip_y, 1.1 * H, 200)
    rib = lambda y: int(0.004 * W + 0.016 * W * (y - tip_y) / H)  # the midrib, thicker towards the stem

    def midrib(layer, shift=(0, 0), extra=1.0):
        for i in range(len(my) - 1):
            cv2.line(layer, (int(mid(my[i]) + shift[0]), int(my[i] + shift[1])),
                     (int(mid(my[i + 1]) + shift[0]), int(my[i + 1] + shift[1])), 255,
                     max(2, int(rib(my[i]) * extra)), cv2.LINE_AA)
    paint(lambda l: midrib(l, off * 2, 1.8), groove, 0.25)
    paint(midrib, vein, 1.0)
    # a soft sheen across the leaf, and a fine grain
    sheen = np.exp(-(((xx * 0.8 + yy * 0.6) - 0.55 * (0.8 * W + 0.6 * H)) / (0.18 * H)) ** 2)[..., None]
    f = f * (1 + 0.10 * sheen) + 6 * sheen
    f *= 1 + 0.025 * cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.5)[..., None]
    img = np.clip(f, 0, 255).astype(np.uint8)
    img[m <= 127] = np.clip(ground, 0, 255).astype(np.uint8)[m <= 127]  # (veins stop at the edge)
    return cv2.GaussianBlur(img, (0, 0), K * 0.4)


def leaf_close(W, H, rng, angle=28):
    """A close-up of the leaf, turned so its midrib runs diagonally, filling the whole panel:
    drawn much larger than the panel, rotated, and cropped near its widest part."""
    Wc, Hc = int(2.6 * W), int(1.7 * H)
    big = leaf(Wc, Hc, rng)
    c = (Wc * 0.47, Hc * 0.58)  # (on the midrib, a little below the middle)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    M[:, 2] += (W / 2 - c[0], H / 2 - c[1])
    return cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def small(n):
    S = n * K
    img = np.zeros((S, S, 4), np.uint8)
    cx, cy, u = S * 0.47, S * 0.50, S * 0.40
    bimg, bmask = butterfly(S, S, cx, cy, u, detail=0.7)
    img[bmask > 127] = np.dstack([bimg, np.full(bmask.shape, 255, np.uint8)])[bmask > 127]
    antennae(img, cx, cy, u, (*BLACK, 255))
    ants(img, bmask, int(0.07 * u), max(3, int(0.045 * u)), int(0.13 * u) or 1)
    pointer(img, cx + 0.48 * u, cy - 0.12 * u, 0.045 * u)
    return img


if __name__ == "__main__":
    for w, h in ((202, 386), (269, 515), (336, 643), (430, 824)):
        cv2.imwrite(f"{OUT}/wizard-{w}.png", cv2.resize(large(w, h), (w, h), interpolation=cv2.INTER_AREA))
    for n in (58, 77, 97, 124):
        cv2.imwrite(f"{OUT}/wizard-small-{n}.png", cv2.resize(small(n), (n, n), interpolation=cv2.INTER_AREA))
