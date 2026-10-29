"""The setup's butterfly logo as an SVG: the same shapes as wizard_images.py (wing outlines,
veins, spots, body, antennae), with the dashed selection outline and the pointer of its small
image, as vector shapes. The black border of each wing is a wide stroke of its outline,
clipped to the wing.

  python butterfly_svg.py [out file, default assets/butterfly.svg]

Needs NumPy and OpenCV, like wizard_images.py."""
import os
import sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "assets", "butterfly.svg")
sys.argv[1:] = []  # (wizard_images reads its own arguments)
sys.path.insert(0, HERE)
import wizard_images as wi

spline, WINGS = wi.spline, wi.WINGS
rgb = lambda bgr: "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])
BLACK, PALE, PEACH, WHITE = (rgb(c) for c in (wi.BLACK, wi.PALE, wi.PEACH, wi.WHITE))

u = 100.0
f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
path = lambda pts: "M" + " L".join(f"{f(x)},{f(y)}" for x, y in pts) + " Z"
out = []
defs = []
border, vein = 0.075 * u, 0.02 * u

for side in (-1, 1):
    for w, (outline, base, (t0, t1), nveins) in enumerate(WINGS):
        curve = np.array([[side * x * u, y * u] for x, y in spline(outline)])
        b = np.array([side * base[0] * u, base[1] * u])
        wid = f"w{'lr'[side > 0]}{w}"
        d = path(curve)
        defs.append(f'<clipPath id="c{wid}"><path d="{d}"/></clipPath>')
        defs.append(f'<radialGradient id="g{wid}" gradientUnits="userSpaceOnUse" cx="{f(b[0])}" cy="{f(b[1])}" '
                    f'r="{f(0.75 * u)}"><stop offset="0" stop-color="{PEACH}"/><stop offset="1" stop-color="{PALE}"/>'
                    f'</radialGradient>')
        g = [f'<g clip-path="url(#c{wid})">',
             f'<path d="{d}" fill="url(#g{wid})"/>',
             f'<path d="{d}" fill="none" stroke="{BLACK}" stroke-width="{f(2 * border)}"/>']
        at = lambda frac: curve[int(len(curve) * frac)]
        ends = [at(t0 + (t1 - t0) * k / (nveins - 1)) for k in range(nveins)]
        for e in ends:
            g.append(f'<line x1="{f(b[0])}" y1="{f(b[1])}" x2="{f(e[0])}" y2="{f(e[1])}" stroke="{BLACK}" '
                     f'stroke-width="{f(vein)}" stroke-linecap="round"/>')
        for k in range(nveins - 1):
            mid = at(t0 + (t1 - t0) * (k + 0.5) / (nveins - 1))
            dv = (b - mid) / np.linalg.norm(b - mid)
            spot = mid + dv * border * 0.55
            r = border * 0.33
            ang = np.degrees(np.arctan2(dv[1], dv[0]))
            g.append(f'<ellipse cx="0" cy="0" rx="{f(r)}" ry="{f(r * 1.3)}" fill="{WHITE}" '
                     f'transform="translate({f(spot[0])},{f(spot[1])}) rotate({f(ang)})"/>')
            if k % 2 == 1:
                dot = b + (mid - b) * 0.62
                g.append(f'<circle cx="{f(dot[0])}" cy="{f(dot[1])}" r="{f(0.035 * u)}" fill="{BLACK}"/>')
        cross = [b + (e - b) * 0.38 for e in ends]
        g.append(f'<polyline points="{" ".join(f"{f(p[0])},{f(p[1])}" for p in cross)}" fill="none" '
                 f'stroke="{BLACK}" stroke-width="{f(vein)}" stroke-linejoin="round"/>')
        g.append("</g>")
        out += g

# body: thorax, a striped abdomen, head with two eyes
out += [f'<ellipse cx="0" cy="{f(-0.02 * u)}" rx="{f(0.055 * u)}" ry="{f(0.16 * u)}" fill="{BLACK}"/>',
        f'<ellipse cx="0" cy="{f(0.30 * u)}" rx="{f(0.045 * u)}" ry="{f(0.30 * u)}" fill="{BLACK}"/>',
        f'<circle cx="0" cy="{f(-0.20 * u)}" r="{f(0.055 * u)}" fill="{BLACK}"/>']
for k in range(5):
    y = (0.10 + 0.085 * k) * u
    out.append(f'<line x1="{f(-0.04 * u)}" y1="{f(y)}" x2="{f(0.04 * u)}" y2="{f(y)}" stroke="{WHITE}" '
               f'stroke-width="{f(0.015 * u)}" stroke-linecap="round"/>')
for x in (-0.022, 0.022):
    out.append(f'<circle cx="{f(x * u)}" cy="{f(-0.215 * u)}" r="{f(0.014 * u)}" fill="{WHITE}"/>')
# antennae
for side in (-1, 1):
    t = np.linspace(0, 1, 40)
    xs, ys = side * (0.02 + 0.30 * t ** 1.4) * u, -(0.24 + 0.50 * t - 0.06 * t * t) * u
    out.append(f'<polyline points="{" ".join(f"{f(x)},{f(y)}" for x, y in zip(xs, ys))}" fill="none" '
               f'stroke="{BLACK}" stroke-width="{f(0.016 * u)}" stroke-linecap="round"/>')
    out.append(f'<circle cx="{f(xs[-1])}" cy="{f(ys[-1])}" r="{f(0.03 * u)}" fill="{BLACK}"/>')

# the selection outline: black and white dashes around the wings and body (not the antennae),
# 0.07 u out from them as in the setup's small logo; worked out on a fine raster, then smoothed
S = 8  # raster pixels per SVG unit
x0, y0, x1, y1 = -1.16 * u, -0.86 * u, 1.16 * u, 0.86 * u
Wr, Hr = int((x1 - x0) * S), int((y1 - y0) * S)
to_r = lambda pts: ((np.asarray(pts) - [x0, y0]) * S).astype(np.int32)
mask = np.zeros((Hr, Wr), np.uint8)
for side in (-1, 1):
    for outline, *_ in WINGS:
        cv2.fillPoly(mask, [to_r([[side * x * u, y * u] for x, y in spline(outline)])], 255)
for (cx_, cy_), (rx, ry) in (((0, -0.02), (0.055, 0.16)), ((0, 0.30), (0.045, 0.30)), ((0, -0.20), (0.055, 0.055))):
    cv2.ellipse(mask, tuple(to_r([[cx_ * u, cy_ * u]])[0]), (int(rx * u * S), int(ry * u * S)), 0, 0, 360, 255, -1)
grow = int(0.07 * u * S)
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
ring = cv2.approxPolyDP(max(contours, key=len), 0.6, True)[:, 0, :] / S + [x0, y0]
ants_d = path(ring)
thick, dash = 0.045 * u, 0.13 * u
out += [f'<path d="{ants_d}" fill="none" stroke="#ffffff" stroke-width="{f(thick)}" stroke-linejoin="round"/>',
        f'<path d="{ants_d}" fill="none" stroke="#191919" stroke-width="{f(thick)}" stroke-linejoin="round" '
        f'stroke-dasharray="{f(dash)} {f(dash)}"/>']
# the pointer clicking the right forewing
s = 0.045 * u
arrow = np.array([[0, 0], [0, 17], [4.5, 13], [7.5, 20], [10.5, 18.8], [7.6, 12], [13, 12]]) * s + [0.48 * u, -0.12 * u]
out.append(f'<path d="{path(arrow)}" fill="#ffffff" stroke="#141414" stroke-width="{f(0.9 * s)}" '
           f'stroke-linejoin="round"/>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{f(x0)} {f(y0)} {f(x1 - x0)} {f(y1 - y0)}" '
       f'width="{f((x1 - x0) * 2)}" height="{f((y1 - y0) * 2)}">\n'
       f'<title>AI Fuzzy Select butterfly</title>\n<defs>\n' + "\n".join(defs) + "\n</defs>\n"
       + "\n".join(out) + "\n</svg>\n")
open(OUT, "w", encoding="utf-8").write(svg)
print(len(svg), "bytes")
