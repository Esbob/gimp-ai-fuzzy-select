"""The README's before/after pictures, made with the installed plug-in's backend on the
groceries photo from SAM 2's repository (which the installer downloads):
  slicing.png        the window with four bags as slices -> GIMP's Layers panel afterwards
  remove-object.png  the window with the bags selected   -> the bags removed
The left halves are the window's own overlay, the removal is LaMa's real result; only the
Layers panel is drawn. Transparent between the panels, for GitHub's light and dark pages.

  python make_examples.py      (with the plug-in's Python: %LOCALAPPDATA%\\gimp-ai-select\\venv)
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
if os.name == "nt":
    GIMP_PROFILES = Path(os.environ["APPDATA"]) / "GIMP"
    INSTALL = Path(os.environ["LOCALAPPDATA"]) / "gimp-ai-select"
    FONT, BOLD = "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"
else:
    GIMP_PROFILES = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "GIMP"
    INSTALL = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "gimp-ai-select"
    FONT = BOLD = "DejaVuSans.ttf"
PLUGIN = max(GIMP_PROFILES.glob("3.*/plug-ins/aiselect"))
PHOTO = INSTALL / "src" / "sam2" / "notebooks" / "images" / "groceries.jpg"
LAYER = "groceries.jpg"  # the layer's name in GIMP, which the slices are named after
ARROW_W = 70

click = lambda x, y: {"sign": 1, "points": [[x, y, 1]]}
BAGS = [(405, 250), (475, 258), (547, 267), (605, 258)]
ON_TOP = [(410, 185), (497, 160), (535, 165), (570, 130), (598, 155)]  # what sticks out of them
REMOVE_GROW = 8  # px: the default 0 left a faint smudge where the bags met the seats


class Backend:
    """The plug-in's backend, spoken to like the window does."""
    def __init__(self):
        settings = (PLUGIN / "aiselect_settings.json").read_text(encoding="utf-8")
        self.proc = subprocess.Popen([sys.executable, str(PLUGIN / "aiselect_server.py"), settings],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def call(self, **req):
        req["id"] = 1
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        if not reply["ok"]:
            raise SystemExit("backend error: " + reply["error"])
        return reply

    def quit(self):
        self.call(cmd="quit")
        self.proc.wait(timeout=30)


def overlay(photo, preview_png):
    """The window's view: its overlay over the photo."""
    p = cv2.imread(str(preview_png), cv2.IMREAD_UNCHANGED)
    p = cv2.resize(p, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_AREA)
    a = p[..., 3:] / 255.0
    return (p[..., :3] * a + photo * (1 - a)).astype(np.uint8)


def rgba(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)


def arrow(h):
    """A grey arrow on transparent, between the panels."""
    img = Image.new("RGBA", (ARROW_W * 4, h * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cy, x0, x1 = h * 2, 14 * 4, (ARROW_W - 14) * 4
    d.line([(x0, cy), (x1 - 30, cy)], fill=(138, 138, 138, 255), width=26)
    d.polygon([(x1, cy), (x1 - 70, cy - 56), (x1 - 70, cy + 56)], fill=(138, 138, 138, 255))
    return img.resize((ARROW_W, h), Image.LANCZOS)


def side_by_side(left, right):
    h = left.height
    out = Image.new("RGBA", (left.width + ARROW_W + right.width, h), (0, 0, 0, 0))
    out.paste(left, (0, 0))
    a = arrow(h)
    out.paste(a, (left.width, 0), a)
    out.paste(right, (left.width + ARROW_W, 0))
    return out


def thumb(layer_rgba, size):
    """A Layers-panel thumbnail: the layer fitted in size x size, on a checkerboard."""
    h, w = layer_rgba.shape[:2]
    k = size / max(w, h)
    tw, th = max(1, round(w * k)), max(1, round(h * k))
    small = cv2.resize(layer_rgba, (tw, th), interpolation=cv2.INTER_AREA).astype(float)
    yy, xx = np.mgrid[0:th, 0:tw]
    bg = np.where(((xx // 6) + (yy // 6)) % 2, 102, 153)[..., None].astype(float)
    a = small[..., 3:] / 255.0
    out = (small[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    return Image.fromarray(np.dstack([out, np.full((th, tw), 255, np.uint8)]))


def layers_panel(photo, labels, h):
    """GIMP's Layers panel after Slice: the group right above the (now hidden) layer, with a
    layer per slice cropped to it, the last first, and the rest."""
    W, s = 420, 2  # drawn at twice the size, then scaled down
    img = Image.new("RGBA", (W * s, h * s), (60, 60, 60, 255))
    d = ImageDraw.Draw(img)
    f, fb = ImageFont.truetype(FONT, 17 * s), ImageFont.truetype(BOLD, 17 * s)
    text, dim = (225, 225, 225, 255), (150, 150, 150, 255)
    d.text((14 * s, 10 * s), "Layers", font=fb, fill=text)
    d.line([(0, 44 * s), (W * s, 44 * s)], fill=(85, 85, 85, 255), width=s)
    rows = [(0, True, f"{LAYER} (slices)", rgba(photo), True)]
    for k in sorted({int(v) for v in np.unique(labels)} - {0}, reverse=True):
        ys, xs = np.nonzero(labels == k)
        crop = rgba(photo)[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        crop[..., 3] = np.where(labels[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == k, 255, 0)
        rows.append((1, True, f"{LAYER} slice {k}", crop, False))
    rest = rgba(photo)
    rest[..., 3] = np.where(labels == 0, 255, 0)
    rows.append((1, True, f"{LAYER} rest", rest, False))
    rows.append((0, False, LAYER, rgba(photo), False))
    row_h = (h - 52) // len(rows)
    for i, (indent, visible, label, layer, is_group) in enumerate(rows):
        cy = (52 + i * row_h) * s + row_h * s // 2
        if visible:  # the eye
            d.ellipse([20 * s - 9, cy - 5, 20 * s + 9, cy + 5], outline=text, width=2)
            d.ellipse([20 * s - 3, cy - 3, 20 * s + 3, cy + 3], fill=text)
        x = (40 + indent * 22) * s
        if is_group:
            d.polygon([(x, cy - 5 * s), (x + 10 * s, cy - 5 * s), (x + 5 * s, cy + 3 * s)], fill=text)
        x += 16 * s
        t = thumb(layer, (row_h - 12) * s)
        img.paste(t, (x, cy - t.height // 2))
        x += (row_h - 12) * s + 10 * s
        d.text((x, cy), label, font=f, fill=text if visible else dim, anchor="lm")
    return img.resize((W, h), Image.LANCZOS)


def main():
    photo = cv2.imread(str(PHOTO))
    h, w = photo.shape[:2]
    work = Path(tempfile.mkdtemp(prefix="aiselect_examples_"))
    backend = Backend()
    backend.call(cmd="load", image=str(PHOTO), preview=str(work / "p.png"), preview_width=w, preview_height=h)

    # Slicing: a click on each bag makes it a slice
    slices = [dict(click(x, y), slice=k + 1) for k, (x, y) in enumerate(BAGS)]
    backend.call(cmd="render", objects=slices, slices=True, preview_out=str(work / "slices.png"))
    backend.call(cmd="render", objects=slices, slices=True, final=True, out=str(work / "slices.raw"))
    labels = np.fromfile(work / "slices.raw", np.uint8).reshape(h, w)
    before = Image.fromarray(rgba(overlay(photo, work / "slices.png")))
    side_by_side(before, layers_panel(photo, labels, h)).save(HERE / "slicing.png", optimize=True)

    # Removing: the bags and what sticks out of them
    objs = [click(x, y) for x, y in BAGS + ON_TOP]
    backend.call(cmd="render", objects=objs, grow=REMOVE_GROW, preview_out=str(work / "remove.png"))
    r = backend.call(cmd="render", objects=objs, grow=REMOVE_GROW, final=True, out=str(work / "remove.raw"),
                     patch_out=str(work / "patch.raw"))
    backend.quit()
    x1, y1, x2, y2 = r["patch_box"]
    patch = np.fromfile(work / "patch.raw", np.uint8).reshape(y2 - y1, x2 - x1, 4)
    after = photo.copy()
    a = patch[..., 3:] / 255.0
    after[y1:y2, x1:x2] = (patch[..., 2::-1] * a + after[y1:y2, x1:x2] * (1 - a)).astype(np.uint8)
    crop = lambda im: Image.fromarray(rgba(im[40:420, 190:790]))  # the boot
    side_by_side(crop(overlay(photo, work / "remove.png")), crop(after)).save(HERE / "remove-object.png",
                                                                              optimize=True)
    for f in ("slicing.png", "remove-object.png"):
        print(HERE / f, Image.open(HERE / f).size)


if __name__ == "__main__":
    main()
