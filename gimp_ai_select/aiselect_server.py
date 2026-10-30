"""
Backend for the GIMP "AI Fuzzy Select" plugin. Runs in the plugin's own virtual
environment (made by install_ai_select.py). It stays alive between plugin runs and keeps
the analysis of recent images, so SAM2 is loaded once and an image is encoded once; each
click then costs only SAM2's small mask decoder.

Two ways to run it:
  aiselect_server.py SETTINGS_JSON            requests on stdin, replies on stdout
  aiselect_server.py --listen SETTINGS_JSON   a local server that stays alive between plugin
      runs (see listen()); the plugin starts it with AISELECT_TOKEN, AISELECT_STATE,
      AISELECT_LOG and AISELECT_KEEPALIVE in the environment

Protocol: one JSON object per line in each direction. In --listen mode a connection
starts with {"cmd": "hello", "token": ...}.
  {"cmd": "load", "image": png, "preview": png, "preview_width": w, "preview_height": h,
   "guide": raw, "model": name}
      encode the image with SAM2 model "model" (one of the settings' "models"; default:
      the settings' "model"). If that fails (a GPU without enough memory, say) it tries
      SAM 2.1 small on the GPU, then small on the CPU; the reply's "model" and "device" say
      which was used, and "fallback" is set if it wasn't the one asked for ("fallback_now"
      if it failed this time, not only before). If nothing works the error has
      "fatal": true and a message for the user. (With "width" and
      "height", "image" is raw 8-bit RGBA instead, and
      "transparent_black" blacks out the hidden colour of fully transparent pixels, as
      GIMP's PNG export does for RGB images); "preview" is where render writes its
      overlay. "guide": the user's GIMP selection (8-bit, image size); the reply's "guide"
      then has "prompts" (a "box" and "point" for the object in each separate area) and
      its "polygons" outline
  {"cmd": "render", "objects": [...], "threshold": 0-100, "granularity": "auto"|"detail"|
   "part"|"whole", "fill_holes": px, "grow": px, "feather": px, "soft": false, "sharp": false,
   "final": false, "soft_preview": false, "out": raw, "preview_out": png}
      objects: [{"sign": 1 or -1, "points": [[x, y, label], ...]} or
                {"sign": 1 or -1, "box": [x1, y1, x2, y2], "points": [[x, y, 1]] (optional)},
                ...] in image pixels;
      the selection is (union of sign 1 objects) minus (union of sign -1 objects).
      Not final: write a tinted preview overlay PNG with an outline to "preview_out"
      (with "soft_preview": "soft" edges too, computed at reduced size;
      "view": "mask" writes the selection as a white-on-black image instead).
      Final: write the full-size 8-bit selection to "out" (optionally with ViTMatte edges).
      With "patch_out", also remove the object: the background LaMa makes up for it (see
      remove_patch), written as 8-bit RGBA for the box in the reply's "patch_box";
      "remove_context": the area around the object LaMa sees, in % of its size (default 20).
      Not final, with "remove_preview": only that patch, as a PNG at full resolution (a
      final render with the same selection and context then reuses it).
      "slices": true slices the image instead: each object has a "slice" number (1-255,
      default 1) and the objects are applied in order, a sign 1 object giving its pixels
      to its slice (taking them from a bigger object's slice, not a smaller one's: a click
      inside a slice splits it) and a sign -1 object giving them back to the rest (slice
      0). Fill holes and
      grow apply to each object; edges stay hard (no soft edges or feather), so every
      pixel belongs to exactly one slice. "snap": px closes gaps in the rest up to that
      wide between two slices or a slice and the image's edge (see snap_gaps; 0 = off).
      The preview colours each slice; final writes the slice number of each pixel (8-bit,
      image size) to "out". The reply's "slices" lists the slices that have pixels:
      {"slice", "area"}, final also "box" [x1, y1, x2, y2]; "snapped" counts the pixels
      that snap gave to a slice.
      "sharp": sharper outlines, worked out in the background; a preview uses the ones
      that are ready and the reply's "sharp_pending" counts those still being worked out
      ("sharp_refined": those used). A final render waits for them.
  {"cmd": "sharp_status"}
      {"pending": number of sharper outlines still being worked out}
  {"cmd": "hover", "point": [x, y], "threshold": 0-100, "granularity": ..., "slices": false}
      the outline a click there would select: {"polygons": [[[x, y], ...], ...]}; with
      "slices", also the slices it would cut into (see hover_splits): "splits" {"click":
      [numbers], "ctrl": [numbers]} and their "slice_outlines" {number: polygons}
  {"cmd": "bye"}   end this connection (in --listen mode the server waits for the next)
  {"cmd": "quit"}  stop the server
Every reply has "id" (echoed from the request) and "ok", plus "error" on failure.
"""

import collections
import contextlib
import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from unittest import mock

# Keep stdout for the protocol only; anything libraries print goes to stderr.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

with open(__file__, "rb") as _f:
    VERSION = hashlib.sha1(_f.read()).hexdigest()[:12]  # lets the plugin spot an outdated server
IMAGE_CACHE_SIZE = 4  # images whose SAM2 analysis is kept (about 16 MB of GPU memory each)
# Full-size results of objects no longer in the selection (undone or replaced clicks) are
# kept up to this many bytes, oldest dropped first: each is 4 bytes per pixel (192 MB on an
# 8000x6000 photo), and keeping every click of a session reached 10 GB in a test.
SPARE_CACHE_BYTES = 1 << 30
# LaMa fills each area on a crop scaled down to at most this many pixels: on real photos 512
# kept walls, kerbs and lines going where 1024 and 2048 smeared a large hole (LaMa was
# trained on small crops, so the hole should be small in pixels too).
LAMA_SIDE = 512
REMOVE_CONTEXT = 20  # around an object to remove, this % of its size is LaMa's context (default)
REMOVE_MARGIN = 0.02  # and this much is filled in with it (its edge pixels)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TINT = (64, 160, 255)  # overlay colour (RGB)
# Slices in the preview, in turn (RGB): colours that tell apart well next to each other.
SLICE_COLOURS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                 (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
                 (0, 128, 128), (170, 110, 40)]


class LoadFailed(Exception):
    """An image couldn't be analysed with any model on any device; the message is for the user."""


def fill_holes(mask, max_area):
    """Fill enclosed holes (not touching the image border) of at most max_area pixels."""
    h, w = mask.shape
    n, cc, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    x, y, cw, ch, area = (stats[:, i] for i in range(5))
    enclosed = (x > 0) & (y > 0) & (x + cw < w) & (y + ch < h)
    small = np.flatnonzero(enclosed & (area <= max_area))
    small = small[small > 0]
    return mask | np.isin(cc, small) if len(small) else mask


def disk_dilate(mask, r):
    """`mask` (bool) grown by a disc of radius r: a distance transform, which unlike cv2.dilate
    with an elliptical kernel costs the same for any r (r reached hundreds of pixels)."""
    return cv2.distanceTransform((~mask).view(np.uint8), cv2.DIST_L2, 5) <= r


def grow_mask(mask, px):
    """Grow (px > 0) or shrink (px < 0) a boolean mask by a round brush."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(px) + 1,) * 2)
    m8 = mask.astype(np.uint8)
    return (cv2.dilate(m8, k) if px > 0 else cv2.erode(m8, k)).astype(bool)


def snap_gaps(labels, image, gap):
    """Slicing: gaps in the rest at most `gap` px wide between two slices, or between a
    slice and the image's edge, go to the slices around them. SAM2's outlines of two
    touching objects rarely meet exactly: on a 4K drawing a tower and the sky behind it
    left a rim of 1500 (5 px) to 8000 (9 px) pixels, the light halo drawn around the tower,
    in the rest. A rest pixel snaps when its distances to the two nearest slices (or the
    nearest slice and the image's edge) add up to at most gap + 1, so rest next to only one
    slice stays rest. A gap is split along the strongest edge in `image` (a watershed from
    the slices around it), so a halo goes with the sky it resembles and an ink line with
    the object. Changes `labels` (image size, 0 = the rest) in place; returns the number of
    pixels that snapped."""
    rest = labels == 0
    if gap <= 0 or rest.all() or not rest.any():
        return 0
    h, w = labels.shape
    # Only near both the rest and a slice can anything snap: work on that part of the image.
    near = rest & disk_dilate(~rest, gap + 1)
    if not near.any():
        return 0
    ys, xs = np.nonzero(near)
    m = gap + 2
    y1, y2, x1, x2 = max(ys.min() - m, 0), min(ys.max() + m + 1, h), max(xs.min() - m, 0), min(xs.max() + m + 1, w)
    crop = labels[y1:y2, x1:x2]
    far = np.float32(gap + 2)  # (no nearer than this: can't snap)
    d1 = np.full(crop.shape, far, np.float32)  # distance to the nearest slice
    d2 = d1.copy()  # and to the nearest other slice
    l1 = np.zeros(crop.shape, np.uint8)  # the nearest slice
    for k in np.flatnonzero(np.bincount(crop.ravel(), minlength=256)[1:]) + 1:
        is_k = crop == k
        rows, cols = np.flatnonzero(is_k.any(1)), np.flatnonzero(is_k.any(0))
        by1, by2 = max(rows[0] - m, 0), min(rows[-1] + m + 1, crop.shape[0])
        bx1, bx2 = max(cols[0] - m, 0), min(cols[-1] + m + 1, crop.shape[1])
        part = crop[by1:by2, bx1:bx2]
        dk = cv2.distanceTransform((part != k).view(np.uint8), cv2.DIST_L2, 5)
        a1, a2, al = d1[by1:by2, bx1:bx2], d2[by1:by2, bx1:bx2], l1[by1:by2, bx1:bx2]
        nearer = dk < a1
        a2[:] = np.where(nearer, a1, np.minimum(a2, dk))
        a1[nearer] = dk[nearer]
        al[nearer] = k
    yy = np.arange(y1, y2, dtype=np.float32)[:, None]
    xx = np.arange(x1, x2, dtype=np.float32)[None, :]
    edge = np.minimum(np.minimum(yy + 1, h - yy), np.minimum(xx + 1, w - xx))  # to the image's edge
    snap = (crop == 0) & (d1 + np.minimum(d2, edge) <= gap + 1)
    if not snap.any():
        return 0
    # Split each gap along the image's strongest edge: a watershed from the slices, with the
    # rest that doesn't snap as a seed of its own so the slices can't flood into it.
    markers = crop.astype(np.int32)
    markers[crop == 0] = 256
    markers[snap] = 0
    cv2.watershed(np.ascontiguousarray(image[y1:y2, x1:x2]), markers)
    won = markers[snap]
    # watershed lines (-1) and anything the rest's seed reached go to the nearest slice
    crop[snap] = np.where((won >= 1) & (won <= 255), won, l1[snap]).astype(np.uint8)
    return int(snap.sum())


class Server:
    def __init__(self, settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu = self.device  # what loading an image tries first (see load_plan)
        self.failed = set()  # (model, device) that failed to load an image; not tried again
        limit = os.environ.get("AISELECT_GPU_LIMIT_MB")  # tests: act like a smaller GPU
        if limit and self.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(limit) * 2 ** 20 / total))
        self.predictor = None
        self.model_name = None  # the SAM2 model loaded (see model_spec)
        self.vitmatte = None
        self.lama = None  # for removing objects, loaded when first needed
        self.last_patch = None  # (image, selection hash, context, box, RGBA): the last removal
        self.slice_state = None  # (image, labels, coarse owner, n, {slice: outline}): the last slicing
        self.lama_device = None
        self.image = None
        # (granularity, points) or ("box", ...) -> (full-res logits, low-res logits), oldest
        # first; see prune_cache
        self.cache = collections.OrderedDict()
        self.cache_used = set()  # keys used by the current render
        # SAM2's analysis of recent images, so reopening the window on the same image is
        # instant: image hash -> (image, predictor state)
        self.analysed = collections.OrderedDict()
        # Sharper outlines take seconds per object, so a background thread works them out
        # (with its own predictor on the same model, and its own CUDA stream) while clicks
        # and hover go on as usual. Each load starts a new generation; results for an
        # earlier image are dropped.
        self.crop_predictor = None
        self.sharp_lock = threading.Condition()
        self.sharp_jobs = []  # [(key, generation, obj, coarse logits, image)], oldest first
        self.sharp_busy = None  # (key, generation) being worked out
        self.sharp_done = {}  # key -> (x1, y1, x2, y2, crop logits), or None: keep the normal one
        self.sharp_merged = {}  # key -> (cut, full-size logits, state) for the latest Threshold
        self.sharp_thread = None
        self.generation = 0
        # One image analysis at a time: two at once (a load while a sharper outline is
        # worked out) left a 4 GB GPU with 0 MB free and took 35-42 s instead of ~15 s.
        # Clicks and hover don't need it.
        self.encoder_lock = threading.Lock()

    # --- models --------------------------------------------------------------------

    def model_spec(self, name=None):
        """(name, config, checkpoint) of the SAM2 model to use: `name` if the settings have
        it, else the settings' default. Settings from before there was a choice name only a
        SAM 2 large checkpoint."""
        models = self.settings.get("models")
        if not models:
            return "sam2-large", "configs/sam2/sam2_hiera_l.yaml", self.settings["sam2Checkpoint"]
        if name not in models:
            name = self.settings.get("model") if self.settings.get("model") in models else next(iter(models))
        return name, models[name]["config"], models[name]["checkpoint"]

    def load_plan(self, wanted):
        """(model, device) pairs to try for an image, best first: the wanted model on the GPU,
        SAM 2.1 small on the GPU, small on the CPU. Pairs that failed before are skipped
        (unless nothing else is left), so a second image doesn't fail through them again."""
        small = "small" if "small" in (self.settings.get("models") or {}) else wanted
        plan = list(dict.fromkeys([(wanted, self.gpu), (small, self.gpu), (small, "cpu")]))
        return [p for p in plan if p not in self.failed] or plan[-1:]

    def analyse(self, name, device, key):
        """Have model `name` on `device` analyse self.image (or reuse its analysis). Returns
        whether the analysis was reused."""
        if self.predictor is None or (name, device) != (self.model_name, self.device):
            if device != self.device:
                self.vitmatte = self.lama = self.lama_device = None  # loaded again on the new device when needed
            self.device = device
            self.load_sam2(name)
        key += name + device
        cached = key in self.analysed
        if cached:
            self.analysed.move_to_end(key)
            pred = self.predictor
            pred._features, pred._orig_hw, pred._is_batch = self.analysed[key]
            pred._is_image_set = True
        else:
            with torch.no_grad(), self.encoder_lock:
                self.predictor.set_image(self.image)
                if self.device == "cuda":
                    # The GPU works asynchronously: "ready" must mean ready (and the lock held
                    # until it is). Only this stream: other work needn't finish.
                    torch.cuda.current_stream().synchronize()
            pred = self.predictor
            self.analysed[key] = (pred._features, pred._orig_hw, pred._is_batch)
            while len(self.analysed) > IMAGE_CACHE_SIZE:
                self.analysed.popitem(last=False)
        return cached

    def load_sam2(self, name=None):
        """Load a SAM2 model in place of the current one (only one is kept in memory)."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        name, config, checkpoint = self.model_spec(name)
        self.predictor = self.crop_predictor = None  # free the old one first
        if self.device == "cuda":
            torch.cuda.empty_cache()
        # Building a model for the CPU, SAM2 still uses the GPU if there is one (it warms up its
        # positional encodings there, and moves a RoPE table there): hide it, or the CPU can't
        # take over from a GPU that is out of memory.
        hide_gpu = (mock.patch.object(torch.cuda, "is_available", lambda: False)
                    if self.device == "cpu" else contextlib.nullcontext())
        with hide_gpu:
            model = build_sam2(config, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.crop_predictor = SAM2ImagePredictor(model)
        self.model_name = name

    def load_vitmatte(self):
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate

        # vitmatteDir holds Matte Anything's ViTMatte config and `modeling` package;
        # "appDir" is the older name for vitmatteDir, still accepted from earlier installs.
        code = self.settings.get("vitmatteDir") or self.settings.get("appDir")
        if not code:
            raise RuntimeError("Soft edges aren't installed (re-run install_ai_select.py "
                               "without --no-soft-edges).")
        checkpoint = self.settings.get("vitmatteCheckpoint") or os.path.join(
            code, "pretrained", "ViTMatte_B_DIS.pth")
        sys.path.insert(0, code)  # the config imports the `modeling` package
        cfg = LazyConfig.load(os.path.join(code, "configs", "matte_anything.py"))
        model = instantiate(cfg.model)
        model.to(self.device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
        self.vitmatte = model

    def free_lama(self):
        """Give LaMa's GPU memory back (about 200 MB): on a 4 GB GPU SAM2 or ViTMatte may need
        it. LaMa loads again in a second or two when an object is next removed."""
        if self.lama is not None and self.lama_device == "cuda":
            self.lama = self.lama_device = None
            torch.cuda.empty_cache()

    def lama_fill(self, image, mask):
        """LaMa's background for the pixels in `mask` (HxW uint8, 1 = fill) of `image` (HxWx3
        uint8, sides a multiple of 8). On a GPU without enough memory left (for loading LaMa
        or for running it), on the CPU, from then on."""
        path = self.settings.get("lamaCheckpoint")
        if not path:
            raise RuntimeError("Removing objects isn't installed: run the installer again.")
        for device in dict.fromkeys([self.lama_device or self.device, "cpu"]):
            try:
                if self.lama is None or device != self.lama_device:
                    self.lama = None  # (frees the GPU's copy first)
                    self.lama = torch.jit.load(path, map_location=device).eval()
                    self.lama_device = device
                with torch.no_grad():
                    img = torch.from_numpy(image).permute(2, 0, 1)[None].float().to(device) / 255
                    out = self.lama(img, torch.from_numpy(mask)[None, None].float().to(device))
                    return out[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
            except RuntimeError as e:  # (inside TorchScript, running out of memory is a RuntimeError)
                if device == "cpu" or "out of memory" not in str(e):
                    raise
                self.lama = None
                torch.cuda.empty_cache()

    def remove_patch(self, mask, path, context=REMOVE_CONTEXT, png=False):
        """For removing the object in `mask` (bool, image size): a patch of background made up
        by LaMa. Each separate part is filled as if the photo had been cropped around it
        first, with `context` % of the part's size on each side, scaled down to LAMA_SIDE px
        and the result scaled back up. The patch covers the part and a margin of
        REMOVE_MARGIN of its size, where its edge colours still show. Both follow the object,
        not the photo: sized by the photo, a street sign in a 4000 px photo was filled at a
        quarter of its resolution, a soft patch. Writes the box around the patch as 8-bit
        RGBA (A = 255 in the patch; as PNG with png=True) and returns the box."""
        key = hashlib.sha1(mask.tobytes()).hexdigest()
        last = self.last_patch
        if last and last[0] is self.image and last[1:3] == (key, context):  # e.g. shown in the window
            box, out = last[3], last[4]
        else:
            box, out = self.make_patch(mask, context)
            self.last_patch = (self.image, key, context, box, out)
        if png:
            cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            with open(path, "wb") as f:
                f.write(out.tobytes())
        return box

    def make_patch(self, mask, context):
        h, w = mask.shape
        image = self.image.copy()  # filled area by area: areas filled earlier are context
        patch = np.zeros((h, w), bool)
        # Each piece of the selection reaches twice its margin (REMOVE_MARGIN of its own size);
        # pieces that reach each other are one part, with the biggest one's margin: a car's
        # side mirror, a few pixels off the car, is filled with the car (on its own it got a
        # tiny margin and left its bright edge behind). Margins by the size of the whole
        # selection instead chained objects at opposite sides of the photo (through stray
        # bits in between) into one part, with one crop over much of the photo.
        n, pieces, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
        reach = np.zeros((h, w), bool)
        margins = [0] * n
        for j in range(1, n):
            x, y, pw, ph = (int(v) for v in stats[j, :4])
            g = margins[j] = max(4, round(max(pw, ph) * REMOVE_MARGIN))
            r = 2 * g
            x1, y1, x2, y2 = max(x - r, 0), max(y - r, 0), min(x + pw + r, w), min(y + ph + r, h)
            reach[y1:y2, x1:x2] |= disk_dilate(pieces[y1:y2, x1:x2] == j, r)
        n, parts, stats_parts, _ = cv2.connectedComponentsWithStats(reach.view(np.uint8), connectivity=8)
        grows = [4] * n
        for j in range(1, len(margins)):
            x, y, pw, ph = (int(v) for v in stats[j, :4])
            part = int(parts[y:y + ph, x:x + pw][pieces[y:y + ph, x:x + pw] == j][0])
            grows[part] = max(grows[part], margins[j])
        # Each part's crop and hole (the part and its margin)
        crops = []
        for i in range(1, n):
            px, py, pw, ph = (int(v) for v in stats_parts[i, :4])
            grow = grows[i]
            side = max(pw, ph) - 4 * grow  # (the part's size, without its reach)
            m = max(16, round(side * context / 100)) + grow
            box = (max(px - m, 0), max(py - m, 0), min(px + pw + m, w), min(py + ph + m, h))
            x1, y1, x2, y2 = box
            crops.append((box, disk_dilate((parts[y1:y2, x1:x2] == i) & mask[y1:y2, x1:x2], grow)))
        # Parts whose crops fit together without making the bigger one larger than LAMA_SIDE
        # (or than it was) are filled in one go: 200 specks of foliage took 200 LaMa runs
        areas = []  # [box, its long side to stay within, [(box, hole), ...]]
        for box, hole in sorted(crops, key=lambda c: -max(c[0][2] - c[0][0], c[0][3] - c[0][1])):
            for area in areas:
                u = (min(area[0][0], box[0]), min(area[0][1], box[1]), max(area[0][2], box[2]),
                     max(area[0][3], box[3]))
                if max(u[2] - u[0], u[3] - u[1]) <= area[1]:
                    area[0] = u
                    area[2].append((box, hole))
                    break
            else:
                areas.append([box, max(LAMA_SIDE, box[2] - box[0], box[3] - box[1]), [(box, hole)]])
        for (x1, y1, x2, y2), _, members in areas:
            hole = np.zeros((y2 - y1, x2 - x1), bool)
            for (bx1, by1, bx2, by2), part_hole in members:
                hole[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] |= part_hole
            crop = image[y1:y2, x1:x2]
            ch, cw = hole.shape
            scale = min(1.0, LAMA_SIDE / max(ch, cw))
            size = (max(8, round(cw * scale)), max(8, round(ch * scale)))
            small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            small_hole = cv2.dilate(cv2.resize(hole.view(np.uint8), size, interpolation=cv2.INTER_NEAREST),
                                    np.ones((3, 3), np.uint8))
            pad = ((0, -size[1] % 8), (0, -size[0] % 8))
            result = self.lama_fill(np.pad(small, pad + ((0, 0),), mode="reflect"),
                                    np.pad(small_hole, pad, mode="reflect"))[:size[1], :size[0]]
            if scale < 1:
                result = cv2.resize(result, (cw, ch), interpolation=cv2.INTER_CUBIC)
            crop[hole] = result[hole]
            patch[y1:y2, x1:x2] |= hole
        x, y, bw, bh = cv2.boundingRect(patch.view(np.uint8))
        out = np.zeros((bh, bw, 4), np.uint8)
        out[..., :3] = image[y:y + bh, x:x + bw]
        out[..., 3] = patch[y:y + bh, x:x + bw] * 255
        return [x, y, x + bw, y + bh], out

    # --- commands ------------------------------------------------------------------

    def cmd_load(self, req):
        t0 = time.time()
        self.last_patch = self.slice_state = None  # (they hold on to the previous image)
        self.free_lama()  # analysing the image needs the GPU memory more
        wanted = self.model_spec(req.get("model"))[0]
        if req.get("width"):  # raw 8-bit RGBA from the plugin
            rgba = np.fromfile(req["image"], np.uint8).reshape(req["height"], req["width"], 4)
            self.image = rgba[..., :3].copy()
            if req.get("transparent_black"):
                self.image[rgba[..., 3] == 0] = 0
        else:
            image = cv2.imread(req["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read {req['image']}")
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.preview_path = req["preview"]
        self.preview_size = (req["preview_width"], req["preview_height"])
        self.cache.clear()
        with self.sharp_lock:
            self.generation += 1
            self.sharp_jobs.clear()
            self.sharp_done.clear()
            self.sharp_merged.clear()
        key = hashlib.sha1(self.image.tobytes()).hexdigest() + str(self.image.shape)
        failures = []
        for name, device in self.load_plan(wanted):
            try:
                cached = self.analyse(name, device, key)
                break
            except Exception as e:
                traceback.print_exc()
                text = f"{type(e).__name__}: {e}"  # (SAM2 wraps an out-of-memory error in its own)
                memory = any(m in text.lower() for m in ("out of memory", "not enough memory"))
                failures.append((name, device, text.splitlines()[0][:200], memory))
                self.failed.add((name, device))
                self.predictor = self.crop_predictor = None
                self.model_name = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            print("could not analyse the image:", failures, flush=True)
            memory = failures and all(f[3] for f in failures)
            raise LoadFailed(
                "AI Fuzzy Select couldn't analyse this image on this computer, neither on the "
                "graphics card nor on the processor" + (": there isn't enough memory." if memory else ".")
                + "\n\nClosing other programs and trying again may help.")
        h, w = self.image.shape[:2]
        reply = {"width": w, "height": h, "seconds": round(time.time() - t0, 2),
                 "device": self.device, "cached": cached, "model": self.model_name}
        if (self.model_name, self.device) != (wanted, self.gpu):
            reply.update(fallback=True, wanted=wanted, fallback_now=bool(failures), gpu=self.gpu == "cuda")
        if req.get("guide"):
            reply["guide"] = self.guide_prompt(req["guide"])
        return reply

    def object_logits(self, points, granularity):
        """Full-res mask logits for one object's clicks. The first click picks one of
        SAM2's three candidates (detail / part / whole); later clicks refine it, feeding
        the previous low-res logits back in as SAM2's own interactive demo does."""
        key = (granularity, tuple(map(tuple, points)))
        self.cache_used.add(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        coords = np.array([p[:2] for p in points], dtype=np.float32)
        labels = np.array([p[2] for p in points], dtype=np.int32)
        with torch.no_grad():
            if len(points) == 1:
                full, low = self.single_click(points[0][:2], granularity)
            else:
                self.object_logits(points[:-1], granularity)
                prev_low = self.cache[(granularity, tuple(map(tuple, points[:-1])))][1]
                masks, _, low = self.predictor.predict(
                    point_coords=coords, point_labels=labels, mask_input=prev_low[None],
                    multimask_output=False, return_logits=True)
                full, low = masks[0], low[0]
        self.cache[key] = (full, low)
        return full

    def single_click(self, point, granularity):
        """Full-res and low-res logits for one click: SAM2's three candidates, picked by
        Granularity. Only candidates that contain the clicked pixel count: SAM2's smallest
        candidate is sometimes a stray fragment elsewhere (31 px, score 0.09, next to a
        clicked disc), and picking it made Ctrl+click and "Detail" do nothing."""
        x, y = point
        with torch.no_grad():
            masks, scores, low = self.predictor.predict(
                point_coords=np.array([[x, y]], dtype=np.float32), point_labels=np.array([1]),
                multimask_output=True, return_logits=True)
        containing = [i for i in range(len(masks)) if masks[i][y, x] > 0] or list(range(len(masks)))
        if granularity == "auto":
            pick = max(containing, key=lambda i: scores[i])
        else:
            by_size = sorted(containing, key=lambda i: int((masks[i] > 0).sum()))
            pick = {"detail": by_size[0], "part": by_size[len(by_size) // 2],
                    "whole": by_size[-1]}[granularity]
        # Copies: a view would keep all three full-size candidates alive in the cache
        return masks[pick].copy(), low[pick].copy()

    def cmd_hover(self, req):
        """The outline a click at req["point"] would select, as polygons in image pixels.
        Not cached (the mouse visits thousands of pixels; each full-res result is several
        MB), and drawn by the window directly instead of via an image file."""
        t0 = time.time()
        cut = (50 - req.get("threshold", 50)) / 25.0
        logits, _ = self.single_click(req["point"], req.get("granularity", "auto"))
        mask = logits > cut
        reply = {"point": req["point"], "polygons": self.outline(mask), "area": int(mask.sum())}
        if req.get("slices"):
            reply["splits"], reply["slice_outlines"] = self.hover_splits(mask)
        return dict(reply, seconds=round(time.time() - t0, 3))

    def box_logits(self, box, point=None):
        """Full-res mask logits for the object inside a box [x1, y1, x2, y2], optionally
        with a click [x, y] inside it. SAM2 recommends a single output for box prompts: the
        box already says how big the object is, so Granularity doesn't apply."""
        key = ("box", tuple(box), tuple(point) if point else None)
        self.cache_used.add(key)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            kwargs = {}
            if point:
                kwargs = {"point_coords": np.array([point], dtype=np.float32),
                          "point_labels": np.array([1])}
            with torch.no_grad():
                masks, _, low = self.predictor.predict(
                    box=np.array(box, dtype=np.float32), multimask_output=False,
                    return_logits=True, **kwargs)
            self.cache[key] = (masks[0], low[0])
        return self.cache[key][0]

    @staticmethod
    def sharp_key(obj, granularity):
        # Granularity doesn't apply to boxes (see box_logits)
        return ("sharp", None if obj.get("box") else granularity, json.dumps(obj.get("box")),
                json.dumps(obj.get("points")))

    def sharp_logits(self, obj, granularity, coarse, cut, wait):
        """Sharper outline for one object: SAM2 again, on a crop around the object, at its
        own 1024 px. SAM2's masks come out at a quarter of its 1024 px input, so on a
        4000 px photo each mask cell covers about 16x16 pixels. On 4000x2600 test images
        with known outlines (20 objects, 150-2000 px, by click and by box) this raised the
        share of the outline within 2 px from 0.49-0.65 to 0.94-0.97 for objects up to
        600 px, 0.76 to 0.92 at 1200 px and 0.63 to 0.74 at 2000 px, never worse, for about
        2 s per object on a GTX 1050 Ti (up to 15 s once the laptop GPU throttles from heat);
        a guided filter on the photo did less. Cached per object.

        Worked out on the background thread (sharp_worker); with `wait` this waits for it,
        otherwise it returns the normal logits until it is ready. Returns (logits, state),
        state "done", "pending" or "skipped" (the normal outline is kept)."""
        key = self.sharp_key(obj, granularity)
        with self.sharp_lock:
            job = (key, self.generation)
            if (key not in self.sharp_done and self.sharp_busy != job
                    and all(j[:2] != job for j in self.sharp_jobs)):
                self.sharp_jobs.append((key, self.generation, obj, coarse, self.image))
                self.sharp_lock.notify_all()
            self.ensure_sharp_worker()
            while wait and key not in self.sharp_done:
                self.sharp_lock.wait(timeout=1)
                self.ensure_sharp_worker()  # never wait on a worker that has stopped
            if key not in self.sharp_done:
                return coarse, "pending"
            refined = self.sharp_done[key]
            merged = self.sharp_merged.get(key)
        if refined is None:
            return coarse, "skipped"
        if merged is None or merged[0] != cut:
            merged = (cut, *self.merge_sharp(coarse, refined, cut, obj))
            with self.sharp_lock:
                self.sharp_merged[key] = merged
        return merged[1], merged[2]

    @staticmethod
    def merge_sharp(coarse, refined, cut, obj):
        """The sharper crop logits pasted into the normal ones, but only near the outline
        at this Threshold: it may move by at most one cell of SAM2's own mask grid (where
        the normal result is coarse), so the object stays the one the user picked."""
        h, w = coarse.shape
        x1, y1, x2, y2, crop = refined
        normal = coarse[y1:y2, x1:x2]
        edge = cv2.morphologyEx((normal > cut).astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        near = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5) <= max(h, w) / 256
        part = np.where(near, crop, normal)
        point = obj["points"][0][:2] if obj.get("points") else None
        if point and coarse[point[1], point[0]] > cut and part[point[1] - y1, point[0] - x1] <= cut:
            return coarse, "skipped"  # never lose the clicked spot
        out = coarse.copy()
        out[y1:y2, x1:x2] = part
        return out, "done"

    def ensure_sharp_worker(self):
        """Start the background worker if it isn't running (call with sharp_lock held)."""
        if self.sharp_thread is None or not self.sharp_thread.is_alive():
            self.sharp_thread = threading.Thread(target=self.sharp_worker, daemon=True)
            self.sharp_thread.start()

    def sharp_worker(self):
        try:
            stream = torch.cuda.Stream() if self.device == "cuda" else None
        except Exception:
            traceback.print_exc()
            stream = None
        while True:
            with self.sharp_lock:
                while not self.sharp_jobs:
                    self.sharp_lock.wait()
                key, generation, obj, coarse, image = self.sharp_jobs.pop(0)
                self.sharp_busy = (key, generation)
            refined = None  # on any failure: keep the normal outline
            try:
                with torch.cuda.stream(stream) if stream and self.device == "cuda" else contextlib.nullcontext():
                    refined = self.refine_in_crop(obj, coarse, image)
            except BaseException:
                traceback.print_exc()
            finally:
                if stream is not None and self.device == "cuda":
                    stream.synchronize()
                    torch.cuda.empty_cache()  # give this stream's memory pool back
                with self.sharp_lock:
                    self.sharp_busy = None
                    if generation == self.generation:
                        self.sharp_done[key] = refined
                    self.sharp_lock.notify_all()

    def cmd_sharp_status(self, req):
        with self.sharp_lock:
            pending = sum(j[1] == self.generation for j in self.sharp_jobs)
            pending += bool(self.sharp_busy and self.sharp_busy[1] == self.generation)
        return {"pending": pending}

    def refine_in_crop(self, obj, coarse, image):
        h, w = coarse.shape
        mask = coarse > 0
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        mx, my = (bx2 - bx1) // 10 + 16, (by2 - by1) // 10 + 16
        x1, y1, x2, y2 = max(bx1 - mx, 0), max(by1 - my, 0), min(bx2 + mx, w), min(by2 + my, h)
        if (x2 - x1) * (y2 - y1) > 0.7 * w * h:
            return None  # the object fills most of the image: nothing to zoom into
        pred = self.crop_predictor
        point = obj["points"][0][:2] if obj.get("points") else None
        with torch.no_grad():
            with self.encoder_lock:
                pred.set_image(np.ascontiguousarray(image[y1:y2, x1:x2]))
                if self.device == "cuda":
                    torch.cuda.current_stream().synchronize()  # done before letting go
            # The object's extent as a box, the click, and the normal mask itself as SAM2's
            # mask prompt keep it on the same object: seen this much larger, SAM2 otherwise
            # often picked another one (on real photos 41 of 207 refinements overlapped the
            # normal result by less than half; with the mask prompt, 23).
            kwargs = {"box": np.array([bx1 - x1, by1 - y1, bx2 - x1, by2 - y1], dtype=np.float32),
                      "mask_input": cv2.resize(coarse[y1:y2, x1:x2], (256, 256),
                                               interpolation=cv2.INTER_AREA)[None]}
            if point:
                kwargs["point_coords"] = np.array([[point[0] - x1, point[1] - y1]], dtype=np.float32)
                kwargs["point_labels"] = np.array([1])
            masks, _, _ = pred.predict(multimask_output=False, return_logits=True, **kwargs)
        return x1, y1, x2, y2, masks[0]  # merged with the normal logits in merge_sharp

    def outline(self, mask):
        """A mask's outline as polygons in image pixels, traced at preview resolution."""
        h, w = mask.shape
        pw, ph = self.preview_size
        small = cv2.resize(mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / pw, h / ph
        polygons = []
        for c in contours:
            c = cv2.approxPolyDP(c, 0.8, True)[:, 0]
            if len(c) >= 3:
                polygons.append([[round(float(x) * sx, 1), round(float(y) * sy, 1)] for x, y in c])
        return polygons

    def guide_prompt(self, path):
        """Turn the user's GIMP selection (a rough lasso, say) into a prompt: its bounding
        box plus a click at the point deepest inside it. On rough lassos around trunks in
        a forest photo this matched the object best (IoU 0.82 on average, 0.97-0.99 on the
        clear trunks); passing the lasso itself as SAM2's mask input did much worse (0.48),
        as SAM2 then follows the lasso's shape instead of the object.

        Each separate area of the selection is its own lasso (a selection around a tree and
        a bench gives two prompts): one box and point for all of them made the box cover
        both and put the point in the tree only. Specks smaller than 0.05% of the image
        are ignored, unless that leaves nothing."""
        h, w = self.image.shape[:2]
        guide = np.fromfile(path, np.uint8).reshape(h, w) > 127
        if not guide.any():
            return None
        n, labels, stats, _ = cv2.connectedComponentsWithStats(guide.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i in np.argsort(-areas) if areas[i] >= max(100, 0.0005 * h * w)]
        keep = keep or [int(np.argmax(areas)) + 1]
        prompts = []
        for i in keep:
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            part = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
            dist = cv2.distanceTransform(np.pad(part, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
            prompts.append({"box": [x, y, x + bw, y + bh], "point": [x + int(px), y + int(py)]})
        return {"prompts": prompts, "polygons": self.outline(guide)}

    def object_masks(self, req):
        """Each object's boolean mask, in order: (object, mask)."""
        # Threshold 50 is SAM2's own cut-off (logit 0); 100 selects more, 0 less. The
        # +-2 logit range keeps uncertain masks controllable: at +-5 a two-click forest
        # selection swung from 0% to 47% of the image.
        cut = (50 - req.get("threshold", 50)) / 25.0
        granularity = req.get("granularity", "auto")
        sharp = bool(req.get("sharp"))
        wanted = {self.sharp_key(o, granularity) for o in req["objects"]} if sharp else set()
        with self.sharp_lock:  # forget queued work for objects that are gone (undo, new click)
            self.sharp_jobs = [j for j in self.sharp_jobs if j[0] in wanted]
            for key in [k for k in self.sharp_merged if k not in wanted]:
                del self.sharp_merged[key]  # full-size, like the cache
        self.cache_used = set()
        self.sharp_pending = self.sharp_refined = 0
        for obj in req["objects"]:
            points = obj.get("points", [])
            if obj.get("box"):
                point = points[0][:2] if points else None  # e.g. from the GIMP selection
                logits = self.box_logits(obj["box"], point)
                points = []  # a box object has no refinement clicks
            elif points:
                logits = self.object_logits(points, granularity)
            else:
                continue
            if sharp and len(obj.get("points", [])) <= 1:
                logits, state = self.sharp_logits(obj, granularity, logits, cut, wait=bool(req.get("final")))
                self.sharp_pending += state == "pending"
                self.sharp_refined += state == "done"
            mask = logits > cut
            # SAM2 sometimes ignores a refinement click (e.g. an "exclude" inside a uniform
            # object). Keep the Fuzzy Select promise anyway: a Shift-clicked spot ends up
            # selected and a Ctrl-clicked spot ends up deselected, by adding or removing
            # the small segment SAM2 finds at that spot.
            for x, y, label in points[1:]:
                if label == 1 and not mask[y, x]:
                    mask |= self.object_logits([[x, y, 1]], "detail") > cut
                elif label == 0 and mask[y, x]:
                    mask &= ~(self.object_logits([[x, y, 1]], "detail") > cut)
            yield obj, mask
        self.prune_cache()

    def selection(self, req):
        """Combined boolean mask, before grow/feather."""
        h, w = self.image.shape[:2]
        add = np.zeros((h, w), bool)
        sub = np.zeros((h, w), bool)
        for obj, mask in self.object_masks(req):
            if obj["sign"] > 0:
                add |= mask
            else:
                sub |= mask
        return add & ~sub

    def slice_labels(self, req):
        """For slicing: the slice number of each pixel (0 = the rest). Where objects overlap,
        the smaller one keeps the pixels, whichever came first: an object clicked inside a
        slice splits it, and a big object clicked later (the car around four shopping bags)
        doesn't take the stray bits of its mask that lie on the smaller ones."""
        h, w = self.image.shape[:2]
        labels = np.zeros((h, w), np.uint8)
        owner = np.full((h, w), np.iinfo(np.uint32).max, np.uint32)  # area of each pixel's object
        fill, grow = float(req.get("fill_holes", 0)), int(req.get("grow", 0))
        for obj, mask in self.object_masks(req):
            if obj["sign"] > 0 and fill > 0 and mask.any():
                mask = fill_holes(mask, np.pi * (fill / 2) ** 2)
            if grow and mask.any():  # growing the result means giving less back to the rest
                mask = grow_mask(mask, grow if obj["sign"] > 0 else -grow)
            if obj["sign"] > 0:
                area = int(mask.sum())
                mask &= owner >= area  # (the same size: the newer one)
                labels[mask] = obj.get("slice", 1)
                owner[mask] = area
            else:  # back to the rest, which later objects can take again
                labels[mask] = 0
                owner[mask] = np.iinfo(np.uint32).max
        # For hover_splits: the object sizes only on every n-th pixel (about 1000 along the long
        # side), which is all it reads; in full they took 4 bytes a pixel (192 MB at 8000 px).
        # It goes by the slices as clicked, before snapping: a snapped pixel is still rest
        # that a click takes, not a slice it would split.
        n = max(1, round(max(h, w) / 1000))
        snap = int(req.get("snap", 0))
        self.slice_state = (self.image, labels.copy() if snap > 0 else labels, owner[::n, ::n].copy(), n, {})
        self.snapped = snap_gaps(labels, self.image, snap)
        return labels

    def hover_splits(self, mask):
        """When slicing: the slices a click on `mask` would cut into, by the same rule as
        slice_labels (a click takes pixels from bigger objects' slices; Ctrl gives back any it
        touches), from the last slicing; and their outlines. Small overlaps (along an outline)
        don't count."""
        state = self.slice_state
        if state is None or state[0] is not self.image:
            return {"click": [], "ctrl": []}, {}
        _, labels, sub_owner, n, outlines = state
        area = int(mask.sum())
        # Counted on every n-th pixel (see slice_labels): on a 4000 px photo every pixel added
        # 50 ms to each hover
        sub_labels, sub_mask = labels[::n, ::n], mask[::n, ::n]
        least = max(50, area // 200) / (n * n)

        def cut(where):
            counts = np.bincount(sub_labels[where], minlength=256)
            return [int(k) for k in np.flatnonzero(counts[1:] >= least) + 1]

        splits = {"click": cut(sub_mask & (sub_owner >= area)), "ctrl": cut(sub_mask)}
        smallest = 0.0005 * labels.size  # (not every small hole in a slice: they hid the outline)
        for k in set(splits["click"] + splits["ctrl"]) - set(outlines):
            outlines[k] = [poly for poly in self.outline(labels == k)
                           if cv2.contourArea(np.array(poly, np.float32)) >= smallest]
        return splits, {str(k): outlines[k] for k in set(splits["click"] + splits["ctrl"])}

    def prune_cache(self):
        """Drop the oldest results of objects not in the current selection beyond
        SPARE_CACHE_BYTES; those in it are always kept."""
        spare = [k for k in self.cache if k not in self.cache_used]
        size = sum(self.cache[k][0].nbytes for k in spare)
        for key in spare:  # oldest first
            if size <= SPARE_CACHE_BYTES:
                break
            size -= self.cache.pop(key)[0].nbytes

    def soft_edges(self, mask, edge, max_side=None):
        """ViTMatte alpha in a band around the mask's edge (on a crop around the mask).
        max_side: for previews, run ViTMatte on the crop scaled down to at most this many
        pixels on its long side (much faster) and scale the alpha back up."""
        if self.vitmatte is None:
            self.free_lama()
            self.load_vitmatte()
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        margin = 5 * edge + 16
        y1, y2 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, h)
        x1, x2 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, w)
        crop = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
        image = np.ascontiguousarray(self.image[y1:y2, x1:x2])
        ch, cw = crop.shape
        scale = min(1.0, max_side / max(ch, cw)) if max_side else 1.0
        if scale < 1:
            size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            crop = ((cv2.resize(crop, size, interpolation=cv2.INTER_AREA) > 127) * 255).astype(np.uint8)
        # Keep the soft band the same width in image pixels at any scale.
        k = max(1, round(edge * scale))
        kernel = np.ones((k, k), np.uint8)
        trimap = np.zeros(crop.shape, np.float32)
        trimap[cv2.dilate(crop, kernel, iterations=5) == 255] = 0.5
        trimap[cv2.erode(crop, kernel, iterations=5) == 255] = 1
        with torch.no_grad():
            alpha = self.vitmatte({
                "image": torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255,
                "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
            })["phas"].flatten(0, 2).cpu().numpy()
        if scale < 1:
            alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((h, w), np.float32)
        out[y1:y2, x1:x2] = alpha
        return out

    def cmd_render(self, req):
        if req.get("slices"):
            return self.render_slices(req)
        t0 = time.time()
        mask = self.selection(req)
        fill = float(req.get("fill_holes", 0))
        if fill > 0 and mask.any():
            mask = fill_holes(mask, np.pi * (fill / 2) ** 2)
        grow = int(req.get("grow", 0))
        if grow and mask.any():
            mask = grow_mask(mask, grow)
        area = int(mask.sum())  # (after shrinking, which can leave nothing)
        if req.get("soft") and mask.any() and (req.get("final") or req.get("soft_preview")):
            # Previews use a reduced-size crop: seconds become a fraction of a second.
            max_side = None if req.get("final") else int(req.get("soft_preview_size", 1024))
            alpha = self.soft_edges(mask, int(req.get("edge", 4)), max_side)
        else:
            alpha = mask.astype(np.float32)
        feather = float(req.get("feather", 0))
        if feather > 0:
            # GIMP's feather radius ~ the width of the transition; sigma = radius / 2.
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather / 2)

        if req.get("final"):
            alpha8 = np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
            with open(req["out"], "wb") as f:
                f.write(np.ascontiguousarray(alpha8).tobytes())
            if req.get("patch_out") and area:
                box = self.remove_patch(alpha > 0.5, req["patch_out"], req.get("remove_context", REMOVE_CONTEXT))
                return {"area": area, "seconds": round(time.time() - t0, 3), "sharp_refined": self.sharp_refined,
                        "patch_box": box}
        elif req.get("remove_preview"):
            box = self.remove_patch(alpha > 0.5, req["remove_preview"], req.get("remove_context", REMOVE_CONTEXT),
                                    png=True) if area else None
            return {"area": area, "seconds": round(time.time() - t0, 3), "patch": req["remove_preview"],
                    "patch_box": box, "sharp_pending": self.sharp_pending}
        else:
            # A fresh file per preview, so the window can read one while the next is written.
            path = req.get("preview_out") or self.preview_path
            self.write_preview(alpha, path, req.get("view", "overlay"))
            return {"area": area, "seconds": round(time.time() - t0, 3), "preview": path,
                    "sharp_pending": self.sharp_pending, "sharp_refined": self.sharp_refined}
        return {"area": area, "seconds": round(time.time() - t0, 3), "sharp_refined": self.sharp_refined}

    def render_slices(self, req):
        t0 = time.time()
        labels = self.slice_labels(req)
        counts = np.bincount(labels.ravel(), minlength=256)
        slices = [{"slice": int(k), "area": int(counts[k])} for k in np.flatnonzero(counts[1:]) + 1]
        reply = {"area": int(counts[1:].sum()), "slices": slices, "snapped": self.snapped}
        if req.get("final"):
            for s in slices:
                x, y, bw, bh = cv2.boundingRect((labels == s["slice"]).view(np.uint8))
                s["box"] = [x, y, x + bw, y + bh]
            with open(req["out"], "wb") as f:
                f.write(labels.tobytes())
        else:
            path = req.get("preview_out") or self.preview_path
            self.write_slice_preview(labels, path, req.get("view", "overlay"))
            reply.update(preview=path, sharp_pending=self.sharp_pending)
        return dict(reply, seconds=round(time.time() - t0, 3), sharp_refined=self.sharp_refined)

    def write_slice_preview(self, labels, path, view="overlay"):
        """Each slice in its own colour, with an outline; the rest is left clear (black in
        the Mask view)."""
        pw, ph = self.preview_size
        small = cv2.resize(labels, (pw, ph), interpolation=cv2.INTER_NEAREST)
        palette = np.zeros((256, 3), np.uint8)
        palette[1:] = [SLICE_COLOURS[i % len(SLICE_COLOURS)] for i in range(255)]
        if view == "mask":
            cv2.imwrite(path, cv2.cvtColor(palette[small], cv2.COLOR_RGB2BGR))
            return
        overlay = np.zeros((ph, pw, 4), np.uint8)
        overlay[..., :3] = palette[small]
        overlay[..., 3] = np.where(small > 0, 120, 0)
        for k in np.unique(small):
            if k == 0:
                continue
            contours, _ = cv2.findContours((small == k).view(np.uint8), cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                xy = contour[:, 0]
                dash = (np.arange(len(xy)) // 4) % 2 == 1
                overlay[xy[:, 1], xy[:, 0]] = (0, 0, 0, 255)
                overlay[xy[dash, 1], xy[dash, 0]] = (255, 255, 255, 255)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))

    def write_preview(self, alpha, path, view="overlay"):
        pw, ph = self.preview_size
        small = cv2.resize(alpha, (pw, ph), interpolation=cv2.INTER_AREA)
        if view == "mask":
            # The selection itself, white on black: partial selection (soft edges,
            # feathering) shows as grey, which the tinted overlay can't make visible.
            cv2.imwrite(path, np.round(np.clip(small, 0, 1) * 255).astype(np.uint8))
            return
        overlay = np.zeros((ph, pw, 4), np.uint8)
        overlay[..., :3] = TINT
        overlay[..., 3] = np.round(np.clip(small, 0, 1) * 110).astype(np.uint8)
        # Outline (where the preview crosses 50%), drawn as black/white dashes like ants.
        edges = (small > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            xy = contour[:, 0]
            dash = (np.arange(len(xy)) // 4) % 2 == 1
            overlay[xy[:, 1], xy[:, 0]] = (0, 0, 0, 255)
            overlay[xy[dash, 1], xy[dash, 0]] = (255, 255, 255, 255)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))


def serve(server, lines, out):
    """Answer requests from `lines` on `out` until "quit" (returns True) or end of input
    (returns False)."""
    for line in lines:
        if not line.strip():
            continue
        req = json.loads(line)
        reply = {"id": req.get("id")}
        try:
            if req["cmd"] not in ("quit", "bye"):
                reply.update(getattr(server, "cmd_" + req["cmd"])(req))
            reply["ok"] = True
        except LoadFailed as e:  # already explained for the user
            reply.update({"ok": False, "error": str(e), "fatal": True})
        except Exception as e:
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"
            if "out of memory" in error.lower() and server.device == "cuda":
                torch.cuda.empty_cache()  # so that trying again can work
                error = gpu_memory_message(error)
            reply.update({"ok": False, "error": error})
        out.write(json.dumps(reply) + "\n")
        out.flush()
        if req["cmd"] == "quit":
            return True
        if req["cmd"] == "bye":
            return False
    return False


def gpu_memory_message(error):
    """A CUDA out-of-memory error explained: the GPU's memory and who uses it (nvidia-smi).
    On a 2 GB GPU, a second backend left from an earlier run (1.6 GB with SAM 2.1 large)
    made a new one fail while loading the model."""
    lines = ["The GPU ran out of memory."]
    try:
        def query(what, fields):
            return subprocess.run(["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        total, used = (x.strip() for x in query("gpu", "memory.total,memory.used")[0].split(","))
        lines[0] = f"The GPU ran out of memory ({used} of {total} in use)."
        apps = [[x.strip() for x in a.split(",")] for a in query("compute-apps", "pid,process_name,used_memory")]
        # (Windows lists every program that draws on screen, without its memory: keep those
        # whose memory is known, and Python processes, which may be backends)
        apps = [a for a in apps if any(c.isdigit() for c in a[2]) or "python" in a[1].lower()]
        if apps:
            lines.append("Using it: " + "; ".join(
                f"process {pid} ({os.path.basename(name)}, {mem})" + (" - this backend" if pid == str(os.getpid()) else "")
                for pid, name, mem in apps) + ".")
    except Exception:
        pass
    lines += ["Another AI Fuzzy Select backend left from an earlier run may be holding it: it stops by "
              "itself after 10 minutes without use, or restart the computer.",
              "On a GPU with less than 3 GB, use SAM 2.1 small: run the installer again with --model small.",
              "", "(" + error.splitlines()[0][:300] + ")"]
    return "\n".join(lines)


def process_start(pid):
    """When process `pid` started, as a string, or None if unknown. Written to the state
    file so the installer and uninstaller can tell this server from a later process that
    happens to get the same pid; they compute it the same way."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            k.OpenProcess.restype = wintypes.HANDLE
            handle = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                times = [wintypes.FILETIME() for _ in range(4)]
                if not k.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                    return None
                return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
            finally:
                k.CloseHandle(handle)
        import subprocess
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def listen(server, settings):
    """Stay alive between plugin runs: accept one connection at a time on a local port and
    exit after AISELECT_KEEPALIVE seconds without one. The port and a random token go to
    the state file (in the user's own temp folder); a connection must start with that token,
    so other users on the machine can't use the server to read images or write files."""
    token = os.environ["AISELECT_TOKEN"]
    state_file = os.environ["AISELECT_STATE"]
    keep_alive = float(os.environ.get("AISELECT_KEEPALIVE", "600"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    state = {"port": sock.getsockname()[1], "pid": os.getpid(), "started": process_start(os.getpid()),
             "token": token, "version": VERSION,
             "settings": settings}
    tmp = state_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_file)
    print(f"listening on port {state['port']} (pid {os.getpid()}, version {VERSION})", flush=True)
    # Waiting for a connection: at least a minute, so that with keepAliveMinutes 0 (stop when
    # the window closes, which sends "quit") the plugin that started it can still connect. A
    # timeout of 0 would make accept() non-blocking and fail at once.
    accept_wait = max(keep_alive, 60)
    # An open connection that sends nothing for this long is dropped. A plugin that went away
    # without closing it (a crash, a killed process) otherwise kept the server waiting on it
    # for good: answering no one, never reaching the idle exit. A window that was only left
    # open reconnects by itself (Backend._exchange).
    idle_wait = float(os.environ.get("AISELECT_IDLE_WAIT", max(keep_alive, 600)))  # (shorter in tests)
    try:
        while True:
            sock.settimeout(accept_wait)
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                print(f"idle for {accept_wait:.0f} s, exiting", flush=True)
                return
            conn.settimeout(10)  # the hello comes at once
            with conn, conn.makefile("r", encoding="utf-8", newline="\n") as rfile, \
                    conn.makefile("w", encoding="utf-8", newline="\n") as wfile:
                try:
                    hello = json.loads(rfile.readline() or "{}")
                    if hello.get("cmd") != "hello" or not hmac.compare_digest(str(hello.get("token", "")), token):
                        continue
                    wfile.write(json.dumps({"id": hello.get("id"), "ok": True, "version": VERSION,
                                            "pid": os.getpid()}) + "\n")
                    wfile.flush()
                    conn.settimeout(idle_wait)
                    if serve(server, rfile, wfile):
                        print("quit requested", flush=True)
                        return
                except socket.timeout:
                    print("connection silent too long (no hello in 10 s, or no request in "
                          f"{idle_wait:.0f} s), closed", flush=True)
                except (OSError, ValueError):
                    traceback.print_exc()  # the plugin went away mid-request; wait for the next
    finally:
        sock.close()
        try:  # remove the state file only if it is still ours (closed first: Windows
            # can't delete an open file)
            with open(state_file, encoding="utf-8") as f:
                ours = json.load(f).get("token") == token
            if ours:
                os.remove(state_file)
        except (OSError, ValueError):
            pass


def main():
    args = sys.argv[1:]
    listening = args[:1] == ["--listen"]
    if listening:
        args = args[1:]
        log = os.environ.get("AISELECT_LOG")
        if log:  # no console to write to: log to a file instead
            sys.stdout = sys.stderr = open(log, "w", encoding="utf-8", buffering=1)
    settings = json.loads(args[0]) if args else {}
    server = Server(settings)
    if listening:
        listen(server, settings)
    else:
        serve(server, sys.stdin, PROTOCOL)


if __name__ == "__main__":
    main()
