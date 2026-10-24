"""
Backend for the GIMP "AI Fuzzy Select" plugin. Runs in the plugin's own virtual
environment (made by install_ai_select.py) and stays alive while the plugin window is
open, so SAM2 is loaded and the image is encoded only once; each click then costs only
SAM2's small mask decoder.

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout.
  {"cmd": "load", "image": png, "preview": png, "preview_width": w, "preview_height": h}
      encode the image; "preview" is where render writes its overlay
  {"cmd": "render", "objects": [...], "threshold": 0-100, "granularity": "auto"|"detail"|
   "part"|"whole", "fill_holes": px, "grow": px, "feather": px, "soft": false,
   "final": false, "soft_preview": false, "out": raw, "preview_out": png}
      objects: [{"sign": 1 or -1, "points": [[x, y, label], ...]} or
                {"sign": 1 or -1, "box": [x1, y1, x2, y2]}, ...] in image pixels;
      the selection is (union of sign 1 objects) minus (union of sign -1 objects).
      Not final: write a tinted preview overlay PNG with an outline to "preview_out"
      (with "soft" and "soft_preview": ViTMatte edges computed at reduced size;
      "view": "mask" writes the selection as a white-on-black image instead).
      Final: write the full-size 8-bit selection to "out" (optionally with ViTMatte edges).
  {"cmd": "quit"}
Every reply has "id" (echoed from the request) and "ok", plus "error" on failure.
"""

import json
import os
import sys
import time
import traceback

# Keep stdout for the protocol only; anything libraries print goes to stderr.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TINT = (64, 160, 255)  # overlay colour (RGB)


def fill_holes(mask, max_area):
    """Fill enclosed holes (not touching the image border) of at most max_area pixels."""
    h, w = mask.shape
    n, cc, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    x, y, cw, ch, area = (stats[:, i] for i in range(5))
    enclosed = (x > 0) & (y > 0) & (x + cw < w) & (y + ch < h)
    small = np.flatnonzero(enclosed & (area <= max_area))
    small = small[small > 0]
    return mask | np.isin(cc, small) if len(small) else mask


class Server:
    def __init__(self, settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = None
        self.vitmatte = None
        self.image = None
        self.cache = {}  # (granularity, points) -> (full-res logits, low-res logits)

    # --- models --------------------------------------------------------------------

    def load_sam2(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2("configs/sam2/sam2_hiera_l.yaml", self.settings["sam2Checkpoint"],
                           device=self.device)
        self.predictor = SAM2ImagePredictor(model)

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

    # --- commands ------------------------------------------------------------------

    def cmd_load(self, req):
        t0 = time.time()
        if self.predictor is None:
            self.load_sam2()
        image = cv2.imread(req["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {req['image']}")
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.preview_path = req["preview"]
        self.preview_size = (req["preview_width"], req["preview_height"])
        self.cache.clear()
        with torch.no_grad():
            self.predictor.set_image(self.image)
        h, w = self.image.shape[:2]
        return {"width": w, "height": h, "seconds": round(time.time() - t0, 2), "device": self.device}

    def object_logits(self, points, granularity):
        """Full-res mask logits for one object's clicks. The first click picks one of
        SAM2's three candidates (detail / part / whole); later clicks refine it, feeding
        the previous low-res logits back in as SAM2's own interactive demo does."""
        key = (granularity, tuple(map(tuple, points)))
        if key in self.cache:
            return self.cache[key][0]
        coords = np.array([p[:2] for p in points], dtype=np.float32)
        labels = np.array([p[2] for p in points], dtype=np.int32)
        with torch.no_grad():
            if len(points) == 1:
                masks, scores, low = self.predictor.predict(
                    point_coords=coords, point_labels=labels, multimask_output=True,
                    return_logits=True)
                # Only candidates that contain the clicked pixel: SAM2's smallest candidate
                # is sometimes a stray fragment elsewhere (31 px, score 0.09, next to a
                # clicked disc), and picking it made Ctrl+click and "Detail" do nothing.
                x, y = points[0][:2]
                containing = [i for i in range(len(masks)) if masks[i][y, x] > 0] or list(range(len(masks)))
                if granularity == "auto":
                    pick = max(containing, key=lambda i: scores[i])
                else:
                    by_size = sorted(containing, key=lambda i: int((masks[i] > 0).sum()))
                    pick = {"detail": by_size[0], "part": by_size[len(by_size) // 2],
                            "whole": by_size[-1]}[granularity]
                full, low = masks[pick], low[pick]
            else:
                self.object_logits(points[:-1], granularity)
                prev_low = self.cache[(granularity, tuple(map(tuple, points[:-1])))][1]
                masks, _, low = self.predictor.predict(
                    point_coords=coords, point_labels=labels, mask_input=prev_low[None],
                    multimask_output=False, return_logits=True)
                full, low = masks[0], low[0]
        self.cache[key] = (full, low)
        return full

    def box_logits(self, box):
        """Full-res mask logits for the object inside a dragged box [x1, y1, x2, y2]. SAM2
        recommends a single output for box prompts: the box already says how big the
        object is, so Granularity doesn't apply."""
        key = ("box", tuple(box))
        if key not in self.cache:
            with torch.no_grad():
                masks, _, low = self.predictor.predict(
                    box=np.array(box, dtype=np.float32), multimask_output=False, return_logits=True)
            self.cache[key] = (masks[0], low[0])
        return self.cache[key][0]

    def selection(self, req):
        """Combined boolean mask, before grow/feather."""
        # Threshold 50 is SAM2's own cut-off (logit 0); 100 selects more, 0 less. The
        # +-2 logit range keeps uncertain masks controllable: at +-5 a two-click forest
        # selection swung from 0% to 47% of the image.
        cut = (50 - req.get("threshold", 50)) / 25.0
        h, w = self.image.shape[:2]
        add = np.zeros((h, w), bool)
        sub = np.zeros((h, w), bool)
        granularity = req.get("granularity", "auto")
        for obj in req["objects"]:
            points = obj.get("points", [])
            if obj.get("box"):
                mask = self.box_logits(obj["box"]) > cut
                points = []  # a box object has no refinement clicks
            elif points:
                mask = self.object_logits(points, granularity) > cut
            else:
                continue
            # SAM2 sometimes ignores a refinement click (e.g. an "exclude" inside a uniform
            # object). Keep the Fuzzy Select promise anyway: a Shift-clicked spot ends up
            # selected and a Ctrl-clicked spot ends up deselected, by adding or removing
            # the small segment SAM2 finds at that spot.
            for x, y, label in points[1:]:
                if label == 1 and not mask[y, x]:
                    mask |= self.object_logits([[x, y, 1]], "detail") > cut
                elif label == 0 and mask[y, x]:
                    mask &= ~(self.object_logits([[x, y, 1]], "detail") > cut)
            if obj["sign"] > 0:
                add |= mask
            else:
                sub |= mask
        return add & ~sub

    def soft_edges(self, mask, edge, max_side=None):
        """ViTMatte alpha in a band around the mask's edge (on a crop around the mask).
        max_side: for previews, run ViTMatte on the crop scaled down to at most this many
        pixels on its long side (much faster) and scale the alpha back up."""
        if self.vitmatte is None:
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
        t0 = time.time()
        mask = self.selection(req)
        fill = float(req.get("fill_holes", 0))
        if fill > 0 and mask.any():
            mask = fill_holes(mask, np.pi * (fill / 2) ** 2)
        area = int(mask.sum())
        grow = int(req.get("grow", 0))
        if grow and area:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1,) * 2)
            m8 = mask.astype(np.uint8)
            mask = (cv2.dilate(m8, k) if grow > 0 else cv2.erode(m8, k)).astype(bool)
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
        else:
            # A fresh file per preview, so the window can read one while the next is written.
            path = req.get("preview_out") or self.preview_path
            self.write_preview(alpha, path, req.get("view", "overlay"))
            return {"area": area, "seconds": round(time.time() - t0, 3), "preview": path}
        return {"area": area, "seconds": round(time.time() - t0, 3)}

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


def main():
    settings = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    server = Server(settings)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        reply = {"id": req.get("id")}
        try:
            if req["cmd"] != "quit":
                reply.update(getattr(server, "cmd_" + req["cmd"])(req))
            reply["ok"] = True
        except Exception as e:
            traceback.print_exc()
            reply.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        PROTOCOL.write(json.dumps(reply) + "\n")
        PROTOCOL.flush()
        if req["cmd"] == "quit":
            break


if __name__ == "__main__":
    main()
