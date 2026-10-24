#!/usr/bin/env python3
"""
Installer for AI Fuzzy Select, a GIMP 3 plugin that works like Fuzzy Select but selects
by object: click an object and SAM2 selects it; Shift+click adds, Ctrl+click removes; the
result is a normal GIMP selection.

What it does:
  1. Creates a virtual environment with the Python running this script (3.10-3.13).
  2. Installs a PyTorch build matching your GPU (CPU-only if there is none).
  3. Installs SAM2 (pinned) and downloads its checkpoint.
  4. Optionally (default on) installs ViTMatte for the "Soft edges" option: detectron2
     and the ViTMatte code from Matte Anything are used as plain source, so no C++/CUDA
     compiler is needed.
  5. Copies the plugin into GIMP 3's plug-ins folder and writes its settings.
  6. Tests the backend on a synthetic image with a known answer, then (if GIMP is found)
     the plugin inside headless GIMP and its window off-screen with GIMP's own Python.

Usage:
  python install_ai_select.py
  python install_ai_select.py --no-soft-edges          # lighter: SAM2 only
  python install_ai_select.py --sam2-checkpoint sam2_hiera_large.pt   # reuse a download
  python install_ai_select.py --test-only

Only the Python standard library is needed to run this script.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_SRC = HERE / "gimp_ai_select"
PLUGIN_FILES = ["aiselect.py", "aiselect_server.py"]

# --- Pinned sources (tested together) ---------------------------------------------

SAM2 = ("facebookresearch/sam2", "2b90b9f5ceec907a1c18123530e92e794ad901a4")
SAM2_CHECKPOINT = ("sam2_hiera_large.pt",
                   "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
# Soft edges: ViTMatte's model code and config live in Matte Anything; its config system
# is detectron2's LazyConfig.
MATTE_ANYTHING = ("hustvl/Matte-Anything", "874951ac0b8d324ad9f885baf1c7de711b8dc1b0")
DETECTRON2 = ("facebookresearch/detectron2", "1e3e13bbf607b54f62205c4c33922521822fb298")
VITMATTE_GDRIVE_ID = "1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW"  # ViTMatte_B_DIS.pth (Matte Anything's README)

TORCH_INDEX = "https://download.pytorch.org/whl/{}"
SOFT_EDGE_PACKAGES = [
    "setuptools<81",  # detectron2's model_zoo still uses pkg_resources
    "fvcore>=0.1.5,<0.1.6",
    "omegaconf>=2.1,<2.4",
    "timm",
    "cloudpickle",
    "pycocotools",  # imported by detectron2.structures, which ViTMatte's backbone pulls in
    "tabulate",
    "termcolor",
    "yacs",
    "gdown",
]

IS_WINDOWS = os.name == "nt"


# --- Output helpers -----------------------------------------------------------------


def step(msg):
    print(f"\n==> {msg}", flush=True)


def info(msg):
    print(f"    {msg}", flush=True)


def ok(msg):
    print(f"    [OK]   {msg}", flush=True)


def fail(msg):
    print(f"    [FAIL] {msg}", flush=True)


class InstallError(Exception):
    pass


def run(cmd, env=None, cwd=None, capture=False):
    """Run a command, streaming output unless capture=True. Raises on failure."""
    printable = " ".join(str(c) for c in cmd)
    info(f"$ {printable}" if len(printable) < 300 else f"$ {printable[:300]}...")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        [str(c) for c in cmd], env=full_env, cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        if capture:
            print(result.stdout)
        raise InstallError(f"Command failed (exit {result.returncode}): {printable[:300]}")
    return result.stdout if capture else ""


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ai-select-installer"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and time.time() - last > 0.5:
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)",
                      end="", flush=True)
                last = time.time()
        if total:
            print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB (100.0%)", flush=True)
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise InstallError(f"Incomplete download of {url}: got {done} of {total} bytes")
    os.replace(tmp, dest)


def fetch_repo(repo, commit, dest):
    """Download a GitHub repo at a pinned commit and extract it to dest (no git needed)."""
    dest = Path(dest)
    if dest.exists():
        ok(f"Already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / f"{dest.name}.zip"
    download(f"https://github.com/{repo}/archive/{commit}.zip", zip_path)
    extract_dir = Path(tempfile.mkdtemp(dir=dest.parent))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    (top,) = extract_dir.iterdir()
    shutil.move(str(top), str(dest))
    shutil.rmtree(extract_dir)
    zip_path.unlink()
    ok(f"{repo} @ {commit[:10]} -> {dest}")
    return dest


# --- Environment detection --------------------------------------------------------


def default_install_dir():
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "gimp-ai-select"


def detect_nvidia_gpu():
    """Return (name, compute_capability) or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        name, _, cap = out.splitlines()[0].rpartition(",")
        return name.strip(), float(cap)
    except Exception:
        return None


def find_gimp_plugins_dir():
    """The plug-ins folder of the newest GIMP 3.x user profile, or None."""
    if IS_WINDOWS:
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "GIMP"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "GIMP"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "GIMP"
    versions = []
    for d in root.iterdir() if root.is_dir() else []:
        try:
            major, minor = (int(x) for x in d.name.split("."))
        except ValueError:
            continue
        if major >= 3:
            versions.append(((major, minor), d))
    return max(versions)[1] / "plug-ins" if versions else None


def find_gimp_console():
    """GIMP's headless executable, used to test the plugin inside real GIMP."""
    for name in ["gimp-console-3", "gimp-console"]:
        if shutil.which(name):
            return Path(shutil.which(name))
    candidates = []
    if IS_WINDOWS:
        for base in [os.environ.get("LOCALAPPDATA", "") + r"\Programs", os.environ.get("ProgramFiles", "")]:
            candidates.append(Path(base) / "GIMP 3" / "bin" / "gimp-console-3.exe")
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/GIMP.app/Contents/MacOS/gimp-console"))
    return next((c for c in candidates if c.exists()), None)


def venv_python(venv_dir):
    return Path(venv_dir) / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def site_packages(py):
    return Path(run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                    capture=True).strip())


# --- Install steps ----------------------------------------------------------------


def create_venv(venv_dir):
    py = venv_python(venv_dir)
    if py.exists():
        ok(f"Reusing existing virtual environment: {venv_dir}")
    else:
        run([sys.executable, "-m", "venv", venv_dir])
        ok(f"Created virtual environment: {venv_dir}")
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"], capture=True)
    return py


def install_sam2(py, index, src):
    run([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", TORCH_INDEX.format(index)])
    run([py, "-m", "pip", "install", "opencv-python-headless", "numpy"])
    sam2 = fetch_repo(*SAM2, src / "sam2")
    # --no-build-isolation: SAM2's build requires torch and would otherwise download a
    # second copy. SAM2_BUILD_CUDA=0: its optional CUDA extension needs a compiler; SAM2
    # works without it.
    run([py, "-m", "pip", "install", "--no-build-isolation", sam2], env={"SAM2_BUILD_CUDA": "0"})


def install_soft_edges(py, src):
    """ViTMatte without compiling anything: detectron2 and Matte Anything's ViTMatte code
    are put on the path as source; Matte Anything uses only detectron2's pure-Python parts."""
    run([py, "-m", "pip", "install", *SOFT_EDGE_PACKAGES])
    # fairscale's old setup.py builds a broken "unknown-0.0.0" wheel in an isolated build
    # with current setuptools; build it against the venv's own setuptools instead.
    run([py, "-m", "pip", "install", "--no-build-isolation", "fairscale"])
    d2 = fetch_repo(*DETECTRON2, src / "detectron2")
    zoo_configs = d2 / "detectron2" / "model_zoo" / "configs"
    if not zoo_configs.exists():  # setup.py normally links these into the package
        shutil.copytree(d2 / "configs", zoo_configs)
    vitmatte = fetch_repo(*MATTE_ANYTHING, src / "Matte-Anything")
    pth = site_packages(py) / "ai_select_sources.pth"
    pth.write_text(f"{d2}\n", encoding="utf-8")
    ok(f"detectron2 registered in {pth}")
    return vitmatte


def get_sam2_checkpoint(dest, existing):
    if existing:
        existing = Path(existing).resolve()
        if not existing.exists():
            raise InstallError(f"--sam2-checkpoint not found: {existing}")
        ok(f"Using existing SAM2 checkpoint: {existing}")
        return existing
    if dest.exists():
        ok(f"Already present: {dest}")
    else:
        info(f"Downloading {SAM2_CHECKPOINT[0]} (~900 MB)")
        download(SAM2_CHECKPOINT[1], dest)
    return dest


def get_vitmatte_checkpoint(py, dest, existing):
    if existing:
        existing = Path(existing).resolve()
        if not existing.exists():
            raise InstallError(f"--vitmatte-checkpoint not found: {existing}")
        ok(f"Using existing ViTMatte checkpoint: {existing}")
        return existing
    if dest.exists():
        ok(f"Already present: {dest}")
        return dest
    info("Downloading ViTMatte_B_DIS.pth from Google Drive")
    try:
        run([py, "-m", "gdown", VITMATTE_GDRIVE_ID, "-O", dest])
    except InstallError:
        raise InstallError(
            "Google Drive refused the ViTMatte download (it rate-limits). Download it from "
            f"https://drive.google.com/file/d/{VITMATTE_GDRIVE_ID}/view, then re-run with "
            "--vitmatte-checkpoint path/to/ViTMatte_B_DIS.pth (or use --no-soft-edges).")
    return dest


def install_plugin(plugins_dir, settings):
    target = Path(plugins_dir) / "aiselect"
    target.mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_FILES:
        shutil.copyfile(PLUGIN_SRC / name, target / name)
        if not IS_WINDOWS:
            os.chmod(target / name, 0o755)
    (target / "aiselect_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    ok(f"Plugin installed: {target}")
    return target


# --- Tests --------------------------------------------------------------------------

IMPORT_TEST = r"""
import sys, torch, cv2, numpy
from sam2.build_sam import build_sam2
print("python", sys.version.split()[0], "| torch", torch.__version__, "| numpy", numpy.__version__,
      "| opencv", cv2.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda")
    (x @ x).sum().item()  # a real kernel catches 'no kernel image' architecture mismatches
    print("cuda_kernel_ok")
if len(sys.argv) > 1:
    import detectron2
    from detectron2.config import LazyConfig
    print("detectron2_ok")
"""

# A soft-edged disc with thin strands over a sky-blue gradient; the true alpha is known.
SYNTHETIC_IMAGE = r"""
import os, sys, numpy as np, cv2
H, W, cx, cy = 600, 800, 400, 300
yy, xx = np.mgrid[0:H, 0:W]
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
disc = np.clip((150 - r) / 8 + 0.5, 0, 1).astype(np.float32)
strands = np.zeros((H, W), np.float32)
for ang in np.linspace(0, 2 * np.pi, 20, endpoint=False):
    end = (int(cx + 260 * np.cos(ang)), int(cy + 260 * np.sin(ang)))
    cv2.line(strands, (cx, cy), end, 1.0, 2, cv2.LINE_AA)
true_alpha = np.maximum(disc, strands * 0.85)
rng = np.random.default_rng(0)
fg = np.dstack([90 + 40 * np.sin(xx / 9), 60 + 30 * np.cos(yy / 7), 30 + 0 * xx]).astype(np.float32)
fg += rng.normal(0, 6, fg.shape)
bg = np.dstack([110 + 0.1 * yy, 170 + 0.05 * yy, 235 + 0 * yy]).astype(np.float32)
img = np.clip(fg * true_alpha[..., None] + bg * (1 - true_alpha[..., None]), 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(sys.argv[1], "input.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
np.save(os.path.join(sys.argv[1], "alpha_true.npy"), true_alpha)
"""

# Drives aiselect_server.py over its protocol like the plugin window does.
BACKEND_TEST = r"""
import json, os, subprocess, sys, time
import numpy as np
work, server, settings, soft = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
true = np.load(os.path.join(work, "alpha_true.npy"))
proc = subprocess.Popen([sys.executable, server, settings], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=open(os.path.join(work, "server.err"), "w"), text=True)
n = [0]
def call(**req):
    n[0] += 1; req["id"] = n[0]
    proc.stdin.write(json.dumps(req) + "\n"); proc.stdin.flush()
    reply = json.loads(proc.stdout.readline())
    if not reply["ok"]:
        raise SystemExit("backend error: " + reply["error"])
    return reply
res = {}
t0 = time.time()
call(cmd="load", image=os.path.join(work, "input.png"), preview=os.path.join(work, "p.png"),
     preview_width=400, preview_height=300)
res["load_s"] = round(time.time() - t0, 1)
out = os.path.join(work, "sel.raw")
def final(objs, **kw):
    call(cmd="render", objects=objs, final=True, out=out, **kw)
    return np.fromfile(out, np.uint8).reshape(true.shape) / 255
click = [{"sign": 1, "points": [[400, 300, 1]]}]
t0 = time.time(); call(cmd="render", objects=click, preview_out=os.path.join(work, "p1.png"))
res["click_ms"] = round(1000 * (time.time() - t0))
hard = final(click)
res["mae_click"] = float(np.abs(hard - true).mean())
res["threshold_areas"] = [call(cmd="render", objects=click, threshold=t, preview_out=os.path.join(work, "p2.png"))["area"]
                          for t in (10, 50, 90)]
# As the window sends them: each click is its own object; Shift adds, Ctrl subtracts.
added = final([{"sign": 1, "points": [[400, 300, 1]]}, {"sign": 1, "points": [[700, 80, 1]]}])
res["shift_adds_spot"] = bool(added[80, 700] > 0.5 and added[300, 400] > 0.5)
removed = final([{"sign": 1, "points": [[700, 80, 1]]}, {"sign": 1, "points": [[400, 300, 1]]},
                 {"sign": -1, "points": [[400, 250, 1]]}])
res["ctrl_removes_spot"] = bool(removed[250, 400] < 0.5 and removed[80, 700] > 0.5)
res["grow_bigger"] = bool((final(click, grow=5) > 0.5).sum() > (hard > 0.5).sum())
f = final(click, feather=10)
res["feather_soft_px"] = int(((f > 0.02) & (f < 0.98)).sum())
if soft:
    t0 = time.time()
    s = final(click, soft=True)
    res["mae_soft"] = float(np.abs(s - true).mean())
    res["soft_s"] = round(time.time() - t0, 1)
call(cmd="quit"); proc.wait(timeout=30)
print("RESULT " + json.dumps(res))
"""

# Runs inside headless GIMP: click the disc through the plugin's scripted interface, save
# the resulting selection, then check "add" keeps an existing selection.
GIMP_TEST = r"""
import gi, json
gi.require_version("Gimp", "3.0"); gi.require_version("Gegl", "0.4"); gi.require_version("Babl", "0.1")
from gi.repository import Gimp, Gegl, Gio, Babl
args = ARGS
report = {}
try:
    img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(args["input"]))
    w, h = img.get_width(), img.get_height()
    proc = Gimp.get_pdb().lookup_procedure("plug-in-ai-select")
    report["registered"] = proc is not None
    def ai_select(operation):
        cfg = proc.create_config()
        cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
        cfg.set_property("image", img)
        cfg.set_core_object_array("drawables", img.get_layers())
        cfg.set_property("objects", json.dumps([{"sign": 1, "points": [[400, 300, 1]]}]))
        cfg.set_property("operation", operation)
        return str(proc.run(cfg).index(0))
    report["status"] = ai_select("replace")
    # Before Selection.save, which selects the channel it creates.
    report["layer_still_selected"] = [l.get_name() for l in img.get_selected_layers()]
    chan = Gimp.Selection.save(img)
    fmt = "Y' u8" if Babl.get_name(chan.get_format()).startswith("Y'") else "Y u8"
    open(args["mask_out"], "wb").write(bytes(chan.get_buffer().get(
        Gegl.Rectangle.new(0, 0, w, h), 1.0, fmt, Gegl.AbyssPolicy.NONE)))
    img.select_rectangle(Gimp.ChannelOps.REPLACE, 5, 5, 40, 40)
    report["status_add"] = ai_select("add")
    report["add_keeps_old"] = Gimp.Selection.value(img, 20, 20) == 255
    report["add_has_disc"] = Gimp.Selection.value(img, 400, 300) == 255
    img.delete()
except Exception:
    import traceback
    report["exception"] = traceback.format_exc()
open(args["report"], "w").write(json.dumps(report))
"""

# Runs with GIMP's bundled Python: builds the real window off-screen, waits for the
# image analysis, clicks, Shift-clicks and Ctrl-clicks, and checks status and cursor.
WINDOW_TEST = r"""
import importlib.util, json, os, sys, time
plugin, settings, image = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location("aiselect", plugin)
ai = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai)
from gi.repository import Gtk, Gdk
ai.AISelectDialog.maximize = lambda self: None
ai.AISelectDialog.show_all = lambda self: None  # keep the real window off the screen
res = {}
backend = ai.Backend(settings)
try:
    dlg = ai.AISelectDialog(backend, image, (800, 600), os.path.join(os.path.dirname(image), "wp.png"),
                            ai.preview_size(800, 600), ai.DEFAULT_OPTIONS,
                            soft_available=ai.soft_edges_available(settings))
    box = dlg.get_child(); dlg.remove(box)
    off = Gtk.OffscreenWindow(); off.set_size_request(1400, 900); off.add(box); off.show_all()
    dlg.set_busy("Loading model and analysing image...")
    def pump(cond, timeout):
        t = time.time()
        while not cond() and time.time() - t < timeout:
            Gtk.main_iteration_do(False); time.sleep(0.005)
        return cond()
    pump(lambda: False, 0.3)
    res["sidebar_disabled_while_loading"] = not dlg.sidebar.get_sensitive()
    res["ready"] = pump(lambda: dlg.ready, 300)
    res["sidebar_enabled_after"] = dlg.sidebar.get_sensitive()
    def click(x, y, state=0):
        ev = Gdk.Event.new(Gdk.EventType.BUTTON_PRESS); b = ev.button
        ox, oy = dlg.image_offset()
        b.button = 1; b.x = ox + (x + .5) * dlg.zoom; b.y = oy + (y + .5) * dlg.zoom; b.state = state
        dlg.set_status("waiting"); dlg.on_press(dlg.area, b)
        pump(lambda: "selected" in dlg.status.get_text(), 30)
        return dlg.status.get_text()
    res["click"] = click(400, 300)
    res["shift_click"] = click(700, 80, Gdk.ModifierType.SHIFT_MASK)
    res["ctrl_click"] = click(400, 250, Gdk.ModifierType.CONTROL_MASK)
    res["objects"] = [o["sign"] for o in dlg.objects]
    e = Gdk.Event.new(Gdk.EventType.KEY_PRESS); e.key.keyval = Gdk.KEY_Shift_L; e.key.state = 0
    dlg.on_key(dlg, e.key)
    res["shift_cursor"] = dlg.area.get_window().get_cursor() is dlg.cursors.get("+")
    res["select_enabled"] = dlg.select_button.get_sensitive()
except Exception:
    import traceback
    res["exception"] = traceback.format_exc()
finally:
    backend.close()
print("RESULT " + json.dumps(res))
"""


def run_backend_test(py, work, plugin_dir, settings, soft):
    proc = subprocess.run([str(py), "-c", BACKEND_TEST, work, plugin_dir / "aiselect_server.py",
                           json.dumps(settings), "1" if soft else "0"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        fail("Backend test crashed:")
        err = Path(work, "server.err")
        tail = (proc.stdout + proc.stderr + (err.read_text(errors="replace") if err.exists() else ""))
        print(textwrap.indent("\n".join(tail.strip().splitlines()[-20:]), "      "))
        return False
    r = json.loads(line[7:])
    info(f"image analysed in {r['load_s']} s; a click updates the preview in {r['click_ms']} ms")
    checks = [
        (r["mae_click"] < 0.03, f"click selects the object (error vs true alpha {r['mae_click']:.4f})"),
        (r["threshold_areas"] == sorted(r["threshold_areas"]),
         f"higher threshold selects more {r['threshold_areas']}"),
        (r["shift_adds_spot"], "Shift+click adds the clicked object, keeping the first"),
        (r["ctrl_removes_spot"], "Ctrl+click subtracts the clicked object, keeping the rest"),
        (r["grow_bigger"], "grow enlarges the selection"),
        (r["feather_soft_px"] > 1000, f"feather softens the edge ({r['feather_soft_px']} partial pixels)"),
    ]
    if soft:
        checks.append((r["mae_soft"] < r["mae_click"],
                       f"soft edges beat the hard mask ({r['mae_soft']:.4f} vs {r['mae_click']:.4f}, "
                       f"{r['soft_s']} s incl. loading ViTMatte)"))
    passed = True
    for good, text in checks:
        (ok if good else fail)(text)
        passed = passed and good
    return passed


def run_gimp_test(py, gimp_console, work):
    args = {"input": str(Path(work, "input.png")), "mask_out": str(Path(work, "gimp_mask.raw")),
            "report": str(Path(work, "gimp_report.json"))}
    script = Path(work, "gimp_test.py")
    script.write_text(GIMP_TEST.replace("ARGS", repr(args)), encoding="utf-8")
    t0 = time.time()
    try:
        subprocess.run([str(gimp_console), "-i", "--batch-interpreter=python-fu-eval",
                        "-b", f"exec(open(r'{script}').read())", "--quit"],
                       capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        fail("Headless GIMP did not finish within 15 minutes")
        return False
    report_file = Path(args["report"])
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    if "exception" in report or not Path(args["mask_out"]).exists():
        fail(f"AI Fuzzy Select failed inside GIMP: {report or 'no report (is the plugin registered?)'}")
        return False
    score = ("import numpy as np, sys; t = np.load(sys.argv[1]); "
             "m = np.fromfile(sys.argv[2], np.uint8).reshape(t.shape) / 255; "
             "print(float(np.abs(m - t).mean()))")
    mae = float(run([py, "-c", score, Path(work, "alpha_true.npy"), args["mask_out"]], capture=True))
    passed = (mae < 0.03 and report.get("add_keeps_old") and report.get("add_has_disc")
              and bool(report.get("layer_still_selected")))
    (ok if passed else fail)(
        f"in GIMP: a click gave a selection with error {mae:.4f}; the layer stayed selected: "
        f"{bool(report.get('layer_still_selected'))}; 'add' kept the old selection: "
        f"{report.get('add_keeps_old')} and added the object: {report.get('add_has_disc')} "
        f"({time.time() - t0:.0f} s incl. GIMP start)")
    return bool(passed)


def run_window_test(gimp_console, plugin_dir, settings, work):
    gimp_python = gimp_console.parent / ("python.exe" if IS_WINDOWS else "python3")
    if not gimp_python.exists():
        info(f"Window test skipped: GIMP's Python not found at {gimp_python}")
        return True
    proc = subprocess.run([str(gimp_python), "-c", WINDOW_TEST, plugin_dir / "aiselect.py",
                           json.dumps(settings), Path(work, "input.png")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    r = json.loads(line[7:]) if line else {"exception": (proc.stdout + proc.stderr)[-2000:]}
    if "exception" in r:
        fail("The window test failed:")
        print(textwrap.indent(r["exception"].strip(), "      "))
        return False
    passed = (r["sidebar_disabled_while_loading"] and r["ready"] and r["sidebar_enabled_after"]
              and "selected" in r["click"] and r["objects"] == [1, 1, -1] and r["shift_cursor"]
              and r["select_enabled"])
    (ok if passed else fail)(
        f"window: options disabled while loading {r['sidebar_disabled_while_loading']}, ready "
        f"{r['ready']}; click/Shift/Ctrl gave objects {r['objects']}; '+' cursor with Shift "
        f"{r['shift_cursor']}; status: {r['ctrl_click']!r}")
    return bool(passed)


def run_tests(py, settings, plugin_dir, soft, expect_cuda, gimp_console):
    step("Running tests")
    try:
        out = run([py, "-c", IMPORT_TEST] + (["soft"] if soft else []), capture=True)
    except InstallError as e:
        fail(f"Import test failed: {e}")
        return False
    for line in out.strip().splitlines():
        info(line)
    if expect_cuda and "cuda_kernel_ok" not in out:
        fail("An NVIDIA GPU was detected but PyTorch cannot use it")
        return False
    ok("Packages import" + (" and CUDA works" if expect_cuda else ""))
    work = Path(tempfile.mkdtemp(prefix="ai_select_test_"))
    try:
        run([py, "-c", SYNTHETIC_IMAGE, work], capture=True)
        passed = run_backend_test(py, work, plugin_dir, settings, soft)
        if gimp_console:
            passed = run_gimp_test(py, gimp_console, work) and passed
            passed = run_window_test(gimp_console, plugin_dir, settings, work) and passed
        else:
            info("GIMP tests skipped: gimp-console not found")
        return passed
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- Main -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Install and test AI Fuzzy Select for GIMP 3.")
    parser.add_argument("--install-dir", type=Path, default=default_install_dir(),
                        help="where the environment, sources and checkpoints go (default: %(default)s)")
    parser.add_argument("--gimp-plugins-dir", type=Path, help="GIMP 3 plug-ins folder (default: auto-detected)")
    parser.add_argument("--no-soft-edges", action="store_true",
                        help="skip ViTMatte (the 'Soft edges' option); SAM2 only")
    parser.add_argument("--sam2-checkpoint", help="use this existing sam2_hiera_large.pt instead of downloading")
    parser.add_argument("--vitmatte-checkpoint", help="use this existing ViTMatte_B_DIS.pth instead of downloading")
    parser.add_argument("--cpu", action="store_true", help="force a CPU-only PyTorch")
    parser.add_argument("--torch-index", help="override the PyTorch wheel index, e.g. cu126, cu130, cpu")
    parser.add_argument("--test-only", action="store_true", help="skip installing, just run the tests")
    args = parser.parse_args()

    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        sys.exit(f"Run this with Python 3.10-3.13 (this is {platform.python_version()}).")
    if platform.architecture()[0] != "64bit":
        sys.exit("A 64-bit Python is required for PyTorch.")

    install_dir = args.install_dir.resolve()
    if IS_WINDOWS and len(str(install_dir)) > 60:
        # PyTorch's deepest file adds ~190 characters; Windows' default limit is 260.
        print(f"WARNING: {install_dir} is a long path; unless Windows long paths are enabled, "
              "installing PyTorch there can fail with 'No such file or directory'. A shorter "
              "--install-dir avoids that.")
    soft = not args.no_soft_edges
    gpu = None if args.cpu else detect_nvidia_gpu()
    if args.torch_index:
        index = args.torch_index
    elif gpu is None or sys.platform == "darwin":
        index = "cpu"
    else:
        # CUDA 13 builds dropped Maxwell/Pascal/Volta; CUDA 12.6 builds still cover sm_50+.
        index = "cu126" if gpu[1] < 7.5 else "cu130"
    plugins_dir = args.gimp_plugins_dir or find_gimp_plugins_dir()
    if plugins_dir is None:
        sys.exit("No GIMP 3 user profile found. Start GIMP 3 once, or pass --gimp-plugins-dir.")
    gimp_console = find_gimp_console()

    print("AI Fuzzy Select installer")
    info(f"Install dir : {install_dir}")
    info(f"GPU         : {f'{gpu[0]} (compute {gpu[1]})' if gpu else 'none detected'}")
    info(f"PyTorch     : {index} wheels")
    info(f"Soft edges  : {'yes (ViTMatte)' if soft else 'no'}")
    info(f"GIMP plugin : {plugins_dir / 'aiselect'}")
    info(f"GIMP tests  : {gimp_console or 'skipped (gimp-console not found)'}")
    if index == "cpu":
        print("\nNote: without an NVIDIA GPU everything runs on the CPU: analysing an image takes "
              "much longer, clicks stay reasonably quick.")

    py = venv_python(install_dir / "venv")
    src = install_dir / "src"
    try:
        if args.test_only:
            settings_file = plugins_dir / "aiselect" / "aiselect_settings.json"
            if not settings_file.exists():
                raise InstallError(f"No installation found ({settings_file} is missing)")
            settings = json.loads(settings_file.read_text(encoding="utf-8"))
            soft = bool(settings.get("vitmatteDir") or settings.get("appDir"))
            py = Path(settings["pythonPath"])
        else:
            step("Creating virtual environment")
            py = create_venv(install_dir / "venv")
            step(f"Installing PyTorch ({index}) and SAM2")
            install_sam2(py, index, src)
            step("Fetching the SAM2 checkpoint")
            sam2_ckpt = get_sam2_checkpoint(install_dir / "checkpoints" / SAM2_CHECKPOINT[0],
                                            args.sam2_checkpoint)
            settings = {"pythonPath": str(py), "sam2Checkpoint": str(sam2_ckpt)}
            if soft:
                step("Installing ViTMatte for soft edges")
                vitmatte_dir = install_soft_edges(py, src)
                vitmatte_ckpt = get_vitmatte_checkpoint(
                    py, install_dir / "checkpoints" / "ViTMatte_B_DIS.pth", args.vitmatte_checkpoint)
                settings.update({"vitmatteDir": str(vitmatte_dir), "vitmatteCheckpoint": str(vitmatte_ckpt)})
            step("Installing the GIMP plugin")
            install_plugin(plugins_dir, settings)
        passed = run_tests(py, settings, plugins_dir / "aiselect", soft, index.startswith("cu"), gimp_console)
    except InstallError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("All tests passed." if passed else "Some tests FAILED - see the output above.")
    print("  In GIMP (restart it if it's open): Select > AI Fuzzy Select...")
    print("=" * 72)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
