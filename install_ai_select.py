#!/usr/bin/env python3
"""
Installer for AI Fuzzy Select, a GIMP 3 plugin that works like Fuzzy Select but selects
by object: click an object and SAM2 selects it; Shift+click adds, Ctrl+click removes; the
result is a normal GIMP selection.

What it does:
  1. Creates a virtual environment with the Python running this script (3.10-3.14).
  2. Installs a PyTorch build matching your GPU (CPU-only if there is none).
  3. Installs SAM2 (pinned) and downloads both SAM 2.1 models, large and small. It starts
     with large on an NVIDIA GPU of 3 GB or more, otherwise small (--model chooses); if the
     image can't be analysed with it, the backend falls back to small, then to the CPU.
  4. Optionally (default on) installs ViTMatte for the "Soft edges" option: detectron2
     and the ViTMatte code from Matte Anything are used as plain source, so no C++/CUDA
     compiler is needed.
  5. Copies the plugin into GIMP 3's plug-ins folder and writes its settings.
  6. Tests the backend on a synthetic image with a known answer, then (if GIMP is found)
     the plugin inside headless GIMP and its window off-screen with GIMP's own Python.

Usage:
  python install_ai_select.py
  python install_ai_select.py --no-soft-edges          # lighter: SAM2 only
  python install_ai_select.py --model small   # start with SAM 2.1 small
  python install_ai_select.py --sam2-checkpoint sam2.1_hiera_large.pt   # reuse a download
  python install_ai_select.py --test-only

Only the Python standard library is needed to run this script.
"""

import argparse
import contextlib
import hashlib
import http.client
import json
import locale
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_SRC = HERE / "gimp_ai_select"
PLUGIN_FILES = ["aiselect.py", "aiselect_server.py"]

# --- Pinned sources (tested together) ---------------------------------------------

SAM2 = ("facebookresearch/sam2", "2b90b9f5ceec907a1c18123530e92e794ad901a4")
# The SAM 2.1 models on offer: name -> (config in the SAM2 source, checkpoint file, MB).
# On real photos large picked whole objects more often; small analyses a photo about 3x
# faster on a GPU (4x on a CPU) with half the GPU memory.
MODELS = {
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt", 900),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt", 185),
}
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{}"
# Pythons the installer and the backend's environment run on. 3.15: with PyTorch 2.14.1 the
# backend crashes (an access violation freeing a tensor made from a NumPy array in another
# thread, which Sharper outlines does); 3.13 and 3.14 with the same PyTorch and NumPy don't.
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
# Soft edges: ViTMatte's model code and config live in Matte Anything; its config system
# is detectron2's LazyConfig.
MATTE_ANYTHING = ("hustvl/Matte-Anything", "874951ac0b8d324ad9f885baf1c7de711b8dc1b0")
DETECTRON2 = ("facebookresearch/detectron2", "1e3e13bbf607b54f62205c4c33922521822fb298")
# LaMa (Apache 2.0) as TorchScript, the file IOPaint (lama-cleaner) uses; for removing objects
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
VITMATTE_GDRIVE_ID = "1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW"  # ViTMatte_B_DIS.pth (Matte Anything's README)
VITMATTE_BYTES = 386838041

TORCH_INDEX = "https://download.pytorch.org/whl/{}"
# For the setup's progress bar: the download of each PyTorch build in MB (PyTorch 2.14.1 for
# Windows; the CUDA 12.6 build bundles more), and how long the tests take in seconds (with
# an NVIDIA GPU, on the CPU), measured on a laptop with a GTX 1050 Ti
TORCH_MB = {"cu130": 2000, "cu126": 2630, "cpu": 126}
TEST_S = {"backend": (100, 45), "gimp": (55, 40), "window": (20, 20)}  # with --full-tests
QUICK_TEST_S = {"backend": (20, 25), "gimp": (15, 15)}  # (GIMP took 5-9 s)
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


class Progress:
    """The Windows setup's progress bar (--progress): lines "@@progress <percent>\t<what>\t
    <detail>" on stdout, which setup.iss reads. Each stage of the install has a weight, about
    the seconds it takes on a laptop with a 100 Mbit/s connection: downloads move with their
    bytes, other stages with the time they usually take (slowing down near the end if they
    take longer). Without --progress it does nothing."""
    MB_PER_S = 10

    def __init__(self, stages=None):
        self.stages = stages or {}  # key -> (what, weight in seconds, moves with the time)
        self.total = sum(w for _, w, _ in self.stages.values()) or 1
        self.key, self.fraction, self.detail, self.since = None, 0.0, "", 0.0
        self.lock = threading.Lock()
        self.last = None
        if self.stages:
            threading.Thread(target=self.tick, daemon=True).start()

    def start(self, key, detail=""):
        keys = list(self.stages)
        if key in keys and (self.key is None or keys.index(key) > keys.index(self.key)):  # only forward
            with self.lock:
                self.key, self.fraction, self.detail, self.since = key, 0.0, detail, time.time()
            self.emit()

    def set(self, fraction, detail=None):
        if self.key:
            with self.lock:
                self.fraction = max(self.fraction, min(fraction, 1.0))
                if detail is not None:
                    self.detail = detail
            self.emit()

    def bytes(self, done, total):
        unit, scale = ("GB", 1e9) if total >= 1e9 else ("MB", 1e6)
        self.set(done / total if total else 0.0,
                 f"{done / scale:.1f} of {total / scale:.1f} {unit}" if unit == "GB"
                 else f"{done / scale:.0f} of {total / scale:.0f} {unit}")

    def tick(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                if not self.key or not self.stages[self.key][2]:
                    continue
                # linear up to 90% at the usual time, then ever slower
                t = (time.time() - self.since) / self.stages[self.key][1]
                self.fraction = max(self.fraction, 0.9 * t if t < 1 else 0.9 + 0.09 * (1 - math.exp(1 - t)))
            self.emit()

    def emit(self, final=False):
        if not self.stages:
            return
        with self.lock:
            keys = list(self.stages)
            done = sum(self.stages[k][1] for k in keys[:keys.index(self.key)]) if self.key else 0
            pct = 100.0 if final else 100.0 * (done + self.stages[self.key][1] * self.fraction) / self.total
            line = f"@@progress {pct:.1f}\t{self.stages[self.key][0] if self.key else ''}\t{self.detail}"
            if line == self.last:
                return
            self.last = line
        print(line, flush=True)


PROGRESS = Progress()


def setup_stages(args, index, models, soft, install_dir, gimp_console):
    """The stages of this install for the setup's progress bar, in the order they run, with
    their weights (see Progress); stages with nothing to do are left out."""
    py, checkpoints = venv_python(install_dir / "venv"), install_dir / "checkpoints"
    cpu = not index.startswith("cu")
    mb = lambda n: f"{n / 1000:.1f} GB" if n >= 1000 else f"{n:.0f} MB"
    stages = {"prepare": ("Getting ready", 15 if py.exists() else 30, True)}
    if not (py.exists() and installed_torch_build(py) == index):
        size = TORCH_MB.get(index, max(TORCH_MB.values()))
        stages["torch_download"] = (f"Downloading the AI engine for {'the processor' if cpu else 'your NVIDIA graphics card'}"
                                    f" ({mb(size)})", size / Progress.MB_PER_S, False)
        stages["torch_install"] = ("Installing the AI engine", 160 if cpu else 210, True)
    stages["sam2"] = ("Installing SAM 2, the AI that finds objects", 60, True)
    for m in models:
        if not (args.sam2_checkpoint and Path(args.sam2_checkpoint).name == MODELS[m][1]) \
                and not (checkpoints / MODELS[m][1]).exists():
            stages[f"model_{m}"] = (f"Downloading the AI model SAM 2.1 {m} ({mb(MODELS[m][2])})",
                                    MODELS[m][2] / Progress.MB_PER_S, False)
    if not (checkpoints / "big-lama.pt").exists():
        stages["lama"] = ("Downloading LaMa, for removing objects (206 MB)", 206 / Progress.MB_PER_S, False)
    if soft:
        stages["soft"] = ("Installing soft edges, for hair and fur", 110, True)
        if not args.vitmatte_checkpoint and not (checkpoints / "ViTMatte_B_DIS.pth").exists():
            stages["vitmatte"] = (f"Downloading the model for soft edges ({mb(VITMATTE_BYTES / 1e6)})",
                                  VITMATTE_BYTES / 1e6 / Progress.MB_PER_S, False)
    stages["plugin"] = ("Installing the plug-in in GIMP", 1, True)
    tests = TEST_S if args.full_tests else QUICK_TEST_S
    stages["test_backend"] = ("Testing the AI", tests["backend"][cpu], True)
    if gimp_console:
        stages["test_gimp"] = ("Testing the plug-in in GIMP", tests["gimp"][cpu], True)
        if args.full_tests:
            stages["test_window"] = ("Testing the AI Fuzzy Select window", tests["window"][cpu], True)
    return stages


def run(cmd, env=None, cwd=None, capture=False, on_line=None):
    """Run a command, streaming output unless capture=True; on_line, if given, gets each line
    of the output instead (stderr included). Raises on failure."""
    printable = " ".join(str(c) for c in cmd)
    info(f"$ {printable}" if len(printable) < 300 else f"$ {printable[:300]}...")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if on_line:
        full_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen([str(c) for c in cmd], env=full_env, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            on_line(line.rstrip("\n"))
        result = subprocess.CompletedProcess(cmd, proc.wait(), "")
    else:
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


def pip_install(py, *args, env=None, download=None, install=None):
    """pip install. For the setup's progress bar its downloads move stage `download`, and
    stage `install` starts when pip starts installing."""
    if not PROGRESS.stages:
        return run([py, "-m", "pip", "install", *args], env=env)
    got = {"done": 0, "file": 0, "now": 0}  # bytes of the files before, this file's size and bytes

    def on_line(line):
        if line.startswith("Progress ") and " of " in line:  # pip's --progress-bar raw
            now, size = (int(x) for x in line[9:].split(" of "))
            if size != got["file"] or now < got["now"]:  # the next file
                got["done"] += got["file"]
                got["file"] = size
            got["now"] = now
            if download and PROGRESS.key == download:
                expected = PROGRESS.stages[download][1] * Progress.MB_PER_S * 1e6
                PROGRESS.bytes(got["done"] + now, max(expected, got["done"] + size))
            return
        print(line, flush=True)
        if line.startswith("Installing collected packages") and install:
            PROGRESS.start(install)

    if download:
        PROGRESS.start(download)
    # (no cache: it would keep another copy of PyTorch's 2 GB download)
    run([py, "-m", "pip", "install", "--progress-bar", "raw", "--no-cache-dir", *args], env=env, on_line=on_line)


@contextlib.contextmanager
def temp_on_disk(install_dir):
    """Temporary files of the install steps go to a folder in the install dir, deleted
    afterwards. pip unpacks the PyTorch packages (several GB on Linux) in the temporary folder,
    and on some systems that is in memory and too small: on Ubuntu 26.04 /tmp is a tmpfs and
    the install failed with a quota error. (Only for the install steps: the plug-in and its
    backend keep their state and log files in the normal temporary folder.)"""
    tmp = Path(install_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    names = ("TMPDIR", "TEMP", "TMP")  # Linux and macOS; Windows
    saved, saved_tempdir = {n: os.environ.get(n) for n in names}, tempfile.tempdir
    os.environ.update({n: str(tmp) for n in names})
    tempfile.tempdir = str(tmp)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        tempfile.tempdir = saved_tempdir
        shutil.rmtree(tmp, ignore_errors=True)


def download(url, dest, report=True, attempts=5):
    """Download url to dest; `report`: the download is the setup's current progress stage.
    An interrupted download is tried again, continuing where it stopped (the server closed
    the connection in the middle of a checkpoint once); so is one an earlier run left as
    .part."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "ai-select-installer"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                resumed = have and resp.status == 206  # (else the server sends it all again)
                done = have if resumed else 0
                total = int(resp.headers.get("Content-Length") or 0) + done
                last = 0.0
                with open(tmp, "ab" if resumed else "wb") as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total and time.time() - last > 0.5:
                            if PROGRESS.stages:
                                if report:
                                    PROGRESS.bytes(done, total)
                            else:
                                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)",
                                      end="", flush=True)
                            last = time.time()
            if total and done != total:
                raise ConnectionError(f"got {done} of {total} bytes")
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 416:
                raise InstallError(f"The download of {url} failed: {e}")
            if e.code == 416:  # the range asked for is past the end: start again
                tmp.unlink(missing_ok=True)
            error = e
        except (OSError, http.client.HTTPException) as e:  # (the connection was reset, a timeout)
            error = e
        if attempt == attempts:
            raise InstallError(f"The download of {url} failed: {error}. Is the computer connected to the "
                               "internet? Running the installer again continues the download.")
        print(f"\n    The download was interrupted ({error}); trying again", flush=True)
        time.sleep(5 * attempt)
    if total:
        print(("" if PROGRESS.stages else "\r") + f"    {done / 1e6:8.1f} / {total / 1e6:.1f} MB (100.0%)",
              flush=True)
    os.replace(tmp, dest)


def fetch_repo(repo, commit, dest):
    """Download a GitHub repo at a pinned commit and extract it to dest (no git needed)."""
    dest = Path(dest)
    if dest.exists():
        ok(f"Already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / f"{dest.name}.zip"
    download(f"https://github.com/{repo}/archive/{commit}.zip", zip_path, report=False)
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
    """Return (name, compute capability or None, driver version, memory in MiB or None) or
    None. Older drivers' nvidia-smi has no compute_cap field: then the capability is None,
    not "no GPU"."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    for fields in ("driver_version,memory.total,compute_cap,name", "driver_version,memory.total,name"):
        try:  # (name last: it may hold commas)
            out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30)
            if out.returncode:
                continue
            parts = out.stdout.strip().splitlines()[0].split(",", fields.count(","))
            driver, memory, name = parts[0].strip(), parts[1].strip(), parts[-1].strip()
            memory = int(float(memory)) if memory.replace(".", "").isdigit() else None
            cap = float(parts[2]) if len(parts) == 4 else None
            return name, cap, driver, memory
        except Exception:
            continue
    return None


def driver_major(version):
    try:
        return int(str(version).split(".")[0])
    except ValueError:
        return 0


def choose_torch_index(gpu):
    """PyTorch wheels for this GPU. CUDA 13 builds need compute 7.5+ (they dropped
    Maxwell/Pascal/Volta) and driver r580+; CUDA 12.6 builds cover compute 5.0+ and need
    driver r528+ (Windows) / r525+ (Linux)."""
    name, cap, driver, memory = gpu
    if cap is not None and cap >= 7.5 and driver_major(driver) >= 580:
        return "cu130"
    return "cu126"


def choose_models(asked, checkpoint, cuda, big_gpu, install_dir, small_only=False):
    """The SAM 2.1 models to install, the one to start with first. With an NVIDIA GPU both:
    if the first can't analyse an image the backend falls back to small (see cmd_load in
    aiselect_server.py). Without one, only small: on a laptop's CPU large took 9 s per image
    against small's 2.4 s, for a 900 MB download. Large is added if --model (`asked`) or
    --sam2-checkpoint asks for it, or kept if it was downloaded before. `small_only`: small,
    also with an NVIDIA GPU (and large only if it was downloaded before)."""
    first = (asked or ["large" if big_gpu and not small_only else "small"])[0]
    if cuda and not small_only:
        return [first] + [m for m in MODELS if m != first]
    given = [m for m in MODELS if checkpoint and Path(checkpoint).name == MODELS[m][1]]
    have = [m for m in MODELS if (Path(install_dir) / "checkpoints" / MODELS[m][1]).exists()]
    return list(dict.fromkeys([first] + (asked or []) + given + have + ["small"]))


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
        # where GIMP's installer says it installed GIMP (any folder), then its default folders
        import winreg
        key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\GIMP-3_is1"
        for root, view in ((winreg.HKEY_CURRENT_USER, 0), (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
                           (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY)):
            try:
                with winreg.OpenKey(root, key, 0, winreg.KEY_READ | view) as k:
                    candidates.append(Path(winreg.QueryValueEx(k, "InstallLocation")[0]) / "bin" / "gimp-console-3.exe")
            except OSError:
                pass
        for base in [os.environ.get("LOCALAPPDATA", "") + r"\Programs", os.environ.get("ProgramFiles", "")]:
            candidates.append(Path(base) / "GIMP 3" / "bin" / "gimp-console-3.exe")
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/GIMP.app/Contents/MacOS/gimp-console"))
    return next((c for c in candidates if c.exists()), None)


def is_store_python(executable=sys.executable, base_prefix=sys.base_prefix):
    """The Microsoft Store's Python: Windows redirects the files it creates under AppData to a
    private folder (AppData/Local/Packages/PythonSoftwareFoundation.Python.../LocalCache),
    so the virtual environment it made there failed to start ("failed to locate pyvenv.cfg"),
    and the plug-in it copied would be where GIMP never looks."""
    return IS_WINDOWS and any("WindowsApps" in p or "PythonSoftwareFoundation.Python" in p
                              for p in (executable, base_prefix))


def venv_python(venv_dir):
    return Path(venv_dir) / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def site_packages(py):
    return Path(run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                    capture=True).strip())


# --- Install steps ----------------------------------------------------------------

SERVER_STATE = Path(tempfile.gettempdir()) / "aiselect_server.json"
SERVER_LOG = Path(tempfile.gettempdir()) / "aiselect_server.log"


def process_start(pid):
    """When process `pid` started, as a string, or None if unknown (computed the same way
    as in aiselect_server.py, which writes its own to the state file)."""
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
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def stop_backend():
    """Stop a running AI Fuzzy Select backend (it stays alive between plugin runs and
    holds files in the virtual environment open). Returns True if one was running."""
    try:
        state = json.loads(SERVER_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    stopped = False
    try:
        with socket.create_connection(("127.0.0.1", state["port"]), timeout=3) as sock:
            f = sock.makefile("rw", encoding="utf-8", newline="\n")
            for req in ({"cmd": "hello", "token": state["token"]}, {"cmd": "quit"}):
                f.write(json.dumps(req) + "\n")
                f.flush()
                f.readline()
            stopped = True
    except (OSError, KeyError, ValueError):
        pass
    pid = state.get("pid")
    if not stopped and pid:
        # Not answering: end it, but only if that pid is still the backend that wrote the
        # state file (same start time). After a crash or reboot the pid may belong to
        # another program by now.
        started = state.get("started")
        if started and process_start(pid) == started:
            try:
                os.kill(pid, signal.SIGTERM)
                stopped = True
            except OSError:
                pass
        elif process_start(pid) is not None:
            print(f"    Note: the backend didn't answer and process {pid} can't be confirmed to be it; "
                  "left running. If it is a stuck AI Fuzzy Select backend, end it in Task Manager.")
    time.sleep(1)  # let it exit and release its files
    try:
        SERVER_STATE.unlink()
    except OSError:
        pass
    return stopped



def create_venv(venv_dir):
    py = venv_python(venv_dir)
    if py.exists():
        out = subprocess.run([str(py), "-c", "import sys, pip; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True)
        version = out.stdout.strip()
        if out.returncode or version not in SUPPORTED_PYTHONS:
            # (the Python it was made with was removed or upgraded: the checkpoints and
            # sources next to it are kept)
            info(f"The existing environment {venv_dir} doesn't work"
                 f"{f' (it uses Python {version})' if version else ''}; making a new one")
            shutil.rmtree(venv_dir)
    if py.exists():
        ok(f"Reusing existing virtual environment: {venv_dir}")
    else:
        try:
            run([sys.executable, "-m", "venv", venv_dir])
        except InstallError:
            # Seconds after the Windows setup had installed Python, ensurepip once failed (exit 1)
            # although it had installed pip, perhaps as a virus scanner held a new file: use the
            # environment if pip works in it, else make it again
            if not py.exists() or subprocess.run([str(py), "-c", "import pip"], capture_output=True).returncode:
                info("Creating the virtual environment failed; trying again")
                time.sleep(5)
                shutil.rmtree(venv_dir, ignore_errors=True)
                run([sys.executable, "-m", "venv", venv_dir])
        ok(f"Created virtual environment: {venv_dir}")
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"], capture=True)
    return py


def installed_torch_build(py):
    """The build of the venv's torch ("cpu", "cu126", ...), or None if there is none."""
    out = subprocess.run([str(py), "-c", "import torch; print(torch.version.cuda or 'cpu')"],
                         capture_output=True, text=True)
    if out.returncode:
        return None
    cuda = out.stdout.strip().splitlines()[-1]
    return "cpu" if cuda == "cpu" else "cu" + cuda.replace(".", "")


def install_sam2(py, index, src):
    # pip treats any installed torch as good enough: replace one of another build (e.g. a
    # CPU build after adding a GPU, or after --torch-index)
    have = installed_torch_build(py)
    known = index == "cpu" or index.startswith("cu")  # (a custom index can't be compared)
    again = ["--force-reinstall", "--no-deps"] if known and have and have != index else []
    if again:
        info(f"Replacing the installed PyTorch ({have}) with the {index} build")
    torch = ["torch", "torchvision", "--index-url", TORCH_INDEX.format(index)]
    if again:
        pip_install(py, *again, *torch, download="torch_download", install="torch_install")
    pip_install(py, *torch, download="torch_download", install="torch_install")
    PROGRESS.start("sam2")
    pip_install(py, "opencv-python-headless", "numpy")
    sam2 = fetch_repo(*SAM2, src / "sam2")
    # --no-build-isolation: SAM2's build requires torch and would otherwise download a
    # second copy. SAM2_BUILD_CUDA=0: its optional CUDA extension needs a compiler; SAM2
    # works without it.
    run([py, "-m", "pip", "install", "--no-build-isolation", sam2], env={"SAM2_BUILD_CUDA": "0"})


def install_soft_edges(py, src):
    """ViTMatte without compiling anything: detectron2 and Matte Anything's ViTMatte code
    are put on the path as source; Matte Anything uses only detectron2's pure-Python parts."""
    PROGRESS.start("soft")
    pip_install(py, *SOFT_EDGE_PACKAGES)
    # fairscale's old setup.py builds a broken "unknown-0.0.0" wheel in an isolated build
    # with current setuptools; build it against the venv's own setuptools instead.
    pip_install(py, "--no-build-isolation", "fairscale")
    d2 = fetch_repo(*DETECTRON2, src / "detectron2")
    zoo_configs = d2 / "detectron2" / "model_zoo" / "configs"
    if not zoo_configs.exists():  # setup.py normally links these into the package
        shutil.copytree(d2 / "configs", zoo_configs)
    vitmatte = fetch_repo(*MATTE_ANYTHING, src / "Matte-Anything")
    pth = site_packages(py) / "ai_select_sources.pth"
    # Python 3.10-3.12 read .pth files in the locale encoding, 3.13 tries UTF-8 first and
    # then the locale: the locale encoding works for all, for a path it can represent
    # (C:\Users\Søren, say).
    try:
        pth.write_bytes(f"{d2}\n".encode(locale.getpreferredencoding(False)))
    except UnicodeEncodeError:
        pth.write_text(f"{d2}\n", encoding="utf-8")
    ok(f"detectron2 registered in {pth}")
    return vitmatte


def get_sam2_checkpoint(name, dest_dir, existing):
    """The checkpoint of model `name`: `existing` if it is that model's file, else the one
    in dest_dir, downloaded if needed."""
    file = MODELS[name][1]
    if existing and Path(existing).name == file:
        existing = Path(existing).resolve()
        if not existing.exists():
            raise InstallError(f"--sam2-checkpoint not found: {existing}")
        ok(f"Using existing SAM2 checkpoint: {existing}")
        return existing
    dest = dest_dir / file
    if dest.exists():
        ok(f"Already present: {dest}")
    else:
        info(f"Downloading {file} (~{MODELS[name][2]} MB)")
        PROGRESS.start(f"model_{name}")
        download(MODEL_URL.format(file), dest)
    return dest


def get_lama_checkpoint(dest):
    if dest.exists():
        ok(f"Already present: {dest}")
        return dest
    info("Downloading big-lama.pt (~206 MB)")
    PROGRESS.start("lama")
    download(LAMA_URL, dest)
    digest = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != LAMA_SHA256:
        dest.unlink()
        raise InstallError(f"The download of {LAMA_URL} isn't the expected file (SHA-256 {digest.hexdigest()})")
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
    PROGRESS.start("vitmatte")
    downloading = threading.Event()
    downloading.set()

    def watch():  # gdown downloads to <name>*.part next to it
        while downloading.is_set():
            try:
                PROGRESS.bytes(sum(f.stat().st_size for f in dest.parent.glob(dest.name + "*.part")),
                               VITMATTE_BYTES)
            except OSError:
                pass
            time.sleep(0.5)

    if PROGRESS.stages:
        threading.Thread(target=watch, daemon=True).start()
    try:
        run([py, "-m", "gdown", VITMATTE_GDRIVE_ID, "-O", dest] + (["--quiet"] if PROGRESS.stages else []))
    except InstallError:
        raise InstallError(
            "Google Drive refused the ViTMatte download (it rate-limits). Download it from "
            f"https://drive.google.com/file/d/{VITMATTE_GDRIVE_ID}/view, then re-run with "
            "--vitmatte-checkpoint path/to/ViTMatte_B_DIS.pth (or use --no-soft-edges).")
    finally:
        downloading.clear()
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
import cv2
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
res["device"] = call(cmd="load", image=os.path.join(work, "input.png"), preview=os.path.join(work, "p.png"),
                     preview_width=400, preview_height=300)["device"]
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
res["mae_box"] = float(np.abs(final([{"sign": 1, "box": [240, 140, 560, 460]}]) - true).mean())
hover = call(cmd="hover", point=[400, 300], threshold=50, granularity="auto")
# A rough lasso (a circle 60 px too big, off-centre) standing in for the GIMP selection.
lasso = np.zeros(true.shape, np.uint8)
cv2.circle(lasso, (415, 290), 210, 255, -1)
lasso.tofile(os.path.join(work, "lasso.raw"))
guide = call(cmd="load", image=os.path.join(work, "input.png"), preview=os.path.join(work, "p.png"),
             preview_width=400, preview_height=300, guide=os.path.join(work, "lasso.raw"))["guide"]
res["mae_guide"] = float(np.abs(final([{"sign": 1, "box": g["box"], "points": [g["point"] + [1]]}
                                        for g in guide["prompts"]]) - true).mean())
# Two separate areas (the disc, and a patch of sky) must give two prompts.
cv2.rectangle(lasso, (650, 40), (760, 120), 255, -1)
lasso.tofile(os.path.join(work, "lasso.raw"))
res["guide_areas"] = len(call(cmd="load", image=os.path.join(work, "input.png"), preview=os.path.join(work, "p.png"),
                              preview_width=400, preview_height=300, guide=os.path.join(work, "lasso.raw"))["guide"]["prompts"])
res["hover_outline"] = len(hover["polygons"]) > 0 and int((hard > 0.5).sum()) == hover["area"]
res["grow_bigger"] = bool((final(click, grow=5) > 0.5).sum() > (hard > 0.5).sum())
# Slices: the disc, a box inside it (which splits it), the sky; then Ctrl gives the box back
# to the rest. Where objects overlap the smaller one keeps the pixels, in either order;
# every pixel is in one slice.
disc, sky = hard > 0.5, final([{"sign": 1, "points": [[700, 80, 1]]}]) > 0.5
inner = final([{"sign": 1, "box": [330, 230, 470, 370]}]) > 0.5
objs = [dict(click[0], slice=1), {"sign": 1, "box": [330, 230, 470, 370], "slice": 2},
        {"sign": 1, "points": [[700, 80, 1]], "slice": 3}]
def slice_render(objects):
    r = call(cmd="render", objects=objects, slices=True, final=True, out=out)
    return np.fromfile(out, np.uint8).reshape(true.shape), r
lab, r = slice_render(objs)
want = {1: disc & ~inner, 2: inner, 3: sky & ~disc & ~inner}
boxes = {s["slice"]: s["box"] for s in r["slices"]}
def bbox(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
lab_rev, _ = slice_render([dict(objs[1], slice=1), dict(objs[0], slice=2), objs[2]])  # box first
res["slices"] = {"right pixels": all(((lab == k) == m).all() for k, m in want.items()),
                 "rest": bool(((lab == 0) == ~(disc | inner | sky)).all()),
                 "split": bool(0 < inner.sum() < disc.sum() < sky.sum() and (inner & disc).sum() > 0.9 * inner.sum()),
                 "either order": bool(((lab_rev == 1) == (lab == 2)).all() and ((lab_rev == 2) == (lab == 1)).all()
                                      and ((lab_rev == 3) == (lab == 3)).all()),
                 "boxes": boxes == {k: bbox(m) for k, m in want.items()}}
lab, r = slice_render(objs + [{"sign": -1, "box": [330, 230, 470, 370]}])
res["slices"]["ctrl gives back"] = bool(((lab == 0) == (~(disc | inner | sky) | inner)).all()
                                        and [s["slice"] for s in r["slices"]] == [1, 3])
t0 = time.time()
r = call(cmd="render", objects=objs, slices=True, preview_out=os.path.join(work, "p5.png"))
res["slice_preview_ms"] = round(1000 * (time.time() - t0))
res["slices"]["preview"] = len(r["slices"]) == 3 and os.path.exists(r["preview"])
# Hovering while slicing: the slices a click would split (a detail inside the disc's slice
# splits it; the sky touches none), with their outlines
call(cmd="render", objects=[{"sign": 1, "box": [240, 140, 560, 460], "slice": 1}], slices=True,
     preview_out=os.path.join(work, "p6.png"))
inside = call(cmd="hover", point=[400, 300], granularity="detail", slices=True)
outside = call(cmd="hover", point=[700, 80], granularity="auto", slices=True)
res["slices"]["hover shows the split"] = (inside["splits"]["click"] == [1] and outside["splits"]["click"] == []
                                          and bool(inside["slice_outlines"].get("1")))
# Snap gaps: the rim of rest between the disc and the sky goes to them; the rest next to only
# one slice (the strands across the sky) stays rest, and the slices keep their own pixels.
pair = [dict(click[0], slice=1), {"sign": 1, "points": [[700, 80, 1]], "slice": 2}]
lab0, _ = slice_render(pair)
r = call(cmd="render", objects=pair, slices=True, final=True, out=out, snap=6)
lab_s = np.fromfile(out, np.uint8).reshape(true.shape)
dist = [cv2.distanceTransform((lab0 != k).astype(np.uint8), cv2.DIST_L2, 5) for k in (1, 2)]
rim = (lab0 == 0) & (dist[0] + dist[1] <= 7)
H, W = true.shape
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
edge = np.minimum(np.minimum(yy + 1, H - yy), np.minimum(xx + 1, W - xx))
lone = (lab0 == 0) & (np.minimum(dist[0], dist[1]) + np.minimum(np.maximum(dist[0], dist[1]), edge) > 7)
res["snap_rim"] = int(rim.sum())
res["snap"] = {"rim closed": bool(rim.any() and (lab_s[rim] != 0).all()),
               "lone rest kept": bool(lone.any() and (lab_s[lone] == 0).all()),
               "slices kept": bool((lab_s[lab0 != 0] == lab0[lab0 != 0]).all()),
               "count": r["snapped"] == int(((lab0 == 0) & (lab_s != 0)).sum())}
t0 = time.time()
res["mae_sharp"] = float(np.abs(final(click, sharp=True) - true).mean())
res["sharp_s"] = round(time.time() - t0, 1)
# In the window, sharper outlines are worked out in the background: a preview comes back at
# once with the work pending, a hover meanwhile is answered at once, and once ready the
# preview uses it.
box = [{"sign": 1, "box": [240, 140, 560, 460]}]
t0 = t_refine = time.time()
first = call(cmd="render", objects=box, sharp=True, preview_out=os.path.join(work, "p3.png"))
res["sharp_preview_ms"] = round(1000 * (time.time() - t0))
res["sharp_pending_first"] = first["sharp_pending"]
t0 = time.time(); call(cmd="hover", point=[400, 300]); res["sharp_hover_ms"] = round(1000 * (time.time() - t0))
t0 = time.time()
while call(cmd="sharp_status")["pending"] and time.time() - t0 < 120:
    time.sleep(0.1)
res["sharp_refine_ms"] = round(1000 * (time.time() - t_refine))
res["sharp_refined_after"] = call(cmd="render", objects=box, sharp=True,
                                  preview_out=os.path.join(work, "p4.png"))["sharp_refined"]
f = final(click, feather=10)
res["feather_soft_px"] = int(((f > 0.02) & (f < 0.98)).sum())
# Removing the disc: the patch covers it, and inside the disc it looks like the sky gradient
# behind it (known here), not like the disc. (Measured between the strands, which LaMa
# rightly continues through the middle.)
if json.loads(settings).get("lamaCheckpoint"):
    t0 = time.time()
    r = call(cmd="render", objects=click, final=True, out=out, patch_out=os.path.join(work, "patch.raw"))
    res["remove_s"] = round(time.time() - t0, 1)
    x1, y1, x2, y2 = r["patch_box"]
    patch = np.fromfile(os.path.join(work, "patch.raw"), np.uint8).reshape(y2 - y1, x2 - x1, 4)
    covered = np.zeros(true.shape, bool)
    covered[y1:y2, x1:x2] = patch[..., 3] == 255
    yy = np.mgrid[0:true.shape[0], 0:true.shape[1]][0]
    sky = np.dstack([110 + 0.1 * yy, 170 + 0.05 * yy, 235 + 0 * yy])
    photo = cv2.cvtColor(cv2.imread(os.path.join(work, "input.png")), cv2.COLOR_BGR2RGB).astype(float)
    strands = np.zeros(true.shape, np.uint8)
    for ang in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        cv2.line(strands, (400, 300), (int(400 + 260 * np.cos(ang)), int(300 + 260 * np.sin(ang))), 1, 15)
    inside = (hard > 0.5) & (strands == 0)
    filled = photo.copy()
    filled[y1:y2, x1:x2][patch[..., 3] == 255] = patch[..., :3][patch[..., 3] == 255]
    res["remove"] = [bool((covered >= (hard > 0.5)).all()), round(float(np.abs(photo - sky)[inside].mean()), 1),
                     round(float(np.abs(filled - sky)[inside].mean()), 1)]
    # The window shows the result first (a PNG); pressing Remove then reuses it, unless the
    # area around the object changed
    def removal(**kw):
        t0 = time.time()
        r = call(cmd="render", objects=click, out=out, **kw)
        return r, time.time() - t0
    shown, t_shown = removal(remove_preview=os.path.join(work, "shown.png"), remove_context=60)
    final_r, t_final = removal(final=True, patch_out=os.path.join(work, "patch2.raw"), remove_context=60)
    x1, y1, x2, y2 = final_r["patch_box"]
    final_patch = np.fromfile(os.path.join(work, "patch2.raw"), np.uint8).reshape(y2 - y1, x2 - x1, 4)
    shown_patch = cv2.cvtColor(cv2.imread(os.path.join(work, "shown.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    other, t_other = removal(final=True, patch_out=os.path.join(work, "patch3.raw"), remove_context=10)
    other_patch = np.fromfile(os.path.join(work, "patch3.raw"), np.uint8)
    res["remove_shown"] = [shown["patch_box"] == final_r["patch_box"] and bool((shown_patch == final_patch).all()),
                           round(t_shown, 2), round(t_final, 2),
                           other_patch.size != final_patch.size or bool((other_patch != final_patch.ravel()).any())]
if soft:
    t0 = time.time()
    s = final(click, soft=True)
    res["mae_soft"] = float(np.abs(s - true).mean())
    res["soft_s"] = round(time.time() - t0, 1)
# Loading falls back when the GPU is too small: SAM 2.1 small on the GPU, then on the CPU
# (a GPU of 1000 MB has no room for large, one of 150 MB for neither)
def fallback(limit_mb):
    p = subprocess.Popen([sys.executable, server, settings], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, env=dict(os.environ, AISELECT_GPU_LIMIT_MB=str(limit_mb)))
    p.stdin.write(json.dumps({"cmd": "load", "image": os.path.join(work, "input.png"), "model": "large",
                              "preview": os.path.join(work, "p.png"), "preview_width": 400,
                              "preview_height": 300}) + "\n")
    p.stdin.flush()
    r = json.loads(p.stdout.readline())
    p.stdin.close(); p.wait(timeout=60)
    return [r.get("model"), r.get("device"), bool(r.get("fallback"))]
if res["device"] == "cuda" and "small" in json.loads(settings).get("models", {}):
    res["fallback_small"] = fallback(1000)
    res["fallback_cpu"] = fallback(150)
res["models"] = {}
for name in json.loads(settings).get("models", {}):
    t0 = time.time()
    used = call(cmd="load", image=os.path.join(work, "input.png"), preview=os.path.join(work, "p.png"),
                preview_width=400, preview_height=300, model=name)["model"]
    res["models"][name] = [used, round(time.time() - t0, 1), float(np.abs(final(click) - true).mean())]
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
    def ai_select(operation, output="selection", drawables=None, image=None, objects=None):
        cfg = proc.create_config()
        cfg.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
        cfg.set_property("image", image or img)
        cfg.set_core_object_array("drawables", drawables or (image or img).get_layers())
        cfg.set_property("objects", json.dumps(objects or [{"sign": 1, "points": [[400, 300, 1]]}]))
        cfg.set_property("operation", operation)
        cfg.set_property("output", output)
        result = proc.run(cfg)
        last_error[0] = None if result.index(0) == Gimp.PDBStatusType.SUCCESS else str(result.index(1))
        return last_error[0] is None
    last_error = [None]  # the plug-in's message when the last call failed
    def value(drawable, x, y):
        f = "Y' u8" if Babl.get_name(drawable.get_format()).startswith("Y'") else "Y u8"
        return bytes(drawable.get_buffer().get(Gegl.Rectangle.new(x, y, 1, 1), 1.0, f,
                                               Gegl.AbyssPolicy.NONE))[0]
    report["status"] = str(ai_select("replace"))
    # Before Selection.save, which selects the channel it creates.
    report["layer_still_selected"] = [l.get_name() for l in img.get_selected_layers()]
    chan = Gimp.Selection.save(img)
    fmt = "Y' u8" if Babl.get_name(chan.get_format()).startswith("Y'") else "Y u8"
    open(args["mask_out"], "wb").write(bytes(chan.get_buffer().get(
        Gegl.Rectangle.new(0, 0, w, h), 1.0, fmt, Gegl.AbyssPolicy.NONE)))
    img.select_rectangle(Gimp.ChannelOps.REPLACE, 5, 5, 40, 40)
    report["status_add"] = str(ai_select("add"))
    report["add_keeps_old"] = Gimp.Selection.value(img, 20, 20) == 255
    report["add_has_disc"] = Gimp.Selection.value(img, 400, 300) == 255
    # Other results: a new layer with the result as its mask, and a channel.
    ai_select("replace", "layer")
    new = img.get_layers()[0]
    report["layer_result"] = (len(img.get_layers()) == 2 and new.get_mask() is not None
                              and value(new.get_mask(), 400, 300) == 255 and value(new.get_mask(), 5, 5) == 0)
    ai_select("replace", "cutout")
    cut = img.get_layers()[0]
    alpha = lambda x, y: bytes(cut.get_buffer().get(Gegl.Rectangle.new(x, y, 1, 1), 1.0, "R'G'B'A u8",
                                                     Gegl.AbyssPolicy.NONE))[3]
    report["cutout_result"] = (len(img.get_layers()) == 3 and cut.get_mask() is None
                               and alpha(400, 300) == 255 and alpha(5, 5) == 0)
    # Removing the object: a layer right above with the sky filled in where the disc was
    if args["remove"]:
        made = ai_select("replace", "remove")
        top = img.get_layers()[0]
        px = lambda layer, x, y: list(bytes(layer.get_buffer().get(Gegl.Rectangle.new(x, y, 1, 1), 1.0,
                                                                    "R'G'B'A u8", Gegl.AbyssPolicy.NONE)))
        _, ox, oy = top.get_offsets()
        centre = px(top, 479 - ox, 313 - oy)  # inside the disc, between two strands
        checks = {"ran": made, "named": top.get_name().endswith("(object removed)"),
                  "cropped": top.get_width() < 400 and 240 <= ox <= 260,
                  "sky where the disc was": centre[3] == 255 and centre[2] > 200 and centre[0] < 170}
        report["remove_result"] = all(checks.values()) or str(dict(checks, error=last_error[0], centre=centre))
        img.remove_layer(top)
    ai_select("replace", "channel")
    ours = [c for c in img.get_channels() if c.get_name().startswith("AI Fuzzy Select")]
    report["channel_result"] = (len(ours) == 1 and value(ours[0], 400, 300) == 255
                                and bool(img.get_selected_layers()))
    # Unusual cases. GIMP ends a plug-in that writes to a pixel-locked layer or its mask:
    # "mask" on one is refused, a cut-out of one works (and stays locked).
    base = img.get_layers()[-1]
    base.set_lock_content(True)
    alpha_at = lambda layer, x, y: bytes(layer.get_buffer().get(Gegl.Rectangle.new(x, y, 1, 1), 1.0,
                                                                "R'G'B'A u8", Gegl.AbyssPolicy.NONE))[3]
    refused = not ai_select("replace", "mask", [base])
    cut_ok = ai_select("replace", "cutout", [base])
    above = img.get_layers()[-2]  # the cut-out goes right above its layer
    locked = {"mask refused": refused, "no mask added": base.get_mask() is None, "cut-out made": cut_ok,
              "cut-out locked": above.get_lock_content(), "cut out": alpha_at(above, 5, 5) == 0}
    report["locked_layer"] = all(locked.values()) or locked
    base.set_lock_content(False)
    # The drawable a script names gets the mask (not the selected layer), and a second run
    # replaces that mask
    img.set_selected_layers([img.get_layers()[0]])
    first = ai_select("replace", "mask", [base])
    mask1 = base.get_mask() is not None and (value(base.get_mask(), 400, 300), value(base.get_mask(), 5, 5))
    second = ai_select("replace", "mask", [base])
    mask2 = base.get_mask() is not None and (value(base.get_mask(), 400, 300), value(base.get_mask(), 5, 5))
    used = {"first run": first, "mask values": mask1, "second run": second, "mask values after": mask2}
    report["drawable_used"] = (first and second and mask1 == (255, 0) and mask2 == (255, 0)) or str(used)
    # A cut-out of a layer group is a normal layer with the outside deleted
    group = Gimp.GroupLayer.new(img, "group")
    img.insert_layer(group, None, 0)
    img.insert_layer(base.copy(), group, 0)
    made = ai_select("replace", "cutout", [group])
    gcut = img.get_layers()[0]
    group_checks = {"ran": made, "not a group": not gcut.is_group(), "no mask left": gcut.get_mask() is None,
                    "outside cut away": bytes(gcut.get_buffer().get(Gegl.Rectangle.new(5, 5, 1, 1), 1.0,
                                                                    "R'G'B'A u8", Gegl.AbyssPolicy.NONE))[3] == 0}
    report["group_cutout"] = all(group_checks.values()) or str(
        dict(group_checks, error=last_error[0], gimp=Gimp.version(), top_layer=gcut.get_name()))
    # Slices: a layer group right above the layer, with a layer per slice (cropped to it)
    # and one for the rest; the layer is hidden, and the group looks exactly like it. Also
    # for a layer that sticks out of the canvas: the rest keeps what's outside. (There just
    # the disc: on the shifted image SAM2 takes the box for the whole disc.)
    def pixels(layer):
        return bytes(layer.get_buffer().get(Gegl.Rectangle.new(0, 0, layer.get_width(), layer.get_height()),
                                            1.0, "R'G'B'A u8", Gegl.AbyssPolicy.NONE))
    disc_slice = {"sign": 1, "points": [[400, 300, 1]], "slice": 1}
    inner_slice = {"sign": 1, "box": [330, 230, 470, 370], "slice": 2}
    sliced = {}
    for case, shift, objects in (("layer", None, [disc_slice, inner_slice]),
                                 ("shifted layer", (-50, -30), [disc_slice])):
        simg = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(args["input"]))
        orig = simg.get_layers()[0]
        name = orig.get_name()
        if shift:
            orig.set_offsets(*shift)
        made = ai_select("replace", "slices", [orig], image=simg, objects=objects)
        top = simg.get_layers()[0]
        kids = [k.get_name() for k in top.get_children()] if top.is_group() else []
        dup = simg.duplicate()
        merged = dup.merge_visible_layers(Gimp.MergeType.EXPAND_AS_NECESSARY)
        same = (merged.get_offsets() == orig.get_offsets() and merged.get_width() == orig.get_width()
                and merged.get_height() == orig.get_height() and pixels(merged) == pixels(orig))
        dup.delete()
        checks = {"ran": made, "group": top.is_group() and top.get_name() == f"{name} (slices)",
                  "right above": [l.get_id() for l in simg.get_layers()] == [top.get_id(), orig.get_id()],
                  "layers": kids == [f"{name} slice {n}" for n in range(len(objects), 0, -1)] + [f"{name} rest"],
                  "cropped": top.is_group() and top.get_children()[0].get_width() < orig.get_width(),
                  "original hidden": not orig.get_visible(), "looks exactly like the original": same}
        sliced[case] = all(checks.values()) or dict(checks, error=last_error[0], layers=kids)
        simg.delete()
    report["slices"] = all(v is True for v in sliced.values()) or str(sliced)
    # A wrong option is refused before anything changes (no stray channel)
    channels = len(img.get_channels())
    report["bad_option"] = not ai_select("Add") and len(img.get_channels()) == channels
    # Nothing visible: a message, not a crash
    hidden = img.duplicate()
    for layer in hidden.get_layers():
        layer.set_visible(False)
    report["all_hidden"] = not ai_select("replace", image=hidden)
    hidden.delete()
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
from gi.repository import Gtk, Gdk, GdkPixbuf
# The plugin hands the window raw 8-bit RGBA, as GIMP exports it (see export_visible)
pb = GdkPixbuf.Pixbuf.new_from_file(image).add_alpha(False, 0, 0, 0)
data, stride = pb.get_pixels(), pb.get_rowstride()
image = os.path.join(os.path.dirname(image), "input.raw")
with open(image, "wb") as f:
    f.write(b"".join(data[y * stride:y * stride + 3200] for y in range(600)))
image = {"image": image, "width": 800, "height": 600, "transparent_black": True}
ai.AISelectDialog.maximize = lambda self: None
ai.AISelectDialog.show_all = lambda self: None  # keep the real window off the screen
res = {}
backend = ai.Backend(settings)
try:
    dlg = ai.AISelectDialog(backend, image, (800, 600), os.path.join(os.path.dirname(image["image"]), "wp.png"),
                            ai.preview_size(800, 600), dict(ai.DEFAULT_OPTIONS),
                            soft_available=ai.soft_edges_available(settings),
                            models=list(settings.get("models", {})))
    box = dlg.get_child(); dlg.remove(box)
    off = Gtk.OffscreenWindow(); off.set_size_request(1400, 900); off.add(box); off.show_all()
    dlg.set_busy("Loading model and analysing image...")
    def pump(cond, timeout):
        t = time.time()
        while not cond() and time.time() - t < timeout:
            Gtk.main_iteration_do(False); time.sleep(0.005)
        return cond()
    # Checked straight away: with the backend still running from an earlier run and this
    # image already analysed, loading can finish within a fraction of a second.
    res["sidebar_disabled_while_loading"] = not dlg.sidebar.get_sensitive()
    res["ready"] = pump(lambda: dlg.ready, 300)
    res["sidebar_enabled_after"] = dlg.sidebar.get_sensitive()
    def mouse(kind, x, y, state=0):
        ev = Gdk.Event.new(kind); ox, oy = dlg.image_offset()
        e = ev.motion if kind == Gdk.EventType.MOTION_NOTIFY else ev.button
        if kind != Gdk.EventType.MOTION_NOTIFY:
            e.button = 1
        e.x = ox + (x + .5) * dlg.zoom; e.y = oy + (y + .5) * dlg.zoom; e.state = state
        handler = {Gdk.EventType.BUTTON_PRESS: dlg.on_press, Gdk.EventType.MOTION_NOTIFY: dlg.on_motion,
                   Gdk.EventType.BUTTON_RELEASE: dlg.on_release}[kind]
        handler(dlg.area, e)
    def click(x, y, state=0, drag_to=None):
        dlg.set_status("waiting")
        mouse(Gdk.EventType.BUTTON_PRESS, x, y, state)
        if drag_to:
            mouse(Gdk.EventType.MOTION_NOTIFY, *drag_to, state)
        mouse(Gdk.EventType.BUTTON_RELEASE, *(drag_to or (x, y)), state)
        pump(lambda: "selected" in dlg.status.get_text(), 30)
        return dlg.status.get_text()
    res["click"] = click(400, 300)
    res["shift_click"] = click(700, 80, Gdk.ModifierType.SHIFT_MASK)
    res["ctrl_click"] = click(400, 250, Gdk.ModifierType.CONTROL_MASK)
    res["objects"] = [o["sign"] for o in dlg.objects]
    res["box_drag"] = click(240, 140, 0, (560, 460))
    res["box_object"] = dlg.objects[-1].get("box") if dlg.objects else None
    e = Gdk.Event.new(Gdk.EventType.KEY_PRESS); e.key.keyval = Gdk.KEY_Shift_L; e.key.state = 0
    dlg.on_key(dlg, e.key)
    res["shift_cursor"] = dlg.area.get_window().get_cursor() is dlg.cursors.get("+")
    res["select_enabled"] = dlg.select_button.get_sensitive()
    # Another model from the Model dropdown: the image is analysed again, the clicks kept
    models = list(settings.get("models", {}))
    if len(models) > 1:
        before = len(dlg.objects)
        dlg.set_status("waiting")
        dlg.model.set_active_id(models[1])
        pump(lambda: dlg.ready and "selected" in dlg.status.get_text(), 300)
        res["model_switch"] = [dlg.options["model"], len(dlg.objects) == before, dlg.status.get_text()]
    # Slices: each click a new slice, Shift adds to the last one (here taking part of the
    # first), Ctrl gives back to the rest;
    # soft edges and feather are off
    dlg.output.set_active_id("slices")
    dlg.clear()
    click(400, 300)
    click(700, 80)
    click(330, 230, Gdk.ModifierType.SHIFT_MASK, (470, 370))  # a box inside the disc
    res["slice_status"] = click(20, 500, Gdk.ModifierType.CONTROL_MASK, (60, 560))  # some sky
    res["slice_numbers"] = [o.get("slice") for o in dlg.objects]
    res["slice_hard_edges"] = not dlg.feather.get_sensitive() and not dlg.soft.get_sensitive()
    # hovering while slicing also brings the slices a click would split
    dlg.hover_polygons = dlg.hover_splits = None
    dlg.hover_at([400, 300])
    pump(lambda: dlg.hover_polygons is not None, 30)
    res["slice_hover"] = dlg.hover_splits is not None and "click" in dlg.hover_splits[0]
    # Snap gaps: shown only when slicing; the status says how many pixels snapped
    visible = dlg.snap_box.get_visible()
    dlg.set_status("waiting")
    dlg.snap.set_value(6)
    pump(lambda: "selected" in dlg.status.get_text(), 30)
    snapped = "px snapped" in dlg.status.get_text()
    dlg.snap.set_value(0)
    dlg.output.set_active_id("selection")
    res["snap_row"] = [visible, snapped, not dlg.snap_box.get_visible()]
    dlg.output.set_active_id("slices")
    if settings.get("lamaCheckpoint"):  # removing an object: also hard edges
        dlg.output.set_active_id("remove")
        res["remove_hard_edges"] = (dlg.output.get_active_id() == "remove" and not dlg.feather.get_sensitive()
                                    and not dlg.soft.get_sensitive() and dlg.select_button.get_label() == "_Remove")
        # the object removed is shown in the Result view, which comes on by itself
        dlg.clear()
        cursor = lambda: dlg.area.get_window().get_cursor()
        click(400, 300)
        waiting = cursor() is not None and cursor() is dlg.cursors.get("progress")  # until it's shown
        shown = pump(lambda: dlg.patch is not None, 120)
        res["remove_shown"] = [shown, dlg.result_button.get_active(), dlg.remove_box.get_visible(),
                               "object removed" in dlg.status.get_text(), waiting,
                               cursor() is not dlg.cursors.get("progress")]
        # the Mask view and back keeps the object removed; another result stops its removal
        dlg.mask_button.set_active(True)
        kept = dlg.patch is not None
        dlg.result_button.set_active(True)
        dlg.output.set_active_id("selection")
        res["remove_switch"] = [kept, not dlg.remove_pending, dlg.remove_timer is None]
        dlg.output.set_active_id("remove")
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
        (r["mae_box"] < 0.03, f"a box around the object selects it (error vs true alpha {r['mae_box']:.4f})"),
        (r["hover_outline"], "hovering outlines exactly what a click there selects"),
        (r["mae_guide"] < 0.03, f"a rough GIMP selection finds the object inside it (error vs true alpha {r['mae_guide']:.4f})"),
        (r["guide_areas"] == 2, f"a GIMP selection with two separate areas gives one prompt each ({r['guide_areas']})"),
        (r["grow_bigger"], "grow enlarges the selection"),
        (all(r["slices"].values()),
         f"slices: a smaller object keeps its pixels in either order, Ctrl gives them back, every pixel in one "
         f"({'OK' if all(r['slices'].values()) else r['slices']}; preview {r['slice_preview_ms']} ms)"),
        (all(r["snap"].values()),
         f"snap gaps gives the rim between two slices to them and keeps other rest "
         f"({r['snap_rim']} px rim; {'OK' if all(r['snap'].values()) else r['snap']})"),
        (r["mae_sharp"] <= r["mae_click"] + 0.002,
         f"sharper outlines work (error vs true alpha {r['mae_sharp']:.4f} vs {r['mae_click']:.4f}, {r['sharp_s']} s)"),
        # A hover during the refinement must not wait for it. (Both slow down together when
        # a laptop GPU throttles from heat, and on a CPU-only install, where the refinement
        # uses every core: compared with the refinement's own time, not a fixed limit.)
        (r["sharp_pending_first"] == 1 and r["sharp_refined_after"] == 1
         and (r["sharp_hover_ms"] < 1500 or r["sharp_hover_ms"] < 0.5 * r["sharp_refine_ms"]
              or r["device"] == "cpu"),
         f"sharper outlines are worked out in the background (preview {r['sharp_preview_ms']} ms, "
         f"hover meanwhile {r['sharp_hover_ms']} ms while the refinement took {r['sharp_refine_ms']} ms, "
         f"then shown: {r['sharp_refined_after'] == 1})"),
        (r["feather_soft_px"] > 1000, f"feather softens the edge ({r['feather_soft_px']} partial pixels)"),
        *([(r["remove"][0] and r["remove"][2] < 20 < r["remove"][1],
             f"removing the object fills it with the background (covered: {r['remove'][0]}; error vs the true "
             f"background {r['remove'][1]} -> {r['remove'][2]}; {r['remove_s']} s incl. loading LaMa)"),
           (r["remove_shown"][0] and r["remove_shown"][3] and r["remove_shown"][2] < r["remove_shown"][1],
            f"the result shown in the window is what Remove makes, reused ({r['remove_shown'][1]} s to show, "
            f"{r['remove_shown'][2]} s on Remove; another area makes a new one: {r['remove_shown'][3]})")]
          if "remove" in r else []),
    ]
    if "fallback_small" in r:
        checks.append((r["fallback_small"] == ["small", "cuda", True] and r["fallback_cpu"] == ["small", "cpu", True],
                       f"too little GPU memory falls back: with 1000 MB to {r['fallback_small'][:2]}, "
                       f"with 150 MB to {r['fallback_cpu'][:2]}"))
    for name, (used, seconds, mae) in r.get("models", {}).items():
        checks.append((used == name and mae < 0.03,
                       f"model SAM 2.1 {name} loads and selects (error vs true alpha {mae:.4f}, "
                       f"switched in {seconds} s)"))
    if soft:
        checks.append((r["mae_soft"] < r["mae_click"],
                       f"soft edges beat the hard mask ({r['mae_soft']:.4f} vs {r['mae_click']:.4f}, "
                       f"{r['soft_s']} s incl. loading ViTMatte)"))
    passed = True
    for good, text in checks:
        (ok if good else fail)(text)
        passed = passed and good
    return passed


def run_gimp_test(py, gimp_console, work, remove):
    args = {"input": str(Path(work, "input.png")), "mask_out": str(Path(work, "gimp_mask.raw")),
            "report": str(Path(work, "gimp_report.json")), "remove": remove}
    script = Path(work, "gimp_test.py")
    script.write_text(GIMP_TEST.replace("ARGS", repr(args)), encoding="utf-8")
    t0 = time.time()
    try:
        subprocess.run([str(gimp_console), "-i", "--batch-interpreter=python-fu-eval",
                        "-b", f"exec(open({str(script)!r}, encoding='utf-8').read())", "--quit"],
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
              and bool(report.get("layer_still_selected")) and report.get("layer_result")
              and report.get("cutout_result") and report.get("channel_result")
              and (not remove or report.get("remove_result") is True))
    (ok if passed else fail)(
        f"in GIMP: a click gave a selection with error {mae:.4f}; the layer stayed selected: "
        f"{bool(report.get('layer_still_selected'))}; 'add' kept the old selection: "
        f"{report.get('add_keeps_old')} and added the object: {report.get('add_has_disc')}; "
        f"new layer with mask: {report.get('layer_result')}; cut-out: {report.get('cutout_result')}; "
        f"channel: {report.get('channel_result')}"
        + (f"; remove the object: {report.get('remove_result')}" if remove else "")
        + f" ({time.time() - t0:.0f} s incl. GIMP start)")
    unusual = {"slices": "slices", "locked layer": "locked_layer", "the script's drawable": "drawable_used",
               "group cut-out": "group_cutout", "a wrong option refused": "bad_option",
               "nothing visible": "all_hidden"}
    unusual_ok = all(report.get(k) is True for k in unusual.values())  # else: the details
    (ok if unusual_ok else fail)("in GIMP, unusual cases: " + "; ".join(
        f"{label}: {report.get(k)}" for label, k in unusual.items()))
    return bool(passed and unusual_ok)


def gimp_python_of(gimp_console):
    """The Python GIMP runs its plug-ins with (next to gimp-console), or None if not found."""
    gimp_python = gimp_console.parent / ("python.exe" if IS_WINDOWS else "python3")
    return gimp_python if gimp_python.exists() else None


def gimp_python_problem(gimp_console):
    """Why GIMP couldn't load the plug-in, or None. Its window draws with cairo: without
    Python's cairo bindings GIMP doesn't load the plug-in at all, so it's missing from the
    Select menu without a word. Debian's gimp package (3.0.4 on Debian 13) doesn't bring them;
    Ubuntu's desktop has them, as does GIMP for Windows. GIMP for Windows can be installed
    without its Python (the Compact install, or Custom without it): then there's no Python
    plug-in at all."""
    if IS_WINDOWS and gimp_console and not gimp_python_of(gimp_console):
        return (f"GIMP ({gimp_console.parent.parent}) was installed without its Python support, which "
                "AI Fuzzy Select needs. Run GIMP's installer again and choose Full (or Custom with "
                "Python plug-in support), then run this installer again.")
    gimp_python = gimp_console and gimp_python_of(gimp_console)
    if not gimp_python:
        return None
    check = subprocess.run([str(gimp_python), "-c", "import cairo, gi; gi.require_foreign('cairo')"],
                           capture_output=True, text=True)
    if check.returncode == 0:
        return None
    return (f"GIMP's Python ({gimp_python}) can't load cairo, which AI Fuzzy Select's window draws "
            "with; without it GIMP doesn't load the plug-in at all. Install Python's cairo bindings "
            "for GIMP (on Debian and Ubuntu: sudo apt install python3-gi-cairo), then run the "
            "installer again.")


def run_window_test(gimp_console, plugin_dir, settings, work):
    gimp_python = gimp_python_of(gimp_console)
    if not gimp_python:
        info(f"Window test skipped: GIMP's Python not found next to {gimp_console}")
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
    switch = r.get("model_switch")
    passed = (r["sidebar_disabled_while_loading"] and r["ready"] and r["sidebar_enabled_after"]
              and "selected" in r["click"] and r["objects"] == [1, 1, -1] and bool(r["box_object"]) and "1 box" in r["box_drag"] and r["shift_cursor"]
              and r["select_enabled"] and (switch is None or (switch[1] and "selected" in switch[2]))
              and r["slice_numbers"] == [1, 2, 2, None] and "2 slices" in r["slice_status"]
              and r["slice_hard_edges"] and r["slice_hover"] and all(r["snap_row"]) and r.get("remove_hard_edges", True)
              and all(r.get("remove_shown", [True])) and all(r.get("remove_switch", [True])))
    (ok if passed else fail)(
        f"window: options disabled while loading {r['sidebar_disabled_while_loading']}, ready "
        f"{r['ready']}; click/Shift/Ctrl gave objects {r['objects']}; a drag made box {r['box_object']}; '+' cursor with Shift "
        f"{r['shift_cursor']}; status: {r['ctrl_click']!r}"
        + (f"; switched to model {switch[0]}, clicks kept {switch[1]}: {switch[2]!r}" if switch else "")
        + f"; slices {r['slice_numbers']}, soft edges and feather off {r['slice_hard_edges']}: {r['slice_status']!r}"
        + f"; hover shows the slices a click would split {r['slice_hover']}"
        + f"; snap gaps shown when slicing, counted in the status, hidden otherwise {r['snap_row']}"
        + (f"; remove the object offered, with soft edges and feather off: {r['remove_hard_edges']}; "
           f"the object removed shown, in the Result view, with the area option, said in the status, busy "
           f"cursor while waiting, and not after: "
           f"{r['remove_shown']}; kept through the Mask view, stopped for another result: "
           f"{r['remove_switch']}" if "remove_hard_edges" in r else ""))
    return bool(passed)


# The quick check after an install: the backend loads the model and one click selects the disc
QUICK_TEST = r"""
import json, os, subprocess, sys, time
import numpy as np
work, server, settings = sys.argv[1], sys.argv[2], sys.argv[3]
true = np.load(os.path.join(work, "alpha_true.npy"))
proc = subprocess.Popen([sys.executable, server, settings], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=open(os.path.join(work, "server.err"), "w"), text=True)
def call(**req):
    req["id"] = 1
    proc.stdin.write(json.dumps(req) + "\n"); proc.stdin.flush()
    reply = json.loads(proc.stdout.readline())
    if not reply["ok"]:
        raise SystemExit("backend error: " + reply["error"])
    return reply
t0 = time.time()
device = call(cmd="load", image=os.path.join(work, "input.png"), preview=os.path.join(work, "p.png"),
              preview_width=400, preview_height=300)["device"]
load_s = round(time.time() - t0, 1)
out = os.path.join(work, "sel.raw")
call(cmd="render", objects=[{"sign": 1, "points": [[400, 300, 1]]}], final=True, out=out)
mae = float(np.abs(np.fromfile(out, np.uint8).reshape(true.shape) / 255 - true).mean())
call(cmd="quit"); proc.wait(timeout=30)
print("RESULT " + json.dumps({"device": device, "load_s": load_s, "mae": mae}))
"""

# ... and GIMP lists the plug-in (it doesn't when it can't load it)
GIMP_CHECK = r"""
import gi, json
gi.require_version("Gimp", "3.0")
from gi.repository import Gimp
open(REPORT, "w").write(json.dumps({"registered": Gimp.get_pdb().lookup_procedure("plug-in-ai-select") is not None}))
"""


def run_quick_test(py, work, plugin_dir, settings, expect_cuda):
    proc = subprocess.run([str(py), "-c", QUICK_TEST, work, plugin_dir / "aiselect_server.py", json.dumps(settings)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        fail("The AI didn't work:")
        err = Path(work, "server.err")
        tail = (proc.stdout + proc.stderr + (err.read_text(errors="replace") if err.exists() else ""))
        print(textwrap.indent("\n".join(tail.strip().splitlines()[-20:]), "      "))
        return False
    r = json.loads(line[7:])
    passed = r["mae"] < 0.03 and (r["device"] == "cuda" or not expect_cuda)
    (ok if passed else fail)(f"a click selects the object (error vs true alpha {r['mae']:.4f}; on the "
                             f"{'graphics card' if r['device'] == 'cuda' else 'processor'}, the image analysed "
                             f"in {r['load_s']} s)")
    return passed


def run_gimp_check(gimp_console, work):
    report = Path(work, "gimp_check.json")
    script = Path(work, "gimp_check.py")
    script.write_text(GIMP_CHECK.replace("REPORT", repr(str(report))), encoding="utf-8")
    t0 = time.time()
    try:
        subprocess.run([str(gimp_console), "-i", "--batch-interpreter=python-fu-eval",
                        "-b", f"exec(open({str(script)!r}, encoding='utf-8').read())", "--quit"],
                       capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        fail("Headless GIMP did not finish within 10 minutes")
        return False
    registered = report.exists() and json.loads(report.read_text()).get("registered")
    (ok if registered else fail)(f"GIMP {'has' if registered else 'did not load'} the plug-in "
                                 f"({time.time() - t0:.0f} s incl. GIMP start)")
    return bool(registered)


def run_tests(py, settings, plugin_dir, soft, expect_cuda, gimp_console, full=False):
    """The quick check, or (full) all the tests: every option, inside GIMP and the window."""
    step("Running tests" if full else "Checking the installation")
    PROGRESS.start("test_backend")
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
        if not full:
            passed = run_quick_test(py, work, plugin_dir, settings, expect_cuda)
            if gimp_console:
                PROGRESS.start("test_gimp")
                passed = run_gimp_check(gimp_console, work) and passed
            return passed
        passed = run_backend_test(py, work, plugin_dir, settings, soft)
        if gimp_console:
            PROGRESS.start("test_gimp")
            passed = run_gimp_test(py, gimp_console, work, bool(settings.get("lamaCheckpoint"))) and passed
            PROGRESS.start("test_window")
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
    parser.add_argument("--model", action="append", choices=list(MODELS),
                        help="SAM 2.1 model to start with (with an NVIDIA GPU both are installed; switch in "
                             "the window; without one only small, unless this asks for large). "
                             "Default: large with an NVIDIA GPU of 3 GB or more, otherwise small")
    parser.add_argument("--small-only", action="store_true",
                        help="install only SAM 2.1 small, also with an NVIDIA GPU (a 900 MB smaller download)")
    parser.add_argument("--sam2-checkpoint", help="use this existing sam2.1_hiera_large.pt or sam2.1_hiera_small.pt "
                                                  "instead of downloading it")
    parser.add_argument("--vitmatte-checkpoint", help="use this existing ViTMatte_B_DIS.pth instead of downloading")
    parser.add_argument("--cpu", action="store_true", help="force a CPU-only PyTorch")
    parser.add_argument("--torch-index", help="override the PyTorch wheel index, e.g. cu126, cu130, cpu")
    parser.add_argument("--test-only", action="store_true", help="skip installing, just run the tests")
    parser.add_argument("--full-tests", action="store_true",
                        help="run all the tests (every option, inside GIMP and its window; a few minutes) "
                             "instead of the quick check that the AI and GIMP work")
    parser.add_argument("--progress", action="store_true",
                        help="report progress for the Windows setup (lines starting with @@ on stdout)")
    args = parser.parse_args()

    if "%d.%d" % sys.version_info[:2] not in SUPPORTED_PYTHONS:
        too_new = sys.version_info[:2] > (3, 14)
        sys.exit(f"This is Python {platform.python_version()}; AI Fuzzy Select needs Python 3.10-3.14."
                 + (" (On Python 3.15, PyTorch crashes the backend: tested with PyTorch 2.14.1.)"
                    if too_new else "")
                 + "\nInstall Python 3.14 from https://www.python.org/downloads/ and run this with it, "
                   "e.g.  py -3.14 install_ai_select.py")
    if platform.architecture()[0] != "64bit":
        sys.exit("A 64-bit Python is required for PyTorch.")
    if is_store_python():
        sys.exit("This is the Microsoft Store version of Python, which can't install AI Fuzzy Select: "
                 "Windows keeps the files it creates under AppData in a private folder, where GIMP "
                 "doesn't see them.\nInstall Python 3.14 from https://www.python.org/downloads/ "
                 "and run the installer with it, e.g.  py -3.14 install_ai_select.py (the Windows setup "
                 "installs one by itself). The Store Python can stay installed.")

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
        index = choose_torch_index(gpu)
    # SAM 2.1 large needs about 1.5 GB of GPU memory, more with soft edges: on a 2 GB GPU it
    # ran out (the desktop uses some too), small needs about 0.6 GB.
    big_gpu = index.startswith("cu") and gpu is not None and (gpu[3] or 0) >= 3000
    if args.small_only and "large" in (args.model or []):
        parser.error("--small-only and --model large don't go together")
    models = choose_models(args.model, args.sam2_checkpoint, index.startswith("cu"), big_gpu, install_dir,
                           args.small_only)
    if args.sam2_checkpoint and Path(args.sam2_checkpoint).name not in [MODELS[m][1] for m in models]:
        sys.exit(f"--sam2-checkpoint must be one of {', '.join(MODELS[m][1] for m in models)} "
                 "(the file name says which model it is)")
    plugins_dir = args.gimp_plugins_dir or find_gimp_plugins_dir()
    if plugins_dir is None:
        sys.exit("No GIMP 3 user profile found. Start GIMP 3 once, or pass --gimp-plugins-dir.")
    gimp_console = find_gimp_console()
    problem = gimp_python_problem(gimp_console)
    if problem:
        sys.exit(problem)

    print("AI Fuzzy Select installer")
    info(f"Install dir : {install_dir}")
    # (no quotes inside the nested f-string: Python 3.10 and 3.11 can't parse that)
    gpu_text = f"{gpu[0]} ({gpu[3] or '?'} MiB, compute {gpu[1] or 'unknown'}, driver {gpu[2]})" if gpu else "none detected"
    info(f"GPU         : {gpu_text}")
    info(f"PyTorch     : {index} wheels")
    info(f"SAM2 models : SAM 2.1 {' and '.join(models)} (starts with {models[0]})")
    info(f"Soft edges  : {'yes (ViTMatte)' if soft else 'no'}")
    info(f"GIMP plugin : {plugins_dir / 'aiselect'}")
    info(f"GIMP tests  : {gimp_console or 'skipped (gimp-console not found)'}")
    if gpu and index.startswith("cu") and driver_major(gpu[2]) < 528:
        print(f"\nWARNING: NVIDIA driver {gpu[2]} is older than PyTorch's GPU builds need (528 or "
              "newer). Update the driver from nvidia.com, or the GPU can't be used.")
    if index == "cpu":
        print("\nNote: without an NVIDIA GPU everything runs on the CPU: analysing an image takes "
              "much longer, clicks stay reasonably quick.")

    py = venv_python(install_dir / "venv")
    src = install_dir / "src"
    if args.progress:
        global PROGRESS
        PROGRESS = Progress(setup_stages(args, index, models, soft, install_dir, gimp_console))
        PROGRESS.start("prepare")
    try:
        if args.test_only:
            settings_file = plugins_dir / "aiselect" / "aiselect_settings.json"
            if not settings_file.exists():
                raise InstallError(f"No installation found ({settings_file} is missing)")
            settings = json.loads(settings_file.read_text(encoding="utf-8"))
            soft = bool(settings.get("vitmatteDir") or settings.get("appDir"))
            py = Path(settings["pythonPath"])
        else:
            if stop_backend():
                info("Stopped the running AI Fuzzy Select backend (it is restarted on next use)")
            with temp_on_disk(install_dir):
                step("Creating virtual environment")
                py = create_venv(install_dir / "venv")
                step(f"Installing PyTorch ({index}) and SAM2")
                install_sam2(py, index, src)
                step("Fetching the SAM2 checkpoints")
                checkpoints = install_dir / "checkpoints"
                installed = {m: get_sam2_checkpoint(m, checkpoints, args.sam2_checkpoint) for m in models}
                settings = {"pythonPath": str(py), "model": models[0],
                            "models": {m: {"config": MODELS[m][0], "checkpoint": str(path)}
                                       for m, path in installed.items()}}
                step("Fetching LaMa for removing objects")
                try:
                    settings["lamaCheckpoint"] = str(get_lama_checkpoint(checkpoints / "big-lama.pt"))
                except (InstallError, OSError) as e:  # (optional: everything else still works)
                    print(f"\nWARNING: LaMa could not be downloaded ({e}). Everything else is installed; "
                          "only \"Remove the object\" is missing. Run the installer again later to add it.")
                if soft:
                    step("Installing ViTMatte for soft edges")
                    vitmatte_dir = install_soft_edges(py, src)
                    vitmatte_ckpt = get_vitmatte_checkpoint(
                        py, install_dir / "checkpoints" / "ViTMatte_B_DIS.pth", args.vitmatte_checkpoint)
                    settings.update({"vitmatteDir": str(vitmatte_dir), "vitmatteCheckpoint": str(vitmatte_ckpt)})
                step("Installing the GIMP plugin")
                PROGRESS.start("plugin")
                install_plugin(plugins_dir, settings)
        passed = run_tests(py, settings, plugins_dir / "aiselect", soft, index.startswith("cu"), gimp_console,
                           args.full_tests)
    except InstallError as e:
        print(f"\nERROR: {e}")
        if args.progress:
            print(f"@@error {e}", flush=True)
        sys.exit(1)
    except OSError as e:  # a download that failed, a full disk
        if not args.progress:
            raise
        print(f"@@error {e}", flush=True)
        sys.exit(1)

    if args.progress:
        PROGRESS.emit(final=True)
        print(f"@@result {'passed' if passed else 'failed'}", flush=True)
        print(f"@@size {sum(f.stat().st_size for f in install_dir.rglob('*') if f.is_file()) // 1024}", flush=True)
    print("\n" + "=" * 72)
    print(("All tests passed." if args.full_tests else "Everything works.") if passed
          else "Some tests FAILED - see the output above.")
    print("  In GIMP (restart it if it's open): Select > AI Fuzzy Select...")
    print("=" * 72)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    if "--progress" in sys.argv:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # setup.iss reads UTF-8
        print(f"@@pid {os.getpid()}", flush=True)  # to stop the install (with its children)
        try:
            main()
        except SystemExit as e:
            if isinstance(e.code, str):  # a reason not to install
                print("@@error " + e.code.replace("\n", " "), flush=True)
            raise
    else:
        main()
