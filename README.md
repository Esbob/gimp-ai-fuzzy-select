# gimp-ai-fuzzy-select

An AI-based fuzzy select tool for GIMP 3. It works like GIMP's **Fuzzy Select**, but it
selects by *object* instead of by *colour*: click on something in the image and
[SAM 2](https://github.com/facebookresearch/sam2) (Meta's Segment Anything Model 2) works
out which pixels belong to it. The result is a normal GIMP selection with marching ants.

## What the plug-in does

**Select > AI Fuzzy Select...** opens a window with your image (what is currently visible) and a
compact options sidebar. GIMP plug-ins can't receive clicks on the canvas itself, so you
click in this window; when you press **Select**, the result becomes GIMP's selection.

### Clicking

| Action | Effect |
|---|---|
| Click | Select the object under the cursor (starts over) |
| Shift+click | Add the object (or part) under the cursor |
| Ctrl+click (or right-click) | Subtract the object (or part) under the cursor |
| Ctrl+Z | Undo the last click |

As with Fuzzy Select, each click picks out an object (or part) of its own, and Shift/Ctrl
add it to or subtract it from what you have so far. Whether a click means the whole object
or a part of it follows the **Granularity** option: with *Auto* a click usually means the
whole object, so to remove just a piece of an object with Ctrl+click, switch Granularity
to *Part* or *Detail* first.

The selection is previewed live as a tinted overlay with a dashed outline. The cursor looks
like GIMP's Fuzzy Select cursor and shows **+** while Shift is held and **−** while Ctrl
is held.

### Options

| Option | What it does |
|---|---|
| Granularity | Auto, Whole object, Part or Detail: whether each click selects, adds or subtracts a whole object or a part of it (e.g. tree → crown → branch) |
| Threshold | Like Fuzzy Select's threshold: higher selects more, lower selects less (50 = SAM 2's own cut-off) |
| Fill holes | Fill enclosed holes up to this size in pixels; 0 keeps every hole (e.g. sky between branches) |
| Grow/shrink | Grow or shrink the selection, in pixels |
| Feather | Soften the selection edge, in pixels |
| Soft edges (ViTMatte) | Refine fuzzy edges such as hair and foliage into a partial selection with [ViTMatte](https://github.com/hustvl/ViTMatte); applied when you press Select |
| On Select | Replace, add to, subtract from or intersect with GIMP's current selection |

Your options are remembered between runs.

### Viewing

The window opens maximized with the image fitted to it. Zoom with the toolbar buttons
(−, +, Fit, 100%, 200%), GIMP's keys (**+**, **−**, **1**, **2**, **Shift+Ctrl+J**) or
Ctrl+scroll around the pointer. Pan by scrolling or by dragging with the middle mouse
button.

### How it works

When the window opens, a backend process (in its own Python environment) loads SAM 2 and
analyses the image once, which takes roughly 10–20 seconds. After that each click only runs
SAM 2's small mask decoder, so updates take about 0.05–0.3 seconds (measured on a GeForce
GTX 1050 Ti from 2018). Clicks are evaluated at SAM 2's working resolution (1024 px on the long
side); on large photos use **Soft edges** or **Feather** for fine edges.

## Requirements

- **GIMP 3** with Python plug-in support (included in the official Windows installer).
  Tested with GIMP 3.2.6.
- **Python 3.10–3.13, 64-bit**, to run the installer. The plug-in's backend runs in a
  virtual environment the installer creates with this Python; GIMP's own Python is not
  used for the models.
- **An NVIDIA GPU is strongly recommended.** Tested on a GeForce GTX 1050 Ti (2018) with 4 GB of
  video memory (driver 581.57). Older GPUs (compute capability below 7.5, e.g. GTX 10xx)
  get PyTorch's CUDA 12.6 build and newer ones the CUDA 13 build, so the NVIDIA driver must
  support that CUDA version. Without an NVIDIA GPU everything runs on the CPU; this is
  untested and analysing the image will be much slower.
- **Disk space:** about 6 GB (PyTorch and other packages 4.4 GB, SAM 2 checkpoint 0.9 GB,
  ViTMatte checkpoint and code 0.5 GB). About 0.5 GB less with `--no-soft-edges`.
- **Internet access** during installation, to download the packages and checkpoints.
- **No C++ or CUDA compiler** is needed.
- **Operating system:** tested on Windows 11 only. The installer contains code for Linux
  and macOS, but **it has not been tested on Linux or macOS**.

## Installation

```
python install_ai_select.py
```

This creates a virtual environment in `%LOCALAPPDATA%\gimp-ai-select` (on Windows),
installs PyTorch for your GPU, SAM 2 and (for soft edges) ViTMatte, downloads the
checkpoints, copies the plug-in into GIMP's `plug-ins` folder and runs a set of tests.
Then restart GIMP and use **Select > AI Fuzzy Select...**.

Useful options:

| Option | Use |
|---|---|
| `--no-soft-edges` | Install SAM 2 only (no ViTMatte); the Soft edges option is greyed out |
| `--sam2-checkpoint PATH` | Use an already downloaded `sam2_hiera_large.pt` |
| `--vitmatte-checkpoint PATH` | Use an already downloaded `ViTMatte_B_DIS.pth` |
| `--install-dir PATH` | Install somewhere else (keep the path short on Windows) |
| `--gimp-plugins-dir PATH` | GIMP's plug-ins folder, if it isn't found automatically |
| `--test-only` | Re-run the tests on an existing installation |

The ViTMatte checkpoint is downloaded from Google Drive, which sometimes refuses
downloads; the installer then explains how to download it by hand.

## Uninstalling

```
python uninstall_ai_select.py
```

This shows what it will remove and asks first: the plug-in from every GIMP 3 profile
(including your remembered options), the install folder with its virtual environment,
sources and downloaded checkpoints, and the plug-in's log file. Checkpoints you gave the
installer with `--sam2-checkpoint` or `--vitmatte-checkpoint` are yours and are kept.
Close GIMP first.

| Option | Use |
|---|---|
| `--dry-run` | Only show what would be removed |
| `--yes` | Don't ask for confirmation |
| `--keep-checkpoints` | Keep the downloaded checkpoints, e.g. for a later reinstall |
| `--install-dir PATH` | The install folder, if it isn't found from the plug-in's settings |
| `--gimp-plugins-dir PATH` | Only remove the plug-in from this plug-ins folder |

## Credits and licences

This plug-in is MIT-licensed (see [LICENSE](LICENSE)). The installer downloads, at pinned
versions:

- [SAM 2](https://github.com/facebookresearch/sam2) and its checkpoint, by Meta (Apache 2.0)
- ViTMatte code and checkpoint from [Matte Anything](https://github.com/hustvl/Matte-Anything) (MIT), for soft edges
- [detectron2](https://github.com/facebookresearch/detectron2) by Meta (Apache 2.0), for loading ViTMatte
