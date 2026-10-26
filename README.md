# gimp-ai-fuzzy-select

An AI-based fuzzy select tool for [GIMP](https://www.gimp.org/) 3. It works like GIMP's **Fuzzy Select**, but it
selects by *object* instead of by *colour*: click on something in the image and
[SAM 2](https://github.com/facebookresearch/sam2) (Meta's Segment Anything Model 2) works
out which pixels belong to it. The result is a normal GIMP selection with marching ants.

<p align="center">
  <a href="https://github.com/Esbob/gimp-ai-fuzzy-select/releases/latest/download/gimp-ai-fuzzy-select.zip"><img src="assets/download.png" width="220" alt="Download"></a><br>
  <sub>For GIMP 3 on Windows and Linux: unzip it, then double-click <code>install.bat</code> (Windows) or run <code>sh install.sh</code> (Linux) · <a href="#installation">installation steps</a> · macOS untested</sub>
</p>

<img width="1920" height="1020" alt="AI_Fuzzy_Select" src="https://github.com/user-attachments/assets/2b375a6d-3563-4bd6-a517-694790976008" />

https://github.com/user-attachments/assets/8a79728e-e93e-48f2-904d-6e7b5a6111e4

## Contents

- [What the plug-in does](#what-the-plug-in-does)
  - [Clicking and dragging](#clicking-and-dragging)
  - [Slicing an image into layers](#slicing-an-image-into-layers)
  - [Options](#options)
  - [Viewing](#viewing)
  - [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Uninstalling](#uninstalling)
- [Credits and licences](#credits-and-licences)

## What the plug-in does

**Select > AI Fuzzy Select...** opens a window with your image (what is currently visible) and a
compact options sidebar. GIMP plug-ins can't receive clicks on the canvas itself, so you
click in this window; when you press **Select**, the result becomes GIMP's selection (or, if you
choose, a new layer with a mask, a cut-out layer, a mask on the current layer, a channel, or
the current layer sliced into a layer per object).

### Clicking and dragging

| Action | Effect |
|---|---|
| Hover | Outline what a click would select (yellow; green with Shift, red with Ctrl) |
| Click | Select the object under the cursor (starts over) |
| Drag | Select the object inside the box you drag (starts over) |
| Shift+click / Shift+drag | Add the object (or part) under the cursor / inside the box |
| Ctrl+click / Ctrl+drag (or with the right button) | Subtract the object (or part) under the cursor / inside the box |
| Ctrl+Z | Undo the last click or box |

As with Fuzzy Select, each click or box picks out an object (or part) of its own, and
Shift/Ctrl add it to or subtract it from what you have so far. A box is often the most
reliable way to get a thin or oddly shaped object that a single click doesn't capture. Whether a click means the whole object
or a part of it follows the **Granularity** option: with *Auto* a click usually means the
whole object, so to remove just a piece of an object with Ctrl+click, switch Granularity
to *Part* or *Detail* first. For a box, the box itself sets the size, so Granularity
doesn't apply.

If the image already has a selection in GIMP, for example a rough lasso drawn with Free
Select, the window shows its outline in cyan and offers **Find object in GIMP's
selection**: it selects the object inside each separate area of your selection (using the
area's bounding box and the point deepest inside it), so a quick, loose lasso around each
object is enough. If one area contains several objects, click the one you mean instead.

Moving the mouse over the image outlines the object a click would select, using the
current Granularity and Threshold, so you can see the result before you click; the
**Show object under cursor** option switches this off. The selection itself is previewed
live as a tinted overlay with a dashed outline. The cursor looks
like GIMP's Fuzzy Select cursor and shows **+** while Shift is held and **−** while Ctrl
is held.

### Slicing an image into layers

With **Result: Slices**, the window cuts the current layer into a layer per object plus
one for everything else (the rest):

| Action | Effect |
|---|---|
| Click / Drag | Make the object a new slice |
| Shift+click / Shift+drag | Add the object to the last slice, e.g. a second part of it |
| Ctrl+click / Ctrl+drag | Give the object back to the rest |

Each slice is shown in its own colour. Clicking an object inside a slice splits it in two:
where objects overlap, the smaller one keeps the pixels, whichever you clicked first. So
you can click a car and then its headlight, or the shopping bags first and then the car
around them.

**Slice** puts the slices and the rest in a new layer group right above the layer, each
slice cropped to its object, and hides the original layer. Every pixel is in exactly one of
them, so together they look exactly like the original layer, and merging the group gives it
back pixel for pixel. That's why slices have hard edges: Soft edges and Feather are off
while slicing.

### Options

| Option | What it does |
|---|---|
| Model | SAM 2.1 large (best quality) or SAM 2.1 small (about 3× faster, half the GPU memory). Switching analyses the image again; your clicks are kept. If the graphics card doesn't have enough memory, AI Fuzzy Select switches to small, and then to the processor (CPU), by itself |
| Granularity | Auto, Whole object, Part or Detail: whether each click selects, adds or subtracts a whole object or a part of it (e.g. tree → crown → branch) |
| Threshold | Like Fuzzy Select's threshold: higher selects more, lower selects less (50 = SAM 2's own cut-off) |
| Fill holes | Fill holes in the selection up to this diameter in pixels, e.g. a window in a wall you want selected as solid. Only holes completely surrounded by the selection are filled, and SAM 2 already includes holes smaller than about 10 px. 0 (default) = off, keeps every hole (e.g. sky between branches) |
| Grow/shrink | Grow or shrink the selection, in pixels |
| Feather | Soften the selection edge, in pixels |
| Sharper outlines | More accurate outlines on large photos, especially for small objects: each object is looked at again on a crop around it. Takes a few seconds per object, in the background; the preview updates when it's ready |
| Soft edges (ViTMatte) | Turn fuzzy edges such as fur and hair into a partial selection with [ViTMatte](https://github.com/hustvl/ViTMatte), picking out individual hairs near the object's outline. On hard edges it mostly makes the outline smoother and more accurate. The effect is best seen in the **Mask** view. The preview shows soft edges a moment after each change; pressing **Select** computes them at full resolution |
| Result | What the main button creates: a **Selection** (default); a **New layer with mask** (a copy of the current layer with the result as its layer mask; non-destructive, so you can fix it later by painting on the mask); a **New layer with only the object** (a plain cut-out: everything outside the object is deleted); a **Mask on current layer**; a **Channel**, leaving GIMP's selection alone; or **Slices** (see [Slicing an image into layers](#slicing-an-image-into-layers)). The button's label changes to match |
| Existing selection in GIMP | Only for a Selection result, and only shown when your image in GIMP already has a selection: whether pressing **Select** replaces it, adds to it, subtracts from it or intersects with it. Without an existing selection, the result simply becomes the selection |

Your options are remembered between runs (saved when you confirm with the main button);
**Reset to defaults** at the bottom of the options puts them all back.

### Viewing

The window opens maximized with the image fitted to it. Zoom with the toolbar buttons
(−, +, Fit, 100%, 200%), GIMP's keys (**+**, **−**, **1**, **2**, **Shift+Ctrl+J**) or
Ctrl+scroll around the pointer. Pan by scrolling or by dragging with the middle mouse
button.

The **View** buttons switch between **Overlay** (the selection as a tinted overlay on the
image) and **Mask** (the selection itself, white on black, like a GIMP channel), also with
the **M** key. Mask view is the way to see soft edges and feathering: partially selected
pixels show as grey.

### How it works

A backend process (in its own Python environment) loads SAM 2 and analyses the image
once. After that each click only runs SAM 2's small mask decoder, so updates take about
0.05–0.3 seconds. Clicks are evaluated at SAM 2's working resolution (1024 px on the long
side); on large photos use **Sharper outlines** for more accurate outlines, and **Soft edges** or
**Feather** for soft ones.

The backend stays loaded between runs and remembers the analysis of the last few images,
so only the first run has to wait for the model. Measured on a GeForce GTX 1050 Ti from
2018:

| Opening the window | Wait |
|---|---|
| First time (loads the model) | about 10 seconds |
| On another image | about 2–3 seconds (analysing the image) |
| On an image analysed recently | instant |

It shuts itself down after 10 minutes without use, which frees the video memory it holds
(about 2 GB with soft edges loaded). To change that, set `"keepAliveMinutes"` in
`aiselect_settings.json` in the plug-in's folder (0 = stop when the window closes). The
installer and uninstaller stop a running backend before changing its files.

<sub>[↑ Back to contents](#contents)</sub>

## Requirements

- **GIMP 3** with Python plug-in support (included in the official Windows installer).
  Tested with GIMP 3.2.6.
- **Python 3.10–3.14, 64-bit**, to run the installer (not 3.15: PyTorch crashes on it). The plug-in's backend runs in a
  virtual environment the installer creates with this Python; GIMP's own Python is not
  used for the models.
- **An NVIDIA GPU is strongly recommended.** Tested on a GeForce GTX 1050 Ti (2018) with 4 GB of
  video memory (driver 581.57). Older GPUs (compute capability below 7.5, e.g. GTX 10xx)
  get PyTorch's CUDA 12.6 build and newer ones the CUDA 13 build, so the NVIDIA driver must
  support that CUDA version. Without an NVIDIA GPU everything runs on the CPU; this is
  untested and analysing the image will be much slower.
- **Disk space:** about 6 GB (PyTorch and other packages 4.4 GB, SAM 2.1 large and small
  1.1 GB, ViTMatte checkpoint and code 0.5 GB); about 0.5 GB less with `--no-soft-edges`.
- **Internet access** during installation, to download the packages and checkpoints.
- **Operating system:** tested on Windows 11 (GIMP 3.2.6) and Ubuntu 26.04 (GIMP 3 from
  apt), both with an NVIDIA GPU. The installer contains code for macOS, but **it has not
  been tested on macOS**.

<sub>[↑ Back to contents](#contents)</sub>

## Installation

On Windows:

1. **Install GIMP 3** from [gimp.org](https://www.gimp.org/downloads/), start it once and close it.
2. **Install Python 3.14** from [python.org](https://www.python.org/downloads/release/python-3149/)
   (the Windows installer, 64-bit; the default settings are fine). Not Python 3.15: PyTorch
   crashes on it.
3. **Download [gimp-ai-fuzzy-select.zip](https://github.com/Esbob/gimp-ai-fuzzy-select/releases/latest/download/gimp-ai-fuzzy-select.zip)**
   and unzip it anywhere (right-click > Extract All).
4. **Double-click `install.bat`** in the unzipped folder. If Windows asks whether to run it,
   choose Run (or More info > Run anyway). It downloads about 4 GB, sets everything up and
   ends with a set of tests.
5. **Start GIMP** (restart it if it's open) and use **Select > AI Fuzzy Select...**

The installer creates a virtual environment in `%LOCALAPPDATA%\gimp-ai-select`, installs
PyTorch for your GPU, SAM 2 and (for soft edges) ViTMatte, downloads the checkpoints, copies
the plug-in into GIMP's `plug-ins` folder and runs the tests. The unzipped folder isn't
needed afterwards, except for uninstalling.

On Linux and macOS (macOS untested): download and unzip the same zip, then run `sh install.sh` in
that folder in a terminal; on macOS you can instead double-click `install.command` (the first
time, right-click it and choose Open). On Ubuntu and Debian, install GIMP and Python's venv
package from apt first: `sudo apt install gimp python3-venv` (tested on Ubuntu 26.04).

With git: `git clone https://github.com/Esbob/gimp-ai-fuzzy-select`, then the same as above
in that folder, or run `python install_ai_select.py` directly.

Installer options go after `install.bat` in a Command Prompt, after `sh install.sh`, or
after `python install_ai_select.py`:

| Option | Use |
|---|---|
| `--model large` / `--model small` | Which SAM 2.1 model to start with (both are installed; switch in the window). Without it: large with an NVIDIA GPU of 3 GB or more, otherwise small |
| `--no-soft-edges` | Install SAM 2 only (no ViTMatte); the Soft edges option is greyed out |
| `--sam2-checkpoint PATH` | Use an already downloaded `sam2.1_hiera_large.pt` or `sam2.1_hiera_small.pt` |
| `--vitmatte-checkpoint PATH` | Use an already downloaded `ViTMatte_B_DIS.pth` |
| `--install-dir PATH` | Install somewhere else (keep the path short on Windows) |
| `--gimp-plugins-dir PATH` | GIMP's plug-ins folder, if it isn't found automatically |
| `--test-only` | Re-run the tests on an existing installation |

The ViTMatte checkpoint is downloaded from Google Drive, which sometimes refuses
downloads; the installer then explains how to download it by hand.

<sub>[↑ Back to contents](#contents)</sub>

## Uninstalling

Double-click `uninstall.bat` (Linux and macOS: `sh uninstall.sh`, or double-click
`uninstall.command` on macOS; or run `python uninstall_ai_select.py`). It shows what it will
remove and asks first: the plug-in from every GIMP 3 profile
(including your remembered options), the install folder with its virtual environment,
sources and downloaded checkpoints, and the plug-in's log file. Checkpoints you gave the
installer with `--sam2-checkpoint` or `--vitmatte-checkpoint` are yours and are kept.
Close GIMP first. Options go after `uninstall.bat`, `sh uninstall.sh` or
`python uninstall_ai_select.py`:

| Option | Use |
|---|---|
| `--dry-run` | Only show what would be removed |
| `--yes` | Don't ask for confirmation |
| `--keep-checkpoints` | Keep the downloaded checkpoints, e.g. for a later reinstall |
| `--install-dir PATH` | The install folder, if it isn't found from the plug-in's settings |
| `--gimp-plugins-dir PATH` | Only remove the plug-in from this plug-ins folder |

<sub>[↑ Back to contents](#contents)</sub>

## Credits and licences

Co-authored with [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5, by Anthropic).

This plug-in is MIT-licensed (see [LICENSE](LICENSE)). The installer downloads, at pinned
versions:

- [SAM 2](https://github.com/facebookresearch/sam2) and its SAM 2.1 checkpoints, by Meta (Apache 2.0)
- ViTMatte code and checkpoint from [Matte Anything](https://github.com/hustvl/Matte-Anything) (MIT), for soft edges
- [detectron2](https://github.com/facebookresearch/detectron2) by Meta (Apache 2.0), for loading ViTMatte

<sub>[↑ Back to contents](#contents)</sub>
