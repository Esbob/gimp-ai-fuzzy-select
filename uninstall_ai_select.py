#!/usr/bin/env python3
"""
Uninstaller for AI Fuzzy Select (see install_ai_select.py).

Removes:
  * the plug-in folder (plug-ins/aiselect) from every GIMP 3.x profile that has it,
    including the remembered dialog options,
  * the install folder (virtual environment, downloaded sources and checkpoints),
  * the plug-in's log files (aiselect.log and aiselect_server.log in the temp folder),
after stopping the plug-in's backend if it is still running.

Checkpoints you pointed the installer at with --sam2-checkpoint / --vitmatte-checkpoint
live outside the install folder and are left alone. It shows what it will delete and
asks first.

Usage:
  python uninstall_ai_select.py              # show the plan, ask, then remove
  python uninstall_ai_select.py --dry-run    # only show the plan
  python uninstall_ai_select.py --yes        # don't ask
  python uninstall_ai_select.py --keep-checkpoints   # keep downloaded checkpoints

Only the Python standard library is needed.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"
PLUGIN_FOLDER = "aiselect"


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


def default_install_dir():
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "gimp-ai-select"


def gimp_profiles_root():
    if IS_WINDOWS:
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "GIMP"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GIMP"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "GIMP"


def installed_plugin_dirs():
    """plug-ins/aiselect in every GIMP 3.x profile."""
    root = gimp_profiles_root()
    found = []
    for profile in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            major = int(profile.name.split(".")[0])
        except ValueError:
            continue
        candidate = profile / "plug-ins" / PLUGIN_FOLDER
        if major >= 3 and candidate.is_dir():
            found.append(candidate)
    return found


def read_settings(plugin_dir):
    try:
        return json.loads((plugin_dir / "aiselect_settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def install_dir_from_settings(settings):
    """The install folder is the parent of the virtual environment the plug-in uses."""
    python = settings.get("pythonPath")
    if not python:
        return None
    venv = Path(python).parent.parent  # venv/Scripts/python.exe or venv/bin/python
    return venv.parent if venv.name == "venv" else None


def looks_like_install_dir(path):
    """Only delete folders the installer made: a venv plus src or checkpoints, and not
    some broad location like the home folder."""
    path = Path(path).resolve()
    protected = {Path.home().resolve(), Path(path.anchor)}
    for var in ("LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "ProgramFiles"):
        if os.environ.get(var):
            protected.add(Path(os.environ[var]).resolve())
    return (path not in protected and (path / "venv").is_dir()
            and ((path / "src").is_dir() or (path / "checkpoints").is_dir()))


def size_of(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def remove_tree(path):
    """rmtree that also clears read-only flags (common in venvs on Windows). Returns the
    paths it couldn't delete, e.g. files in use by a running GIMP or backend."""
    failed = []

    def on_error(func, failed_path, exc_info):
        try:
            os.chmod(failed_path, stat.S_IWRITE)
            func(failed_path)
        except OSError:
            failed.append(failed_path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda f, p, e: on_error(f, p, None))
    else:
        shutil.rmtree(path, onerror=on_error)
    return failed


def main():
    parser = argparse.ArgumentParser(description="Uninstall AI Fuzzy Select for GIMP 3.")
    parser.add_argument("--install-dir", type=Path,
                        help="the install folder (default: read from the plug-in's settings, "
                             f"else {default_install_dir()})")
    parser.add_argument("--gimp-plugins-dir", type=Path, action="append",
                        help="a GIMP plug-ins folder to remove the plug-in from (default: every "
                             "GIMP 3 profile); can be given more than once")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="keep the downloaded model checkpoints (install-dir/checkpoints)")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be removed")
    parser.add_argument("--yes", "-y", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args()

    if args.gimp_plugins_dir:
        plugin_dirs = [d / PLUGIN_FOLDER for d in args.gimp_plugins_dir if (d / PLUGIN_FOLDER).is_dir()]
    else:
        plugin_dirs = installed_plugin_dirs()

    # Settings name the install folder and any checkpoints outside it.
    settings = next((s for s in map(read_settings, plugin_dirs) if s), {})
    install_dir = args.install_dir or install_dir_from_settings(settings) or default_install_dir()
    install_dir = Path(install_dir).resolve()

    plan = [(d, "plug-in") for d in plugin_dirs]
    keep_notes = []
    if install_dir.exists():
        if not looks_like_install_dir(install_dir):
            keep_notes.append(f"{install_dir} (doesn't look like an AI Fuzzy Select install "
                              "folder: no venv with src/checkpoints next to it)")
        elif args.keep_checkpoints and (install_dir / "checkpoints").is_dir():
            plan += [(p, "install folder") for p in sorted(install_dir.iterdir()) if p.name != "checkpoints"]
            keep_notes.append(f"{install_dir / 'checkpoints'} (--keep-checkpoints)")
        else:
            plan.append((install_dir, "install folder"))
    for log_file in (Path(tempfile.gettempdir()) / "aiselect.log", SERVER_LOG):
        if log_file.exists():
            plan.append((log_file, "log file"))
    for key in ("sam2Checkpoint", "vitmatteCheckpoint"):
        ckpt = settings.get(key)
        if ckpt and install_dir not in Path(ckpt).resolve().parents:
            keep_notes.append(f"{ckpt} (yours, passed to the installer)")

    if not plan:
        print("Nothing to remove: AI Fuzzy Select doesn't seem to be installed.")
        return 0

    print("AI Fuzzy Select uninstaller\n\nWill remove:")
    for path, what in plan:
        size = size_of(path) if path.is_dir() else path.stat().st_size
        print(f"  {path}  ({what}, {size / 1e6:,.0f} MB)" if size >= 1e6 else f"  {path}  ({what})")
    if keep_notes:
        print("\nWill keep:")
        for note in keep_notes:
            print(f"  {note}")
    if args.dry_run:
        print("\nDry run: nothing was removed.")
        return 0
    if not args.yes:
        try:
            answer = input("\nRemove these? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Cancelled; nothing was removed.")
            return 1

    if stop_backend():
        print("Stopped the running AI Fuzzy Select backend.")
    leftovers = []
    for path, _ in plan:
        if path.is_dir():
            leftovers += remove_tree(path)
        else:
            try:
                path.unlink()
            except OSError:
                leftovers.append(str(path))
    if leftovers:
        print(f"\nSome files could not be removed ({len(leftovers)}), probably because GIMP or "
              "the AI Fuzzy Select window is still open. Close GIMP and run this again. First few:")
        for p in leftovers[:5]:
            print(f"  {p}")
        return 1
    print("\nAI Fuzzy Select has been removed. Restart GIMP if it's open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
