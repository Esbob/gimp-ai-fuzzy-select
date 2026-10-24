#!/usr/bin/env python3
"""
GIMP 3 plugin: Select > AI Fuzzy Select...

An AI "fuzzy select": instead of growing a selection over similar colours, it selects
the object you click on, using SAM2. GIMP plugins can't receive clicks on the canvas, so
the clicking happens in the plugin's own window, with a live preview:

    hover           outline what a click would select
    click / drag    select the object under the cursor / inside the dragged box
    Shift+...       add the object (or part) under the cursor / inside the box
    Ctrl+...        subtract the object (or part) under the cursor / inside the box
    Ctrl+Z          undo the last click     Ctrl+scroll: zoom

Whether a click means a whole object or a part of it follows the Granularity option.

"Select" turns the result into a normal GIMP selection (replacing, adding to,
subtracting from or intersecting the current one), with optional grow/shrink,
feathering and ViTMatte soft edges.

The models run in the plugin's own Python environment (made by install_ai_select.py)
through aiselect_server.py, which stays alive while the window is open: SAM2 encodes the image once, then each click only
runs its small decoder.
"""

import copy
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Babl", "0.1")
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Babl, Gdk, GdkPixbuf, Gegl, Gimp, GimpUi, Gio, GLib, GObject, Gtk

import cairo

PROC_NAME = "plug-in-ai-select"
PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
SETTINGS_FILE = os.path.join(PLUGIN_DIR, "aiselect_settings.json")
OPTIONS_FILE = os.path.join(PLUGIN_DIR, "aiselect_last_options.json")
SERVER = os.path.join(PLUGIN_DIR, "aiselect_server.py")
PREVIEW_MAX = 2560  # long side of the selection overlay the backend renders

DEFAULT_OPTIONS = {
    "output": "selection", "hover": True, "granularity": "auto", "threshold": 50, "fill_holes": 0, "grow": 0, "feather": 0,
    "soft": False, "edge": 4, "operation": "replace",
}
RENDER_OPTIONS = ("threshold", "granularity", "fill_holes", "grow", "feather", "soft", "edge")
# Result choices: dropdown text and the label of the dialog's main button.
OUTPUTS = {
    "selection": ("Selection", "_Select"),
    "layer": ("New layer with mask", "_Create layer"),
    "cutout": ("New layer with only the object", "_Create cut-out"),
    "mask": ("Mask on current layer", "_Add mask"),
    "channel": ("Channel", "_Save channel"),
}
OPERATIONS = {
    "replace": Gimp.ChannelOps.REPLACE, "add": Gimp.ChannelOps.ADD,
    "subtract": Gimp.ChannelOps.SUBTRACT, "intersect": Gimp.ChannelOps.INTERSECT,
}


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# --- Backend process ----------------------------------------------------------------


SERVER_STATE = os.path.join(tempfile.gettempdir(), "aiselect_server.json")
SERVER_LOG = os.path.join(tempfile.gettempdir(), "aiselect_server.log")


def server_version():
    with open(SERVER, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


class Backend:
    """Talks to aiselect_server.py one request at a time on a worker thread.

    The server stays alive between plugin runs (for keepAliveMinutes, default 10), so SAM2
    is loaded only once: the first run starts it, later runs reconnect via the port and
    token in SERVER_STATE. A server from older plugin files, or other settings, is
    replaced. Connecting happens on the worker thread, so the window appears at once.

    Interactive requests can be 'coalesced' by group: a newer request replaces a queued,
    unsent one of the same group (e.g. "preview"), so dragging a slider doesn't build up a
    backlog, while fast previews and slower soft-edge previews don't cancel each other."""

    def __init__(self, settings):
        self.settings = settings
        self.sock = self.rfile = self.wfile = None
        self.queue = []
        self.cond = threading.Condition()
        self.next_id = 0
        threading.Thread(target=self._worker, daemon=True).start()

    # --- connecting ---

    def _connect(self):
        state = load_json(SERVER_STATE, None)
        if state and self._try(state):
            if state.get("version") == server_version() and state.get("settings") == self.settings:
                return
            self._send({"cmd": "quit"})  # outdated code or other settings: replace it
            self._disconnect()
            time.sleep(0.5)
        self._start()

    def _try(self, state):
        try:
            sock = socket.create_connection(("127.0.0.1", state["port"]), timeout=3)
            sock.settimeout(None)
            self.sock = sock
            self.rfile = sock.makefile("r", encoding="utf-8", newline="\n")
            self.wfile = sock.makefile("w", encoding="utf-8", newline="\n")
            reply = self._send({"cmd": "hello", "token": state["token"]})
            if reply and reply.get("ok"):
                return True
        except (OSError, KeyError, ValueError):
            pass
        self._disconnect()
        return False

    def _start(self):
        token = secrets.token_hex(16)
        env = dict(os.environ, AISELECT_TOKEN=token, AISELECT_STATE=SERVER_STATE,
                   AISELECT_LOG=SERVER_LOG,
                   AISELECT_KEEPALIVE=str(60 * float(self.settings.get("keepAliveMinutes", 10))))
        kwargs = {"start_new_session": True}
        if os.name == "nt":  # no console window; not tied to this plugin process
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen(
            [self.settings["pythonPath"], SERVER, "--listen", json.dumps(self.settings)], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        t0 = time.time()
        while time.time() - t0 < 120:
            state = load_json(SERVER_STATE, None)
            if state and state.get("token") == token and self._try(state):
                return
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        raise RuntimeError("The AI Fuzzy Select backend didn't start.\n\n" + log_tail())

    def _disconnect(self):
        for f in (self.rfile, self.wfile, self.sock):
            try:
                if f:
                    f.close()
            except OSError:
                pass
        self.sock = self.rfile = self.wfile = None

    def _send(self, req):
        self.wfile.write(json.dumps(req) + "\n")
        self.wfile.flush()
        line = self.rfile.readline()
        return json.loads(line) if line else None

    # --- requests ---

    def request(self, req, callback, coalesce=False, main_loop=True):
        with self.cond:
            if coalesce:
                self.queue = [q for q in self.queue if q[2] != coalesce]
            self.next_id += 1
            req["id"] = self.next_id
            self.queue.append((req, callback, coalesce, main_loop))
            self.cond.notify()

    def call(self, req, timeout=600):
        """Blocking request (used after the window has closed)."""
        done = threading.Event()
        box = {}

        def store(reply):
            box["reply"] = reply
            done.set()

        self.request(req, store, main_loop=False)
        t0 = time.time()
        while not done.wait(0.1):
            Gimp.progress_pulse()
            if time.time() - t0 > timeout:
                return {"ok": False, "error": "The AI Fuzzy Select backend timed out."}
        return box["reply"]

    def _worker(self):
        while True:
            with self.cond:
                while not self.queue:
                    self.cond.wait()
                req, callback, _, main_loop = self.queue.pop(0)
            reply = None
            try:
                if self.sock is None and req.get("cmd") not in ("bye", "quit"):
                    self._connect()
                if self.sock is not None:
                    reply = self._send(req)
                elif req.get("cmd") in ("bye", "quit"):
                    reply = {"ok": True}
            except RuntimeError as e:
                reply = {"ok": False, "error": str(e)}
            except (OSError, ValueError):
                reply = None
            if reply is None:
                self._disconnect()
                reply = {"ok": False, "error": "The AI Fuzzy Select backend stopped.\n\n" + log_tail()}
            if main_loop:
                GLib.idle_add(callback, reply)
            else:
                callback(reply)
            if req.get("cmd") in ("bye", "quit"):
                self._disconnect()
                return

    def close(self):
        """End this connection; the server stays loaded for the next run (unless
        keepAliveMinutes is 0)."""
        stop = float(self.settings.get("keepAliveMinutes", 10)) <= 0
        done = threading.Event()
        self.request({"cmd": "quit" if stop else "bye"}, lambda r: done.set(), main_loop=False)
        done.wait(10)


def log_tail(lines=15):
    try:
        with open(SERVER_LOG, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:]).strip()
    except OSError:
        return ""


# --- Window -------------------------------------------------------------------------


SIDEBAR_CSS = b"""
.aiselect-sidebar, .aiselect-sidebar * { font-size: 9pt; }
.aiselect-sidebar button { padding: 2px 6px; min-height: 0; }
.aiselect-sidebar spinbutton entry { min-height: 0; padding-top: 1px; padding-bottom: 1px; }
.aiselect-sidebar spinbutton button { padding: 0 4px; }
.aiselect-sidebar combobox button { padding-top: 1px; padding-bottom: 1px; }
.aiselect-sidebar .heading { font-weight: bold; margin-top: 6px; }
.aiselect-toolbar button { padding: 1px 6px; min-height: 0; }
.aiselect-toolbar label { font-size: 9pt; }
.aiselect-busy { background-color: rgba(20, 20, 20, 0.75); border-radius: 10px; padding: 24px 32px; }
.aiselect-busy label { color: white; font-size: 13pt; }
.aiselect-busy spinner { color: white; }
"""

ZOOM_STEPS = [0.03125, 0.0625, 0.125, 0.1667, 0.25, 0.3333, 0.5, 0.6667, 1, 1.5, 2, 3, 4, 5.5, 8, 11, 16]


class AISelectDialog(Gtk.Dialog):
    """The click-to-select window: a compact options sidebar on the left (like GIMP's
    dock) and the image, as large as possible, on the right. The image is drawn at full
    resolution; the selection overlay comes from the backend at preview resolution.
    `zoom` is screen pixels per image pixel, as in GIMP."""

    def __init__(self, backend, image_png, image_size, preview_png, preview_size, options,
                 soft_available=True, has_selection=True, guide_raw=None):
        super().__init__(title="AI Fuzzy Select", modal=True)
        self.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        self.select_button = self.add_button("_Select", Gtk.ResponseType.OK)
        self.select_button.set_sensitive(False)

        self.backend = backend
        self.image_size = image_size
        self.preview_size = preview_size
        self.preview_png = preview_png
        self.options = dict(options)
        self.soft_available = soft_available
        self.has_selection = has_selection  # the image already has a selection to combine with
        if not soft_available:
            self.options["soft"] = False
        self.objects = []  # the backend's prompt: [{"sign": 1/-1, "points": [[x, y, label]]}]
        self.history = []
        self.ready = False
        self.busy = True  # set_busy() below; on_draw may run first
        self.zoom = 1.0
        self.fit_mode = True  # keep fitting the image to the view until the user zooms
        self.base = surface_from_pixbuf(GdkPixbuf.Pixbuf.new_from_file(image_png))
        self.base_scaled = None  # base pre-scaled for the current zoom, when zoomed out
        self.overlay = None
        self.preview_count = 0
        self.version = 0  # bumped on every change; older preview results are discarded
        self.soft_timer = None
        self.pan_start = None
        self.drag = None  # a left/right button press in progress: click or box
        self.guide = None  # prompt for the object inside GIMP's selection, and its outline
        # Hover preview: the outline a click under the mouse would select.
        self.hover_target = None  # image point under the mouse, or None
        self.hover_polygons = None  # outline for the last hover reply
        self.hover_busy = False  # a hover request is in flight
        self.cursors = {}
        self.modifiers = 0  # Shift/Ctrl currently held, for the cursor's +/- sign

        self.size_to_screen()
        self.build_ui()
        self.connect("key-press-event", self.on_key)
        self.connect("key-release-event", self.on_key_release)
        self.set_busy("Loading model and analysing image...")
        load = {"cmd": "load", "image": image_png, "preview": preview_png,
                "preview_width": preview_size[0], "preview_height": preview_size[1]}
        if guide_raw:
            load["guide"] = guide_raw  # GIMP's current selection, to start from
        self.backend.request(load, self.on_loaded)

    # --- layout ---

    def size_to_screen(self):
        """Open maximized. The un-maximized size (what Windows restores to when the window
        is dragged) leaves room for the title bar and taskbar: at the full work-area size,
        the title bar pushed Cancel/Select below the bottom of the screen."""
        # A normal window, not a dialog: Windows gives dialogs no maximize button and may
        # ignore maximizing them.
        self.set_type_hint(Gdk.WindowTypeHint.NORMAL)
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0) if display else None
        if monitor is not None:
            area = monitor.get_workarea()
            self.set_default_size(int(area.width * 0.9), int(area.height * 0.85))
        else:
            self.set_default_size(1400, 850)
        self.set_position(Gtk.WindowPosition.CENTER)
        # Maximize once the window is on screen; requested earlier, it isn't always honoured.
        self.connect("map", lambda w: GLib.idle_add(w.maximize))

    def build_ui(self):
        provider = Gtk.CssProvider()
        provider.load_from_data(SIDEBAR_CSS)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), provider,
                                                 Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self.get_content_area().set_border_width(0)
        paned = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.get_content_area().pack_start(paned, True, True, 0)

        # Left: options, scrollable in case the screen is short.
        side_scroll = Gtk.ScrolledWindow()
        side_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        side_scroll.set_size_request(240, -1)
        # The rows' widgets expand within the sidebar; without this, GTK propagates that
        # up and the sidebar takes half the window from the image.
        side_scroll.set_hexpand(False)
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        side.set_border_width(6)
        side.get_style_context().add_class("aiselect-sidebar")
        side_scroll.add(side)
        self.sidebar = side
        paned.pack_start(side_scroll, False, False, 0)
        paned.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 0)
        self.build_sidebar(side)

        # Right: zoom toolbar over the image.
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        paned.pack_start(right, True, True, 0)
        right.pack_start(self.build_toolbar(), False, False, 0)
        self.area = Gtk.DrawingArea()
        self.area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK
                             | Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.ENTER_NOTIFY_MASK
                             | Gdk.EventMask.LEAVE_NOTIFY_MASK | Gdk.EventMask.SCROLL_MASK)
        self.area.connect("draw", self.on_draw)
        self.area.connect("button-press-event", self.on_press)
        self.area.connect("button-release-event", self.on_release)
        self.area.connect("motion-notify-event", self.on_motion)
        self.area.connect("scroll-event", self.on_scroll)
        self.area.connect("enter-notify-event", lambda w, e: self.update_cursor(e.state))
        self.area.connect("leave-notify-event", lambda w, e: self.clear_hover())
        self.area.connect("realize", lambda w: self.update_cursor())
        self.scroller = Gtk.ScrolledWindow()
        self.scroller.add(self.area)
        self.scroller.connect("size-allocate", self.on_view_resized)
        view = Gtk.Overlay()
        view.add(self.scroller)
        right.pack_start(view, True, True, 0)

        # Busy indicator in the middle of the image: a large spinner and a message.
        self.busy_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        self.busy_box.get_style_context().add_class("aiselect-busy")
        self.busy_box.set_halign(Gtk.Align.CENTER)
        self.busy_box.set_valign(Gtk.Align.CENTER)
        # Drawn by hand: GTK 3's Gtk.Spinner draws its icon small whatever its size.
        self.spinner = Gtk.DrawingArea()
        self.spinner.set_size_request(96, 96)
        self.spinner.set_halign(Gtk.Align.CENTER)
        self.spinner.connect("draw", self.draw_spinner)
        self.spin_angle = 0.0
        self.busy_box.pack_start(self.spinner, False, False, 0)
        self.busy_label = Gtk.Label()
        self.busy_label.set_justify(Gtk.Justification.CENTER)
        self.busy_label.set_line_wrap(True)
        self.busy_label.set_max_width_chars(60)
        self.busy_box.pack_start(self.busy_label, False, False, 0)
        view.add_overlay(self.busy_box)
        self.show_all()

    def build_sidebar(self, side):
        def heading(text):
            label = Gtk.Label(label=text, xalign=0)
            label.get_style_context().add_class("heading")
            side.pack_start(label, False, False, 0)

        help_label = Gtk.Label(xalign=0)
        help_label.set_line_wrap(True)
        help_label.set_markup(
            "<b>Click</b> an object to select it\n"
            "<b>Drag</b> a box around an object\n"
            "<b>Shift</b> add · <b>Ctrl</b> remove (click or drag)\n"
            "<b>Ctrl+Z</b> undo\n"
            "<b>Ctrl+scroll</b> or <b>+ − 1</b> zoom\n"
            "<b>Middle-drag</b> or scroll to pan")
        side.pack_start(help_label, False, False, 0)

        heading("Clicks")
        self.hover = Gtk.CheckButton(label="Show object under cursor")
        self.hover.set_active(self.options["hover"])
        self.hover.set_tooltip_text("While you move the mouse, outline what a click would select "
                                    "(yellow; green with Shift, red with Ctrl)")
        self.hover.connect("toggled", self.on_hover_toggled)
        side.pack_start(self.hover, False, False, 0)
        buttons = Gtk.Box(spacing=4)
        for text, handler in [("Undo", lambda b: self.undo()), ("Clear", lambda b: self.clear())]:
            button = Gtk.Button(label=text)
            button.connect("clicked", handler)
            buttons.pack_start(button, True, True, 0)
        side.pack_start(buttons, False, False, 2)
        self.guide_button = Gtk.Button(label="Find object in GIMP's selection")
        self.guide_button.set_tooltip_text(
            "Your selection in GIMP (shown as a thin cyan outline) is used as a rough guide: "
            "the object inside each separate area of it is selected, starting over like a "
            "click. A rough lasso is enough; if one area contains several objects, click the "
            "one you mean instead.")
        self.guide_button.connect("clicked", lambda b: self.use_guide())
        self.guide_button.set_no_show_all(True)  # shown once the image is loaded
        side.pack_start(self.guide_button, False, False, 2)

        heading("Selection")
        grid = Gtk.Grid(column_spacing=6, row_spacing=3)
        side.pack_start(grid, False, False, 0)
        row = [0]

        def add_row(label, widget, tooltip=None):
            text = Gtk.Label(label=label, xalign=0)
            grid.attach(text, 0, row[0], 1, 1)
            widget.set_hexpand(True)
            grid.attach(widget, 1, row[0], 1, 1)
            if tooltip:
                text.set_tooltip_text(tooltip)
                widget.set_tooltip_text(tooltip)
            row[0] += 1

        def spin(lo, hi, key):
            widget = Gtk.SpinButton.new_with_range(lo, hi, 1)
            widget.set_value(self.options[key])
            widget.set_width_chars(4)
            widget.connect("value-changed", lambda w: self.option_changed(key, w.get_value_as_int()))
            return widget

        self.granularity = Gtk.ComboBoxText()
        for key, text in [("auto", "Auto"), ("whole", "Whole object"), ("part", "Part"),
                          ("detail", "Detail")]:
            self.granularity.append(key, text)
        self.granularity.set_active_id(self.options["granularity"])
        self.granularity.connect("changed", lambda w: self.option_changed("granularity", w.get_active_id()))
        add_row("Granularity", self.granularity,
                "Whether each click selects, adds or subtracts a whole object or a part of it: "
                "SAM2's best guess (Auto), the whole object, a part or a detail")

        self.threshold = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        self.threshold.set_value(self.options["threshold"])
        self.threshold.set_value_pos(Gtk.PositionType.RIGHT)
        self.threshold.connect("value-changed", lambda w: self.option_changed("threshold", w.get_value()))
        add_row("Threshold", self.threshold, "Higher selects more, lower selects less (50 = SAM2's own cut-off)")

        self.fill_holes = spin(0, 500, "fill_holes")
        add_row("Fill holes (px)", self.fill_holes,
                "Fills holes in the selection up to this diameter in pixels, e.g. a window in "
                "a wall you want selected as solid. Only holes completely surrounded by the "
                "selection are filled; gaps open to the outside are not holes. SAM2 already "
                "includes holes smaller than about 10 px by itself. 0 = off, keeps every hole "
                "(e.g. sky between branches).")
        self.grow = spin(-100, 100, "grow")
        add_row("Grow/shrink (px)", self.grow, "Grow (positive) or shrink (negative) the selection")
        self.feather = spin(0, 200, "feather")
        add_row("Feather (px)", self.feather, "Soften the selection edge by this radius")

        self.soft = Gtk.CheckButton(label="Soft edges (ViTMatte)")
        self.soft.set_active(self.options["soft"])
        self.soft.set_tooltip_text(
            "Turns fuzzy edges such as fur and hair into a partial selection, picking out "
            "individual hairs near the object's outline. On hard edges it mostly makes the "
            "outline smoother and more accurate.\n\n"
            "The effect is best seen in the Mask view (View: Mask above the image, or the M "
            "key), where partially selected pixels show as grey. The preview shows soft edges "
            "a moment after each change; pressing Select computes them at full resolution.")
        self.soft.connect("toggled", lambda w: self.option_changed("soft", w.get_active()))
        if not self.soft_available:
            self.soft.set_sensitive(False)
            self.soft.set_tooltip_text("Not installed (install_ai_select.py --no-soft-edges)")
        grid.attach(self.soft, 0, row[0], 2, 1)
        row[0] += 1

        heading("Result")
        self.output = Gtk.ComboBoxText()
        for key, (text, _) in OUTPUTS.items():
            self.output.append(key, text)
        self.output.set_active_id(self.options["output"])
        self.output.set_tooltip_text(
            "What the button at the bottom right creates: a selection; a copy of the current "
            "layer with the result as its layer mask (non-destructive: paint on the mask to fix "
            "it later); a copy with everything outside the object deleted (a plain cut-out); "
            "the result as the current layer's mask; or a channel, leaving GIMP's selection alone")
        self.output.connect("changed", lambda w: self.on_output_changed(w.get_active_id()))
        side.pack_start(self.output, False, False, 0)

        # How to combine with GIMP's existing selection: only meaningful for a selection
        # result, and only if there is a selection.
        self.existing_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        if self.has_selection:
            label = Gtk.Label(label="Existing selection in GIMP", xalign=0)
            label.get_style_context().add_class("heading")
            self.existing_box.pack_start(label, False, False, 0)
            self.operation = Gtk.ComboBoxText()
            for key, text in [("replace", "Replace it"), ("add", "Add to it"),
                              ("subtract", "Subtract from it"), ("intersect", "Intersect with it")]:
                self.operation.append(key, text)
            self.operation.set_active_id(self.options["operation"])
            self.operation.set_tooltip_text(
                "Your image in GIMP already has a selection. When you press the Select button, the AI "
                "selection replaces it, is added to it, is subtracted from it, or keeps only "
                "where the two overlap")
            self.operation.connect("changed", lambda w: self.options.update(operation=w.get_active_id()))
            self.existing_box.pack_start(self.operation, False, False, 0)
        self.existing_box.set_no_show_all(True)
        for child in self.existing_box.get_children():
            child.show()
        side.pack_start(self.existing_box, False, False, 0)

        reset = Gtk.Button(label="Reset to defaults")
        reset.set_tooltip_text("Put all options back to their defaults (your clicks are kept). "
                               "Saved when you confirm with the main button.")
        reset.connect("clicked", lambda b: self.reset_options())
        side.pack_start(reset, False, False, 8)
        GLib.idle_add(lambda: self.on_output_changed(self.options["output"]) and False)

        self.status = Gtk.Label(xalign=0)
        self.status.set_line_wrap(True)
        self.status.set_max_width_chars(30)
        side.pack_end(self.status, False, False, 4)

    def build_toolbar(self):
        bar = Gtk.Box(spacing=4)
        bar.set_border_width(3)
        bar.get_style_context().add_class("aiselect-toolbar")
        for text, tip, handler in [
            ("−", "Zoom out (−)", lambda b: self.zoom_step(-1)),
            ("+", "Zoom in (+)", lambda b: self.zoom_step(1)),
            ("Fit", "Fit image in window (Shift+Ctrl+J)", lambda b: self.zoom_fit()),
            ("100%", "Actual pixels (1)", lambda b: self.set_zoom(1.0)),
            ("200%", "Zoom 2:1 (2)", lambda b: self.set_zoom(2.0)),
        ]:
            button = Gtk.Button(label=text)
            button.set_tooltip_text(tip)
            button.set_focus_on_click(False)
            button.connect("clicked", handler)
            bar.pack_start(button, False, False, 0)
        self.zoom_label = Gtk.Label(label="")
        self.zoom_label.set_width_chars(7)
        bar.pack_start(self.zoom_label, False, False, 6)

        # Overlay: tinted selection over the image. Mask: the selection itself, white on
        # black, where soft edges and feathering show as grey.
        bar.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)
        bar.pack_start(Gtk.Label(label="View:"), False, False, 2)
        self.overlay_button = Gtk.RadioButton.new_with_label_from_widget(None, "Overlay")
        self.overlay_button.set_tooltip_text("The selection as a tinted overlay on the image (M toggles)")
        self.mask_button = Gtk.RadioButton.new_with_label_from_widget(self.overlay_button, "Mask")
        self.mask_button.set_tooltip_text("The selection itself, white on black: soft edges and "
                                          "feathering show as grey (M toggles)")
        for button in (self.overlay_button, self.mask_button):
            button.set_mode(False)  # look like toggle buttons, not radio dots
            button.set_focus_on_click(False)
            bar.pack_start(button, False, False, 0)
        self.mask_button.connect("toggled", lambda b: self.render())
        size = Gtk.Label(label=f"{self.image_size[0]} × {self.image_size[1]} px")
        size.set_opacity(0.7)
        bar.pack_end(size, False, False, 6)
        return bar

    def set_status(self, text):
        self.status.set_text(text)

    def reset_options(self):
        """Every option back to DEFAULT_OPTIONS, through the widgets so the preview follows."""
        d = DEFAULT_OPTIONS
        self.hover.set_active(d["hover"])
        self.granularity.set_active_id(d["granularity"])
        self.threshold.set_value(d["threshold"])
        self.fill_holes.set_value(d["fill_holes"])
        self.grow.set_value(d["grow"])
        self.feather.set_value(d["feather"])
        if self.soft_available:
            self.soft.set_active(d["soft"])
        self.output.set_active_id(d["output"])
        if self.has_selection:
            self.operation.set_active_id(d["operation"])
        self.options.update(d, soft=d["soft"] and self.soft_available)
        self.set_status("Options reset to their defaults.")

    def on_output_changed(self, output):
        self.options["output"] = output
        self.existing_box.set_visible(output == "selection" and self.has_selection)
        self.select_button.set_label(OUTPUTS[output][1])

    def set_busy(self, message, error=False):
        """Busy: grey the image, disable the sidebar, show a spinner and `message` in the
        middle (with elapsed seconds). message=None ends it; error=True keeps the message
        but stops the spinner."""
        self.busy = message is not None and not error
        self.sidebar.set_sensitive(message is None)
        self.busy_message = message
        self.busy_started = time.time()
        if message is None:
            self.busy_box.hide()
        else:
            self.busy_label.set_text(message)
            self.busy_box.show_all()
            if error:
                self.spinner.hide()
            else:
                GLib.timeout_add(40, self.busy_tick)
        self.update_cursor()
        self.area.queue_draw()

    def busy_tick(self):
        if not self.busy:
            return False
        self.spin_angle = (self.spin_angle + 0.25) % 6.2832
        self.spinner.queue_draw()
        seconds = int(time.time() - self.busy_started)
        text = f"{self.busy_message}\n{seconds} s"
        if self.busy_label.get_text() != text:
            self.busy_label.set_text(text)
        return True

    def draw_spinner(self, widget, cr):
        size = min(widget.get_allocated_width(), widget.get_allocated_height())
        c, r = size / 2, size / 2 - 6
        cr.set_line_width(8)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_source_rgba(1, 1, 1, 0.2)
        cr.arc(c, c, r, 0, 6.2832)
        cr.stroke()
        cr.set_source_rgba(1, 1, 1, 0.95)
        cr.arc(c, c, r, self.spin_angle, self.spin_angle + 1.8)
        cr.stroke()
        return False

    # --- zoom & drawing ---

    def image_offset(self):
        """Where the image's top-left sits in the drawing area (centred when smaller)."""
        alloc = self.area.get_allocation()
        w, h = self.image_size[0] * self.zoom, self.image_size[1] * self.zoom
        return max(0.0, (alloc.width - w) / 2), max(0.0, (alloc.height - h) / 2)

    def set_zoom(self, zoom, anchor=None, fit=False):
        """Zoom keeping the image point under `anchor` (viewport coords, default the centre)
        where it is."""
        self.fit_mode = fit
        hadj, vadj = self.scroller.get_hadjustment(), self.scroller.get_vadjustment()
        if anchor is None:
            anchor = (hadj.get_page_size() / 2, vadj.get_page_size() / 2)
        ox, oy = self.image_offset()
        ix = (hadj.get_value() + anchor[0] - ox) / self.zoom
        iy = (vadj.get_value() + anchor[1] - oy) / self.zoom
        self.zoom = min(max(zoom, ZOOM_STEPS[0]), ZOOM_STEPS[-1])
        self.base_scaled = None
        w, h = self.image_size
        self.area.set_size_request(max(1, int(w * self.zoom)), max(1, int(h * self.zoom)))
        self.zoom_label.set_text(f"{self.zoom * 100:.0f}%")

        def restore():
            nox, noy = self.image_offset()
            hadj.set_value(ix * self.zoom + nox - anchor[0])
            vadj.set_value(iy * self.zoom + noy - anchor[1])
            return False

        GLib.idle_add(restore)
        self.area.queue_draw()

    def zoom_step(self, direction, anchor=None):
        if direction > 0:
            zoom = next((z for z in ZOOM_STEPS if z > self.zoom * 1.001), ZOOM_STEPS[-1])
        else:
            zoom = next((z for z in reversed(ZOOM_STEPS) if z < self.zoom / 1.001), ZOOM_STEPS[0])
        self.set_zoom(zoom, anchor)

    def zoom_fit(self):
        alloc = self.scroller.get_allocation()
        if alloc.width > 1 and alloc.height > 1:
            w, h = self.image_size
            self.set_zoom(min((alloc.width - 2) / w, (alloc.height - 2) / h), fit=True)
        return False

    def on_view_resized(self, widget, allocation):
        if self.fit_mode:
            GLib.idle_add(self.zoom_fit)

    def on_draw(self, widget, cr):
        ox, oy = self.image_offset()
        cr.translate(ox, oy)
        if self.zoom < 1:
            # Pre-scale once per zoom level: smooth and fast to repaint.
            if self.base_scaled is None:
                self.base_scaled = scaled_surface(self.base, self.zoom)
            cr.set_source_surface(self.base_scaled, 0, 0)
            cr.paint()
        else:
            cr.save()
            cr.scale(self.zoom, self.zoom)
            cr.set_source_surface(self.base, 0, 0)
            cr.get_source().set_filter(cairo.FILTER_NEAREST)  # show real pixels when zoomed in
            cr.paint()
            cr.restore()
        if self.busy:
            # Grey the image out while it's being analysed: desaturate, then darken.
            cr.rectangle(0, 0, self.image_size[0] * self.zoom, self.image_size[1] * self.zoom)
            cr.save()
            cr.clip_preserve()
            cr.set_operator(cairo.OPERATOR_HSL_SATURATION)
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.fill_preserve()
            cr.set_operator(cairo.OPERATOR_OVER)
            cr.set_source_rgba(0, 0, 0, 0.45)
            cr.fill()
            cr.restore()
        if self.overlay is not None:
            cr.save()
            s = self.zoom * self.image_size[0] / self.preview_size[0]
            cr.scale(s, s)
            cr.set_source_surface(self.overlay, 0, 0)
            cr.get_source().set_filter(cairo.FILTER_GOOD if s < 1 else cairo.FILTER_BILINEAR)
            cr.paint()
            cr.restore()
        if self.guide:  # GIMP's selection, for reference: thin cyan dashes
            for rgb, width, dash in (((0, 0, 0), 2.5, []), ((0.2, 0.85, 1.0), 1.2, [6, 4])):
                for poly in self.guide["polygons"]:
                    cr.move_to(poly[0][0] * self.zoom, poly[0][1] * self.zoom)
                    for x, y in poly[1:]:
                        cr.line_to(x * self.zoom, y * self.zoom)
                    cr.close_path()
                cr.set_source_rgba(*rgb, 0.8)
                cr.set_line_width(width)
                cr.set_dash(dash)
                cr.stroke()
            cr.set_dash([])
        # Boxes and click markers: green = include, red = exclude.
        for obj in self.objects:
            if obj.get("box"):
                x1, y1, x2, y2 = (v * self.zoom for v in obj["box"])
                colour = (0.1, 0.75, 0.2) if obj["sign"] > 0 else (0.9, 0.15, 0.15)
                for rgb, width in (((1, 1, 1), 3), (colour, 1.5)):
                    cr.rectangle(x1, y1, x2 - x1, y2 - y1)
                    cr.set_source_rgb(*rgb)
                    cr.set_line_width(width)
                    cr.stroke()
            for x, y, label in obj.get("points", []):
                include = label == 1 and obj["sign"] > 0
                px, py = (x + 0.5) * self.zoom, (y + 0.5) * self.zoom
                cr.arc(px, py, 6, 0, 6.2832)
                cr.set_source_rgb(1, 1, 1)
                cr.fill()
                cr.arc(px, py, 4, 0, 6.2832)
                cr.set_source_rgb(0.1, 0.75, 0.2) if include else cr.set_source_rgb(0.9, 0.15, 0.15)
                cr.fill()
        ctrl_with_nothing = self.modifiers & Gdk.ModifierType.CONTROL_MASK and not self.objects
        if self.hover_polygons and not self.drag and self.pan_start is None and not ctrl_with_nothing:
            # What a click would select: yellow, or green/red when Shift/Ctrl would add/remove.
            # (Ctrl with nothing selected yet does nothing, so nothing is shown.)
            if self.modifiers & Gdk.ModifierType.CONTROL_MASK:
                colour = (0.95, 0.2, 0.2)
            elif self.modifiers & Gdk.ModifierType.SHIFT_MASK:
                colour = (0.3, 0.9, 0.3)
            else:
                colour = (1.0, 0.85, 0.1)
            for rgb, width in (((0, 0, 0), 3.5), (colour, 1.8)):
                for poly in self.hover_polygons:
                    cr.move_to(poly[0][0] * self.zoom, poly[0][1] * self.zoom)
                    for x, y in poly[1:]:
                        cr.line_to(x * self.zoom, y * self.zoom)
                    cr.close_path()
                cr.set_source_rgb(*rgb)
                cr.set_line_width(width)
                cr.stroke()
        if self.drag and self.drag["dragging"]:  # the box being dragged, as marching ants
            (x1, y1), (x2, y2) = self.drag["start"], self.drag["end"]
            cr.rectangle(min(x1, x2) * self.zoom, min(y1, y2) * self.zoom,
                         abs(x2 - x1) * self.zoom, abs(y2 - y1) * self.zoom)
            cr.set_line_width(1.5)
            cr.set_source_rgb(0, 0, 0)
            cr.stroke_preserve()
            cr.set_dash([5, 5])
            cr.set_source_rgb(1, 1, 1)
            cr.stroke()
            cr.set_dash([])
        return False

    def on_scroll(self, widget, event):
        if not event.state & Gdk.ModifierType.CONTROL_MASK:
            return False  # let the scrolled window pan
        ok, dx, dy = event.get_scroll_deltas()
        up = event.direction == Gdk.ScrollDirection.UP or (ok and dy < 0)
        hadj, vadj = self.scroller.get_hadjustment(), self.scroller.get_vadjustment()
        self.zoom_step(1 if up else -1, (event.x - hadj.get_value(), event.y - vadj.get_value()))
        return True

    # --- mouse ---

    def image_point(self, wx, wy):
        ox, oy = self.image_offset()
        x, y = (wx - ox) / self.zoom, (wy - oy) / self.zoom
        if not (0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]):
            return None
        return [int(x), int(y)]

    def image_point_clamped(self, wx, wy):
        """Like image_point, but kept inside the image (for box corners)."""
        ox, oy = self.image_offset()
        return (min(max((wx - ox) / self.zoom, 0.0), float(self.image_size[0])),
                min(max((wy - oy) / self.zoom, 0.0), float(self.image_size[1])))

    def on_press(self, widget, event):
        if event.type != Gdk.EventType.BUTTON_PRESS:
            return False
        if event.button == 2:  # middle-drag pans
            hadj, vadj = self.scroller.get_hadjustment(), self.scroller.get_vadjustment()
            self.pan_start = (event.x_root, event.y_root, hadj.get_value(), vadj.get_value())
            self.update_cursor()
            return True
        if event.button not in (1, 3):
            return False
        if not self.ready:
            self.set_status("Still analysing the image, one moment...")
            return True
        self.clear_hover()
        # Decided on release: a press that doesn't move is a click, a drag is a box.
        self.drag = {
            "button": event.button, "widget": (event.x, event.y), "dragging": False,
            "point": self.image_point(event.x, event.y),
            "start": self.image_point_clamped(event.x, event.y),
            "end": self.image_point_clamped(event.x, event.y),
            "shift": bool(event.state & Gdk.ModifierType.SHIFT_MASK),
            "ctrl": bool(event.state & Gdk.ModifierType.CONTROL_MASK) or event.button == 3,
        }
        return True

    def on_motion(self, widget, event):
        if self.drag is not None:
            wx, wy = self.drag["widget"]
            if abs(event.x - wx) > 4 or abs(event.y - wy) > 4:
                self.drag["dragging"] = True
            if self.drag["dragging"]:
                self.drag["end"] = self.image_point_clamped(event.x, event.y)
                self.area.queue_draw()
            return True
        if self.pan_start is None:
            self.update_cursor(event.state)
            self.hover_at(self.image_point(event.x, event.y))
            return False
        x0, y0, h0, v0 = self.pan_start
        self.scroller.get_hadjustment().set_value(h0 - (event.x_root - x0))
        self.scroller.get_vadjustment().set_value(v0 - (event.y_root - y0))
        return True

    def on_release(self, widget, event):
        if event.button == 2:
            self.pan_start = None
            self.update_cursor(event.state)
            return True
        drag, self.drag = self.drag, None
        if drag is None or event.button != drag["button"]:
            return False
        if drag["dragging"]:
            (x1, y1), (x2, y2) = drag["start"], self.image_point_clamped(event.x, event.y)
            box = [int(min(x1, x2)), int(min(y1, y2)),
                   int(round(max(x1, x2))), int(round(max(y1, y2)))]
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                self.add_prompt({"box": box}, drag["shift"], drag["ctrl"])
            else:
                self.area.queue_draw()  # too small to be a box: just clear the rubber band
        elif drag["point"] is not None:
            self.add_click(drag["point"], drag["shift"], drag["ctrl"])
        return True

    # --- clicks ---

    def use_guide(self):
        """Select the object inside each separate area of GIMP's selection (its bounding box
        plus a click at the point deepest inside it), starting over, as one undo step."""
        if self.guide and self.ready:
            self.history.append(copy.deepcopy(self.objects))
            self.objects = [{"sign": 1, "box": g["box"], "points": [g["point"] + [1]]}
                            for g in self.guide["prompts"]]
            self.render()

    def add_click(self, point, shift=False, ctrl=False):
        self.add_prompt({"points": [point + [1]]}, shift, ctrl)

    def add_prompt(self, prompt, shift=False, ctrl=False):
        """Like Fuzzy Select: a plain click or box starts over with that object, Shift adds
        it and Ctrl subtracts it. Each click or box is its own SAM2 prompt; for clicks,
        Granularity decides whether it means a whole object or a part, while a box says
        that itself."""
        if ctrl and not self.objects:
            return  # nothing to remove from
        self.history.append(copy.deepcopy(self.objects))
        new = dict(prompt, sign=-1 if ctrl else 1)
        if shift or ctrl:
            self.objects.append(new)
        else:
            self.objects = [new]
        self.render()

    def undo(self):
        if self.history:
            self.objects = self.history.pop()
            self.render()

    def clear(self):
        if self.objects:
            self.history.append(copy.deepcopy(self.objects))
            self.objects = []
            self.render()

    # --- cursor ---

    MODIFIER_KEYS = {
        Gdk.KEY_Shift_L: Gdk.ModifierType.SHIFT_MASK, Gdk.KEY_Shift_R: Gdk.ModifierType.SHIFT_MASK,
        Gdk.KEY_Control_L: Gdk.ModifierType.CONTROL_MASK, Gdk.KEY_Control_R: Gdk.ModifierType.CONTROL_MASK,
    }

    def update_cursor(self, state=None):
        """Fuzzy-Select-style cursor over the image: '+' with Shift, '-' with Ctrl; the
        busy cursor while analysing and a move cursor while panning."""
        if state is not None:
            self.modifiers = state & (Gdk.ModifierType.SHIFT_MASK | Gdk.ModifierType.CONTROL_MASK)
        window = self.area.get_window()
        if window is None:
            return
        if self.busy:
            key = "wait"
        elif self.pan_start is not None:
            key = "move"
        elif self.modifiers & Gdk.ModifierType.CONTROL_MASK:
            key = "-"
        elif self.modifiers & Gdk.ModifierType.SHIFT_MASK:
            key = "+"
        else:
            key = None
        if key not in self.cursors:
            display = window.get_display()
            if key in ("wait", "move"):
                self.cursors[key] = Gdk.Cursor.new_from_name(display, key)
            else:
                self.cursors[key] = make_tool_cursor(display, key)
        window.set_cursor(self.cursors[key])
        if self.hover_polygons:
            self.area.queue_draw()  # the hover outline's colour follows Shift/Ctrl

    def on_key_release(self, widget, event):
        if event.keyval in self.MODIFIER_KEYS:
            # The event's state is from before the release.
            self.update_cursor(event.state & ~self.MODIFIER_KEYS[event.keyval])
        return False

    def on_key(self, widget, event):
        if event.keyval in self.MODIFIER_KEYS:
            # The event's state is from before the press.
            self.update_cursor(event.state | self.MODIFIER_KEYS[event.keyval])
            return False
        ctrl = event.state & Gdk.ModifierType.CONTROL_MASK
        shift = event.state & Gdk.ModifierType.SHIFT_MASK
        key = event.keyval
        if ctrl and key in (Gdk.KEY_z, Gdk.KEY_Z):
            self.undo()
        elif ctrl and shift and key in (Gdk.KEY_j, Gdk.KEY_J):
            self.zoom_fit()
        elif isinstance(self.get_focus(), (Gtk.SpinButton, Gtk.Entry)):
            return False  # typing a number: don't steal + - 1 2
        elif key in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
            self.zoom_step(1)
        elif key in (Gdk.KEY_minus, Gdk.KEY_KP_Subtract):
            self.zoom_step(-1)
        elif key in (Gdk.KEY_1, Gdk.KEY_KP_1):
            self.set_zoom(1.0)
        elif key in (Gdk.KEY_2, Gdk.KEY_KP_2):
            self.set_zoom(2.0)
        elif key in (Gdk.KEY_m, Gdk.KEY_M):
            # A radio button can't be switched off directly: switch the other one on.
            other = self.overlay_button if self.mask_button.get_active() else self.mask_button
            other.set_active(True)
        else:
            return False
        return True


    def option_changed(self, key, value):
        self.options[key] = value
        if self.objects:
            self.render()
        if key in ("granularity", "threshold") and self.hover_target:
            self.hover_polygons = None
            self.hover_at(self.hover_target)

    # --- hover preview ---

    def hover_at(self, point):
        """Ask for the outline a click at `point` would select. Only one request is in
        flight; when it returns, the newest mouse position is requested next."""
        self.hover_target = point
        if point is None:
            self.clear_hover()
            return
        if (not self.options["hover"] or not self.ready or self.busy or self.hover_busy
                or self.drag or self.pan_start is not None):
            return
        self.hover_busy = True
        self.backend.request({"cmd": "hover", "point": point, "threshold": self.options["threshold"],
                              "granularity": self.options["granularity"]},
                             self.on_hover, coalesce="hover")

    def on_hover(self, reply):
        self.hover_busy = False
        if reply.get("ok") and self.hover_target is not None and self.options["hover"]:
            self.hover_polygons = reply["polygons"]
            self.area.queue_draw()
        if self.hover_target is not None and self.hover_target != reply.get("point"):
            self.hover_at(self.hover_target)  # the mouse moved on meanwhile
        return False

    def clear_hover(self):
        self.hover_target = None
        if self.hover_polygons:
            self.hover_polygons = None
            self.area.queue_draw()

    def on_hover_toggled(self, button):
        self.options["hover"] = button.get_active()
        if not self.options["hover"]:
            self.clear_hover()

    # --- backend ---

    def render_request(self, final=False, out=None):
        req = {"cmd": "render", "objects": self.objects, "final": final}
        for key in RENDER_OPTIONS:
            req[key] = self.options[key]
        if out:
            req["out"] = out
        return req

    def preview_request(self, soft):
        self.preview_count += 1
        req = self.render_request()
        req["preview_out"] = f"{os.path.splitext(self.preview_png)[0]}_{self.preview_count}.png"
        req["soft_preview"] = soft
        req["view"] = "mask" if self.mask_button.get_active() else "overlay"
        return req

    def render(self):
        """Fast preview right away; with Soft edges on, a soft-edge preview follows once
        the clicks and options have been still for a moment."""
        self.area.queue_draw()  # show the new click marker right away
        self.select_button.set_sensitive(bool(self.objects))
        self.version += 1
        if self.soft_timer:
            GLib.source_remove(self.soft_timer)
            self.soft_timer = None
        if not self.objects:
            self.overlay = None
            self.set_status("Click an object to select it.")
            return
        version = self.version
        self.backend.request(self.preview_request(soft=False),
                             lambda reply: self.on_rendered(reply, version), coalesce="preview")
        if self.options["soft"] and self.soft_available:
            self.soft_timer = GLib.timeout_add(500, self.request_soft_preview, version)

    def request_soft_preview(self, version):
        self.soft_timer = None
        if version == self.version:
            self.backend.request(self.preview_request(soft=True),
                                 lambda reply: self.on_rendered(reply, version, soft=True),
                                 coalesce="soft")
        return False

    def on_loaded(self, reply):
        if not reply["ok"]:
            self.set_busy("Could not analyse the image:\n\n" + reply["error"], error=True)
            self.set_status("Error: " + reply["error"])
            return False
        self.set_busy(None)
        self.ready = True
        how = ("image analysis reused" if reply.get("cached")
               else f"analysed in {reply['seconds']} s")
        hint = "Click an object to select it."
        self.guide = reply.get("guide")
        if self.guide:
            self.guide_button.show()
            hint = ("Click an object, or press \"Find object in GIMP's selection\" to start "
                    "from your selection (cyan).")
            self.area.queue_draw()
        self.set_status(f"Ready ({reply['device']}, {how}). {hint}")
        return False

    def on_rendered(self, reply, version, soft=False):
        if not reply["ok"]:
            self.set_status("Error: " + reply["error"])
            return False
        stale = version != self.version or not self.objects  # changed while in flight
        if not stale:
            self.overlay = surface_from_pixbuf(GdkPixbuf.Pixbuf.new_from_file(reply["preview"]))
        try:
            os.remove(reply["preview"])
        except OSError:
            pass
        if stale:
            return False
        pct = 100 * reply["area"] / (self.image_size[0] * self.image_size[1])
        boxes = sum(1 for o in self.objects if o.get("box"))
        clicks = len(self.objects) - boxes
        parts = [f"{clicks} click{'s' * (clicks != 1)}"] if clicks else []
        parts += [f"{boxes} box{'es' * (boxes != 1)}"] if boxes else []
        status = f"{pct:.1f}% of the image selected ({', '.join(parts)}"
        if soft:
            status += f", soft edges shown, {reply['seconds']:.1f} s)"
            if not self.mask_button.get_active():
                status += " - switch the view to Mask to see them"
        elif self.options["soft"] and self.soft_available:
            status += ") - refining soft edges..."
        else:
            status += f", {reply['seconds'] * 1000:.0f} ms)"
        self.set_status(status)
        self.area.queue_draw()
        return False


def find_fuzzy_select_icon():
    """GIMP's own Fuzzy Select tool icon: from the user's icon theme, else the default."""
    dirs = []
    try:
        dirs.append(Gimp.icon_theme_dir().get_path())
    except Exception:  # not running inside GIMP (tests)
        pass
    try:
        dirs.append(os.path.join(Gimp.data_directory(), "icons", "Default"))
    except Exception:
        pass
    dirs.append(os.path.join(os.path.dirname(sys.executable), "..", "share", "gimp", "3.0",
                             "icons", "Default"))
    for d in dirs:
        for name in ("gimp-tool-fuzzy-select.svg", "gimp-tool-fuzzy-select-symbolic.svg"):
            path = os.path.join(d, "scalable", "apps", name)
            if os.path.exists(path):
                return path
    return None


def make_tool_cursor(display, sign=None):
    """A GIMP-style tool cursor: a small crosshair whose centre is the hotspot (the pixel
    that gets clicked), the Fuzzy Select icon below right, and a '+' or '-' for Shift/Ctrl."""
    size, hx, hy = 32, 8, 8
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    cr = cairo.Context(surface)
    cr.set_line_cap(cairo.LINE_CAP_SQUARE)

    def stroke_outlined(path):
        for rgb, width in (((1, 1, 1), 3), ((0, 0, 0), 1)):
            path()
            cr.set_source_rgb(*rgb)
            cr.set_line_width(width)
            cr.stroke()

    def crosshair():
        for (x1, y1), (x2, y2) in [((hx, 1), (hx, hy - 3)), ((hx, hy + 3), (hx, 2 * hy - 1)),
                                   ((1, hy), (hx - 3, hy)), ((hx + 3, hy), (2 * hx - 1, hy))]:
            cr.move_to(x1 + 0.5, y1 + 0.5)
            cr.line_to(x2 + 0.5, y2 + 0.5)

    stroke_outlined(crosshair)

    icon_path = find_fuzzy_select_icon()
    ix, iy = 14, 14
    if icon_path:
        icon = GdkPixbuf.Pixbuf.new_from_file_at_size(icon_path, 16, 16)
        # A white halo (the icon's shape, offset in every direction) keeps it visible on
        # dark images; then the icon itself.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            Gdk.cairo_set_source_pixbuf(cr, icon, ix + dx, iy + dy)
            shape = cr.get_source()
            cr.set_source_rgba(1, 1, 1, 0.9)
            cr.mask(shape)
        Gdk.cairo_set_source_pixbuf(cr, icon, ix, iy)
        cr.paint()
    else:  # no icon found: a plain wand
        stroke_outlined(lambda: (cr.move_to(ix + 4.5, iy + 4.5), cr.line_to(ix + 14.5, iy + 14.5)))

    if sign:
        sx, sy, r = 25, 6, 3  # centre and half-length of the sign

        def sign_path():
            cr.move_to(sx - r + 0.5, sy + 0.5)
            cr.line_to(sx + r + 0.5, sy + 0.5)
            if sign == "+":
                cr.move_to(sx + 0.5, sy - r + 0.5)
                cr.line_to(sx + 0.5, sy + r + 0.5)

        stroke_outlined(sign_path)

    pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, size, size)
    return Gdk.Cursor.new_from_pixbuf(display, pixbuf, hx, hy)


def surface_from_pixbuf(pixbuf):
    """Convert once to a cairo surface; painting a pixbuf converts it on every redraw."""
    fmt = cairo.FORMAT_ARGB32 if pixbuf.get_has_alpha() else cairo.FORMAT_RGB24
    surface = cairo.ImageSurface(fmt, pixbuf.get_width(), pixbuf.get_height())
    cr = cairo.Context(surface)
    Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
    cr.paint()
    return surface


def scaled_surface(surface, scale):
    w, h = max(1, int(surface.get_width() * scale)), max(1, int(surface.get_height() * scale))
    out = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    cr = cairo.Context(out)
    cr.scale(scale, scale)
    cr.set_source_surface(surface, 0, 0)
    cr.get_source().set_filter(cairo.FILTER_GOOD)
    cr.paint()
    return out


# --- GIMP side ------------------------------------------------------------------------


def export_selection(image, path):
    """Write GIMP's selection as 8-bit values (image size), read in its own storage format
    so the values aren't gamma-converted."""
    selection = image.get_selection()
    w, h = image.get_width(), image.get_height()
    fmt = "Y' u8" if Babl.get_name(selection.get_format()).startswith("Y'") else "Y u8"
    data = selection.get_buffer().get(Gegl.Rectangle.new(0, 0, w, h), 1.0, fmt,
                                      Gegl.AbyssPolicy.NONE)
    with open(path, "wb") as f:
        f.write(bytes(data))


def export_visible(image, path):
    dup = image.duplicate()
    dup.merge_visible_layers(Gimp.MergeType.CLIP_TO_IMAGE)
    proc = Gimp.get_pdb().lookup_procedure("file-png-export")
    config = proc.create_config()
    config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
    config.set_property("image", dup)
    config.set_property("file", Gio.File.new_for_path(path))
    proc.run(config)
    dup.delete()


def write_mask(drawable, data):
    """Write 8-bit values into a channel or layer mask the drawable's size, in its own colour
    model so 50% stays 50% ("Y u8" into a Y' channel would gamma-convert partial values)."""
    w, h = drawable.get_width(), drawable.get_height()
    fmt = "Y' u8" if Babl.get_name(drawable.get_format()).startswith("Y'") else "Y u8"
    buffer = drawable.get_buffer()
    buffer.set(Gegl.Rectangle.new(0, 0, w, h), fmt, data)
    buffer.flush()
    drawable.update(0, 0, w, h)


def crop_to_layer(data, image_w, image_h, layer):
    """The part of an image-sized 8-bit mask that falls on `layer`. Layers can be smaller
    than the image, shifted, or stick out of it; outside the image counts as unselected."""
    _, ox, oy = layer.get_offsets()
    lw, lh = layer.get_width(), layer.get_height()
    out = bytearray(lw * lh)
    x0, x1 = max(ox, 0), min(ox + lw, image_w)
    if x1 > x0:
        for y in range(max(oy, 0), min(oy + lh, image_h)):
            dst = (y - oy) * lw + (x0 - ox)
            out[dst:dst + x1 - x0] = data[y * image_w + x0:y * image_w + x1]
    return bytes(out)


class KeepSelectedItems:
    """GIMP's Channels dock selects a newly added channel, which deselects the user's
    layers; then Copy etc. silently do nothing. Restore the selected items afterwards."""

    def __init__(self, image):
        self.image = image

    def __enter__(self):
        image = self.image
        self.layers, self.channels = image.get_selected_layers(), image.get_selected_channels()
        self.paths = image.get_selected_paths()

    def __exit__(self, *exc):
        image = self.image
        if self.layers:
            image.set_selected_layers(self.layers)
        elif self.channels:
            image.set_selected_channels(self.channels)
        if self.paths:  # GIMP rejects an empty list here
            image.set_selected_paths(self.paths)
        log(f"selected items restored: layers={[l.get_name() for l in image.get_selected_layers()]} "
            f"channels={[c.get_name() for c in image.get_selected_channels()]}")
        return False


def apply_selection(image, data, operation):
    """Turn an image-sized 8-bit mask into GIMP's selection."""
    w, h = image.get_width(), image.get_height()
    with KeepSelectedItems(image):
        channel = Gimp.Channel.new(image, "AI Fuzzy Select", w, h, 100, Gegl.Color.new("black"))
        channel.set_visible(False)  # a visible channel flashes as an overlay on the canvas
        image.insert_channel(channel, None, 0)
        write_mask(channel, data)
        image.select_item(OPERATIONS[operation], channel)
        image.remove_channel(channel)


def apply_result(image, raw_file, output, operation, layer):
    """Create what the user chose under Result. `layer` is the layer that was selected
    when the plugin started (for "layer" and "mask"). Returns an error message or None."""
    w, h = image.get_width(), image.get_height()
    with open(raw_file, "rb") as f:
        data = f.read()
    if output == "selection":
        apply_selection(image, data, operation)
    elif output == "channel":
        with KeepSelectedItems(image):
            channel = Gimp.Channel.new(image, "AI Fuzzy Select", w, h, 50, Gegl.Color.new("black"))
            channel.set_visible(False)
            image.insert_channel(channel, None, 0)
            write_mask(channel, data)
    elif output in ("layer", "cutout"):
        if layer is not None:  # a copy of the current layer, right above it
            new = layer.copy()
            new.set_name(f"{layer.get_name()} (AI Fuzzy Select)")
            image.insert_layer(new, layer.get_parent(), image.get_item_position(layer))
        else:  # no layer selected: what's visible
            new = Gimp.Layer.new_from_visible(image, image, "AI Fuzzy Select")
            image.insert_layer(new, None, 0)
        mask = new.get_mask()  # a copied layer may bring its own mask; replace its contents
        if mask is None:
            mask = new.create_mask(Gimp.AddMaskType.WHITE)
            new.add_mask(mask)
        write_mask(mask, crop_to_layer(data, w, h, new))
        if output == "cutout":  # make it permanent: everything outside the object deleted
            if not new.has_alpha():
                new.add_alpha()
            new.remove_mask(Gimp.MaskApplyMode.APPLY)
        image.set_selected_layers([new])
    elif output == "mask":
        if layer is None:
            return "Select a layer first: \"Mask on current layer\" needs one."
        mask = layer.get_mask()  # an existing mask gets the new contents
        if mask is None:
            mask = layer.create_mask(Gimp.AddMaskType.WHITE)
            layer.add_mask(mask)
        write_mask(mask, crop_to_layer(data, w, h, layer))
    return None


def preview_size(w, h):
    scale = min(1.0, PREVIEW_MAX / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


LOG_FILE = os.path.join(tempfile.gettempdir(), "aiselect.log")


def log(message):
    """Append to %TEMP%\\aiselect.log (rewritten each run), for diagnosing problems."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {message}\n")
    except OSError:
        pass


def selection_summary(image):
    _, non_empty, x1, y1, x2, y2 = Gimp.Selection.bounds(image)
    return f"non_empty={non_empty} bounds=({x1},{y1})-({x2},{y2})"


def ai_select(procedure, run_mode, image, config):
    """Returns an error message, or None."""
    try:
        open(LOG_FILE, "w").close()
    except OSError:
        pass
    log(f"run_mode={run_mode} image={image.get_width()}x{image.get_height()} "
        f"precision={image.get_precision().value_nick} selection before: {selection_summary(image)}")
    try:
        error = ai_select_steps(procedure, run_mode, image, config)
    except Exception:
        import traceback
        log("EXCEPTION\n" + traceback.format_exc())
        error = "AI Fuzzy Select failed unexpectedly:\n\n" + traceback.format_exc(limit=3)
    log(f"returning error={error!r}; selection now: {selection_summary(image)}")
    return error


def soft_edges_available(settings):
    return bool(settings.get("vitmatteDir") or settings.get("appDir"))


def ai_select_steps(procedure, run_mode, image, config):
    settings = load_json(SETTINGS_FILE, None)
    if not settings:
        return "AI Fuzzy Select isn't set up. Run install_ai_select.py."
    options = dict(DEFAULT_OPTIONS, **load_json(OPTIONS_FILE, {}))
    work = tempfile.mkdtemp(prefix="aiselect_")
    backend = None
    try:
        image_png = os.path.join(work, "image.png")
        export_visible(image, image_png)
        size = (image.get_width(), image.get_height())
        psize = preview_size(*size)
        preview_png = os.path.join(work, "preview.png")
        backend = Backend(settings)
        # Without an existing selection, "Existing selection in GIMP" is hidden and the result simply
        # becomes the selection; the remembered choice is kept for next time.
        has_selection = Gimp.Selection.bounds(image)[1]
        selected_layers = image.get_selected_layers()
        layer = selected_layers[0] if selected_layers else None  # for "layer"/"mask" results

        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init("aiselect")
            guide_raw = None
            if has_selection:  # offered as a starting point in the window
                guide_raw = os.path.join(work, "gimp_selection.raw")
                export_selection(image, guide_raw)
            dialog = AISelectDialog(backend, image_png, size, preview_png, psize, options,
                                    soft_available=soft_edges_available(settings),
                                    has_selection=has_selection, guide_raw=guide_raw)
            response = dialog.run()
            objects, options = dialog.objects, dialog.options
            dialog.destroy()
            log(f"dialog response={response} objects={json.dumps(objects)} options={json.dumps(options)}")
            if response != Gtk.ResponseType.OK or not objects:
                return None
            with open(OPTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(options, f)
        else:  # scripted use: the clicks come as JSON in the "objects" argument
            objects = json.loads(config.get_property("objects") or "[]")
            for key in ("threshold", "granularity", "grow", "feather", "soft", "operation", "output"):
                options[key] = config.get_property(key)
            options["fill_holes"] = config.get_property("fill-holes")
            reply = backend.call({"cmd": "load", "image": image_png, "preview": preview_png,
                                  "preview_width": psize[0], "preview_height": psize[1]})
            if not reply["ok"]:
                return reply["error"]

        Gimp.progress_init("AI Fuzzy Select: creating the result...")
        out = os.path.join(work, "selection.raw")
        req = {"cmd": "render", "objects": objects, "final": True, "out": out}
        for key in RENDER_OPTIONS:
            req[key] = options[key]
        reply = backend.call(req)
        log(f"final render reply={json.dumps(reply)} out_size="
            f"{os.path.getsize(out) if os.path.exists(out) else 'missing'}")
        if not reply["ok"]:
            return reply["error"]
        if reply.get("area", 0) == 0:
            return "The selection came out empty (try a higher Threshold or another click)."
        output = options.get("output", "selection")
        if output not in OUTPUTS:
            return f"Unknown result type {output!r}."
        image.undo_group_start()
        try:
            interactive = run_mode == Gimp.RunMode.INTERACTIVE
            operation = options["operation"] if has_selection or not interactive else "replace"
            error = apply_result(image, out, output, operation, layer)
        finally:
            image.undo_group_end()
        if error:
            return error
        log(f"applied ({output}, {operation}): {selection_summary(image)}")
        Gimp.displays_flush()
        return None
    finally:
        if backend is not None:
            backend.close()
        Gimp.progress_end()
        shutil.rmtree(work, ignore_errors=True)


class AISelect(Gimp.PlugIn):
    def do_query_procedures(self):
        return [PROC_NAME]

    def do_set_i18n(self, name):
        return False

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, self.run, None)
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE
                                       | Gimp.ProcedureSensitivityMask.NO_DRAWABLES)
        procedure.set_menu_label("_AI Fuzzy Select...")
        procedure.add_menu_path("<Image>/Select")
        procedure.set_documentation(
            "Select the object you click on (AI fuzzy select)",
            "Like Fuzzy Select, but grows the selection over the clicked object as SAM2 sees "
            "it instead of over similar colours. Click to select, Shift+click to add, "
            "Ctrl+click to remove; the result becomes a normal selection.",
            name)
        procedure.set_attribution("AI Fuzzy Select plugin", "MIT", "2026")
        rw = GObject.ParamFlags.READWRITE
        # Used when called from scripts (the window passes its own state).
        procedure.add_string_argument("objects", "Objects (JSON)",
                                      'Clicks: [{"sign": 1, "points": [[x, y, 1]]}]', "[]", rw)
        procedure.add_int_argument("threshold", "Threshold", "0-100, 50 = SAM2 default", 0, 100, 50, rw)
        procedure.add_string_argument("granularity", "Granularity", "auto, whole, part or detail", "auto", rw)
        procedure.add_int_argument("fill-holes", "Fill holes", "Fill enclosed holes up to this diameter (px); 0 = off", 0, 500, 0, rw)
        procedure.add_int_argument("grow", "Grow/shrink", "Pixels", -100, 100, 0, rw)
        procedure.add_int_argument("feather", "Feather", "Pixels", 0, 200, 0, rw)
        procedure.add_boolean_argument("soft", "Soft edges", "ViTMatte soft edges", False, rw)
        procedure.add_string_argument("operation", "Operation", "replace, add, subtract or intersect",
                                      "replace", rw)
        procedure.add_string_argument("output", "Result", "selection, layer, cutout, mask or channel",
                                      "selection", rw)
        return procedure

    def run(self, procedure, run_mode, image, drawables, config, data):
        error = ai_select(procedure, run_mode, image, config)
        if error:
            if run_mode == Gimp.RunMode.INTERACTIVE:
                Gimp.message(error)
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(error))
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


if __name__ == "__main__":
    Gimp.main(AISelect.__gtype__, sys.argv)
