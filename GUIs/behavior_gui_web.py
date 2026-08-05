"""Panel (browser) GUI for browsing NDNF behavior pipeline data.

Structural port of behavior_gui.py (Tkinter) for use over a remote VS Code
Server / code-server session, where there is no local X display for Tkinter
to draw a native window into. This runs a small local web server instead,
and is viewed in a browser tab through VS Code's automatic port forwarding.

Same DataJoint config handling as behavior_gui.py (gui_config.json next to
this file remembers the path to your dj_local_conf*.json), and the same
underlying plotting code (ndnf_pipeline.plot.behavior_plots /
videography_plots) - only the widget layer changed.

Run with, from the repo root:
    panel serve GUIs/behavior_gui_web.py --show
which opens a local browser tab automatically (use this on a machine with a
display, e.g. testing locally).

On a *remote* VS Code Server / code-server box, skip --show and forward the
port instead:
    panel serve GUIs/behavior_gui_web.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=*
Then open VS Code's "Ports" panel, forward 5006 (or let auto-forwarding pick
it up), and open the forwarded URL in your local browser.

--allow-websocket-origin=* matters here: the whole UI is painted over a
websocket back to this process, and Bokeh (which Panel is built on) rejects
websocket connections whose Origin header doesn't match a known host. The
forwarded URL VS Code hands you is almost never literally "localhost:5006"
(it's a proxied domain/port), so without this flag the page loads as a blank
shell - HTML/CSS arrive, but no widgets ever get painted into it.

`python GUIs/behavior_gui_web.py` also works and starts the same server.

This is a single-user tool, same as the Tkinter version: it drives one
shared DataJoint connection, so don't point multiple browser tabs at
different DataJoint configs at the same time.
"""
import queue
import sys
import threading
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # figures are built headless, then embedded as images below
import matplotlib.pyplot as plt
import numpy as np

import panel as pn
import holoviews as hv
from holoviews import streams

pn.extension("modal", notifications=True, sizing_mode="stretch_width")
hv.extension("bokeh")

GUIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUIS_DIR.parent
sys.path.insert(0, str(GUIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import dj_connection


class _Guard:
    """Suppress one specific callback while a widget's .value/.options are being set from
    code, without touching the widget's other watchers.

    Mirrors how Tkinter only fires <<...Selected>>/<<ListboxSelect>> on real user
    interaction, never on a plain var.set()/listbox.insert() from code - callers pair
    `with guard:` around the programmatic update with an explicit call to the relevant
    handler afterwards.

    Deliberately NOT implemented with param.parameterized.discard_events(widget): that
    discards *all* events on the widget, including Panel's own internal watcher that syncs
    the value to the browser - so the widget would keep the new value server-side (plots
    would look right) while the dropdown shown in the browser never visibly updates.
    """

    def __init__(self):
        self.suspended = False

    def __enter__(self):
        self.suspended = True
        return self

    def __exit__(self, *exc):
        self.suspended = False

    def wrap(self, handler):
        def wrapped(event=None):
            if not self.suspended:
                handler(event)
        return wrapped


# ---------------------------------------------------------------------------
# small shared controls (structural port of the Tkinter helper Frames)
# ---------------------------------------------------------------------------

class UniformRangeControl(pn.Row):
    """Checkbox + numeric input controlling a symmetric +/- force range, e.g. [-20,20,-20,20]."""

    def __init__(self, on_change, default_value=20.0, default_enabled=True, **params):
        self.on_change = on_change
        self.enabled_cb = pn.widgets.Checkbox(name="Uniform force range ±", value=default_enabled)
        self.value_input = pn.widgets.FloatInput(value=default_value, width=70, step=1)
        super().__init__(self.enabled_cb, self.value_input, pn.widgets.StaticText(value="g", margin=(8, 0)),
                          **params)
        self.enabled_cb.param.watch(lambda e: self.on_change(), "value")
        self.value_input.param.watch(lambda e: self.on_change(), "value")

    @property
    def enabled(self):
        return self.enabled_cb.value

    def get_range(self):
        value = self.value_input.value if self.value_input.value is not None else 20.0
        return np.asarray([-1, 1, -1, 1]) * value


class PerfAxisControls(pn.Row):
    """Log/linear Y-axis toggle + rolling-mean window size, for a quiescence/response panel."""

    def __init__(self, on_change, default_window=10, **params):
        self.on_change = on_change
        self._default_window = default_window
        self.log_cb = pn.widgets.Checkbox(name="Log Y axis", value=False)
        self.window_input = pn.widgets.IntInput(name="Rolling mean window", value=default_window,
                                                 start=1, width=90)
        super().__init__(self.log_cb, self.window_input, **params)
        self.log_cb.param.watch(lambda e: self.on_change(), "value")
        self.window_input.param.watch(lambda e: self.on_change(), "value")

    @property
    def log_scale(self):
        return self.log_cb.value

    def get_smoothing_window(self):
        window = self.window_input.value
        return window if window and window >= 1 else self._default_window


class EpochSelector(pn.Row):
    """Checkboxes restricting the force histogram/trajectory (and force-vs-time) panels to
    samples within selected trial epoch(s): quiescence, response, reward consumption.

    All three checked (the default) is equivalent to no restriction at all.
    """
    EPOCHS = ('quiescence', 'response', 'reward')
    LABELS = {'quiescence': 'Quiescence', 'response': 'Response', 'reward': 'Reward'}

    def __init__(self, on_change, default_selected=EPOCHS, **params):
        self.on_change = on_change
        self.checkboxes = {epoch: pn.widgets.Checkbox(name=self.LABELS[epoch], value=epoch in default_selected)
                            for epoch in self.EPOCHS}
        for cb in self.checkboxes.values():
            cb.param.watch(lambda e: self.on_change(), "value")
        super().__init__(pn.widgets.StaticText(value="Epochs:", margin=(8, 5)),
                          *self.checkboxes.values(), **params)

    def get_selected_epochs(self):
        return tuple(epoch for epoch in self.EPOCHS if self.checkboxes[epoch].value)


class TraceFilterControls(pn.Row):
    """Dropdown + millisecond-based parameter inputs to smooth each trial's force trace
    before it feeds the histogram/trajectory/force-vs-time panels: none, boxcar (moving
    average), median, gaussian, or Savitzky-Golay (a local polynomial fit -- keeps peak shape
    better than the other three, at the cost of needing a window wide enough to fit the
    chosen order).

    Window/sigma are entered in milliseconds rather than samples, since the raw sample count
    for a given duration depends on the (block-specific) force trace sampling rate; a gray
    "(~N samples)" hint next to each input shows what that currently works out to, updated via
    set_sample_interval() whenever the selected block's sampling rate becomes known. Only the
    parameter input(s) relevant to the selected method are enabled.
    """
    METHODS = ('none', 'boxcar', 'median', 'gaussian', 'savgol')
    METHOD_LABELS = {
        'none': 'None', 'boxcar': 'Boxcar (moving avg)', 'median': 'Median',
        'gaussian': 'Gaussian', 'savgol': 'Savitzky-Golay',
    }
    RELEVANT_PARAMS = {
        'none': (), 'boxcar': ('window',), 'median': ('window',),
        'gaussian': ('sigma',), 'savgol': ('window', 'polyorder'),
    }

    def __init__(self, on_change, **params):
        self.on_change = on_change
        self._sample_interval_s = None  # seconds/sample for the currently selected block
        self.method_select = pn.widgets.Select(name="Trace filter",
                                                 options=[self.METHOD_LABELS[m] for m in self.METHODS],
                                                 value=self.METHOD_LABELS['none'], width=170)
        self.window_ms_input = pn.widgets.FloatInput(name="window (ms)", value=50.0, start=0, width=90)
        self.window_hint = pn.widgets.StaticText(value="", margin=(22, 0, 0, 5))
        self.sigma_ms_input = pn.widgets.FloatInput(name="sigma (ms)", value=20.0, start=0, width=90)
        self.sigma_hint = pn.widgets.StaticText(value="", margin=(22, 0, 0, 5))
        self.polyorder_input = pn.widgets.IntInput(name="poly order", value=3, start=0, width=90)
        self.method_select.param.watch(lambda e: self._on_method_changed(), "value")
        self.window_ms_input.param.watch(lambda e: self._on_param_changed(), "value")
        self.sigma_ms_input.param.watch(lambda e: self._on_param_changed(), "value")
        self.polyorder_input.param.watch(lambda e: self.on_change(), "value")
        super().__init__(self.method_select, self.window_ms_input, self.window_hint,
                          self.sigma_ms_input, self.sigma_hint, self.polyorder_input, **params)
        self._update_enabled_state()
        self._update_hints()

    def _method_key(self):
        label = self.method_select.value
        for key, lbl in self.METHOD_LABELS.items():
            if lbl == label:
                return key
        return 'none'

    def _on_method_changed(self):
        self._update_enabled_state()
        self.on_change()

    def _on_param_changed(self):
        self._update_hints()
        self.on_change()

    def _update_enabled_state(self):
        relevant = self.RELEVANT_PARAMS[self._method_key()]
        self.window_ms_input.disabled = 'window' not in relevant
        self.sigma_ms_input.disabled = 'sigma' not in relevant
        self.polyorder_input.disabled = 'polyorder' not in relevant

    def set_sample_interval(self, sample_interval_s):
        """Called by the tab when the selected block's force-trace sampling rate becomes
        known (or unknown, with None), so the "(~N samples)" hints stay accurate."""
        self._sample_interval_s = sample_interval_s
        self._update_hints()

    def _update_hints(self):
        window_ms = self.window_ms_input.value or 0.0
        sigma_ms = self.sigma_ms_input.value or 0.0
        if self._sample_interval_s:
            # imported here (not at module load) so constructing this widget before a
            # DataJoint connection exists never touches the ndnf_pipeline package -- see
            # set_sample_interval(), which is the only caller that can make this branch true
            from ndnf_pipeline.plot.behavior_plots import ms_to_samples, ms_to_samples_float
            window_samples = ms_to_samples(window_ms, self._sample_interval_s)
            sigma_samples = ms_to_samples_float(sigma_ms, self._sample_interval_s)
            self.window_hint.value = f"(~{window_samples} samples)"
            self.sigma_hint.value = f"(~{sigma_samples:.1f} samples)"
        else:
            self.window_hint.value = "(rate unknown)"
            self.sigma_hint.value = "(rate unknown)"

    def get_filter_kwargs(self):
        return dict(filter_method=self._method_key(),
                    filter_window_ms=self.window_ms_input.value if self.window_ms_input.value is not None else 50.0,
                    filter_sigma_ms=self.sigma_ms_input.value if self.sigma_ms_input.value is not None else 20.0,
                    filter_polyorder=self.polyorder_input.value if self.polyorder_input.value is not None else 3)


def make_plot_pane():
    """A Matplotlib pane analogous to Tkinter's PlotPanel; swap figures via show_figure().

    No fixed height: the pane fills the available width and lets height follow the figure's
    own aspect ratio (Matplotlib pane's default fixed_aspect=True) instead of squashing a
    portrait figure - e.g. Block Detail's 12x19in figure - down into a small fixed box. A
    tall figure then means scrolling to see all of it, which beats shrinking it to be tiny.
    """
    return pn.pane.Matplotlib(None, tight=True, format="png", dpi=110, sizing_mode="stretch_width")


def show_figure(pane, fig):
    old = pane.object
    pane.object = fig
    if old is not None and old is not fig:
        plt.close(old)


# ---------------------------------------------------------------------------
# plot tabs
# ---------------------------------------------------------------------------

class SessionOverviewTab:
    """The 'all session' plot: every block of the selected session, side by side."""

    title = "Session Overview"

    def __init__(self, app):
        self.app = app
        self.range_control = UniformRangeControl(on_change=self.refresh)
        self.perf_controls = PerfAxisControls(on_change=self.refresh)
        refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary", width=90)
        refresh_btn.on_click(lambda e: self.refresh())

        self.pane = make_plot_pane()
        controls = pn.Row(self.range_control, self.perf_controls, refresh_btn)
        self.panel = pn.Column(controls, self.pane)

    def on_session_changed(self):
        self.refresh()

    def refresh(self, *_):
        subject_id, session = self.app.get_selected_subject_session()
        if subject_id is None or session is None:
            self.pane.object = None
            return
        from ndnf_pipeline.plot.behavior_plots import plot_session_blocks_overview

        def work():
            return plot_session_blocks_overview(
                subject_id, session,
                force_uniform_range=self.range_control.enabled,
                uniform_force_range=self.range_control.get_range(),
                perf_log_yscale=self.perf_controls.log_scale,
                perf_smoothing_window=self.perf_controls.get_smoothing_window(),
                epochs=self.app.epoch_selector.get_selected_epochs(),
                **self.app.filter_controls.get_filter_kwargs())

        self.app.run_plot(work, self.pane)


class BlockDetailTab:
    """4-panel detail figure for a single block of the selected session.

    A trial multi-select restricts which trials feed the force-distribution and
    trajectory panels; the performance (quiescence/response) panel always shows every
    trial in the block regardless of that selection.
    """

    title = "Block Detail"

    def __init__(self, app):
        self.app = app
        self._block_guard = _Guard()
        self._trial_guard = _Guard()
        self.block_select = pn.widgets.Select(name="Block", options=[], width=100)
        self.range_control = UniformRangeControl(on_change=self.refresh)
        self.perf_controls = PerfAxisControls(on_change=self.refresh)
        refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary", width=90)

        self.block_select.param.watch(self._block_guard.wrap(lambda e: self.on_block_selected()), "value")
        refresh_btn.on_click(lambda e: self.refresh())

        self.trial_select = pn.widgets.MultiSelect(
            name="Trials for 2D hist & trajectories (none = all)", options=[], size=20, width=150)
        self.trial_select.param.watch(self._trial_guard.wrap(lambda e: self.refresh()), "value")
        all_btn = pn.widgets.Button(name="All", width=60)
        none_btn = pn.widgets.Button(name="None", width=60)
        all_btn.on_click(lambda e: self.select_all_trials())
        none_btn.on_click(lambda e: self.select_no_trials())

        self.pane = make_plot_pane()
        controls = pn.Row(self.block_select, self.range_control, self.perf_controls, refresh_btn)
        left = pn.Column(self.trial_select, pn.Row(all_btn, none_btn), width=170)
        body = pn.Row(left, self.pane)
        self.panel = pn.Column(controls, body)

    def on_session_changed(self):
        subject_id, session = self.app.get_selected_subject_session()
        if subject_id is None or session is None:
            with self._block_guard:
                self.block_select.options = []
            with self._trial_guard:
                self.trial_select.options = []
            self.pane.object = None
            self.app.update_shared_sample_interval()
            return
        try:
            blocks = sorted((self.app.experiment.Block()
                              & {"subject_id": subject_id, "session": session}).fetch("block").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        values = [str(b) for b in blocks]
        with self._block_guard:
            self.block_select.options = values
            if values:
                self.block_select.value = values[0]
        if values:
            self.on_block_selected()
        else:
            with self._trial_guard:
                self.trial_select.options = []
            self.pane.object = None
            self.app.update_shared_sample_interval()

    def on_block_selected(self, *_):
        self.reload_trials()
        self.app.update_shared_sample_interval()
        self.refresh()

    def reload_trials(self):
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.block_select.value
        if subject_id is None or session is None or not block_str:
            with self._trial_guard:
                self.trial_select.options = []
            return
        key = {"subject_id": subject_id, "session": session, "block": int(block_str)}
        try:
            trials = sorted((self.app.experiment.BehaviorTrial() & key).fetch("trial").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        with self._trial_guard:
            self.trial_select.options = [str(t) for t in trials]
            self.trial_select.value = []

    def select_all_trials(self):
        with self._trial_guard:
            self.trial_select.value = list(self.trial_select.options)
        self.refresh()

    def select_no_trials(self):
        with self._trial_guard:
            self.trial_select.value = []
        self.refresh()

    def get_selected_trials(self):
        return [int(t) for t in self.trial_select.value]

    def refresh(self, *_):
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.block_select.value
        if subject_id is None or session is None or not block_str:
            return
        block = int(block_str)
        selected_trials = self.get_selected_trials()
        from ndnf_pipeline.plot.behavior_plots import plot_block_force_figure

        def work():
            return plot_block_force_figure(
                subject_id, session, block,
                force_uniform_range=self.range_control.enabled,
                uniform_force_range=self.range_control.get_range(),
                perf_log_yscale=self.perf_controls.log_scale,
                perf_smoothing_window=self.perf_controls.get_smoothing_window(),
                trials=selected_trials or None,
                epochs=self.app.epoch_selector.get_selected_epochs(),
                **self.app.filter_controls.get_filter_kwargs())

        self.app.run_plot(work, self.pane)


class SubjectTrendTab:
    """Trial count / session length / hit rate across all sessions of the selected subject."""

    title = "Subject Trend"

    def __init__(self, app):
        self.app = app
        refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary", width=90)
        refresh_btn.on_click(lambda e: self.refresh())
        self.pane = make_plot_pane()
        self.panel = pn.Column(pn.Row(refresh_btn), self.pane)

    def on_subject_changed(self):
        self.refresh()

    def refresh(self, *_):
        subject_id, _session = self.app.get_selected_subject_session()
        if subject_id is None:
            self.pane.object = None
            return
        from ndnf_pipeline.plot.behavior_plots import plot_subject_behavior_trend

        def work():
            return plot_subject_behavior_trend(self.app.experiment, subject_id)

        self.app.run_plot(work, self.pane)


class TrialsPerMouseTab:
    """Trials-per-mouse bar chart plus overlaid per-session trends, for a selectable set of mice."""

    title = "Trials per Mouse"

    def __init__(self, app):
        self.app = app
        self._mice_loaded = False
        self._mouse_guard = _Guard()

        self.mouse_select = pn.widgets.MultiSelect(name="Mice to show", options=[], size=20, width=180)
        self.mouse_select.param.watch(self._mouse_guard.wrap(lambda e: self.refresh()), "value")
        all_btn = pn.widgets.Button(name="All", width=70)
        none_btn = pn.widgets.Button(name="None", width=70)
        reload_btn = pn.widgets.Button(name="Reload mouse list", width=160)
        all_btn.on_click(lambda e: self.select_all())
        none_btn.on_click(lambda e: self.select_none())
        reload_btn.on_click(lambda e: self.reload_mice())

        self.bar_pane = make_plot_pane()
        self.trend_pane = make_plot_pane()

        left = pn.Column(self.mouse_select, pn.Row(all_btn, none_btn), reload_btn, width=200)
        right = pn.Column(
            pn.pane.Markdown("**Total trials** (selected mice, or all mice if none selected)"),
            self.bar_pane,
            pn.pane.Markdown("**Session trends** (selected mice, color-coded)"),
            self.trend_pane,
        )
        self.panel = pn.Row(left, right)

    def on_subject_changed(self):
        # the mouse list only needs populating once a connection is established
        if not self._mice_loaded:
            self.reload_mice()

    def reload_mice(self):
        if self.app.lab is None or self.app.experiment is None:
            return
        try:
            mouse_ids = sorted(self.app.lab.Subject.fetch("subject_id").tolist())
            trial_nums = [len(self.app.experiment.SessionTrial() & {"subject_id": m}) for m in mouse_ids]
        except Exception as exc:
            self.app.report_error(exc)
            return
        previously_selected = set(self.get_selected_mice())
        mice_with_trials = [m for m, n in zip(mouse_ids, trial_nums) if n > 0]
        with self._mouse_guard:
            self.mouse_select.options = mice_with_trials
            self.mouse_select.value = [m for m in mice_with_trials if m in previously_selected]
        self._mice_loaded = True
        self.refresh()

    def select_all(self):
        with self._mouse_guard:
            self.mouse_select.value = list(self.mouse_select.options)
        self.refresh()

    def select_none(self):
        with self._mouse_guard:
            self.mouse_select.value = []
        self.refresh()

    def get_selected_mice(self):
        return list(self.mouse_select.value)

    def refresh(self, *_):
        if self.app.lab is None or self.app.experiment is None:
            return
        selected = self.get_selected_mice()
        from ndnf_pipeline.plot.behavior_plots import plot_trials_per_mouse, plot_subject_behavior_trends

        def work_bar():
            return plot_trials_per_mouse(self.app.lab, self.app.experiment, subject_ids=selected or None)

        def work_trend():
            return plot_subject_behavior_trends(self.app.experiment, selected)

        self.app.run_plot(work_bar, self.bar_pane)
        self.app.run_plot(work_trend, self.trend_pane)


class WaterRestrictionTab:
    """Weight / water-consumed logs for every mouse on water restriction; selected subject is highlighted."""

    title = "Water Restriction"

    def __init__(self, app):
        self.app = app
        refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary", width=90)
        refresh_btn.on_click(lambda e: self.refresh())
        self.pane = make_plot_pane()
        self.panel = pn.Column(pn.Row(refresh_btn), self.pane)

    def on_subject_changed(self):
        self.refresh()

    def refresh(self, *_):
        subject_id, _session = self.app.get_selected_subject_session()
        from ndnf_pipeline.plot.behavior_plots import plot_water_restriction_overview

        def work():
            return plot_water_restriction_overview(self.app.lab, highlight_subject_id=subject_id)

        self.app.run_plot(work, self.pane)


# ---------------------------------------------------------------------------
# video generation tab
# ---------------------------------------------------------------------------

class VideoGenerationTab:
    """Generate an annotated trial-range video for one block, with an interactive crop tool.

    The block and trial range are inherited from the Block Detail tab (its block dropdown and
    its trial multi-select - "none selected" there means "every trial in the block") rather than
    picked here, so the video always matches whatever block/trials you're already looking at.

    Cropping uses a HoloViews BoxEdit rectangle drawn over the loaded frame (drag corners/edges
    to resize, drag the body to move) in place of Tkinter's matplotlib RectangleSelector.

    Wraps ndnf_pipeline.plot.videography_plots (load_trial_video_data / preview_last_frame /
    render_trial_video) - see that module for what each rendering parameter controls.
    """

    title = "Generate Video"

    def __init__(self, app):
        self.app = app
        # per-camera-slot crop state (1 = primary camera, 2 = optional second camera)
        self._cam = {
            1: dict(crop=(0, 0, 0, 0), frame_shape=None, box_stream=None),
            2: dict(crop=(0, 0, 0, 0), frame_shape=None, box_stream=None),
        }
        self._preview_clims = None
        self._preview_clims_2 = None
        self._output_dir = None
        self._render_queue = queue.Queue()
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None

        self.inherited_label = pn.pane.Markdown(
            "*(switch to the Block Detail tab and pick a block first)*")
        sync_btn = pn.widgets.Button(name="Refresh from Block Detail", width=180)
        sync_btn.on_click(lambda e: self.sync_from_block_detail())
        selectors = pn.Row(pn.widgets.StaticText(value="From Block Detail tab:", margin=(8, 5)),
                            self.inherited_label, sync_btn)

        self.camera_select = pn.widgets.Select(name="Camera 1", options=[], width=180)
        self.use_camera_2 = pn.widgets.Checkbox(name="Add a 2nd camera (stacked below camera 1)", value=False)
        self.camera_2_select = pn.widgets.Select(name="Camera 2", options=[], width=180, disabled=True)
        self.use_camera_2.param.watch(lambda e: self._on_camera2_toggle(), "value")
        cameras_row = pn.Row(self.camera_select, self.use_camera_2, self.camera_2_select)

        self.pad_start = pn.widgets.FloatInput(name="pad start (s)", value=1.0, width=90)
        self.pad_end = pn.widgets.FloatInput(name="pad end (s)", value=1.0, width=90)
        self.tail = pn.widgets.FloatInput(name="tail (s)", value=5, width=80)
        self.force_limit = pn.widgets.FloatInput(name="force limit (g)", value=10.0, width=90)
        self.clim_low = pn.widgets.FloatInput(name="clim low%", value=0, width=70)
        self.clim_high = pn.widgets.FloatInput(name="clim high%", value=95, width=70)
        self.speed = pn.widgets.FloatInput(name="speed", value=1.0, width=70)
        self.fps = pn.widgets.IntInput(name="fps", value=20, width=70)
        self.lang = pn.widgets.Select(name="lang", options=["en", "hu"], value="en", width=70)
        params_row = pn.Row(self.pad_start, self.pad_end, self.tail, self.force_limit,
                             self.clim_low, self.clim_high, self.speed, self.fps, self.lang)

        # --- crop tool (camera 1) ---
        self.crop_label = {1: pn.widgets.StaticText(value="crop: left=0 right=0 top=0 bottom=0"),
                            2: pn.widgets.StaticText(value="crop: left=0 right=0 top=0 bottom=0")}
        self.crop_image_pane = {1: pn.pane.HoloViews(height=340, sizing_mode="stretch_width"),
                                 2: pn.pane.HoloViews(height=340, sizing_mode="stretch_width")}

        load_1 = pn.widgets.Button(name="Load frame", width=100)
        apply_1 = pn.widgets.Button(name="Apply crop", width=100)
        reset_1 = pn.widgets.Button(name="Reset crop", width=100)
        load_1.on_click(lambda e: self.load_frame_for_cropping(1))
        apply_1.on_click(lambda e: self.apply_crop(1))
        reset_1.on_click(lambda e: self.reset_crop(1))
        actions_cam1 = pn.Row(pn.widgets.StaticText(value="Camera 1 crop:", margin=(8, 5)),
                               load_1, apply_1, reset_1, self.crop_label[1])

        self.load_2_btn = pn.widgets.Button(name="Load frame", width=100, disabled=True)
        self.apply_2_btn = pn.widgets.Button(name="Apply crop", width=100, disabled=True)
        self.reset_2_btn = pn.widgets.Button(name="Reset crop", width=100, disabled=True)
        self.load_2_btn.on_click(lambda e: self.load_frame_for_cropping(2))
        self.apply_2_btn.on_click(lambda e: self.apply_crop(2))
        self.reset_2_btn.on_click(lambda e: self.reset_crop(2))
        actions_cam2 = pn.Row(pn.widgets.StaticText(value="Camera 2 crop:", margin=(8, 5)),
                               self.load_2_btn, self.apply_2_btn, self.reset_2_btn, self.crop_label[2])

        crop_body = pn.Row(
            pn.Column(pn.pane.Markdown("**Camera 1 crop** (drag the box edges)"), self.crop_image_pane[1]),
            pn.Column(pn.pane.Markdown("**Camera 2 crop** (drag the box edges)"), self.crop_image_pane[2]),
        )

        # --- preview / render ---
        self.preview_pane = make_plot_pane()
        self.preview_modal = pn.Modal(self.preview_pane, name="Video preview")

        preview_btn = pn.widgets.Button(name="Preview", width=90)
        preview_btn.on_click(lambda e: self.do_preview())

        self.output_dir_input = pn.widgets.TextInput(name="Output folder", width=340)
        cfg = dj_connection.load_gui_config()
        self.output_dir_input.value = cfg.get("video_output_dir") or str(Path.home())
        self.output_name_input = pn.widgets.TextInput(name="Output filename", value="video.mp4", width=260)

        self.render_btn = pn.widgets.Button(name="Render video", button_type="primary", width=120)
        self.render_btn.on_click(lambda e: self.start_render())
        self.render_progress = pn.indicators.Progress(active=False, visible=False, width=150,
                                                        bar_color="info")

        actions2 = pn.Row(preview_btn, self.output_dir_input, self.output_name_input,
                           self.render_btn, self.render_progress)

        self.panel = pn.Column(
            selectors, cameras_row, params_row,
            actions_cam1, actions_cam2, crop_body,
            actions2, self.preview_modal,
        )

    # --- inherit block/trials from the Block Detail tab ---
    def on_session_changed(self):
        self.reload_cameras()
        self.sync_from_block_detail()

    def reload_cameras(self):
        subject_id, session = self.app.get_selected_subject_session()
        if subject_id is None or session is None:
            self.camera_select.options = []
            self.camera_2_select.options = []
            return
        try:
            from ndnf_pipeline import videography
            cameras = sorted(set((videography.VideoRecording()
                                   & {"subject_id": subject_id, "session": session}).fetch("device").tolist()))
        except Exception as exc:
            self.app.report_error(exc)
            return
        self.camera_select.options = cameras
        if cameras:
            self.camera_select.value = cameras[0]
        self.camera_2_select.options = cameras
        if cameras:
            self.camera_2_select.value = cameras[1] if len(cameras) > 1 else cameras[0]

    def _on_camera2_toggle(self):
        enabled = self.use_camera_2.value
        self.camera_2_select.disabled = not enabled
        for btn in (self.load_2_btn, self.apply_2_btn, self.reset_2_btn):
            btn.disabled = not enabled

    def sync_from_block_detail(self):
        """Re-read the Block Detail tab's block + trial selection (none selected there = every
        trial in the block) and convert it into the block-relative trial_start/trial_end indices
        load_trial_video_data expects."""
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.app.block_detail_tab.block_select.value
        if subject_id is None or session is None or not block_str:
            self.inherited_label.object = "*(switch to the Block Detail tab and pick a block first)*"
            return
        try:
            all_trials = sorted((self.app.experiment.BehaviorTrial()
                                  & {"subject_id": subject_id, "session": session, "block": int(block_str)}
                                  ).fetch("trial").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        if not all_trials:
            self.inherited_label.object = f"Block {block_str}: no trials in this block"
            return
        selected = self.app.block_detail_tab.get_selected_trials()
        chosen = sorted(t for t in selected if t in all_trials) if selected else all_trials
        if not chosen:
            self.inherited_label.object = f"Block {block_str}: the selected trials aren't in this block"
            return
        self._inherited_block = int(block_str)
        self._inherited_trial_start = all_trials.index(min(chosen))
        self._inherited_trial_end = all_trials.index(max(chosen))
        if selected:
            self.inherited_label.object = (f"Block {block_str}: trials {min(chosen)}-{max(chosen)} "
                                            f"({len(chosen)} of {len(all_trials)} selected)")
        else:
            self.inherited_label.object = (f"Block {block_str}: all {len(all_trials)} trials "
                                            f"(trial numbers {all_trials[0]}-{all_trials[-1]})")

    # --- widget value parsing ---
    def _current_params(self):
        """Read + validate the inherited block/trial-range plus the camera selection(s). Returns
        load_trial_video_data kwargs, or None (after reporting the problem) if something
        required is missing/invalid."""
        subject_id, session = self.app.get_selected_subject_session()
        camera = self.camera_select.value
        camera_2 = self.camera_2_select.value if self.use_camera_2.value else None
        if subject_id is None or session is None:
            self.app.report_error(RuntimeError("Select a subject and session first."))
            return None
        if self._inherited_block is None:
            self.app.report_error(RuntimeError(
                "Pick a block (and, optionally, trials) on the Block Detail tab first, "
                "then click 'Refresh from Block Detail'."))
            return None
        if not camera:
            self.app.report_error(RuntimeError("No camera recordings available for this session."))
            return None
        if self.use_camera_2.value and not camera_2:
            self.app.report_error(RuntimeError("Pick a second camera, or uncheck 'Add a 2nd camera'."))
            return None
        return dict(subject_id=subject_id, session=session, block=self._inherited_block,
                    trial_start=self._inherited_trial_start, trial_end=self._inherited_trial_end,
                    camera_name=camera, camera_name_2=camera_2,
                    pad_start=self.pad_start.value or 1.0, pad_end=self.pad_end.value or 1.0)

    def _render_kwargs(self):
        return dict(
            clim_pct=(self.clim_low.value or 0, self.clim_high.value or 95),
            tail_s=self.tail.value or 5,
            force_axis_limit=self.force_limit.value or 10.0,
            lang=self.lang.value or "en",
        )

    # --- crop tool ---
    def _build_crop_view(self, slot, rgb, crop=(0, 0, 0, 0)):
        """rgb: HxWx3 float array in [0,1] (as returned by _prepare_frame). Displays it with an
        editable HoloViews rectangle seeded from `crop`, and wires up a fresh BoxEdit stream."""
        h, w = rgb.shape[:2]
        left, right, top, bottom = crop
        x0, x1 = left, max(w - right, left + 1)
        y0, y1 = bottom, max(h - top, bottom + 1)
        img = hv.RGB(rgb, bounds=(0, 0, w, h)).opts(
            responsive=True, aspect=w / h if h else 1, xaxis=None, yaxis=None,
            toolbar="above", active_tools=["box_edit"])
        rectangles = hv.Rectangles([(x0, y0, x1, y1)]).opts(
            fill_alpha=0.15, fill_color="red", line_color="red", line_width=2)
        box_stream = streams.BoxEdit(source=rectangles, num_objects=1)
        self._cam[slot]["box_stream"] = box_stream
        self._cam[slot]["frame_shape"] = (h, w)
        self.crop_image_pane[slot].object = img * rectangles

    def load_frame_for_cropping(self, slot):
        params = self._current_params()
        if params is None:
            return
        if slot == 2 and not params.get("camera_name_2"):
            self.app.report_error(RuntimeError("Pick a second camera first."))
            return
        self.app.status.object = f"Loading frame (camera {slot})..."
        try:
            from ndnf_pipeline.plot.videography_plots import load_trial_video_data, preview_last_frame, _prepare_frame
            data = load_trial_video_data(**params)
            render_kwargs = self._render_kwargs()
            dashboard_fig, last_frame, clims, last_frame_2, clims_2 = preview_last_frame(
                data, crop=(0, 0, 0, 0), crop_2=(0, 0, 0, 0) if params.get("camera_name_2") else None,
                clim_pct=render_kwargs["clim_pct"])
            plt.close(dashboard_fig)
            frame = last_frame if slot == 1 else last_frame_2
            frame_clims = clims if slot == 1 else clims_2
            rgb = _prepare_frame(frame, (0, 0, 0, 0), frame_clims)
            cam = self._cam[slot]
            self._build_crop_view(slot, rgb, crop=cam["crop"])
            self.app.status.object = f"Frame loaded (camera {slot}) - drag the crop box, then click 'Apply crop'."
        except Exception as exc:
            self.app.report_error(exc)
            self.app.status.object = "Failed to load frame - see error notification."

    def apply_crop(self, slot):
        cam = self._cam[slot]
        box_stream = cam["box_stream"]
        frame_shape = cam["frame_shape"]
        if box_stream is None or frame_shape is None:
            self.app.report_error(RuntimeError("Load a frame for cropping first."))
            return
        h, w = frame_shape
        data = box_stream.data
        if not data or not data.get("x0"):
            cam["crop"] = (0, 0, 0, 0)
        else:
            x_min, x_max = sorted((data["x0"][-1], data["x1"][-1]))
            y_min, y_max = sorted((data["y0"][-1], data["y1"][-1]))
            left = int(round(max(0, x_min)))
            right = int(round(max(0, w - x_max)))
            top = int(round(max(0, h - y_max)))
            bottom = int(round(max(0, y_min)))
            cam["crop"] = (left, right, top, bottom)
        left, right, top, bottom = cam["crop"]
        self.crop_label[slot].value = f"crop: left={left} right={right} top={top} bottom={bottom}"

    def reset_crop(self, slot):
        cam = self._cam[slot]
        cam["crop"] = (0, 0, 0, 0)
        self.crop_label[slot].value = "crop: left=0 right=0 top=0 bottom=0"
        if cam["frame_shape"] is not None:
            h, w = cam["frame_shape"]
            rectangles = hv.Rectangles([(0, 0, w, h)]).opts(
                fill_alpha=0.15, fill_color="red", line_color="red", line_width=2)
            box_stream = streams.BoxEdit(source=rectangles, num_objects=1)
            cam["box_stream"] = box_stream
            img = self.crop_image_pane[slot].object.get(0) if self.crop_image_pane[slot].object else None
            if img is not None:
                self.crop_image_pane[slot].object = img * rectangles

    # --- preview / render ---
    def do_preview(self):
        params = self._current_params()
        if params is None:
            return
        self.app.status.object = "Building preview..."
        try:
            from ndnf_pipeline.plot.videography_plots import load_trial_video_data, preview_last_frame
            data = load_trial_video_data(**params)
            render_kwargs = self._render_kwargs()
            crop_2 = self._cam[2]["crop"] if params.get("camera_name_2") else None
            fig, last_frame, clims, last_frame_2, clims_2 = preview_last_frame(
                data, crop=self._cam[1]["crop"], crop_2=crop_2, **render_kwargs)
            self._preview_clims = clims
            self._preview_clims_2 = clims_2
            show_figure(self.preview_pane, fig)
            self.preview_modal.show()
            self.app.status.object = "Preview ready."
        except Exception as exc:
            self.app.report_error(exc)
            self.app.status.object = "Preview failed - see error notification."

    def start_render(self):
        output_dir = (self.output_dir_input.value or "").strip()
        output_name = (self.output_name_input.value or "").strip()
        if not output_dir or not output_name:
            self.app.report_error(RuntimeError("Set an output folder and filename first."))
            return
        output_path = str(Path(output_dir) / output_name)
        params = self._current_params()
        if params is None:
            return
        render_kwargs = self._render_kwargs()
        speed = self.speed.value or 1.0
        fps = self.fps.value or 20
        crop = self._cam[1]["crop"]
        crop_2 = self._cam[2]["crop"] if params.get("camera_name_2") else None
        clims = self._preview_clims
        clims_2 = self._preview_clims_2 if params.get("camera_name_2") else None

        cfg = dj_connection.load_gui_config()
        cfg["video_output_dir"] = output_dir
        dj_connection.save_gui_config(cfg)

        self.render_btn.disabled = True
        self.render_progress.visible = True
        self.render_progress.active = True
        self.app.status.object = "Rendering video... this can take a while."

        def work():
            try:
                from ndnf_pipeline.plot.videography_plots import load_trial_video_data, render_trial_video
                data = load_trial_video_data(**params)
                render_trial_video(data, output_path, video_fps=fps, playback_speed=speed,
                                    crop=crop, crop_2=crop_2, clims=clims, clims_2=clims_2, **render_kwargs)
                self._render_queue.put(("done", output_path))
            except Exception as exc:
                self._render_queue.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()
        self._poll_handle = pn.state.add_periodic_callback(self._poll_render_queue, period=300)

    def _poll_render_queue(self):
        try:
            status, payload = self._render_queue.get_nowait()
        except queue.Empty:
            return
        self._poll_handle.stop()
        self.render_progress.active = False
        self.render_progress.visible = False
        self.render_btn.disabled = False
        if status == "done":
            self.app.status.object = f"Video saved: {payload}"
            pn.state.notifications.success(f"Video saved:\n{payload}", duration=0)
        else:
            self.app.status.object = "Render failed - see error notification."
            self.app.report_error(payload)


# ---------------------------------------------------------------------------
# main app
# ---------------------------------------------------------------------------

class BehaviorGUI:
    def __init__(self):
        self.lab = None
        self.experiment = None
        self._sessions_by_label = {}

        self.status = pn.pane.Markdown("Not connected.",
                                        styles={"font-size": "0.85em", "color": "var(--panel-secondary-color, #666)"})

        self._subject_guard = _Guard()
        self._session_guard = _Guard()
        self.subject_select = pn.widgets.Select(name="Subject", options=[], width=160)
        self.session_select = pn.widgets.Select(name="Session", options=[], width=280)
        self.subject_select.param.watch(self._subject_guard.wrap(lambda e: self.on_subject_selected()), "value")
        self.session_select.param.watch(self._session_guard.wrap(lambda e: self.on_session_selected()), "value")

        reload_btn = pn.widgets.Button(name="Reload subjects", width=130)
        reload_btn.on_click(lambda e: self.reload_subjects())
        settings_btn = pn.widgets.Button(name="DataJoint config...", width=150)
        settings_btn.on_click(lambda e: self.settings_modal.show())

        self.config_file_selector = pn.widgets.FileSelector(str(Path.home()), file_pattern="*.json")
        use_config_btn = pn.widgets.Button(name="Use this config", button_type="primary", width=140)
        use_config_btn.on_click(lambda e: self._use_selected_config())
        self.settings_modal = pn.Modal(
            pn.Column(
                pn.pane.Markdown("Select your DataJoint config file (dj_local_conf*.json):"),
                self.config_file_selector, use_config_btn),
            name="DataJoint connection")

        top_bar = pn.Row(self.subject_select, self.session_select, reload_btn,
                          pn.Spacer(sizing_mode="stretch_width"), settings_btn)

        # Epoch selection and trace filtering apply to force data, not to a specific tab, so one
        # shared instance of each lives here (above the tabs) and is read by both Session
        # Overview and Block Detail -- changing either is immediately reflected in both, rather
        # than each tab tracking its own, independently-driftable copy.
        self.epoch_selector = EpochSelector(on_change=self._on_shared_force_controls_changed)
        self.filter_controls = TraceFilterControls(on_change=self._on_shared_force_controls_changed)
        shared_force_controls = pn.Column(self.epoch_selector, self.filter_controls)

        self.session_overview_tab = SessionOverviewTab(self)
        self.block_detail_tab = BlockDetailTab(self)
        self.subject_trend_tab = SubjectTrendTab(self)
        self.trials_per_mouse_tab = TrialsPerMouseTab(self)
        self.water_restriction_tab = WaterRestrictionTab(self)
        self.video_generation_tab = VideoGenerationTab(self)
        self._tabs = (self.session_overview_tab, self.block_detail_tab, self.subject_trend_tab,
                      self.trials_per_mouse_tab, self.water_restriction_tab, self.video_generation_tab)

        self.tabs_widget = pn.Tabs(*[(t.title, t.panel) for t in self._tabs], dynamic=False)
        # the video tab inherits its block/trial selection from Block Detail, which can change
        # without a session change (different block, different trial selection) - resync on
        # every tab switch so it's never showing a stale block/trial range
        self.tabs_widget.param.watch(self._on_tab_changed, "active")

        self.layout = pn.Column(
            pn.pane.Markdown("# NDNF Behavior Pipeline Viewer"),
            top_bar,
            shared_force_controls,
            self.settings_modal,
            self.tabs_widget,
            self.status,
        )

        pn.state.onload(self.startup_connect)

    def _on_tab_changed(self, event):
        if self._tabs[event.new] is self.video_generation_tab:
            self.video_generation_tab.sync_from_block_detail()

    def _on_shared_force_controls_changed(self):
        self.session_overview_tab.refresh()
        self.block_detail_tab.refresh()

    def update_shared_sample_interval(self):
        """Refresh the shared filter controls' "(~N samples)" hints for whichever block is
        current: Block Detail's own selection if it has one, else the session's first block
        (so the hint is still useful from the Session Overview tab, which has no single
        block of its own)."""
        subject_id, session = self.get_selected_subject_session()
        if subject_id is None or session is None:
            self.filter_controls.set_sample_interval(None)
            return
        block_str = self.block_detail_tab.block_select.value
        if block_str:
            block = int(block_str)
        else:
            try:
                blocks = sorted((self.experiment.Block()
                                  & {"subject_id": subject_id, "session": session}).fetch("block").tolist())
            except Exception as exc:
                self.report_error(exc)
                self.filter_controls.set_sample_interval(None)
                return
            block = blocks[0] if blocks else None
        if block is None:
            self.filter_controls.set_sample_interval(None)
            return
        from ndnf_pipeline.plot.behavior_plots import estimate_force_sample_interval

        try:
            sample_interval_s = estimate_force_sample_interval(subject_id, session, block)
        except Exception as exc:
            self.report_error(exc)
            sample_interval_s = None
        self.filter_controls.set_sample_interval(sample_interval_s)

    # --- connection ---
    def startup_connect(self):
        dj_config_path = dj_connection.get_dj_config_path()
        if not dj_config_path or not Path(dj_config_path).exists():
            self.status.object = "No DataJoint config selected yet - use 'DataJoint config...' above."
            self.settings_modal.show()
            return
        self.connect(dj_config_path)

    def _use_selected_config(self):
        selected = self.config_file_selector.value
        if not selected:
            pn.state.notifications.warning("Pick a dj_local_conf*.json file first.")
            return
        path = selected[0]
        dj_connection.set_dj_config_path(path)
        self.settings_modal.hide()
        self.connect(path)

    def connect(self, dj_config_path):
        self.status.object = f"Connecting using {dj_config_path} ..."
        try:
            conn, lab, experiment = dj_connection.connect(dj_config_path)
        except Exception as exc:
            self.report_error(exc)
            self.status.object = "Connection failed. Use 'DataJoint config...' above."
            return
        self.lab = lab
        self.experiment = experiment
        self.status.object = f"Connected as {conn.get_user()}  |  config: {dj_config_path}"
        self.reload_subjects()

    # --- subject / session population ---
    def reload_subjects(self):
        if self.lab is None:
            return
        try:
            subject_ids = sorted(self.lab.Subject.fetch("subject_id").tolist())
        except Exception as exc:
            self.report_error(exc)
            return
        with self._subject_guard:
            self.subject_select.options = subject_ids
            if subject_ids and self.subject_select.value not in subject_ids:
                self.subject_select.value = self._most_recent_subject(subject_ids)
        if subject_ids:
            self.on_subject_selected()

    def _most_recent_subject(self, subject_ids):
        """Subject with the most recent session, falling back to the first subject alphabetically."""
        try:
            subs, dates, times = self.experiment.Session().fetch('subject_id', 'session_date', 'session_time')
        except Exception:
            subs = []
        if len(subs):
            latest_idx = max(range(len(subs)), key=lambda i: (dates[i], times[i]))
            return subs[latest_idx]
        return subject_ids[0]

    def on_subject_selected(self):
        self.populate_sessions()
        self._notify_tabs("on_subject_changed")

    def populate_sessions(self):
        subject_id = self.subject_select.value
        self._sessions_by_label = {}
        if not subject_id or self.experiment is None:
            with self._session_guard:
                self.session_select.options = []
            self._notify_tabs("on_session_changed")
            return
        try:
            sessions, dates = (self.experiment.Session() & {"subject_id": subject_id}).fetch(
                "session", "session_date", order_by="session DESC")
        except Exception as exc:
            self.report_error(exc)
            return
        labels = [f"{s} — {d}" for s, d in zip(sessions, dates)]
        self._sessions_by_label = dict(zip(labels, sessions.tolist()))
        with self._session_guard:
            self.session_select.options = labels
            if labels:
                self.session_select.value = labels[0]
        self.on_session_selected()

    def on_session_selected(self):
        self._notify_tabs("on_session_changed")

    def _notify_tabs(self, hook_name):
        for tab in self._tabs:
            hook = getattr(tab, hook_name, None)
            if hook is not None:
                hook()

    def get_selected_subject_session(self):
        subject_id = self.subject_select.value or None
        session = self._sessions_by_label.get(self.session_select.value)
        return subject_id, session

    # --- plotting / error helpers ---
    def run_plot(self, work_fn, pane):
        self.status.object = "Plotting..."
        try:
            fig = work_fn()
        except Exception as exc:
            self.report_error(exc)
            self.status.object = "Plot failed - see error notification."
            return
        show_figure(pane, fig)
        self.status.object = "Ready."

    def report_error(self, exc):
        traceback.print_exc()
        pn.state.notifications.error(str(exc), duration=0)


def create_app():
    gui = BehaviorGUI()
    return gui.layout


# `panel serve` (and `pn.serve` below) re-execute this module once per browser
# session, so module-level state here is already per-session - the standard
# Panel pattern is to build + .servable() at the top level when running under
# a server, and only fall back to pn.serve(...) for `python this_file.py`.
if pn.state.curdoc:
    create_app().servable(title="NDNF Behavior Pipeline Viewer")

if __name__ == "__main__":
    pn.serve(create_app, port=5006, address="0.0.0.0", show=False,
              allow_websocket_origin=["*"], title="NDNF Behavior Pipeline Viewer")
