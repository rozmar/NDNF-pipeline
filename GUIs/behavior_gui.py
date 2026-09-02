"""Tkinter GUI for browsing NDNF behavior pipeline data.

Connects to the DataJoint server (credentials come from a dj_local_conf.json
whose path is remembered in gui_config.json, see gui_config.example.json),
then lets you pick a subject and session and view plots for it across tabs.

Run with:  python GUIs/behavior_gui.py
"""
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("Agg")  # figures are built headless, then re-parented into a Tk canvas below
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import numpy as np

GUIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUIS_DIR.parent
sys.path.insert(0, str(GUIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import dj_connection


class UniformRangeControl(ttk.Frame):
    """Checkbox + numeric entry controlling a symmetric +/- force range, e.g. [-20,20,-20,20]."""

    def __init__(self, master, on_change, default_value=20.0, default_enabled=True):
        super().__init__(master)
        self.on_change = on_change
        self.enabled_var = tk.BooleanVar(value=default_enabled)
        self.value_var = tk.StringVar(value=str(default_value))
        ttk.Checkbutton(self, text="Uniform force range ±", variable=self.enabled_var,
                         command=self.on_change).pack(side="left")
        entry = ttk.Entry(self, textvariable=self.value_var, width=6)
        entry.pack(side="left", padx=(2, 2))
        entry.bind("<Return>", lambda e: self.on_change())
        entry.bind("<FocusOut>", lambda e: self.on_change())
        ttk.Label(self, text="g").pack(side="left")

    @property
    def enabled(self):
        return self.enabled_var.get()

    def get_range(self):
        try:
            value = float(self.value_var.get())
        except ValueError:
            value = 20.0
            self.value_var.set(str(value))
        return np.asarray([-1, 1, -1, 1]) * value


class PerfAxisControls(ttk.Frame):
    """Log/linear Y-axis toggle + rolling-mean window size, for a quiescence/response panel."""

    def __init__(self, master, on_change, default_window=10):
        super().__init__(master)
        self.on_change = on_change
        self.log_yscale = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Log Y axis", variable=self.log_yscale,
                         command=self.on_change).pack(side="left")
        ttk.Label(self, text="Rolling mean window:").pack(side="left", padx=(10, 0))
        self.smoothing_window = tk.StringVar(value=str(default_window))
        entry = ttk.Entry(self, textvariable=self.smoothing_window, width=4)
        entry.pack(side="left", padx=(2, 0))
        entry.bind("<Return>", lambda e: self.on_change())
        entry.bind("<FocusOut>", lambda e: self.on_change())
        self._default_window = default_window

    @property
    def log_scale(self):
        return self.log_yscale.get()

    def get_smoothing_window(self):
        try:
            window = int(self.smoothing_window.get())
            if window < 1:
                raise ValueError
        except ValueError:
            window = self._default_window
            self.smoothing_window.set(str(window))
        return window


class EpochSelector(ttk.Frame):
    """Checkboxes restricting the force histogram/trajectory (and force-vs-time) panels to
    samples within selected trial epoch(s): quiescence, response, reward consumption.

    All three checked (the default) is equivalent to no restriction at all.
    """
    EPOCHS = ('quiescence', 'response', 'reward')
    LABELS = {'quiescence': 'Quiescence', 'response': 'Response', 'reward': 'Reward'}

    def __init__(self, master, on_change, default_selected=EPOCHS):
        super().__init__(master)
        self.on_change = on_change
        ttk.Label(self, text="Epochs:").pack(side="left")
        self.vars = {}
        for epoch in self.EPOCHS:
            var = tk.BooleanVar(value=epoch in default_selected)
            self.vars[epoch] = var
            ttk.Checkbutton(self, text=self.LABELS[epoch], variable=var,
                             command=self.on_change).pack(side="left", padx=(4, 0))

    def get_selected_epochs(self):
        return tuple(epoch for epoch in self.EPOCHS if self.vars[epoch].get())


class TraceFilterControls(ttk.Frame):
    """Dropdown + millisecond-based parameter entries to smooth each trial's force trace
    before it feeds the histogram/trajectory/force-vs-time panels: none, boxcar (moving
    average), median, gaussian, or Savitzky-Golay (a local polynomial fit -- keeps peak shape
    better than the other three, at the cost of needing a window wide enough to fit the
    chosen order).

    Window/sigma are entered in milliseconds rather than samples, since the raw sample count
    for a given duration depends on the (block-specific) force trace sampling rate; a gray
    "(~N samples)" hint next to each entry shows what that currently works out to, updated via
    set_sample_interval() whenever the selected block's sampling rate becomes known. Only the
    parameter entry (entries) relevant to the selected method are enabled.
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

    def __init__(self, master, on_change):
        super().__init__(master)
        self.on_change = on_change
        self._sample_interval_s = None  # seconds/sample for the currently selected block
        ttk.Label(self, text="Trace filter:").pack(side="left")
        self.method_var = tk.StringVar(value=self.METHOD_LABELS['none'])
        self.method_cb = ttk.Combobox(self, textvariable=self.method_var, state="readonly", width=17,
                                       values=[self.METHOD_LABELS[m] for m in self.METHODS])
        self.method_cb.pack(side="left", padx=(4, 10))
        self.method_cb.bind("<<ComboboxSelected>>", lambda e: self._on_method_changed())

        ttk.Label(self, text="window (ms):").pack(side="left")
        self.window_ms_var = tk.StringVar(value="50")
        self.window_entry = ttk.Entry(self, textvariable=self.window_ms_var, width=6)
        self.window_entry.pack(side="left", padx=(2, 2))
        self.window_entry.bind("<Return>", lambda e: self._on_param_changed())
        self.window_entry.bind("<FocusOut>", lambda e: self._on_param_changed())
        self.window_hint_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.window_hint_var, foreground="gray").pack(side="left", padx=(0, 10))

        ttk.Label(self, text="sigma (ms):").pack(side="left")
        self.sigma_ms_var = tk.StringVar(value="20")
        self.sigma_entry = ttk.Entry(self, textvariable=self.sigma_ms_var, width=6)
        self.sigma_entry.pack(side="left", padx=(2, 2))
        self.sigma_entry.bind("<Return>", lambda e: self._on_param_changed())
        self.sigma_entry.bind("<FocusOut>", lambda e: self._on_param_changed())
        self.sigma_hint_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.sigma_hint_var, foreground="gray").pack(side="left", padx=(0, 10))

        ttk.Label(self, text="poly order:").pack(side="left")
        self.polyorder_var = tk.StringVar(value="3")
        self.polyorder_entry = ttk.Entry(self, textvariable=self.polyorder_var, width=4)
        self.polyorder_entry.pack(side="left", padx=(2, 0))
        self.polyorder_entry.bind("<Return>", lambda e: self.on_change())
        self.polyorder_entry.bind("<FocusOut>", lambda e: self.on_change())

        self._update_enabled_state()
        self._update_hints()

    def _method_key(self):
        label = self.method_var.get()
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
        self.window_entry.config(state="normal" if 'window' in relevant else "disabled")
        self.sigma_entry.config(state="normal" if 'sigma' in relevant else "disabled")
        self.polyorder_entry.config(state="normal" if 'polyorder' in relevant else "disabled")

    def set_sample_interval(self, sample_interval_s):
        """Called by the tab when the selected block's force-trace sampling rate becomes
        known (or unknown, with None), so the "(~N samples)" hints stay accurate."""
        self._sample_interval_s = sample_interval_s
        self._update_hints()

    def _update_hints(self):
        window_ms = self._safe_float(self.window_ms_var, 50.0)
        sigma_ms = self._safe_float(self.sigma_ms_var, 20.0)
        if self._sample_interval_s:
            # imported here (not at module load) so constructing this widget before a
            # DataJoint connection exists never touches the ndnf_pipeline package -- see
            # set_sample_interval(), which is the only caller that can make this branch true
            from ndnf_pipeline.plot.behavior_plots import ms_to_samples, ms_to_samples_float
            window_samples = ms_to_samples(window_ms, self._sample_interval_s)
            sigma_samples = ms_to_samples_float(sigma_ms, self._sample_interval_s)
            self.window_hint_var.set(f"(~{window_samples} samples)")
            self.sigma_hint_var.set(f"(~{sigma_samples:.1f} samples)")
        else:
            self.window_hint_var.set("(rate unknown)")
            self.sigma_hint_var.set("(rate unknown)")

    def _safe_float(self, var, default):
        try:
            return float(var.get())
        except ValueError:
            var.set(str(default))
            return default

    def get_filter_kwargs(self):
        method = self._method_key()
        window_ms = self._safe_float(self.window_ms_var, 50.0)
        sigma_ms = self._safe_float(self.sigma_ms_var, 20.0)
        try:
            polyorder = int(self.polyorder_var.get())
        except ValueError:
            polyorder = 3
            self.polyorder_var.set(str(polyorder))
        return dict(filter_method=method, filter_window_ms=window_ms, filter_sigma_ms=sigma_ms,
                    filter_polyorder=polyorder)


class PlotPanel(ttk.Frame):
    """A frame holding a matplotlib canvas + navigation toolbar; swap figures via show_figure()."""

    def __init__(self, master):
        super().__init__(master)
        self._canvas = None
        self._toolbar = None
        self._fig = None
        self._placeholder = ttk.Label(self, text="No plot yet.", anchor="center")
        self._placeholder.pack(fill="both", expand=True)

    def show_figure(self, fig):
        self.clear()
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.draw()
        self._toolbar = NavigationToolbar2Tk(self._canvas, self)
        self._toolbar.update()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        if self._placeholder is not None:
            self._placeholder.destroy()
            self._placeholder = None


class SessionOverviewTab(ttk.Frame):
    """The 'all session' plot: every block of the selected session, side by side."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        self.range_control = UniformRangeControl(controls, on_change=self.refresh)
        self.range_control.pack(side="left")
        self.perf_controls = PerfAxisControls(controls, on_change=self.refresh)
        self.perf_controls.pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Refresh", command=lambda: self.refresh(force=True)).pack(side="left", padx=(10, 0))
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto refresh", variable=self.auto_refresh_var).pack(side="left", padx=(6, 0))
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_session_changed(self):
        self.refresh()

    def refresh(self, force=False):
        if not force and not self.auto_refresh_var.get():
            return
        subject_id = self.app.subject_var.get() or None
        sessions = self.app.get_selected_sessions()
        if subject_id is None or not sessions:
            self.panel.clear()
            return
        from ndnf_pipeline.plot.behavior_plots import plot_session_blocks_overview

        def work():
            return plot_session_blocks_overview(
                subject_id, sessions,
                force_uniform_range=self.range_control.enabled,
                uniform_force_range=self.range_control.get_range(),
                perf_log_yscale=self.perf_controls.log_scale,
                perf_smoothing_window=self.perf_controls.get_smoothing_window(),
                epochs=self.app.epoch_selector.get_selected_epochs(),
                **self.app.filter_controls.get_filter_kwargs())

        self.app.run_plot(work, self.panel)


class BlockDetailTab(ttk.Frame):
    """4-panel detail figure for a single block of the selected session.

    A trial multi-select restricts which trials feed the force-distribution and
    trajectory panels; the performance (quiescence/response) panel always shows every
    trial in the block regardless of that selection.
    """

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._blocks_by_label = {}  # block_var label -> (session, block)
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Label(controls, text="Block:").pack(side="left")
        self.block_var = tk.StringVar()
        self.block_cb = ttk.Combobox(controls, textvariable=self.block_var, state="readonly", width=10)
        self.block_cb.pack(side="left", padx=(4, 10))
        self.block_cb.bind("<<ComboboxSelected>>", lambda e: self.on_block_selected())
        self.range_control = UniformRangeControl(controls, on_change=self.refresh)
        self.range_control.pack(side="left", padx=(10, 0))
        self.perf_controls = PerfAxisControls(controls, on_change=self.refresh)
        self.perf_controls.pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Refresh", command=lambda: self.refresh(force=True)).pack(side="left", padx=(10, 0))
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto refresh", variable=self.auto_refresh_var).pack(side="left", padx=(6, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(5, 0), pady=5)
        ttk.Label(left, text="Trials for 2D hist\n& trajectories\n(none = all):", justify="left").pack(anchor="w")
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="y", expand=True)
        self.trial_listbox = tk.Listbox(list_frame, selectmode="extended", exportselection=False,
                                         width=8, height=22)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.trial_listbox.yview)
        self.trial_listbox.configure(yscrollcommand=scrollbar.set)
        self.trial_listbox.pack(side="left", fill="y")
        scrollbar.pack(side="left", fill="y")
        self.trial_listbox.bind("<<ListboxSelect>>", lambda e: self.refresh())
        button_row = ttk.Frame(left)
        button_row.pack(fill="x", pady=(5, 0))
        ttk.Button(button_row, text="All", command=self.select_all_trials).pack(side="left")
        ttk.Button(button_row, text="None", command=self.select_no_trials).pack(side="left", padx=(4, 0))

        self.panel = PlotPanel(body)
        self.panel.pack(side="left", fill="both", expand=True)

    def on_session_changed(self):
        subject_id = self.app.subject_var.get() or None
        sessions = self.app.get_selected_sessions()
        self.block_cb["values"] = []
        self.block_var.set("")
        self._blocks_by_label = {}
        if subject_id is None or not sessions:
            self.trial_listbox.delete(0, "end")
            self.panel.clear()
            self.app.update_shared_sample_interval()
            return
        multi_session = len(sessions) > 1
        try:
            pairs = []
            for session in sessions:
                blocks = sorted((self.app.experiment.Block()
                                  & {"subject_id": subject_id, "session": session}).fetch("block").tolist())
                pairs.extend((session, block) for block in blocks)
        except Exception as exc:
            self.app.report_error(exc)
            return
        labels = [f"s{session} b{block}" if multi_session else str(block) for session, block in pairs]
        self._blocks_by_label = dict(zip(labels, pairs))
        self.block_cb["values"] = labels
        if labels:
            self.block_var.set(labels[0])
            self.on_block_selected()
        else:
            self.trial_listbox.delete(0, "end")
            self.panel.clear()
            self.app.update_shared_sample_interval()

    def on_block_selected(self):
        self.reload_trials()
        self.app.update_shared_sample_interval()
        self.refresh()

    def get_selected_session_block(self):
        """(session, block) for whatever's currently picked in block_var, or (None, None)."""
        return self._blocks_by_label.get(self.block_var.get(), (None, None))

    def reload_trials(self):
        subject_id = self.app.subject_var.get() or None
        session, block = self.get_selected_session_block()
        self.trial_listbox.delete(0, "end")
        if subject_id is None or session is None:
            return
        key = {"subject_id": subject_id, "session": session, "block": block}
        try:
            trials = sorted((self.app.experiment.BehaviorTrial() & key).fetch("trial").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        for t in trials:
            self.trial_listbox.insert("end", t)

    def select_all_trials(self):
        self.trial_listbox.selection_set(0, "end")
        self.refresh()

    def select_no_trials(self):
        self.trial_listbox.selection_clear(0, "end")
        self.refresh()

    def get_selected_trials(self):
        return [int(self.trial_listbox.get(i)) for i in self.trial_listbox.curselection()]

    def refresh(self, force=False):
        if not force and not self.auto_refresh_var.get():
            return
        subject_id = self.app.subject_var.get() or None
        session, block = self.get_selected_session_block()
        if subject_id is None or session is None:
            return
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

        self.app.run_plot(work, self.panel)


class SubjectTrendTab(ttk.Frame):
    """Trial count / session length / hit rate across all sessions of the selected subject."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Button(controls, text="Refresh", command=lambda: self.refresh(force=True)).pack(side="left")
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto refresh", variable=self.auto_refresh_var).pack(side="left", padx=(6, 0))
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_subject_changed(self):
        self.refresh()

    def refresh(self, force=False):
        if not force and not self.auto_refresh_var.get():
            return
        subject_id, _ = self.app.get_selected_subject_session()
        if subject_id is None:
            self.panel.clear()
            return
        from ndnf_pipeline.plot.behavior_plots import plot_subject_behavior_trend

        def work():
            return plot_subject_behavior_trend(self.app.experiment, subject_id)

        self.app.run_plot(work, self.panel)


class TrialsPerMouseTab(ttk.Frame):
    """Trials-per-mouse bar chart plus overlaid per-session trends, for a selectable set of mice."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._mice_loaded = False

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(5, 0), pady=5)
        ttk.Label(left, text="Mice to show:").pack(anchor="w")
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="y", expand=True)
        self.mouse_listbox = tk.Listbox(list_frame, selectmode="extended", exportselection=False,
                                         width=16, height=22)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.mouse_listbox.yview)
        self.mouse_listbox.configure(yscrollcommand=scrollbar.set)
        self.mouse_listbox.pack(side="left", fill="y")
        scrollbar.pack(side="left", fill="y")
        self.mouse_listbox.bind("<<ListboxSelect>>", lambda e: self.refresh())

        button_row = ttk.Frame(left)
        button_row.pack(fill="x", pady=(5, 0))
        ttk.Button(button_row, text="All", command=self.select_all).pack(side="left")
        ttk.Button(button_row, text="None", command=self.select_none).pack(side="left", padx=(4, 0))
        ttk.Button(left, text="Reload mouse list", command=self.reload_mice).pack(fill="x", pady=(5, 0))
        refresh_row = ttk.Frame(left)
        refresh_row.pack(fill="x", pady=(5, 0))
        ttk.Button(refresh_row, text="Refresh", command=lambda: self.refresh(force=True)).pack(side="left")
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(refresh_row, text="Auto refresh", variable=self.auto_refresh_var).pack(side="left", padx=(6, 0))

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        bar_frame = ttk.LabelFrame(right, text="Total trials (selected mice, or all mice if none selected)")
        bar_frame.pack(fill="both", expand=True, padx=5, pady=(5, 2))
        self.bar_panel = PlotPanel(bar_frame)
        self.bar_panel.pack(fill="both", expand=True)
        trend_frame = ttk.LabelFrame(right, text="Session trends (selected mice, color-coded)")
        trend_frame.pack(fill="both", expand=True, padx=5, pady=(2, 5))
        self.trend_panel = PlotPanel(trend_frame)
        self.trend_panel.pack(fill="both", expand=True)

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
        self.mouse_listbox.delete(0, "end")
        for i, m in enumerate(mice_with_trials):
            self.mouse_listbox.insert("end", m)
            if m in previously_selected:
                self.mouse_listbox.selection_set(i)
        self._mice_loaded = True
        self.refresh()

    def select_all(self):
        self.mouse_listbox.selection_set(0, "end")
        self.refresh()

    def select_none(self):
        self.mouse_listbox.selection_clear(0, "end")
        self.refresh()

    def get_selected_mice(self):
        return [self.mouse_listbox.get(i) for i in self.mouse_listbox.curselection()]

    def refresh(self, force=False):
        if not force and not self.auto_refresh_var.get():
            return
        if self.app.lab is None or self.app.experiment is None:
            return
        selected = self.get_selected_mice()
        from ndnf_pipeline.plot.behavior_plots import plot_trials_per_mouse, plot_subject_behavior_trends

        def work_bar():
            return plot_trials_per_mouse(self.app.lab, self.app.experiment, subject_ids=selected or None)

        def work_trend():
            return plot_subject_behavior_trends(self.app.experiment, selected)

        self.app.run_plot(work_bar, self.bar_panel)
        self.app.run_plot(work_trend, self.trend_panel)


class WaterRestrictionTab(ttk.Frame):
    """Weight / water-consumed logs for every mouse on water restriction; selected subject is highlighted."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Button(controls, text="Refresh", command=lambda: self.refresh(force=True)).pack(side="left")
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto refresh", variable=self.auto_refresh_var).pack(side="left", padx=(6, 0))
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_subject_changed(self):
        self.refresh()

    def refresh(self, force=False):
        if not force and not self.auto_refresh_var.get():
            return
        subject_id, _ = self.app.get_selected_subject_session()
        from ndnf_pipeline.plot.behavior_plots import plot_water_restriction_overview

        def work():
            return plot_water_restriction_overview(self.app.lab, highlight_subject_id=subject_id)

        self.app.run_plot(work, self.panel)


class VideoGenerationTab(ttk.Frame):
    """Generate an annotated trial-range video for one block, with an interactive crop tool.

    The block and trial range are inherited from the Block Detail tab (its block dropdown and
    its trial multi-select — "none selected" there means "every trial in the block") rather than
    picked here, so the video always matches whatever block/trials you're already looking at.

    Wraps ndnf_pipeline.plot.videography_plots (load_trial_video_data / preview_last_frame /
    render_trial_video) — see that module for what each rendering parameter controls.
    """

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        # per-camera-slot crop state (1 = primary camera, 2 = optional second camera)
        self._cam = {
            1: dict(crop=(0, 0, 0, 0), frame_shape=None, selector=None),
            2: dict(crop=(0, 0, 0, 0), frame_shape=None, selector=None),
        }
        self._preview_clims = None
        self._preview_clims_2 = None
        self._output_path = None
        self._render_queue = queue.Queue()
        self._render_progress_state = (0, 0)  # (n_done, n_total), written by the render thread
        self._inherited_session = None
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None

        selectors = ttk.Frame(self)
        selectors.pack(fill="x", padx=5, pady=(5, 0))
        ttk.Label(selectors, text="From Block Detail tab:").pack(side="left")
        self.inherited_label_var = tk.StringVar(value="(switch to the Block Detail tab and pick a block first)")
        ttk.Label(selectors, textvariable=self.inherited_label_var, foreground="gray").pack(side="left", padx=(4, 10))
        ttk.Button(selectors, text="Refresh from Block Detail", command=self.sync_from_block_detail).pack(side="left")

        cameras_row = ttk.Frame(self)
        cameras_row.pack(fill="x", padx=5, pady=(2, 0))
        ttk.Label(cameras_row, text="Camera 1:").pack(side="left")
        self.camera_var = tk.StringVar()
        self.camera_cb = ttk.Combobox(cameras_row, textvariable=self.camera_var, state="readonly", width=14)
        self.camera_cb.pack(side="left", padx=(4, 10))

        self.use_camera_2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(cameras_row, text="Add a 2nd camera (stacked below camera 1)",
                        variable=self.use_camera_2, command=self._on_camera2_toggle).pack(side="left", padx=(10, 0))
        ttk.Label(cameras_row, text="Camera 2:").pack(side="left", padx=(10, 0))
        self.camera_2_var = tk.StringVar()
        self.camera_2_cb = ttk.Combobox(cameras_row, textvariable=self.camera_2_var, state="disabled", width=14)
        self.camera_2_cb.pack(side="left", padx=(4, 10))

        params = ttk.Frame(self)
        params.pack(fill="x", padx=5, pady=(4, 0))
        self.pad_start_var = tk.StringVar(value="1.0")
        self.pad_end_var = tk.StringVar(value="1.0")
        self.tail_var = tk.StringVar(value="5")
        self.force_limit_var = tk.StringVar(value="10.0")
        self.clim_low_var = tk.StringVar(value="0")
        self.clim_high_var = tk.StringVar(value="95")
        self.speed_var = tk.StringVar(value="1.0")
        self.fps_var = tk.StringVar(value="20")
        self.lang_var = tk.StringVar(value="en")
        # imported here (not at module load) to match this tab's other lazy ndnf_pipeline
        # imports below - just a plain dict, so it doesn't actually need a DataJoint connection
        from ndnf_pipeline.plot.videography_plots import VIDEO_QUALITY_PRESETS, VIDEO_QUALITY_DEFAULT
        self._quality_presets = VIDEO_QUALITY_PRESETS
        self.quality_var = tk.StringVar(value=VIDEO_QUALITY_DEFAULT)

        def _labeled_entry(parent, label, var, width=5):
            ttk.Label(parent, text=label).pack(side="left")
            ttk.Entry(parent, textvariable=var, width=width).pack(side="left", padx=(2, 10))

        _labeled_entry(params, "pad start (s):", self.pad_start_var)
        _labeled_entry(params, "pad end (s):", self.pad_end_var)
        _labeled_entry(params, "tail (s):", self.tail_var)
        _labeled_entry(params, "force limit (g):", self.force_limit_var)
        _labeled_entry(params, "clim low%:", self.clim_low_var, width=4)
        _labeled_entry(params, "high%:", self.clim_high_var, width=4)
        _labeled_entry(params, "speed:", self.speed_var, width=4)
        _labeled_entry(params, "fps:", self.fps_var, width=4)
        ttk.Label(params, text="lang:").pack(side="left")
        ttk.Combobox(params, textvariable=self.lang_var, state="readonly", width=4,
                     values=["en", "hu"]).pack(side="left", padx=(2, 10))
        ttk.Label(params, text="quality:").pack(side="left")
        ttk.Combobox(params, textvariable=self.quality_var, state="readonly", width=8,
                     values=list(self._quality_presets.keys())).pack(side="left", padx=(2, 0))

        actions_cam1 = ttk.Frame(self)
        actions_cam1.pack(fill="x", padx=5, pady=(4, 2))
        ttk.Label(actions_cam1, text="Camera 1 crop:").pack(side="left")
        ttk.Button(actions_cam1, text="Load frame", command=lambda: self.load_frame_for_cropping(1)).pack(side="left", padx=(6, 0))
        ttk.Button(actions_cam1, text="Apply crop", command=lambda: self.apply_crop(1)).pack(side="left", padx=(6, 0))
        ttk.Button(actions_cam1, text="Reset crop", command=lambda: self.reset_crop(1)).pack(side="left", padx=(6, 0))
        self.crop_label_var = tk.StringVar(value="crop: left=0 right=0 top=0 bottom=0")
        ttk.Label(actions_cam1, textvariable=self.crop_label_var).pack(side="left", padx=(10, 0))

        actions_cam2 = ttk.Frame(self)
        actions_cam2.pack(fill="x", padx=5, pady=(0, 2))
        ttk.Label(actions_cam2, text="Camera 2 crop:").pack(side="left")
        self.load_frame_2_btn = ttk.Button(actions_cam2, text="Load frame", state="disabled",
                                            command=lambda: self.load_frame_for_cropping(2))
        self.load_frame_2_btn.pack(side="left", padx=(6, 0))
        self.apply_crop_2_btn = ttk.Button(actions_cam2, text="Apply crop", state="disabled",
                                            command=lambda: self.apply_crop(2))
        self.apply_crop_2_btn.pack(side="left", padx=(6, 0))
        self.reset_crop_2_btn = ttk.Button(actions_cam2, text="Reset crop", state="disabled",
                                            command=lambda: self.reset_crop(2))
        self.reset_crop_2_btn.pack(side="left", padx=(6, 0))
        self.crop_label_var_2 = tk.StringVar(value="crop: left=0 right=0 top=0 bottom=0")
        ttk.Label(actions_cam2, textvariable=self.crop_label_var_2).pack(side="left", padx=(10, 0))

        actions2 = ttk.Frame(self)
        actions2.pack(fill="x", padx=5, pady=(2, 4))
        ttk.Button(actions2, text="Preview", command=self.do_preview).pack(side="left")
        ttk.Button(actions2, text="Choose output file...", command=self.choose_output_path).pack(side="left", padx=(6, 0))
        self.output_label_var = tk.StringVar(value="(no output file chosen)")
        ttk.Label(actions2, textvariable=self.output_label_var, foreground="gray").pack(side="left", padx=(6, 0))
        self.render_button = ttk.Button(actions2, text="Render video", command=self.start_render)
        self.render_button.pack(side="left", padx=(10, 0))
        self.render_progress = ttk.Progressbar(actions2, mode="indeterminate", length=120)
        self.render_progress.pack(side="left", padx=(10, 0))

        # preview opens in its own pop-out window (see _show_preview_window) instead of an
        # embedded panel, so it renders at the dashboard figure's true aspect ratio instead of
        # being stretched to fill whatever space an embedded panel happens to have
        self._preview_window = None
        self._preview_fig = None

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        crop_frame = ttk.LabelFrame(body, text="Camera 1 crop selection (drag the box edges)")
        crop_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.crop_panel = PlotPanel(crop_frame)
        self.crop_panel.pack(fill="both", expand=True)
        crop_frame_2 = ttk.LabelFrame(body, text="Camera 2 crop selection (drag the box edges)")
        crop_frame_2.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.crop_panel_2 = PlotPanel(crop_frame_2)
        self.crop_panel_2.pack(fill="both", expand=True)
        self._crop_panels = {1: self.crop_panel, 2: self.crop_panel_2}
        self._crop_label_vars = {1: self.crop_label_var, 2: self.crop_label_var_2}

    # --- inherit block/trials from the Block Detail tab ---
    def on_session_changed(self):
        # sync first so _inherited_session reflects the actual session of whatever block is
        # selected in Block Detail (which may not be the app's "primary" selected session, once
        # more than one session is selected) - reload_cameras then uses that.
        self.sync_from_block_detail()
        self.reload_cameras()

    def reload_cameras(self):
        subject_id = self.app.subject_var.get() or None
        session = self._inherited_session
        if session is None:
            _, session = self.app.get_selected_subject_session()
        self.camera_cb["values"] = []
        self.camera_var.set("")
        self.camera_2_cb["values"] = []
        self.camera_2_var.set("")
        if subject_id is None or session is None:
            return
        try:
            from ndnf_pipeline import videography
            cameras = sorted(set((videography.VideoRecording()
                                   & {"subject_id": subject_id, "session": session}).fetch("device").tolist()))
        except Exception as exc:
            self.app.report_error(exc)
            return
        self.camera_cb["values"] = cameras
        self.camera_2_cb["values"] = cameras
        if cameras:
            self.camera_var.set(cameras[0])
            self.camera_2_var.set(cameras[1] if len(cameras) > 1 else cameras[0])

    def _on_camera2_toggle(self):
        enabled = self.use_camera_2.get()
        self.camera_2_cb.config(state="readonly" if enabled else "disabled")
        state = "normal" if enabled else "disabled"
        self.load_frame_2_btn.config(state=state)
        self.apply_crop_2_btn.config(state=state)
        self.reset_crop_2_btn.config(state=state)

    def sync_from_block_detail(self):
        """Re-read the Block Detail tab's block + trial selection (none selected there = every
        trial in the block) and convert it into the block-relative trial_start/trial_end indices
        load_trial_video_data expects. The session comes from whichever (session, block) Block
        Detail currently has picked - not necessarily the app's "primary" selected session, once
        more than one session is selected there."""
        self._inherited_session = None
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None
        subject_id = self.app.subject_var.get() or None
        session, block = self.app.block_detail_tab.get_selected_session_block()
        if subject_id is None or session is None:
            self.inherited_label_var.set("(switch to the Block Detail tab and pick a block first)")
            return
        block_label = self.app.block_detail_tab.block_var.get()
        try:
            all_trials = sorted((self.app.experiment.BehaviorTrial()
                                  & {"subject_id": subject_id, "session": session, "block": block}
                                  ).fetch("trial").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        if not all_trials:
            self.inherited_label_var.set(f"Block {block_label}: no trials in this block")
            return
        selected = self.app.block_detail_tab.get_selected_trials()
        chosen = sorted(t for t in selected if t in all_trials) if selected else all_trials
        if not chosen:
            self.inherited_label_var.set(f"Block {block_label}: the selected trials aren't in this block")
            return
        self._inherited_session = session
        self._inherited_block = block
        self._inherited_trial_start = all_trials.index(min(chosen))
        self._inherited_trial_end = all_trials.index(max(chosen))
        if selected:
            self.inherited_label_var.set(f"Block {block_label}: trials {min(chosen)}-{max(chosen)} "
                                          f"({len(chosen)} of {len(all_trials)} selected)")
        else:
            self.inherited_label_var.set(f"Block {block_label}: all {len(all_trials)} trials "
                                          f"(trial numbers {all_trials[0]}-{all_trials[-1]})")

    # --- widget value parsing ---
    def _read_float(self, var, default):
        try:
            return float(var.get())
        except ValueError:
            var.set(str(default))
            return default

    def _read_int(self, var, default):
        try:
            return int(var.get())
        except ValueError:
            var.set(str(default))
            return default

    def _current_params(self):
        """Read + validate the inherited block/trial-range plus the camera selection(s). Returns
        load_trial_video_data kwargs, or None (after reporting the problem) if something
        required is missing/invalid."""
        subject_id = self.app.subject_var.get() or None
        session = self._inherited_session
        camera = self.camera_var.get()
        camera_2 = self.camera_2_var.get() if self.use_camera_2.get() else None
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
        if self.use_camera_2.get() and not camera_2:
            self.app.report_error(RuntimeError("Pick a second camera, or uncheck 'Add a 2nd camera'."))
            return None
        return dict(subject_id=subject_id, session=session, block=self._inherited_block,
                    trial_start=self._inherited_trial_start, trial_end=self._inherited_trial_end,
                    camera_name=camera, camera_name_2=camera_2,
                    pad_start=self._read_float(self.pad_start_var, 1.0),
                    pad_end=self._read_float(self.pad_end_var, 1.0),
                    # same shared trace-filter control used by Session Overview / Block Detail,
                    # so the video's trace is smoothed the same way as whatever you were just
                    # looking at there - see load_trial_video_data's filter_method docs
                    **self.app.filter_controls.get_filter_kwargs())

    def _quality_preset(self):
        return self._quality_presets.get(self.quality_var.get(), self._quality_presets['high'])

    def _render_kwargs(self):
        return dict(
            clim_pct=(self._read_float(self.clim_low_var, 0), self._read_float(self.clim_high_var, 95)),
            tail_s=self._read_float(self.tail_var, 5),
            force_axis_limit=self._read_float(self.force_limit_var, 10.0),
            lang=self.lang_var.get() or "en",
            dpi=self._quality_preset()['dpi'],
        )

    # --- actions ---
    def load_frame_for_cropping(self, slot):
        params = self._current_params()
        if params is None:
            return
        if slot == 2 and not params.get("camera_name_2"):
            self.app.report_error(RuntimeError("Pick a second camera first."))
            return
        self.app.status_var.set(f"Loading frame (camera {slot})...")
        self.app.update_idletasks()
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
            h, w = frame.shape[:2]
            fig, ax = plt.subplots(figsize=(7, max(4.0, 7 * h / w)))
            ax.imshow(rgb)
            ax.set_title("Drag the box edges/corners to set the crop, then click 'Apply crop'", fontsize=9)
            cam = self._cam[slot]
            left, right, top, bottom = cam["crop"]
            selector = RectangleSelector(ax, onselect=lambda *a: None, useblit=False, interactive=True,
                                          button=[1], minspanx=5, minspany=5, spancoords="pixels",
                                          drag_from_anywhere=True)
            selector.extents = (left, max(w - right, left + 1), top, max(h - bottom, top + 1))
            cam["selector"] = selector
            cam["frame_shape"] = (h, w)
            self._crop_panels[slot].show_figure(fig)
            self.app.status_var.set(f"Frame loaded (camera {slot}) - adjust the crop box, then click 'Apply crop'.")
        except Exception as exc:
            self.app.report_error(exc)
            self.app.status_var.set("Failed to load frame - see error dialog.")

    def apply_crop(self, slot):
        cam = self._cam[slot]
        if cam["selector"] is None or cam["frame_shape"] is None:
            self.app.report_error(RuntimeError("Load a frame for cropping first."))
            return
        xmin, xmax, ymin, ymax = cam["selector"].extents
        h, w = cam["frame_shape"]
        left = int(round(max(0, xmin)))
        right = int(round(max(0, w - xmax)))
        top = int(round(max(0, ymin)))
        bottom = int(round(max(0, h - ymax)))
        cam["crop"] = (left, right, top, bottom)
        self._crop_label_vars[slot].set(f"crop: left={left} right={right} top={top} bottom={bottom}")

    def reset_crop(self, slot):
        cam = self._cam[slot]
        cam["crop"] = (0, 0, 0, 0)
        self._crop_label_vars[slot].set("crop: left=0 right=0 top=0 bottom=0")
        if cam["selector"] is not None and cam["frame_shape"] is not None:
            h, w = cam["frame_shape"]
            cam["selector"].extents = (0, w, 0, h)

    def do_preview(self):
        params = self._current_params()
        if params is None:
            return
        self.app.status_var.set("Building preview...")
        self.app.update_idletasks()
        try:
            from ndnf_pipeline.plot.videography_plots import load_trial_video_data, preview_last_frame
            data = load_trial_video_data(**params)
            render_kwargs = self._render_kwargs()
            crop_2 = self._cam[2]["crop"] if params.get("camera_name_2") else None
            fig, last_frame, clims, last_frame_2, clims_2 = preview_last_frame(
                data, crop=self._cam[1]["crop"], crop_2=crop_2, **render_kwargs)
            self._preview_clims = clims
            self._preview_clims_2 = clims_2
            self._show_preview_window(fig)
            self.app.status_var.set("Preview ready.")
        except Exception as exc:
            self.app.report_error(exc)
            self.app.status_var.set("Preview failed - see error dialog.")

    def _show_preview_window(self, fig):
        """Show fig in its own Toplevel, sized to its true aspect ratio (capped so it fits on
        screen) and non-resizable, so the dashboard is never stretched out of proportion."""
        if self._preview_window is not None and self._preview_window.winfo_exists():
            self._preview_window.destroy()
        if self._preview_fig is not None:
            plt.close(self._preview_fig)

        fig_w_in, fig_h_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        native_w, native_h = fig_w_in * dpi, fig_h_in * dpi
        max_w, max_h = 1400, 900
        scale = min(1.0, max_w / native_w, max_h / native_h)
        win_w, win_h = int(native_w * scale), int(native_h * scale + 40)  # + toolbar height

        win = tk.Toplevel(self.app)
        win.title("Video preview")
        win.geometry(f"{win_w}x{win_h}")
        win.resizable(False, False)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._preview_window = win
        self._preview_fig = fig

    def choose_output_path(self):
        subject_id = self.app.subject_var.get() or None
        session = self._inherited_session
        camera = self.camera_var.get()
        camera_suffix = camera
        if self.use_camera_2.get() and self.camera_2_var.get():
            camera_suffix = f"{camera}+{self.camera_2_var.get()}"
        quality = self.quality_var.get()
        default_name = f"video_{quality}.mp4"
        if subject_id and session is not None and self._inherited_block is not None and camera:
            default_name = (f"{subject_id}_s{session}_b{self._inherited_block}_"
                             f"t{self._inherited_trial_start}-{self._inherited_trial_end}_"
                             f"{camera_suffix}_{quality}.mp4")
        cfg = dj_connection.load_gui_config()
        initial_dir = cfg.get("video_output_dir") or str(Path.home())
        path = filedialog.asksaveasfilename(
            title="Save video as", defaultextension=".mp4", initialfile=default_name,
            initialdir=initial_dir, filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")])
        if not path:
            return
        self._output_path = path
        self.output_label_var.set(path)
        cfg["video_output_dir"] = str(Path(path).parent)
        dj_connection.save_gui_config(cfg)

    def start_render(self):
        if self._output_path is None:
            self.app.report_error(RuntimeError("Choose an output file first."))
            return
        params = self._current_params()
        if params is None:
            return
        render_kwargs = self._render_kwargs()
        bitrate = self._quality_preset()['bitrate']
        speed = self._read_float(self.speed_var, 1.0)
        fps = self._read_int(self.fps_var, 20)
        crop = self._cam[1]["crop"]
        crop_2 = self._cam[2]["crop"] if params.get("camera_name_2") else None
        clims = self._preview_clims
        clims_2 = self._preview_clims_2 if params.get("camera_name_2") else None
        output_path = self._output_path

        self.render_button.config(state="disabled")
        self.render_progress.config(mode="indeterminate")
        self.render_progress.start(50)
        self._render_progress_state = (0, 0)
        self.app.status_var.set("Rendering video... this can take a while.")

        def work():
            try:
                from ndnf_pipeline.plot.videography_plots import (
                    load_trial_video_data, render_trial_video, save_render_params)
                data = load_trial_video_data(**params)
                # called from the render worker thread, not the Tk main loop - so it only
                # stashes the numbers; _poll_render_queue (running via after()) is what
                # actually touches the render_progress widget
                _, used_clims, used_clims_2 = render_trial_video(
                    data, output_path, video_fps=fps, playback_speed=speed,
                    crop=crop, crop_2=crop_2, clims=clims, clims_2=clims_2, bitrate=bitrate,
                    progress_callback=lambda n_done, n_total: setattr(
                        self, '_render_progress_state', (n_done, n_total)),
                    **render_kwargs)
                params_path = save_render_params(
                    data, output_path, video_fps=fps, playback_speed=speed,
                    crop=crop, crop_2=crop_2, clims=used_clims, clims_2=used_clims_2,
                    bitrate=bitrate, **render_kwargs)
                self._render_queue.put(("done", (output_path, params_path)))
            except Exception as exc:
                self._render_queue.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()
        self.after(200, self._poll_render_queue)

    def _poll_render_queue(self):
        n_done, n_total = self._render_progress_state
        if n_total > 0:
            if str(self.render_progress["mode"]) != "determinate":
                self.render_progress.stop()
                self.render_progress.config(mode="determinate", maximum=n_total)
            self.render_progress["value"] = n_done
        try:
            status, payload = self._render_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_render_queue)
            return
        self.render_progress.stop()
        self.render_progress.config(mode="indeterminate", value=0)
        self.render_button.config(state="normal")
        if status == "done":
            output_path, params_path = payload
            self.app.status_var.set(f"Video saved: {output_path}  (params: {params_path})")
            messagebox.showinfo("Video saved", f"Saved to:\n{output_path}\n\nParams saved:\n{params_path}")
        else:
            self.app.status_var.set("Render failed - see error dialog.")
            self.app.report_error(payload)


class BehaviorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NDNF Behavior Pipeline Viewer")
        self.geometry("1250x900")

        self.lab = None
        self.experiment = None
        self._sessions_by_label = {}

        self._build_menu()
        self._build_top_bar()
        self._build_shared_force_controls()
        self._build_tabs()
        self._build_status_bar()

        self.after(100, self.startup_connect)

    # --- layout ---
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Change DataJoint config...", command=self.change_dj_config)
        file_menu.add_command(label="Reload subjects", command=self.reload_subjects)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_top_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Label(bar, text="Subject:").pack(side="left")
        self.subject_var = tk.StringVar()
        self.subject_cb = ttk.Combobox(bar, textvariable=self.subject_var, state="readonly", width=20)
        self.subject_cb.pack(side="left", padx=(4, 16))
        self.subject_cb.bind("<<ComboboxSelected>>", lambda e: self.on_subject_selected())

        ttk.Label(bar, text="Session(s):").pack(side="left")
        session_frame = ttk.Frame(bar)
        session_frame.pack(side="left", padx=(4, 4))
        self.session_listbox = tk.Listbox(session_frame, selectmode="extended", exportselection=False,
                                           width=26, height=4)
        session_scrollbar = ttk.Scrollbar(session_frame, orient="vertical", command=self.session_listbox.yview)
        self.session_listbox.configure(yscrollcommand=session_scrollbar.set)
        self.session_listbox.pack(side="left")
        session_scrollbar.pack(side="left", fill="y")
        self.session_listbox.bind("<<ListboxSelect>>", lambda e: self.on_session_selected())
        ttk.Label(bar, text="(ctrl/shift-click to compare several)", foreground="gray").pack(side="left", padx=(4, 16))

    def _build_shared_force_controls(self):
        """Epoch selection and trace filtering apply to force data, not to a specific tab, so
        one shared instance of each lives here (above the tab notebook) and is read by both
        Session Overview and Block Detail -- changing either is immediately reflected in both,
        rather than each tab tracking its own, independently-driftable copy."""
        row = ttk.Frame(self)
        row.pack(fill="x", padx=8, pady=(0, 4))
        self.epoch_selector = EpochSelector(row, on_change=self._on_shared_force_controls_changed)
        self.epoch_selector.pack(side="left")
        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=8, pady=(0, 6))
        self.filter_controls = TraceFilterControls(row2, on_change=self._on_shared_force_controls_changed)
        self.filter_controls.pack(side="left")

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
        block_session, block = self.block_detail_tab.get_selected_session_block()
        if block is not None:
            session = block_session
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

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.session_overview_tab = SessionOverviewTab(self.notebook, self)
        self.block_detail_tab = BlockDetailTab(self.notebook, self)
        self.subject_trend_tab = SubjectTrendTab(self.notebook, self)
        self.trials_per_mouse_tab = TrialsPerMouseTab(self.notebook, self)
        self.water_restriction_tab = WaterRestrictionTab(self.notebook, self)
        self.video_generation_tab = VideoGenerationTab(self.notebook, self)
        self.notebook.add(self.session_overview_tab, text="Session Overview")
        self.notebook.add(self.block_detail_tab, text="Block Detail")
        self.notebook.add(self.subject_trend_tab, text="Subject Trend")
        self.notebook.add(self.trials_per_mouse_tab, text="Trials per Mouse")
        self.notebook.add(self.water_restriction_tab, text="Water Restriction")
        self.notebook.add(self.video_generation_tab, text="Generate Video")
        self._tabs = (self.session_overview_tab, self.block_detail_tab, self.subject_trend_tab,
                       self.trials_per_mouse_tab, self.water_restriction_tab, self.video_generation_tab)
        # the video tab inherits its block/trial selection from Block Detail, which can change
        # without a session change (different block, different trial selection) — resync on
        # every tab switch so it's never showing a stale block/trial range
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        if self.notebook.nametowidget(self.notebook.select()) is self.video_generation_tab:
            self.video_generation_tab.sync_from_block_detail()

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Not connected.")
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken").pack(fill="x", side="bottom")

    # --- connection ---
    def startup_connect(self):
        dj_config_path = dj_connection.get_dj_config_path()
        if not dj_config_path or not Path(dj_config_path).exists():
            dj_config_path = self.prompt_for_dj_config()
            if not dj_config_path:
                messagebox.showwarning("No config", "No DataJoint config file was selected. Closing.")
                self.destroy()
                return
        self.connect(dj_config_path)

    def prompt_for_dj_config(self):
        path = filedialog.askopenfilename(
            title="Select your DataJoint config file (dj_local_conf*.json)",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
        if path:
            dj_connection.set_dj_config_path(path)
        return path

    def connect(self, dj_config_path):
        self.status_var.set(f"Connecting using {dj_config_path} ...")
        self.update_idletasks()
        try:
            conn, lab, experiment = dj_connection.connect(dj_config_path)
        except Exception as exc:
            self.report_error(exc)
            self.status_var.set("Connection failed. Use File > Change DataJoint config...")
            return
        self.lab = lab
        self.experiment = experiment
        self.status_var.set(f"Connected as {conn.get_user()}  |  config: {dj_config_path}")
        self.reload_subjects()

    def change_dj_config(self):
        path = self.prompt_for_dj_config()
        if path:
            messagebox.showinfo(
                "Restart required",
                "The new DataJoint config path has been saved. Please restart the "
                "application for the new connection to take effect.")

    # --- subject / session population ---
    def reload_subjects(self):
        if self.lab is None:
            return
        try:
            subject_ids = sorted(self.lab.Subject.fetch("subject_id").tolist())
        except Exception as exc:
            self.report_error(exc)
            return
        self.subject_cb["values"] = subject_ids
        if subject_ids and self.subject_var.get() not in subject_ids:
            self.subject_var.set(self._most_recent_subject(subject_ids))
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
        subject_id = self.subject_var.get()
        self.session_listbox.delete(0, "end")
        self._sessions_by_label = {}
        if not subject_id or self.experiment is None:
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
        for label in labels:
            self.session_listbox.insert("end", label)
        # default to the most recent session only - multi-session comparison is opt-in
        # (ctrl/shift-click more sessions in the listbox)
        if labels:
            self.session_listbox.selection_set(0)
        self.on_session_selected()

    def on_session_selected(self):
        self._notify_tabs("on_session_changed")

    def _notify_tabs(self, hook_name):
        for tab in self._tabs:
            hook = getattr(tab, hook_name, None)
            if hook is not None:
                hook()

    def get_selected_sessions(self):
        """Every currently selected session, sorted ascending (chronological by session id)."""
        labels = [self.session_listbox.get(i) for i in self.session_listbox.curselection()]
        return sorted(self._sessions_by_label[label] for label in labels)

    def get_selected_subject_session(self):
        """Subject + the "primary" (first) selected session, for tabs that only need one -
        Subject Trend / Trials per Mouse / Water Restriction use just the subject, and the
        shared sample-interval hint falls back to this when Block Detail has no block picked."""
        subject_id = self.subject_var.get() or None
        sessions = self.get_selected_sessions()
        return subject_id, (sessions[0] if sessions else None)

    # --- plotting / error helpers ---
    def run_plot(self, work_fn, panel):
        self.status_var.set("Plotting...")
        self.update_idletasks()
        try:
            fig = work_fn()
        except Exception as exc:
            self.report_error(exc)
            self.status_var.set("Plot failed - see error dialog.")
            return
        panel.show_figure(fig)
        self.status_var.set("Ready.")

    def report_error(self, exc):
        # traceback.print_exc() would print nothing useful here ("NoneType: None"): it reads
        # the *currently being handled* exception, but errors from the video-render worker
        # thread arrive here later, via the render queue, from inside an after()-scheduled poll
        # with no exception in flight - exc.__traceback__ is preserved on the exception object
        # itself regardless of which thread/context it's reported from, so use that instead.
        traceback.print_exception(exc)
        messagebox.showerror("Error", str(exc))


def main():
    app = BehaviorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
