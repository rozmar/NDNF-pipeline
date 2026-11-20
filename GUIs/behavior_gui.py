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
import offline_data


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
#oioioioi

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


class PlotStyleControl(ttk.Frame):
    """Line-vs-bar toggle for per-session trend plots."""

    def __init__(self, master, on_change, default_style='line'):
        super().__init__(master)
        self.on_change = on_change
        ttk.Label(self, text="Style:").pack(side="left")
        self.style_var = tk.StringVar(value=default_style)
        for style, text in (('line', 'Line'), ('bar', 'Bar')):
            ttk.Radiobutton(self, text=text, value=style, variable=self.style_var,
                             command=self.on_change).pack(side="left", padx=(4, 0))

    @property
    def style(self):
        return self.style_var.get()


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
        if force:
            dj_connection.query_cache.clear()  # Refresh button: always go back to the database
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


# sentinel stored in BlockDetailTab._blocks_by_label for its "All blocks" pseudo-entry -
# distinguishable from any real block number (an int) so callers can tell them apart
ALL_BLOCKS = "all"


class BlockDetailTab(ttk.Frame):
    """6-panel detail figure for a single block of the selected session, or (via the "All
    blocks" entry in the block dropdown) for every block of the session laid out on one
    continuous trial-number/session-time axis - so a trial selection can span a block
    boundary, e.g. the last trial of one block through the first of the next.

    A trial multi-select restricts which trials feed the force-distribution and
    trajectory panels; the performance (quiescence/response) panel always shows every trial
    in the block (or session, in "All blocks" mode) regardless of that selection.
    """

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._blocks_by_label = {}  # block_var label -> (session, block) or (session, ALL_BLOCKS)
        self._trial_list = []  # trial number for each row of trial_listbox, in display order
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
        self.trial_listbox.bind(
            "<<ListboxSelect>>", lambda e: self.app.debounce("trial_select", 150, self.refresh))
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
        self.trial_listbox.delete(0, "end")
        self.panel.clear()
        if subject_id is None or not sessions:
            self.app.update_shared_sample_interval()
            return
        multi_session = len(sessions) > 1

        def work():
            pairs = []
            for session in sessions:
                blocks = offline_data.session_blocks(self.app.experiment, subject_id, session)
                pairs.extend((session, block) for block in blocks)
            return pairs

        def on_success(pairs):
            labels = [f"s{session} b{block}" if multi_session else str(block) for session, block in pairs]
            self._blocks_by_label = dict(zip(labels, pairs))
            # "All blocks" lays every block of the session out on one continuous trial axis (see
            # plot_session_force_figure) - only offered for a single selected session, since
            # spanning several would make its target-LUT columns and trial list ambiguous about
            # which session's blocks they belong to
            if not multi_session and labels:
                all_label = "All blocks"
                self._blocks_by_label[all_label] = (sessions[0], ALL_BLOCKS)
                labels = labels + [all_label]
            self.block_cb["values"] = labels
            if labels:
                self.block_var.set(labels[0])
                self.on_block_selected()
            else:
                self.app.update_shared_sample_interval()

        self.app.run_bg(work, on_success, group="block_detail_blocks")

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
        self._trial_list = []
        if subject_id is None or session is None:
            return

        def work():
            if block == ALL_BLOCKS:
                # every trial of the session, each row tagged with its own block so a boundary-
                # spanning selection stays legible
                return offline_data.session_trials_with_blocks(self.app.experiment, subject_id, session)
            trials = offline_data.block_trials(self.app.experiment, subject_id, session, block)
            return [(int(t), None) for t in trials]

        def on_success(rows):
            for t, b in rows:
                self._trial_list.append(t)
                self.trial_listbox.insert("end", f"{t} (b{b})" if b is not None else t)

        self.app.run_bg(work, on_success, group="block_detail_trials")

    def select_all_trials(self):
        self.trial_listbox.selection_set(0, "end")
        self.refresh()

    def select_no_trials(self):
        self.trial_listbox.selection_clear(0, "end")
        self.refresh()

    def get_selected_trials(self):
        return [self._trial_list[i] for i in self.trial_listbox.curselection()]

    def refresh(self, force=False):
        if force:
            dj_connection.query_cache.clear()  # Refresh button: always go back to the database
        if not force and not self.auto_refresh_var.get():
            return
        subject_id = self.app.subject_var.get() or None
        session, block = self.get_selected_session_block()
        if subject_id is None or session is None:
            return
        selected_trials = self.get_selected_trials()

        if block == ALL_BLOCKS:
            from ndnf_pipeline.plot.behavior_plots import plot_session_force_figure

            def work():
                return plot_session_force_figure(
                    subject_id, session,
                    force_uniform_range=self.range_control.enabled,
                    uniform_force_range=self.range_control.get_range(),
                    perf_log_yscale=self.perf_controls.log_scale,
                    perf_smoothing_window=self.perf_controls.get_smoothing_window(),
                    trials=selected_trials or None,
                    epochs=self.app.epoch_selector.get_selected_epochs(),
                    **self.app.filter_controls.get_filter_kwargs())
        else:
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
    """Trial count / session duration / reward count / success rate for the selected
    subject's currently-selected session(s) -- a learning-curve view for judging day-to-day
    improvement. Select multiple sessions in the session list (ctrl/shift-click) to compare
    across days; with none selected, every session the subject has is shown."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Button(controls, text="Refresh", command=lambda: self.refresh(force=True)).pack(side="left")
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Auto refresh", variable=self.auto_refresh_var).pack(side="left", padx=(6, 0))
        self.all_sessions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Show all sessions", variable=self.all_sessions_var,
                        command=lambda: self.refresh(force=True)).pack(side="left", padx=(10, 0))
        self.style_control = PlotStyleControl(controls, on_change=lambda: self.refresh(force=True))
        self.style_control.pack(side="left", padx=(10, 0))
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_subject_changed(self):
        self.refresh()

    def on_session_changed(self):
        self.refresh()

    def refresh(self, force=False):
        if force:
            dj_connection.query_cache.clear()  # Refresh button: always go back to the database
        if not force and not self.auto_refresh_var.get():
            return
        subject_id, _ = self.app.get_selected_subject_session()
        if subject_id is None:
            self.panel.clear()
            return
        sessions = None if self.all_sessions_var.get() else (self.app.get_selected_sessions() or None)
        plot_style = self.style_control.style
        from ndnf_pipeline.plot.behavior_plots import plot_subject_behavior_trend

        def work():
            return plot_subject_behavior_trend(self.app.experiment, subject_id, sessions=sessions,
                                                plot_style=plot_style)

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
        self.mouse_listbox.bind(
            "<<ListboxSelect>>", lambda e: self.app.debounce("mouse_select", 150, self.refresh))

        button_row = ttk.Frame(left)
        button_row.pack(fill="x", pady=(5, 0))
        ttk.Button(button_row, text="All", command=self.select_all).pack(side="left")
        ttk.Button(button_row, text="None", command=self.select_none).pack(side="left", padx=(4, 0))
        ttk.Button(left, text="Reload mouse list", command=lambda: self.reload_mice(force=True)).pack(fill="x", pady=(5, 0))
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
        trend_controls = ttk.Frame(trend_frame)
        trend_controls.pack(fill="x", padx=5, pady=(2, 0))
        self.style_control = PlotStyleControl(trend_controls, on_change=lambda: self.refresh(force=True))
        self.style_control.pack(side="left")
        self.trend_panel = PlotPanel(trend_frame)
        self.trend_panel.pack(fill="both", expand=True)

    def on_subject_changed(self):
        # the mouse list only needs populating once a connection is established
        if not self._mice_loaded:
            self.reload_mice()

    def reload_mice(self, force=False):
        if self.app.lab is None or self.app.experiment is None:
            return
        if force:
            dj_connection.query_cache.clear()  # Reload button: always go back to the database

        def work():
            mouse_ids = sorted(self.app.lab.Subject.fetch("subject_id").tolist())
            trial_nums = [len(self.app.experiment.SessionTrial() & {"subject_id": m}) for m in mouse_ids]
            return [m for m, n in zip(mouse_ids, trial_nums) if n > 0]

        def on_success(mice_with_trials):
            previously_selected = set(self.get_selected_mice())
            self.mouse_listbox.delete(0, "end")
            for i, m in enumerate(mice_with_trials):
                self.mouse_listbox.insert("end", m)
                if m in previously_selected:
                    self.mouse_listbox.selection_set(i)
            self._mice_loaded = True
            self.refresh()

        self.app.run_bg(work, on_success, group="reload_mice")

    def select_all(self):
        self.mouse_listbox.selection_set(0, "end")
        self.refresh()

    def select_none(self):
        self.mouse_listbox.selection_clear(0, "end")
        self.refresh()

    def get_selected_mice(self):
        return [self.mouse_listbox.get(i) for i in self.mouse_listbox.curselection()]

    def refresh(self, force=False):
        if force:
            dj_connection.query_cache.clear()  # Refresh button: always go back to the database
        if not force and not self.auto_refresh_var.get():
            return
        if self.app.lab is None or self.app.experiment is None:
            return
        selected = self.get_selected_mice()
        from ndnf_pipeline.plot.behavior_plots import plot_trials_per_mouse, plot_subject_behavior_trends

        def work_bar():
            return plot_trials_per_mouse(self.app.lab, self.app.experiment, subject_ids=selected or None)

        def work_trend():
            return plot_subject_behavior_trends(self.app.experiment, selected, plot_style=self.style_control.style)

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
        if force:
            dj_connection.query_cache.clear()  # Refresh button: always go back to the database
        if not force and not self.auto_refresh_var.get():
            return
        subject_id, _ = self.app.get_selected_subject_session()
        from ndnf_pipeline.plot.behavior_plots import plot_water_restriction_overview

        def work():
            return plot_water_restriction_overview(self.app.lab, highlight_subject_id=subject_id)

        self.app.run_plot(work, self.panel)


class VideoGenerationTab(ttk.Frame):
    """Generate an annotated trial-range video for one block (or, via Block Detail's "All
    blocks" entry, a range spanning a block boundary), with an interactive crop tool.

    The block and trial range are inherited from the Block Detail tab (its block dropdown and
    its trial multi-select — "none selected" there means "every trial in the block/session")
    rather than picked here, so the video always matches whatever block/trials you're already
    looking at. In "All blocks" mode, the target-LUT overlay switches to match whichever block
    the currently-playing trial belongs to.

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
        # separate from _inherited_block being None, which (once synced) legitimately means
        # "All blocks" mode rather than "nothing picked yet"
        self._inherited_ready = False

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

        def work():
            return offline_data.session_cameras(subject_id, session)

        def on_success(cameras):
            self.camera_cb["values"] = cameras
            self.camera_2_cb["values"] = cameras
            if cameras:
                self.camera_var.set(cameras[0])
                self.camera_2_var.set(cameras[1] if len(cameras) > 1 else cameras[0])

        self.app.run_bg(work, on_success, group="video_cameras")

    def _on_camera2_toggle(self):
        enabled = self.use_camera_2.get()
        self.camera_2_cb.config(state="readonly" if enabled else "disabled")
        state = "normal" if enabled else "disabled"
        self.load_frame_2_btn.config(state=state)
        self.apply_crop_2_btn.config(state=state)
        self.reset_crop_2_btn.config(state=state)

    def sync_from_block_detail(self):
        """Re-read the Block Detail tab's block + trial selection (none selected there = every
        trial in the block/session) and convert it into the trial_start/trial_end
        load_trial_video_data expects: block-relative indices for a single picked block, or
        actual (session-absolute) trial numbers when Block Detail's "All blocks" entry is picked
        - see load_trial_video_data's own docstring for why those two mean different things.
        The session comes from whichever (session, block) Block Detail currently has picked -
        not necessarily the app's "primary" selected session, once more than one is selected
        there."""
        self._inherited_session = None
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None
        self._inherited_ready = False
        subject_id = self.app.subject_var.get() or None
        session, block = self.app.block_detail_tab.get_selected_session_block()
        if subject_id is None or session is None:
            self.inherited_label_var.set("(switch to the Block Detail tab and pick a block first)")
            return
        block_label = self.app.block_detail_tab.block_var.get()
        all_blocks_mode = (block == ALL_BLOCKS)
        # read now, not inside the background callback below -- the trial selection can change
        # while the fetch is still in flight, and we want the selection as of this call
        selected = self.app.block_detail_tab.get_selected_trials()

        def work():
            return offline_data.block_trials(self.app.experiment, subject_id, session,
                                             None if all_blocks_mode else block)

        def on_success(all_trials):
            if not all_trials:
                self.inherited_label_var.set(f"Block {block_label}: no trials in this block")
                return
            chosen = sorted(t for t in selected if t in all_trials) if selected else all_trials
            if not chosen:
                self.inherited_label_var.set(f"Block {block_label}: the selected trials aren't in this block")
                return
            self._inherited_session = session
            self._inherited_block = None if all_blocks_mode else block
            if all_blocks_mode:
                self._inherited_trial_start = min(chosen)
                self._inherited_trial_end = max(chosen)
            else:
                self._inherited_trial_start = all_trials.index(min(chosen))
                self._inherited_trial_end = all_trials.index(max(chosen))
            self._inherited_ready = True
            if selected:
                self.inherited_label_var.set(f"Block {block_label}: trials {min(chosen)}-{max(chosen)} "
                                              f"({len(chosen)} of {len(all_trials)} selected)")
            else:
                self.inherited_label_var.set(f"Block {block_label}: all {len(all_trials)} trials "
                                              f"(trial numbers {all_trials[0]}-{all_trials[-1]})")

        self.app.run_bg(work, on_success, group="video_sync")

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
        if not self._inherited_ready:
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
        return self._quality_presets.get(self.quality_var.get(), self._quality_presets['ultra'])

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
        if subject_id and session is not None and self._inherited_ready and camera:
            block_label = f"b{self._inherited_block}" if self._inherited_block is not None else "allblocks"
            default_name = (f"{subject_id}_s{session}_{block_label}_"
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


class DownloadDialog(tk.Toplevel):
    """File > Download sessions...: save the picked sessions of a subject to a local folder, so
    browsing them afterwards needs no network round trips (see offline_data.py). Optionally
    includes what the Generate Video tab needs: the per-trial video data and the raw video files.

    Downloaded sessions are read as they were when downloaded - Refresh doesn't refetch them;
    downloading again does. The data download runs as one job per session on the app's
    background worker (between the GUI's own fetches), the video copy on its own thread since
    it only touches files.
    """

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Download sessions")
        self.geometry("520x520")
        self.transient(app)
        self._busy = False
        self._copy_progress = queue.Queue()

        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Label(row, text="Subject:").pack(side="left")
        self.subject_var = tk.StringVar(value=app.subject_var.get())
        subject_cb = ttk.Combobox(row, textvariable=self.subject_var, state="readonly", width=20,
                                  values=list(app.subject_cb["values"]))
        subject_cb.pack(side="left", padx=4)
        subject_cb.bind("<<ComboboxSelected>>", lambda e: self.load_sessions())

        ttk.Label(top, text="Sessions (ctrl/shift-click for several):").pack(anchor="w", pady=(8, 2))
        list_frame = ttk.Frame(top)
        list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_frame, selectmode="extended", exportselection=False)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")
        buttons = ttk.Frame(top)
        buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(buttons, text="All", command=lambda: self.listbox.selection_set(0, "end")).pack(side="left")
        ttk.Button(buttons, text="None", command=lambda: self.listbox.selection_clear(0, "end")).pack(
            side="left", padx=4)

        self.include_videos = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Include videos (video files + data for the Generate Video tab)",
                        variable=self.include_videos).pack(anchor="w", pady=(8, 0))

        actions = ttk.Frame(top)
        actions.pack(fill="x", pady=(8, 0))
        self.download_btn = ttk.Button(actions, text="Download to folder...", command=self.start_download)
        self.download_btn.pack(side="left")
        self.remove_btn = ttk.Button(actions, text="Remove local copy", command=self.remove_selected)
        self.remove_btn.pack(side="left", padx=4)
        ttk.Button(actions, text="Close", command=self.destroy).pack(side="right")

        self.progress = ttk.Progressbar(top, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="Downloaded sessions are marked (local). Refresh doesn't "
                                             "refetch them - download again to update one.")
        ttk.Label(top, textvariable=self.status_var, wraplength=490, justify="left").pack(anchor="w", pady=(4, 0))

        self._sessions = []  # (session, session_date) per listbox row
        self.load_sessions()

    def load_sessions(self):
        subject_id = self.subject_var.get()
        self.listbox.delete(0, "end")
        self._sessions = []
        if not subject_id:
            return

        def work():
            return (self.app.experiment.Session() & {"subject_id": subject_id}).fetch(
                "session", "session_date", order_by="session DESC")

        def on_success(result):
            if not self.winfo_exists() or subject_id != self.subject_var.get():
                return
            sessions, dates = result
            downloaded = offline_data.downloaded_sessions(subject_id)
            self._sessions = list(zip(sessions.tolist(), dates))
            for session, date in self._sessions:
                entry = downloaded.get(session)
                if entry is None:
                    label = f"{session} — {date}"
                else:
                    videos = ", with videos" if entry["videos"] else ""
                    label = f"{session} — {date}   (local{videos}, {entry['downloaded_at'][:10]})"
                self.listbox.insert("end", label)

        self.app.run_bg(work, on_success, group="download_dialog_sessions")

    def _selected(self):
        return [self._sessions[i] for i in self.listbox.curselection()]

    def _set_busy(self, busy):
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.download_btn.config(state=state)
        self.remove_btn.config(state=state)

    def start_download(self):
        subject_id = self.subject_var.get()
        selected = self._selected()
        if not subject_id or not selected:
            messagebox.showinfo("Nothing selected", "Pick one or more sessions to download.", parent=self)
            return
        dirs = offline_data.download_dirs()
        folder = filedialog.askdirectory(
            parent=self, title="Download sessions to folder",
            initialdir=str(dirs[-1]) if dirs else str(Path.home()))
        if not folder:
            return
        folder = Path(folder)
        include_videos = self.include_videos.get()
        offline_data.register_download_dir(folder)
        self._set_busy(True)
        self.progress.config(maximum=len(selected), value=0)
        all_videos = []
        problems = []

        def download_next(i):
            if i == len(selected):
                on_data_done()
                return
            session, date = selected[i]
            self.status_var.set(f"Downloading {subject_id} session {session} ({i + 1}/{len(selected)})...")

            def work():
                return offline_data.download_session_data(
                    self.app.experiment, folder, subject_id, session, include_videos)

            def on_success(result):
                keys, video_files, session_problems = result
                videos = video_files if include_videos else []
                offline_data.record_download(folder, subject_id, session, date, keys, videos)
                all_videos.extend(videos)
                problems.extend(f"s{session} {p}" for p in session_problems)
                if self.winfo_exists():
                    self.progress.config(value=i + 1)
                download_next(i + 1)

            def on_error(exc):
                self.app.report_error(exc)
                if self.winfo_exists():
                    self._set_busy(False)
                    self.status_var.set(f"Download failed at session {session} - see error dialog. "
                                        f"Sessions before it were saved.")
                self.app.populate_sessions()

            self.app.run_bg(work, on_success, on_error)

        def on_data_done():
            if all_videos:
                self.status_var.set(f"Copying {len(all_videos)} video file(s)...")
                threading.Thread(target=copy_videos, daemon=True).start()
                self.after(200, poll_copy)
            else:
                finish()

        def copy_videos():
            try:
                offline_data.copy_videos(all_videos, folder,
                                         progress=lambda done, total: self._copy_progress.put((done, total)))
                self._copy_progress.put("done")
            except Exception as exc:
                self._copy_progress.put(exc)

        def poll_copy():
            try:
                while True:
                    item = self._copy_progress.get_nowait()
                    if item == "done":
                        finish()
                        return
                    if isinstance(item, Exception):
                        self.app.report_error(item)
                        finish(f"Video copy failed - see error dialog. The data was saved; "
                               f"download again with videos to retry.")
                        return
                    done, total = item
                    if self.winfo_exists():
                        self.progress.config(maximum=max(total, 1), value=done)
                        self.status_var.set(f"Copying videos: {done / 1024 ** 3:.2f} of "
                                            f"{total / 1024 ** 3:.2f} GB")
            except queue.Empty:
                pass
            self.after(200, poll_copy)

        def finish(message=None):
            self.app.populate_sessions()
            if not self.winfo_exists():
                return
            self._set_busy(False)
            self.load_sessions()
            if message is None:
                message = f"Downloaded {len(selected)} session(s) to {folder}."
                if problems:
                    message += (f" {len(problems)} view(s) couldn't be built (they fail the same way "
                                f"in the GUI), e.g. {problems[0]}")
            self.status_var.set(message)

        download_next(0)

    def remove_selected(self):
        subject_id = self.subject_var.get()
        downloaded = offline_data.downloaded_sessions(subject_id)
        selected = [session for session, _ in self._selected() if session in downloaded]
        if not selected:
            messagebox.showinfo("Nothing to remove", "None of the selected sessions are downloaded.", parent=self)
            return
        if not messagebox.askyesno(
                "Remove local copy",
                f"Delete the local copy of {len(selected)} session(s)? They'll be read from the "
                f"database again.", parent=self):
            return
        for session in selected:
            offline_data.remove_download(downloaded[session]["folder"], subject_id, session)
        self.status_var.set(f"Removed the local copy of {len(selected)} session(s).")
        self.load_sessions()
        self.app.populate_sessions()


class BehaviorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NDNF Behavior Pipeline Viewer")
        self.geometry("1250x900")

        self.lab = None
        self.experiment = None
        self._sessions_by_label = {}

        # background-task infrastructure: DataJoint fetches and matplotlib figure building are
        # both slow enough (network round-trips, plot layout) that doing them on the Tk main
        # thread visibly freezes the UI -- every call site that used to do this work inline now
        # goes through run_bg()/run_plot(), which hands it to a single serial worker thread (one
        # thread, not a pool: DataJoint connections and the Agg-backend matplotlib globals used
        # here aren't safe for concurrent access) and delivers the result back on the main thread
        # via after()-polling. A frozen main thread was also why dragging across several days in
        # the session listbox could end up selecting more than intended: queued mouse-motion
        # events only got processed once the blocking call returned, in a burst.
        self._debounce_jobs = {}
        self._bg_lock = threading.Lock()
        self._bg_queue = queue.Queue()
        self._bg_results = queue.Queue()
        self._bg_request_seq = 0
        self._bg_pending = {}       # request_id -> (on_success, on_error, group)
        self._bg_group_active = {}  # group -> request_id of the only one worth running/delivering
        threading.Thread(target=self._bg_worker, daemon=True).start()
        self.after(50, self._poll_bg_results)

        self._build_menu()
        self._build_top_bar()
        self._build_training_log()
        self._build_shared_force_controls()
        self._build_tabs()
        self._build_status_bar()

        self.after(100, self.startup_connect)

    # --- layout ---
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Change DataJoint config...", command=self.change_dj_config)
        file_menu.add_command(label="Reload subjects", command=lambda: self.reload_subjects(force=True))
        file_menu.add_command(label="Download sessions...", command=self.open_download_dialog)
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
        self.session_listbox.bind(
            "<<ListboxSelect>>", lambda e: self.debounce("session_select", 150, self.on_session_selected))
        ttk.Label(bar, text="(ctrl/shift-click to compare several)", foreground="gray").pack(side="left", padx=(4, 16))
        # whether the selected session(s) ran with the Instantaneous setting ON - stored per block
        # as Block.feedback_type ('*_position' = instantaneous, '*_speed' = accumulating)
        self.instantaneous_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.instantaneous_var, justify="left").pack(side="left")

    def _build_training_log(self):
        """Read-only box with the NDNF procedures sheet's training-log entries (weight, water,
        training type, rig, preset, notes, behavior notes) for each selected session's date."""
        frame = ttk.LabelFrame(self, text="Training log (NDNF procedures sheet)")
        frame.pack(fill="x", padx=8, pady=(0, 6))
        self.training_log_text = tk.Text(frame, height=4, wrap="word", relief="flat",
                                         background=self.cget("background"), font=("TkDefaultFont", 9))
        log_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.training_log_text.yview)
        self.training_log_text.configure(yscrollcommand=log_scrollbar.set, state="disabled")
        self.training_log_text.tag_configure("head", font=("TkDefaultFont", 9, "bold"))
        self.training_log_text.tag_configure("muted", foreground="gray")
        self.training_log_text.pack(side="left", fill="x", expand=True, padx=(4, 0), pady=2)
        log_scrollbar.pack(side="left", fill="y")

    def _set_training_log(self, chunks):
        """chunks: [(text, tag or None)]"""
        widget = self.training_log_text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for text, tag in chunks:
            widget.insert("end", text, tag or ())
        widget.configure(state="disabled")

    def update_training_log(self):
        subject_id = self.subject_var.get() or None
        sessions = self.get_selected_sessions()
        self._set_training_log([])
        if subject_id is None or not sessions:
            return
        self._set_training_log([("Loading...", "muted")])

        def work():
            results = [(session, *offline_data.session_procedures_log(self.experiment, self.lab, subject_id, session))
                       for session in sessions]
            for _, _, entries, _ in results:
                for e in entries:
                    # live lookup in the Telekinesis presets folder; a downloaded session keeps
                    # the match saved at download time when that folder isn't on this computer
                    try:
                        match = offline_data.preset_instantaneous(e["preset"])
                    except Exception:
                        match = None
                    if match is not None or "preset_match" not in e:
                        e["preset_match"] = match
            return results

        def fmt_preset(e):
            """'preset X (Instantaneous ON)', from the Telekinesis presets folder."""
            match = e.get("preset_match")
            if match is None:
                return f"preset {e['preset']} (no matching preset file)"
            stem, exact, flags = match  # a list, not a tuple, when read back from a download
            if all(flags):
                inst = "Instantaneous ON"
            elif not any(flags):
                inst = "Instantaneous OFF"
            else:
                inst = "Instantaneous " + ", ".join(f"b{i} {'ON' if f else 'OFF'}" for i, f in enumerate(flags))
            name = e["preset"] if exact else f"{e['preset']} -> {stem}"
            return f"preset {name} ({inst})"

        def fmt_entry(e):
            parts = [e["time"]]
            for key in ("training_type", "rig"):
                if e[key]:
                    parts.append(e[key])
            if e["preset"]:
                parts.append(fmt_preset(e))
            if e["weight"]:
                parts.append(f"weight {e['weight']} g" + (f" -> {e['weight_after']} g" if e["weight_after"] else ""))
            if e["water_given"]:
                parts.append(f"water given {e['water_given']} ml")
            if e["water_earned"]:
                parts.append(f"earned {e['water_earned']} ml")
            if e["experimenter"]:
                parts.append(e["experimenter"])
            lines = ["  " + "  |  ".join(parts) + "\n"]
            if e["notes"]:
                lines.append(f"    Notes: {e['notes']}\n")
            if e["behavior_notes"]:
                lines.append(f"    Behavior notes: {e['behavior_notes']}\n")
            return "".join(lines)

        def on_success(results):
            chunks = []
            for session, session_date, entries, source in results:
                chunks.append((f"Session {session} ({session_date})", "head"))
                if source == "datajoint":
                    chunks.append(("  - sheet CSV not reachable, DataJoint log only "
                                   "(no training type / rig / preset / behavior notes)", "muted"))
                chunks.append(("\n", None))
                if not entries:
                    chunks.append(("  no training-log entry for this date\n", "muted"))
                for e in entries:
                    chunks.append((fmt_entry(e), None))
            self._set_training_log(chunks)

        def on_error(exc):
            self._set_training_log([(f"Training log unavailable: {exc}", "muted")])

        self.run_bg(work, on_success, on_error, group="training_log")

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
        from ndnf_pipeline.plot.behavior_plots import estimate_force_sample_interval

        def work():
            resolved_block = block
            # "All blocks" mode has no single block of its own either - same fallback as no
            # block picked yet, below
            if resolved_block is None or resolved_block == ALL_BLOCKS:
                blocks = offline_data.session_blocks(self.experiment, subject_id, session)
                resolved_block = blocks[0] if blocks else None
            if resolved_block is None:
                return None
            return estimate_force_sample_interval(subject_id, session, resolved_block)

        def on_success(sample_interval_s):
            self.filter_controls.set_sample_interval(sample_interval_s)

        def on_error(exc):
            self.report_error(exc)
            self.filter_controls.set_sample_interval(None)

        self.run_bg(work, on_success, on_error, group="sample_interval")

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
        offline_data.register_saved_download_dirs()
        self.status_var.set(f"Connected as {conn.get_user()}  |  config: {dj_config_path}")
        self.reload_subjects()

    def open_download_dialog(self):
        if self.experiment is None:
            messagebox.showinfo("Not connected", "Connect to the database first.")
            return
        DownloadDialog(self)

    def change_dj_config(self):
        path = self.prompt_for_dj_config()
        if path:
            messagebox.showinfo(
                "Restart required",
                "The new DataJoint config path has been saved. Please restart the "
                "application for the new connection to take effect.")

    # --- subject / session population ---
    def reload_subjects(self, force=False):
        if self.lab is None:
            return
        if force:
            dj_connection.query_cache.clear()  # Reload button: always go back to the database

        def work():
            subject_ids = sorted(self.lab.Subject.fetch("subject_id").tolist())
            most_recent = self._most_recent_subject(subject_ids) if subject_ids else None
            return subject_ids, most_recent

        def on_success(result):
            subject_ids, most_recent = result
            self.subject_cb["values"] = subject_ids
            if subject_ids and self.subject_var.get() not in subject_ids:
                self.subject_var.set(most_recent)
            if subject_ids:
                self.on_subject_selected()

        self.run_bg(work, on_success, group="reload_subjects")

    def _most_recent_subject(self, subject_ids):
        """Subject with the most recent session, falling back to the first subject alphabetically.
        Called from reload_subjects' background worker, not the main thread."""
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
            self.on_session_selected()
            return

        def work():
            return (self.experiment.Session() & {"subject_id": subject_id}).fetch(
                "session", "session_date", order_by="session DESC")

        def on_success(result):
            sessions, dates = result
            downloaded = offline_data.downloaded_sessions(subject_id)
            labels = [f"{s} — {d}" + ("  (local)" if s in downloaded else "")
                      for s, d in zip(sessions.tolist(), dates)]
            self._sessions_by_label = dict(zip(labels, sessions.tolist()))
            for label in labels:
                self.session_listbox.insert("end", label)
            # default to the most recent session only - multi-session comparison is opt-in
            # (ctrl/shift-click more sessions in the listbox)
            if labels:
                self.session_listbox.selection_set(0)
            self.on_session_selected()

        self.run_bg(work, on_success, group="populate_sessions")

    def on_session_selected(self):
        self.update_instantaneous_label()
        self.update_training_log()
        self._notify_tabs("on_session_changed")

    def update_instantaneous_label(self):
        """Show whether each selected session ran with the Instantaneous setting ON or OFF
        (per block, when a session mixes both)."""
        subject_id = self.subject_var.get() or None
        sessions = self.get_selected_sessions()
        self.instantaneous_var.set("")
        if subject_id is None or not sessions:
            return

        def describe(block_types):
            states = [(b, "ON" if f.endswith("_position") else "OFF" if f.endswith("_speed") else "?")
                      for b, f in block_types]
            distinct = {state for _, state in states}
            if not states:
                return "?"
            if len(distinct) == 1:
                return distinct.pop()
            return "mixed (" + ", ".join(f"b{b} {state}" for b, state in states) + ")"

        def describe_recorded(flags):
            if not flags:
                return "not recorded"
            # None = no action_type saved: task versions from before the Instantaneous option
            # existed, which always integrated (OFF)
            states = ["ON" if f else "OFF" for f in flags]
            if len(set(states)) == 1:
                return states[0]
            return "mixed (" + ", ".join(f"b{i + 1} {s}" for i, s in enumerate(states)) + ")"

        def work():
            results = []
            for session in sessions:
                # the raw data on Z: holds what the task actually ran; the database's
                # Block.feedback_type was mis-ingested for most sessions, so it's a last resort
                try:
                    flags = offline_data.session_recorded_instantaneous(self.experiment, subject_id, session)
                except Exception:
                    flags = None
                if flags is not None:
                    results.append((session, describe_recorded(flags)))
                else:
                    results.append((session, describe(offline_data.session_block_feedback_types(
                        self.experiment, subject_id, session)) + " (database - raw data on Z: not reachable)"))
            return results

        def on_success(results):
            if len(results) == 1:
                text = f"Instantaneous: {results[0][1]}"
            else:
                text = "Instantaneous:\n" + "\n".join(f"s{s}: {d}" for s, d in results)
            self.instantaneous_var.set(text)

        def on_error(exc):
            # e.g. a session downloaded for offline use before this query existed
            self.instantaneous_var.set("Instantaneous: unknown")

        self.run_bg(work, on_success, on_error, group="instantaneous_label")

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

    # --- background work / plotting / error helpers ---
    def debounce(self, key, delay_ms, fn):
        """Collapse a burst of calls under the same key into a single call to fn(), delay_ms
        after the last call -- used on the multi-select listboxes so a click-drag across several
        rows runs fn() (which kicks off a chain of DataJoint fetches and plots) once, not once
        per row passed over."""
        job = self._debounce_jobs.pop(key, None)
        if job is not None:
            self.after_cancel(job)
        self._debounce_jobs[key] = self.after(delay_ms, fn)

    def run_bg(self, fn, on_success, on_error=None, group=None):
        """Run fn() on the background worker thread; on completion, on_success(result) (or
        on_error(exc), defaulting to report_error) is called on the main thread. If `group` is
        given, only the most recently submitted request in that group is actually run and
        delivered -- an in-flight fetch/plot for the same group that gets superseded by a newer
        one (e.g. the session selection changing again before the first fetch returns) is
        silently dropped instead of landing late and clobbering the newer result."""
        with self._bg_lock:
            self._bg_request_seq += 1
            request_id = self._bg_request_seq
            if group is not None:
                self._bg_group_active[group] = request_id
            self._bg_pending[request_id] = (on_success, on_error, group)
        self._bg_queue.put((request_id, group, fn))
        return request_id

    def _bg_worker(self):
        while True:
            request_id, group, fn = self._bg_queue.get()
            with self._bg_lock:
                superseded = group is not None and self._bg_group_active.get(group) != request_id
            if superseded:
                with self._bg_lock:
                    self._bg_pending.pop(request_id, None)
                continue
            try:
                result, status = fn(), "ok"
            except Exception as exc:
                result, status = exc, "error"
            self._bg_results.put((request_id, status, result))

    def _poll_bg_results(self):
        try:
            while True:
                request_id, status, payload = self._bg_results.get_nowait()
                with self._bg_lock:
                    entry = self._bg_pending.pop(request_id, None)
                    if entry is not None and entry[2] is not None \
                            and self._bg_group_active.get(entry[2]) != request_id:
                        entry = None  # a newer request in the same group superseded this one
                if entry is None:
                    continue
                on_success, on_error, _ = entry
                if status == "ok":
                    on_success(payload)
                else:
                    (on_error or self.report_error)(payload)
        except queue.Empty:
            pass
        self.after(50, self._poll_bg_results)

    def run_plot(self, work_fn, panel, group=None):
        self.status_var.set("Plotting...")

        def on_success(fig):
            panel.show_figure(fig)
            self.status_var.set("Ready.")

        def on_error(exc):
            self.report_error(exc)
            self.status_var.set("Plot failed - see error dialog.")

        self.run_bg(work_fn, on_success, on_error, group=group if group is not None else panel)

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
