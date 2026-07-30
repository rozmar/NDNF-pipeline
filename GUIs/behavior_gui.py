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
        self.subtract_median = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Subtract force median", variable=self.subtract_median,
                         command=self.refresh).pack(side="left")
        self.range_control = UniformRangeControl(controls, on_change=self.refresh)
        self.range_control.pack(side="left", padx=(10, 0))
        self.perf_controls = PerfAxisControls(controls, on_change=self.refresh)
        self.perf_controls.pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=(10, 0))
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_session_changed(self):
        self.refresh()

    def refresh(self):
        subject_id, session = self.app.get_selected_subject_session()
        if subject_id is None or session is None:
            self.panel.clear()
            return
        from ndnf_pipeline.plot.behavior_plots import plot_session_blocks_overview

        def work():
            return plot_session_blocks_overview(
                subject_id, session,
                subtract_force_median=self.subtract_median.get(),
                force_uniform_range=self.range_control.enabled,
                uniform_force_range=self.range_control.get_range(),
                perf_log_yscale=self.perf_controls.log_scale,
                perf_smoothing_window=self.perf_controls.get_smoothing_window())

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
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Label(controls, text="Block:").pack(side="left")
        self.block_var = tk.StringVar()
        self.block_cb = ttk.Combobox(controls, textvariable=self.block_var, state="readonly", width=8)
        self.block_cb.pack(side="left", padx=(4, 10))
        self.block_cb.bind("<<ComboboxSelected>>", lambda e: self.on_block_selected())
        self.subtract_median = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Subtract force median", variable=self.subtract_median,
                         command=self.refresh).pack(side="left")
        self.range_control = UniformRangeControl(controls, on_change=self.refresh)
        self.range_control.pack(side="left", padx=(10, 0))
        self.perf_controls = PerfAxisControls(controls, on_change=self.refresh)
        self.perf_controls.pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=(10, 0))

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
        subject_id, session = self.app.get_selected_subject_session()
        self.block_cb["values"] = []
        self.block_var.set("")
        if subject_id is None or session is None:
            self.trial_listbox.delete(0, "end")
            self.panel.clear()
            return
        try:
            blocks = sorted((self.app.experiment.Block()
                              & {"subject_id": subject_id, "session": session}).fetch("block").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        values = [str(b) for b in blocks]
        self.block_cb["values"] = values
        if values:
            self.block_var.set(values[0])
            self.on_block_selected()
        else:
            self.trial_listbox.delete(0, "end")
            self.panel.clear()

    def on_block_selected(self):
        self.reload_trials()
        self.refresh()

    def reload_trials(self):
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.block_var.get()
        self.trial_listbox.delete(0, "end")
        if subject_id is None or session is None or not block_str:
            return
        key = {"subject_id": subject_id, "session": session, "block": int(block_str)}
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

    def refresh(self):
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.block_var.get()
        if subject_id is None or session is None or not block_str:
            return
        block = int(block_str)
        selected_trials = self.get_selected_trials()
        from ndnf_pipeline.plot.behavior_plots import plot_block_force_figure

        def work():
            return plot_block_force_figure(
                subject_id, session, block,
                subtract_force_median=self.subtract_median.get(),
                force_uniform_range=self.range_control.enabled,
                uniform_force_range=self.range_control.get_range(),
                perf_log_yscale=self.perf_controls.log_scale,
                perf_smoothing_window=self.perf_controls.get_smoothing_window(),
                trials=selected_trials or None)

        self.app.run_plot(work, self.panel)


class SubjectTrendTab(ttk.Frame):
    """Trial count / session length / hit rate across all sessions of the selected subject."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left")
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_subject_changed(self):
        self.refresh()

    def refresh(self):
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

    def refresh(self):
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
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left")
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_subject_changed(self):
        self.refresh()

    def refresh(self):
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
        self._crop = (0, 0, 0, 0)
        self._crop_frame_shape = None
        self._crop_selector = None
        self._preview_clims = None
        self._output_path = None
        self._render_queue = queue.Queue()
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None

        selectors = ttk.Frame(self)
        selectors.pack(fill="x", padx=5, pady=(5, 0))
        ttk.Label(selectors, text="From Block Detail tab:").pack(side="left")
        self.inherited_label_var = tk.StringVar(value="(switch to the Block Detail tab and pick a block first)")
        ttk.Label(selectors, textvariable=self.inherited_label_var, foreground="gray").pack(side="left", padx=(4, 10))
        ttk.Button(selectors, text="Refresh from Block Detail", command=self.sync_from_block_detail).pack(side="left")

        ttk.Label(selectors, text="Camera:").pack(side="left", padx=(16, 0))
        self.camera_var = tk.StringVar()
        self.camera_cb = ttk.Combobox(selectors, textvariable=self.camera_var, state="readonly", width=14)
        self.camera_cb.pack(side="left", padx=(4, 10))

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
                     values=["en", "hu"]).pack(side="left", padx=(2, 0))

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=5, pady=(4, 4))
        ttk.Button(actions, text="Load frame for cropping", command=self.load_frame_for_cropping).pack(side="left")
        ttk.Button(actions, text="Apply crop from selection", command=self.apply_crop).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Reset crop", command=self.reset_crop).pack(side="left", padx=(6, 0))
        self.crop_label_var = tk.StringVar(value="crop: left=0 right=0 top=0 bottom=0")
        ttk.Label(actions, textvariable=self.crop_label_var).pack(side="left", padx=(10, 0))

        actions2 = ttk.Frame(self)
        actions2.pack(fill="x", padx=5, pady=(0, 4))
        ttk.Button(actions2, text="Preview", command=self.do_preview).pack(side="left")
        ttk.Button(actions2, text="Choose output file...", command=self.choose_output_path).pack(side="left", padx=(6, 0))
        self.output_label_var = tk.StringVar(value="(no output file chosen)")
        ttk.Label(actions2, textvariable=self.output_label_var, foreground="gray").pack(side="left", padx=(6, 0))
        self.render_button = ttk.Button(actions2, text="Render video", command=self.start_render)
        self.render_button.pack(side="left", padx=(10, 0))
        self.render_progress = ttk.Progressbar(actions2, mode="indeterminate", length=120)
        self.render_progress.pack(side="left", padx=(10, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        crop_frame = ttk.LabelFrame(body, text="Crop selection (drag the box edges)")
        crop_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.crop_panel = PlotPanel(crop_frame)
        self.crop_panel.pack(fill="both", expand=True)
        preview_frame = ttk.LabelFrame(body, text="Preview (last frame)")
        preview_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.preview_panel = PlotPanel(preview_frame)
        self.preview_panel.pack(fill="both", expand=True)

    # --- inherit block/trials from the Block Detail tab ---
    def on_session_changed(self):
        self.reload_cameras()
        self.sync_from_block_detail()

    def reload_cameras(self):
        subject_id, session = self.app.get_selected_subject_session()
        self.camera_cb["values"] = []
        self.camera_var.set("")
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
        if cameras:
            self.camera_var.set(cameras[0])

    def sync_from_block_detail(self):
        """Re-read the Block Detail tab's block + trial selection (none selected there = every
        trial in the block) and convert it into the block-relative trial_start/trial_end indices
        load_trial_video_data expects."""
        self._inherited_block = None
        self._inherited_trial_start = None
        self._inherited_trial_end = None
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.app.block_detail_tab.block_var.get()
        if subject_id is None or session is None or not block_str:
            self.inherited_label_var.set("(switch to the Block Detail tab and pick a block first)")
            return
        try:
            all_trials = sorted((self.app.experiment.BehaviorTrial()
                                  & {"subject_id": subject_id, "session": session, "block": int(block_str)}
                                  ).fetch("trial").tolist())
        except Exception as exc:
            self.app.report_error(exc)
            return
        if not all_trials:
            self.inherited_label_var.set(f"Block {block_str}: no trials in this block")
            return
        selected = self.app.block_detail_tab.get_selected_trials()
        chosen = sorted(t for t in selected if t in all_trials) if selected else all_trials
        if not chosen:
            self.inherited_label_var.set(f"Block {block_str}: the selected trials aren't in this block")
            return
        self._inherited_block = int(block_str)
        self._inherited_trial_start = all_trials.index(min(chosen))
        self._inherited_trial_end = all_trials.index(max(chosen))
        if selected:
            self.inherited_label_var.set(f"Block {block_str}: trials {min(chosen)}-{max(chosen)} "
                                          f"({len(chosen)} of {len(all_trials)} selected)")
        else:
            self.inherited_label_var.set(f"Block {block_str}: all {len(all_trials)} trials "
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
        """Read + validate the inherited block/trial-range plus the camera selection. Returns
        load_trial_video_data kwargs, or None (after reporting the problem) if something
        required is missing/invalid."""
        subject_id, session = self.app.get_selected_subject_session()
        camera = self.camera_var.get()
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
        return dict(subject_id=subject_id, session=session, block=self._inherited_block,
                    trial_start=self._inherited_trial_start, trial_end=self._inherited_trial_end,
                    camera_name=camera,
                    pad_start=self._read_float(self.pad_start_var, 1.0),
                    pad_end=self._read_float(self.pad_end_var, 1.0))

    def _render_kwargs(self):
        return dict(
            clim_pct=(self._read_float(self.clim_low_var, 0), self._read_float(self.clim_high_var, 95)),
            tail_s=self._read_float(self.tail_var, 5),
            force_axis_limit=self._read_float(self.force_limit_var, 10.0),
            lang=self.lang_var.get() or "en",
        )

    # --- actions ---
    def load_frame_for_cropping(self):
        params = self._current_params()
        if params is None:
            return
        self.app.status_var.set("Loading frame...")
        self.app.update_idletasks()
        try:
            from ndnf_pipeline.plot.videography_plots import load_trial_video_data, preview_last_frame, _prepare_frame
            data = load_trial_video_data(**params)
            render_kwargs = self._render_kwargs()
            dashboard_fig, last_frame, clims = preview_last_frame(data, crop=(0, 0, 0, 0),
                                                                    clim_pct=render_kwargs["clim_pct"])
            plt.close(dashboard_fig)
            rgb = _prepare_frame(last_frame, (0, 0, 0, 0), clims)
            h, w = last_frame.shape[:2]
            fig, ax = plt.subplots(figsize=(7, max(4.0, 7 * h / w)))
            ax.imshow(rgb)
            ax.set_title("Drag the box edges/corners to set the crop, then click 'Apply crop'", fontsize=9)
            left, right, top, bottom = self._crop
            selector = RectangleSelector(ax, onselect=lambda *a: None, useblit=False, interactive=True,
                                          button=[1], minspanx=5, minspany=5, spancoords="pixels",
                                          drag_from_anywhere=True)
            selector.extents = (left, max(w - right, left + 1), top, max(h - bottom, top + 1))
            self._crop_selector = selector
            self._crop_frame_shape = (h, w)
            self.crop_panel.show_figure(fig)
            self.app.status_var.set("Frame loaded - adjust the crop box, then click 'Apply crop'.")
        except Exception as exc:
            self.app.report_error(exc)
            self.app.status_var.set("Failed to load frame - see error dialog.")

    def apply_crop(self):
        if self._crop_selector is None or self._crop_frame_shape is None:
            self.app.report_error(RuntimeError("Load a frame for cropping first."))
            return
        xmin, xmax, ymin, ymax = self._crop_selector.extents
        h, w = self._crop_frame_shape
        left = int(round(max(0, xmin)))
        right = int(round(max(0, w - xmax)))
        top = int(round(max(0, ymin)))
        bottom = int(round(max(0, h - ymax)))
        self._crop = (left, right, top, bottom)
        self.crop_label_var.set(f"crop: left={left} right={right} top={top} bottom={bottom}")

    def reset_crop(self):
        self._crop = (0, 0, 0, 0)
        self.crop_label_var.set("crop: left=0 right=0 top=0 bottom=0")
        if self._crop_selector is not None and self._crop_frame_shape is not None:
            h, w = self._crop_frame_shape
            self._crop_selector.extents = (0, w, 0, h)

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
            fig, last_frame, clims = preview_last_frame(data, crop=self._crop, **render_kwargs)
            self._preview_clims = clims
            self.preview_panel.show_figure(fig)
            self.app.status_var.set("Preview ready.")
        except Exception as exc:
            self.app.report_error(exc)
            self.app.status_var.set("Preview failed - see error dialog.")

    def choose_output_path(self):
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.block_var.get()
        camera = self.camera_var.get()
        default_name = "video.mp4"
        if subject_id and session is not None and block_str and camera:
            default_name = (f"{subject_id}_s{session}_b{block_str}_"
                             f"t{self.trial_start_var.get()}-{self.trial_end_var.get()}_{camera}.mp4")
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
        speed = self._read_float(self.speed_var, 1.0)
        fps = self._read_int(self.fps_var, 20)
        crop = self._crop
        clims = self._preview_clims
        output_path = self._output_path

        self.render_button.config(state="disabled")
        self.render_progress.start(50)
        self.app.status_var.set("Rendering video... this can take a while.")

        def work():
            try:
                from ndnf_pipeline.plot.videography_plots import load_trial_video_data, render_trial_video
                data = load_trial_video_data(**params)
                render_trial_video(data, output_path, video_fps=fps, playback_speed=speed,
                                    crop=crop, clims=clims, **render_kwargs)
                self._render_queue.put(("done", output_path))
            except Exception as exc:
                self._render_queue.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()
        self.after(200, self._poll_render_queue)

    def _poll_render_queue(self):
        try:
            status, payload = self._render_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_render_queue)
            return
        self.render_progress.stop()
        self.render_button.config(state="normal")
        if status == "done":
            self.app.status_var.set(f"Video saved: {payload}")
            messagebox.showinfo("Video saved", f"Saved to:\n{payload}")
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

        ttk.Label(bar, text="Session:").pack(side="left")
        self.session_var = tk.StringVar()
        self.session_cb = ttk.Combobox(bar, textvariable=self.session_var, state="readonly", width=28)
        self.session_cb.pack(side="left", padx=(4, 16))
        self.session_cb.bind("<<ComboboxSelected>>", lambda e: self.on_session_selected())

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
        self.session_cb["values"] = []
        self.session_var.set("")
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
        self.session_cb["values"] = labels
        if labels:
            self.session_var.set(labels[0])
        self.on_session_selected()

    def on_session_selected(self):
        self._notify_tabs("on_session_changed")

    def _notify_tabs(self, hook_name):
        for tab in self._tabs:
            hook = getattr(tab, hook_name, None)
            if hook is not None:
                hook()

    def get_selected_subject_session(self):
        subject_id = self.subject_var.get() or None
        session = self._sessions_by_label.get(self.session_var.get())
        return subject_id, session

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
        traceback.print_exc()
        messagebox.showerror("Error", str(exc))


def main():
    app = BehaviorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
