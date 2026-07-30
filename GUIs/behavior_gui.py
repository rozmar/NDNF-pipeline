"""Tkinter GUI for browsing NDNF behavior pipeline data.

Connects to the DataJoint server (credentials come from a dj_local_conf.json
whose path is remembered in gui_config.json, see gui_config.example.json),
then lets you pick a subject and session and view plots for it across tabs.

Run with:  python GUIs/behavior_gui.py
"""
import sys
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("Agg")  # figures are built headless, then re-parented into a Tk canvas below
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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
                uniform_force_range=self.range_control.get_range())

        self.app.run_plot(work, self.panel)


class BlockDetailTab(ttk.Frame):
    """4-panel detail figure for a single block of the selected session."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=5, pady=5)
        ttk.Label(controls, text="Block:").pack(side="left")
        self.block_var = tk.StringVar()
        self.block_cb = ttk.Combobox(controls, textvariable=self.block_var, state="readonly", width=8)
        self.block_cb.pack(side="left", padx=(4, 10))
        self.block_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        self.subtract_median = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Subtract force median", variable=self.subtract_median,
                         command=self.refresh).pack(side="left")
        self.range_control = UniformRangeControl(controls, on_change=self.refresh)
        self.range_control.pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left", padx=(10, 0))
        self.panel = PlotPanel(self)
        self.panel.pack(fill="both", expand=True)

    def on_session_changed(self):
        subject_id, session = self.app.get_selected_subject_session()
        self.block_cb["values"] = []
        self.block_var.set("")
        if subject_id is None or session is None:
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
            self.refresh()
        else:
            self.panel.clear()

    def refresh(self):
        subject_id, session = self.app.get_selected_subject_session()
        block_str = self.block_var.get()
        if subject_id is None or session is None or not block_str:
            return
        block = int(block_str)
        from ndnf_pipeline.plot.behavior_plots import plot_block_force_figure

        def work():
            return plot_block_force_figure(
                subject_id, session, block,
                subtract_force_median=self.subtract_median.get(),
                force_uniform_range=self.range_control.enabled,
                uniform_force_range=self.range_control.get_range())

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
        self.notebook.add(self.session_overview_tab, text="Session Overview")
        self.notebook.add(self.block_detail_tab, text="Block Detail")
        self.notebook.add(self.subject_trend_tab, text="Subject Trend")
        self.notebook.add(self.trials_per_mouse_tab, text="Trials per Mouse")
        self.notebook.add(self.water_restriction_tab, text="Water Restriction")
        self._tabs = (self.session_overview_tab, self.block_detail_tab, self.subject_trend_tab,
                       self.trials_per_mouse_tab, self.water_restriction_tab)

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
            self.subject_var.set(subject_ids[0])
        if subject_ids:
            self.on_subject_selected()

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
