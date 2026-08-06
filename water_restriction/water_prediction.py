"""
Water restriction water-amount recommender.

Small desktop app (Tkinter) that reproduces the pipeline built in
Notebooks/water_prediction_model.ipynb:

  1. refresh the data (sync metadata from Google Drive, then ingest it
     into the DataJoint database) -- its own button, since you often want
     to retrain without re-syncing/re-ingesting every time,
  2. rebuild the "transitions" training table (weight/water-restriction
     log entries + matching environment sensor readings) and fit the
     mixed-effects decay model on all of it, with all predictors
     standardized exactly as in the notebook -- also its own button,
  3. let the user type in a mouse's current state and get a recommended
     water amount (mL) back.

Run it with the same Python environment used for the notebook, e.g.:

    /Users/moldor/Documents/KOKI/NDNF-pipeline/.venv/bin/python water_prediction.py

"""

import os
import sys
import queue
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
except ImportError:
    sys.exit(
        "Tkinter is not available in this Python installation, so the GUI can't start.\n"
        "This project's .venv uses a uv-managed standalone Python build, which on macOS "
        "sometimes ships without Tk support.\n"
        "Fixes to try:\n"
        "  1) Run this script with your Mac's system Python 3 instead "
        "(e.g. `/usr/bin/python3 water_prediction.py`, after `pip install --user "
        "datajoint pandas numpy statsmodels` there), or\n"
        "  2) Install Tk for this venv's Python, e.g. `brew install tcl-tk` and "
        "reinstall/rebuild the venv's Python with Tk support (uv python install "
        "--reinstall may be needed), or\n"
        "  3) `brew install python-tk@3.11` and point this venv at that Python build."
    )

# --------------------------------------------------------------------------
# Configuration (mirrors the notebook's manual choices)
# --------------------------------------------------------------------------

# Same local DataJoint config file used in Notebooks/water_prediction_model.ipynb
DJ_CONFIG_PATH = os.environ.get(
    "NDNF_DJ_CONFIG",
    "/Users/moldor/Documents/KOKI/secrets/dj_local_conf_NDNF_behavior_KOKI.json",
)

# The notebook picks one environment sensor / one consecutive recording block
# by hand ("Replacing sensors is not yet handled."). Keep the same choice here.
SENSOR_ID = "NDNF-#3"
GROUP_ID = 0
GAP_THRESHOLD_HOURS = 1.5

# Default training window shown (and used) in the GUI -- this is the period
# that's effectively in use right now. Edit these two strings, or just change
# them in the GUI itself before clicking "Refresh data & train model"; leaving
# either field blank in the GUI falls back to auto-detecting the sensor's
# first consecutive recording block (the original notebook behaviour).
DEFAULT_WINDOW_START = "2026-05-28 14:00:00"
DEFAULT_WINDOW_END = "2026-07-21 23:59:59"

# Model formula, adapted from the notebook's last mixedlm fit: mean_humidity
# and age_days dropped (by request) so the user doesn't need to supply them.
FULL_FORMULA = (
    "decay ~ standardize(mean_temp) "
    "+ standardize(weight_t) + standardize(wr_start_weight) "
    "+ standardize(water_given_t) + standardize(elapsed_days) "
    "+ C(sex)"
)

# Bisection search bounds for recommended water (mL).
WATER_SEARCH_LO = 0.0
WATER_SEARCH_HI = 5.0
WATER_SEARCH_TOL = 1e-3

DEFAULT_TARGET_PCT = 85.0
DEFAULT_DAYS_UNTIL_NEXT = 1.0


def _ensure_repo_on_path():
    """Same trick as the notebook's first cell: find the repo root (the
    parent directory that contains the ndnf_pipeline package) and put it
    on sys.path so `from ndnf_pipeline import ...` works regardless of
    where this script lives."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parent.parents]:
        if (parent / "ndnf_pipeline").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError(
        "Could not locate the ndnf_pipeline package. Place this script "
        "somewhere inside the NDNF-pipeline repository."
    )


_ensure_repo_on_path()


# --------------------------------------------------------------------------
# Data refresh / ingest / model training
# (these are plain functions so they can be run from a background thread)
# --------------------------------------------------------------------------

def connect_datajoint(log):
    import datajoint as dj

    log(f"Loading DataJoint config from {DJ_CONFIG_PATH} ...")
    dj.config.load(DJ_CONFIG_PATH)
    dj.conn()
    log("Connected to DataJoint.")
    return dj


def refresh_metadata(dj, log, sync_google=True):
    """Mirrors ndnf_pipeline_main.py steps 2-3, minus behavior-session
    ingestion (not needed for the water model, and it requires raw
    session folders that may not be available on this machine)."""
    if sync_google:
        from ndnf_pipeline.utils.google_notebook import update_metadata

        spreadsheet_names = dj.config.get("metadata.spreadsheet_names", [])
        if not spreadsheet_names:
            log("No 'metadata.spreadsheet_names' configured, skipping Google sync.")
        for spreadsheet in spreadsheet_names:
            log(f"Syncing metadata spreadsheet '{spreadsheet}' from Google Drive ...")
            update_metadata(
                spreadsheet,
                dj.config["path.metadata"],
                dj.config["path.google_creds_json"],
            )
    else:
        log("Skipping Google Drive sync (using metadata files already on disk).")

    from ndnf_pipeline.ingest.ingest_metadata import ingest_metadata

    log("Ingesting metadata (subjects, water restriction logs, environment sensors) ...")
    ingest_metadata(dj)
    log("Ingest complete.")


def build_transitions(log, window_start=None, window_end=None):
    """Rebuild the 'transitions' table exactly as in the notebook: one row
    per consecutive pair of water-restriction log entries for a mouse,
    restricted to a training time window, with matching mean temperature /
    humidity merged in.

    If window_start / window_end are given (as strings or Timestamps), they
    are used directly as the training window. If either is left as None, it
    falls back to the notebook's original behaviour of auto-detecting the
    sensor's first consecutive recording block (a run of readings with no
    gap longer than GAP_THRESHOLD_HOURS)."""
    from ndnf_pipeline import lab, environment

    log(f"Fetching recording timestamps for sensor {SENSOR_ID} ...")
    times = (environment.EnvSensorRecording & {"sensor_id": SENSOR_ID}).fetch(
        "recording_datetime"
    )
    times = pd.to_datetime(pd.Series(times)).sort_values().reset_index(drop=True)
    if times.empty:
        raise RuntimeError(f"No recordings found for sensor {SENSOR_ID}.")

    if window_start is not None and window_end is not None:
        env_start, env_end = pd.Timestamp(window_start), pd.Timestamp(window_end)
        log(f"Using user-specified training window {env_start} to {env_end} "
            f"(sensor {SENSOR_ID} has data from {times.min()} to {times.max()}).")
    else:
        log(f"No window given -- auto-detecting recording block {GROUP_ID} for sensor {SENSOR_ID} ...")
        gap_threshold = pd.Timedelta(hours=GAP_THRESHOLD_HOURS)
        group_ids = (times.diff() > gap_threshold).cumsum()
        first_run = times[group_ids == GROUP_ID]
        auto_start, auto_end = first_run.min(), first_run.max()
        env_start = pd.Timestamp(window_start) if window_start is not None else auto_start
        env_end = pd.Timestamp(window_end) if window_end is not None else auto_end
        log(f"Using window {env_start} to {env_end}.")

    log("Fetching water restriction logs in that window ...")
    wr_log = (
        lab.WaterRestriction.WaterRestrictionLog
        & f"log_datetime between '{env_start}' and '{env_end}'"
    ).fetch(format="frame").reset_index()
    if wr_log.empty:
        raise RuntimeError("No WaterRestrictionLog rows found in the sensor window.")
    for col in ["weight", "weight_after_watering", "water_given"]:
        wr_log[col] = pd.to_numeric(wr_log[col], errors="coerce")
    wr_log["log_datetime"] = pd.to_datetime(wr_log["log_datetime"])
    wr_log = wr_log.sort_values(["subject_id", "log_datetime"])

    dob_df = lab.Subject().fetch(format="frame")[["date_of_birth"]].reset_index()
    wr_log = wr_log.merge(dob_df, on="subject_id", how="left")
    wr_log["date_of_birth"] = pd.to_datetime(wr_log["date_of_birth"])
    wr_log["age_days"] = (wr_log["log_datetime"] - wr_log["date_of_birth"]).dt.days

    wr_master = lab.WaterRestriction().fetch(format="frame").reset_index()
    wr_master["wr_start_weight"] = pd.to_numeric(wr_master["wr_start_weight"], errors="coerce")
    wr_log = wr_log.merge(
        wr_master[["subject_id", "water_restriction_id", "wr_start_weight"]],
        on=["subject_id", "water_restriction_id"],
        how="left",
    )
    wr_log = wr_log.sort_values(["subject_id", "log_datetime"])

    log("Building weight/water transitions ...")
    rows = []
    for subject_id, grp in wr_log.groupby("subject_id"):
        grp = grp.sort_values("log_datetime").reset_index(drop=True)
        for i in range(len(grp) - 1):
            t, t1 = grp.loc[i], grp.loc[i + 1]
            if pd.isna(t["weight_after_watering"]):
                continue
            elapsed_days = (t1["log_datetime"] - t["log_datetime"]).total_seconds() / 86400
            if elapsed_days <= 0:
                continue
            decay = t["weight_after_watering"] - t1["weight"]
            rows.append(
                {
                    "subject_id": subject_id,
                    "t_start": t["log_datetime"],
                    "t_end": t1["log_datetime"],
                    "elapsed_days": elapsed_days,
                    "water_given_t": t["weight_after_watering"] - t["weight"],
                    "decay": decay,
                    "wr_start_weight": t["wr_start_weight"],
                    "weight_t": t["weight"],
                    "age_days": t["age_days"],
                    "net_change": t1["weight"] - t["weight"],
                }
            )
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        raise RuntimeError("No valid transitions could be built from the water restriction logs.")

    sex_df = lab.Subject().fetch(format="frame")[["sex"]].reset_index()[["subject_id", "sex"]]
    transitions = transitions.merge(sex_df, on="subject_id", how="left")

    log("Merging in mean temperature / humidity per transition ...")
    env_all = (
        (environment.EnvSensorRecording.Channel & {"sensor_id": SENSOR_ID})
        & "channel_name in ('Temperature', 'Humidity')"
    ).fetch(format="frame").reset_index()
    env_all["recording_datetime"] = pd.to_datetime(env_all["recording_datetime"])

    def get_env_means(t_start, t_end):
        mask = (env_all["recording_datetime"] >= t_start) & (env_all["recording_datetime"] < t_end)
        sub = env_all[mask]
        return pd.Series(
            {
                "mean_temp": sub.loc[sub["channel_name"] == "Temperature", "value_avg"].mean(),
                "mean_humidity": sub.loc[sub["channel_name"] == "Humidity", "value_avg"].mean(),
            }
        )

    env_features = transitions.apply(lambda r: get_env_means(r["t_start"], r["t_end"]), axis=1)
    transitions = pd.concat([transitions, env_features], axis=1)
    # mean_humidity is kept around (unused by FULL_FORMULA right now) but isn't
    # required to be present -- only mean_temp actually feeds into the model.
    transitions = transitions.dropna(subset=["mean_temp"])
    if transitions.empty:
        raise RuntimeError("All transitions were dropped for missing environment data.")

    log(f"Built {len(transitions)} transitions across {transitions['subject_id'].nunique()} mice.")
    return transitions


def fit_model(transitions, log):
    import statsmodels.formula.api as smf

    log("Fitting mixed-effects model (random intercept per mouse) ...")
    model = smf.mixedlm(FULL_FORMULA, data=transitions, groups=transitions["subject_id"])
    result = model.fit()
    log("Model fit complete.")
    log(str(result.summary()))
    return result


def compute_metrics(cv_results):
    actual = cv_results["actual"]
    predicted = cv_results["predicted"]
    error = actual - predicted
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())
    corr = actual.corr(predicted)
    ss_res = (error ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def cross_validate_leave_one_mouse_out(transitions, formula, target_col="decay"):
    import warnings
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    predictions = []
    for held_out in transitions["subject_id"].unique():
        train = transitions[transitions["subject_id"] != held_out]
        test = transitions[transitions["subject_id"] == held_out]
        if test.empty or train["subject_id"].nunique() < 2:
            continue
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", ConvergenceWarning)
            m = smf.mixedlm(formula, data=train, groups=train["subject_id"]).fit()
        pred = m.predict(test)
        predictions.append(pd.DataFrame({"subject_id": held_out, "actual": test[target_col], "predicted": pred}))
    if not predictions:
        return None, None
    cv_results = pd.concat(predictions)
    return cv_results, compute_metrics(cv_results)


def recommend_water(
    model_result,
    current_weight,
    wr_start_weight,
    target_pct,
    days_until_next,
    mean_temp_forecast,
    sex,
    subject_id=None,
    lo=WATER_SEARCH_LO,
    hi=WATER_SEARCH_HI,
    tol=WATER_SEARCH_TOL,
):
    """Corrected version of the notebook's recommend_water_search: finds the
    water amount (mL) such that the model's predicted weight at the next
    check-in is (approximately) target_pct% of the mouse's water-restriction
    starting weight. Uses exactly the predictors FULL_FORMULA was fit on
    (mean_humidity and age_days are intentionally not part of the model)."""
    target_weight_g = wr_start_weight * target_pct / 100.0

    b_i = 0.0
    random_effects = getattr(model_result, "random_effects", {})
    if subject_id is not None and subject_id in random_effects:
        b_i = random_effects[subject_id].get("Group", 0.0)

    def predicted_weight_next(water_given):
        new_row = pd.DataFrame(
            {
                "mean_temp": [mean_temp_forecast],
                "weight_t": [current_weight],
                "wr_start_weight": [wr_start_weight],
                "water_given_t": [water_given],
                "elapsed_days": [days_until_next],
                "sex": [sex],
            }
        )
        pred_decay = model_result.predict(new_row).iloc[0] + b_i
        return (current_weight + water_given) - pred_decay

    lo_val, hi_val = lo, hi
    reached_bound = False
    if predicted_weight_next(hi_val) < target_weight_g:
        reached_bound = True
    elif predicted_weight_next(lo_val) >= target_weight_g:
        # even giving (almost) no water keeps it above target
        hi_val = lo_val
    else:
        while hi_val - lo_val > tol:
            mid = (lo_val + hi_val) / 2
            if predicted_weight_next(mid) < target_weight_g:
                lo_val = mid
            else:
                hi_val = mid

    water_ml = (lo_val + hi_val) / 2
    predicted_weight = predicted_weight_next(water_ml)
    return water_ml, predicted_weight, target_weight_g, reached_bound, b_i


def fetch_subject_lookup(log):
    """subject_id -> {sex, wr_start_weight} for autofill when a subject is picked in the GUI."""
    from ndnf_pipeline import lab

    subjects = lab.Subject().fetch(format="frame")[["sex"]].reset_index()
    wr_master = lab.WaterRestriction().fetch(format="frame").reset_index()
    wr_master["wr_start_weight"] = pd.to_numeric(wr_master["wr_start_weight"], errors="coerce")
    merged = subjects.merge(wr_master[["subject_id", "wr_start_weight"]], on="subject_id", how="left")
    lookup = {}
    for _, row in merged.iterrows():
        lookup[row["subject_id"]] = {
            "sex": row["sex"],
            "wr_start_weight": row["wr_start_weight"],
        }
    log(f"Subject lookup ready ({len(lookup)} subjects).")
    return lookup


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class WaterPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Water restriction — recommended water amount")
        self.root.geometry("780x900")

        self.msg_queue = queue.Queue()
        self.model_result = None
        self.transitions = None
        self.subject_lookup = {}

        self._build_widgets()
        self.root.after(150, self._poll_queue)

    # ---- layout -----------------------------------------------------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        # --- 1. Refresh data ---
        refresh_frame = ttk.LabelFrame(self.root, text="1. Refresh data (sync + ingest)")
        refresh_frame.pack(fill="x", **pad)

        self.sync_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            refresh_frame, text="Also sync metadata from Google Drive (slower, needs internet + credentials)",
            variable=self.sync_var,
        ).pack(anchor="w", padx=8, pady=(6, 0))

        refresh_btn_row = ttk.Frame(refresh_frame)
        refresh_btn_row.pack(fill="x", padx=8, pady=6)
        self.refresh_btn = ttk.Button(refresh_btn_row, text="Refresh data", command=self.on_refresh)
        self.refresh_btn.pack(side="left")
        self.refresh_status_label = ttk.Label(refresh_btn_row, text="Not refreshed yet.")
        self.refresh_status_label.pack(side="left", padx=12)

        # --- 2. Train model ---
        train_frame = ttk.LabelFrame(self.root, text="2. Train model")
        train_frame.pack(fill="x", **pad)

        window_row = ttk.Frame(train_frame)
        window_row.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(window_row, text="Training window start:").grid(row=0, column=0, sticky="w")
        self.window_start_var = tk.StringVar(value=DEFAULT_WINDOW_START)
        ttk.Entry(window_row, textvariable=self.window_start_var, width=22).grid(row=0, column=1, padx=(4, 16))
        ttk.Label(window_row, text="Training window end:").grid(row=0, column=2, sticky="w")
        self.window_end_var = tk.StringVar(value=DEFAULT_WINDOW_END)
        ttk.Entry(window_row, textvariable=self.window_end_var, width=22).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(
            train_frame, text="(this is the period currently used to train the model; leave either box blank to "
            "auto-detect the sensor's first recording block instead)",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", padx=8, pady=(0, 4))

        train_btn_row = ttk.Frame(train_frame)
        train_btn_row.pack(fill="x", padx=8, pady=6)
        self.train_btn = ttk.Button(train_btn_row, text="Train model", command=self.on_train)
        self.train_btn.pack(side="left")
        self.train_status_label = ttk.Label(train_btn_row, text="Not trained yet.")
        self.train_status_label.pack(side="left", padx=12)

        # --- shared log for both steps above ---
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=False, **pad)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=10, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=8)

        form = ttk.LabelFrame(self.root, text="3. Enter the mouse's current state")
        form.pack(fill="x", **pad)

        self.fields = {}

        def add_field(row, label, key, default="", combo_values=None):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            if combo_values is not None:
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(form, textvariable=var, values=combo_values, width=30, state="normal")
            else:
                var = tk.StringVar(value=default)
                widget = ttk.Entry(form, textvariable=var, width=32)
            widget.grid(row=row, column=1, sticky="w", padx=8, pady=4)
            self.fields[key] = var
            return widget

        self.subject_combo = add_field(
            0, "Subject ID (pick a trained mouse, or leave/type a new one):", "subject_id", combo_values=[]
        )
        self.subject_combo.bind("<<ComboboxSelected>>", self.on_subject_selected)

        add_field(1, "Current weight (g):", "current_weight")
        add_field(2, "Water-restriction starting weight (g):", "wr_start_weight")
        add_field(3, "Target weight (% of starting weight):", "target_pct", default=str(DEFAULT_TARGET_PCT))
        add_field(4, "Days until next weighing:", "days_until_next", default=str(DEFAULT_DAYS_UNTIL_NEXT))
        add_field(5, "Forecast mean temperature (°C):", "mean_temp")
        self.sex_combo = add_field(6, "Sex:", "sex", combo_values=["M", "F"])

        self.compute_btn = ttk.Button(form, text="Compute recommended water", command=self.on_compute)
        self.compute_btn.grid(row=7, column=0, columnspan=2, pady=10)
        self.compute_btn.state(["disabled"])

        result_frame = ttk.LabelFrame(self.root, text="4. Result")
        result_frame.pack(fill="both", expand=True, **pad)
        self.result_label = ttk.Label(
            result_frame, text="Train the model first, then fill in the form above.",
            font=("TkDefaultFont", 12, "bold"), wraplength=700, justify="left",
        )
        self.result_label.pack(anchor="w", padx=8, pady=8)
        self.detail_label = ttk.Label(result_frame, text="", wraplength=700, justify="left")
        self.detail_label.pack(anchor="w", padx=8, pady=4)

    # ---- logging helpers ---------------------------------------------

    def _log(self, text):
        # called from the worker thread -> just queue it
        self.msg_queue.put(("log", text))

    def _append_log_line(self, text):
        # main-thread-only: actually write into the log widget
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_queue(self):
        # IMPORTANT: the `self.root.after(...)` reschedule lives in `finally`
        # so a bug in a message handler (which would otherwise silently kill
        # all future polling -- log/status updates included) can never stop
        # the loop; it just gets reported and polling continues.
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                try:
                    if kind == "log":
                        self._append_log_line(payload)
                    elif kind == "refresh_done":
                        self._on_refresh_done()
                    elif kind == "refresh_error":
                        self._on_refresh_error(payload)
                    elif kind == "train_done":
                        self._on_train_done(payload)
                    elif kind == "train_error":
                        self._on_train_error(payload)
                except Exception:
                    self._append_log_line(
                        f"Internal GUI error while handling a '{kind}' message:\n{traceback.format_exc()}"
                    )
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._poll_queue)

    # ---- refresh data ---------------------------------------------------

    def on_refresh(self):
        self.refresh_btn.state(["disabled"])
        self.refresh_status_label.config(text="Working ...")
        sync_google = self.sync_var.get()
        thread = threading.Thread(target=self._refresh_worker, args=(sync_google,), daemon=True)
        thread.start()

    def _refresh_worker(self, sync_google):
        try:
            dj = connect_datajoint(self._log)
            refresh_metadata(dj, self._log, sync_google=sync_google)
            self._log("Data refresh complete.")
            self.msg_queue.put(("refresh_done", None))
        except Exception:
            self.msg_queue.put(("refresh_error", traceback.format_exc()))

    def _on_refresh_done(self):
        self.refresh_status_label.config(text="Data refreshed. You can train the model now.")
        self.refresh_btn.state(["!disabled"])

    def _on_refresh_error(self, tb_text):
        self._append_log_line("ERROR during data refresh:\n" + tb_text)
        self.refresh_status_label.config(text="Refresh failed — see log.")
        self.refresh_btn.state(["!disabled"])
        messagebox.showerror("Data refresh failed", tb_text.splitlines()[-1])

    # ---- training -----------------------------------------------------

    def on_train(self):
        window_start_str = self.window_start_var.get().strip()
        window_end_str = self.window_end_var.get().strip()
        try:
            window_start = pd.Timestamp(window_start_str) if window_start_str else None
            window_end = pd.Timestamp(window_end_str) if window_end_str else None
        except (ValueError, TypeError) as exc:
            messagebox.showerror(
                "Invalid training window",
                f"Couldn't parse the training window start/end as dates: {exc}\n"
                "Use a format like 2026-05-28 14:00:00, or clear the box to auto-detect.",
            )
            return

        self.train_btn.state(["disabled"])
        self.compute_btn.state(["disabled"])
        self.train_status_label.config(text="Working ...")
        thread = threading.Thread(target=self._train_worker, args=(window_start, window_end), daemon=True)
        thread.start()

    def _train_worker(self, window_start, window_end):
        try:
            # cheap no-op if a connection from an earlier Refresh already exists
            connect_datajoint(self._log)
            transitions = build_transitions(self._log, window_start=window_start, window_end=window_end)
            result = fit_model(transitions, self._log)

            self._log("Running leave-one-mouse-out cross-validation for a sanity check ...")
            cv_results, errors = cross_validate_leave_one_mouse_out(transitions, FULL_FORMULA)
            if errors:
                self._log(
                    "LOMO CV -> MAE={mae:.3f} g, RMSE={rmse:.3f} g, R²={r2:.3f}, corr={corr:.3f}".format(**errors)
                )
            else:
                self._log("LOMO CV skipped (not enough mice).")

            self._log("Fetching subject lookup table for autofill ...")
            subject_lookup = fetch_subject_lookup(self._log)

            self._log("All done — the form below is now enabled.")
            self.msg_queue.put(
                ("train_done", {"transitions": transitions, "result": result, "subject_lookup": subject_lookup})
            )
        except Exception:
            self.msg_queue.put(("train_error", traceback.format_exc()))

    def _on_train_done(self, payload):
        self.transitions = payload["transitions"]
        self.model_result = payload["result"]
        self.subject_lookup = payload["subject_lookup"]

        subject_ids = sorted(self.transitions["subject_id"].unique())
        self.subject_combo["values"] = subject_ids

        sexes = sorted(x for x in self.transitions["sex"].dropna().unique())
        if sexes:
            self.sex_combo["values"] = sexes

        # default the forecast-temperature field to the average environmental
        # mean_temp seen across all training transitions -- a neutral
        # starting point; override it with an actual forecast if you have one.
        self.fields["mean_temp"].set(f"{self.transitions['mean_temp'].mean():.1f}")

        self.train_status_label.config(
            text=f"Trained on {len(self.transitions)} transitions from {len(subject_ids)} mice."
        )
        self.train_btn.state(["!disabled"])
        self.compute_btn.state(["!disabled"])
        self.result_label.config(text="Model ready. Fill in the form and click Compute.")

    def _on_train_error(self, tb_text):
        self._append_log_line("ERROR during training:\n" + tb_text)
        self.train_status_label.config(text="Training failed — see log.")
        self.train_btn.state(["!disabled"])
        messagebox.showerror("Training failed", tb_text.splitlines()[-1])

    # ---- autofill on subject selection --------------------------------

    def on_subject_selected(self, _event=None):
        subject_id = self.fields["subject_id"].get().strip()
        info = self.subject_lookup.get(subject_id)
        if not info:
            return
        if info.get("sex"):
            self.fields["sex"].set(info["sex"])
        if info.get("wr_start_weight") is not None and not pd.isna(info["wr_start_weight"]):
            self.fields["wr_start_weight"].set(f"{info['wr_start_weight']:.2f}")

    # ---- compute --------------------------------------------------------

    def on_compute(self):
        try:
            subject_id = self.fields["subject_id"].get().strip() or None
            current_weight = float(self.fields["current_weight"].get())
            wr_start_weight = float(self.fields["wr_start_weight"].get())
            target_pct = float(self.fields["target_pct"].get())
            days_until_next = float(self.fields["days_until_next"].get())
            mean_temp = float(self.fields["mean_temp"].get())
            sex = self.fields["sex"].get().strip()
        except ValueError:
            messagebox.showerror("Invalid input", "Please fill in every field with a valid number.")
            return

        if not sex:
            messagebox.showerror("Invalid input", "Please select or enter a sex (matching the training data).")
            return

        try:
            water_ml, predicted_weight, target_weight_g, reached_bound, b_i = recommend_water(
                self.model_result,
                current_weight=current_weight,
                wr_start_weight=wr_start_weight,
                target_pct=target_pct,
                days_until_next=days_until_next,
                mean_temp_forecast=mean_temp,
                sex=sex,
                subject_id=subject_id,
            )
        except Exception as exc:
            messagebox.showerror("Could not compute a recommendation", str(exc))
            return

        self.result_label.config(text=f"Recommended water: {water_ml:.2f} mL")
        note = ""
        if reached_bound:
            note = (
                f"\nNote: even {WATER_SEARCH_HI:g} mL (the search ceiling) isn't predicted to reach "
                "the target weight — the recommendation above is that ceiling, not a converged solution."
            )
        used_b_i = f"{b_i:.3f} g" if subject_id in getattr(self.model_result, "random_effects", {}) else "0 g (mouse not in training data)"
        self.detail_label.config(
            text=(
                f"Target weight: {target_weight_g:.2f} g  |  "
                f"Predicted weight at next check-in with this much water: {predicted_weight:.2f} g\n"
                f"Mouse-specific random intercept used: {used_b_i}{note}"
            )
        )


def main():
    root = tk.Tk()
    WaterPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
