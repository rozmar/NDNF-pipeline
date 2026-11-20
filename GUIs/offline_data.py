"""Downloading sessions to local disk, so the behavior GUI browses them without the network.

A download folder (picked by the user, remembered in gui_config.json under "download_dirs")
holds:
- queries/: the result of every database query the GUI makes when showing a session, saved
  by dj_connection.QueryCache.recording() and served from there afterwards (no round trip).
- videos/: optionally, the session's raw video files, laid out like dj.config['path.raw_data'];
  videography_plots.resolve_video_path() prefers these over the network share.
- extras/<subject>_s<session>.json: what the GUI reads from files rather than the database
  (the Instantaneous setting recorded in the raw data on Z:, the training-log entries from the
  NDNF procedures sheet with their preset match), read before trying Z:.
- downloads.json: which sessions are in the folder, and which query/video files are theirs.

When adding something to the GUI that shows per-session data, add it to download_session_data
too: database queries are covered by calling the same function inside the recording() block;
anything read from files (Z:, the sheet CSV, the presets folder) goes into extras.

The cache is keyed on the exact SQL, so a download only helps if it runs the very same queries
the GUI does - which is why download_session_data() calls the same plot functions, and why the
GUI's own session/block/trial list queries live here as functions both share.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import dj_connection

MANIFEST_NAME = "downloads.json"
COPY_CHUNK_BYTES = 8 * 1024 ** 2


# --- the GUI's own small per-session queries (shared with download_session_data) ---
def session_blocks(experiment, subject_id, session):
    return sorted((experiment.Block() & {"subject_id": subject_id, "session": session}).fetch("block").tolist())


def session_block_feedback_types(experiment, subject_id, session):
    """[(block, feedback_type)] for the session, ordered by block. A '*_position' feedback type
    means the block ran with the Instantaneous setting ON; '*_speed' means it was OFF
    (accumulating/speed feedback)."""
    blocks, feedback_types = (experiment.Block() & {"subject_id": subject_id, "session": session}
                              ).fetch("block", "feedback_type", order_by="block")
    return [(int(b), str(f)) for b, f in zip(blocks.tolist(), feedback_types.tolist())]


_SESSION_FOLDER_FORMATS = ('%Y%m%dT%H%M%S', '%Y-%m-%dT%H%M%SZ', '%Y-%m-%dT%H%M%S.%fZ')  # as ingest_behavior
_recorded_instantaneous_cache = {}


def _session_raw_folders(subject_id, rig, session_date):
    """The raw behavior folders ingest_behavior assigns to this session: every
    <raw_data>/behavior/<rig>/<subject>/<subject>_<UTC time> folder whose local date (Budapest,
    or Los Angeles for AIND rigs) is the session date. None if the raw data isn't reachable."""
    import datajoint as dj
    from datetime import timezone
    from zoneinfo import ZoneInfo
    raw = dj.config.get("path.raw_data")
    if not raw:
        return None
    subject_dir = Path(raw) / "behavior" / rig / subject_id
    if not subject_dir.is_dir():
        return None
    tz = ZoneInfo("America/Los_Angeles" if "AIND" in rig else "Europe/Budapest")
    folders = []
    for folder in sorted(subject_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(subject_id + "_"):
            continue
        stamp = folder.name[len(subject_id) + 1:]
        for fmt in _SESSION_FOLDER_FORMATS:
            try:
                local = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc).astimezone(tz)
                break
            except ValueError:
                continue
        else:
            continue
        if local.date() == session_date:
            folders.append(folder)
    return folders


def _folder_action_types(folder):
    """action_type ('instantaneous' / 'integrated' / None) of every block that actually started
    in this raw session folder, from the ActiveBlock software events logged during the run;
    falls back to the blocks declared in the saved task logic when there are no such events."""
    events_file = folder / "behavior" / "SoftwareEvents" / "ActiveBlock.json"
    types = []
    if events_file.exists():
        with open(events_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trial_statistics = json.loads(line).get("data", {}).get("trial_statistics", {})
                types.append(((trial_statistics.get("response_period") or {}).get("action") or {}).get("action_type"))
        if types:
            return types
    for rel in ("behavior/Logs/tasklogic_output.json", "other/Config/tasklogic_output.json"):
        logic_file = folder / rel
        if logic_file.exists():
            with open(logic_file, encoding="utf-8") as f:
                logic = json.load(f)
            for block in logic.get("task_parameters", {}).get("environment", {}).get("block_statistics", []):
                types.append(((block.get("trial_statistics", {}).get("response_period") or {}).get("action") or {}
                              ).get("action_type"))
            return types
    return types


def _extras_path(folder, subject_id, session):
    return Path(folder) / "extras" / f"{subject_id}_s{session}.json"


def _downloaded_extra(subject_id, session, name):
    """A value saved by download_session_data for this session (the parts of the GUI that read
    files rather than the database: the raw-data Instantaneous setting and the training-log
    sheet), or None if the session isn't downloaded or predates that."""
    entry = downloaded_sessions(subject_id).get(session)
    if entry is None:
        return None
    try:
        with open(_extras_path(entry["folder"], subject_id, session), encoding="utf-8") as f:
            return json.load(f).get(name)
    except (OSError, ValueError):
        return None


def session_recorded_instantaneous(experiment, subject_id, session):
    """[instantaneous? (True/False/None=not recorded)] for each block the task actually ran in
    this session, read from the raw data the Telekinesis task saved - the "Instantaneous"
    checkbox of task_gui.py, as it was at run time. None if the raw data isn't reachable."""
    import pandas as pd
    downloaded = _downloaded_extra(subject_id, session, "instantaneous_recorded")
    if downloaded is not None:
        return downloaded
    rig, session_date = (experiment.Session() & {"subject_id": subject_id, "session": session}
                         ).fetch1("rig", "session_date")
    session_date = pd.Timestamp(session_date).date()
    key = (subject_id, rig, session_date)
    if key in _recorded_instantaneous_cache:
        return _recorded_instantaneous_cache[key]
    folders = _session_raw_folders(subject_id, rig, session_date)
    if not folders:
        return None
    flags = []
    for folder in folders:
        for action_type in _folder_action_types(folder):
            flags.append(True if action_type == "instantaneous" else False if action_type == "integrated" else None)
    # sessions from earlier days don't change any more; today's may still be running
    if session_date < datetime.now().date():
        _recorded_instantaneous_cache[key] = flags
    return flags


PROCEDURES_LOG_CSV ="NDNF procedures_Water Restriction.csv"
_procedures_csv_cache = {}  # path -> (mtime, DataFrame)


def _read_procedures_csv():
    """The 'Water Restriction' tab of the NDNF procedures Google sheet, as synced to
    dj.config['path.metadata'] by google_notebook.update_metadata - or None if unreachable
    (e.g. offline, no Z: drive). Re-read only when the file changes."""
    import datajoint as dj
    import pandas as pd
    metadata_path = dj.config.get("path.metadata")
    if not metadata_path:
        return None
    path = Path(metadata_path) / PROCEDURES_LOG_CSV
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _procedures_csv_cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    df = pd.read_csv(path)
    df["_datetime"] = pd.to_datetime(df["Timestamp"].astype(str).str.strip(),
                                     format="%m/%d/%Y %H:%M:%S", errors="coerce")
    _procedures_csv_cache[str(path)] = (mtime, df)
    return df


def session_procedures_log(experiment, lab, subject_id, session, use_sheet=True, use_download=True):
    """(session_date, entries, source) - the training-log entries for this subject on the
    session's date. Each entry is a dict with the sheet's fields (missing ones = None).

    source is 'sheet' when read from the synced Google sheet CSV (every field), or 'datajoint'
    when falling back to lab.WaterRestriction.WaterRestrictionLog, which only holds weight,
    weight after watering, water given and notes (no training type / rig / behavior notes /
    preset / water earned)."""
    import pandas as pd
    if use_sheet and use_download:
        downloaded = _downloaded_extra(subject_id, session, "training_log")
        if downloaded is not None:
            return (datetime.strptime(downloaded["session_date"], "%Y-%m-%d").date(),
                    downloaded["entries"], downloaded["source"])
    session_date = (experiment.Session() & {"subject_id": subject_id, "session": session}
                    ).fetch1("session_date")
    session_date = pd.Timestamp(session_date).date()

    def clean(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        return text or None

    df = None
    if use_sheet:
        try:
            df = _read_procedures_csv()
        except Exception:
            df = None
    if df is not None:
        rows = df[(df["Mouse ID"].astype(str).str.strip() == subject_id)
                  & (df["_datetime"].dt.date == session_date)].sort_values("_datetime")
        entries = [{
            "time": row["_datetime"].strftime("%H:%M"),
            "experimenter": clean(row.get("Experimenter")),
            "training_type": clean(row.get("Training type")),
            "rig": clean(row.get("Rig")),
            "preset": clean(row.get("Preset used")),
            "weight": clean(row.get("Weight (g)")),
            "weight_after": clean(row.get("Weight after watering (g)")),
            "water_given": clean(row.get("Water given (ml)")),
            "water_earned": clean(row.get("Water earned during the task (ml)")),
            "notes": clean(row.get("Notes")),
            "behavior_notes": clean(row.get("Behavior notes")),
        } for _, row in rows.iterrows()]
        return session_date, entries, "sheet"

    log = (lab.WaterRestriction.WaterRestrictionLog() & {"subject_id": subject_id}
           & f'DATE(log_datetime) = "{session_date.isoformat()}"')
    entries = [{
        "time": pd.Timestamp(r["log_datetime"]).strftime("%H:%M"),
        "experimenter": clean(r["user_name"]),
        "training_type": None, "rig": None, "preset": None,
        "weight": clean(r["weight"]),
        "weight_after": clean(r["weight_after_watering"]),
        "water_given": clean(r["water_given"]),
        "water_earned": clean(r["water_earned"]),
        "notes": clean(r["notes"]),
        "behavior_notes": None,
    } for r in log.fetch(as_dict=True, order_by="log_datetime")]
    return session_date, entries, "datajoint"


PRESETS_DIR_DEFAULT = r"C:\Aind.Behavior.Telekinesis\local\presets"
_presets_cache = {"key": None, "presets": {}}  # key = (dir, newest mtime, file count)


def presets_dir():
    """The Telekinesis task GUI's presets folder (gui_config.json "presets_dir" overrides)."""
    return Path(dj_connection.load_gui_config().get("presets_dir") or PRESETS_DIR_DEFAULT)


def _normalize_preset_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _load_presets():
    """{normalized name: (file stem, [instantaneous_mode per block])} for every preset JSON
    (subfolders included), re-read whenever a file is added or changed. A block's setting
    falls back to the preset's top-level value, as the task GUI does ({**preset, **block})."""
    folder = presets_dir()
    files = sorted(folder.rglob("*.json")) if folder.is_dir() else []
    key = (str(folder), max((f.stat().st_mtime for f in files), default=0), len(files))
    if _presets_cache["key"] == key:
        return _presets_cache["presets"]
    presets = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                params = json.load(fp)
        except (OSError, ValueError):
            continue
        blocks = params.get("blocks") or [{}]
        flags = [bool({**params, **block}.get("instantaneous_mode", False)) for block in blocks]
        presets.setdefault(_normalize_preset_name(f.stem), (f.stem, flags))
    _presets_cache.update(key=key, presets=presets)
    return presets


def preset_instantaneous(preset_name):
    """(matched preset file name, exact match?, [instantaneous_mode per block]) for a "Preset
    used" entry of the training log, or None if no preset file matches. Names typed in the
    sheet are matched ignoring case/spaces/punctuation, then as a suffix of a file name
    ('Deletion_2' -> '33_Deletion_2'), then by closest spelling ('Shspe_Dist_01')."""
    import difflib
    if not preset_name:
        return None
    presets = _load_presets()
    wanted = _normalize_preset_name(preset_name)
    if not wanted:
        return None
    if wanted in presets:
        stem, flags = presets[wanted]
        return stem, True, flags
    suffix_matches = [k for k in presets if k.endswith(wanted)]
    if len(suffix_matches) == 1:
        stem, flags = presets[suffix_matches[0]]
        return stem, False, flags
    close = difflib.get_close_matches(wanted, list(presets), n=1, cutoff=0.85)
    if close:
        stem, flags = presets[close[0]]
        return stem, False, flags
    return None


def _cache_datajoint_procedures_log(experiment, subject_id, session):
    """Run the DataJoint fallback of session_procedures_log so a download keeps it - offline,
    the synced sheet CSV is usually out of reach."""
    from ndnf_pipeline import lab
    return session_procedures_log(experiment, lab, subject_id, session, use_sheet=False)


def block_trials(experiment, subject_id, session, block):
    key = {"subject_id": subject_id, "session": session}
    if block is not None:
        key["block"] = block
    return sorted((experiment.BehaviorTrial() & key).fetch("trial").tolist())


def session_trials_with_blocks(experiment, subject_id, session):
    trials, trial_blocks = (experiment.BehaviorTrial() & {"subject_id": subject_id, "session": session}
                            ).fetch("trial", "block", order_by="trial")
    return [(int(t), int(b)) for t, b in zip(trials.tolist(), trial_blocks.tolist())]


def session_cameras(subject_id, session):
    from ndnf_pipeline import videography
    return sorted(set((videography.VideoRecording()
                       & {"subject_id": subject_id, "session": session}).fetch("device").tolist()))


# --- download folders ---
def download_dirs():
    return [Path(d) for d in dj_connection.load_gui_config().get("download_dirs", [])]


def register_download_dir(folder):
    """Serve folder's downloaded queries and videos (and remember it for next time)."""
    from ndnf_pipeline.plot import videography_plots
    folder = Path(folder)
    cfg = dj_connection.load_gui_config()
    dirs = cfg.get("download_dirs", [])
    if str(folder) not in dirs:
        cfg["download_dirs"] = dirs + [str(folder)]
        dj_connection.save_gui_config(cfg)
    dj_connection.query_cache.add_download_dir(folder / "queries")
    videos = str(folder / "videos")
    if videos not in videography_plots.LOCAL_VIDEO_ROOTS:
        videography_plots.LOCAL_VIDEO_ROOTS.append(videos)


def register_saved_download_dirs():
    for folder in download_dirs():
        if folder.is_dir():
            register_download_dir(folder)


def load_manifest(folder):
    path = Path(folder) / MANIFEST_NAME
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"sessions": {}}


def save_manifest(folder, manifest):
    path = Path(folder) / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _session_id(subject_id, session):
    return f"{subject_id}/{session}"


def downloaded_sessions(subject_id):
    """{session: manifest entry (plus "folder")} for every downloaded session of subject_id."""
    found = {}
    for folder in download_dirs():
        for entry in load_manifest(folder)["sessions"].values():
            if entry["subject_id"] == subject_id:
                found[entry["session"]] = {**entry, "folder": str(folder)}
    return found


# --- downloading ---
def download_session_data(experiment, folder, subject_id, session, include_video_data):
    """Run (and save into folder/queries) every query the GUI makes for this session: what the
    Session Overview / Block Detail tabs and their block/trial lists need, and - if
    include_video_data - what the Generate Video tab needs for any trial range and camera.

    Must run on the GUI's background worker thread, like every other query (the DataJoint
    connection isn't thread-safe). Returns (query keys saved, VideoFile.file_path of each of the
    session's video files, problems: messages for any view that couldn't be built - those fail
    the same way in the GUI itself, e.g. a block without force data).
    """
    import matplotlib.pyplot as plt
    from ndnf_pipeline.plot.behavior_plots import (
        plot_session_blocks_overview, plot_session_force_figure, plot_block_force_figure,
        estimate_force_sample_interval)

    problems = []

    def run(label, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            problems.append(f"{label}: {exc}")
            return None
        if isinstance(result, plt.Figure):
            plt.close(result)
        return result

    with dj_connection.query_cache.recording(Path(folder) / "queries") as saved_keys:
        blocks = session_blocks(experiment, subject_id, session)
        session_block_feedback_types(experiment, subject_id, session)
        run("training log", _cache_datajoint_procedures_log, experiment, subject_id, session)
        all_trials = session_trials_with_blocks(experiment, subject_id, session)
        block_trials(experiment, subject_id, session, None)
        cameras = session_cameras(subject_id, session)
        run("session overview", plot_session_blocks_overview, subject_id, [session])
        run("all blocks", plot_session_force_figure, subject_id, session)
        for block in blocks:
            block_trials(experiment, subject_id, session, block)
            run(f"block {block}", plot_block_force_figure, subject_id, session, block)
            run(f"block {block} sample interval", estimate_force_sample_interval, subject_id, session, block)

        video_files = []
        if cameras:
            import datajoint as dj
            from ndnf_pipeline import videography
            from ndnf_pipeline.plot import videography_plots
            video_files = (videography.VideoFile() & {"subject_id": subject_id, "session": session}
                           ).fetch("file_path").tolist()
            if include_video_data and all_trials:
                key = {"subject_id": subject_id, "session": session}
                for camera in cameras:
                    # one call per block plus one across the session covers every per-trial query
                    # (they're keyed on the trial alone) any trial range of the Generate Video tab
                    # makes; only the video-file lookup below stops at the range's first trial
                    for block in blocks:
                        n_trials = len(block_trials(experiment, subject_id, session, block))
                        if n_trials:
                            run(f"video data, {camera} block {block}", videography_plots.load_trial_video_data,
                                subject_id, session, block, 0, n_trials - 1, camera)
                    run(f"video data, {camera} all blocks", videography_plots.load_trial_video_data,
                        subject_id, session, None, all_trials[0][0], all_trials[-1][0], camera)
                    for trial, _ in all_trials:
                        try:
                            videography_plots._camera_video_file(videography, dj, key, camera, [trial])
                        except RuntimeError:
                            pass  # this trial has no video for this camera
        saved = list(saved_keys)

    # the parts of the GUI that read files instead of the database - saved as they are now, so a
    # downloaded session shows them without Z: (re-download to refresh)
    extras = {}
    instantaneous = run("instantaneous setting (raw data on Z:)", _live_recorded_instantaneous,
                        experiment, subject_id, session)
    if instantaneous is not None:
        extras["instantaneous_recorded"] = instantaneous
    training_log = run("training log (procedures sheet)", session_procedures_log,
                       experiment, _lab_module(), subject_id, session, use_download=False)
    if training_log is not None:
        session_date, entries, source = training_log
        for e in entries:
            e["preset_match"] = preset_instantaneous(e.get("preset"))
        extras["training_log"] = {"session_date": session_date.isoformat(), "entries": entries,
                                  "source": source}
    path = _extras_path(folder, subject_id, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(extras, f, indent=2)
    return saved, video_files, problems


def _lab_module():
    from ndnf_pipeline import lab
    return lab


def _live_recorded_instantaneous(experiment, subject_id, session):
    """session_recorded_instantaneous, read from Z: even if the session was downloaded before."""
    import pandas as pd
    rig, session_date = (experiment.Session() & {"subject_id": subject_id, "session": session}
                         ).fetch1("rig", "session_date")
    folders = _session_raw_folders(subject_id, rig, pd.Timestamp(session_date).date())
    if not folders:
        raise RuntimeError("raw data folder not found")
    return [True if t == "instantaneous" else False if t == "integrated" else None
            for folder in folders for t in _folder_action_types(folder)]


def copy_videos(file_paths, folder, progress=None):
    """Copy the given VideoFile.file_path files from the raw-data share to folder/videos,
    skipping any already there in full. progress(bytes_done, bytes_total) is called as it goes.
    Returns the file_paths copied (or already present)."""
    import datajoint as dj
    from ndnf_pipeline.plot import videography_plots
    jobs = []
    for file_path in file_paths:
        relative = file_path.replace("\\", "/")
        source = videography_plots._resolve_case_insensitive(dj.config["path.raw_data"], relative)
        target = Path(folder) / "videos" / Path(*[p for p in relative.split("/") if p])
        jobs.append((file_path, Path(source), target, os.path.getsize(source)))
    total = sum(size for *_, size in jobs)
    done = 0
    for _, source, target, size in jobs:
        if not (target.exists() and target.stat().st_size == size):
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with open(source, "rb") as src, open(tmp, "wb") as dst:
                while chunk := src.read(COPY_CHUNK_BYTES):
                    dst.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            os.replace(tmp, target)
        else:
            done += size
        if progress:
            progress(done, total)
    return [file_path for file_path, *_ in jobs]


def record_download(folder, subject_id, session, session_date, query_keys, videos):
    """Note in folder's manifest that this session is downloaded. A re-download replaces the
    session's previous entry; query files only it used are deleted."""
    manifest = load_manifest(folder)
    sid = _session_id(subject_id, session)
    old = manifest["sessions"].get(sid)
    manifest["sessions"][sid] = {
        "subject_id": subject_id, "session": session, "session_date": str(session_date),
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "queries": sorted(query_keys), "videos": sorted(videos)}
    if old is not None:
        _delete_unused(folder, manifest, old["queries"], old["videos"])
    save_manifest(folder, manifest)


def remove_download(folder, subject_id, session):
    """Delete a downloaded session's files from folder (any shared with another downloaded
    session stay) - the GUI then reads that session from the database again."""
    manifest = load_manifest(folder)
    entry = manifest["sessions"].pop(_session_id(subject_id, session), None)
    if entry is None:
        return
    _delete_unused(folder, manifest, entry["queries"], entry["videos"])
    _extras_path(folder, subject_id, session).unlink(missing_ok=True)
    save_manifest(folder, manifest)


def _delete_unused(folder, manifest, query_keys, videos):
    still_used_queries = {k for e in manifest["sessions"].values() for k in e["queries"]}
    still_used_videos = {v for e in manifest["sessions"].values() for v in e["videos"]}
    unused = [k for k in query_keys if k not in still_used_queries]
    dj_connection.query_cache.forget_downloaded(unused)
    for key in unused:
        (Path(folder) / "queries" / f"{key}.pkl").unlink(missing_ok=True)
    for file_path in videos:
        if file_path not in still_used_videos:
            relative = [p for p in file_path.replace("\\", "/").split("/") if p]
            (Path(folder) / "videos" / Path(*relative)).unlink(missing_ok=True)

