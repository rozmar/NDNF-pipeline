"""DataJoint connection handling for the GUIs.

Stores the path to the user's real dj_local_conf.json (the file with database
credentials, e.g. dj_local_conf_NDNF_behavior_KOKI.json) in gui_config.json,
next to this file. gui_config.json is git-ignored since the path is
machine/user specific; gui_config.example.json documents the format.
"""
import json
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
GUI_CONFIG_PATH = GUI_DIR / "gui_config.json"


def load_gui_config():
    if GUI_CONFIG_PATH.exists():
        with open(GUI_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_gui_config(config):
    with open(GUI_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_dj_config_path():
    return load_gui_config().get("dj_config_path")


def set_dj_config_path(path):
    config = load_gui_config()
    config["dj_config_path"] = str(path)
    save_gui_config(config)


def connect(dj_config_path):
    """Load the DataJoint config from dj_config_path and connect.

    Returns (connection, lab, experiment) so callers get already-connected
    schema modules; experiment is imported here (not at module load time)
    since it requires a working connection/config.
    """
    import datajoint as dj
    dj.config.load(dj_config_path)
    conn = dj.conn(reset=True)
    from ndnf_pipeline import lab, experiment
    return conn, lab, experiment
