"""DataJoint connection handling for the GUIs.

Stores the path to the user's real dj_local_conf.json (the file with database
credentials, e.g. dj_local_conf_NDNF_behavior_KOKI.json) in gui_config.json,
next to this file. gui_config.json is git-ignored since the path is
machine/user specific; gui_config.example.json documents the format.

Optional gui_config.json keys tuning the query cache (see QueryCache):
query_cache_mb, query_cache_ttl_min, query_cache_disk_mb (0 = no disk cache),
query_cache_dir.
"""
import contextlib
import hashlib
import json
import os
import pickle
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
GUI_CONFIG_PATH = GUI_DIR / "gui_config.json"

QUERY_CACHE_MB_DEFAULT = 2048
QUERY_CACHE_TTL_MIN_DEFAULT = 30
QUERY_CACHE_DISK_MB_DEFAULT = 5120
QUERY_CACHE_DIR_DEFAULT = str(Path.home() / ".cache" / "ndnf_pipeline" / "query_cache")


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
    for c in (conn, lab.schema.connection, experiment.schema.connection):
        query_cache.install(c)  # no-op for a connection it's already installed on
    return conn, lab, experiment


class QueryCache:
    """Cache of SELECT results, installed underneath a DataJoint connection.

    Every plot pulls the raw per-trial force/lickport blobs from the database, and the GUIs
    redraw on every selection/filter change - fine on the lab network, but from outside it
    (measured from home: ~260 ms per round trip, ~1 MB/s) each redraw re-downloads tens of
    MB and pays a round trip per query. Caching the raw rows (blobs still packed) means only
    the first look at a session hits the network.

    Two layers:
    - memory: every SELECT, LRU-capped at max_bytes. An entry older than ttl_s is refetched
      (small results) or revalidated (large ones, see below), so data still being ingested -
      e.g. a session recorded today - shows up without a manual refresh.
    - disk (disk_dir, capped at disk_max_bytes): results of at least DISK_MIN_BYTES, i.e. the
      blob fetches, so reopening the GUI doesn't re-download sessions already looked at.

    A large result coming from disk, or past its ttl in memory, is only reused once a
    SELECT COUNT(*) of the same query (one round trip, no blobs) still returns the cached
    row count - new trials appended to a session invalidate it. clear() (wired to the GUIs'
    Refresh / Reload buttons) empties both layers, and so does any write through the
    connection, so it never serves rows older than the GUI's own changes.

    On top of both: downloaded sessions (see offline_data.py). Queries run inside recording()
    are saved into a download folder, and every folder registered with add_download_dir() is
    served from as-is - no ttl, no COUNT(*) check, no eviction, not touched by clear() - so
    browsing a downloaded session needs no round trip at all. Re-downloading a session is how
    it gets refreshed.
    """
    DISK_MIN_BYTES = 256 * 1024
    _CACHEABLE = re.compile(r"\s*SELECT\b", re.IGNORECASE)
    _WRITE = re.compile(r"\s*(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|TRUNCATE)\b", re.IGNORECASE)
    _LOCKING = re.compile(r"\bFOR\s+UPDATE\b|\bLOCK\s+IN\s+SHARE\s+MODE\b", re.IGNORECASE)
    # DataJoint names subqueries `$1a2b`-style with a per-process counter, so the same
    # query built twice gets different aliases - strip them for the key (as dj's own
    # disk query cache does)
    _ALIAS = re.compile(r"`\$\w+`")
    # DataJoint's own schema lookups - never pinned into a download, they aren't session data
    _SCHEMA_LOOKUP = re.compile(r"\binformation_schema\b", re.IGNORECASE)

    def __init__(self, max_bytes, ttl_s, disk_dir=None, disk_max_bytes=0):
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        self.disk_dir = Path(disk_dir) if disk_dir and disk_max_bytes > 0 else None
        self.disk_max_bytes = disk_max_bytes
        self._lock = threading.Lock()
        self._entries = OrderedDict()  # key -> (checked_at, nbytes, rows)
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self._downloaded = {}  # key -> path of its pickle in a download folder
        self._recording = threading.local()  # .rows: key -> rows, while recording() on this thread

    def add_download_dir(self, queries_dir):
        """Serve every query saved in queries_dir (a download folder's queries/ subfolder)."""
        paths = {path.stem: path for path in Path(queries_dir).glob("*.pkl")}
        with self._lock:
            self._downloaded.update(paths)

    def forget_downloaded(self, keys):
        with self._lock:
            for key in keys:
                self._downloaded.pop(key, None)
                if key in self._entries:
                    self._drop(key)

    @contextlib.contextmanager
    def recording(self, queries_dir):
        """Within this block, SELECTs run on this thread go to the database (not the cache) and
        are saved to queries_dir, which is then served from like add_download_dir. Yields the
        set of keys saved, so the caller can tell which files belong to what it downloaded."""
        queries_dir = Path(queries_dir)
        queries_dir.mkdir(parents=True, exist_ok=True)
        self._recording.dir = queries_dir
        self._recording.rows = {}
        try:
            yield self._recording.rows.keys()
        finally:
            self._recording.dir = None
            self._recording.rows = None

    def install(self, conn):
        if getattr(conn, "_ndnf_query_cache", None) is self:
            return
        CachedCursor = _cached_cursor_class()
        uncached_query = conn.query
        host = f"{conn.conn_info.get('host')}:{conn.conn_info.get('port')}"

        def query(sql, args=(), *, as_dict=False, **kwargs):
            if self._WRITE.match(sql):
                self.clear()
            if not self._CACHEABLE.match(sql) or self._LOCKING.search(sql) or conn.in_transaction:
                return uncached_query(sql, args, as_dict=as_dict, **kwargs)

            def count_rows():
                return uncached_query(f"SELECT COUNT(*) FROM ({sql}) AS `_ndnf_cache_check`",
                                      args).fetchall()[0][0]

            key = hashlib.sha256(
                repr((host, self._ALIAS.sub("", sql), args, as_dict)).encode()).hexdigest()
            recorded = getattr(self._recording, "rows", None)
            if recorded is not None and not self._SCHEMA_LOOKUP.search(sql):
                if key not in recorded:  # the same query can come up more than once per download
                    recorded[key] = uncached_query(sql, args, as_dict=as_dict, **kwargs).fetchall()
                    self._save_downloaded(key, recorded[key])
                return CachedCursor(recorded[key])
            rows = self._get(key, count_rows)
            if rows is None:
                rows = uncached_query(sql, args, as_dict=as_dict, **kwargs).fetchall()
                self._put(key, rows)
            return CachedCursor(rows)

        conn.query = query
        conn._ndnf_query_cache = self

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0
            if self.disk_dir is not None and self.disk_dir.is_dir():
                for path in self.disk_dir.glob("*.pkl"):
                    path.unlink(missing_ok=True)

    @property
    def total_bytes(self):
        return self._total_bytes

    def _get(self, key, count_rows):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            downloaded = self._downloaded.get(key)
        if downloaded is not None:
            rows = entry[2] if entry is not None else self._load_downloaded(key, downloaded)
            if rows is not None:
                self.hits += 1
                return rows
        if entry is not None:
            checked_at, nbytes, rows = entry
            if time.monotonic() - checked_at <= self.ttl_s:
                self.hits += 1
                return rows
            if nbytes >= self.DISK_MIN_BYTES and count_rows() == len(rows):
                self._put(key, rows, to_disk=False)  # still current: restart its ttl
                self.hits += 1
                return rows
            with self._lock:
                if key in self._entries:
                    self._drop(key)
            self.misses += 1
            return None

        rows = self._disk_read(key)
        if rows is not None and count_rows() == len(rows):
            self._put(key, rows, to_disk=False)
            self.hits += 1
            return rows
        self.misses += 1
        return None

    def _put(self, key, rows, to_disk=True):
        nbytes = _approx_nbytes(rows)
        if to_disk and nbytes >= self.DISK_MIN_BYTES:
            self._disk_write(key, rows)
        if nbytes > self.max_bytes:
            return
        with self._lock:
            if key in self._entries:
                self._drop(key)
            self._entries[key] = (time.monotonic(), nbytes, rows)
            self._total_bytes += nbytes
            while self._total_bytes > self.max_bytes:
                self._drop(next(iter(self._entries)))

    def _drop(self, key):
        _, nbytes, _ = self._entries.pop(key)
        self._total_bytes -= nbytes

    def _save_downloaded(self, key, rows):
        path = self._recording.dir / f"{key}.pkl"
        _write_pickle(path, rows)  # unlike the disk cache, a failed download should say so
        with self._lock:
            self._downloaded[key] = path
        self._put(key, rows, to_disk=False)

    def _load_downloaded(self, key, path):
        try:
            with open(path, "rb") as f:
                rows = pickle.load(f)
        except Exception:  # deleted or damaged since: fall back to the database
            with self._lock:
                self._downloaded.pop(key, None)
            return None
        self._put(key, rows, to_disk=False)
        return rows

    # the disk layer is best-effort: any failure (unwritable dir, corrupt or half-written
    # file, disk full) just falls back to the database
    def _disk_read(self, key):
        if self.disk_dir is None:
            return None
        path = self.disk_dir / f"{key}.pkl"
        try:
            with open(path, "rb") as f:
                rows = pickle.load(f)
            os.utime(path)  # mtime doubles as last-used time for eviction
            return rows
        except FileNotFoundError:
            return None
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _disk_write(self, key, rows):
        if self.disk_dir is None:
            return
        try:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            _write_pickle(self.disk_dir / f"{key}.pkl", rows)
            files = sorted(self.disk_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
            total = sum(p.stat().st_size for p in files)
            for old in files:
                if total <= self.disk_max_bytes:
                    break
                total -= old.stat().st_size
                old.unlink(missing_ok=True)
        except Exception:
            pass


def _write_pickle(path, rows):
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _approx_nbytes(rows):
    total = 0
    for row in rows:
        for value in (row.values() if isinstance(row, dict) else row):
            total += len(value) if isinstance(value, (bytes, bytearray)) else 32
    return total


def _make_query_cache():
    cfg = load_gui_config()
    return QueryCache(
        max_bytes=int(cfg.get("query_cache_mb", QUERY_CACHE_MB_DEFAULT)) * 1024 ** 2,
        ttl_s=float(cfg.get("query_cache_ttl_min", QUERY_CACHE_TTL_MIN_DEFAULT)) * 60,
        disk_dir=cfg.get("query_cache_dir", QUERY_CACHE_DIR_DEFAULT),
        disk_max_bytes=int(cfg.get("query_cache_disk_mb", QUERY_CACHE_DISK_MB_DEFAULT)) * 1024 ** 2)


query_cache = _make_query_cache()


def _cached_cursor_class():
    from datajoint.connection import EmulatedCursor

    class CachedCursor(EmulatedCursor):
        # EmulatedCursor.fetchone raises StopIteration past the last row where a real cursor
        # returns None - which breaks fetch1() (it calls fetchone a second time to check
        # there's exactly one row)
        def fetchone(self):
            return next(self._iter, None)

    return CachedCursor
