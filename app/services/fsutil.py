"""
Small filesystem helpers shared by the storage and knowledge-store layers.

All JSON writes go through `atomic_write_json` so that a crash, a killed
server, or two threads writing at once can never leave a half-written file
on disk. This is what previously allowed the knowledge base to be corrupted
or emptied mid-operation.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.config import Config


def now_ist_str():
    """Current timestamp in the configured timezone, as a display string."""
    try:
        return datetime.now(ZoneInfo(Config.SCHEDULE_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_json(path, payload, indent=2):
    """
    Writes JSON to `path` atomically: serialise to a temp file in the same
    directory, flush to disk, then os.replace() over the target.
    Either the old file or the complete new file exists - never a partial one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        raise


def read_json(path, default=None):
    """Reads JSON, returning `default` if the file is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[fsutil] Could not read {path.name}: {e}")
        return default


def atomic_write_bytes(path, write_fn):
    """
    Atomically produces a binary file. `write_fn(tmp_path_str)` must write the
    payload to the given temp path. Used for embeddings.npy and faiss.index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp_name)
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        raise
