"""
atomic_io.py
-------------

This module provides simple atomic file write helpers with a cooperative
file‑based locking mechanism.  When multiple parts of the bot attempt to
write to the same JSON file concurrently, naive writes can lead to
corruption or lost updates.  To mitigate this, `safe_write_json`
performs writes through a temporary file and atomically replaces the
target on completion.  It also obtains a `.lock` file to serialize
writers across threads and processes.  If the lock is already held by
another writer, calls will block briefly and retry until the lock is
released.  The lock is implemented using ``os.O_EXCL`` so it is
portable across platforms and does not rely on external packages.

Typical usage::

    from atomic_io import safe_write_json, file_lock
    safe_write_json(Path("metrics/metrics.json"), data)
    
    # Or using context manager:
    with file_lock(Path("myfile.json")):
        # do atomic operations
        pass

The helper functions catch and suppress all exceptions internally;
callers should not rely on them raising exceptions on failure.  If
atomic writes fail for any reason, the original file is left
untouched.

CHANGELOG:
- v1.0: Initial version with safe_write_json
- v1.1: Added file_lock context manager
- v1.2: Added safe_append_jsonl function
- v1.3: Added safe_read_json function
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Tuple, Optional, Union


def _acquire_lock(lock_path: Path, timeout: float = 5.0, poll_interval: float = 0.05) -> Tuple[int, Path] | None:
    """
    Acquire an exclusive file lock.  The lock is represented by creating
    a companion ``.lock`` file next to the target.  If the lock already
    exists, this function waits until it is released or the timeout is
    reached.  Returns the file descriptor and the lock file path on
    success, or ``None`` if the lock could not be acquired.

    Args:
        lock_path: Path to the lock file (should end with ``.lock``).
        timeout: Maximum seconds to wait for the lock.
        poll_interval: Seconds between lock acquisition attempts.
    """
    end_time = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            # Use os.O_EXCL to ensure we fail if the file already exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            # Write PID for debugging (optional)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            except Exception:
                pass
            return fd, lock_path
        except FileExistsError:
            # lock is held by another process
            if time.monotonic() >= end_time:
                return None
            time.sleep(poll_interval)
        except Exception:
            return None


def _release_lock(fd: int, lock_path: Path) -> None:
    """
    Release a previously acquired file lock.  Closes the file descriptor
    and removes the lock file.  Exceptions are suppressed.
    """
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        os.unlink(str(lock_path))
    except Exception:
        pass


@contextmanager
def file_lock(path: Union[Path, str], timeout: float = 5.0):
    """
    Context manager for file locking.
    
    Acquires an exclusive lock on the given file path and releases it
    when the context exits. This is useful for ensuring atomic operations
    on files that may be accessed by multiple processes or threads.
    
    Args:
        path: Path to the file to lock (a .lock file will be created next to it)
        timeout: Maximum seconds to wait for the lock
        
    Yields:
        None
        
    Raises:
        TimeoutError: If the lock could not be acquired within the timeout
        
    Usage::
    
        with file_lock(Path("myfile.json")):
            # Read, modify, and write the file atomically
            data = json.loads(Path("myfile.json").read_text())
            data["key"] = "value"
            Path("myfile.json").write_text(json.dumps(data))
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock = _acquire_lock(lock_path, timeout=timeout)
    if lock is None:
        raise TimeoutError(f"Could not acquire lock for {path} within {timeout}s")
    fd, lp = lock
    try:
        yield
    finally:
        _release_lock(fd, lp)


def safe_write_json(path: Union[Path, str], data: Any) -> bool:
    """
    Atomically write JSON data to the given path.  This function
    acquires a `.lock` next to the target file to prevent concurrent
    writers from clobbering each other.  The data is first written to
    a temporary file and then replaced over the target using
    ``os.replace``, which is atomic on POSIX and Windows.  If locking
    fails, or writing fails, the function falls back silently.

    Args:
        path: Destination file path.
        data: A JSON‑serialisable object to write.
        
    Returns:
        True if write succeeded, False otherwise.
    """
    try:
        path = Path(path)
        lock_path = path.with_suffix(path.suffix + ".lock")
        # Acquire lock
        lock = _acquire_lock(lock_path)
        if lock is None:
            # Could not acquire lock within timeout
            return False
        fd, lp = lock
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to temporary file in same directory
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            # Atomically replace
            os.replace(str(tmp_path), str(path))
            return True
        finally:
            _release_lock(fd, lp)
    except Exception:
        # Silently ignore all failures – atomic write is best effort
        return False


def safe_read_json(path: Union[Path, str], default: Any = None) -> Any:
    """
    Safely read JSON data from the given path with locking.
    
    Args:
        path: Source file path.
        default: Value to return if file doesn't exist or read fails.
        
    Returns:
        Parsed JSON data or default value.
    """
    try:
        path = Path(path)
        if not path.exists():
            return default
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock = _acquire_lock(lock_path, timeout=2.0)
        if lock is None:
            # Could not acquire lock, try reading anyway
            return json.loads(path.read_text(encoding="utf-8"))
        fd, lp = lock
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        finally:
            _release_lock(fd, lp)
    except Exception:
        return default


def safe_append_jsonl(path: Union[Path, str], record: Any) -> bool:
    """
    Append a JSON record to a JSONL file with the same lock discipline.

    - Her kayıt tek satır JSON olarak yazılır.
    - Lock ile yarış koşulları engellenir.
    
    Args:
        path: Path to the JSONL file.
        record: A JSON-serializable object to append.
        
    Returns:
        True if append succeeded, False otherwise.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(path, timeout=10):
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return True
    except Exception:
        return False


def safe_update_json(path: Union[Path, str], updates: dict) -> bool:
    """
    Atomically update specific keys in a JSON file.
    
    Reads the existing JSON, merges updates, and writes back atomically.
    
    Args:
        path: Path to the JSON file.
        updates: Dictionary of key-value pairs to update.
        
    Returns:
        True if update succeeded, False otherwise.
    """
    try:
        path = Path(path)
        with file_lock(path, timeout=10):
            # Read existing data
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        data = {}
                except Exception:
                    data = {}
            else:
                data = {}
            
            # Merge updates
            data.update(updates)
            
            # Write back
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(str(tmp_path), str(path))
            return True
    except Exception:
        return False
