"""proc_cleanup.py -- reap a process AND all its descendants, so a trial leaves nothing behind.

A trial launcher spawns the driver as a subprocess, which may itself spawn workers.  proc.terminate()
only kills the direct child -- on Windows it cannot even run a clean SIGTERM handler -- so grandchildren
linger (we accumulated 16 stale python processes across runs, contending for resources and making a
slow turn look 'hung').  kill_tree kills the whole tree.

Use in a launcher's finally so the tree is reaped on ANY exit (break / exception / wall cap):
    proc = subprocess.Popen([...])
    try:
        ... run the trial ...
    finally:
        proc_cleanup.kill_tree(proc.pid)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys


def kill_tree(pid) -> None:
    """Terminate ``pid`` and every descendant.  Never raises (best-effort cleanup)."""
    if pid is None:
        return
    try:
        if sys.platform.startswith("win"):
            # /T = tree (the process and any children it started), /F = force
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return
        # POSIX: kill the whole process group if the child leads one, else the lone pid
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
        except Exception:
            os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def reap_self_children() -> None:
    """atexit-friendly: kill any direct children of THIS process (graceful-exit cleanup).
    Uses psutil when available; otherwise a no-op (kill_tree from the launcher is the backstop)."""
    try:
        import psutil  # optional
        me = psutil.Process(os.getpid())
        for ch in me.children(recursive=True):
            try:
                ch.kill()
            except Exception:
                pass
    except Exception:
        pass
