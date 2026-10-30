"""Bound the entire Docker archive transfer, including blocking pipe reads and writes."""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time


def copy_archive(command, operation, *, upload, timeout):
    # Import lazily: transport exposes the public errors and calls this helper.
    from runtime.transport import DockerError, DockerPathMissing

    if timeout <= 0:
        raise ValueError('transfer timeout must be positive')
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE if upload else subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL if upload else subprocess.PIPE,
                                       stderr=errors, start_new_session=True)
        except OSError as exc:
            raise DockerError(str(exc)) from exc
        expired = threading.Event()
        deadline = time.monotonic() + timeout

        def kill():
            # The CLI can exit while a descendant still holds the archive pipe open.
            # Its process group remains ours until that last descendant exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                process.kill()

        def expire():
            expired.set()
            kill()

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            value = operation(process)
            if upload:
                process.stdin.close()
                process.stdin = None
            process.communicate(timeout=max(0.001, deadline - time.monotonic()))
            if expired.is_set():
                raise DockerError(f'Docker archive transfer timed out after {timeout:g}s')
            if process.returncode:
                errors.seek(0)
                message = errors.read(65536).decode(errors='replace').strip()
                error = DockerPathMissing if 'Could not find the file ' in message and ' in container ' in message else DockerError
                raise error(message or f'Docker archive transfer exited {process.returncode}')
            return value
        except BaseException as exc:
            kill()
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as reap_error:
                if hasattr(exc, 'add_note'):
                    exc.add_note(f'failed to reap Docker archive process: {reap_error}')
            if expired.is_set() or isinstance(exc, subprocess.TimeoutExpired):
                raise DockerError(f'Docker archive transfer timed out after {timeout:g}s') from exc
            # A missing source yields no tar stream; preserve that distinction for optional artifacts.
            import tarfile
            if isinstance(exc, (tarfile.ReadError, BrokenPipeError)):
                errors.seek(0)
                message = errors.read(65536).decode(errors='replace').strip()
                if message:
                    error = DockerPathMissing if 'Could not find the file ' in message and ' in container ' in message else DockerError
                    raise error(message) from exc
                raise DockerError(f'Docker archive transfer failed: {exc}') from exc
            raise
        finally:
            timer.cancel()
            timer.join()
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
