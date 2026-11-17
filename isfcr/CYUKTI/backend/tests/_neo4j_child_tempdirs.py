"""Parent-side lifecycle of child processes that may die by SIGKILL, and of their temporary directories.

Each child's install_child_guards() makes a cyukti-test-state-* directory and registers its removal
with atexit, which SIGKILL skips. The name is random, so the parent can't know it. So every child is
started through ChildProcesses, which gives it a private directory as its TMPDIR, keeps its Popen,
and removes that directory only once the child's exit is verified. Anything unexpected is reported
and left in place.

Identity and exit: only the Popen object is ever used. Its poll and wait reap this exact child, and
it won't send a signal once the child is reaped, so a reused pid can't hit an unrelated process. A
child counts as exited only when waitpid has returned for it.
"""
import os
import re
import shutil
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

_STATE_DIRECTORY = re.compile(r"^cyukti-test-state-[a-z0-9_]{8}$")


class ChildTempCleanupError(Exception):
    """A child's exit couldn't be verified, or its directory couldn't be removed or held something
    unexpected. What it names has been left in place."""


def child_environment(area: str) -> dict:
    """The environment for a child whose temporary directory belongs in `area`."""
    return {**os.environ, "TMPDIR": area}


def remove_child_state(area: str) -> None:
    """Removes the state directories in `area`, after checking each one's name, type and owner.
    Raises if anything didn't pass or couldn't be removed. Only call once `area`'s child has exited."""
    problems = []
    with os.scandir(area) as entries:
        for entry in list(entries):
            info = entry.stat(follow_symlinks=False)
            if not _STATE_DIRECTORY.match(entry.name):
                problems.append(f"{entry.name}: unexpected name")
            elif not stat.S_ISDIR(info.st_mode):
                problems.append(f"{entry.name}: not a real directory")
            elif info.st_uid != os.getuid():
                problems.append(f"{entry.name}: owned by uid {info.st_uid}")
            else:
                try:
                    shutil.rmtree(entry.path)
                except OSError as error:
                    problems.append(f"{entry.name}: {type(error).__name__}")
    if problems:
        raise ChildTempCleanupError(f"in {area}: {'; '.join(problems)}")


@contextmanager
def child_temp_area(parent: str) -> Iterator[str]:
    """A private directory with no process behind it, for exercising remove_child_state's checks.
    Children must be started through ChildProcesses instead."""
    area = os.path.join(parent, "child-tmp")
    os.mkdir(area, 0o700)
    try:
        yield area
    finally:
        remove_child_state(area)
        os.rmdir(area)


@dataclass
class _Child:
    process: subprocess.Popen
    directory: str
    device: int
    inode: int


class ChildProcesses:
    """The children of one test. cleanup() handles each independently and can safely be repeated:
    a child that fails stays registered with its directory kept, and the next call tries again."""

    def __init__(self, parent: str, *, terminate_timeout: float = 5.0, kill_timeout: float = 5.0) -> None:
        self._root = os.path.join(parent, "child-processes")
        os.mkdir(self._root, 0o700)
        self._terminate_timeout = terminate_timeout
        self._kill_timeout = kill_timeout
        self._children: list[_Child] = []
        self._started = 0
        self._root_removed = False

    @property
    def directories(self) -> tuple[str, ...]:
        """The private directory of each child not yet cleaned up, in start order."""
        return tuple(child.directory for child in self._children)

    def spawn(self, argv: list, **popen_arguments: Any) -> subprocess.Popen:
        """Starts and registers a child. Its environment is set here, so it can't bypass the directory."""
        if "env" in popen_arguments:
            raise TypeError("ChildProcesses sets the child's environment")
        self._started += 1
        directory = os.path.join(self._root, f"child-{self._started}")
        os.mkdir(directory, 0o700)
        info = os.lstat(directory)
        try:
            process = subprocess.Popen(argv, env=child_environment(directory), **popen_arguments)
        except BaseException:
            os.rmdir(directory)
            raise
        self._children.append(_Child(process, directory, info.st_dev, info.st_ino))
        return process

    def run(self, argv: list, *, timeout: float, capture_output: bool = False, **popen_arguments: Any) -> subprocess.CompletedProcess:
        """Like subprocess.run. On a timeout the child stays registered, and cleanup() deals with it."""
        if capture_output:
            popen_arguments.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process = self.spawn(argv, **popen_arguments)
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)

    def _verify_exit(self, child: _Child) -> None:
        """Returns only once waitpid has returned for this child, terminating it if need be."""
        process = child.process
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=self._terminate_timeout)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=self._kill_timeout)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ChildTempCleanupError(f"child pid {process.pid}: exit not verified ({type(error).__name__}), "
                                            f"directory kept: {child.directory}") from error
        if process.returncode is None:
            raise ChildTempCleanupError(f"child pid {process.pid}: exit not verified, directory kept: {child.directory}")

    def _remove_directory(self, child: _Child) -> None:
        pid = child.process.pid
        try:
            info = os.lstat(child.directory)
        except OSError as error:
            raise ChildTempCleanupError(f"child pid {pid}: {child.directory}: {type(error).__name__}") from error
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (child.device, child.inode):
            raise ChildTempCleanupError(f"child pid {pid}: {child.directory} is no longer the directory it was given")
        if info.st_uid != os.getuid():
            raise ChildTempCleanupError(f"child pid {pid}: {child.directory} is owned by uid {info.st_uid}")
        try:
            remove_child_state(child.directory)
        except ChildTempCleanupError as error:
            raise ChildTempCleanupError(f"child pid {pid}: {error}") from error
        try:
            os.rmdir(child.directory)
        except OSError as error:
            raise ChildTempCleanupError(f"child pid {pid}: {child.directory}: {type(error).__name__}") from error

    @staticmethod
    def _close_streams(child: _Child) -> None:
        for stream in (child.process.stdin, child.process.stdout, child.process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass  # buffered input to a process that has gone; nothing depends on it

    def cleanup(self) -> None:
        """Verifies each child's exit, then removes its directory, then the root once empty. Raises
        ChildTempCleanupError naming every child it couldn't finish, after trying all of them."""
        failures = []
        for child in list(self._children):
            try:
                self._verify_exit(child)
                self._close_streams(child)
                self._remove_directory(child)
            except Exception as error:
                failures.append(str(error) if isinstance(error, ChildTempCleanupError)
                                else f"child pid {child.process.pid}: {type(error).__name__}, directory kept: {child.directory}")
            else:
                self._children.remove(child)
        if failures:
            raise ChildTempCleanupError("; ".join(failures))
        if not self._root_removed:
            try:
                os.rmdir(self._root)
            except OSError as error:
                raise ChildTempCleanupError(f"{self._root}: {type(error).__name__}") from error
            self._root_removed = True


@contextmanager
def child_processes(parent: str, **timeouts: float) -> Iterator[ChildProcesses]:
    """ChildProcesses, cleaned up however the block ends. If the block failed, that failure is what
    propagates, carrying any cleanup failure as a note, so neither hides the other."""
    children = ChildProcesses(parent, **timeouts)
    try:
        yield children
    except BaseException as original:
        try:
            children.cleanup()
        except Exception as cleanup_error:
            original.add_note(f"child cleanup also failed: {cleanup_error}")
        raise
    else:
        children.cleanup()
