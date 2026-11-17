"""The lifecycle of test children: a directory is never removed while its child may be alive.

These use real processes and check real process state. The children here run only short standard-library
snippets and never import backend code, so they don't need install_child_guards. Each ends itself
within a minute, so a failed test can't leave one running for long. Faults that a real process can't
produce, such as a child that won't die, are injected at the Popen, with a real child behind it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _neo4j_child_tempdirs as tempdirs
from tests._neo4j_child_tempdirs import ChildProcesses, ChildTempCleanupError, child_processes

MAKE_STATE = 'import tempfile; tempfile.mkdtemp(prefix="cyukti-test-state-"); '
HOLD = MAKE_STATE + 'import time; print("ready", flush=True); time.sleep(60)'
IGNORE_TERM = MAKE_STATE + 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready", flush=True); time.sleep(60)'
KILL_SELF = MAKE_STATE + 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)'
EXIT_CLEANLY = 'import sys; sys.exit(0)'
EXIT_LEAVING_STATE = MAKE_STATE + 'import os; os._exit(0)'


@pytest.fixture
def children(tmp_path):
    instance = ChildProcesses(str(tmp_path), terminate_timeout=2.0, kill_timeout=2.0)
    yield instance
    for child in instance._children:  # whatever a test injected, nothing outlives it
        subprocess.Popen.kill(child.process)
        subprocess.Popen.wait(child.process, 5)
        for stream in (child.process.stdout, child.process.stdin):
            if stream:
                stream.close()


def _start(children, code):
    process = children.spawn([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "ready", "the child has to be running for this to mean anything"
    return process


def _alive(process):
    return process.poll() is None


def _gone(pid):
    """The kernel's own view, not Popen's."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _state_entries(directory):
    return [name for name in os.listdir(directory) if name.startswith("cyukti-test-state-")]


def _fail_waiting(monkeypatch, process):
    def never(timeout=None):
        raise subprocess.TimeoutExpired("child", timeout)
    monkeypatch.setattr(process, "wait", never)


# ================================================================ normal and killed children

def test_a_child_that_exits_normally_has_its_directory_cleaned(children):
    result = children.run([sys.executable, "-c", EXIT_CLEANLY], timeout=30)
    assert result.returncode == 0
    [directory] = children.directories
    children.cleanup()
    assert not os.path.exists(directory)


def test_a_child_that_left_state_behind_has_it_cleaned_too(children):
    children.run([sys.executable, "-c", EXIT_LEAVING_STATE], timeout=30)
    [directory] = children.directories
    assert len(_state_entries(directory)) == 1
    children.cleanup()
    assert not os.path.exists(directory)


def test_a_child_already_killed_is_cleaned_without_being_signalled(children, monkeypatch):
    result = children.run([sys.executable, "-c", KILL_SELF], timeout=30)
    assert result.returncode == -signal.SIGKILL
    [directory] = children.directories
    assert len(_state_entries(directory)) == 1, "SIGKILL skipped any cleanup of its own"
    process = children._children[0].process

    def must_not_signal():
        raise AssertionError("an exited child was signalled")

    monkeypatch.setattr(process, "terminate", must_not_signal)
    monkeypatch.setattr(process, "kill", must_not_signal)
    children.cleanup()
    assert not os.path.exists(directory)


def test_a_child_killed_by_the_test_itself_is_reaped_then_cleaned(children):
    process = _start(children, HOLD)
    os.kill(process.pid, signal.SIGKILL)
    [directory] = children.directories
    children.cleanup()
    assert process.returncode == -signal.SIGKILL and _gone(process.pid)
    assert not os.path.exists(directory)


# ================================================================ alive at teardown

def test_a_running_child_is_terminated_and_its_exit_verified_before_removal(children, monkeypatch):
    process = _start(children, HOLD)
    [directory] = children.directories
    assert _alive(process) and _state_entries(directory)
    seen = []
    real_rmtree = tempdirs.shutil.rmtree

    def recording(path):
        seen.append((process.returncode, _gone(process.pid)))
        real_rmtree(path)

    monkeypatch.setattr(tempdirs.shutil, "rmtree", recording)
    children.cleanup()
    assert seen == [(-signal.SIGTERM, True)], "exited, and known to have, before anything was removed"
    assert not os.path.exists(directory)


def test_a_child_ignoring_termination_is_escalated_to_kill(tmp_path):
    children = ChildProcesses(str(tmp_path), terminate_timeout=0.3, kill_timeout=5.0)
    process = _start(children, IGNORE_TERM)
    [directory] = children.directories
    children.cleanup()
    assert process.returncode == -signal.SIGKILL, "SIGTERM was ignored, so only the kill could have ended it"
    assert _gone(process.pid) and not os.path.exists(directory)


def test_if_terminate_itself_fails_kill_is_still_tried(children, monkeypatch):
    process = _start(children, HOLD)
    [directory] = children.directories

    def refused():
        raise PermissionError("injected")

    monkeypatch.setattr(process, "terminate", refused)
    children.cleanup()
    assert process.returncode == -signal.SIGKILL and not os.path.exists(directory)


# ================================================================ exit can't be verified

def test_a_child_whose_exit_cannot_be_confirmed_keeps_its_directory(children, monkeypatch):
    process = _start(children, HOLD)
    [directory] = children.directories
    _fail_waiting(monkeypatch, process)
    with pytest.raises(ChildTempCleanupError) as caught:
        children.cleanup()
    assert str(process.pid) in str(caught.value) and directory in str(caught.value)
    assert "exit not verified" in str(caught.value) and "directory kept" in str(caught.value)
    assert _state_entries(directory), "kept, with what's in it"
    monkeypatch.undo()
    process.wait(10)  # the injected wait hid a real exit: the kill did land
    children.cleanup()
    assert not os.path.exists(directory)


def test_a_child_that_cannot_be_terminated_stays_alive_with_its_directory_kept(children, monkeypatch):
    process = _start(children, HOLD)
    [directory] = children.directories

    def refused():
        raise PermissionError("injected")

    monkeypatch.setattr(process, "terminate", refused)
    monkeypatch.setattr(process, "kill", refused)
    with pytest.raises(ChildTempCleanupError) as caught:
        children.cleanup()
    assert str(process.pid) in str(caught.value) and directory in str(caught.value) and "PermissionError" in str(caught.value)
    assert _alive(process) and _state_entries(directory), "really still running, and nothing was removed"
    monkeypatch.undo()
    children.cleanup()
    assert process.returncode == -signal.SIGTERM and not os.path.exists(directory)


def test_a_wait_that_returns_without_an_exit_is_not_believed(children, monkeypatch):
    process = _start(children, IGNORE_TERM)
    [directory] = children.directories
    monkeypatch.setattr(process, "wait", lambda timeout=None: None)
    monkeypatch.setattr(process, "terminate", lambda: None)
    with pytest.raises(ChildTempCleanupError, match="exit not verified"):
        children.cleanup()
    assert _alive(process) and _state_entries(directory)


def test_a_finished_childs_streams_are_closed(children):
    process = _start(children, HOLD)
    children.cleanup()
    assert process.stdout.closed


def test_an_unexpected_error_while_stopping_a_child_also_keeps_its_directory(children, monkeypatch):
    process = _start(children, HOLD)
    [directory] = children.directories

    def broken():
        raise RuntimeError("injected")

    monkeypatch.setattr(process, "terminate", broken)
    with pytest.raises(ChildTempCleanupError, match="RuntimeError"):
        children.cleanup()
    assert _alive(process) and _state_entries(directory)


# ================================================================ refusals still apply

def _exited_child_with_directory(children):
    children.run([sys.executable, "-c", EXIT_CLEANLY], timeout=30)
    return children.directories[-1]


@pytest.mark.parametrize("name", ["unrelated", "cyukti-test-state-short", "xcyukti-test-state-abcd1234"])
def test_an_unexpected_entry_is_refused_and_kept(children, name):
    directory = _exited_child_with_directory(children)
    os.mkdir(os.path.join(directory, name))
    with pytest.raises(ChildTempCleanupError, match="unexpected name"):
        children.cleanup()
    assert os.path.isdir(os.path.join(directory, name))


def test_a_symlink_entry_is_refused_and_never_followed(children, tmp_path):
    directory = _exited_child_with_directory(children)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    os.symlink(outside, os.path.join(directory, "cyukti-test-state-abcd1234"))
    with pytest.raises(ChildTempCleanupError, match="not a real directory"):
        children.cleanup()
    assert (outside / "keep").read_text() == "keep"


def test_an_entry_owned_by_someone_else_is_refused(children, monkeypatch):
    directory = _exited_child_with_directory(children)
    os.mkdir(os.path.join(directory, "cyukti-test-state-abcd1234"))
    uid = os.getuid()
    monkeypatch.setattr(tempdirs.os, "getuid", lambda: uid if sys._getframe(1).f_code.co_name != "remove_child_state" else uid + 1)
    with pytest.raises(ChildTempCleanupError, match="owned by uid"):
        children.cleanup()
    assert os.path.isdir(os.path.join(directory, "cyukti-test-state-abcd1234"))


def test_a_private_directory_owned_by_someone_else_is_refused(children, monkeypatch):
    directory = _exited_child_with_directory(children)
    uid = os.getuid()
    monkeypatch.setattr(tempdirs.os, "getuid", lambda: uid + 1)
    with pytest.raises(ChildTempCleanupError, match="owned by uid"):
        children.cleanup()
    assert os.path.isdir(directory)


def test_a_private_directory_replaced_by_a_symlink_is_refused_and_not_followed(children, tmp_path):
    directory = _exited_child_with_directory(children)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    os.rmdir(directory)
    os.symlink(outside, directory)
    with pytest.raises(ChildTempCleanupError, match="no longer the directory it was given"):
        children.cleanup()
    assert (outside / "keep").read_text() == "keep" and os.path.islink(directory)


def test_a_private_directory_replaced_by_another_directory_is_refused(children):
    directory = _exited_child_with_directory(children)
    os.rename(directory, directory + "-moved")
    os.mkdir(directory, 0o700)
    with pytest.raises(ChildTempCleanupError, match="no longer the directory it was given"):
        children.cleanup()
    assert os.path.isdir(directory) and os.path.isdir(directory + "-moved")


def test_a_missing_private_directory_is_reported(children):
    directory = _exited_child_with_directory(children)
    os.rmdir(directory)
    with pytest.raises(ChildTempCleanupError, match="FileNotFoundError"):
        children.cleanup()


# ================================================================ repeated cleanup

def test_cleanup_called_twice_does_nothing_more(children, monkeypatch):
    process = _start(children, HOLD)
    children.cleanup()
    assert process.returncode == -signal.SIGTERM

    def must_not_run(*args, **kwargs):
        raise AssertionError("a second cleanup acted again")

    monkeypatch.setattr(process, "terminate", must_not_run)
    monkeypatch.setattr(tempdirs.shutil, "rmtree", must_not_run)
    children.cleanup()
    assert children.directories == ()


def test_a_failed_cleanup_fails_the_same_way_again_and_then_can_finish(children, monkeypatch):
    process = _start(children, HOLD)
    [directory] = children.directories

    def refused():
        raise PermissionError("injected")

    monkeypatch.setattr(process, "terminate", refused)
    monkeypatch.setattr(process, "kill", refused)
    messages = []
    for _ in range(2):
        with pytest.raises(ChildTempCleanupError) as caught:
            children.cleanup()
        messages.append(str(caught.value))
    assert messages[0] == messages[1] and _alive(process) and os.path.isdir(directory)
    monkeypatch.undo()
    children.cleanup()
    assert not os.path.exists(directory) and children.directories == ()


# ================================================================ several children

def test_each_child_gets_its_own_directory_as_its_temp(children):
    print_temp = 'import tempfile; print(tempfile.gettempdir())'
    first = children.run([sys.executable, "-c", print_temp], timeout=30, capture_output=True, text=True)
    second = children.run([sys.executable, "-c", print_temp], timeout=30, capture_output=True, text=True)
    assert [first.stdout.strip(), second.stdout.strip()] == list(children.directories)
    assert first.stdout != second.stdout


def test_several_children_are_each_handled_by_their_own_state(children):
    running = _start(children, HOLD)
    ignoring = _start(children, HOLD)
    killed = children.run([sys.executable, "-c", KILL_SELF], timeout=30)
    clean = children.run([sys.executable, "-c", EXIT_CLEANLY], timeout=30)
    directories = children.directories
    assert len(set(directories)) == 4
    children.cleanup()
    assert running.returncode == ignoring.returncode == -signal.SIGTERM
    assert killed.returncode == -signal.SIGKILL and clean.returncode == 0
    assert not any(os.path.exists(directory) for directory in directories)


def test_one_child_failing_leaves_its_directory_while_the_others_are_handled(children, monkeypatch, tmp_path):
    good_before = _start(children, HOLD)
    bad = _start(children, HOLD)
    good_after = _start(children, HOLD)
    before_dir, bad_dir, after_dir = children.directories
    _fail_waiting(monkeypatch, bad)
    with pytest.raises(ChildTempCleanupError) as caught:
        children.cleanup()
    assert good_before.returncode == -signal.SIGTERM and good_after.returncode == -signal.SIGTERM
    assert not os.path.exists(before_dir) and not os.path.exists(after_dir), "the others were finished regardless"
    assert os.path.isdir(bad_dir) and _state_entries(bad_dir), "the failed one was left alone"
    assert children.directories == (bad_dir,)
    assert bad_dir in str(caught.value) and before_dir not in str(caught.value) and after_dir not in str(caught.value)
    assert (tmp_path / "child-processes").is_dir(), "the root holds a kept directory"
    monkeypatch.undo()
    bad.wait(10)
    children.cleanup()
    assert not (tmp_path / "child-processes").exists()


def test_every_failure_is_named_when_several_children_fail(children, monkeypatch):
    first, second = _start(children, HOLD), _start(children, HOLD)
    _fail_waiting(monkeypatch, first)
    _fail_waiting(monkeypatch, second)
    with pytest.raises(ChildTempCleanupError) as caught:
        children.cleanup()
    assert str(first.pid) in str(caught.value) and str(second.pid) in str(caught.value)
    monkeypatch.undo()
    first.wait(10)
    second.wait(10)


def test_an_unrelated_process_is_left_alone(children):
    """Started directly, so the helper has no way of knowing about it."""
    unrelated = subprocess.Popen([sys.executable, "-c", 'import time; print("ready", flush=True); time.sleep(60)'], stdout=subprocess.PIPE, text=True)
    try:
        assert unrelated.stdout.readline().strip() == "ready"
        _start(children, HOLD)
        children.cleanup()
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(10)
        unrelated.stdout.close()


# ================================================================ starting children

def test_a_child_that_fails_to_start_leaves_nothing_registered(children):
    with pytest.raises(OSError):
        children.spawn(["/nonexistent/program"])
    assert children.directories == ()
    children.cleanup()


def test_the_environment_cannot_bypass_the_private_directory(children):
    with pytest.raises(TypeError):
        children.spawn([sys.executable, "-c", "pass"], env={})
    assert children.directories == ()


def test_a_run_that_times_out_leaves_the_child_for_cleanup(children):
    with pytest.raises(subprocess.TimeoutExpired):
        children.run([sys.executable, "-c", 'import time; time.sleep(60)'], timeout=0.2)
    process = children._children[0].process
    assert _alive(process)
    children.cleanup()
    assert process.returncode == -signal.SIGTERM


# ================================================================ failures don't hide each other

def test_the_block_cleans_up_a_running_child_when_it_ends_normally(tmp_path):
    with child_processes(str(tmp_path)) as children:
        process = _start(children, HOLD)
    assert process.returncode == -signal.SIGTERM and not (tmp_path / "child-processes").exists()


def test_the_block_cleans_up_when_the_test_fails_and_the_failure_stays_visible(tmp_path):
    with pytest.raises(ValueError, match="the test failed") as caught:
        with child_processes(str(tmp_path)) as children:
            process = _start(children, HOLD)
            raise ValueError("the test failed")
    assert process.returncode == -signal.SIGTERM
    assert getattr(caught.value, "__notes__", []) == [], "cleanup worked, so there's nothing to add"
    assert not (tmp_path / "child-processes").exists()


def test_a_cleanup_failure_doesnt_replace_the_tests_failure(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="the test failed") as caught:
        with child_processes(str(tmp_path), terminate_timeout=0.2, kill_timeout=0.2) as children:
            process = _start(children, HOLD)
            _fail_waiting(monkeypatch, process)
            raise ValueError("the test failed")
    notes = caught.value.__notes__
    assert len(notes) == 1 and "child cleanup also failed" in notes[0] and str(process.pid) in notes[0], "both are there"
    monkeypatch.undo()
    process.wait(10)
    children.cleanup()


def test_a_cleanup_failure_is_raised_when_nothing_else_failed(tmp_path, monkeypatch):
    with pytest.raises(ChildTempCleanupError) as caught:
        with child_processes(str(tmp_path), terminate_timeout=0.2, kill_timeout=0.2) as children:
            process = _start(children, HOLD)
            _fail_waiting(monkeypatch, process)
    assert str(process.pid) in str(caught.value)
    monkeypatch.undo()
    process.wait(10)
    children.cleanup()


def test_an_interrupt_is_preserved_too(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with child_processes(str(tmp_path)) as children:
            process = _start(children, HOLD)
            raise KeyboardInterrupt
    assert process.returncode == -signal.SIGTERM


def test_every_child_in_the_journal_crash_and_fault_tests_goes_through_the_lifecycle():
    """A child started any other way could outlive its directory or leak one."""
    forbidden = ["subprocess." + "run(", "subprocess." + "Popen(", "dict(" + "os.environ)", "child_" + "environment("]
    for name in ("test_neo4j_write_journal_crash.py", "test_neo4j_write_journal_faults.py"):
        source = (Path(__file__).parent / name).read_text(encoding="utf-8")
        for text in forbidden:
            assert text not in source, (name, text)
