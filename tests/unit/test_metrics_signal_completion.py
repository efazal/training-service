"""Real SIGTERM at the success boundary must not interrupt driver finalization."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("stage", "status"),
    [
        ("success_selected", "FINISHED"),
        ("terminal_event", "FINISHED"),
        ("close", "FINISHED"),
        ("failed_finalization", "FAILED"),
        ("cancelled_finalization", "KILLED"),
    ],
)
def test_sigterm_preserves_selected_outcome_and_cleanup(stage, status):
    script = """
import os
import signal
import sys
from types import SimpleNamespace
from training_service import ray_entrypoint

stage = sys.argv[1]
sent = False
def terminate():
    global sent
    sent = True
    os.kill(os.getpid(), signal.SIGTERM)
    print('SIGNAL_RETURNED', flush=True)

def trace(frame, event, arg):
    if (not sent and event == 'line' and frame.f_code is ray_entrypoint.main.__code__):
        success = stage == 'success_selected' and frame.f_locals.get('status') == 'FINISHED'
        finalizing = (stage in {'failed_finalization', 'cancelled_finalization'}
                      and frame.f_locals.get('finalizing'))
        if success or finalizing:
            # Deliver a real signal before finally can install SIG_IGN.
            assert callable(signal.getsignal(signal.SIGTERM))
            terminate()
    return trace

def emit(metrics=None, *, tags=None):
    if tags['training.phase'] == 'completed':
        if stage == 'terminal_event':
            terminate()
        assert 'job/duration_seconds' in metrics
        assert 'job/finished_at_unix_seconds' in metrics
    print('PHASE:' + tags['training.phase'], flush=True)

def close(status):
    if stage == 'close':
        terminate()
    print('CLOSED:' + status, flush=True)

record = SimpleNamespace(emit=emit, start_sampling=lambda: None, close=close)
sys.modules['training_service_metrics_runtime'] = SimpleNamespace(current_recorder=lambda: record)
def execute(cfg):
    if stage == 'failed_finalization':
        raise ValueError('training failed')
    if stage == 'cancelled_finalization':
        raise KeyboardInterrupt('training interrupted')
ray_entrypoint._execute = execute
previous = signal.getsignal(signal.SIGTERM)
sys.settrace(trace)
try:
    try:
        ray_entrypoint.main({'mlflow': {'run_id': 'synthetic'}})
    except (ValueError, KeyboardInterrupt) as error:
        assert type(error) is {'failed_finalization': ValueError,
                               'cancelled_finalization': KeyboardInterrupt}[stage]
        print('EXPECTED_FAILURE', flush=True)
finally:
    sys.settrace(None)
    print('RESTORED:' + str(signal.getsignal(signal.SIGTERM) == previous), flush=True)
assert sent
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, stage],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("SIGNAL_RETURNED\n") == 1
    assert result.stdout.count("CLOSED:" + status + "\n") == 1
    phase = {"FINISHED": "completed", "FAILED": "failed", "KILLED": "cancelled"}[status]
    assert "PHASE:" + phase + "\n" in result.stdout
    assert ("EXPECTED_FAILURE\n" in result.stdout) == (status != "FINISHED")
    assert "RESTORED:True\n" in result.stdout
