"""Deterministic concurrent POST regressions: CAS losers never clean up the winner."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event, local
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kubernetes.client.exceptions import ApiException
from tests.unit import test_submission as submission_tests
from tests.unit.test_metrics_recovery_review import CONFIG, PROJECT, RUN
from tests.unit.test_metrics_recovery_review import server as server

from training_service import jobs
from training_service.auth import CallerIdentity
from training_service.errors import ApiError
from training_service.metrics import reservation, tracking


@pytest.fixture
def race(server, monkeypatch):
    tracking_client, state = server
    core = submission_tests._mock_sdk(monkeypatch)
    monkeypatch.setattr(jobs, "resolve_local_queue", lambda *args: None)
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "verify_training_access", lambda project: None)
    tracking_client.delete_run = Mock()
    discard = Mock(wraps=core.delete_namespaced_config_map)
    monkeypatch.setattr(core, "delete_namespaced_config_map", discard)
    sdk_create = Mock()
    original_submit = submission_tests.FakeRayJob.submit

    def submit(job):
        sdk_create(job.kwargs["job_name"])
        if sdk_create.call_count > 1:
            # A real API rejects a second create for the already accepted RayJob.
            raise jobs._SdkSubmissionError(ApiException(status=409))
        return original_submit(job)

    monkeypatch.setattr(submission_tests.FakeRayJob, "submit", submit)
    return SimpleNamespace(
        backend=jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        core=core,
        tracking=tracking_client,
        state=state,
        discard=discard,
        sdk_create=sdk_create,
        actor=local(),
    )


def post(race, actor):
    race.actor.name = actor
    try:
        return race.backend._submit_sync(
            PROJECT, submission_tests._sample(), CallerIdentity("alice", None, ()), "caller"
        )
    except Exception as error:
        return error


def assert_winner_preserved(race, winner_markers, loser):
    # Check destructive effects first so the old implementation fails on the actual regression.
    race.tracking.delete_run.assert_not_called()
    race.discard.assert_not_called()
    race.sdk_create.assert_called_once_with(submission_tests._sample().name)
    assert race.state["creates"] == 1
    assert len(winner_markers) == 1
    assert race.core.config_maps == winner_markers
    marker = next(iter(race.core.config_maps.values()))
    assert marker["data"]["phase"] == "submitted"
    assert marker["data"]["submit_attempted"] == "true"
    assert marker["data"]["run_id"] == RUN
    params = {item["key"]: item["value"] for item in race.state["run"]["data"]["params"]}
    assert params["training.submission_id"] == marker["metadata"]["uid"]
    assert params["training.rayjob_uid"] == marker["data"]["rayjob_uid"] == "created-rayjob-uid"
    assert isinstance(loser, ApiError)
    assert (loser.status_code, loser.code) == (409, "conflict")


def test_stale_submission_claim_conflict_does_not_clean_up_successful_post(race, monkeypatch):
    """Both reserve the same run; the loser applies its stale claim after the winner finishes."""
    ready = {actor: Event() for actor in ("winner", "loser")}
    release = {actor: Event() for actor in ("winner", "loser")}
    markers = {}
    original_update = reservation.Reservation.update

    def update(marker, **values):
        if values == {"submit_attempted": "true"}:
            actor = race.actor.name
            assert marker.data["phase"] == "reserved"
            assert marker.data["submit_attempted"] == "false"
            markers[actor] = marker
            ready[actor].set()
            assert release[actor].wait(timeout=5), f"{actor} submission claim was not released"
        return original_update(marker, **values)

    monkeypatch.setattr(reservation.Reservation, "update", update)
    with ThreadPoolExecutor(max_workers=2) as workers:
        try:
            winner = workers.submit(post, race, "winner")
            assert ready["winner"].wait(timeout=5), "winner did not reach its submission claim"
            loser = workers.submit(post, race, "loser")
            assert ready["loser"].wait(timeout=5), "loser did not reserve the shared run"
            assert markers["winner"].submission_id == markers["loser"].submission_id
            release["winner"].set()
            assert winner.result(timeout=5) == submission_tests._sample().name
            winner_markers = deepcopy(race.core.config_maps)
            release["loser"].set()
            loser_result = loser.result(timeout=5)
        finally:
            for event in release.values():
                event.set()

    assert_winner_preserved(race, winner_markers, loser_result)
    assert markers["loser"].data["submit_attempted"] == "true"  # Refreshed by the CAS conflict.


def test_create_claim_refresh_rejects_already_submitted_post_before_search(race, monkeypatch):
    """The loser holds prepared while another POST claims, submits, and binds the same marker."""
    ready, release = Event(), Event()
    refreshed = {}
    original_claim = reservation.Reservation.claim_create
    search = Mock(wraps=race.tracking.find_run)
    monkeypatch.setattr(race.tracking, "find_run", search)

    def claim(marker):
        if race.actor.name != "loser":
            return original_claim(marker)
        assert marker.data["phase"] == "prepared"
        ready.set()
        assert release.wait(timeout=5), "loser create claim was not released"
        result = original_claim(marker)
        assert result is False
        refreshed.update(marker.data)
        return result

    monkeypatch.setattr(reservation.Reservation, "claim_create", claim)
    with ThreadPoolExecutor(max_workers=2) as workers:
        try:
            loser = workers.submit(post, race, "loser")
            assert ready.wait(timeout=5), "loser did not reach its prepared create claim"
            winner = workers.submit(post, race, "winner")
            assert winner.result(timeout=5) == submission_tests._sample().name
            winner_markers = deepcopy(race.core.config_maps)
            release.set()
            loser_result = loser.result(timeout=5)
        finally:
            release.set()

    assert_winner_preserved(race, winner_markers, loser_result)
    assert refreshed["phase"] == "submitted"
    assert refreshed["submit_attempted"] == "true"
    search.assert_not_called()
