"""Durable reservation ownership, Kubernetes CAS, and delayed GC regression tests."""

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from training_service.errors import ApiError
from training_service.kubernetes_client import TIMEOUT
from training_service.metrics import reservation

PROJECT, JOB = "team-alpha", "example"
DIGEST, URI, EXPERIMENT = "a" * 64, "https://mlflow.example.test", "training"
LABEL = "training.rhoai.io/metrics-reservation"
NAME = "training-metrics-" + sha256(JOB.encode()).hexdigest()[:32]
RAYJOB = {
    "apiVersion": "ray.io/v1",
    "kind": "RayJob",
    "metadata": {"name": JOB, "namespace": PROJECT, "uid": "rayjob-uid"},
}


class ConfigMaps:
    """API state enforces real UID/resourceVersion preconditions before mutation."""

    def __init__(self, models):
        self.models, self.resource, self.attempt = models, None, 0
        self.core = Mock(spec=client.CoreV1Api)
        for verb in ("create", "read", "patch", "delete"):
            getattr(self.core, f"{verb}_namespaced_config_map").side_effect = getattr(self, verb)

    def response(self):
        value = deepcopy(self.resource)
        if not self.models:
            return value
        meta = value["metadata"]
        return client.V1ConfigMap(
            metadata=client.V1ObjectMeta(
                name=meta["name"],
                namespace=meta["namespace"],
                uid=meta["uid"],
                resource_version=meta["resourceVersion"],
                labels=meta.get("labels"),
                owner_references=meta.get("ownerReferences"),
            ),
            data=value["data"],
        )

    def create(self, *, namespace, body, _request_timeout):
        assert namespace == PROJECT and _request_timeout == TIMEOUT
        if self.resource is not None:
            raise ApiException(status=409)
        self.attempt += 1
        self.resource = deepcopy(body)
        self.resource["metadata"].update(
            namespace=namespace, uid=f"attempt-{self.attempt}", resourceVersion="1"
        )
        return self.response()

    def read(self, *, name, namespace, _request_timeout):
        assert (name, namespace, _request_timeout) == (NAME, PROJECT, TIMEOUT)
        if self.resource is None:
            raise ApiException(status=404)
        return self.response()

    def patch(self, *, name, namespace, body, _request_timeout):
        self.read(name=name, namespace=namespace, _request_timeout=_request_timeout)
        meta = self.resource["metadata"]
        if any(body["metadata"][key] != meta[key] for key in ("uid", "resourceVersion")):
            raise ApiException(status=409)
        self.resource["data"].update(body["data"])
        meta.update(deepcopy(body["metadata"]))
        meta["resourceVersion"] = str(int(meta["resourceVersion"]) + 1)
        return self.response()

    def delete(self, *, name, namespace, body, _request_timeout):
        self.read(name=name, namespace=namespace, _request_timeout=_request_timeout)
        assert set(body) == {"preconditions"}
        assert set(body["preconditions"]) == {"uid"}
        if body["preconditions"]["uid"] != self.resource["metadata"]["uid"]:
            raise ApiException(status=409)
        self.resource = None


@pytest.fixture(params=[False, True], ids=["dict", "V1ConfigMap"])
def store(request):
    return ConfigMaps(request.param)


def acquire(store, **changes):
    values = {"request_digest": DIGEST, "tracking_uri": URI, "experiment_name": EXPERIMENT}
    return reservation.acquire(store.core, PROJECT, JOB, **(values | changes))


def assert_conflict(action):
    with pytest.raises(ApiError) as error:
        action()
    assert (error.value.status_code, error.value.code) == (409, "conflict")
    assert "reconcile" in error.value.message.lower()
    return error.value


def test_create_records_only_identifiers_and_uses_server_uid(store):
    marker = acquire(store)
    assert marker.submission_id == "attempt-1"
    assert store.resource["metadata"]["name"] == NAME
    assert store.resource["metadata"]["labels"] == {LABEL: "true"}
    assert marker.data == {
        "request_digest": DIGEST,
        "tracking_uri": URI,
        "experiment_name": EXPERIMENT,
        "phase": "prepared",
        "job_name": JOB,
        "submit_attempted": "false",
    }
    assert "ownerReferences" not in store.resource["metadata"]
    assert all(isinstance(value, str) for value in marker.data.values())
    snapshot = marker.data
    snapshot["request_digest"] = "tampered"
    assert marker.data["request_digest"] == DIGEST


def test_acquire_conflict_reads_existing_without_resetting_it(store):
    first = acquire(store)
    assert first.claim_create()
    first.update(phase="reserved", run_id="run", experiment_id="7", submit_attempted="true")
    before = deepcopy(store.resource)
    second = acquire(store)
    assert second.submission_id == first.submission_id
    assert second.data == first.data
    assert store.resource == before
    store.core.read_namespaced_config_map.assert_called_once_with(
        name=NAME, namespace=PROJECT, _request_timeout=TIMEOUT
    )


@pytest.mark.parametrize("field", ["request_digest", "tracking_uri", "experiment_name"])
def test_acquire_rejects_changed_request_identity_without_mutation(store, field):
    acquire(store)
    before = deepcopy(store.resource)
    assert_conflict(lambda: acquire(store, **{field: "different"}))
    assert store.resource == before
    store.core.patch_namespaced_config_map.assert_not_called()
    store.core.delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("metadata", "labels", {}),
        ("metadata", "labels", {LABEL: "false"}),
        ("metadata", "name", "unrelated"),
        ("metadata", "namespace", "other-project"),
        ("metadata", "uid", ""),
        ("metadata", "resourceVersion", ""),
        ("data", "job_name", "different-job"),
        ("data", "phase", "invalid"),
    ],
)
def test_acquire_guards_unrelated_or_invalid_marker(store, section, key, value):
    acquire(store)
    store.resource[section][key] = value
    before = deepcopy(store.resource)
    assert_conflict(lambda: acquire(store))
    assert store.resource == before
    store.core.patch_namespaced_config_map.assert_not_called()
    store.core.delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize("bad_data", [None, [], "not-a-map", {"phase": False}, {7: "value"}])
def test_acquire_requires_a_string_dictionary(store, bad_data):
    acquire(store)
    store.resource["data"] = bad_data
    assert_conflict(lambda: acquire(store))
    store.core.patch_namespaced_config_map.assert_not_called()


def test_two_prepared_readers_only_one_claims_create(store):
    winner, loser = acquire(store), acquire(store)
    assert winner.claim_create() is True
    assert loser.claim_create() is False
    assert loser.data["phase"] == "creating"
    assert store.resource["data"]["phase"] == "creating"
    assert store.core.patch_namespaced_config_map.call_count == 2
    assert store.core.patch_namespaced_config_map.call_args.kwargs["body"]["metadata"] == {
        "uid": "attempt-1",
        "resourceVersion": "1",
    }


@pytest.mark.parametrize("phase", ["creating", "reserved", "submitted", "abandoned"])
def test_other_phases_never_claim_or_patch(store, phase):
    marker = acquire(store)
    marker.update(phase=phase)
    store.core.patch_namespaced_config_map.reset_mock()
    assert marker.claim_create() is False
    store.core.patch_namespaced_config_map.assert_not_called()


def test_claim_conflict_returns_false_even_if_phase_is_still_prepared(store):
    marker = acquire(store)
    store.resource["metadata"]["resourceVersion"] = "2"
    assert marker.claim_create() is False
    assert marker.data["phase"] == "prepared"
    assert store.core.patch_namespaced_config_map.call_count == 1


@pytest.mark.parametrize("change", ["uid", *reservation.IDENTITY, "label", "gone"])
def test_claim_cas_conflict_revalidates_identity(store, change):
    marker = acquire(store)
    store.resource["metadata"]["resourceVersion"] = "2"
    if change == "uid":
        store.resource["metadata"]["uid"] = "replacement"
    elif change in reservation.IDENTITY:
        store.resource["data"][change] = "different"
    elif change == "label":
        store.resource["metadata"]["labels"] = {}
    else:
        store.resource = None
        store.core.patch_namespaced_config_map.side_effect = ApiException(status=409)
    before = deepcopy(store.resource)
    assert_conflict(marker.claim_create)
    assert store.resource == before


def test_lost_claim_response_is_reconciled_without_second_create_claim(store):
    marker = acquire(store)

    def lost_response(**kwargs):
        store.patch(**kwargs)
        raise OSError("lost response")

    store.core.patch_namespaced_config_map.side_effect = lost_response
    with pytest.raises(OSError):
        marker.claim_create()
    recovered = reservation.load(store.core, PROJECT, JOB)
    assert recovered.submission_id == marker.submission_id
    assert recovered.data["phase"] == "creating"
    store.core.patch_namespaced_config_map.reset_mock()
    assert recovered.claim_create() is False
    store.core.patch_namespaced_config_map.assert_not_called()


def test_lost_marker_creation_response_recovers_same_server_uid(store):
    def lost_response(**kwargs):
        store.create(**kwargs)
        raise OSError("lost response")

    store.core.create_namespaced_config_map.side_effect = lost_response
    with pytest.raises(OSError):
        acquire(store)
    store.core.create_namespaced_config_map.side_effect = store.create
    assert acquire(store).submission_id == "attempt-1"
    assert store.attempt == 1


def test_update_merges_data_and_conflicts_without_overwriting_concurrent_changes(store):
    first, stale = acquire(store), acquire(store)
    first.update(run_id="run", experiment_id="7", phase="reserved")
    before = deepcopy(store.resource)
    assert_conflict(lambda: stale.update(submit_attempted="true"))
    assert store.resource == before
    assert stale.data["run_id"] == "run"
    stale.update(submit_attempted="true")
    assert stale.data == before["data"] | {"submit_attempted": "true"}


@pytest.mark.parametrize(
    "changes",
    [
        {"token": "secret"},
        {"payload": "{}"},
        {"phase": "invalid"},
        {"submit_attempted": True},
        {"run_id": None},
    ],
)
def test_update_rejects_non_identifier_or_non_string_data(store, changes):
    marker = acquire(store)
    with pytest.raises(ValueError):
        marker.update(**changes)
    store.core.patch_namespaced_config_map.assert_not_called()


def test_update_cannot_change_request_identity_or_an_existing_binding(store):
    marker = acquire(store)
    marker.bind_job(RAYJOB)
    store.core.patch_namespaced_config_map.reset_mock()
    for changes in ({"request_digest": "other"}, {"rayjob_uid": "other"}):
        assert_conflict(lambda changes=changes: marker.update(**changes))
    store.core.patch_namespaced_config_map.assert_not_called()


def test_bind_records_exact_uid_and_only_complete_attaches_owner(store):
    marker = acquire(store)
    marker.update(run_id="run", experiment_id="7", phase="reserved")
    marker.bind_job(RAYJOB)
    assert marker.data["rayjob_uid"] == RAYJOB["metadata"]["uid"]
    assert "ownerReferences" not in store.resource["metadata"]
    marker.complete(RAYJOB)
    assert marker.data["phase"] == "submitted"
    assert store.resource["metadata"]["ownerReferences"] == [
        {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "name": JOB,
            "uid": "rayjob-uid",
        }
    ]
    assert marker.data["run_id"] == "run"
    marker.complete(RAYJOB)
    before = deepcopy(store.resource)
    other = deepcopy(RAYJOB)
    other["metadata"]["uid"] = "replacement-job"
    assert_conflict(lambda: marker.bind_job(other))
    assert_conflict(lambda: marker.complete(other))
    assert store.resource == before


@pytest.mark.parametrize("field,value", [("uid", ""), ("name", "other"), ("namespace", "other")])
def test_bind_rejects_missing_uid_or_wrong_job_identity(store, field, value):
    marker = acquire(store)
    job = deepcopy(RAYJOB)
    job["metadata"][field] = value
    assert_conflict(lambda: marker.complete(job))
    store.core.patch_namespaced_config_map.assert_not_called()


def test_discard_uses_uid_precondition_and_recreation_has_a_new_submission_id(store):
    first = acquire(store)
    first.discard()
    assert reservation.load(store.core, PROJECT, JOB) is None
    second = acquire(store)
    assert first.submission_id != second.submission_id
    before = deepcopy(store.resource)
    assert_conflict(first.discard)
    assert store.resource == before
    second.discard()
    second.discard()


@pytest.mark.parametrize("operation", ["claim", "update", "bind", "complete"])
def test_same_name_and_resource_version_cannot_mutate_replacement_uid(store, operation):
    stale = acquire(store)
    stale.discard()
    replacement = acquire(store)
    assert replacement.submission_id != stale.submission_id
    assert store.resource["metadata"]["resourceVersion"] == "1"
    before = deepcopy(store.resource)
    actions = {
        "claim": stale.claim_create,
        "update": lambda: stale.update(run_id="run"),
        "bind": lambda: stale.bind_job(RAYJOB),
        "complete": lambda: stale.complete(RAYJOB),
    }
    assert_conflict(actions[operation])
    assert store.resource == before


def test_owner_reference_conflict_keeps_marker_for_reconciliation(store):
    marker = acquire(store)

    def conflict_on_owner(**kwargs):
        if "ownerReferences" in kwargs["body"]["metadata"]:
            store.resource["metadata"]["resourceVersion"] = "3"
            raise ApiException(status=409)
        return store.patch(**kwargs)

    store.core.patch_namespaced_config_map.side_effect = conflict_on_owner
    assert_conflict(lambda: marker.complete(RAYJOB))
    assert marker.data["rayjob_uid"] == "rayjob-uid"
    assert "ownerReferences" not in store.resource["metadata"]


def test_load_validates_minimal_owned_marker_and_does_not_create(store):
    assert reservation.load(store.core, PROJECT, JOB) is None
    marker = acquire(store)
    store.resource["data"] = {"run_id": "run", "tracking_uri": URI}
    loaded = reservation.load(store.core, PROJECT, JOB)
    assert loaded.submission_id == marker.submission_id
    assert loaded.data == store.resource["data"]
    store.resource["metadata"]["labels"] = {}
    assert_conflict(lambda: reservation.load(store.core, PROJECT, JOB))
    assert store.core.create_namespaced_config_map.call_count == 1


@pytest.mark.parametrize("field", ["name", "uid", "resourceVersion"])
def test_load_rejects_incomplete_or_wrong_server_identity(store, field):
    acquire(store)
    store.resource["metadata"][field] = ""
    assert_conflict(lambda: reservation.load(store.core, PROJECT, JOB))


@pytest.mark.parametrize("uid", ["with_underscore", "with space", "quote'", "a/b", "a" * 129])
def test_submission_id_rejects_unsafe_mlflow_filter_values(store, uid):
    acquire(store)
    store.resource["metadata"]["uid"] = uid
    assert_conflict(lambda: acquire(store))
    assert_conflict(lambda: reservation.load(store.core, PROJECT, JOB))
    store.core.patch_namespaced_config_map.assert_not_called()
    store.core.delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize("uid", ["f01e2d3c-4b5a-6789-abcd-0123456789ef", "fake-ID-1", "a" * 128])
def test_submission_id_accepts_kubernetes_uuid_and_safe_fake_ids(store, uid):
    acquire(store)
    store.resource["metadata"]["uid"] = uid
    assert acquire(store).submission_id == uid
    assert reservation.load(store.core, PROJECT, JOB).submission_id == uid


@pytest.mark.parametrize("operation", ["create", "read", "patch", "delete"])
@pytest.mark.parametrize("status", [403, 500])
def test_other_kubernetes_errors_are_left_for_caller_translation(store, operation, status):
    marker = acquire(store)
    getattr(store.core, f"{operation}_namespaced_config_map").side_effect = ApiException(
        status=status
    )
    actions = {
        "create": lambda: acquire(store),
        "read": lambda: reservation.load(store.core, PROJECT, JOB),
        "patch": marker.claim_create,
        "delete": marker.discard,
    }
    with pytest.raises(ApiException) as error:
        actions[operation]()
    assert error.value.status == status


def test_uses_only_supplied_client_without_loading_or_persisting_credentials(store, monkeypatch):
    token = "Bearer caller-only-test-credential"
    store.core.api_client = SimpleNamespace(
        configuration=SimpleNamespace(api_key={"authorization": token})
    )
    defaults = client.Configuration.get_default_copy().api_key.copy()
    forbidden = Mock(side_effect=AssertionError("unexpected configuration or credential IO"))
    with monkeypatch.context() as patch:
        for name in ("load_incluster_config", "load_kube_config"):
            patch.setattr(config, name, forbidden)
        for name in ("CoreV1Api", "ApiClient", "Configuration"):
            patch.setattr(client, name, forbidden)
        patch.setattr("builtins.open", forbidden)
        marker = acquire(store)
        marker.claim_create()
        marker.update(run_id="run", experiment_id="7", phase="reserved")
        marker.complete(RAYJOB)
        assert reservation.load(store.core, PROJECT, JOB).submission_id == marker.submission_id
        marker.discard()
    forbidden.assert_not_called()
    assert client.Configuration.get_default_copy().api_key == defaults
    assert store.core.api_client.configuration.api_key == {"authorization": token}
    for call in store.core.mock_calls:
        assert token not in json.dumps(call.kwargs)
