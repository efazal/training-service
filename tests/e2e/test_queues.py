"""Queue discovery and queue selection over the deployed HTTP API; no training starts."""

import pytest

from tests.e2e.support import namespace, workload

KUEUE_UNUSABLE = {"kueue_not_installed", "kueue_project_excluded"}
MISSING_CLAIM = "e2e-missing-input"


def discover(api, project, expected=200):
    return api.request("GET", f"/api/v1/projects/{project}/queues", expected=expected)


def unusable(api, response):
    """Check a 409 discovery response and return its documented code."""
    code = api.object(response).get("code")
    assert code in KUEUE_UNUSABLE
    api.error(response, code)
    return code


def queue_names(api, project, response=None):
    """Check a usable discovery response and return its LocalQueue names."""
    body = api.object(response or discover(api, project))
    assert set(body) == {"kueue_enabled", "queues"}
    # Readiness and project eligibility passed; this is not an admission guarantee.
    assert body["kueue_enabled"] is True
    assert isinstance(body["queues"], list)
    for queue in body["queues"]:
        assert {"name"} <= queue.keys() <= {"name", "cluster_queue"}
        assert all(isinstance(value, str) and value for value in queue.values())
    names = [queue["name"] for queue in body["queues"]]
    assert len(names) == len(set(names))
    return names


def selected_queue(pytestconfig):
    if pytestconfig.getoption("queue_profile") != "with-kueue":
        pytest.skip("Pass --job-file and --queue-profile with-kueue to select a prepared queue")
    return workload(pytestconfig.getoption("job_file"), "with-kueue")["queue_name"]


def submit(api, pytestconfig, missing_job_id, queue, expected):
    """POST a request that cannot train: its storage claim is deliberately missing."""
    payload = {
        "name": "e2e-" + missing_job_id[-16:],
        "model": {"uri": f"pvc://{MISSING_CLAIM}/model/"},
        "dataset": {"uri": f"pvc://{MISSING_CLAIM}/data.jsonl"},
        "algorithm": {"name": "lora_sft", "parameters": {}},
        "output": {"destination_uri": f"pvc://{MISSING_CLAIM}/output/"},
        "queue_name": queue,
    }
    try:
        response = api.request("POST", api.jobs, expected=expected, json=payload)
        api.error(api.request("GET", f"{api.jobs}/{payload['name']}", expected=404), "not_found")
        return response
    finally:
        api.cleanup(payload["name"], pytestconfig.getoption("cleanup_timeout"), payload)


def test_queue_discovery(api, pytestconfig):
    profile = pytestconfig.getoption("queue_profile")
    if profile == "no-kueue":
        assert unusable(api, discover(api, api.project, 409)) == "kueue_not_installed"
    elif profile == "with-kueue":
        assert selected_queue(pytestconfig) in queue_names(api, api.project)
    elif profile == "default-queue":
        assert queue_names(api, api.project)
    else:
        response = discover(api, api.project, (200, 409))
        if response.status_code == 409:
            unusable(api, response)
        else:
            queue_names(api, api.project, response)


def test_selected_queue_reaches_storage_lookup(api, pytestconfig, missing_job_id):
    """A discoverable queue is accepted: the next rejection is the missing claim."""
    queue = selected_queue(pytestconfig)
    response = submit(api, pytestconfig, missing_job_id, queue, 404)
    api.error(
        response,
        "not_found",
        mentions=(f"PersistentVolumeClaim '{MISSING_CLAIM}'", f"project '{api.project}'"),
    )


def test_missing_queue_selection_creates_no_job(api, pytestconfig, missing_job_id):
    profile = pytestconfig.getoption("queue_profile")
    queue = "e2e-missing-" + missing_job_id[-16:]
    expected = {"with-kueue": 404, "default-queue": 404, "no-kueue": 409}.get(profile, (404, 409))
    response = submit(api, pytestconfig, missing_job_id, queue, expected)
    if response.status_code == 409:
        code = api.object(response).get("code")
        assert code in ({"kueue_not_installed"} if profile == "no-kueue" else KUEUE_UNUSABLE)
        api.error(response, code)
    else:
        api.error(
            response,
            "not_found",
            mentions=(f"LocalQueue '{queue}'", f"project '{api.project}'"),
        )


@pytest.mark.parametrize("queue", ["Bad_Queue", "a..b", "-edge", "a" * 64])
def test_malformed_queue_selection_creates_no_job(api, pytestconfig, missing_job_id, queue):
    response = submit(api, pytestconfig, missing_job_id, queue, 422)
    api.error(response, "validation_error", mentions=("queue_name",))


def test_queue_selection_is_scoped_to_requested_project(api, pytestconfig, missing_job_id):
    """A queue from another project must not be selectable in the test project."""
    option = pytestconfig.getoption("other_project_queue")
    if not option:
        pytest.skip("Pass --other-project-queue PROJECT/QUEUE to prove project scoping")
    other, separator, queue = option.partition("/")
    try:
        if not separator or namespace(other) == api.project:
            raise ValueError
    except ValueError:
        raise pytest.UsageError(
            "--other-project-queue must be OTHER_PROJECT/QUEUE, not the test project"
        ) from None
    assert queue in queue_names(api, other), "Queue is not discoverable in the other project"
    # An excluded test project rejects every queue, which would prove nothing about scoping.
    assert queue not in queue_names(api, api.project), (
        "Choose a queue name that does not also exist in the test project"
    )
    response = submit(api, pytestconfig, missing_job_id, queue, 404)
    api.error(
        response,
        "not_found",
        mentions=(f"LocalQueue '{queue}'", f"project '{api.project}'"),
    )
