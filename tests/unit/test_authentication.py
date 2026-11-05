"""Focused authentication-boundary tests without a live OpenShift cluster."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import Depends, Request
from fastapi.testclient import TestClient
from kubernetes import client
from urllib3.exceptions import HTTPError

from training_service.auth import (
    _TOKEN_REVIEW_TIMEOUT_SECONDS,
    AuthenticationUnavailable,
    CallerIdentity,
    KubernetesTokenAuthenticator,
)
from training_service.main import create_app
from training_service_api.models.extra_models import TokenModel
from training_service_api.security_api import get_token_OpenShiftBearer


class FakeAuthenticator:
    """Records the token passed across the boundary and returns a selected result."""

    def __init__(self, result: CallerIdentity | None) -> None:
        self.result = result
        self.tokens: list[str] = []

    async def authenticate(self, token: str) -> CallerIdentity | None:
        self.tokens.append(token)
        return self.result


class UnavailableAuthenticator:
    """Simulates an unavailable OpenShift TokenReview service."""

    async def authenticate(self, token: str) -> CallerIdentity | None:
        raise AuthenticationUnavailable


def protected_client(result: CallerIdentity | None) -> tuple[TestClient, FakeAuthenticator]:
    authenticator = FakeAuthenticator(result)
    application = create_app(authenticator)

    @application.get("/identity")
    async def identity(
        request: Request,
        _: TokenModel = Depends(get_token_OpenShiftBearer),
    ) -> dict[str, object]:
        caller = request.state.caller_identity
        return {"username": caller.username, "uid": caller.uid, "groups": list(caller.groups)}

    return TestClient(application), authenticator


@pytest.mark.parametrize(
    "header",
    [None, "Basic credential", "Bearer", "Bearer  ", "Bearer token extra"],
)
def test_missing_or_malformed_bearer_credential_returns_401(header: str | None) -> None:
    client, authenticator = protected_client(CallerIdentity("alice", "uid-1", ("team-a",)))
    headers = {} if header is None else {"Authorization": header}

    response = client.get("/api/v1/algorithms", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {
        "code": "unauthorized",
        "message": "A valid OpenShift bearer token is required.",
    }
    assert authenticator.tokens == []


def test_authentication_backend_unavailable_returns_contract_503() -> None:
    client = TestClient(create_app(UnavailableAuthenticator()))

    response = client.get(
        "/api/v1/algorithms",
        headers={"Authorization": "Bearer accepted-token"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "authentication_unavailable",
        "message": "OpenShift token validation is temporarily unavailable.",
    }


def test_rejected_token_returns_401_without_exposing_it() -> None:
    client, authenticator = protected_client(None)
    token = "not-a-real-token"

    response = client.get("/api/v1/algorithms", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert token not in response.text
    assert authenticator.tokens == [token]


def test_verified_identity_is_propagated_without_the_bearer_token() -> None:
    client, authenticator = protected_client(CallerIdentity("alice", "uid-1", ("team-a", "team-b")))

    response = client.get("/identity", headers={"Authorization": "Bearer accepted-token"})

    assert response.status_code == 200
    assert response.json() == {
        "username": "alice",
        "uid": "uid-1",
        "groups": ["team-a", "team-b"],
    }
    assert authenticator.tokens == ["accepted-token"]


def test_authenticated_generated_route_reaches_unconfigured_discovery(monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_SERVICE_CATALOG_PATH", "/missing/catalog.json")
    client, _ = protected_client(CallerIdentity("alice", "uid-1", ()))

    response = client.get("/api/v1/algorithms", headers={"Authorization": "Bearer accepted-token"})

    assert response.status_code == 503
    assert response.json()["code"] == "backend_unavailable"


def test_token_review_uses_a_finite_timeout() -> None:
    authenticator = KubernetesTokenAuthenticator()
    api = Mock()
    api.create_token_review.return_value = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token="unused"),
        status=client.V1TokenReviewStatus(
            authenticated=True,
            user=client.V1UserInfo(
                username="alice",
                uid="uid-1",
                groups=["team-a"],
                extra={"scopes.authorization.openshift.io": ["user:full"]},
            ),
        ),
    )
    authenticator._api = api

    identity = authenticator._review("accepted-token")

    assert identity == CallerIdentity(
        "alice",
        "uid-1",
        ("team-a",),
        {"scopes.authorization.openshift.io": ("user:full",)},
    )
    api.create_token_review.assert_called_once()
    assert (
        api.create_token_review.call_args.kwargs["_request_timeout"]
        == _TOKEN_REVIEW_TIMEOUT_SECONDS
    )


def test_token_review_status_error_is_authentication_unavailable() -> None:
    authenticator = KubernetesTokenAuthenticator()
    api = Mock()
    api.create_token_review.return_value = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token="unused"),
        status=client.V1TokenReviewStatus(authenticated=False, error="token check unavailable"),
    )
    authenticator._api = api

    with pytest.raises(AuthenticationUnavailable):
        authenticator._review("accepted-token")


@pytest.mark.parametrize(
    "error",
    ["invalid bearer token", "token lookup failed", "[invalid bearer token, token lookup failed]"],
)
def test_openshift_token_review_rejection_returns_401(error: str) -> None:
    authenticator = KubernetesTokenAuthenticator()
    authenticator._api = Mock()
    authenticator._api.create_token_review.return_value = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token="unused"),
        status=client.V1TokenReviewStatus(authenticated=False, error=error),
    )
    http = TestClient(create_app(authenticator))
    response = http.get("/api/v1/algorithms", headers={"Authorization": "Bearer invalid-token"})
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


def test_token_rejection_mixed_with_backend_failure_remains_unavailable() -> None:
    authenticator = KubernetesTokenAuthenticator()
    authenticator._api = Mock()
    authenticator._api.create_token_review.return_value = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token="unused"),
        status=client.V1TokenReviewStatus(
            authenticated=False, error="[invalid bearer token, authentication webhook timed out]"
        ),
    )
    with pytest.raises(AuthenticationUnavailable):
        authenticator._review("invalid-token")


def test_token_review_client_initializes_once_for_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticator = KubernetesTokenAuthenticator()
    api = Mock()
    api.create_token_review.return_value = client.V1TokenReview(
        spec=client.V1TokenReviewSpec(token="unused"),
        status=client.V1TokenReviewStatus(authenticated=False),
    )
    load_config = Mock(side_effect=lambda: time.sleep(0.01))
    create_api = Mock(return_value=api)
    monkeypatch.setattr("training_service.auth.config.load_incluster_config", load_config)
    monkeypatch.setattr("training_service.auth.client.AuthenticationV1Api", create_api)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(authenticator._review, ["token-one", "token-two"]))

    load_config.assert_called_once()
    create_api.assert_called_once()


@pytest.mark.anyio
async def test_transport_failure_is_sanitized_as_authentication_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticator = KubernetesTokenAuthenticator()

    def raise_transport_error(token: str) -> CallerIdentity | None:
        raise HTTPError("connection failed")

    monkeypatch.setattr(authenticator, "_review", raise_transport_error)

    with pytest.raises(AuthenticationUnavailable):
        await authenticator.authenticate("accepted-token")
