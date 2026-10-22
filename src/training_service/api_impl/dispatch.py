"""Select independently implemented API operations without import-order coupling."""

from typing import Any

from training_service.errors import ApiError


def implementation_for(base: Any, operation: str) -> Any:
    """Prefer explicit operation owners; support the existing submission handler."""
    owners = [
        implementation
        for implementation in base.subclasses
        if operation in getattr(implementation, "implemented_operations", ())
    ]
    if len(owners) > 1:
        raise RuntimeError(f"Multiple implementations registered for {operation}")
    if owners:
        return owners[0]()
    # Older handlers implement every method, including fail-closed stubs.
    # An explicit owner must take precedence regardless of registration order.
    legacy = [
        implementation
        for implementation in base.subclasses
        if not hasattr(implementation, "implemented_operations")
    ]
    if len(legacy) > 1:
        raise RuntimeError(f"Multiple legacy implementations registered for {operation}")
    if legacy:
        return legacy[0]()
    raise ApiError(501, "not_implemented", "This operation is not implemented.")
