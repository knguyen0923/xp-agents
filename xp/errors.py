from __future__ import annotations


class ProtocolError(Exception):
    """A research-protocol rule was violated."""


class ConfigMismatchError(ProtocolError):
    """An existing result was produced under a different config."""


class ProvisionalResultError(ProtocolError):
    """A provisional (boundary-optimum) result was used in a final output."""
