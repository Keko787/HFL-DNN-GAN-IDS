"""Identifier newtypes.

Plain ``str`` aliases keep the wire format trivial while letting type
checkers catch swaps (``DeviceID`` passed where ``MuleID`` was expected).
"""

from __future__ import annotations

from typing import NewType

DeviceID = NewType("DeviceID", str)
MuleID = NewType("MuleID", str)
ServerID = NewType("ServerID", str)
