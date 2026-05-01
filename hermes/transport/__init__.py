"""HERMES transports.

Each tier-boundary edge has a transport module. Phase 1 shipped the
dock-link loopback (Mule <-> Cluster). Phase 2 adds the RF-link loopback
(Mule <-> Edge Device). Sprint 2 (Phase 6) adds the TCP variants and a
wireless channel-emulator stub for AERPAW-shaped local emulation.
"""

from .channel_emulator import ChannelEmulator, no_op_emulator
from .cloud_link import CloudLink, CloudLinkError, HTTPCloudLink, MockTier3Server
from .dock_link import DockLink, DockLinkError, LoopbackDockLink
from .rf_link import RFLink, RFLinkError, LoopbackRFLink
from .tcp_dock_link import TCPDockLinkClient, TCPDockLinkServer
from .tcp_rf_link import TCPRFLinkClient, TCPRFLinkServer
from .wire import WireError, decode_message, encode_message, recv_message, send_message

__all__ = [
    # Loopback (Sprint 1)
    "DockLink",
    "DockLinkError",
    "LoopbackDockLink",
    "RFLink",
    "RFLinkError",
    "LoopbackRFLink",
    # Sprint 2 — TCP + emulator + cloud link
    "ChannelEmulator",
    "CloudLink",
    "CloudLinkError",
    "HTTPCloudLink",
    "MockTier3Server",
    "no_op_emulator",
    "TCPDockLinkClient",
    "TCPDockLinkServer",
    "TCPRFLinkClient",
    "TCPRFLinkServer",
    "WireError",
    "decode_message",
    "encode_message",
    "recv_message",
    "send_message",
]
