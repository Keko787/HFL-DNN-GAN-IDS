"""Amendment 10: the SO_SNDTIMEO value is packed per OS.

Winsock reads a DWORD of milliseconds, POSIX a ``struct timeval``. The dock
link packed the POSIX layout everywhere, so on Windows its 60 s send bound
read as 60 ms, and a bound under 1 s read as none at all. Both encodings are
checked here on any host; the last tests read the option back from a real
socket on this one.
"""

from __future__ import annotations

import socket
import struct
import sys

import pytest

from hermes.transport.tcp_dock_link import set_send_timeout, sndtimeo_optval


def read_sndtimeo_s(sock: socket.socket) -> float:
    """The send timeout ``sock`` holds, in seconds, decoded per OS."""
    if sys.platform.startswith("win"):
        raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, 4)
        return struct.unpack("=I", raw[:4])[0] / 1000.0
    raw = sock.getsockopt(
        socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.calcsize("ll"),
    )
    sec, usec = struct.unpack("ll", raw)
    return sec + usec / 1e6


@pytest.mark.parametrize(
    "timeout_s, ms",
    [
        (60.0, 60_000),
        (30.0, 30_000),
        (0.3, 300),
        (1.5, 1_500),
        (0.0004, 1),   # a positive bound never packs as 0 (= no bound)
        (0.0, 0),
        (None, 0),
    ],
)
def test_windows_packs_a_dword_of_milliseconds(timeout_s, ms):
    assert sndtimeo_optval(timeout_s, platform="win32") == struct.pack("=I", ms)


def test_the_legacy_packing_reads_as_milliseconds_on_windows():
    # What the dock link used to pass: Winsock takes the first DWORD, tv_sec,
    # as milliseconds.
    legacy = struct.pack("ll", 60, 0)
    assert struct.unpack("=I", legacy[:4])[0] == 60
    assert sndtimeo_optval(60.0, platform="win32") != legacy[:4]
    # And a sub-second bound packed tv_sec = 0, which Winsock reads as none.
    assert struct.unpack("=I", struct.pack("ll", 0, 300_000)[:4])[0] == 0


def test_windows_clamps_to_the_largest_dword():
    assert sndtimeo_optval(1e12, platform="win32") == struct.pack("=I", 0xFFFFFFFF)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize(
    "timeout_s, sec, usec",
    [
        (60.0, 60, 0),
        (30.0, 30, 0),
        (1.5, 1, 500_000),
        (0.3, 0, 300_000),
        (2.9999999, 3, 0),   # rounding carries into seconds
        (1e-9, 0, 1),        # a positive bound never packs as 0
        (0.0, 0, 0),
        (None, 0, 0),
    ],
)
def test_posix_packs_seconds_and_microseconds(platform, timeout_s, sec, usec):
    assert sndtimeo_optval(timeout_s, platform=platform) == struct.pack("ll", sec, usec)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_rejects_timeouts_that_cannot_be_packed(bad, platform):
    with pytest.raises(ValueError, match="send timeout"):
        sndtimeo_optval(bad, platform=platform)


@pytest.mark.parametrize("timeout_s", [60.0, 30.0, 1.5])
def test_set_send_timeout_reads_back_on_this_host(timeout_s):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        assert set_send_timeout(s, timeout_s) is True
        # Linux may round to its timer tick; Windows stores milliseconds.
        assert read_sndtimeo_s(s) == pytest.approx(timeout_s, abs=0.02)
    finally:
        s.close()


def test_set_send_timeout_none_clears_the_bound():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        set_send_timeout(s, 5.0)
        assert set_send_timeout(s, None) is True
        assert read_sndtimeo_s(s) == 0.0
    finally:
        s.close()


def test_set_send_timeout_reports_a_refused_option():
    class _Refusing:
        def setsockopt(self, *_a):
            raise OSError("option not supported")

    assert set_send_timeout(_Refusing(), 1.0) is False
