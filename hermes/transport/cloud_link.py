"""Sprint 2 — outbound-only cloud link (Tier-2 ↔ Tier-3).

AERPAW blocks inbound traffic from outside the testbed (slides 30–32
of the deck), so the Tier-2 cluster cannot accept connections from
Tier-3. The cloud link inverts the polarity: the *cluster* makes
outbound HTTP requests to a Tier-3 endpoint, both to ship the cluster
partial upward and to poll for generator refinements.

Documented limitations / Sprint-3+ backlog:

* **S2-L4** — :class:`HTTPCloudLink` uses ``urllib.request`` synchronously,
  with no retry / backoff. A flaky Tier-3 returns errors directly. When
  Tier-3 grows authentication, this swaps to ``httpx`` or ``requests``
  with proper retry semantics. Today's local emulation never sees flaky
  endpoints so the simple form is enough.
* **S2-L5** — :class:`MockTier3Server` doesn't simulate latency or 5xx
  errors. Fault-injection in chunk O can extend it (drop probability,
  status-code injection) when those tests need it; the surface is
  small enough that retrofitting won't churn the public API.
* **S2-H2** — the handler class is bound late inside ``__init__`` to
  close over ``self``; one mock instance per test, no cross-talk
  observed. Subclassing the handler would need an explicit factory.

Pattern (slides 30–32, "reverse-SSH or HTTP polling"):

* ``POST {base_url}/partials`` — cluster ships its post-FedAvg
  discriminator weights to Tier-3 for cross-cluster aggregation.
* ``GET {base_url}/refinement/{cluster_id}`` — cluster polls for the
  next generator refinement. ``204 No Content`` means "nothing yet,
  try again later"; ``200 OK`` returns a pickled
  :class:`GeneratorRefinement`.

Both directions ride on the same length-prefix / pickle pair as the
RF and dock links — see :mod:`hermes.transport.wire`. Tier-3 isn't
constrained to AERPAW's no-inbound rule (it lives on Chameleon /
AWS), but keeping the polarity outbound-from-cluster keeps the design
portable when AERPAW returns.

This module ships:

* :class:`CloudLink` — abstract surface.
* :class:`HTTPCloudLink` — outbound HTTP client. Uses ``urllib`` to
  avoid an extra dependency.
* :class:`MockTier3Server` — in-process HTTP server for tests +
  Sprint 2 demos.

Sprint 6 (real AERPAW + real Tier-3) replaces ``MockTier3Server`` with
the actual Chameleon endpoint URL; ``HTTPCloudLink`` itself doesn't
change.
"""

from __future__ import annotations

import http.server
import logging
import pickle
import socketserver
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from hermes.types import ClusterPartialUpload, GeneratorRefinement

log = logging.getLogger(__name__)


class CloudLinkError(IOError):
    """Raised on cloud-link failure (HTTP error, timeout, decode failure)."""


# --------------------------------------------------------------------------- #
# Abstract surface
# --------------------------------------------------------------------------- #

class CloudLink(ABC):
    """Outbound-only cluster ↔ Tier-3 link.

    Mirrors the symmetric server/client convention of :class:`RFLink`
    and :class:`DockLink`, except both methods here are *outbound*
    from the cluster. Tier-3 has no method to push inbound — the AERPAW
    inbound restriction is structural in this ABC.
    """

    @abstractmethod
    def send_partial(self, partial: ClusterPartialUpload) -> None:
        """Outbound POST: ship the cluster's partial to Tier-3."""

    @abstractmethod
    def poll_refinement(
        self, cluster_id: str, *, timeout_s: float = 5.0
    ) -> Optional[GeneratorRefinement]:
        """Outbound GET: ask Tier-3 for the next refinement.

        Returns ``None`` if Tier-3 has nothing pending (HTTP 204).
        Raises :class:`CloudLinkError` on transport / decode failure.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""


# --------------------------------------------------------------------------- #
# HTTP implementation (production + tests)
# --------------------------------------------------------------------------- #

_PICKLE_MIME = "application/octet-stream"


class HTTPCloudLink(CloudLink):
    """Outbound-only HTTP client against a Tier-3 endpoint.

    Constructor takes the base URL (e.g. ``http://tier3.example/api``).
    The two endpoints are:

    * ``POST  {base_url}/partials``           — body: pickled ``ClusterPartialUpload``
    * ``GET   {base_url}/refinement/{cluster_id}`` — body: pickled ``GeneratorRefinement`` or 204
    """

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_s
        self._closed = False

    @property
    def base_url(self) -> str:
        return self._base_url

    def send_partial(self, partial: ClusterPartialUpload) -> None:
        self._raise_if_closed()
        body = pickle.dumps(partial, protocol=pickle.HIGHEST_PROTOCOL)
        req = urllib.request.Request(
            f"{self._base_url}/partials",
            data=body,
            method="POST",
            headers={"Content-Type": _PICKLE_MIME},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status not in (200, 202, 204):
                    raise CloudLinkError(
                        f"send_partial: unexpected status {resp.status}"
                    )
        except urllib.error.URLError as e:
            raise CloudLinkError(f"send_partial: {e}") from e

    def poll_refinement(
        self, cluster_id: str, *, timeout_s: float = 5.0
    ) -> Optional[GeneratorRefinement]:
        self._raise_if_closed()
        url = f"{self._base_url}/refinement/{cluster_id}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status == 204:
                    return None
                if resp.status != 200:
                    raise CloudLinkError(
                        f"poll_refinement: unexpected status {resp.status}"
                    )
                body = resp.read()
                try:
                    return pickle.loads(body)
                except Exception as e:  # pragma: no cover
                    raise CloudLinkError(
                        f"poll_refinement: pickle decode failed: {e}"
                    ) from e
        except urllib.error.URLError as e:
            raise CloudLinkError(f"poll_refinement: {e}") from e

    def close(self) -> None:
        self._closed = True

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise CloudLinkError("cloud link closed")


# --------------------------------------------------------------------------- #
# Mock Tier-3 server — in-process for tests + Sprint 2 demos
# --------------------------------------------------------------------------- #

class MockTier3Server:
    """In-process HTTP server that mimics Tier-3's inbox + outbox.

    Usage:

        srv = MockTier3Server()
        srv.start()
        # tests can read srv.received_partials and write
        # srv.queued_refinement = GeneratorRefinement(...)
        link = HTTPCloudLink(base_url=srv.url)
        ...
        srv.stop()

    Endpoints:

    * ``POST /partials`` → reads pickled :class:`ClusterPartialUpload`,
      appends to ``received_partials``. Always 202.
    * ``GET /refinement/{cluster_id}`` → if ``queued_refinement`` is
      set, return its pickled body and clear the slot; otherwise 204.

    Thread-safety: the server runs in its own thread. The accessors
    (``received_partials``, ``queued_refinement``) take a lock so the
    caller can read / write them concurrently with handler activity.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._lock = threading.RLock()
        self._received: list = []
        # S2-M5: FIFO queue of pending refinements. Tests can stage
        # several at once and the cluster will drain them in order
        # across multiple GET polls.
        self._queued: list = []

        # Late-bind the handler against this instance — easiest way to
        # share state without subclassing the global handler class.
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            # Disable noisy default access logs.
            def log_message(self, fmt, *args):  # noqa: D401, N802
                log.debug("MockTier3Server " + fmt, *args)

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.endswith("/partials"):
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    payload = pickle.loads(body)
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                if not isinstance(payload, ClusterPartialUpload):
                    self.send_response(400)
                    self.end_headers()
                    return
                with outer._lock:
                    outer._received.append(payload)
                self.send_response(202)
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if not self.path.startswith("/refinement/"):
                    self.send_response(404)
                    self.end_headers()
                    return
                with outer._lock:
                    pending = outer._queued.pop(0) if outer._queued else None
                if pending is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                body = pickle.dumps(pending, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_response(200)
                self.send_header("Content-Type", _PICKLE_MIME)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._handler_cls = _Handler
        self._httpd: Optional[socketserver.TCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: int = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def port(self) -> int:
        return self._port

    @property
    def received_partials(self) -> list:
        with self._lock:
            return list(self._received)

    def queue_refinement(self, refinement: GeneratorRefinement) -> None:
        """Append a refinement to the FIFO queue. Multiple calls stack up."""
        with self._lock:
            self._queued.append(refinement)

    @property
    def pending_refinements(self) -> int:
        """How many refinements are waiting to be polled."""
        with self._lock:
            return len(self._queued)

    def start(self) -> None:
        if self._httpd is not None:
            return
        # Bind on the requested port (0 → ephemeral).
        self._httpd = socketserver.TCPServer((self._host, self._port), self._handler_cls)
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="MockTier3Server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
