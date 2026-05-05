"""Server-side topology config for Experiment 1.

Three configuration modes coexist:

1. **Explicit JSON** — ``--topology configs/exp1_aerpaw.json``. The
   paper-run mode; reproducible.
2. **Explicit CLI** — ``--client d1@192.168.1.11:partition=0`` flags.
   Convenient for one-off runs.
3. **Discovery** — ``--discover --n-clients 4``. The server accepts
   the first N clients that REGISTER (with unique partitions
   covering 0..N-1). Dev-mode convenience.

All three populate the same :class:`Exp1Topology` dataclass. The server's
registration validator reads the topology to accept/reject each
connecting client; the rest of the protocol is mode-agnostic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ClientSlot:
    """One expected (or accepted) client.

    In explicit modes the slots come from config; in discovery mode
    they're filled in as clients connect. ``host`` is ``None`` in
    discovery mode until a client claims the slot.
    """

    client_id: str
    data_partition: int
    host: Optional[str] = None  # None = "any IP" (discovery mode)


@dataclass
class Exp1Topology:
    """Server-side topology: bind address + expected clients."""

    bind_host: str = "0.0.0.0"
    bind_port: int = 9000
    registration_timeout_s: float = 60.0
    # Either fully populated (explicit modes) or partially populated
    # with placeholder slots (discovery mode).
    clients: List[ClientSlot] = field(default_factory=list)
    # Discovery mode flag. When True the server accepts any IP for any
    # slot; when False the server requires the connecting IP to match
    # ``ClientSlot.host`` (warn-only unless ``strict_ip=True``).
    discover: bool = False
    strict_ip: bool = False
    # B_nominal carried for documentation only — link shaping is the
    # operator's responsibility (tc/netem, AERPAW, Chameleon).
    B_nominal_mbps: float = 10.0

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    def expected_partitions(self) -> List[int]:
        return [s.data_partition for s in self.clients]

    def expected_client_ids(self) -> List[str]:
        return [s.client_id for s in self.clients]

    # ------------------------------------------------------------------ #
    # JSON round-trip
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        payload = {
            "server": {
                "host": self.bind_host,
                "port": self.bind_port,
                "registration_timeout_s": self.registration_timeout_s,
            },
            "clients": [
                {
                    "client_id": s.client_id,
                    "host": s.host,
                    "data_partition": s.data_partition,
                }
                for s in self.clients
            ],
            "shared_link": {
                "B_nominal_mbps": self.B_nominal_mbps,
                "shaping_owner": "operator",
            },
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "Exp1Topology":
        raw = json.loads(payload)
        server = raw.get("server", {})
        clients = [
            ClientSlot(
                client_id=c["client_id"],
                data_partition=int(c["data_partition"]),
                host=c.get("host"),
            )
            for c in raw.get("clients", [])
        ]
        link = raw.get("shared_link", {})
        return cls(
            bind_host=server.get("host", "0.0.0.0"),
            bind_port=int(server.get("port", 9000)),
            registration_timeout_s=float(
                server.get("registration_timeout_s", 60.0)
            ),
            clients=clients,
            B_nominal_mbps=float(link.get("B_nominal_mbps", 10.0)),
        )

    @classmethod
    def from_file(cls, path: Path) -> "Exp1Topology":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if not self.discover and not self.clients:
            raise ValueError(
                "explicit-mode topology must declare at least one client; "
                "use --discover for discovery mode"
            )
        if self.discover and self.clients and any(
            s.client_id != "" or s.host is not None
            for s in self.clients
        ):
            # Discovery mode shouldn't have pre-populated client IDs/hosts.
            # Slots in discovery mode carry only the expected partition.
            raise ValueError(
                "discovery-mode topology must not pre-populate client_id "
                "or host on slots; clients self-declare"
            )
        # Partition coverage: must be 0..N-1 with no duplicates.
        partitions = self.expected_partitions()
        n = len(partitions)
        if sorted(partitions) != list(range(n)):
            raise ValueError(
                f"client partitions must cover 0..{n - 1} exactly once, "
                f"got {sorted(partitions)}"
            )
        # Unique client_ids (explicit mode only — discovery has empty IDs).
        if not self.discover:
            ids = self.expected_client_ids()
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate client_id in topology: {ids}")


# --------------------------------------------------------------------------- #
# CLI parsing — three modes via argparse
# --------------------------------------------------------------------------- #

# Format: "client_id@host:partition=N"  or  "client_id@host"  (auto-partition)
_CLIENT_FLAG_RE = re.compile(
    r"^(?P<id>[A-Za-z0-9_-]+)@(?P<host>[^:]+)(?::partition=(?P<part>\d+))?$"
)


def add_topology_arguments(parser: argparse.ArgumentParser) -> None:
    """Wire the three topology modes onto an ``argparse.ArgumentParser``.

    Mutually exclusive: ``--topology FILE`` vs. ``--client ...`` vs.
    ``--discover --n-clients N``. Validation defers to
    :func:`build_topology_from_args` so the parser stays simple.
    """
    g = parser.add_argument_group("topology")
    g.add_argument(
        "--topology",
        type=Path,
        default=None,
        help=(
            "Explicit topology JSON. Overrides any --client / --discover "
            "flags. Use this for paper-run reproducibility."
        ),
    )
    g.add_argument(
        "--client",
        action="append",
        default=[],
        metavar="CLIENT_ID@HOST[:partition=N]",
        help=(
            "Explicit client entry; pass once per client. Example: "
            "'d1@192.168.1.11:partition=0'. If :partition= is omitted, "
            "partitions auto-assign in flag order."
        ),
    )
    g.add_argument(
        "--discover",
        action="store_true",
        help="Accept the first --n-clients connections as the topology.",
    )
    g.add_argument(
        "--n-clients",
        type=int,
        default=None,
        help="Required with --discover. Number of clients to accept.",
    )
    g.add_argument(
        "--bind-host",
        default="0.0.0.0",
        help="Server bind address (default: 0.0.0.0 = all interfaces).",
    )
    g.add_argument(
        "--bind-port",
        type=int,
        default=9000,
        help="Server bind port (default: 9000).",
    )
    g.add_argument(
        "--strict-ip",
        action="store_true",
        help=(
            "Reject a connecting client whose source IP does not match "
            "its slot's expected host. Default: warn-only."
        ),
    )
    g.add_argument(
        "--registration-timeout-s",
        type=float,
        default=60.0,
        help="Wait this long for all clients to REGISTER before exiting.",
    )


def build_topology_from_args(args: argparse.Namespace) -> Exp1Topology:
    """Resolve the three topology modes into one :class:`Exp1Topology`.

    Precedence: ``--topology`` (JSON) > explicit ``--client`` flags >
    ``--discover``. Mutually exclusive — passing more than one raises
    ``argparse.ArgumentTypeError``.
    """
    n_explicit_modes = sum([
        args.topology is not None,
        bool(args.client),
        args.discover,
    ])
    if n_explicit_modes == 0:
        raise argparse.ArgumentTypeError(
            "must specify exactly one of --topology, --client, or --discover"
        )
    if n_explicit_modes > 1:
        raise argparse.ArgumentTypeError(
            "--topology / --client / --discover are mutually exclusive"
        )

    if args.topology is not None:
        topo = Exp1Topology.from_file(args.topology)
        # CLI overrides for bind args take precedence over JSON.
        # (Lets a paper-run JSON be portable across environments.)
        if args.bind_host != "0.0.0.0":
            topo.bind_host = args.bind_host
        if args.bind_port != 9000:
            topo.bind_port = args.bind_port
        if args.strict_ip:
            topo.strict_ip = True
        topo.validate()
        return topo

    if args.client:
        clients: List[ClientSlot] = []
        for i, raw in enumerate(args.client):
            m = _CLIENT_FLAG_RE.match(raw)
            if not m:
                raise argparse.ArgumentTypeError(
                    f"--client value {raw!r} doesn't match "
                    f"CLIENT_ID@HOST[:partition=N]"
                )
            partition = int(m.group("part")) if m.group("part") else i
            clients.append(ClientSlot(
                client_id=m.group("id"),
                host=m.group("host"),
                data_partition=partition,
            ))
        topo = Exp1Topology(
            bind_host=args.bind_host,
            bind_port=args.bind_port,
            registration_timeout_s=args.registration_timeout_s,
            clients=clients,
            strict_ip=args.strict_ip,
        )
        topo.validate()
        return topo

    # Discovery mode.
    if args.n_clients is None or args.n_clients < 1:
        raise argparse.ArgumentTypeError(
            "--discover requires --n-clients N (with N ≥ 1)"
        )
    # Pre-populate placeholder slots — partitions 0..N-1 — that get
    # filled in as clients register. Slot client_id stays empty until
    # the client claims it.
    placeholder_slots = [
        ClientSlot(client_id="", data_partition=i, host=None)
        for i in range(args.n_clients)
    ]
    # Validation in discovery mode rejects pre-populated IDs/hosts; our
    # placeholders satisfy that, so validate() will pass.
    topo = Exp1Topology(
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        registration_timeout_s=args.registration_timeout_s,
        clients=placeholder_slots,
        discover=True,
    )
    topo.validate()
    return topo
