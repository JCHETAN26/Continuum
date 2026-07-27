"""Docker connection setup shared by the Testcontainers integration suite.

Two Docker Desktop (macOS) behaviours break the defaults, so both are corrected
here rather than in every test. Each setting is only applied when the caller has
not already chosen a value, so CI and ad-hoc runs stay overridable.

1. Socket path. Docker Desktop makes its socket the current context's endpoint at
   ``~/.docker/run/docker.sock``. Testcontainers bind-mounts that path into the
   Ryuk reaper container, and the Docker Desktop VM cannot mount a path under the
   user's home into ``/host_mnt`` — creation fails with "operation not supported".
   The ``/var/run/docker.sock`` symlink points at the same daemon and *is*
   mountable. On Linux CI this resolves to the same socket, so it is a no-op.

2. Ryuk. The reaper races its own port publish: ``get_exposed_port(8080)`` can be
   queried before Docker has mapped it, which fails the first test and leaves a
   container whose name then 409s every subsequent test. Ryuk is a safety net for
   crashed runs, and every test here manages its containers with a ``with`` block
   that stops them deterministically, so turning it off costs no cleanup.
"""

import os
from pathlib import Path

_DEFAULT_SOCKET = Path("/var/run/docker.sock")


def pytest_configure(config):  # noqa: ARG001 - pytest hook signature
    if not os.getenv("DOCKER_HOST") and _DEFAULT_SOCKET.exists():
        os.environ["DOCKER_HOST"] = f"unix://{_DEFAULT_SOCKET}"

    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
