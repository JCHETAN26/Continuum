import argparse
import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HealthTarget:
    name: str
    url: str


@dataclass(frozen=True)
class HealthResult:
    name: str
    healthy: bool
    detail: str


DEFAULT_TARGETS = (
    HealthTarget("ingest", "http://localhost:8000/health"),
    HealthTarget("drift", "http://localhost:8001/health"),
    HealthTarget("server", "http://localhost:8002/health"),
    HealthTarget("trainer-api", "http://localhost:8003/health"),
    HealthTarget("linguistic-drift", "http://localhost:8004/health"),
    HealthTarget("dashboard", "http://localhost:3000"),
)


async def check_target(client: httpx.AsyncClient, target: HealthTarget) -> HealthResult:
    try:
        response = await client.get(target.url)
        if 200 <= response.status_code < 300:
            return HealthResult(target.name, True, f"{response.status_code} {target.url}")
        return HealthResult(target.name, False, f"{response.status_code} {target.url}")
    except httpx.HTTPError as error:
        return HealthResult(target.name, False, f"{target.url}: {error}")


async def check_targets(
    client: httpx.AsyncClient, targets: Sequence[HealthTarget]
) -> list[HealthResult]:
    return await asyncio.gather(*(check_target(client, target) for target in targets))


async def wait_for_stack(
    timeout_seconds: float,
    interval_seconds: float,
    targets: Sequence[HealthTarget] = DEFAULT_TARGETS,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    last_results: list[HealthResult] = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            last_results = await check_targets(client, targets)
            print_results(last_results)
            if all(result.healthy for result in last_results):
                return 0
            await asyncio.sleep(interval_seconds)

    print(f"Timed out after {timeout_seconds:.0f}s waiting for stack health.")
    if last_results:
        print_results(last_results)
    return 1


def print_results(results: Sequence[HealthResult]) -> None:
    for result in results:
        prefix = "PASS" if result.healthy else "WAIT"
        print(f"{prefix} {result.name}: {result.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the local Continuum stack to be healthy."
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(wait_for_stack(args.timeout, args.interval)))


if __name__ == "__main__":
    main()
