import argparse
import socket
import sys
from collections.abc import Sequence


def check_tcp(target: str, timeout: float) -> str | None:
    host, _, port_text = target.rpartition(":")
    if not host or not port_text:
        return f"invalid tcp target: {target}"
    try:
        with socket.create_connection((host, int(port_text)), timeout=timeout):
            return None
    except OSError as error:
        return f"{target}: {error}"


def run_checks(targets: Sequence[str], timeout: float) -> int:
    errors = [error for target in targets if (error := check_tcp(target, timeout))]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Container dependency health probe.")
    parser.add_argument("--tcp", action="append", default=[], help="host:port TCP target")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    raise SystemExit(run_checks(args.tcp, args.timeout))


if __name__ == "__main__":
    main()
