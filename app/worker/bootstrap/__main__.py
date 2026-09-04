from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout

from app.contracts.worker.v11 import (
    CONTRACT_VERSION,
    WorkerContractError,
    load_document,
)
from app.worker.application.events import WorkerEventEmitter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transformer-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--contract-version", type=int, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--contract-version", type=int, required=True)
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("--attempt", type=int, required=True)
    run_parser.add_argument("--attempt-id", required=True)
    run_parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    if args.contract_version != CONTRACT_VERSION:
        print("unsupported worker contract version", file=sys.stderr)
        return 2
    if args.command == "inspect":
        from app.worker.application.capabilities import inspect_capabilities

        try:
            print(json.dumps(
                inspect_capabilities(),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ))
            return 0
        except BaseException as exc:
            print(
                f"{type(exc).__name__}: worker capability inspection failed",
                file=sys.stderr,
            )
            return 1
    emitter = WorkerEventEmitter(
        job_id=args.job_id,
        attempt=args.attempt,
        attempt_id=args.attempt_id,
        stream=sys.stdout.buffer,
    )
    try:
        manifest = load_document(args.manifest, "command-manifest")
        from app.worker.application.executor import WorkerApplication

        with redirect_stdout(sys.stderr):
            WorkerApplication(emitter, sys.stdin.buffer).run(manifest)
        return 0
    except BaseException as exc:
        code, message = _safe_error(exc)
        try:
            emitter.error(code, message)
        except Exception:
            pass
        diagnostic = str(exc) or "worker execution failed"
        print(f"{type(exc).__name__}: {diagnostic}", file=sys.stderr)
        return 1


def _safe_error(exc: BaseException) -> tuple[str, str]:
    from app.worker.application.errors import WorkerExecutionError

    if isinstance(exc, WorkerExecutionError):
        return exc.code, exc.message
    if isinstance(exc, WorkerContractError):
        return (
            "WORKER_PROTOCOL_VIOLATION",
            "worker command manifest is invalid",
        )
    text = str(exc).lower()
    if "cuda" in text and (
        "out of memory" in text or "outofmemoryerror" in text
    ):
        return "CUDA_OUT_OF_MEMORY", "CUDA execution ran out of memory"
    if "cuda" in text and any(fragment in text for fragment in (
        "device has been lost",
        "cuda_error_device_lost",
        "driver shutting down",
        "initialization error",
        "not available",
    )):
        return "DEVICE_LOST", "CUDA device became unavailable during execution"
    if isinstance(exc, (OSError, ValueError)):
        return "MALFORMED_OUTPUT", "worker input or output is invalid"
    return "SUBPROCESS_FAILED", "worker execution failed"


if __name__ == "__main__":
    raise SystemExit(main())
