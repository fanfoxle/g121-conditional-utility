"""Bounded stdin/stdout judge for complete CodeContests Python programs.

The outer seam reuses the already-qualified CodeContests namespace/chroot
launcher.  This file is also copied into that sandbox as a standalone trusted
harness, so package imports are deliberately conditional.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence

if __package__:
    from .code_exec import (
        _prepare_staging,
        _remove_staging,
        _sandbox_call,
    )


PROGRAM_JUDGE_LIMITS = {
    "wall_timeout_seconds": 4.0,
    "cpu_seconds": 8,
    "address_space_bytes": 1_024 * 1_024 * 1_024,
    "processes": 32,
    "open_files": 64,
    "output_bytes": 1_024 * 1_024,
    "source_bytes": 512 * 1_024,
}
PROGRAM_VERDICT_CATEGORIES = (
    "pass",
    "wrong_answer",
    "compile_failure",
    "runtime_failure",
    "timeout",
    "runner_infrastructure_error",
)


def build_judged_payload(source: str) -> bytes:
    """Construct the sole LF-normalized, otherwise exact executable payload."""

    if not isinstance(source, str) or not source:
        raise ValueError("program must be nonempty text")
    return source.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _lf_bytes(value: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError("judge text payload must be text")
    return value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_request(
    program: str,
    tests: Sequence[Mapping[str, object]],
    wall_timeout_seconds: float,
) -> tuple[dict[str, object], ...]:
    if not isinstance(program, str) or not program:
        raise ValueError("program must be nonempty text")
    if len(_lf_bytes(program)) > PROGRAM_JUDGE_LIMITS["source_bytes"]:
        raise ValueError("program exceeds the frozen source limit")
    if not tests:
        raise ValueError("at least one frozen test is required")
    if not 0 < wall_timeout_seconds <= PROGRAM_JUDGE_LIMITS["wall_timeout_seconds"]:
        raise ValueError("wall timeout exceeds the frozen program limit")
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for test in tests:
        try:
            family = str(test["family"])
            ordinal = int(test["ordinal"])
            stdin = test["input"]
            expected = test["expected_output"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frozen test is malformed") from exc
        if family not in {"public", "private", "generated"} or ordinal < 0:
            raise ValueError("frozen test identity is invalid")
        if isinstance(ordinal, bool) or not isinstance(stdin, str) or not isinstance(expected, str):
            raise ValueError("frozen test payload is invalid")
        identity = (family, ordinal)
        if identity in seen:
            raise ValueError("duplicate frozen test identity")
        seen.add(identity)
        normalized.append(
            {
                "family": family,
                "ordinal": ordinal,
                "input": stdin.replace("\r\n", "\n").replace("\r", "\n"),
                "expected_output": expected.replace("\r\n", "\n").replace("\r", "\n"),
            }
        )
    return tuple(normalized)


def judge_codecontests_program(
    program: str,
    tests: Sequence[Mapping[str, object]],
    *,
    sandbox_base: Path,
    wall_timeout_seconds: float = 4.0,
    stop_on_first_failure: bool = False,
) -> dict[str, object]:
    """Compile once and execute one fresh bounded process per frozen test."""

    judged_payload = build_judged_payload(program)
    judged_program = judged_payload.decode("utf-8")
    frozen_tests = _validate_request(judged_program, tests, wall_timeout_seconds)
    staging, payload = _prepare_staging(sandbox_base)
    try:
        shutil.copy2(Path(__file__).resolve(), payload / "evalplus_probe.py")
        result = _sandbox_call(
            staging=staging,
            payload_dir=payload,
            mode="program_judge",
            request={
                "program": judged_program,
                "tests": frozen_tests,
                "wall_timeout_seconds": wall_timeout_seconds,
                "stop_on_first_failure": stop_on_first_failure,
            },
            outer_timeout_seconds=(
                15.0 + len(frozen_tests) * (wall_timeout_seconds + 1.0)
            ),
        )
    finally:
        _remove_staging(sandbox_base)
    if result.get("category") == "sandbox_infrastructure_error":
        result["category"] = "runner_infrastructure_error"
    category = result.get("category")
    if category not in PROGRAM_VERDICT_CATEGORIES:
        return {
            "schema": "codecontests-v2-program-judge-v1",
            "category": "runner_infrastructure_error",
            "detail": "invalid_trusted_harness_category",
            "complete_contract": False,
            "test_executions": 0,
            "test_verdicts": (),
        }
    result["schema"] = "codecontests-v2-program-judge-v1"
    result["judged_payload_sha256"] = _sha256(judged_payload)
    result["complete_contract"] = bool(
        category != "runner_infrastructure_error"
        and category != "compile_failure"
        and result.get("test_executions") == len(frozen_tests)
    )
    return result


def judge_scope_pc_program(program: str, tests: Sequence[Mapping[str, object]], *, sandbox_base: Path) -> dict[str, object]:
    """SCOPE complete-program endpoint, including its explicit empty-output outcome."""
    if not isinstance(program, str):
        raise TypeError("SCOPE program must be text")
    if not program.strip():
        return {"category": "empty_program", "passed": False, "test_executions": 0,
                "complete_contract": False, "infrastructure_error": False}
    result = judge_codecontests_program(program, tests, sandbox_base=sandbox_base)
    if result["category"] == "runner_infrastructure_error":
        raise RuntimeError(f"SCOPE program judge infrastructure error: {result}")
    return {**result, "passed": result["category"] == "pass" and result["complete_contract"],
            "infrastructure_error": False}


def _child_limits() -> None:
    os.setsid()
    for kind, value in (
        (resource.RLIMIT_CPU, PROGRAM_JUDGE_LIMITS["cpu_seconds"]),
        (resource.RLIMIT_AS, PROGRAM_JUDGE_LIMITS["address_space_bytes"]),
        (resource.RLIMIT_NPROC, PROGRAM_JUDGE_LIMITS["processes"]),
        (resource.RLIMIT_NOFILE, PROGRAM_JUDGE_LIMITS["open_files"]),
        (resource.RLIMIT_FSIZE, PROGRAM_JUDGE_LIMITS["output_bytes"]),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(kind, (int(value), int(value)))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = (ctypes.c_uint32,)
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    )
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = (ctypes.c_void_p,)
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise OSError("seccomp_init failed")
    try:
        deny = 0x00050000 | errno.EPERM
        for name in (
            b"connect",
            b"bind",
            b"listen",
            b"accept",
            b"accept4",
            b"sendto",
            b"sendmsg",
            b"sendmmsg",
            b"recvfrom",
            b"recvmsg",
            b"recvmmsg",
            b"shutdown",
            b"ptrace",
            b"process_vm_readv",
            b"process_vm_writev",
            b"pidfd_getfd",
            b"open_by_handle_at",
            b"mount",
            b"umount2",
            b"pivot_root",
            b"chroot",
            b"setns",
            b"unshare",
            b"bpf",
            b"perf_event_open",
            b"io_uring_setup",
            b"io_uring_enter",
            b"io_uring_register",
        ):
            syscall = seccomp.seccomp_syscall_resolve_name(name)
            if syscall >= 0 and seccomp.seccomp_rule_add(context, deny, syscall, 0) != 0:
                raise OSError(f"seccomp rule failed for {name.decode()}")
        if seccomp.seccomp_load(context) != 0:
            raise OSError(ctypes.get_errno(), "seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def _execute_one(program_path: Path, stdin: str, *, timeout: float) -> dict[str, object]:
    suffix = f"{time.monotonic_ns()}_{os.getpid()}"
    stdout_path = Path("/tmp") / f"cc_stdout_{suffix}"
    stderr_path = Path("/tmp") / f"cc_stderr_{suffix}"
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                ["/usr/bin/python3", "-I", "-S", str(program_path)],
                stdin=subprocess.PIPE,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd="/tmp",
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/tmp",
                    "TMPDIR": "/tmp",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONHASHSEED": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                },
                preexec_fn=_child_limits,
            )
            timed_out = False
            try:
                process.communicate(_lf_bytes(stdin), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stdout = stdout_path.read_bytes()[: PROGRAM_JUDGE_LIMITS["output_bytes"] + 1]
        stderr = stderr_path.read_bytes()[: PROGRAM_JUDGE_LIMITS["output_bytes"] + 1]
    except Exception as exc:
        return {
            "category": "runner_infrastructure_error",
            "detail": type(exc).__name__,
            "observed_output_sha256": None,
        }
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
    if timed_out:
        category = "timeout"
    elif len(stdout) > PROGRAM_JUDGE_LIMITS["output_bytes"] or len(stderr) > PROGRAM_JUDGE_LIMITS["output_bytes"]:
        category = "runtime_failure"
    elif process.returncode != 0:
        category = "runtime_failure"
    else:
        category = "pass"
    return {
        "category": category,
        "detail": "program_execution",
        "observed_output_sha256": _sha256(stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")),
        "stdout_tokens": stdout.split() if category == "pass" else None,
    }


def _inside_program_judge(request: Mapping[str, object]) -> dict[str, object]:
    program = request.get("program")
    tests = request.get("tests")
    timeout = request.get("wall_timeout_seconds")
    stop_on_first_failure = request.get("stop_on_first_failure", False)
    if not isinstance(program, str) or not isinstance(tests, list) or not isinstance(timeout, (int, float)) or type(stop_on_first_failure) is not bool:
        return {
            "category": "runner_infrastructure_error",
            "detail": "malformed_trusted_request",
            "test_executions": 0,
            "test_verdicts": [],
        }
    try:
        compile(program, "<codecontests-program>", "exec")
    except (SyntaxError, ValueError, OverflowError):
        return {
            "category": "compile_failure",
            "detail": "python_compile_failure",
            "test_executions": 0,
            "test_verdicts": [],
        }
    program_path = Path("/tmp/codecontests_program.py")
    program_path.write_bytes(_lf_bytes(program))
    verdicts: list[dict[str, object]] = []
    overall = "pass"
    try:
        for test in tests:
            family = str(test["family"])
            ordinal = int(test["ordinal"])
            stdin = str(test["input"])
            expected = str(test["expected_output"])
            execution = _execute_one(program_path, stdin, timeout=float(timeout))
            category = str(execution["category"])
            if category == "pass" and execution.pop("stdout_tokens") != _lf_bytes(expected).split():
                category = "wrong_answer"
            execution.pop("stdout_tokens", None)
            verdicts.append(
                {
                    "family": family,
                    "ordinal": ordinal,
                    "category": category,
                    "input_sha256": _sha256(_lf_bytes(stdin)),
                    "expected_output_sha256": _sha256(_lf_bytes(expected)),
                    "observed_output_sha256": execution["observed_output_sha256"],
                }
            )
            if overall == "pass" and category != "pass":
                overall = category
            if stop_on_first_failure and category != "pass":
                break
    except Exception as exc:
        overall = "runner_infrastructure_error"
        verdicts.append(
            {
                "family": None,
                "ordinal": None,
                "category": overall,
                "detail": type(exc).__name__,
            }
        )
    finally:
        program_path.unlink(missing_ok=True)
    return {
        "category": overall,
        "detail": "complete_sampled_contract",
        "test_executions": len(verdicts),
        "test_verdicts": verdicts,
    }


def _inside_main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        result = _inside_program_judge(request)
    except Exception as exc:
        result = {
            "category": "runner_infrastructure_error",
            "detail": type(exc).__name__,
            "test_executions": 0,
            "test_verdicts": [],
        }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    parser_mode = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[1] == "--inside-mode" else None
    raise SystemExit(_inside_main() if parser_mode == "program_judge" else 2)
