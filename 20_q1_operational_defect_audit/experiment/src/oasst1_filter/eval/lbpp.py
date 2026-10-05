"""Fixed-revision LBPP decoding, assembly, and isolated Python judging."""

from __future__ import annotations

import base64
import binascii
import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickletools
import re
try:
    import resource
except ImportError:  # pragma: no cover - the execution backend is WSL Linux only
    resource = None  # type: ignore[assignment]
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


_SAFE_PICKLE_GRAMMARS = {
    ("PROTO", "FRAME", "SHORT_BINUNICODE", "MEMOIZE", "STOP"),
    ("PROTO", "FRAME", "BINUNICODE", "MEMOIZE", "STOP"),
}
_TASK_ID_PATTERN = re.compile(r"lbpp/python/[0-9]{3}\Z")
_TEST_MARKER_PATTERN = re.compile(r"(?m)^# TEST$")
_REQUIRED_ROW_FIELDS = frozenset(
    {
        "task_id", "language", "instruction", "signature", "completion",
        "test_setup", "test_list", "test_file",
    }
)
_DISPOSITION_TO_COUNT = {
    "canary": "N_canary",
    "duplicate": "N_duplicate",
    "used": "N_used",
    "overlap": "N_overlap",
    "history_unknown": "N_history_unknown",
    "static_unsupported": "N_static_unsupported",
    "static_candidate": "N_static",
}
LBPP_LIMITS: dict[str, int | float] = {
    "task_wall_seconds": 60.0,
    "task_cpu_seconds": 60,
    "memory_bytes": 2 * 1024 * 1024 * 1024,
    "processes_and_threads": 16,
    "stdout_bytes": 1024 * 1024,
    "stderr_bytes": 1024 * 1024,
    "writable_tmp_bytes": 64 * 1024 * 1024,
    "source_bytes": 1024 * 1024,
    "open_files": 64,
    "sandbox_setup_grace_seconds": 20.0,
}

_SANDBOX_SCRIPT = r"""
set -eu
umask 0022
ROOTFS="$1"
PAYLOAD="$2"
DEPS="$3"
mount --make-rprivate /
mount -t tmpfs -o size=16m,mode=0755,nosuid,nodev tmpfs "$ROOTFS"
mkdir -p "$ROOTFS/usr" "$ROOTFS/etc" "$ROOTFS/proc" "$ROOTFS/dev" \
  "$ROOTFS/tmp" "$ROOTFS/payload" "$ROOTFS/taskdeps" "$ROOTFS/home" \
  "$ROOTFS/root" "$ROOTFS/mnt"
ln -s usr/bin "$ROOTFS/bin"
ln -s usr/sbin "$ROOTFS/sbin"
ln -s usr/lib "$ROOTFS/lib"
ln -s usr/lib64 "$ROOTFS/lib64"
printf 'root:x:0:0:root:/root:/bin/false\nnobody:x:65534:65534:nobody:/tmp:/bin/false\n' > "$ROOTFS/etc/passwd"
printf 'root:x:0:\nnogroup:x:65534:\n' > "$ROOTFS/etc/group"
mount --bind /usr "$ROOTFS/usr"
mount -o remount,bind,ro "$ROOTFS/usr"
mount --bind "$PAYLOAD" "$ROOTFS/payload"
mount -o remount,bind,ro "$ROOTFS/payload"
if [ "$DEPS" != "-" ]; then
  mount --bind "$DEPS" "$ROOTFS/taskdeps"
  mount -o remount,bind,ro "$ROOTFS/taskdeps"
fi
mount -t proc -o nosuid,nodev,noexec,hidepid=2 proc "$ROOTFS/proc"
mount -t tmpfs -o size=4m,mode=0755,nosuid,noexec tmpfs "$ROOTFS/dev"
mknod -m 666 "$ROOTFS/dev/null" c 1 3
mknod -m 666 "$ROOTFS/dev/zero" c 1 5
mknod -m 444 "$ROOTFS/dev/urandom" c 1 9
chown 0:0 "$ROOTFS/dev/null" "$ROOTFS/dev/zero" "$ROOTFS/dev/urandom"
ln -s /proc/self/fd "$ROOTFS/dev/fd"
ln -s /proc/self/fd/0 "$ROOTFS/dev/stdin"
ln -s /proc/self/fd/1 "$ROOTFS/dev/stdout"
ln -s /proc/self/fd/2 "$ROOTFS/dev/stderr"
mount -o remount,ro "$ROOTFS/dev"
mount -t tmpfs -o size=64m,mode=1777,nosuid,nodev tmpfs "$ROOTFS/tmp"
mount -o remount,ro "$ROOTFS"
exec chroot "$ROOTFS" /usr/bin/setpriv \
  --reuid 65534 --regid 65534 --clear-groups \
  --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
  /usr/bin/env -i PATH=/usr/bin:/bin HOME=/tmp TMPDIR=/tmp \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/taskdeps CUDA_VISIBLE_DEVICES= \
  /usr/bin/python3 /payload/lbpp_probe.py --inside-mode judge
"""

_CGROUP_LAUNCH_SCRIPT = r"""
set -eu
CGROUP="$1"
CPU="$2"
shift 2
echo $$ > "$CGROUP/cgroup.procs"
exec /usr/bin/taskset -c "$CPU" "$@"
"""

_RUNNER_SOURCE = """\
import os
import runpy
import traceback

try:
    runpy.run_path('/tmp/test_file.py', run_name='__main__')
except AssertionError:
    os._exit(20)
except BaseException:
    traceback.print_exc()
    os._exit(21)
os._exit(0)
"""


def _validate_json_shape(value: Any, *, depth: int = 0) -> None:
    if depth > 20:
        raise ValueError("decoded JSON nesting exceeds limit")
    if value is None or type(value) in (bool, int, float):
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 1024 * 1024:
            raise ValueError("decoded JSON string exceeds limit")
        return
    if isinstance(value, list):
        if len(value) > 10_000:
            raise ValueError("decoded JSON list exceeds limit")
        for item in value:
            _validate_json_shape(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 10_000 or any(not isinstance(key, str) for key in value):
            raise ValueError("decoded JSON object exceeds limit")
        for item in value.values():
            _validate_json_shape(item, depth=depth + 1)
        return
    raise ValueError("decoded JSON contains an unsupported value")


def decode_lbpp_cell(
    encoded: str,
    *,
    expected_type: type,
    max_compressed_bytes: int = 1024 * 1024,
    max_decompressed_bytes: int = 1024 * 1024,
) -> Any:
    """Decode LBPP's base64/zlib/pickle(JSON) cell without unpickling.

    Only the observed literal-string pickle grammar is accepted.  In
    particular, no GLOBAL/REDUCE or object-construction opcode is executable.
    """

    if not isinstance(encoded, str):
        raise TypeError("encoded LBPP cell must be a string")
    try:
        compressed = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("invalid base64 LBPP cell") from exc
    if len(compressed) > max_compressed_bytes:
        raise ValueError("compressed size exceeds limit")
    decompressor = zlib.decompressobj()
    try:
        payload = decompressor.decompress(compressed, max_decompressed_bytes + 1)
        payload += decompressor.flush()
    except zlib.error as exc:
        raise ValueError("invalid zlib LBPP cell") from exc
    if (
        len(payload) > max_decompressed_bytes
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise ValueError("decompressed size or stream boundary exceeds limit")
    try:
        operations = list(pickletools.genops(payload))
    except Exception as exc:
        raise ValueError("invalid pickle grammar") from exc
    grammar = tuple(operation.name for operation, _argument, _position in operations)
    if grammar not in _SAFE_PICKLE_GRAMMARS or len(operations) != 5:
        raise ValueError("pickle grammar is not the literal-string contract")
    json_text = operations[2][1]
    if not isinstance(json_text, str):
        raise ValueError("pickle grammar did not contain a string literal")
    try:
        decoded = json.loads(json_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("pickle string does not contain valid JSON") from exc
    if not isinstance(decoded, expected_type):
        raise ValueError("decoded JSON type does not match contract")
    _validate_json_shape(decoded)
    return decoded


def _normalize_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def render_nat_lbpp_prompt(*, instruction: str, signature: str) -> str:
    """Render only the two model-visible LBPP fields under the frozen contract."""

    if not isinstance(instruction, str) or not isinstance(signature, str):
        raise TypeError("LBPP prompt fields must be strings")

    def commented(field: str) -> str:
        return "\n".join(f"# {line}" for line in _normalize_lf(field).strip().split("\n"))

    return (
        "# Task: implement the Python module specified below.\n"
        "# Output only the complete source of code.py, including any imports, helper functions, and classes.\n"
        "# Do not use Markdown fences or explanatory prose.\n"
        "# Requirements:\n"
        f"{commented(instruction)}\n"
        "# Required interface:\n"
        f"{commented(signature)}\n"
        "# Complete code.py:\n"
    )


def consume_nat_lbpp_completion(
    sequence: Sequence[int],
    *,
    input_width: int,
    tokenizer: Any,
    eos_token_id: int = 151_643,
    max_new_tokens: int = 2_048,
) -> dict[str, Any]:
    """Consume the literal new-token suffix without source-level repair."""

    if type(input_width) is not int or input_width < 0 or input_width > len(sequence):
        raise ValueError("invalid generation input width")
    suffix = [int(token) for token in sequence[input_width:]]
    if len(suffix) > max_new_tokens:
        raise ValueError("generation exceeds the frozen new-token budget")
    eos_reached = eos_token_id in suffix
    retained = suffix[: suffix.index(eos_token_id)] if eos_reached else suffix
    completion = _normalize_lf(
        tokenizer.decode(
            retained,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    return {
        "completion": completion,
        "generated_token_ids": retained,
        "generated_tokens": len(retained),
        "eos_reached": eos_reached,
        "length_cap_reached": not eos_reached and len(suffix) == max_new_tokens,
        "empty": not completion.strip(),
    }


def assemble_complete_program(model_output: str) -> str:
    """Use one complete ``code.py`` output without prompt concatenation/truncation."""

    if not isinstance(model_output, str):
        raise TypeError("complete program output must be a string")
    normalized = _normalize_lf(model_output)
    if not normalized.strip():
        raise ValueError("complete program output is empty")
    if len(normalized.encode("utf-8")) > 1024 * 1024:
        raise ValueError("complete program output exceeds limit")
    return normalized


def build_lbpp_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the official Python ``test_file`` + adjacent ``code.py`` contract."""

    task_id = row.get("task_id")
    instruction = row.get("instruction")
    signature = row.get("signature")
    if (
        not isinstance(task_id, str)
        or _TASK_ID_PATTERN.fullmatch(task_id) is None
        or not isinstance(instruction, str)
        or not isinstance(signature, str)
    ):
        raise ValueError("malformed LBPP identity or prompt fields")
    completion = decode_lbpp_cell(row.get("completion"), expected_type=str)
    test_setup = decode_lbpp_cell(row.get("test_setup"), expected_type=str)
    test_list = decode_lbpp_cell(row.get("test_list"), expected_type=list)
    test_file = decode_lbpp_cell(row.get("test_file"), expected_type=str)
    if not all(isinstance(test, str) and test.strip() for test in test_list):
        raise ValueError("LBPP test_list must contain nonempty strings")
    normalized_test = _normalize_lf(test_file)
    normalized_setup = _normalize_lf(test_setup)
    if not normalized_test.startswith(normalized_setup):
        raise ValueError("official test_file does not preserve test_setup prefix")
    if any(_normalize_lf(test).strip() not in normalized_test for test in test_list):
        raise ValueError("official test_file does not cover every test_list entry")
    expected_test_count = len(_TEST_MARKER_PATTERN.findall(normalized_test))
    if not test_list or expected_test_count < len(test_list):
        raise ValueError("official test_file has no complete executable test contract")
    try:
        ast.parse(normalized_test, filename=f"{task_id}:test_file")
    except SyntaxError as exc:
        raise ValueError("official test_file is not valid Python") from exc
    prompt = f"{_normalize_lf(instruction).strip()}\n\n{_normalize_lf(signature).strip()}\n"
    return {
        "task_id": task_id,
        "prompt": prompt,
        "canonical_program": assemble_complete_program(completion),
        "test_source": normalized_test,
        "test_setup": normalized_setup,
        "test_list": test_list,
        "expected_test_count": expected_test_count,
        "test_mapping": "official_test_file_with_adjacent_code_py",
        "program_assembly": "one_complete_code_py_output_normalized_to_lf",
    }


def select_lbpp_concurrency(
    *, effective_cpus: int, available_memory_bytes: int
) -> int | None:
    """Choose the largest frozen rung allowed by CPU and memory headroom."""

    if type(effective_cpus) is not int or type(available_memory_bytes) is not int:
        raise TypeError("LBPP resource observations must be integers")
    if effective_cpus < 0 or available_memory_bytes < 0:
        raise ValueError("LBPP resource observations cannot be negative")
    reserve = 2 * 1024 * 1024 * 1024
    per_task = int(LBPP_LIMITS["memory_bytes"])
    for concurrency in (4, 2, 1):
        if effective_cpus >= concurrency and available_memory_bytes >= concurrency * per_task + reserve:
            return concurrency
    return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _child_limits() -> None:
    if resource is None:
        raise RuntimeError("LBPP resource limits require POSIX")
    os.setsid()
    for kind, value in (
        (resource.RLIMIT_CPU, int(LBPP_LIMITS["task_cpu_seconds"])),
        (resource.RLIMIT_AS, int(LBPP_LIMITS["memory_bytes"])),
        (resource.RLIMIT_NPROC, int(LBPP_LIMITS["processes_and_threads"])),
        (resource.RLIMIT_FSIZE, int(LBPP_LIMITS["stdout_bytes"])),
        (resource.RLIMIT_NOFILE, int(LBPP_LIMITS["open_files"])),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(kind, (value, value))


def _inside_judge(request: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    program = request.get("program")
    test_source = request.get("test_source")
    expected_test_count = request.get("expected_test_count")
    expected_cpu = request.get("expected_cpu")
    outer_net_namespace = request.get("outer_net_namespace")
    if (
        not isinstance(program, str)
        or not isinstance(test_source, str)
        or type(expected_test_count) is not int
        or expected_test_count < 1
        or type(expected_cpu) is not int
        or not isinstance(outer_net_namespace, str)
    ):
        return {
            "category": "sandbox_infrastructure_error",
            "detail": "malformed_trusted_payload",
            "test_executions": 0,
        }
    if any(
        len(value.encode("utf-8")) > int(LBPP_LIMITS["source_bytes"])
        for value in (program, test_source)
    ):
        return {
            "category": "sandbox_infrastructure_error",
            "detail": "source_limit_exceeded",
            "test_executions": 0,
        }
    if len(_TEST_MARKER_PATTERN.findall(test_source)) != expected_test_count:
        return {
            "category": "sandbox_infrastructure_error",
            "detail": "test_count_contract_mismatch",
            "test_executions": 0,
        }
    try:
        compile(program, "code.py", "exec")
    except (SyntaxError, MemoryError, RecursionError, OverflowError) as exc:
        return {
            "category": "compile_failure",
            "detail": type(exc).__name__,
            "test_executions": 0,
            "wall_seconds": time.perf_counter() - started,
        }
    try:
        compile(test_source, "test_file.py", "exec")
    except SyntaxError as exc:
        return {
            "category": "sandbox_infrastructure_error",
            "detail": f"invalid_test_contract:{type(exc).__name__}",
            "test_executions": 0,
            "wall_seconds": time.perf_counter() - started,
        }
    Path("/tmp/code.py").write_text(program, encoding="utf-8", newline="\n")
    Path("/tmp/test_file.py").write_text(test_source, encoding="utf-8", newline="\n")
    Path("/tmp/lbpp_runner.py").write_text(_RUNNER_SOURCE, encoding="utf-8", newline="\n")
    stdout_path = Path("/tmp/candidate.stdout")
    stderr_path = Path("/tmp/candidate.stderr")
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            ["/usr/bin/python3", "/tmp/lbpp_runner.py"],
            cwd="/tmp",
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
                "TMPDIR": "/tmp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": "/taskdeps",
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            preexec_fn=_child_limits,
        )
        timed_out = False
        try:
            process.wait(timeout=float(request.get("wall_timeout_seconds", 60.0)))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    if timed_out:
        category, detail, test_executions = "timeout", "task_wall_limit", 0
    elif len(stdout) >= int(LBPP_LIMITS["stdout_bytes"]) or len(stderr) >= int(
        LBPP_LIMITS["stderr_bytes"]
    ):
        category, detail, test_executions = "resource_limit", "output_limit", 0
    elif process.returncode == 0:
        category, detail, test_executions = "pass", "all_official_tests_completed", expected_test_count
    elif process.returncode == 20:
        category, detail, test_executions = "wrong_answer", "assertion_failure", 0
    elif process.returncode == 21:
        category, detail, test_executions = "runtime_failure", "test_process_exception", 0
    else:
        category, detail, test_executions = "runtime_failure", "test_process_failed", 0
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "category": category,
        "detail": detail,
        "returncode": process.returncode,
        "test_executions": test_executions,
        "expected_test_count": expected_test_count,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "wall_seconds": time.perf_counter() - started,
        "isolation": {
            "uid_nonroot": os.getuid() == 65534,
            "gid_nonroot": os.getgid() == 65534,
            "single_cpu_affinity": affinity == [expected_cpu],
            "cpu_affinity": affinity,
            "network_namespace_fresh": os.readlink("/proc/self/ns/net")
            != outer_net_namespace,
            "workspace_hidden": not Path("/mnt/d/study/hq_datasets_v2").exists(),
            "gpu_hidden": not Path("/dev/dxg").exists()
            and not tuple(Path("/dev").glob("nvidia*")),
            "root_read_only": _write_probe_fails(Path("/root_write_probe")),
            "tmp_writable": _write_probe_succeeds(Path("/tmp/write_probe")),
        },
    }


def _write_probe_succeeds(path: Path) -> bool:
    try:
        path.write_bytes(b"x")
        path.unlink()
    except OSError:
        return False
    return True


def _write_probe_fails(path: Path) -> bool:
    return not _write_probe_succeeds(path)


def _cgroup_values(cgroup: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in ("memory.max", "memory.peak", "pids.max", "pids.peak"):
        path = cgroup / name
        if path.is_file():
            raw = path.read_text("ascii").strip()
            values[name.replace(".", "_")] = int(raw) if raw.isdigit() else raw
    events: dict[str, int] = {}
    path = cgroup / "memory.events"
    if path.is_file():
        for line in path.read_text("ascii").splitlines():
            key, value = line.split()
            events[key] = int(value)
    values["memory_events"] = events
    return values


def _release_cgroup(cgroup: Path) -> bool:
    if not cgroup.exists():
        return True
    kill_path = cgroup / "cgroup.kill"
    if kill_path.is_file():
        try:
            kill_path.write_text("1", encoding="ascii")
        except OSError:
            pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            if (cgroup / "pids.current").read_text("ascii").strip() == "0":
                cgroup.rmdir()
                return True
        except (FileNotFoundError, OSError):
            if not cgroup.exists():
                return True
        time.sleep(0.02)
    return False


def judge_lbpp_program(
    program: str,
    *,
    test_source: str,
    expected_test_count: int,
    task_id: str,
    sandbox_base: Path,
    cpu_id: int,
    dependency_path: Path | None = None,
    wall_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Judge one complete ``code.py`` under a private namespace and cgroup."""

    started = time.perf_counter()
    if os.name != "posix" or os.geteuid() != 0:
        raise RuntimeError("LBPP sandbox requires root inside WSL Linux")
    if not isinstance(task_id, str) or _TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError("malformed LBPP task_id")
    if not isinstance(program, str) or not isinstance(test_source, str):
        raise TypeError("LBPP program and test source must be strings")
    if type(expected_test_count) is not int or expected_test_count < 1:
        return {
            "task_id": task_id,
            "category": "sandbox_infrastructure_error",
            "detail": "zero_tests_not_executable",
            "test_executions": 0,
            "sandbox_released": True,
            "cgroup_released": True,
            "wall_seconds": time.perf_counter() - started,
        }
    if not 0 < float(wall_timeout_seconds) <= float(LBPP_LIMITS["task_wall_seconds"]):
        raise ValueError("LBPP task wall timeout exceeds frozen limit")
    affinity = os.sched_getaffinity(0)
    if type(cpu_id) is not int or cpu_id not in affinity:
        raise ValueError("LBPP logical CPU is unavailable")
    dependencies = "-"
    if dependency_path is not None:
        resolved_dependencies = Path(dependency_path).resolve()
        if not resolved_dependencies.is_dir():
            raise ValueError("LBPP dependency path is unavailable")
        dependencies = str(resolved_dependencies)
    sandbox_base = Path(sandbox_base).resolve()
    sandbox_base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="lbpp_task_", dir=sandbox_base))
    payload = staging / "payload"
    rootfs = staging / "rootfs"
    payload.mkdir()
    rootfs.mkdir()
    shutil.copy2(Path(__file__).resolve(), payload / "lbpp_probe.py")
    (payload / "lbpp_probe.py").chmod(0o444)
    cgroup_root = Path("/sys/fs/cgroup")
    cgroup = cgroup_root / f"lbpp_q_{os.getpid()}_{secrets.token_hex(6)}"
    process: subprocess.Popen[bytes] | None = None
    result: dict[str, Any]
    resource_values: dict[str, Any] = {}
    cgroup_released = False
    try:
        if cgroup_root.stat().st_dev == 0 or not (cgroup_root / "cgroup.controllers").is_file():
            raise RuntimeError("cgroup v2 is unavailable")
        cgroup.mkdir(mode=0o755)
        (cgroup / "memory.max").write_text(str(int(LBPP_LIMITS["memory_bytes"])), "ascii")
        if (cgroup / "memory.swap.max").is_file():
            (cgroup / "memory.swap.max").write_text("0", "ascii")
        (cgroup / "pids.max").write_text(
            str(int(LBPP_LIMITS["processes_and_threads"])), "ascii"
        )
        namespace_command = [
            "/usr/bin/unshare",
            "--mount",
            "--uts",
            "--ipc",
            "--net",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "--propagation",
            "private",
            "/bin/bash",
            "-c",
            _SANDBOX_SCRIPT,
            "sandbox",
            str(rootfs),
            str(payload),
            dependencies,
        ]
        command = [
            "/bin/bash",
            "-c",
            _CGROUP_LAUNCH_SCRIPT,
            "launcher",
            str(cgroup),
            str(cpu_id),
            *namespace_command,
        ]
        request = {
            "program": assemble_complete_program(program),
            "test_source": _normalize_lf(test_source),
            "expected_test_count": expected_test_count,
            "expected_cpu": cpu_id,
            "outer_net_namespace": os.readlink("/proc/self/ns/net"),
            "wall_timeout_seconds": float(wall_timeout_seconds),
        }
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
        )
        try:
            stdout, stderr = process.communicate(
                _canonical_bytes(request),
                timeout=float(wall_timeout_seconds)
                + float(LBPP_LIMITS["sandbox_setup_grace_seconds"]),
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            result = {
                "category": "timeout",
                "detail": "outer_task_wall_limit",
                "test_executions": 0,
            }
        else:
            if process.returncode != 0:
                result = {
                    "category": "sandbox_infrastructure_error",
                    "detail": "namespace_setup_failed",
                    "returncode": process.returncode,
                    "stderr_sha256": _sha256_bytes(stderr),
                    "stderr_bytes": len(stderr),
                    "test_executions": 0,
                }
            elif len(stdout) > int(LBPP_LIMITS["stdout_bytes"]):
                result = {
                    "category": "sandbox_infrastructure_error",
                    "detail": "trusted_harness_output_limit",
                    "test_executions": 0,
                }
            else:
                try:
                    decoded = json.loads(stdout.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    decoded = None
                if not isinstance(decoded, dict) or not isinstance(
                    decoded.get("category"), str
                ):
                    result = {
                        "category": "sandbox_infrastructure_error",
                        "detail": "invalid_trusted_harness_result",
                        "stdout_sha256": _sha256_bytes(stdout),
                        "stderr_sha256": _sha256_bytes(stderr),
                        "test_executions": 0,
                    }
                else:
                    result = decoded
        resource_values = _cgroup_values(cgroup)
        if resource_values.get("memory_events", {}).get("oom_kill", 0):
            result.update(
                category="resource_limit", detail="whole_tree_memory_limit", test_executions=0
            )
    except BaseException as exc:
        result = {
            "category": "sandbox_infrastructure_error",
            "detail": f"sandbox_exception:{type(exc).__name__}",
            "test_executions": 0,
        }
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        cgroup_released = _release_cgroup(cgroup)
        shutil.rmtree(staging, ignore_errors=True)
    isolation = result.get("isolation")
    if isinstance(isolation, dict) and not all(isolation.values()):
        result.update(
            category="sandbox_infrastructure_error",
            detail="isolation_contract_failed",
            test_executions=0,
        )
    result.update(
        task_id=task_id,
        sandbox_backend="namespace_chroot_cgroup_v2",
        resource_limits=dict(LBPP_LIMITS),
        resource_observation=resource_values,
        sandbox_released=not staging.exists(),
        cgroup_released=cgroup_released,
        wall_seconds=time.perf_counter() - started,
    )
    return result


def classify_asset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    used_ids: Iterable[str],
    overlap_ids: Iterable[str],
    history_unknown_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Give every raw row exactly one predeclared D1 disposition."""

    used, overlap, unknown = set(used_ids), set(overlap_ids), set(history_unknown_ids)
    counts = Counter(row.get("task_id") for row in rows)
    classified: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        task_id = row.get("task_id")
        if task_id == "lbpp/python/042":
            disposition = "canary"
        elif not isinstance(task_id, str) or counts[task_id] != 1:
            disposition = "duplicate"
        elif task_id in used:
            disposition = "used"
        elif task_id in overlap:
            disposition = "overlap"
        elif task_id in unknown:
            disposition = "history_unknown"
        elif (
            not _REQUIRED_ROW_FIELDS.issubset(row)
            or row.get("language") != "python"
            or any(not isinstance(row.get(field), str) for field in _REQUIRED_ROW_FIELDS)
        ):
            disposition = "static_unsupported"
        else:
            disposition = "static_candidate"
        classified.append(
            {"raw_index": index, "task_id": task_id, "disposition": disposition}
        )
    return classified


def summarize_static_dispositions(
    classified: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(item.get("disposition") for item in classified)
    unknown = set(counts) - set(_DISPOSITION_TO_COUNT)
    if unknown:
        raise ValueError(f"unknown LBPP disposition(s): {sorted(unknown)!r}")
    summary = {"N_raw": len(classified)}
    for disposition, label in _DISPOSITION_TO_COUNT.items():
        summary[label] = counts[disposition]
    if sum(summary[label] for label in _DISPOSITION_TO_COUNT.values()) != len(classified):
        raise ValueError("LBPP D1 disposition ledger does not conserve raw rows")
    return summary


def resolve_lbpp_terminal(
    *,
    d1: Mapping[str, int],
    d2: Mapping[str, int],
    d4: Mapping[str, int],
    behavior_passed: bool,
    resource_limit_reached: bool = False,
) -> str:
    d1_parts = (
        "N_canary", "N_duplicate", "N_used", "N_overlap",
        "N_history_unknown", "N_static_unsupported", "N_static",
    )
    if sum(int(d1.get(key, -1)) for key in d1_parts) != int(d1.get("N_raw", -2)):
        raise ValueError("LBPP D1 ledger does not conserve")
    if int(d2.get("N_contract_ready", -1)) + int(
        d2.get("N_semantic_unresolved", -1)
    ) != int(d1.get("N_static", -2)):
        raise ValueError("LBPP D2 ledger does not conserve")
    d4_parts = (
        "N_reference_qualified", "N_reference_not_qualified",
        "N_infra_unresolved", "N_resource_stability_unresolved", "N_not_run",
    )
    if sum(int(d4.get(key, -1)) for key in d4_parts) != int(
        d2.get("N_contract_ready", -2)
    ):
        raise ValueError("LBPP D4 ledger does not conserve")
    if resource_limit_reached:
        return "LBPP_RESOURCE_LIMIT"
    unresolved = (
        int(d1.get("N_history_unknown", 0))
        + int(d2.get("N_semantic_unresolved", 0))
        + int(d4.get("N_infra_unresolved", 0))
        + int(d4.get("N_resource_stability_unresolved", 0))
        + int(d4.get("N_not_run", 0))
    )
    if unresolved or not behavior_passed:
        return "LBPP_QUALIFICATION_UNRESOLVED"
    if int(d4.get("N_reference_qualified", 0)) < 80:
        return "LBPP_DESIGN_NOT_QUALIFIED"
    return "LBPP_CPU_QUALIFIED_FOR_REVIEW"


def save_qualification_receipt(
    path: Path, *, identity: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    path = Path(path)
    payload = {
        "schema": "lbpp-qualification-receipt-v1",
        "identity": dict(identity),
        "result": dict(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def load_qualification_receipt(
    path: Path, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "lbpp-qualification-receipt-v1":
        raise ValueError("LBPP receipt schema is invalid")
    if value.get("identity") != dict(expected_identity):
        raise ValueError("LBPP receipt identity differs")
    if not isinstance(value.get("result"), dict):
        raise ValueError("LBPP receipt result is invalid")
    return value


def _inside_main(mode: str) -> int:
    try:
        request = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
        if mode != "judge":
            raise ValueError("unknown LBPP inside mode")
        result = _inside_judge(request)
    except BaseException as exc:
        result = {
            "category": "sandbox_infrastructure_error",
            "detail": f"trusted_harness_exception:{type(exc).__name__}",
            "test_executions": 0,
        }
    sys.stdout.buffer.write(_canonical_bytes(result))
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-mode", choices=("judge",), required=True)
    arguments = parser.parse_args()
    return _inside_main(arguments.inside_mode)


if __name__ == "__main__":
    raise SystemExit(_cli())
