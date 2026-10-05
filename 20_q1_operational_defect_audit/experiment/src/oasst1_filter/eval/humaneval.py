"""Pinned HumanEval+ loading, assembly, and qualification scoring seams."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import errno
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence


HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    "v0.1.10/HumanEvalPlus.jsonl.gz"
)
HUMANEVAL_PLUS_TAG = "v0.1.10"
HUMANEVAL_PLUS_SHA256 = "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
HUMANEVAL_PLUS_LOGICAL_SHA256 = (
    "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
)
EVALPLUS_REVISION = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
EVALPLUS_SOURCE_ARCHIVE_SHA256 = (
    "d11dc64dfee6858a52c493466aa91a7bd0e3d50deda9a586144c1ba127c0e869"
)


_HUMANEVAL_ISOLATION_SCRIPT = r'''
set -eu
ROOTFS="$1"
PAYLOAD="$2"
mount --make-rprivate /
mount -t tmpfs -o size=512m,mode=0755,nosuid,nodev tmpfs "$ROOTFS"
mkdir -p "$ROOTFS/usr" "$ROOTFS/proc" "$ROOTFS/dev" "$ROOTFS/tmp" "$ROOTFS/payload" "$ROOTFS/etc"
ln -s usr/bin "$ROOTFS/bin"
ln -s usr/sbin "$ROOTFS/sbin"
ln -s usr/lib "$ROOTFS/lib"
ln -s usr/lib64 "$ROOTFS/lib64"
printf 'nobody:x:65534:65534:nobody:/tmp:/bin/false\n' > "$ROOTFS/etc/passwd"
printf 'nogroup:x:65534:\n' > "$ROOTFS/etc/group"
mount --bind /usr "$ROOTFS/usr"
mount -o remount,bind,ro "$ROOTFS/usr"
mount --bind "$PAYLOAD" "$ROOTFS/payload"
mount -o remount,bind,ro "$ROOTFS/payload"
mount -t proc -o nosuid,nodev,noexec,hidepid=2 proc "$ROOTFS/proc"
mount -t tmpfs -o size=4m,mode=0755,nosuid,noexec tmpfs "$ROOTFS/dev"
mknod -m 666 "$ROOTFS/dev/null" c 1 3
mknod -m 666 "$ROOTFS/dev/zero" c 1 5
mknod -m 444 "$ROOTFS/dev/urandom" c 1 9
ln -s /proc/self/fd "$ROOTFS/dev/fd"
ln -s /proc/self/fd/0 "$ROOTFS/dev/stdin"
ln -s /proc/self/fd/1 "$ROOTFS/dev/stdout"
ln -s /proc/self/fd/2 "$ROOTFS/dev/stderr"
mount -o remount,ro "$ROOTFS/dev"
mount -t tmpfs -o size=256m,mode=1777,nosuid,nodev tmpfs "$ROOTFS/tmp"
mount -o remount,ro "$ROOTFS"
exec chroot "$ROOTFS" /usr/bin/setpriv \
  --reuid 65534 --regid 65534 --clear-groups \
  --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
  /usr/bin/prlimit --as=4294967296 --nproc=64 --nofile=128 --fsize=1048576 --core=0 -- \
  /usr/bin/env -i PATH=/usr/bin:/bin PYTHONPATH=/payload HOME=/tmp TMPDIR=/tmp \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES= EVALPLUS_MAX_MEMORY_BYTES=4294967296 \
  /usr/bin/python3 -m oasst1_filter.eval.humaneval --inside-score-stdio
'''


@dataclass(frozen=True)
class HumanEvalAssetAuthority:
    url: str
    tag: str
    compressed_sha256: str
    logical_sha256: str


OFFICIAL_HUMANEVAL_AUTHORITY = HumanEvalAssetAuthority(
    url=HUMANEVAL_PLUS_URL,
    tag=HUMANEVAL_PLUS_TAG,
    compressed_sha256=HUMANEVAL_PLUS_SHA256,
    logical_sha256=HUMANEVAL_PLUS_LOGICAL_SHA256,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _normalize_lf(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def assemble_humaneval_program(prompt: str, suffix: str) -> str:
    """Join an official base prompt and completion without changing code semantics."""

    if not isinstance(prompt, str) or not isinstance(suffix, str):
        raise TypeError("HumanEval prompt and suffix must be strings")
    return _normalize_lf(prompt) + _normalize_lf(suffix)


def load_humaneval_plus_asset(
    path: Path, *, authority: HumanEvalAssetAuthority = OFFICIAL_HUMANEVAL_AUTHORITY
) -> tuple[Mapping[str, Any], ...]:
    """Load the pinned complete release after validating both stored and logical bytes."""

    path = Path(path)
    compressed = path.read_bytes()
    if _sha256_bytes(compressed) != authority.compressed_sha256:
        raise ValueError("HumanEval+ compressed asset identity differs from authority")
    with gzip.open(path, "rb") as handle:
        logical = handle.read()
    if _sha256_bytes(logical) != authority.logical_sha256:
        raise ValueError("HumanEval+ logical line identity differs from authority")

    raw_lines = logical.splitlines()
    if len(raw_lines) != 164:
        raise ValueError("HumanEval+ must contain exactly 164 logical records")
    records = [json.loads(line) for line in raw_lines]
    expected_ids = [f"HumanEval/{index}" for index in range(164)]
    if [row.get("task_id") for row in records] != expected_ids:
        raise ValueError("HumanEval+ task identities must be exactly HumanEval/0..163 in order")

    string_fields = ("task_id", "prompt", "canonical_solution", "entry_point", "contract", "test")
    for row in records:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in string_fields):
            raise ValueError("HumanEval+ required string schema is invalid")
        if not isinstance(row.get("base_input"), list) or not isinstance(row.get("plus_input"), list):
            raise ValueError("HumanEval+ base/plus input schema is invalid")
        if not isinstance(row.get("atol"), (int, float)) or isinstance(row.get("atol"), bool):
            raise ValueError("HumanEval+ tolerance schema is invalid")
    return tuple(_deep_freeze(row) for row in records)


def _evaluator_identity_error(source_root: Path, archive: Path) -> str | None:
    if not archive.is_file() or _sha256_bytes(archive.read_bytes()) != EVALPLUS_SOURCE_ARCHIVE_SHA256:
        return "EvalPlus source archive identity differs from freeze"
    humaneval_source = source_root / "evalplus/data/humaneval.py"
    if not humaneval_source.is_file():
        return "EvalPlus source tree is missing HumanEval support"
    text = humaneval_source.read_text("utf-8")
    if f'HUMANEVAL_PLUS_VERSION = "{HUMANEVAL_PLUS_TAG}"' not in text:
        return "frozen EvalPlus source does not support HumanEval+ v0.1.10"
    return None


def score_humaneval_program(
    program: str,
    *,
    task: Mapping[str, Any],
    evaluator_source_root: Path,
    evaluator_archive: Path,
    sandbox_base: Path,
    python_executable: Path = Path(sys.executable),
    outer_timeout_seconds: float = 180.0,
    require_isolation: bool = False,
) -> dict[str, Any]:
    """Score one complete program through the frozen evaluator in a private process tree."""

    if not isinstance(program, str):
        raise TypeError("HumanEval program must be a string")
    sandbox_base = Path(sandbox_base)
    sandbox_base.mkdir(parents=True, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="humaneval_", dir=sandbox_base))
    sandbox_id = sandbox.name
    started = time.perf_counter()
    result: dict[str, Any]
    try:
        identity_error = _evaluator_identity_error(
            Path(evaluator_source_root), Path(evaluator_archive)
        )
        if identity_error is not None:
            result = {"verdict": "infrastructure_error", "detail": identity_error}
        elif require_isolation:
            result = _score_humaneval_isolated_batch(
                requests=({"program": program, "task": task},),
                evaluator_source_root=Path(evaluator_source_root),
                sandbox=sandbox,
                outer_timeout_seconds=outer_timeout_seconds,
                workers=1,
            )[0]
        else:
            request_path = sandbox / "request.json"
            result_path = sandbox / "result.json"
            request_path.write_text(
                json.dumps(
                    {"program": program, "task": _json_ready(task)}, separators=(",", ":")
                ),
                "utf-8",
            )
            env = os.environ.copy()
            evaluator_root = str(Path(evaluator_source_root))
            package_source_root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (
                    evaluator_root,
                    package_source_root,
                    env.get("PYTHONPATH", ""),
                )
                if part
            )
            env["HOME"] = str(sandbox)
            env["TMPDIR"] = str(sandbox / "tmp")
            (sandbox / "tmp").mkdir()
            command = [
                str(python_executable),
                "-m",
                "oasst1_filter.eval.humaneval",
                "--inside-score",
                str(request_path),
                str(result_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=sandbox,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=float(outer_timeout_seconds),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result = {
                    "verdict": "infrastructure_error",
                    "detail": "outer_evaluator_timeout",
                }
            else:
                if completed.returncode != 0 or not result_path.is_file():
                    result = {
                        "verdict": "infrastructure_error",
                        "detail": "frozen_evaluator_process_failed",
                        "returncode": completed.returncode,
                        "stderr_sha256": _sha256_bytes(completed.stderr),
                        "stderr_tail": completed.stderr.decode(
                            "utf-8", errors="replace"
                        )[-2048:],
                    }
                else:
                    try:
                        result = json.loads(result_path.read_text("utf-8"))
                    except (OSError, ValueError):
                        result = {
                            "verdict": "infrastructure_error",
                            "detail": "invalid_frozen_evaluator_result",
                        }
        result.update(
            task_id=task.get("task_id"),
            evaluator_revision=EVALPLUS_REVISION,
            sandbox_id=sandbox_id,
            wall_seconds=time.perf_counter() - started,
        )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
    result["sandbox_released"] = not sandbox.exists()
    return result


def _score_humaneval_isolated_batch(
    *, requests: Sequence[Mapping[str, Any]], evaluator_source_root: Path,
    sandbox: Path, outer_timeout_seconds: float, workers: int,
) -> list[dict[str, Any]]:
    """Run one private chroot/seccomp batch through the trusted frozen evaluator."""

    if os.name != "posix" or workers not in (1, 2, 4) or not requests:
        return [{"verdict": "infrastructure_error", "detail": "required_chroot_isolation_unavailable"} for _ in requests]
    sandbox = Path(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    if workers > 1:
        buckets: list[list[tuple[int, Mapping[str, Any]]]] = [[] for _ in range(workers)]
        for index, request_row in enumerate(requests):
            buckets[index % workers].append((index, request_row))

        def run_slot(item: tuple[int, list[tuple[int, Mapping[str, Any]]]]):
            slot, values = item
            return values, _score_humaneval_isolated_batch(
                requests=[value for _index, value in values],
                evaluator_source_root=evaluator_source_root,
                sandbox=sandbox / f"slot_{slot}",
                outer_timeout_seconds=outer_timeout_seconds,
                workers=1,
            )

        merged: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for values, outcomes in pool.map(run_slot, enumerate(buckets)):
                for (index, _value), outcome in zip(values, outcomes, strict=True):
                    merged[index] = outcome
        return [merged[index] for index in range(len(requests))]
    request = json.dumps(
        {"requests": [
            {"program": str(item["program"]), "task": _json_ready(item["task"])}
            for item in requests
        ], "workers": workers}, separators=(",", ":")
    ).encode("utf-8")
    import importlib.util
    dependency_paths = []
    for package in ("numpy", "psutil"):
        spec = importlib.util.find_spec(package)
        if spec is None or spec.submodule_search_locations is None:
            return [{"verdict": "infrastructure_error", "detail": f"chroot_dependency_unavailable:{package}"} for _ in requests]
        dependency_paths.append(str(Path(next(iter(spec.submodule_search_locations)))))
    command = [
        "/usr/bin/sudo", "-n", "/usr/bin/python3", str(Path(__file__).resolve()),
        "--root-chroot-batch-stdio", str(sandbox), str(evaluator_source_root),
        *dependency_paths,
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
    )
    try:
        stdout, stderr = process.communicate(request, timeout=float(outer_timeout_seconds))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return [{"verdict": "infrastructure_error", "detail": "outer_chroot_evaluator_timeout"} for _ in requests]
    if process.returncode != 0:
        detail = {
            "verdict": "infrastructure_error",
            "detail": "chroot_seccomp_evaluator_failed",
            "returncode": process.returncode,
            "stderr_sha256": _sha256_bytes(stderr),
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-2048:],
        }
        return [dict(detail) for _ in requests]
    try:
        values = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        values = None
    if not isinstance(values, list) or len(values) != len(requests) or any(
        not isinstance(value, dict)
        or value.get("verdict") not in {"pass", "fail", "timeout", "infrastructure_error"}
        for value in values or ()
    ):
        return [{"verdict": "infrastructure_error", "detail": "invalid_chroot_evaluator_result"} for _ in requests]
    for value in values:
        value["isolation"] = {
            "backend": "chroot_seccomp", "network": "seccomp_denied",
            "workspace_mounted": False, "model_mounted": False,
            "credentials_mounted": False, "address_space_bytes": 4_294_967_296,
            "process_limit": 64,
        }
    return values


def _score_humaneval_in_namespace(
    *, program: str, task: Mapping[str, Any], evaluator_source_root: Path,
    sandbox: Path, outer_timeout_seconds: float,
) -> dict[str, Any]:
    """Run the frozen evaluator without mounting project, model, network, or credentials."""

    if os.name != "posix" or not Path("/usr/bin/unshare").is_file():
        return {"verdict": "infrastructure_error", "detail": "required_namespace_isolation_unavailable"}
    payload = sandbox / "payload"
    module_dir = payload / "oasst1_filter/eval"
    module_dir.mkdir(parents=True)
    (payload / "oasst1_filter/__init__.py").write_text("", "utf-8")
    (module_dir / "__init__.py").write_text("", "utf-8")
    shutil.copy2(Path(__file__).resolve(), module_dir / "humaneval.py")
    source_package = evaluator_source_root / "evalplus"
    if not source_package.is_dir():
        return {"verdict": "infrastructure_error", "detail": "isolated_evalplus_source_missing"}
    shutil.copytree(source_package, payload / "evalplus")
    for path in payload.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    rootfs = sandbox / "rootfs"
    rootfs.mkdir(mode=0o755)
    privilege_prefix = [] if os.geteuid() == 0 else ["/usr/bin/sudo", "-n"]
    command = privilege_prefix + [
        "/usr/bin/unshare", "--mount", "--uts", "--ipc", "--net", "--pid",
        "--fork", "--kill-child=SIGKILL", "--propagation", "private",
        "/bin/bash", "-c", _HUMANEVAL_ISOLATION_SCRIPT, "sandbox",
        str(rootfs), str(payload),
    ]
    request = json.dumps(
        {"program": program, "task": _json_ready(task)}, separators=(",", ":")
    ).encode("utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C.UTF-8"},
    )
    try:
        stdout, stderr = process.communicate(request, timeout=float(outer_timeout_seconds))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {"verdict": "infrastructure_error", "detail": "outer_isolated_evaluator_timeout"}
    if process.returncode != 0:
        return {
            "verdict": "infrastructure_error",
            "detail": "namespace_isolated_evaluator_failed",
            "returncode": process.returncode,
            "stderr_sha256": _sha256_bytes(stderr),
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-2048:],
        }
    try:
        result = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {"verdict": "infrastructure_error", "detail": "invalid_isolated_evaluator_result"}
    if not isinstance(result, dict) or result.get("verdict") not in {
        "pass", "fail", "timeout", "infrastructure_error"
    }:
        return {"verdict": "infrastructure_error", "detail": "malformed_isolated_evaluator_result"}
    result["isolation"] = {
        "backend": "namespace_chroot",
        "network_namespace": "private",
        "workspace_mounted": False,
        "model_mounted": False,
        "credentials_mounted": False,
        "address_space_bytes": 4_294_967_296,
        "process_limit": 64,
    }
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def execute_humaneval_canonical_qualification(
    *,
    tasks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    workers: int,
    output_dir: Path,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    sandbox_base: Path,
    python_executable: Path = Path(sys.executable),
    development: bool = False,
) -> dict[str, Any]:
    """Run one recoverable canonical qualification rung at C1 or C4."""

    if workers not in (1, 4) and not (development and workers in (1, 2, 3, 4)):
        raise ValueError("canonical qualification workers must be C1 or C4")
    ordered = list(tasks)
    task_ids = [task.get("task_id") for task in ordered]
    if len(set(task_ids)) != len(task_ids) or (
        not development and task_ids != [f"HumanEval/{index}" for index in range(164)]
    ) or (development and not 1 <= len(ordered) <= 8):
        raise ValueError("canonical qualification task population is invalid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, Mapping[str, Any], str, Path]] = []
    receipts: dict[int, dict[str, Any]] = {}
    for index, task in enumerate(ordered):
        task_bytes = json.dumps(
            _json_ready(task), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        task_sha256 = _sha256_bytes(task_bytes)
        path = output_dir / f"{index:03d}.json"
        if path.is_file():
            value = json.loads(path.read_text("utf-8"))
            if (
                value.get("task_id") != task["task_id"]
                or value.get("task_sha256") != task_sha256
                or value.get("evaluator_revision") != EVALPLUS_REVISION
                or value.get("verdict") not in {"pass", "fail", "timeout", "infrastructure_error"}
            ):
                raise ValueError("canonical qualification receipt identity differs")
            receipts[index] = value
        else:
            pending.append((index, task, task_sha256, path))

    def score(item: tuple[int, Mapping[str, Any], str, Path]) -> tuple[int, dict[str, Any], Path]:
        index, task, task_sha256, path = item
        program = assemble_humaneval_program(
            str(task["prompt"]), str(task["canonical_solution"])
        )
        scored = score_humaneval_program(
            program,
            task=task,
            evaluator_source_root=evaluator_source_root,
            evaluator_archive=evaluator_archive,
            sandbox_base=sandbox_base,
            python_executable=python_executable,
        )
        receipt = {
            "schema": "humaneval-canonical-qualification-receipt-v1",
            "task_id": task["task_id"],
            "task_sha256": task_sha256,
            "evaluator_revision": EVALPLUS_REVISION,
            "verdict": scored["verdict"],
            "base_status": scored.get("base_status"),
            "plus_status": scored.get("plus_status"),
            "wall_seconds": scored["wall_seconds"],
            "sandbox_released": scored["sandbox_released"],
        }
        return index, receipt, path

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score, item) for item in pending]
            for future in as_completed(futures):
                index, receipt, path = future.result()
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")), "utf-8"
                )
                temporary.replace(path)
                receipts[index] = receipt
    ordered_receipts = [receipts[index] for index in range(len(ordered))]
    counts = {
        verdict: sum(row["verdict"] == verdict for row in ordered_receipts)
        for verdict in ("pass", "fail", "timeout", "infrastructure_error")
    }
    return {
        "schema": "humaneval-canonical-qualification-rung-v1",
        "workers": workers,
        "receipt_count": len(ordered_receipts),
        "new_positions": len(pending),
        "counts": counts,
        "task_verdicts": [
            {"task_id": row["task_id"], "verdict": row["verdict"]}
            for row in ordered_receipts
        ],
        "wall_seconds": sum(float(row["wall_seconds"]) for row in ordered_receipts),
        "all_sandboxes_released": all(row["sandbox_released"] for row in ordered_receipts),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "utf-8",
    )
    temporary.replace(path)


def _formal_task_sha256(task: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            _json_ready(task), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )


def _load_formal_generation_position(
    path: Path,
    *,
    state_id: str,
    task: Mapping[str, Any],
    task_index: int,
    design_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text("utf-8"))
    completion = value.get("completion")
    if not (
        value.get("schema") == "nat-replica-x164-generation-position-v1"
        and value.get("state_id") == state_id
        and value.get("task_id") == task.get("task_id")
        and value.get("task_index") == task_index
        and value.get("task_sha256") == _formal_task_sha256(task)
        and value.get("design_sha256") == design_sha256
        and isinstance(completion, str)
        and value.get("completion_sha256")
        == _sha256_bytes(completion.encode("utf-8"))
        and type(value.get("generated_tokens")) is int
        and 1 <= value["generated_tokens"] <= 1024
        and value.get("normal_termination") is True
    ):
        raise ValueError("HumanEval formal generation position identity differs")
    return value


def execute_humaneval_formal_generation(
    *,
    state_id: str,
    tasks: Sequence[Mapping[str, Any]],
    model_path: Path,
    adapter_path: Path | None,
    output_dir: Path,
    design_sha256: str,
    development: bool = False,
) -> dict[str, Any]:
    """Generate one BASE or adapted HumanEval+ state with position recovery."""

    ordered = list(tasks)
    task_ids = [task.get("task_id") for task in ordered]
    if (
        len(set(task_ids)) != len(task_ids)
        or (
            not development
            and task_ids != [f"HumanEval/{index}" for index in range(164)]
        )
        or (development and not 1 <= len(ordered) <= 8)
    ):
        raise ValueError("HumanEval formal generation task population is invalid")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("HumanEval formal generation state is invalid")
    if len(design_sha256) != 64:
        raise ValueError("HumanEval formal design identity is invalid")
    output_dir = Path(output_dir)
    pending: list[tuple[int, Mapping[str, Any], Path]] = []
    positions: dict[int, dict[str, Any]] = {}
    for index, task in enumerate(ordered):
        path = output_dir / f"{index:03d}.json"
        restored = _load_formal_generation_position(
            path,
            state_id=state_id,
            task=task,
            task_index=index,
            design_sha256=design_sha256,
        )
        if restored is None:
            pending.append((index, task, path))
        else:
            positions[index] = restored
    load_seconds = 0.0
    peak_reserved_bytes = None
    if pending:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("NAT-REPLICA deterministic CUDA environment missing")
        import torch
        from peft import PeftModel
        from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

        from .codecontests import CODECONTESTS_STOP_STRINGS, extract_codecontests_completion
        from .run_codecontests import FORMAL_GENERATION_SEED, codecontests_generation_kwargs
        from ..train.modeling import configure_determinism

        configure_determinism(FORMAL_GENERATION_SEED)
        load_started = time.perf_counter()
        tokenizer = Qwen2Tokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
            raise ValueError("NAT-REPLICA tokenizer differs from the frozen BASE")
        model = Qwen2ForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda:0")
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model, str(adapter_path), is_trainable=False
            )
            if any(
                parameter.dtype != torch.float32
                for name, parameter in model.named_parameters()
                if "lora_" in name
            ):
                raise ValueError("NAT-REPLICA LoRA parameters must remain FP32")
        model.eval()
        load_seconds = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats()
        try:
            for index, task, path in pending:
                prompt = str(task["prompt"])
                encoded = tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                    return_tensors="pt",
                )
                prompt_tokens = int(encoded["input_ids"].shape[1])
                encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                started = time.perf_counter()
                with torch.inference_mode():
                    sequence = model.generate(
                        **encoded,
                        **codecontests_generation_kwargs(
                            pad_token_id=151643, eos_token_id=151643
                        ),
                        stop_strings=list(CODECONTESTS_STOP_STRINGS),
                        tokenizer=tokenizer,
                    ).detach().cpu().tolist()[0]
                suffix_tokens = sequence[prompt_tokens:]
                if not suffix_tokens or len(suffix_tokens) > 1024:
                    raise RuntimeError("HumanEval formal generation did not terminate normally")
                completion = extract_codecontests_completion(
                    sequence, prompt_tokens, tokenizer
                )
                generated_tokens = (
                    suffix_tokens.index(151643) + 1
                    if 151643 in suffix_tokens
                    else len(suffix_tokens)
                )
                receipt = {
                    "schema": "nat-replica-x164-generation-position-v1",
                    "state_id": state_id,
                    "task_id": task["task_id"],
                    "task_index": index,
                    "task_sha256": _formal_task_sha256(task),
                    "design_sha256": design_sha256,
                    "completion": completion,
                    "completion_sha256": _sha256_bytes(completion.encode("utf-8")),
                    "generated_tokens": generated_tokens,
                    "decode_seconds": time.perf_counter() - started,
                    "normal_termination": True,
                }
                _atomic_json(path, receipt)
                positions[index] = receipt
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "schema": "nat-replica-x164-generation-state-v1",
        "state_id": state_id,
        "completed_positions": len(positions),
        "new_positions": len(pending),
        "load_seconds": load_seconds,
        "decode_seconds": sum(
            float(row.get("decode_seconds", 0.0)) for row in positions.values()
        ),
        "peak_reserved_bytes": peak_reserved_bytes,
        "design_sha256": design_sha256,
    }
    _atomic_json(output_dir / "COMPLETE.json", result)
    return result


def execute_humaneval_formal_scoring(
    *,
    state_id: str,
    tasks: Sequence[Mapping[str, Any]],
    generation_dir: Path,
    output_dir: Path,
    design_sha256: str,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    sandbox_base: Path,
    workers: int = 4,
    python_executable: Path = Path(sys.executable),
    development: bool = False,
    require_isolation: bool = False,
) -> dict[str, Any]:
    """Score one state with per-position recovery and explicit infra nulls."""

    ordered = list(tasks)
    if workers not in (1, 2, 4) or (
        not development
        and [task.get("task_id") for task in ordered]
        != [f"HumanEval/{index}" for index in range(164)]
    ) or (development and not 1 <= len(ordered) <= 8):
        raise ValueError("HumanEval formal scoring geometry is invalid")
    output_dir = Path(output_dir)
    pending: list[tuple[int, Mapping[str, Any], dict[str, Any], Path]] = []
    receipts: dict[int, dict[str, Any]] = {}
    for index, task in enumerate(ordered):
        generation = _load_formal_generation_position(
            Path(generation_dir) / f"{index:03d}.json",
            state_id=state_id,
            task=task,
            task_index=index,
            design_sha256=design_sha256,
        )
        if generation is None:
            raise FileNotFoundError("HumanEval formal score requires closed generation")
        path = output_dir / f"{index:03d}.json"
        if path.is_file():
            value = json.loads(path.read_text("utf-8"))
            if not (
                value.get("schema") == "nat-replica-x164-score-position-v1"
                and value.get("state_id") == state_id
                and value.get("task_id") == task.get("task_id")
                and value.get("task_index") == index
                and value.get("design_sha256") == design_sha256
                and value.get("completion_sha256")
                == generation["completion_sha256"]
                and value.get("verdict")
                in {"pass", "fail", "timeout", "infrastructure_error"}
                and value.get("Y")
                == (
                    1
                    if value["verdict"] == "pass"
                    else None
                    if value["verdict"] == "infrastructure_error"
                    else 0
                )
            ):
                raise ValueError("HumanEval formal score position identity differs")
            receipts[index] = value
        else:
            pending.append((index, task, generation, path))

    def score(
        item: tuple[int, Mapping[str, Any], dict[str, Any], Path]
    ) -> tuple[int, dict[str, Any], Path]:
        index, task, generation, path = item
        outcome = score_humaneval_program(
            assemble_humaneval_program(str(task["prompt"]), generation["completion"]),
            task=task,
            evaluator_source_root=evaluator_source_root,
            evaluator_archive=evaluator_archive,
            sandbox_base=sandbox_base,
            python_executable=python_executable,
            require_isolation=require_isolation,
        )
        verdict = outcome["verdict"]
        receipt = {
            "schema": "nat-replica-x164-score-position-v1",
            "state_id": state_id,
            "task_id": task["task_id"],
            "task_index": index,
            "design_sha256": design_sha256,
            "completion_sha256": generation["completion_sha256"],
            "verdict": verdict,
            "Y": 1 if verdict == "pass" else None if verdict == "infrastructure_error" else 0,
            "base_status": outcome.get("base_status"),
            "plus_status": outcome.get("plus_status"),
            "detail": outcome.get("detail"),
            "stderr_tail": outcome.get("stderr_tail"),
            "wall_seconds": outcome.get("wall_seconds"),
            "sandbox_released": outcome.get("sandbox_released") is True,
        }
        return index, receipt, path

    if pending and require_isolation:
        batch_root = Path(sandbox_base) / "private_slots"
        outcomes = _score_humaneval_isolated_batch(
            requests=[{
                "program": assemble_humaneval_program(
                    str(task["prompt"]), generation["completion"]
                ),
                "task": task,
            } for _index, task, generation, _path in pending],
            evaluator_source_root=Path(evaluator_source_root),
            sandbox=batch_root, outer_timeout_seconds=7_500.0,
            workers=workers,
        )
        shutil.rmtree(batch_root, ignore_errors=True)
        batch_released = not batch_root.exists()
        for (index, task, generation, path), outcome in zip(
            pending, outcomes, strict=True
        ):
            verdict = outcome["verdict"]
            receipt = {
                "schema": "nat-replica-x164-score-position-v1",
                "state_id": state_id, "task_id": task["task_id"],
                "task_index": index, "design_sha256": design_sha256,
                "completion_sha256": generation["completion_sha256"],
                "verdict": verdict,
                "Y": 1 if verdict == "pass" else None if verdict == "infrastructure_error" else 0,
                "base_status": outcome.get("base_status"),
                "plus_status": outcome.get("plus_status"),
                "detail": outcome.get("detail"),
                "stderr_tail": outcome.get("stderr_tail"),
                "wall_seconds": outcome.get("wall_seconds"),
                "sandbox_released": batch_released,
                "isolation": outcome.get("isolation"),
            }
            _atomic_json(path, receipt)
            receipts[index] = receipt
    elif pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score, item) for item in pending]
            for future in as_completed(futures):
                index, receipt, path = future.result()
                _atomic_json(path, receipt)
                receipts[index] = receipt
    ordered_receipts = [receipts[index] for index in range(len(ordered))]
    counts = {
        verdict: sum(row["verdict"] == verdict for row in ordered_receipts)
        for verdict in ("pass", "fail", "timeout", "infrastructure_error")
    }
    result = {
        "schema": "nat-replica-x164-score-state-v1",
        "state_id": state_id,
        "completed_positions": len(ordered_receipts),
        "new_positions": len(pending),
        "counts": counts,
        "valid_positions": len(ordered_receipts) - counts["infrastructure_error"],
        "all_sandboxes_released": all(
            row["sandbox_released"] for row in ordered_receipts
        ),
        "design_sha256": design_sha256,
    }
    _atomic_json(output_dir / "COMPLETE.json", result)
    return result


def plan_humaneval_probe_workers(
    task_ids: list[str] | tuple[str, ...], *, concurrency: int
) -> tuple[tuple[str, ...], ...]:
    """Partition the fixed first eight tasks into C4, C2, or C1 processes."""

    expected = [f"HumanEval/{index}" for index in range(8)]
    if list(task_ids) != expected or concurrency not in (1, 2, 4):
        raise ValueError("HumanEval probe geometry is not the fixed first-eight schedule")
    width = 8 // concurrency
    return tuple(
        tuple(expected[start : start + width]) for start in range(0, 8, width)
    )


def execute_humaneval_probe_worker(
    *,
    tasks: list[Mapping[str, Any]],
    model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load one independent BASE state and generate its fixed probe slice."""

    task_ids = [task.get("task_id") for task in tasks]
    if not task_ids or len(task_ids) not in (2, 4, 8):
        raise ValueError("probe worker task count is invalid")
    start = int(str(task_ids[0]).split("/")[-1])
    expected_slice = [f"HumanEval/{index}" for index in range(start, start + len(task_ids))]
    if task_ids != expected_slice or any(
        task_id not in {f"HumanEval/{index}" for index in range(8)} for task_id in task_ids
    ):
        raise ValueError("probe worker tasks are not a contiguous first-eight slice")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("NAT-REPLICA deterministic CUDA environment missing")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "COMPLETE.json"
    if complete_path.is_file():
        completed = json.loads(complete_path.read_text("utf-8"))
        if completed.get("task_ids") != task_ids or completed.get("completed_positions") != len(tasks):
            raise ValueError("probe worker recovery identity differs")
        return completed

    import torch
    from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

    from .codecontests import CODECONTESTS_STOP_STRINGS, extract_codecontests_completion
    from .run_codecontests import FORMAL_GENERATION_SEED, codecontests_generation_kwargs
    from ..train.modeling import configure_determinism

    configure_determinism(FORMAL_GENERATION_SEED)
    load_started = time.perf_counter()
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
        raise ValueError("NAT-REPLICA tokenizer EOS/pad differs from the frozen BASE")
    base = Qwen2ForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda:0")
    base.eval()
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    receipts: list[dict[str, Any]] = []
    decode_seconds = 0.0
    try:
        for index, task in enumerate(tasks):
            prompt = str(task["prompt"])
            encoded = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            prompt_tokens = int(encoded["input_ids"].shape[1])
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                sequence = base.generate(
                    **encoded,
                    **codecontests_generation_kwargs(
                        pad_token_id=151643, eos_token_id=151643
                    ),
                    stop_strings=list(CODECONTESTS_STOP_STRINGS),
                    tokenizer=tokenizer,
                ).detach().cpu().tolist()[0]
            suffix_tokens = sequence[prompt_tokens:]
            if not suffix_tokens or len(suffix_tokens) > 1024:
                raise RuntimeError("HumanEval probe generation suffix is incomplete")
            completion = extract_codecontests_completion(sequence, prompt_tokens, tokenizer)
            generated_tokens = (
                suffix_tokens.index(151643) + 1
                if 151643 in suffix_tokens
                else len(suffix_tokens)
            )
            elapsed = time.perf_counter() - started
            decode_seconds += elapsed
            private_receipt = {
                "schema": "nat-replica-x164-private-probe-completion-v1",
                "task_id": task["task_id"],
                "completion": completion,
                "completion_sha256": _sha256_bytes(completion.encode("utf-8")),
                "generated_tokens": generated_tokens,
                "decode_seconds": elapsed,
            }
            path = output_dir / f"{index:02d}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(private_receipt, sort_keys=True, separators=(",", ":")), "utf-8"
            )
            temporary.replace(path)
            receipts.append(
                {
                    "task_id": task["task_id"],
                    "completion_sha256": private_receipt["completion_sha256"],
                    "generated_tokens": generated_tokens,
                    "decode_seconds": elapsed,
                }
            )
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        del base
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "schema": "nat-replica-x164-probe-worker-v1",
        "task_ids": task_ids,
        "completed_positions": len(receipts),
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "generated_tokens": sum(int(row["generated_tokens"]) for row in receipts),
        "peak_reserved_bytes": peak_reserved,
        "completion_receipts": receipts,
    }
    temporary = complete_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")), "utf-8")
    temporary.replace(complete_path)
    return result


def _evaluate_score_request(request: Mapping[str, Any]) -> dict[str, Any]:
    task = request["task"]
    program = request["program"]
    from evalplus.eval import untrusted_check
    from evalplus.gen.util import trusted_exec

    reference = assemble_humaneval_program(task["prompt"], task["canonical_solution"])
    base_expected, base_times = trusted_exec(
        reference,
        task["base_input"],
        task["entry_point"],
        record_time=True,
        output_not_none=False,
    )
    plus_expected, plus_times = trusted_exec(
        reference,
        task["plus_input"],
        task["entry_point"],
        record_time=True,
        output_not_none=False,
    )
    base_status = untrusted_check(
        "humaneval",
        program,
        task["base_input"],
        task["entry_point"],
        expected=base_expected,
        atol=task["atol"],
        ref_time=base_times,
    )[0]
    plus_status = untrusted_check(
        "humaneval",
        program,
        task["plus_input"],
        task["entry_point"],
        expected=plus_expected,
        atol=task["atol"],
        ref_time=plus_times,
    )
    plus_status = plus_status[0]
    if base_status == plus_status == "pass":
        verdict = "pass"
    elif "timeout" in (base_status, plus_status):
        verdict = "timeout"
    else:
        verdict = "fail"
    return {"verdict": verdict, "base_status": base_status, "plus_status": plus_status}


def _install_humaneval_seccomp() -> None:
    """Deny networking and cross-process/namespace escape in the chroot."""

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = (ctypes.c_uint32,)
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
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
            b"connect", b"bind", b"listen", b"accept", b"accept4",
            b"sendto", b"sendmsg", b"sendmmsg", b"recvfrom", b"recvmsg",
            b"recvmmsg", b"shutdown", b"ptrace", b"process_vm_readv",
            b"process_vm_writev", b"pidfd_getfd", b"open_by_handle_at",
            b"mount", b"umount2", b"pivot_root", b"chroot", b"setns",
            b"unshare", b"bpf", b"perf_event_open", b"io_uring_setup",
            b"io_uring_enter", b"io_uring_register",
        ):
            syscall = seccomp.seccomp_syscall_resolve_name(name)
            if syscall >= 0 and seccomp.seccomp_rule_add(context, deny, syscall, 0) != 0:
                raise OSError(f"seccomp rule failed for {name.decode()}")
        if seccomp.seccomp_load(context) != 0:
            raise OSError(ctypes.get_errno(), "seccomp_load failed")
    finally:
        seccomp.seccomp_release(context)


def _inside_score_batch_stdio() -> int:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    rows = request.get("requests")
    workers = request.get("workers")
    if not isinstance(rows, list) or workers not in (1, 2, 4):
        return 2
    _install_humaneval_seccomp()
    values = []
    for row in rows:
        values.append(_evaluate_score_request(row))
        for path in Path("/tmp").iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    sys.stdout.write(json.dumps(values, separators=(",", ":")))
    return 0


def _root_chroot_batch_stdio(
    sandbox: Path, evaluator_source_root: Path,
    numpy_source: Path, psutil_source: Path,
) -> int:
    """Trusted root-only setup; untrusted code starts only after chroot+setuid."""

    if os.geteuid() != 0:
        return 3
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from oasst1_filter.eval.code_exec import (
        _build_chroot_runtime, _install_chroot_path, _ldd_dependencies,
    )
    rootfs = _build_chroot_runtime(Path(sandbox))
    private_shm = rootfs / "dev/shm"
    private_shm.mkdir(exist_ok=True)
    private_shm.chmod(0o1777)
    # EvalPlus consults psutil only to clamp its already-frozen 4 GiB limit.
    # Supply a synthetic, non-host /proc view so the child cannot enumerate
    # host processes while retaining the evaluator's memory-limit semantics.
    private_proc = rootfs / "proc"
    private_proc.mkdir(exist_ok=True)
    (private_proc / "meminfo").write_text(
        "MemTotal:        4194304 kB\n"
        "MemFree:         4194304 kB\n"
        "MemAvailable:    4194304 kB\n"
        "Buffers:               0 kB\n"
        "Cached:                0 kB\n"
        "SReclaimable:          0 kB\n"
        "Shmem:                 0 kB\n",
        "ascii",
    )
    (private_proc / "meminfo").chmod(0o444)
    private_proc.chmod(0o555)
    payload = rootfs / "payload"
    for existing in tuple(payload.iterdir()):
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()
    module_dir = payload / "oasst1_filter/eval"
    module_dir.mkdir(parents=True)
    (payload / "oasst1_filter/__init__.py").write_text("", "utf-8")
    (module_dir / "__init__.py").write_text("", "utf-8")
    shutil.copy2(Path(__file__).resolve(), module_dir / "humaneval.py")
    shutil.copytree(Path(evaluator_source_root) / "evalplus", payload / "evalplus")
    native_sources: list[Path] = []
    for package, source in (("numpy", Path(numpy_source)), ("psutil", Path(psutil_source))):
        if not source.is_dir() or source.name != package:
            raise RuntimeError(f"chroot dependency is unavailable: {package}")
        shutil.copytree(source, payload / package)
        native_sources.extend(source.rglob("*.so"))
        sibling = source.with_name(f"{package}.libs")
        if sibling.is_dir():
            shutil.copytree(sibling, payload / sibling.name)
            native_sources.extend(sibling.rglob("*.so*"))
    for dependency in _ldd_dependencies(tuple(native_sources)):
        # NumPy's bundled libraries are reported by ``ldd`` through paths such
        # as ``numpy/linalg/../../numpy.libs/libgfortran...``.  They already
        # live at the matching RPATH below /payload; copying that non-normalized
        # host path into the rootfs aliases a later canonical destination and
        # can raise SameFileError.  Only system dependencies need installation.
        resolved = dependency.resolve()
        resolved_destination = rootfs / resolved.relative_to("/")
        if not resolved_destination.exists():
            _install_chroot_path(resolved, rootfs)
        if ".." in dependency.parts or dependency == resolved:
            continue
        # The rootfs deliberately has independent /lib and /usr/lib trees.
        # Recreating a host symlink such as /lib/.../libstdc++.so.6 would leave
        # a broken relative target, so materialize its resolved file as an alias.
        destination = rootfs / dependency.relative_to("/")
        if not destination.exists() and not destination.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(resolved, destination)
            except OSError:
                shutil.copy2(resolved, destination)
    for path in payload.rglob("*"):
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    payload.chmod(0o555)

    request = sys.stdin.buffer.read()
    command = [
        "/usr/bin/python3", "-m", "oasst1_filter.eval.humaneval",
        "--inside-score-batch-stdio",
    ]
    def enter_chroot() -> None:
        os.chroot(rootfs)
        os.chdir("/tmp")
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        for kind, limit in (
            (resource.RLIMIT_AS, 4_294_967_296),
            (resource.RLIMIT_NPROC, 64), (resource.RLIMIT_NOFILE, 128),
            (resource.RLIMIT_FSIZE, 1_048_576), (resource.RLIMIT_CORE, 0),
        ):
            resource.setrlimit(kind, (limit, limit))

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
        preexec_fn=enter_chroot,
        env={
            "PATH": "/usr/bin:/bin", "PYTHONPATH": "/payload",
            "HOME": "/tmp", "TMPDIR": "/tmp", "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "EVALPLUS_MAX_MEMORY_BYTES": "4294967296",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        },
    )
    stdout, stderr = process.communicate(request)
    if process.returncode != 0:
        sys.stderr.buffer.write(stderr[-4096:])
        return process.returncode
    sys.stdout.buffer.write(stdout)
    return 0


def _inside_score(request_path: Path, result_path: Path) -> int:
    result_path.write_text(
        json.dumps(
            _evaluate_score_request(json.loads(request_path.read_text("utf-8"))),
            separators=(",", ":"),
        ), "utf-8"
    )
    return 0


def _inside_score_stdio() -> int:
    os.environ.pop("EVALPLUS_TIMEOUT_PER_TASK", None)
    os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = "4294967296"
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    sys.stdout.write(json.dumps(_evaluate_score_request(request), separators=(",", ":")))
    return 0


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--inside-score-stdio":
        return _inside_score_stdio()
    if len(argv) == 2 and argv[1] == "--inside-score-batch-stdio":
        return _inside_score_batch_stdio()
    if len(argv) == 6 and argv[1] == "--root-chroot-batch-stdio":
        return _root_chroot_batch_stdio(
            Path(argv[2]), Path(argv[3]), Path(argv[4]), Path(argv[5])
        )
    if len(argv) == 4 and argv[1] == "--inside-score":
        return _inside_score(Path(argv[2]), Path(argv[3]))
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
