"""Stable CodeContests V1 clean-exec scoring core.

Mechanically promoted from the qualified F0d evaluator. Historical endpoint
state, authorization, generation, and one-shot orchestration are intentionally
excluded; this module owns only deterministic MBPP+ normalization and isolated
execution.
"""
from __future__ import annotations
import argparse
import atexit
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import ctypes
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import random
try:
    import resource
except ImportError:
    resource = None
import shutil
import signal
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
import unicodedata
import unittest
PUBLIC_NAMESPACE = 'Q1_CODECONTESTS_F0A_EVALPLUS_MBPP_PLUS_V1'
PARTITION_MODULUS = 5
TASK_ID_PATTERN = re.compile('Mbpp/(?:0|[1-9][0-9]*)\\Z')
TASK_ID_BYTES_PATTERN = re.compile(b'"task_id"\\s*:\\s*"(Mbpp/(?:0|[1-9][0-9]*))"')
MBPP_OUTPUT_NOT_NONE_TASKS = ('check_str', 'text_match_three', 'text_starta_endb')
MBPP_OUTPUT_SET_EQ_TASKS = ('similar_elements', 'find_char_long', 'common_in_nested_lists', 'extract_singly', 'larg_nnum', 'intersection_array', 'find_dissimilar', 'Diff')
LIMITS: dict[str, int | float] = {'per_program_wall_seconds': 5.0, 'per_program_cpu_seconds': 8, 'per_solution_wall_seconds': 180.0, 'address_space_bytes': 1024 * 1024 * 1024, 'processes': 32, 'open_files': 64, 'file_bytes': 1024 * 1024, 'stdout_bytes': 1024 * 1024, 'stderr_bytes': 1024 * 1024, 'source_bytes': 512 * 1024, 'single_test_input_bytes': 2 * 1024 * 1024, 'single_expected_output_bytes': 2 * 1024 * 1024, 'solution_payload_bytes': 64 * 1024 * 1024, 'tmpfs_bytes': 64 * 1024 * 1024, 'sandbox_setup_grace_seconds': 20.0}
VERDICT_CATEGORIES = ('pass', 'wrong_answer', 'compile_failure', 'runtime_failure', 'timeout', 'sandbox_infrastructure_error')
DIRECT_EXEC_ENV = 'CODECONTESTS_EVAL_DIRECT_EXEC'
SANDBOX_BACKEND_ENV = 'CODECONTESTS_EVAL_SANDBOX_BACKEND'
SANDBOX_BACKENDS = ('namespace_chroot', 'chroot_seccomp')

def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload, encoding='utf-8', newline='\n')
    os.replace(temporary, path)

def normalize_lf(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')

def serialize_prompt(prompt: str) -> str:
    """Apply the frozen base-model direct-completion serialization."""
    if not isinstance(prompt, str):
        raise TypeError('prompt must be a string')
    return normalize_lf(prompt).strip() + '\n'

def prompt_fingerprint(prompt: str) -> str:
    normalized = unicodedata.normalize('NFC', normalize_lf(prompt)).strip()
    return _sha256_bytes(normalized.encode('utf-8'))

def partition_record(task_id: str) -> dict[str, Any]:
    """Return the prospective SHA-256 partition decision for one strict task ID."""
    if not isinstance(task_id, str) or TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValueError('malformed MBPP task_id')
    digest = _sha256_bytes(f'{PUBLIC_NAMESPACE}|{task_id}'.encode('utf-8'))
    return {'task_id': task_id, 'partition_sha256': digest, 'qualification': int(digest, 16) % PARTITION_MODULUS == 0}

def _task_sort_key(task_id: str) -> tuple[int, str]:
    return (int(task_id.split('/', 1)[1]), task_id)

def _task_id_from_raw_line(raw_line: bytes) -> str:
    matches = TASK_ID_BYTES_PATTERN.findall(raw_line)
    if len(matches) != 1:
        raise ValueError('row must contain exactly one strict task_id field')
    return matches[0].decode('ascii')

def scan_partition(path: Path, *, expected_rows: int | None=None, load_qualification: bool=True) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Scan every row by task ID only; JSON-decode qualification rows only.

    When ``load_qualification`` is false, no dataset row is JSON-decoded.  The
    production partition freeze uses that mode before the second-pass dry run.
    """
    manifest: list[dict[str, str]] = []
    qualification: list[dict[str, Any]] = []
    seen: set[str] = set()
    with gzip.open(path, 'rb') as handle:
        for raw_line in handle:
            task_id = _task_id_from_raw_line(raw_line)
            if task_id in seen:
                raise ValueError('duplicate task_id')
            seen.add(task_id)
            decision = partition_record(task_id)
            manifest.append({'task_id': task_id, 'partition_sha256': decision['partition_sha256']})
            if load_qualification and decision['qualification']:
                row = json.loads(raw_line)
                if not isinstance(row, dict) or row.get('task_id') != task_id:
                    raise ValueError('qualification row task_id mismatch')
                qualification.append(row)
    if expected_rows is not None and len(manifest) != expected_rows:
        raise ValueError('unexpected task universe size')
    manifest.sort(key=lambda row: _task_sort_key(row['task_id']))
    qualification.sort(key=lambda row: _task_sort_key(row['task_id']))
    return (manifest, qualification)

def deserialize_mbpp_inputs(task_id: str, inputs: list) -> list | tuple:
    number = int(task_id.split('/')[-1])
    if number in [2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429, 470, 560, 579, 596, 616, 630, 726, 740, 744, 809]:
        modified = [[tuple(item) for item in inp] for inp in inputs]
    elif number in [63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780]:
        modified = [[[tuple(item) for item in nested] for nested in inp] for inp in inputs]
    elif number in [75, 413, 444, 753]:
        modified = [[[tuple(item) for item in inp[0]]] + [inp[1]] for inp in inputs]
    elif number in [106, 750]:
        modified = [[inp[0]] + [tuple(inp[1])] for inp in inputs]
    elif number == 115:
        modified = [[[set(item) if isinstance(item, list) and len(item) else {} for item in inp[0]]] for inp in inputs]
    elif number == 124:
        modified = [(float(inp[0]), complex(inp[1])) for inp in inputs]
    elif number in [250, 405, 446, 617, 720, 763, 808]:
        modified = [[tuple(inp[0])] + [inp[1]] for inp in inputs]
    elif number in [259, 401, 445]:
        modified = [[[tuple(item) for item in nested] for nested in inp] for inp in inputs]
        modified = [[tuple(item) for item in inp] for inp in modified]
    elif number == 278:
        modified = [[[tuple(item) if isinstance(item, list) else item for item in inp[0]]] for inp in inputs]
        modified = [[tuple(item) for item in inp] for inp in modified]
    elif number == 307:
        modified = [[tuple(inp[0])] + [inp[1], inp[2]] for inp in inputs]
    elif number == 722:
        modified = [[{key: tuple(value) for (key, value) in inp[0].items()}] + inp[1:] for inp in inputs]
    elif number == 252:
        modified = [[complex(inp[0])] for inp in inputs]
    elif number in [580, 615, 791]:

        def turn_all_lists_into_tuples(value: Any) -> Any:
            if isinstance(value, list):
                return tuple((turn_all_lists_into_tuples(item) for item in value))
            return value
        modified = [turn_all_lists_into_tuples(inp) for inp in inputs]
    else:
        modified = inputs
    return modified

def _surface_area(base_edge: Any, height: Any) -> int:
    slant_height = math.sqrt((base_edge / 2) ** 2 + height ** 2)
    return round(base_edge ** 2 + 4 * (base_edge * slant_height) / 2)

def _digit_distance(num1: Any, num2: Any) -> int:
    (left, right) = (str(num1), str(num2))
    width = max(len(left), len(right))
    return sum((abs(int(a) - int(b)) for (a, b) in zip(left.zfill(width), right.zfill(width))))

def _is_floats(value: Any) -> bool:
    if isinstance(value, float):
        return True
    return isinstance(value, (list, tuple)) and bool(value) and all((isinstance(item, float) for item in value))

def _rectangular_numeric(value: Any) -> tuple[tuple[int, ...], list[Any]]:
    """Convert the supported np.allclose domain to shape plus row-major values."""
    if type(value) in (bool, int, float, complex):
        return ((), [value])
    if type(value) not in (list, tuple):
        raise TypeError('unsupported allclose value')
    children = [_rectangular_numeric(item) for item in value]
    if not children:
        return ((0,), [])
    child_shape = children[0][0]
    if any((shape != child_shape for (shape, _flat) in children[1:])):
        raise ValueError('ragged allclose value')
    return ((len(children),) + child_shape, [item for (_shape, flat) in children for item in flat])

def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    width = max(len(left), len(right))
    padded_left = (1,) * (width - len(left)) + left
    padded_right = (1,) * (width - len(right)) + right
    output = []
    for (left_dim, right_dim) in zip(padded_left, padded_right):
        if left_dim == right_dim:
            output.append(left_dim)
        elif left_dim == 1:
            output.append(right_dim)
        elif right_dim == 1:
            output.append(left_dim)
        else:
            raise ValueError('incompatible broadcast shapes')
    return tuple(output)

def _flat_index(shape: tuple[int, ...], output_index: tuple[int, ...]) -> int:
    if not shape:
        return 0
    aligned = output_index[-len(shape):]
    offset = 0
    for (dimension, index) in zip(shape, aligned):
        offset = offset * dimension + (0 if dimension == 1 else index)
    return offset

def _all_indices(shape: tuple[int, ...]) -> Sequence[tuple[int, ...]]:
    indices: list[tuple[int, ...]] = [()]
    for dimension in shape:
        indices = [prefix + (index,) for prefix in indices for index in range(dimension)]
    return indices

def _scalar_close(actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    try:
        if actual == expected:
            return True
        if isinstance(actual, complex) or isinstance(expected, complex):
            if any((math.isnan(component) or math.isinf(component) for component in (complex(actual).real, complex(actual).imag, complex(expected).real, complex(expected).imag))):
                return False
        else:
            if math.isnan(float(actual)) or math.isnan(float(expected)):
                return False
            if math.isinf(float(actual)) or math.isinf(float(expected)):
                return False
        return abs(actual - expected) <= atol + rtol * abs(expected)
    except Exception:
        return False

def _numpy_allclose_equivalent(actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    """Pure-Python np.allclose for numeric scalars/rectangular sequences.

    This covers Python bool/int/float/complex scalars and rectangular nested
    list/tuple values, including NumPy broadcasting, NaN, and infinity rules.
    Ragged, object, ndarray, and custom-number values fail closed; an unsupported
    canonical output is classified by the scorer as an environment/data block.
    """
    try:
        (actual_shape, actual_flat) = _rectangular_numeric(actual)
        (expected_shape, expected_flat) = _rectangular_numeric(expected)
        output_shape = _broadcast_shape(actual_shape, expected_shape)
        return all((_scalar_close(actual_flat[_flat_index(actual_shape, index)], expected_flat[_flat_index(expected_shape, index)], rtol=rtol, atol=atol) for index in _all_indices(output_shape)))
    except Exception:
        return False

class OfficialComparatorState:
    """Pinned EvalPlus comparator with split-local mutable ``atol`` state."""

    def __init__(self, atol: float):
        self.atol = atol

    def compare(self, entry_point: str, inp: Sequence[Any], actual: Any, expected: Any) -> bool:
        try:
            exact_match = bool(actual == expected)
        except Exception:
            exact_match = False
        try:
            if entry_point == 'are_equivalent':
                exact_match = True
            elif entry_point == 'sum_div':
                exact_match = exact_match or actual == 0
            elif entry_point == 'surface_Area':
                exact_match = exact_match or abs(actual - _surface_area(*inp)) <= self.atol
            elif entry_point == 'digit_distance_nums':
                exact_match = exact_match or actual == _digit_distance(*inp)
            elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                exact_match = set(actual) == set(expected)
            elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                if isinstance(actual, bool):
                    exact_match = actual == expected
                else:
                    exact_match = expected == (actual is not None)
        except Exception:
            return False
        if self.atol == 0 and _is_floats(expected):
            self.atol = 1e-06
        if not exact_match and self.atol != 0:
            if type(actual) is not type(expected):
                return False
            if isinstance(expected, (list, tuple)) and len(actual) != len(expected):
                return False
            return _numpy_allclose_equivalent(actual, expected, rtol=1e-07, atol=self.atol)
        return exact_match

def official_compare(entry_point: str, inp: Sequence[Any], actual: Any, expected: Any, atol: float) -> bool:
    """One-call convenience wrapper; split scoring uses persistent state."""
    return OfficialComparatorState(atol).compare(entry_point, inp, actual, expected)

def aggregate_task_results(outcomes: Sequence[Mapping[str, Any]], *, expected_base: int, expected_plus: int) -> dict[str, Any]:
    if expected_base < 1 or expected_plus < 1:
        raise ValueError('base and plus groups must both be nonempty')
    group_counts = {'base': Counter(), 'plus': Counter()}
    for outcome in outcomes:
        group = outcome.get('group')
        category = outcome.get('category')
        if group not in group_counts or category not in VERDICT_CATEGORIES:
            raise ValueError('malformed per-test outcome')
        group_counts[group][category] += 1
    expected_total = expected_base + expected_plus
    complete = sum(group_counts['base'].values()) == expected_base and sum(group_counts['plus'].values()) == expected_plus
    pass_count = sum((counter['pass'] for counter in group_counts.values()))
    all_pass = complete and pass_count == expected_total
    precedence = ('sandbox_infrastructure_error', 'timeout', 'compile_failure', 'runtime_failure', 'wrong_answer')
    overall = 'pass'
    for category in precedence:
        if any((counter[category] for counter in group_counts.values())):
            overall = category
            break
    return {'category': overall, 'all_tests_pass': all_pass, 'test_executions': len(outcomes), 'test_passes': pass_count, 'test_pass_fraction': pass_count / expected_total, 'group_counts': {group: dict(sorted(counter.items())) for (group, counter) in group_counts.items()}}
SANDBOX_SCRIPT = '\nset -eu\nROOTFS="$1"\nPAYLOAD="$2"\nMODE="$3"\nmount --make-rprivate /\nmount -t tmpfs -o size=16m,mode=0755,nosuid,nodev tmpfs "$ROOTFS"\nmkdir -p "$ROOTFS/usr" "$ROOTFS/etc" "$ROOTFS/proc" "$ROOTFS/dev" \\\n  "$ROOTFS/tmp" "$ROOTFS/payload" "$ROOTFS/home" "$ROOTFS/root" "$ROOTFS/mnt"\nln -s usr/bin "$ROOTFS/bin"\nln -s usr/sbin "$ROOTFS/sbin"\nln -s usr/lib "$ROOTFS/lib"\nln -s usr/lib64 "$ROOTFS/lib64"\nprintf \'root:x:0:0:root:/root:/bin/false\\nnobody:x:65534:65534:nobody:/tmp:/bin/false\\n\' > "$ROOTFS/etc/passwd"\nprintf \'root:x:0:\\nnogroup:x:65534:\\n\' > "$ROOTFS/etc/group"\nmount --bind /usr "$ROOTFS/usr"\nmount -o remount,bind,ro "$ROOTFS/usr"\nmount --bind "$PAYLOAD" "$ROOTFS/payload"\nmount -o remount,bind,ro "$ROOTFS/payload"\nmount -t proc -o nosuid,nodev,noexec,hidepid=2 proc "$ROOTFS/proc"\nmount -t tmpfs -o size=4m,mode=0755,nosuid,noexec tmpfs "$ROOTFS/dev"\nmknod -m 666 "$ROOTFS/dev/null" c 1 3\nmknod -m 666 "$ROOTFS/dev/zero" c 1 5\nmknod -m 444 "$ROOTFS/dev/urandom" c 1 9\nln -s /proc/self/fd "$ROOTFS/dev/fd"\nln -s /proc/self/fd/0 "$ROOTFS/dev/stdin"\nln -s /proc/self/fd/1 "$ROOTFS/dev/stdout"\nln -s /proc/self/fd/2 "$ROOTFS/dev/stderr"\nmount -o remount,ro "$ROOTFS/dev"\nmount -t tmpfs -o size=64m,mode=1777,nosuid,nodev tmpfs "$ROOTFS/tmp"\nmount -o remount,ro "$ROOTFS"\nexec chroot "$ROOTFS" /usr/bin/setpriv \\\n  --reuid 65534 --regid 65534 --clear-groups \\\n  --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \\\n  /usr/bin/prlimit --as=1073741824 --nproc=32 --nofile=64 \\\n  --fsize=1048576 --core=0 -- \\\n  /usr/bin/env -i PATH=/usr/bin:/bin HOME=/tmp TMPDIR=/tmp \\\n  LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \\\n  CUDA_VISIBLE_DEVICES= \\\n  /usr/bin/python3 /payload/evalplus_probe.py --inside-mode "$MODE"\n'

SANDBOX_SCRIPT = SANDBOX_SCRIPT.replace(
    "\nset -eu\n", "\nset -eu\numask 0022\n", 1
).replace(
    "mknod -m 444 \"$ROOTFS/dev/urandom\" c 1 9\n",
    "mknod -m 444 \"$ROOTFS/dev/urandom\" c 1 9\n"
    "chown 0:0 \"$ROOTFS/dev/null\" \"$ROOTFS/dev/zero\" \"$ROOTFS/dev/urandom\"\n"
    "chmod 0666 \"$ROOTFS/dev/null\" \"$ROOTFS/dev/zero\"\n"
    "chmod 0444 \"$ROOTFS/dev/urandom\"\n"
    "test ! -L \"$ROOTFS/dev/null\" && test ! -L \"$ROOTFS/dev/zero\" && test ! -L \"$ROOTFS/dev/urandom\"\n"
    "test \"$(stat -c '%F:%t:%T:%u:%g:%a' \"$ROOTFS/dev/null\")\" = \"character special file:1:3:0:0:666\"\n"
    "test \"$(stat -c '%F:%t:%T:%u:%g:%a' \"$ROOTFS/dev/zero\")\" = \"character special file:1:5:0:0:666\"\n"
    "test \"$(stat -c '%F:%t:%T:%u:%g:%a' \"$ROOTFS/dev/urandom\")\" = \"character special file:1:9:0:0:444\"\n",
    1,
)


def _staging_path(base: Path) -> Path:
    resolved = base.resolve()
    staging = resolved / '_task_staging'
    if staging.parent != resolved or staging.name != '_task_staging':
        raise RuntimeError('invalid staging target')
    return staging

def _sandbox_backend() -> str:
    backend = os.environ.get(SANDBOX_BACKEND_ENV, 'namespace_chroot')
    if backend not in SANDBOX_BACKENDS:
        raise RuntimeError('unsupported CodeContests sandbox backend')
    return backend

def _install_chroot_path(source: Path, rootfs: Path) -> None:
    destination = rootfs / source.relative_to('/')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    elif source.is_dir():
        destination.mkdir(exist_ok=True)
        destination.chmod(stat.S_IMODE(source.stat().st_mode))
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

def _install_chroot_tree(source: Path, rootfs: Path) -> None:
    _install_chroot_path(source, rootfs)
    for directory, names, files in os.walk(source, followlinks=False):
        base = Path(directory)
        for name in names:
            _install_chroot_path(base / name, rootfs)
        for name in files:
            _install_chroot_path(base / name, rootfs)

def _ldd_dependencies(paths: Sequence[Path]) -> tuple[Path, ...]:
    completed = subprocess.run(
        ['/usr/bin/ldd', *(str(path) for path in paths)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C'},
    )
    dependencies = set()
    for line in completed.stdout.splitlines():
        for rendered in re.findall(r'(?:=>\s+)?(/[^\s(]+)', line):
            candidate = Path(rendered)
            if candidate.is_file():
                dependencies.add(candidate)
    return tuple(sorted(dependencies))


_PRIVATE_DEVICE_ACCESS_CONTRACT = (
    ("null", 1, 3, 0o666),
    ("zero", 1, 5, 0o666),
    ("urandom", 1, 9, 0o444),
)


def _validate_private_device_access_contract(rootfs: Path) -> None:
    """Fail closed unless one unpublished/private rootfs has exact devices."""

    if rootfs.is_symlink():
        raise RuntimeError("private device access contract rejected redirected rootfs")
    try:
        root_info = os.lstat(rootfs)
        dev = rootfs / "dev"
        dev_info = os.lstat(dev)
    except OSError as exc:
        raise RuntimeError("private device access contract is incomplete") from exc
    for label, info in (("rootfs", root_info), ("dev", dev_info)):
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o755
        ):
            raise RuntimeError(
                f"private device access contract rejected {label} traversal"
            )
    if dev.is_symlink() or dev.resolve(strict=True).parent != rootfs.resolve(strict=True):
        raise RuntimeError("private device access contract rejected redirected dev")
    for name, major, minor, mode in _PRIVATE_DEVICE_ACCESS_CONTRACT:
        node = dev / name
        if node.is_symlink():
            raise RuntimeError(
                f"private device access contract rejected redirected {name}"
            )
        try:
            info = os.lstat(node)
        except OSError as exc:
            raise RuntimeError(
                f"private device access contract is missing {name}"
            ) from exc
        if (
            not stat.S_ISCHR(info.st_mode)
            or os.major(info.st_rdev) != major
            or os.minor(info.st_rdev) != minor
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != mode
        ):
            raise RuntimeError(
                f"private device access contract rejected invalid {name}"
            )

def _build_chroot_runtime(base: Path) -> Path:
    if os.geteuid() != 0:
        raise RuntimeError('chroot/seccomp sandbox requires root entry')
    rootfs = base.resolve() / '_chroot_runtime'
    if rootfs.parent != base.resolve() or rootfs.name != '_chroot_runtime':
        raise RuntimeError('invalid chroot runtime target')
    if rootfs.is_symlink():
        raise RuntimeError('private device access contract rejected redirected rootfs')
    if rootfs.is_dir():
        _validate_private_device_access_contract(rootfs)
        return rootfs
    if rootfs.exists():
        raise RuntimeError('private device access contract rejected invalid rootfs')
    temporary = rootfs.with_name(f'{rootfs.name}.tmp_{os.getpid()}')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, mode=0o755)
    temporary.chmod(0o755)
    os.chown(temporary, 0, 0)
    try:
        python_binary = Path('/usr/bin/python3').resolve()
        stdlib = Path('/usr/lib/python3.10')
        if not python_binary.is_file() or not stdlib.is_dir():
            raise RuntimeError('cloud sandbox Python runtime is unavailable')
        _install_chroot_path(python_binary, temporary)
        python_link = temporary / 'usr/bin/python3'
        python_link.parent.mkdir(parents=True, exist_ok=True)
        python_link.symlink_to(python_binary.name)
        _install_chroot_tree(stdlib, temporary)
        dynamic_modules = tuple((stdlib / 'lib-dynload').glob('*.so'))
        dependencies = set(_ldd_dependencies((python_binary, *dynamic_modules)))
        seccomp_library = Path('/lib/x86_64-linux-gnu/libseccomp.so.2')
        if not seccomp_library.is_file():
            raise RuntimeError('cloud sandbox libseccomp is unavailable')
        dependencies.add(seccomp_library)
        dependencies.update(_ldd_dependencies((seccomp_library,)))
        for dependency in sorted(dependencies):
            _install_chroot_path(dependency.resolve(), temporary)
            if dependency != dependency.resolve():
                destination = temporary / dependency.relative_to('/')
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(dependency.resolve(), destination)
                except OSError:
                    shutil.copy2(dependency.resolve(), destination)
        for source in (
            Path('/etc/ld.so.cache'),
            Path('/etc/localtime'),
        ):
            if source.is_file():
                _install_chroot_path(source, temporary)
        for directory, mode in (
            ('payload', 0o755),
            ('tmp', 0o1777),
            ('dev', 0o755),
        ):
            target = temporary / directory
            target.mkdir(exist_ok=True)
            target.chmod(mode)
        for name, major, minor, mode in _PRIVATE_DEVICE_ACCESS_CONTRACT:
            node = temporary / 'dev' / name
            os.mknod(
                node,
                stat.S_IFCHR | mode,
                os.makedev(major, minor),
            )
            os.chown(node, 0, 0)
            os.chmod(node, mode)
        _validate_private_device_access_contract(temporary)
        os.replace(temporary, rootfs)
        _validate_private_device_access_contract(rootfs)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    atexit.register(shutil.rmtree, rootfs, True)
    return rootfs

def _sync_chroot_payload(rootfs: Path, payload_dir: Path) -> None:
    destination = rootfs / 'payload'
    destination.chmod(0o755)
    for existing in destination.iterdir():
        if not existing.is_file():
            raise RuntimeError('cloud sandbox payload contains an invalid entry')
        existing.unlink()
    for source in payload_dir.iterdir():
        if not source.is_file():
            raise RuntimeError('cloud sandbox source payload is invalid')
        target = destination / source.name
        shutil.copy2(source, target)
        target.chmod(0o444)
    destination.chmod(0o555)

def _chroot_preexec(rootfs: Path) -> Callable[[], None]:
    def enter() -> None:
        os.chroot(rootfs)
        os.chdir('/tmp')
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')
        for kind, value in (
            (resource.RLIMIT_AS, int(LIMITS['address_space_bytes'])),
            (resource.RLIMIT_NPROC, int(LIMITS['processes'])),
            (resource.RLIMIT_NOFILE, int(LIMITS['open_files'])),
            (resource.RLIMIT_FSIZE, int(LIMITS['file_bytes'])),
            (resource.RLIMIT_CORE, 0),
        ):
            resource.setrlimit(kind, (value, value))
    return enter

def _prepare_staging(base: Path) -> tuple[Path, Path]:
    if os.name != 'posix' or not Path('/proc/self/ns/pid').is_file():
        raise RuntimeError('sandbox preflight must run inside WSL Linux')
    backend = _sandbox_backend()
    if os.geteuid() != 0 and os.environ.get(DIRECT_EXEC_ENV) != '1':
        raise RuntimeError(f'{backend} sandbox setup requires root entry')
    staging = _staging_path(base)
    if staging.exists():
        raise RuntimeError('task staging already exists')
    payload = staging / 'payload'
    roots = staging / 'roots'
    payload.mkdir(parents=True)
    roots.mkdir()
    shutil.copy2(Path(__file__).resolve(), payload / 'evalplus_probe.py')
    contract = {'namespace': PUBLIC_NAMESPACE, 'outer_namespaces': {name: os.readlink(f'/proc/self/ns/{name}') for name in ('pid', 'net', 'mnt', 'ipc', 'uts')}, 'workspace_paths_that_must_be_hidden': ['/mnt/d/study/hq_datasets_v2', '/mnt/d/study/PT'], 'limits': LIMITS}
    _write_json(payload / 'contract.json', contract)
    _write_json(staging / '.owner.json', {'namespace': PUBLIC_NAMESPACE, 'target': str(staging.resolve())})
    return (staging, payload)

def _remove_staging(base: Path) -> bool:
    staging = _staging_path(base)
    if not staging.exists():
        return True
    owner = staging / '.owner.json'
    if not owner.is_file():
        raise RuntimeError('refusing to remove unowned staging')
    value = json.loads(owner.read_text('utf-8'))
    if value != {'namespace': PUBLIC_NAMESPACE, 'target': str(staging.resolve())}:
        raise RuntimeError('refusing to remove staging with invalid owner marker')
    shutil.rmtree(staging)
    return not staging.exists()

def _sandbox_call(*, staging: Path, payload_dir: Path, mode: str, request: Mapping[str, Any], outer_timeout_seconds: float | None = None) -> dict[str, Any]:
    roots = staging / 'roots'
    direct_exec = os.environ.get(DIRECT_EXEC_ENV) == '1'
    backend = 'direct_exec' if direct_exec else _sandbox_backend()
    namespace_rootfs = None
    request_bytes = _canonical_bytes(request)
    timeout_seconds = min(float(LIMITS['per_solution_wall_seconds']) + float(LIMITS['sandbox_setup_grace_seconds']), float(LIMITS['sandbox_setup_grace_seconds']) + (len(request.get('base_input', [])) + len(request.get('plus_input', [])) + 2) * (2 * float(LIMITS['per_program_wall_seconds']) + 2.0)) if outer_timeout_seconds is None else float(outer_timeout_seconds)
    if not 0 < timeout_seconds <= 24 * 60 * 60:
        raise ValueError('outer sandbox timeout is invalid')
    preexec_fn = None
    if direct_exec:
        privilege_prefix = [] if os.geteuid() == 0 else ['/usr/bin/sudo', '-n']
        command = privilege_prefix + [
            '/usr/bin/setpriv', '--reuid', '65534', '--regid', '65534',
            '--clear-groups', '--bounding-set=-all', '--inh-caps=-all',
            '--ambient-caps=-all', '--no-new-privs',
            '/usr/bin/prlimit', '--as=1073741824', '--nproc=32', '--nofile=64',
            '--fsize=1048576', '--core=0', '--',
            '/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'HOME=/tmp', 'TMPDIR=/tmp',
            'LANG=C.UTF-8', 'LC_ALL=C.UTF-8', 'PYTHONHASHSEED=0',
            'PYTHONDONTWRITEBYTECODE=1', 'CUDA_VISIBLE_DEVICES=',
            '/usr/bin/python3', str(payload_dir / 'evalplus_probe.py'),
            '--inside-mode', mode,
        ]
    elif backend == 'chroot_seccomp':
        cloud_rootfs = _build_chroot_runtime(staging.parent)
        _sync_chroot_payload(cloud_rootfs, payload_dir)
        command = ['/usr/bin/python3', '/payload/evalplus_probe.py', '--inside-mode', mode]
        preexec_fn = _chroot_preexec(cloud_rootfs)
    else:
        namespace_rootfs = Path(tempfile.mkdtemp(prefix='root_', dir=roots))
        command = ['/usr/bin/unshare', '--mount', '--uts', '--ipc', '--net', '--pid', '--fork', '--kill-child=SIGKILL', '--propagation', 'private', '/bin/bash', '-c', SANDBOX_SCRIPT, 'sandbox', str(namespace_rootfs), str(payload_dir), mode]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C.UTF-8'}, preexec_fn=preexec_fn)
    try:
        (stdout, stderr) = process.communicate(request_bytes, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        (stdout, stderr) = process.communicate()
        result: dict[str, Any] = {'category': 'sandbox_infrastructure_error', 'detail': 'outer_sandbox_timeout', 'test_executions': 0}
    else:
        if process.returncode != 0:
            detail = 'direct_exec_setup_failed' if direct_exec else f'{backend}_setup_failed'
            result = {'category': 'sandbox_infrastructure_error', 'detail': detail, 'returncode': process.returncode, 'stderr_sha256': _sha256_bytes(stderr), 'stderr_bytes': len(stderr), 'test_executions': 0}
        elif len(stdout) > int(LIMITS['stdout_bytes']):
            result = {'category': 'sandbox_infrastructure_error', 'detail': 'trusted_harness_output_limit', 'test_executions': 0}
        else:
            try:
                result = json.loads(stdout.decode('utf-8'))
            except Exception:
                result = {'category': 'sandbox_infrastructure_error', 'detail': 'invalid_trusted_harness_result', 'stdout_sha256': _sha256_bytes(stdout), 'stderr_sha256': _sha256_bytes(stderr), 'test_executions': 0}
    result['sandbox_call_wall_seconds'] = time.perf_counter() - started
    result['sandbox_backend'] = backend
    if namespace_rootfs is not None:
        try:
            namespace_rootfs.rmdir()
        except OSError:
            result = {'category': 'sandbox_infrastructure_error', 'detail': 'rootfs_mountpoint_not_empty_after_namespace_exit', 'test_executions': 0, 'sandbox_call_wall_seconds': time.perf_counter() - started}
    return result

def _attempt_write(path: str) -> bool:
    try:
        with open(path, 'ab') as handle:
            handle.write(b'x')
    except OSError:
        return False
    return True

def _inside_isolation() -> dict[str, Any]:
    contract = json.loads(Path('/payload/contract.json').read_text('utf-8'))
    status = {}
    for line in Path('/proc/self/status').read_text('utf-8').splitlines():
        if ':' in line:
            (key, value) = line.split(':', 1)
            status[key] = value.strip()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.25)
        network_unusable = probe.connect_ex(('198.18.0.1', 9)) != 0
        probe.close()
    except OSError:
        network_unusable = True
    tmp_probe = Path('/tmp/isolation_write_probe')
    try:
        tmp_probe.write_text('ok', encoding='utf-8')
        tmp_write = tmp_probe.read_text('utf-8') == 'ok'
        tmp_probe.unlink()
    except OSError:
        tmp_write = False
    gpu_paths = list(Path('/dev').glob('nvidia*'))
    checks = {'uid_nonroot': os.getuid() == 65534, 'gid_nonroot': os.getgid() == 65534, 'effective_capabilities_zero': int(status.get('CapEff', '1'), 16) == 0, 'no_new_privileges': status.get('NoNewPrivs') == '1', 'network_namespace_fresh': os.readlink('/proc/self/ns/net') != contract['outer_namespaces']['net'], 'network_unusable': network_unusable, 'pid_namespace_fresh': os.readlink('/proc/self/ns/pid') != contract['outer_namespaces']['pid'], 'mount_namespace_fresh': os.readlink('/proc/self/ns/mnt') != contract['outer_namespaces']['mnt'], 'workspace_hidden': all((not Path(path).exists() for path in contract['workspace_paths_that_must_be_hidden'])), 'home_credentials_hidden': all((not Path(path).exists() for path in ('/root/.aws', '/root/.config/gcloud', '/root/.azure', '/root/.ssh', '/home/.aws'))), 'gpu_devices_hidden': not Path('/dev/dxg').exists() and (not gpu_paths), 'root_write_fails': not _attempt_write('/root_write_probe'), 'input_write_fails': not _attempt_write('/payload/contract.json'), 'tmp_write_succeeds': tmp_write, 'address_space_limit': resource.getrlimit(resource.RLIMIT_AS)[0] == int(LIMITS['address_space_bytes']), 'process_limit': resource.getrlimit(resource.RLIMIT_NPROC)[0] == int(LIMITS['processes']), 'file_limit': resource.getrlimit(resource.RLIMIT_FSIZE)[0] == int(LIMITS['file_bytes']), 'open_file_limit': resource.getrlimit(resource.RLIMIT_NOFILE)[0] == int(LIMITS['open_files'])}
    return {'category': 'pass' if all(checks.values()) else 'sandbox_infrastructure_error', 'detail': 'isolation_contract', 'checks': checks, 'all_passed': all(checks.values()), 'uid': os.getuid(), 'gid': os.getgid(), 'test_executions': 0}

def _child_limits() -> None:
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (int(LIMITS['per_program_cpu_seconds']), int(LIMITS['per_program_cpu_seconds'])))
    for (kind, value) in ((resource.RLIMIT_AS, int(LIMITS['address_space_bytes'])), (resource.RLIMIT_NPROC, int(LIMITS['processes'])), (resource.RLIMIT_FSIZE, int(LIMITS['file_bytes'])), (resource.RLIMIT_NOFILE, int(LIMITS['open_files'])), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (value, value))

def _run_child(argv: Sequence[str], stdin_bytes: bytes, *, wall_timeout_seconds: float | None=None) -> dict[str, Any]:
    suffix = _sha256_bytes(('\x00'.join(argv) + str(time.monotonic_ns())).encode())[:16]
    stdout_path = Path('/tmp') / f'stdout_{suffix}'
    stderr_path = Path('/tmp') / f'stderr_{suffix}'
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with stdout_path.open('wb') as stdout_handle, stderr_path.open('wb') as stderr_handle:
        process = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=stdout_handle, stderr=stderr_handle, cwd='/tmp', env={'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}, preexec_fn=_child_limits)
        timed_out = False
        try:
            process.communicate(stdin_bytes, timeout=float(LIMITS['per_program_wall_seconds']) if wall_timeout_seconds is None else wall_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    stdout = stdout_path.read_bytes()[:int(LIMITS['stdout_bytes']) + 1]
    stderr = stderr_path.read_bytes()[:int(LIMITS['stderr_bytes']) + 1]
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'returncode': process.returncode, 'timed_out': timed_out, 'stdout': stdout, 'stderr': stderr, 'stdout_limit_exceeded': len(stdout) > int(LIMITS['stdout_bytes']), 'stderr_limit_exceeded': len(stderr) > int(LIMITS['stderr_bytes']), 'wall_seconds': time.perf_counter() - started, 'cpu_seconds': after.ru_utime + after.ru_stime - (before.ru_utime + before.ru_stime)}

class _OpaqueNonNone:
    pass

def _wire_encode(value: Any, *, allow_opaque: bool=False) -> Any:
    """Encode outputs to bounded JSON data; never use pickle across a boundary."""
    if value is None:
        return {'t': 'none'}
    if type(value) is bool:
        return {'t': 'bool', 'v': value}
    if type(value) is int:
        return {'t': 'int', 'v': str(value)}
    if type(value) is float:
        if math.isnan(value):
            rendered = 'nan'
        elif math.isinf(value):
            rendered = '+inf' if value > 0 else '-inf'
        else:
            rendered = value.hex()
        return {'t': 'float', 'v': rendered}
    if type(value) is complex:
        return {'t': 'complex', 'r': _wire_encode(float(value.real)), 'i': _wire_encode(float(value.imag))}
    if type(value) is str:
        return {'t': 'str', 'v': value}
    if type(value) is bytes:
        return {'t': 'bytes', 'v': value.hex()}
    if type(value) in (list, tuple):
        return {'t': 'list' if type(value) is list else 'tuple', 'v': [_wire_encode(item, allow_opaque=allow_opaque) for item in value]}
    if type(value) in (set, frozenset):
        encoded = [_wire_encode(item, allow_opaque=allow_opaque) for item in value]
        encoded.sort(key=lambda item: _canonical_bytes(item))
        return {'t': 'set' if type(value) is set else 'frozenset', 'v': encoded}
    if type(value) is dict:
        pairs = [[_wire_encode(key, allow_opaque=allow_opaque), _wire_encode(item, allow_opaque=allow_opaque)] for (key, item) in value.items()]
        pairs.sort(key=lambda item: _canonical_bytes(item[0]))
        return {'t': 'dict', 'v': pairs}
    if allow_opaque:
        return {'t': 'opaque_non_none'}
    raise TypeError(f'unsupported canonical output type: {type(value).__name__}')

def _wire_decode(value: Any, *, depth: int=0) -> Any:
    if depth > 100 or not isinstance(value, Mapping) or type(value.get('t')) is not str:
        raise ValueError('malformed output wire value')
    tag = value['t']
    if tag == 'none' and set(value) == {'t'}:
        return None
    if tag == 'bool' and type(value.get('v')) is bool:
        return value['v']
    if tag == 'int' and type(value.get('v')) is str:
        return int(value['v'])
    if tag == 'float' and type(value.get('v')) is str:
        rendered = value['v']
        if rendered == 'nan':
            return float('nan')
        if rendered == '+inf':
            return float('inf')
        if rendered == '-inf':
            return float('-inf')
        return float.fromhex(rendered)
    if tag == 'complex':
        return complex(_wire_decode(value.get('r'), depth=depth + 1), _wire_decode(value.get('i'), depth=depth + 1))
    if tag == 'str' and type(value.get('v')) is str:
        return value['v']
    if tag == 'bytes' and type(value.get('v')) is str:
        return bytes.fromhex(value['v'])
    if tag in ('list', 'tuple', 'set', 'frozenset') and type(value.get('v')) is list:
        items = [_wire_decode(item, depth=depth + 1) for item in value['v']]
        if tag == 'list':
            return items
        if tag == 'tuple':
            return tuple(items)
        if tag == 'set':
            return set(items)
        return frozenset(items)
    if tag == 'dict' and type(value.get('v')) is list:
        result = {}
        for pair in value['v']:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError('malformed dict wire pair')
            result[_wire_decode(pair[0], depth=depth + 1)] = _wire_decode(pair[1], depth=depth + 1)
        return result
    if tag == 'opaque_non_none' and set(value) == {'t'}:
        return _OpaqueNonNone()
    raise ValueError('unsupported output wire tag')

class _PerCallTimeout(Exception):
    pass

def _call_with_wall_limit(function: Any, arguments: Sequence[Any]) -> Any:

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise _PerCallTimeout('per-input wall limit')
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(LIMITS['per_program_wall_seconds']))
    try:
        return function(*arguments)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

def _inside_split(role: str) -> int:
    """Execute one fresh oracle/candidate base-or-plus namespace."""
    try:
        request = json.loads(sys.stdin.buffer.read().decode('utf-8') or '{}')
        code = request.get('code')
        task_id = request.get('task_id')
        entry_point = request.get('entry_point')
        raw_inputs = request.get('inputs')
        if role not in ('oracle', 'candidate') or type(code) is not str or type(task_id) is not str or (type(entry_point) is not str) or (type(raw_inputs) is not list) or (not raw_inputs) or (len(code.encode('utf-8')) > int(LIMITS['source_bytes'])):
            raise ValueError('malformed split payload')
        deserialized = deserialize_mbpp_inputs(task_id, raw_inputs)
        inputs = list(deserialized)
        namespace: dict[str, Any] = {}
        try:
            compiled = compile(code, '<solution>', 'exec')
        except BaseException as exc:
            result = {'phase_status': 'compile_failure', 'detail': type(exc).__name__, 'call_results': []}
        else:
            try:
                with open(os.devnull, 'w', encoding='utf-8') as sink:
                    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        exec(compiled, namespace)
                function = namespace[entry_point]
                if not callable(function):
                    raise TypeError('entry point is not callable')
            except BaseException as exc:
                result = {'phase_status': 'load_failure', 'detail': type(exc).__name__, 'call_results': []}
            else:
                import copy
                call_results = []
                for inp in inputs:
                    try:
                        arguments = copy.deepcopy(inp) if role == 'oracle' else inp
                        with open(os.devnull, 'w', encoding='utf-8') as sink:
                            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                                output = _call_with_wall_limit(function, arguments)
                        if role == 'oracle' and entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                            output = output is not None
                        encoded = _wire_encode(output, allow_opaque=role == 'candidate')
                        call_results.append({'status': 'ok', 'output': encoded})
                    except _PerCallTimeout:
                        call_results.append({'status': 'timeout'})
                    except BaseException as exc:
                        call_results.append({'status': 'exception', 'detail': type(exc).__name__})
                result = {'phase_status': 'completed', 'detail': 'all_split_calls_attempted', 'call_results': call_results}
    except BaseException as exc:
        result = {'phase_status': 'harness_failure', 'detail': type(exc).__name__, 'call_results': []}
    sys.stdout.buffer.write(_canonical_bytes(result))
    return 0

def _run_split_phase(*, role: str, request: Mapping[str, Any], wall_timeout_seconds: float) -> dict[str, Any]:
    execution = _run_child(['/usr/bin/python3', str(Path(__file__).resolve()), '--inside-split', role], _canonical_bytes(request), wall_timeout_seconds=wall_timeout_seconds)
    summary = {'role': role, 'wall_seconds': execution['wall_seconds'], 'cpu_seconds': execution['cpu_seconds']}
    if execution['timed_out']:
        return {**summary, 'phase_status': 'process_timeout', 'call_results': []}
    if execution['stdout_limit_exceeded'] or execution['stderr_limit_exceeded']:
        return {**summary, 'phase_status': 'output_limit', 'call_results': []}
    if execution['returncode'] != 0:
        return {**summary, 'phase_status': 'process_failure', 'call_results': []}
    try:
        decoded = json.loads(execution['stdout'].decode('utf-8'))
    except Exception:
        return {**summary, 'phase_status': 'invalid_harness_result', 'call_results': []}
    if not isinstance(decoded, Mapping) or type(decoded.get('call_results')) is not list:
        return {**summary, 'phase_status': 'invalid_harness_result', 'call_results': []}
    return {**summary, **decoded}

def _phase_wall_budget(input_count: int, remaining: float) -> float:
    prospective = input_count * float(LIMITS['per_program_wall_seconds']) + 2.0
    return max(0.1, min(prospective, remaining))

def _inside_mbpp_score(request: Mapping[str, Any]) -> dict[str, Any]:
    """Trusted split orchestrator running wholly inside the namespace/chroot."""
    started = time.perf_counter()
    required = ('task_id', 'entry_point', 'stored_prompt', 'canonical_solution', 'candidate_solution', 'base_input', 'plus_input', 'atol')
    if not isinstance(request, Mapping) or any((field not in request for field in required)):
        return {'category': 'sandbox_infrastructure_error', 'detail': 'malformed_trusted_payload', 'test_executions': 0}
    task_id = request['task_id']
    entry_point = request['entry_point']
    stored_prompt = request['stored_prompt']
    canonical_solution = request['canonical_solution']
    candidate_solution = request['candidate_solution']
    base_input = request['base_input']
    plus_input = request['plus_input']
    atol = request['atol']
    if type(task_id) is not str or TASK_ID_PATTERN.fullmatch(task_id) is None or type(entry_point) is not str or (type(stored_prompt) is not str) or (type(canonical_solution) is not str) or (type(candidate_solution) is not str) or (type(base_input) is not list) or (type(plus_input) is not list) or (not base_input) or (not plus_input) or (type(atol) not in (int, float)) or (not math.isfinite(atol)) or (atol < 0):
        return {'category': 'sandbox_infrastructure_error', 'detail': 'malformed_trusted_payload', 'test_executions': 0}
    oracle_code = stored_prompt + canonical_solution
    if any((len(code.encode('utf-8')) > int(LIMITS['source_bytes']) for code in (oracle_code, candidate_solution))) or len(_canonical_bytes(request)) > int(LIMITS['solution_payload_bytes']):
        return {'category': 'sandbox_infrastructure_error', 'detail': 'payload_exceeds_frozen_limit', 'test_executions': 0}
    groups = ('base', 'plus')
    raw_groups = {'base': base_input, 'plus': plus_input}
    phase_records: list[dict[str, Any]] = []
    expected_by_group: dict[str, list[Any]] = {}
    deadline = started + float(LIMITS['per_solution_wall_seconds'])
    for group in groups:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return {'category': 'sandbox_infrastructure_error', 'detail': 'oracle_per_solution_wall_limit', 'test_executions': 0, 'split_phase_executions': len(phase_records)}
        phase = _run_split_phase(role='oracle', request={'code': oracle_code, 'task_id': task_id, 'entry_point': entry_point, 'inputs': raw_groups[group]}, wall_timeout_seconds=_phase_wall_budget(len(raw_groups[group]), remaining))
        phase_records.append({'phase': f'oracle_{group}', 'phase_status': phase['phase_status'], 'wall_seconds': phase['wall_seconds'], 'cpu_seconds': phase['cpu_seconds'], 'call_count': len(phase['call_results'])})
        calls = phase['call_results']
        if phase['phase_status'] != 'completed' or len(calls) != len(raw_groups[group]) or any((call.get('status') != 'ok' for call in calls)):
            return {'category': 'sandbox_infrastructure_error', 'detail': 'canonical_oracle_failure', 'oracle_failure_phase': f'oracle_{group}', 'test_executions': 0, 'split_phase_executions': len(phase_records), 'phase_records': phase_records, 'wall_seconds': time.perf_counter() - started}
        try:
            expected_by_group[group] = [_wire_decode(call['output']) for call in calls]
        except Exception:
            return {'category': 'sandbox_infrastructure_error', 'detail': 'unsupported_canonical_output', 'oracle_failure_phase': f'oracle_{group}', 'test_executions': 0, 'split_phase_executions': len(phase_records), 'phase_records': phase_records, 'wall_seconds': time.perf_counter() - started}
    outcomes = []
    candidate_calls_attempted = 0
    for group in groups:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            for _ in raw_groups[group]:
                outcomes.append({'group': group, 'category': 'timeout'})
            continue
        phase = _run_split_phase(role='candidate', request={'code': candidate_solution, 'task_id': task_id, 'entry_point': entry_point, 'inputs': raw_groups[group]}, wall_timeout_seconds=_phase_wall_budget(len(raw_groups[group]), remaining))
        phase_records.append({'phase': f'candidate_{group}', 'phase_status': phase['phase_status'], 'wall_seconds': phase['wall_seconds'], 'cpu_seconds': phase['cpu_seconds'], 'call_count': len(phase['call_results'])})
        calls = phase['call_results']
        if phase['phase_status'] == 'compile_failure':
            outcomes.extend(({'group': group, 'category': 'compile_failure'} for _ in raw_groups[group]))
            continue
        if phase['phase_status'] in ('load_failure', 'process_failure', 'output_limit', 'invalid_harness_result'):
            outcomes.extend(({'group': group, 'category': 'runtime_failure'} for _ in raw_groups[group]))
            continue
        if phase['phase_status'] == 'process_timeout':
            outcomes.extend(({'group': group, 'category': 'timeout'} for _ in raw_groups[group]))
            continue
        if phase['phase_status'] != 'completed' or len(calls) != len(raw_groups[group]):
            return {'category': 'sandbox_infrastructure_error', 'detail': 'candidate_harness_contract_failure', 'test_executions': candidate_calls_attempted, 'split_phase_executions': len(phase_records), 'phase_records': phase_records, 'wall_seconds': time.perf_counter() - started}
        candidate_calls_attempted += len(calls)
        comparator = OfficialComparatorState(atol)
        for (inp, expected, call) in zip(deserialize_mbpp_inputs(task_id, raw_groups[group]), expected_by_group[group], calls):
            status = call.get('status')
            if status == 'timeout':
                category = 'timeout'
            elif status != 'ok':
                category = 'runtime_failure'
            else:
                try:
                    actual = _wire_decode(call['output'])
                    matched = comparator.compare(entry_point, inp, actual, expected)
                except Exception:
                    matched = False
                category = 'pass' if matched else 'wrong_answer'
            outcomes.append({'group': group, 'category': category})
    aggregate = aggregate_task_results(outcomes, expected_base=len(base_input), expected_plus=len(plus_input))
    aggregate.update({'detail': 'all_base_plus_calls_attempted', 'candidate_call_executions': candidate_calls_attempted, 'split_phase_executions': len(phase_records), 'phase_records': phase_records, 'fresh_namespace_per_split': True, 'namespace_reused_within_split': True, 'oracle_inputs_deepcopied': True, 'candidate_inputs_explicitly_deepcopied': False, 'host_untrusted_pickle_deserialized': False, 'oracle_code_contract': 'exact stored prompt plus exact stored canonical_solution', 'candidate_code_contract': 'serialized prompt plus extracted completion', 'wall_seconds': time.perf_counter() - started})
    return aggregate

def _inside_main(mode: str) -> int:
    try:
        request = json.loads(sys.stdin.buffer.read().decode('utf-8') or '{}')
        if mode == 'isolation':
            result = _inside_isolation()
        elif mode == 'mbpp-score':
            result = _inside_mbpp_score(request)
        else:
            raise ValueError('unknown inside mode')
    except BaseException as exc:
        result = {'category': 'sandbox_infrastructure_error', 'detail': f'trusted_harness_exception:{type(exc).__name__}', 'test_executions': 0}
    sys.stdout.buffer.write(_canonical_bytes(result))
    return 0

def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-mode", choices=("isolation", "mbpp-score"))
    parser.add_argument("--inside-split", choices=("oracle", "candidate"))
    arguments = parser.parse_args()
    if (arguments.inside_mode is None) == (arguments.inside_split is None):
        parser.error("select exactly one inside mode")
    if arguments.inside_mode is not None:
        return _inside_main(arguments.inside_mode)
    return _inside_split(arguments.inside_split)


if __name__ == "__main__":
    raise SystemExit(_cli())
