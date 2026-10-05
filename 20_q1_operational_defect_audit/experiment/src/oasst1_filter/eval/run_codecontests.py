"""Formal sealed eligibility, adapter generation, and compact scoring workers."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import errno
import gc
import gzip
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import wait as wait_connections
import os
from pathlib import Path
import random
import shutil
import signal
import threading
import time
from typing import Any, Mapping, Sequence

from ..data.provenance import sha256_file
from ..protocol import CODECONTESTS_DESIGN_ID, CODECONTESTS_PROTOCOL_ID
from .code_exec import (
    _canonical_bytes,
    _prepare_staging,
    _remove_staging,
    _sandbox_call,
    _task_id_from_raw_line,
    _task_sort_key,
    partition_record,
    prompt_fingerprint,
    serialize_prompt,
)
from .codecontests import (
    codecontests_generation_kwargs,
    extract_codecontests_completion,
)


EVALPLUS_ASSET_SHA256 = "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
EVALPLUS_ASSET_BYTES = 336_032
EXPECTED_ROWS = 378
EXPECTED_SEALED = 290
MAX_NEW_TOKENS = 1_024
MODEL_MAX_POSITION_EMBEDDINGS = 131_072
REQUIRED_STRING_FIELDS = (
    "task_id",
    "prompt",
    "entry_point",
    "canonical_solution",
    "contract",
    "assertion",
)
REQUIRED_LIST_FIELDS = ("base_input", "plus_input")
FORMAL_GENERATION_SEED = 20_260_829


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_codecontests_sealed_rows(
    path: Path, *, completed_training_jobs: int
) -> tuple[dict[str, Any], ...]:
    """Open exactly the sealed 290 rows, gated by all 24 terminal adapters."""

    if completed_training_jobs != 24:
        raise PermissionError("sealed MBPP+ rows require 24 terminal adapters")
    if path.stat().st_size != EVALPLUS_ASSET_BYTES or sha256_file(path) != EVALPLUS_ASSET_SHA256:
        raise ValueError("MBPP+ asset identity differs from freeze")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            total += 1
            task_id = _task_id_from_raw_line(raw_line)
            if task_id in seen:
                raise ValueError("duplicate MBPP+ task identity")
            seen.add(task_id)
            if not partition_record(task_id)["qualification"]:
                row = json.loads(raw_line)
                if not isinstance(row, dict) or row.get("task_id") != task_id:
                    raise ValueError("sealed MBPP+ row identity mismatch")
                rows.append(row)
    if total != EXPECTED_ROWS or len(rows) != EXPECTED_SEALED:
        raise ValueError("sealed MBPP+ population size differs from freeze")
    rows.sort(key=lambda row: _task_sort_key(row["task_id"]))
    return tuple(rows)


def load_codecontests_endpoint_member_rows(
    path: Path, members: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """Stream the asset and JSON-decode only the already sealed endpoint members."""

    if path.stat().st_size != EVALPLUS_ASSET_BYTES or sha256_file(path) != EVALPLUS_ASSET_SHA256:
        raise ValueError("MBPP+ asset identity differs from freeze")
    if not members or any(
        member.get("eligible_order") != index
        or not isinstance(member.get("task_id"), str)
        or not isinstance(member.get("sealed_record_sha256"), str)
        for index, member in enumerate(members)
    ):
        raise ValueError("endpoint member identity/order is invalid")
    wanted = {str(member["task_id"]): member for member in members}
    if len(wanted) != len(members):
        raise ValueError("endpoint member identities are duplicated")
    rows: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    total = 0
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            total += 1
            task_id = _task_id_from_raw_line(raw_line)
            if task_id in seen:
                raise ValueError("duplicate MBPP+ task identity")
            seen.add(task_id)
            member = wanted.get(task_id)
            if member is None:
                continue
            row = json.loads(raw_line)
            if (
                not isinstance(row, dict)
                or row.get("task_id") != task_id
                or member.get("sealed_record_sha256") != _sha256(_canonical_bytes(row))
            ):
                raise ValueError("endpoint member row identity differs from freeze")
            rows[task_id] = row
    if total != EXPECTED_ROWS or set(rows) != set(wanted):
        raise ValueError("endpoint member population differs from freeze")
    return rows


def assess_codecontests_sealed_row(
    row: Mapping[str, Any], tokenizer: Any, *, codecontest_fingerprints: set[str]
) -> dict[str, Any]:
    """Apply the frozen model-independent sealed eligibility checks."""

    if not isinstance(row, Mapping):
        return {"eligible": False, "reason": "malformed_row"}
    for field in REQUIRED_STRING_FIELDS:
        if type(row.get(field)) is not str:
            return {"eligible": False, "reason": f"malformed_{field}"}
    for field in REQUIRED_LIST_FIELDS:
        if type(row.get(field)) is not list or not row[field]:
            return {"eligible": False, "reason": f"malformed_{field}"}
    atol = row.get("atol")
    if type(atol) not in (int, float) or not math.isfinite(atol) or atol < 0:
        return {"eligible": False, "reason": "malformed_atol"}
    try:
        decision = partition_record(row["task_id"])
    except ValueError:
        return {"eligible": False, "reason": "malformed_task_id"}
    if decision["qualification"]:
        return {"eligible": False, "reason": "not_in_sealed_partition"}
    if not row["prompt"].strip() or not row["entry_point"].strip() or not row["canonical_solution"].strip():
        return {"eligible": False, "reason": "empty_required_text"}
    fingerprint = prompt_fingerprint(row["prompt"])
    if fingerprint in codecontest_fingerprints:
        return {"eligible": False, "reason": "exact_codecontests_prompt_overlap"}
    prompt = serialize_prompt(row["prompt"])
    try:
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    except Exception:
        return {"eligible": False, "reason": "tokenization_failed"}
    if prompt_tokens + MAX_NEW_TOKENS > MODEL_MAX_POSITION_EMBEDDINGS:
        return {"eligible": False, "reason": "context_limit_exceeded"}
    return {
        "eligible": True,
        "reason": "eligible",
        "prompt_tokens": prompt_tokens,
        "prompt_fingerprint_sha256": fingerprint,
        "serialized_prompt_sha256": _sha256(prompt.encode("utf-8")),
        "base_input_count": len(row["base_input"]),
        "plus_input_count": len(row["plus_input"]),
    }


def load_codecontests_description_fingerprints(dataset_root: Path) -> set[str]:
    """Read only CodeContests descriptions for the frozen exact-overlap gate."""

    import pyarrow.parquet as pq

    paths = tuple(sorted((dataset_root / "data").glob("*.parquet")))
    if len(paths) != 41:
        raise ValueError("CodeContests description shard count differs from freeze")
    fingerprints: set[str] = set()
    for path in paths:
        table = pq.read_table(path, columns=["description"])
        for value in table.column("description").to_pylist():
            if isinstance(value, str):
                fingerprints.add(prompt_fingerprint(value))
    if len(fingerprints) != 12_567:
        raise ValueError("CodeContests description fingerprint population differs from freeze")
    return fingerprints


def _score_one(
    *, staging: Path, payload: Path, row: Mapping[str, Any], candidate_solution: str
) -> dict[str, Any]:
    return _sandbox_call(
        staging=staging,
        payload_dir=payload,
        mode="mbpp-score",
        request={
            "task_id": row["task_id"],
            "entry_point": row["entry_point"],
            "stored_prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "candidate_solution": candidate_solution,
            "base_input": row["base_input"],
            "plus_input": row["plus_input"],
            "atol": row["atol"],
        },
    )


def _scoring_thread_children() -> set[int]:
    path = Path(f"/proc/self/task/{threading.get_native_id()}/children")
    return {int(value) for value in path.read_text().split()}


def _remove_scoring_sandbox(base: Path, *, prior_children: set[int]) -> None:
    """Reclaim only the private directories of this complete scoring call."""

    if base.is_symlink() or base.resolve() != base.absolute():
        raise RuntimeError("refusing to clean a redirected scoring sandbox")
    # The bound judge normally waits for its child. A transport interruption can
    # leave that child alive. Only children created by this calling thread during
    # this synchronous scoring call (and their descendants) belong to this call.
    direct = _scoring_thread_children() - prior_children
    pending = list(direct)
    descendants = set(direct)
    while pending:
        pid = pending.pop()
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except FileNotFoundError:
            continue
        for value in children:
            child = int(value)
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    for pid in sorted(descendants, reverse=True):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while direct:
        for pid in tuple(direct):
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                direct.remove(pid)
            else:
                if reaped:
                    direct.remove(pid)
        if direct:
            if time.monotonic() >= deadline:
                raise RuntimeError("scoring child could not be reaped; sandbox retained")
            time.sleep(0.01)
    _remove_staging(base)
    # The bound judge registers an atexit fallback for its sibling runtime.
    # A public scoring call must close it before attempting the parent rmdir.
    for name in ("_chroot_runtime", f"_chroot_runtime.tmp_{os.getpid()}"):
        runtime = base / name
        if runtime.is_symlink() or runtime.is_mount():
            raise RuntimeError("refusing to clean a redirected chroot runtime")
        if runtime.exists():
            if runtime.resolve().parent != base.resolve():
                raise RuntimeError("chroot runtime escapes the owned scoring sandbox")
            shutil.rmtree(runtime)
    if base.exists():
        # Unknown entries are not ours to remove: fail closed, without recursion.
        base.rmdir()


def freeze_codecontests_endpoint_eligibility(
    *,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    codecontest_fingerprints: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze one ordered eligible set after canonical-oracle clean-exec checks."""

    if len(rows) != EXPECTED_SEALED:
        raise ValueError("CodeContests endpoint requires the sealed 290 candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    static = [
        assess_codecontests_sealed_row(
            row, tokenizer, codecontest_fingerprints=codecontest_fingerprints
        )
        for row in rows
    ]
    candidates = [index for index, decision in enumerate(static) if decision["eligible"]]
    sandbox_base = output_dir / "oracle_sandbox"
    sandbox_base.mkdir(parents=True, exist_ok=False)
    staging = None
    oracle: dict[int, dict[str, Any]] = {}
    primary_error: BaseException | None = None
    prior_children = _scoring_thread_children()
    try:
        staging, payload = _prepare_staging(sandbox_base)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    _score_one,
                    staging=staging,
                    payload=payload,
                    row=rows[index],
                    candidate_solution=str(rows[index]["prompt"])
                    + str(rows[index]["canonical_solution"]),
                ): index
                for index in candidates
            }
            for future in as_completed(futures):
                oracle[futures[future]] = future.result()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _remove_scoring_sandbox(sandbox_base, prior_children=prior_children)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.cleanup_errors = [{
                "exception_type": type(cleanup_error).__name__,
                "errno": getattr(cleanup_error, "errno", None),
                "message": str(cleanup_error),
            }]
    unresolved = [
        index
        for index, result in oracle.items()
        if result.get("category") == "sandbox_infrastructure_error"
        and result.get("detail") != "canonical_oracle_failure"
    ]
    if unresolved:
        raise RuntimeError("sealed canonical-oracle checks contain infrastructure errors")
    members = []
    exclusions = Counter()
    for sealed_order, (row, decision) in enumerate(zip(rows, static)):
        if not decision["eligible"]:
            exclusions[str(decision["reason"])] += 1
            continue
        result = oracle[sealed_order]
        if result.get("all_tests_pass") is not True:
            exclusions["canonical_oracle_not_all_tests_pass"] += 1
            continue
        members.append(
            {
                "eligible_order": len(members),
                "sealed_order": sealed_order,
                "task_id": row["task_id"],
                "partition_sha256": partition_record(row["task_id"])["partition_sha256"],
                "sealed_record_sha256": _sha256(_canonical_bytes(row)),
                "prompt_tokens": decision["prompt_tokens"],
                "prompt_fingerprint_sha256": decision["prompt_fingerprint_sha256"],
                "serialized_prompt_sha256": decision["serialized_prompt_sha256"],
                "base_input_count": decision["base_input_count"],
                "plus_input_count": decision["plus_input_count"],
            }
        )
    if len(members) < 250:
        raise RuntimeError("sealed eligible task count is below 250")
    manifest = {
        "schema": "codecontests-v1-sealed-eligibility-v1",
        "protocol_id": CODECONTESTS_PROTOCOL_ID,
        "design_id": CODECONTESTS_DESIGN_ID,
        "sealed_candidate_count": len(rows),
        "eligible_count": len(members),
        "members": members,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "unresolved_infrastructure_errors": 0,
        "contains_raw_prompt_tests_canonical_or_output": False,
    }
    _atomic_json(output_dir / "SEALED_ELIGIBILITY.json", manifest)
    return manifest


def _validate_members(
    members: Sequence[Mapping[str, Any]], rows_by_id: Mapping[str, Mapping[str, Any]],
    *, development: bool = False,
) -> list[Mapping[str, Any]]:
    ordered = []
    for index, member in enumerate(members):
        task_id = member.get("task_id")
        row = rows_by_id.get(str(task_id))
        if (
            row is None
            or member.get("eligible_order") != index
            or member.get("sealed_record_sha256") != _sha256(_canonical_bytes(row))
        ):
            raise ValueError("sealed member identity differs from frozen rows")
        ordered.append(row)
    if development:
        if not 1 <= len(ordered) <= 8 or any(
            not str(row["task_id"]).startswith("Mbpp/")
            or not str(row["task_id"]).split("/")[-1].isdigit()
            or int(str(row["task_id"]).split("/")[-1]) < 10000 for row in ordered
        ):
            raise ValueError("development fixtures must be at most eight synthetic non-endpoint IDs")
    elif len(ordered) < 250:
        raise ValueError("frozen eligible set is too small")
    return ordered


def generate_nat_replace_completion(model, tokenizer, *, prompt, prompt_tokens, device):
    """One original MBPP direct completion, with no BRANCH clamp or probability hook."""
    import torch
    from .codecontests import CODECONTESTS_STOP_STRINGS
    if tokenizer.eos_token_id != 151643 or tokenizer.pad_token_id != 151643:
        raise ValueError("NAT-REPLACE tokenizer EOS/pad differs")
    encoded = tokenizer(serialize_prompt(prompt), add_special_tokens=True,
                        truncation=False, return_tensors="pt")
    if tuple(encoded["input_ids"].shape) != (1, prompt_tokens):
        raise ValueError("NAT-REPLACE prompt token binding differs")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(**encoded,
            **codecontests_generation_kwargs(pad_token_id=151643, eos_token_id=151643),
            stop_strings=list(CODECONTESTS_STOP_STRINGS), tokenizer=tokenizer)
    if generated.ndim != 2 or generated.shape[0] != 1:
        raise RuntimeError("NAT-REPLACE generation did not return one sequence")
    sequence = generated.detach().cpu().tolist()[0]
    suffix = sequence[prompt_tokens:]
    if not suffix or len(suffix) > 1024 or sequence[:prompt_tokens] != encoded["input_ids"].detach().cpu().tolist()[0]:
        raise RuntimeError("NAT-REPLACE generation publication is incomplete")
    completion = extract_codecontests_completion(sequence, prompt_tokens, tokenizer)
    return dict(completion=completion, completion_sha256=_sha256(completion.encode("utf-8")),
                generated_tokens=suffix.index(151643)+1 if 151643 in suffix else len(suffix),
                normal_termination=True, length_capped=len(suffix) == 1024 and 151643 not in suffix)


def execute_nat_replace_generation(*, state_id, tasks, model_path, adapter_path,
                                   output_dir, design_sha256, development=False):
    """One loaded state, microbatch one, atomically recoverable fixed MBPP positions."""
    if len({t["task_id"] for t in tasks}) != len(tasks) or (
            not development and len(tasks) != 285) or (development and not 1 <= len(tasks) <= 8):
        raise ValueError("NAT-REPLACE generation task panel differs")
    if development and any(not t["task_id"].startswith("Mbpp/") or int(t["task_id"].split("/")[-1]) < 10000 for t in tasks):
        raise ValueError("NAT-REPLACE development panel must be synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, task in enumerate(tasks):
        binding = dict(state_id=state_id, task_id=task["task_id"], design_sha256=design_sha256,
                       task_sha256=_sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()))
        path = output_dir / f"{index:03d}.json"
        if path.exists():
            value = json.loads(path.read_text("utf-8"))
            if (any(value.get(k) != v for k, v in binding.items())
                    or value.get("normal_termination") is not True
                    or not isinstance(value.get("completion"), str)
                    or _sha256(value["completion"].encode()) != value.get("completion_sha256")
                    or type(value.get("generated_tokens")) is not int
                    or not 1 <= value["generated_tokens"] <= 1024):
                raise ValueError("NAT-REPLACE saved generation binding is invalid")
        else:
            pending.append((path, task, binding))
    started = time.perf_counter()
    peak_reserved = 0
    if pending:
        import torch
        from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
        from peft import PeftModel
        from ..train.modeling import configure_determinism
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("NAT-REPLACE deterministic CUDA environment missing")
        configure_determinism(FORMAL_GENERATION_SEED)
        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        base = Qwen2ForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0")
        model = base if adapter_path is None else PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        try:
            for path, task, binding in pending:
                value = generate_nat_replace_completion(model, tokenizer, prompt=task["prompt"],
                                prompt_tokens=int(task["prompt_tokens"]), device="cuda:0")
                _atomic_json(path, {**binding, **value})
            peak_reserved = int(torch.cuda.max_memory_reserved())
        finally:
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
    result = dict(state_id=state_id, design_sha256=design_sha256,
                  completed_positions=len(tasks), new_positions=len(pending),
                  process_seconds=time.perf_counter()-started, peak_reserved_bytes=peak_reserved)
    _atomic_json(output_dir / "COMPLETE.json", result)
    return result


def _validate_nat_lbpp_generation_receipt(value, *, binding):
    if (
        value.get("schema") != "nat-lbpp-fix16-generation-v1"
        or any(value.get(key) != expected for key, expected in binding.items())
        or value.get("normal_termination") is not True
        or not isinstance(value.get("completion"), str)
        or _sha256(value["completion"].encode("utf-8")) != value.get("completion_sha256")
        or not isinstance(value.get("generated_token_ids"), list)
        or type(value.get("generated_tokens")) is not int
        or value["generated_tokens"] != len(value["generated_token_ids"])
        or not 0 <= value["generated_tokens"] <= 2_048
        or type(value.get("eos_reached")) is not bool
        or type(value.get("length_cap_reached")) is not bool
        or type(value.get("empty")) is not bool
        or value["empty"] != (not value["completion"].strip())
    ):
        raise ValueError("NAT-LBPP-FIX16 saved generation binding is invalid")


def execute_nat_lbpp_fixed16_generation(
    *, state_id, tasks, model_path, adapter_path, output_dir, execution_sha256
):
    """Generate one literal complete-file suffix for every bound LBPP prompt."""

    from .lbpp import consume_nat_lbpp_completion

    if not 1 <= len(tasks) <= 161 or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("NAT-LBPP-FIX16 generation panel is invalid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, task in enumerate(tasks):
        prompt = task.get("prompt")
        binding = {
            "state_id": state_id,
            "task_id": task.get("task_id"),
            "execution_sha256": execution_sha256,
            "prompt_sha256": task.get("prompt_sha256"),
        }
        if not isinstance(prompt, str) or _sha256(prompt.encode("utf-8")) != binding["prompt_sha256"]:
            raise ValueError("NAT-LBPP-FIX16 prompt binding is invalid")
        path = output_dir / f"{index:03d}.json"
        if path.exists():
            _validate_nat_lbpp_generation_receipt(json.loads(path.read_text("utf-8")), binding=binding)
        else:
            pending.append((path, task, binding))

    started = time.perf_counter()
    complete_path = output_dir / "COMPLETE.json"
    if not pending and complete_path.exists():
        complete = json.loads(complete_path.read_text("utf-8"))
        if (
            complete.get("schema") != "nat-lbpp-fix16-generation-complete-v1"
            or complete.get("state_id") != state_id
            or complete.get("execution_sha256") != execution_sha256
            or complete.get("completed_positions") != len(tasks)
        ):
            raise ValueError("NAT-LBPP-FIX16 generation terminal binding is invalid")
        return complete
    peak_reserved = 0
    effective_config = None
    if pending:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from ..train.modeling import configure_determinism

        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("NAT-LBPP-FIX16 deterministic CUDA environment missing")
        configure_determinism(FORMAL_GENERATION_SEED)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        if tokenizer.eos_token_id != 151_643 or tokenizer.pad_token_id != 151_643:
            raise ValueError("NAT-LBPP-FIX16 tokenizer EOS/pad differs")
        base = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="eager",
        ).to("cuda:0")
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
        model.eval()
        active_adapter = model.active_adapter
        lora_keys = [name for name in model.state_dict() if "lora_" in name]
        if active_adapter != "default" or not lora_keys or model.training:
            raise RuntimeError("NAT-LBPP-FIX16 adapter activation is invalid")
        generation_config = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1,
            max_new_tokens=2_048, min_new_tokens=0, min_length=0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
            eos_token_id=151_643, pad_token_id=151_643, use_cache=True,
            forced_bos_token_id=None, forced_eos_token_id=None,
            bad_words_ids=None, suppress_tokens=None, begin_suppress_tokens=None,
            sequence_bias=None,
        )
        effective_config = generation_config.to_dict()
        model_loaded_unix = time.time()
        barrier_released_unix = None
        barrier_dir_text = os.environ.get("NAT_LBPP_BARRIER_DIR")
        if barrier_dir_text:
            barrier_dir = Path(barrier_dir_text)
            barrier_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(barrier_dir / f"ready-{state_id}-{os.getpid()}.json", {
                "state_id": state_id, "pid": os.getpid(), "loaded_unix": model_loaded_unix,
            })
            barrier_target = int(os.environ.get("NAT_LBPP_BARRIER_TARGET", "4"))
            barrier_deadline = time.monotonic() + 600
            while len(list(barrier_dir.glob("ready-*.json"))) < barrier_target:
                if time.monotonic() >= barrier_deadline:
                    raise TimeoutError("NAT-LBPP-FIX16 launch barrier timed out")
                time.sleep(0.1)
            barrier_released_unix = time.time()
        torch.cuda.reset_peak_memory_stats()
        generation_started_unix = time.time()
        try:
            for path, task, binding in pending:
                encoded = tokenizer(
                    task["prompt"], add_special_tokens=False, truncation=False,
                    return_tensors="pt",
                )
                input_width = int(encoded["input_ids"].shape[1])
                if input_width != int(task["prompt_tokens"]):
                    raise ValueError("NAT-LBPP-FIX16 prompt token count differs")
                prompt_ids = encoded["input_ids"].detach().cpu().tolist()[0]
                encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(**encoded, generation_config=generation_config)
                sequence = generated.detach().cpu().tolist()[0]
                if sequence[:input_width] != prompt_ids:
                    raise RuntimeError("NAT-LBPP-FIX16 generated prefix differs")
                consumed = consume_nat_lbpp_completion(
                    sequence, input_width=input_width, tokenizer=tokenizer
                )
                _atomic_json(path, {
                    "schema": "nat-lbpp-fix16-generation-v1", **binding, **consumed,
                    "completion_sha256": _sha256(consumed["completion"].encode("utf-8")),
                    "normal_termination": True,
                })
                del encoded, generated
            peak_reserved = int(torch.cuda.max_memory_reserved())
        finally:
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "schema": "nat-lbpp-fix16-generation-complete-v1",
        "state_id": state_id, "execution_sha256": execution_sha256,
        "completed_positions": len(tasks), "new_positions": len(pending),
        "process_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": peak_reserved, "generation_config": effective_config,
        "model_loaded_unix": model_loaded_unix if pending else None,
        "barrier_released_unix": barrier_released_unix if pending else None,
        "generation_started_unix": generation_started_unix if pending else None,
        "generation_ended_unix": time.time() if pending else None,
        "active_adapter": active_adapter if pending else None,
        "lora_key_count": len(lora_keys) if pending else None,
    }
    _atomic_json(complete_path, result)
    return result


def execute_codecontests_adapter_generation(
    *,
    members: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    development: bool = False,
) -> dict[str, Any]:
    """Generate one deterministic completion per frozen eligible task."""

    rows = _validate_members(members, rows_by_id, development=development)
    raw_path = output_dir / "RAW_GENERATIONS.jsonl"
    terminal_path = output_dir / "GENERATION_RESULT.json"
    if raw_path.exists() or terminal_path.exists():
        raise FileExistsError("CodeContests generation output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    import numpy as np
    import torch
    from ..train.modeling import configure_determinism
    from peft import PeftModel
    from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

    random.seed(FORMAL_GENERATION_SEED)
    np.random.seed(FORMAL_GENERATION_SEED)
    torch.manual_seed(FORMAL_GENERATION_SEED)
    torch.cuda.manual_seed_all(FORMAL_GENERATION_SEED)
    configure_determinism(917100 if development else FORMAL_GENERATION_SEED)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id != 151_643 or tokenizer.pad_token_id != 151_643:
        raise RuntimeError("CodeContests generation tokenizer identity mismatch")
    tokenizer.padding_side = "left"
    model = None
    generated_rows = []
    batch_metrics = []
    started = time.perf_counter()
    peak_reserved = 0
    try:
        base = Qwen2ForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda:0")
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
        model.eval()
        for batch_index, start in enumerate(range(0, len(rows), 8)):
            batch_rows = rows[start : start + 8]
            batch_members = members[start : start + 8]
            encoded = tokenizer(
                [serialize_prompt(str(row["prompt"])) for row in batch_rows],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            observed = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
            expected = [int(member["prompt_tokens"]) for member in batch_members]
            if observed != expected:
                raise RuntimeError("CodeContests formal prompt token counts differ from freeze")
            padded_width = int(encoded["input_ids"].shape[1])
            encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
            torch.cuda.reset_peak_memory_stats(0)
            batch_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    **codecontests_generation_kwargs(
                        pad_token_id=int(tokenizer.pad_token_id),
                        eos_token_id=int(tokenizer.eos_token_id),
                    ),
                    stop_strings=["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n\"\"\"", "\nassert"],
                    tokenizer=tokenizer,
                )
            torch.cuda.synchronize(0)
            values = generated.detach().cpu().tolist()
            for offset, sequence in enumerate(values):
                completion = extract_codecontests_completion(sequence, padded_width, tokenizer)
                suffix = sequence[padded_width:]
                try:
                    generated_tokens = suffix.index(int(tokenizer.eos_token_id)) + 1
                except ValueError:
                    generated_tokens = len(suffix)
                generated_rows.append(
                    {
                        "task_id": batch_members[offset]["task_id"],
                        "completion": completion,
                        "completion_sha256": _sha256(completion.encode("utf-8")),
                        "generated_tokens": generated_tokens,
                    }
                )
            batch_peak = int(torch.cuda.max_memory_reserved(0))
            peak_reserved = max(peak_reserved, batch_peak)
            batch_metrics.append(
                {
                    "batch_index": batch_index,
                    "size": len(batch_rows),
                    "wall_seconds": time.perf_counter() - batch_started,
                    "peak_reserved_bytes": batch_peak,
                }
            )
            del encoded, generated, values
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
    with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in generated_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    compact = [
        {key: row[key] for key in ("task_id", "completion_sha256", "generated_tokens")}
        for row in generated_rows
    ]
    result = {
            "schema": "codecontests-development-generation-v1" if development else "codecontests-v1-formal-generation-v1",
        "task_count": len(generated_rows),
        "output_sha256": _sha256(_canonical_bytes(compact)),
        "raw_generation_file": {"bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)},
        "batch_metrics": batch_metrics,
        "peak_reserved_bytes": peak_reserved,
        "process_seconds": time.perf_counter() - started,
        "finite": True,
    }
    _atomic_json(terminal_path, result)
    return result


def _interrupt_scoring(signum: int, _frame: object) -> None:
    # A second termination request must not interrupt ownership cleanup.
    signal.signal(signum, signal.SIG_IGN)
    # OSError subclasses can be caught by copy fallbacks in the bound staging
    # builder. Termination must unwind all the way to this caller's finally.
    raise KeyboardInterrupt("scoring controller received external termination")


def execute_nat_replace_scoring(*, state_id, members, rows_by_id, generation_dir,
                               output_dir, design_sha256, development=False):
    """Serial private judge per state; only the outer queue provides four global slots."""
    rows = _validate_members(members, rows_by_id, development=development)
    if not development and len(rows) != 285:
        raise ValueError("NAT-REPLACE scoring requires the fixed E285")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, (member, row) in enumerate(zip(members, rows, strict=True)):
        raw = json.loads((generation_dir / f"{index:03d}.json").read_text("utf-8"))
        if (raw.get("state_id") != state_id or raw.get("task_id") != member["task_id"]
                or raw.get("design_sha256") != design_sha256 or raw.get("normal_termination") is not True
                or not isinstance(raw.get("completion"), str)
                or _sha256(raw["completion"].encode()) != raw.get("completion_sha256")):
            raise ValueError("NAT-REPLACE scoring generation binding differs")
        binding = dict(state_id=state_id, task_id=member["task_id"], design_sha256=design_sha256,
                       completion_sha256=raw["completion_sha256"],
                       sealed_record_sha256=member["sealed_record_sha256"])
        path = output_dir / f"{index:03d}.json"
        if path.exists():
            value = json.loads(path.read_text("utf-8"))
            if any(value.get(k) != v for k, v in binding.items()) or type(value.get("Y")) is not int or value["Y"] not in (0, 1):
                raise ValueError("NAT-REPLACE saved score binding differs")
        else:
            pending.append((path, row, raw, binding))
    started = time.perf_counter()
    if pending:
        os.umask(0o022)
        sandbox = output_dir.absolute() / "sandbox"
        sandbox.mkdir(exist_ok=False)
        prior_children = _scoring_thread_children()
        previous = signal.signal(signal.SIGTERM, _interrupt_scoring)
        try:
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            try:
                staging, payload = _prepare_staging(sandbox)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            for path, row, raw, binding in pending:
                score = _score_one(staging=staging, payload=payload, row=row,
                          candidate_solution=serialize_prompt(str(row["prompt"])) + raw["completion"])
                if score.get("category") == "sandbox_infrastructure_error" or type(score.get("all_tests_pass")) is not bool:
                    raise RuntimeError("NAT-REPLACE judge infrastructure requires repair; Y remains missing")
                _atomic_json(path, {**binding, "Y": int(score["all_tests_pass"]),
                                    "category": score.get("category"),
                                    "test_executions": score.get("test_executions", 0)})
        finally:
            try:
                _remove_scoring_sandbox(sandbox, prior_children=prior_children)
            finally:
                signal.signal(signal.SIGTERM, previous)
    result = dict(state_id=state_id, design_sha256=design_sha256,
                  completed_positions=len(rows), new_positions=len(pending),
                  process_seconds=time.perf_counter()-started)
    _atomic_json(output_dir / "COMPLETE.json", result)
    return result


def execute_nat_lbpp_fixed16_scoring(
    *, state_id, tasks, generation_dir, output_dir, execution_sha256,
    sandbox_base, cpu_id, dependency_path=None
):
    """Score fixed completions, preserving candidate failures as zero and infra as missing."""

    from .lbpp import judge_lbpp_program

    if not 1 <= len(tasks) <= 161 or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("NAT-LBPP-FIX16 scoring panel is invalid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for index, task in enumerate(tasks):
        generation = json.loads((Path(generation_dir) / f"{index:03d}.json").read_text("utf-8"))
        generation_binding = {
            "state_id": state_id, "task_id": task["task_id"],
            "execution_sha256": execution_sha256,
            "prompt_sha256": task["prompt_sha256"],
        }
        _validate_nat_lbpp_generation_receipt(generation, binding=generation_binding)
        binding = {
            "state_id": state_id, "task_id": task["task_id"],
            "execution_sha256": execution_sha256,
            "completion_sha256": generation["completion_sha256"],
            "contract_sha256": task["contract_sha256"],
        }
        path = output_dir / f"{index:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text("utf-8"))
            if (
                saved.get("schema") != "nat-lbpp-fix16-score-v1"
                or any(saved.get(key) != expected for key, expected in binding.items())
                or type(saved.get("Y")) is not int or saved["Y"] not in (0, 1)
            ):
                raise ValueError("NAT-LBPP-FIX16 saved score binding is invalid")
        else:
            pending.append((path, task, generation, binding))
    started = time.perf_counter()
    for path, task, generation, binding in pending:
        if generation["empty"]:
            judged = {"category": "empty_output", "detail": "whitespace_only", "test_executions": 0}
            outcome = 0
        else:
            judged = judge_lbpp_program(
                generation["completion"], test_source=task["test_source"],
                expected_test_count=int(task["expected_test_count"]), task_id=task["task_id"],
                sandbox_base=Path(sandbox_base), cpu_id=int(cpu_id),
                dependency_path=Path(dependency_path) if dependency_path else None,
            )
            if judged.get("category") == "sandbox_infrastructure_error":
                raise RuntimeError("NAT-LBPP-FIX16 judge infrastructure requires repair; Y remains missing")
            outcome = int(judged.get("category") == "pass")
        _atomic_json(path, {
            "schema": "nat-lbpp-fix16-score-v1", **binding, "Y": outcome,
            "category": judged.get("category"), "detail": judged.get("detail"),
            "test_executions": judged.get("test_executions", 0),
            "wall_seconds": judged.get("wall_seconds", 0.0),
        })
    result = {
        "schema": "nat-lbpp-fix16-score-complete-v1", "state_id": state_id,
        "execution_sha256": execution_sha256, "completed_positions": len(tasks),
        "new_positions": len(pending), "process_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "COMPLETE.json", result)
    return result


def execute_codecontests_scoring(
    *,
    members: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    generation_dir: Path,
    output_dir: Path,
    retain_generation: bool = False,
    development: bool = False,
) -> dict[str, Any]:
    """Score one checkpoint and delete its transient code after acceptance."""

    os.umask(0o022)
    rows = _validate_members(members, rows_by_id, development=development)
    raw_path = generation_dir / "RAW_GENERATIONS.jsonl"
    generation = json.loads((generation_dir / "GENERATION_RESULT.json").read_text("utf-8"))
    identity = generation.get("raw_generation_file", {})
    if raw_path.stat().st_size != identity.get("bytes") or sha256_file(raw_path) != identity.get("sha256"):
        raise ValueError("raw generation identity differs from accepted generation")
    generated = [json.loads(line) for line in raw_path.read_text("utf-8").splitlines()]
    if [row["task_id"] for row in generated] != [member["task_id"] for member in members]:
        raise ValueError("generated task order differs from eligible freeze")
    output_dir.mkdir(parents=True, exist_ok=True)
    sandbox_base = output_dir.absolute() / "sandbox"
    sandbox_base.mkdir(exist_ok=False)
    staging = None
    results: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    primary_error: BaseException | None = None
    prior_children = _scoring_thread_children()
    original_sigterm = None
    if threading.current_thread() is threading.main_thread():
        original_sigterm = signal.signal(signal.SIGTERM, _interrupt_scoring)
    try:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            staging, payload = _prepare_staging(sandbox_base)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        for index, (row, completion) in enumerate(zip(rows, generated)):
            if _sha256(str(completion["completion"]).encode("utf-8")) != completion["completion_sha256"]:
                raise ValueError("generated completion digest mismatch")
            results[index] = _score_one(
                staging=staging,
                payload=payload,
                row=row,
                candidate_solution=serialize_prompt(str(row["prompt"])) + str(completion["completion"]),
            )
            if results[index].get("category") == "sandbox_infrastructure_error":
                raise RuntimeError("checkpoint scoring contains unresolved infrastructure errors")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _remove_scoring_sandbox(sandbox_base, prior_children=prior_children)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.cleanup_errors = [{
                "exception_type": type(cleanup_error).__name__,
                "errno": getattr(cleanup_error, "errno", None),
                "message": str(cleanup_error),
            }]
        finally:
            if original_sigterm is not None:
                signal.signal(signal.SIGTERM, original_sigterm)
    compact = []
    for index, (member, completion) in enumerate(zip(members, generated)):
        scored = results[index]
        compact.append(
            {
                "eligible_order": index,
                "task_id": member["task_id"],
                "completion_sha256": completion["completion_sha256"],
                "generated_tokens": completion["generated_tokens"],
                "category": scored.get("category"),
                "detail": scored.get("detail"),
                "all_tests_pass": scored.get("all_tests_pass", False),
                "test_executions": scored.get("test_executions", 0),
                "test_passes": scored.get("test_passes", 0),
                "test_pass_fraction": scored.get("test_pass_fraction", 0.0),
                "group_counts": scored.get("group_counts", {}),
            }
        )
    infrastructure_errors = sum(row["category"] == "sandbox_infrastructure_error" for row in compact)
    if infrastructure_errors:
        raise RuntimeError("checkpoint scoring contains unresolved infrastructure errors")
    result = {
        "schema": "codecontests-development-score-v1" if development else "codecontests-v1-formal-score-v1",
        "task_count": len(compact),
        "pass_count": sum(row["all_tests_pass"] is True for row in compact),
        "pass_fraction": sum(row["all_tests_pass"] is True for row in compact) / len(compact),
        "infrastructure_error_count": 0,
        "tasks": compact,
        "scoring_wall_seconds": time.perf_counter() - started,
        "raw_completion_persisted": True,
    }
    _atomic_json(output_dir / "SCORE_RESULT.json", result)
    if retain_generation:
        return result
    try:
        raw_path.unlink()
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
        result["raw_cleanup_error"] = {
            "exception_type": type(exc).__name__, "errno": exc.errno,
            "message": "accepted raw copy retained by filesystem policy",
        }
    else:
        result["raw_completion_persisted"] = False
    _atomic_json(output_dir / "SCORE_RESULT.json", result)
    return result


def execute_codecontests_reference_bridge(
    *,
    members: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Run only the fixed endpoint reference programs through the full scorer."""

    os.umask(0o022)
    rows = _validate_members(members, rows_by_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    sandbox_base = output_dir.absolute() / "sandbox"
    sandbox_base.mkdir(exist_ok=False)
    prior_children = _scoring_thread_children()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    primary_error: BaseException | None = None
    try:
        staging, payload = _prepare_staging(sandbox_base)
        for index, (member, row) in enumerate(zip(members, rows)):
            verdict = _score_one(
                staging=staging,
                payload=payload,
                row=row,
                candidate_solution=serialize_prompt(str(row["prompt"]))
                + str(row["canonical_solution"]),
            )
            category = verdict.get("category")
            if category == "sandbox_infrastructure_error":
                raise RuntimeError("endpoint reference bridge contains infrastructure errors")
            results.append(
                {
                    "eligible_order": index,
                    "task_id": member["task_id"],
                    "category": category,
                    "all_tests_pass": verdict.get("all_tests_pass") is True,
                    "test_executions": verdict.get("test_executions", 0),
                }
            )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _remove_scoring_sandbox(sandbox_base, prior_children=prior_children)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.cleanup_errors = [{
                "exception_type": type(cleanup_error).__name__,
                "errno": getattr(cleanup_error, "errno", None),
                "message": str(cleanup_error),
            }]
    result = {
        "schema": "codecontests-s12b-endpoint-reference-bridge-v1",
        "task_count": len(results),
        "pass_count": sum(row["all_tests_pass"] for row in results),
        "infrastructure_error_count": 0,
        "tasks": results,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "REFERENCE_BRIDGE_RESULT.json", result)
    return result


def _reference_scoring_slot(base, indexed_rows, connection, deadline_unix):
    """One spawned, single-thread owner; reuse its runtime only sequentially."""
    signal.signal(signal.SIGTERM, _interrupt_scoring)
    os.umask(0o022)
    prior_children = _scoring_thread_children()
    failure = None
    try:
        base.mkdir()
        staging, payload = _prepare_staging(base)
        for index, row in indexed_rows:
            if deadline_unix is not None and time.time() >= deadline_unix:
                raise TimeoutError("endpoint qualification deadline expired")
            verdict = _score_one(
                staging=staging, payload=payload, row=row,
                candidate_solution=serialize_prompt(str(row["prompt"]))
                + str(row["canonical_solution"]),
            )
            connection.send(("result", index, verdict))
    except BaseException as exc:
        failure = {
            "exception_type": type(exc).__name__, "errno": getattr(exc, "errno", None),
            "message": str(exc)[:2048],
        }
    finally:
        try:
            _remove_scoring_sandbox(base, prior_children=prior_children)
        except BaseException as exc:
            cleanup = {"exception_type": type(exc).__name__, "errno": getattr(exc, "errno", None),
                       "message": str(exc)[:2048]}
            if failure is None:
                failure = cleanup
            else:
                failure["cleanup_error"] = cleanup
        try:
            connection.send(("error", failure) if failure else ("closed",))
        finally:
            connection.close()


def execute_codecontests_reference_qualification(
    *,
    members: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    workers: int,
    progress_callback=None,
    deadline_unix: float | None = None,
) -> dict[str, object]:
    """Execute the real 285-reference component in serial or canonical-C4 mode."""

    if workers not in {1, 4}:
        raise ValueError("endpoint reference qualification workers must be one or four")
    os.umask(0o022)
    rows = _validate_members(members, rows_by_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    sandbox_base = output_dir.absolute() / "sandbox"
    sandbox_base.mkdir(exist_ok=False)
    results: list[dict[str, Any] | None] = [None] * len(rows)
    context = multiprocessing.get_context("spawn")
    slots = []
    pending = {}
    primary_error = None
    original_sigterm = None
    if threading.current_thread() is threading.main_thread():
        original_sigterm = signal.signal(signal.SIGTERM, _interrupt_scoring)
    try:
        for slot in range(workers):
            receive, send = context.Pipe(duplex=False)
            process = context.Process(
                target=_reference_scoring_slot,
                args=(sandbox_base / f"slot_{slot}",
                      [(i, rows[i]) for i in range(slot, len(rows), workers)],
                      send, deadline_unix),
            )
            process.start()
            send.close()
            slots.append(process)
            pending[receive] = slot
        while pending:
            if deadline_unix is not None and time.time() >= deadline_unix:
                raise TimeoutError("endpoint qualification deadline expired")
            for connection in wait_connections(list(pending), timeout=0.5):
                message = connection.recv()
                if message[0] == "closed":
                    pending.pop(connection)
                    connection.close()
                    continue
                if message[0] == "error":
                    detail = message[1]
                    if detail["exception_type"] == "TimeoutError":
                        raise TimeoutError(detail["message"])
                    error = RuntimeError(f"endpoint scoring slot failed: {detail}")
                    error.worker_failure = detail
                    raise error
                _, index, verdict = message
                if index % workers != pending[connection] or results[index] is not None:
                    raise RuntimeError("endpoint scoring slot returned an invalid or duplicate key")
                results[index] = {
                    "eligible_order": index,
                    "task_id": members[index]["task_id"],
                    "category": verdict.get("category"),
                    "all_tests_pass": verdict.get("all_tests_pass") is True,
                    "test_executions": verdict.get("test_executions", 0),
                }
                if progress_callback is not None:
                    progress_callback(sum(row is not None for row in results), len(results))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            if pending:
                for process in slots:
                    if process.is_alive():
                        process.terminate()
            for process in slots:
                process.join(timeout=30)
                if process.is_alive():
                    raise RuntimeError("endpoint slot cleanup did not finish; owned sandbox retained")
            for connection in pending:
                connection.close()
            sandbox_base.rmdir()
        except BaseException as exc:
            if primary_error is None:
                raise
            primary_error.cleanup_errors = [{"exception_type": type(exc).__name__,
                "errno": getattr(exc, "errno", None), "message": str(exc)[:2048]}]
        finally:
            if original_sigterm is not None:
                signal.signal(signal.SIGTERM, original_sigterm)
    if any(row is None for row in results):
        raise RuntimeError("endpoint reference qualification is incomplete")
    compact = tuple(row for row in results if row is not None)
    result = {
        "schema": "codecontests-s12c-endpoint-reference-qualification-v1",
        "execution_mode": "canonical_c4" if workers == 4 else "serial_c1",
        "task_count": len(compact),
        "pass_count": sum(row["all_tests_pass"] for row in compact),
        "infrastructure_error_count": sum(
            row["category"] == "sandbox_infrastructure_error" for row in compact
        ),
        "tasks": compact,
    }
    _atomic_json(output_dir / "REFERENCE_QUALIFICATION_RESULT.json", result)
    return result


def _load_rows_by_members(asset: Path, members: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    rows = load_codecontests_sealed_rows(asset, completed_training_jobs=24)
    wanted = {str(member["task_id"]) for member in members}
    return {row["task_id"]: row for row in rows if row["task_id"] in wanted}


def generate_branch_pc_completion(model, tokenizer, geometry, condition, *, device, eos_ids, pad_id,
                                  observe=True):
    """Actual greedy first-forward observation; only tokens supplied by the contract."""
    import torch
    from transformers import GenerationConfig
    request = geometry["conditions"][condition]
    ids = torch.tensor([request["input_ids"]], device=device)
    summary = {}

    def capture(module, args, output):
        if summary:
            return None
        logits = output.logits[0, -1].detach().float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("BRANCH nonfinite raw logits")
        logp = torch.log_softmax(logits, dim=-1)
        token_ids = torch.arange(logits.numel(), device=logits.device)
        a_r, a_m = geometry["aR"], geometry["aM"]
        for label, target in (("R", a_r), ("M", a_m)):
            value = float(logp[target].cpu())
            probability = math.exp(value)
            summary.update({f"logp_{label}": value, f"p_{label}": probability,
                            f"underflow_{label}": probability == 0.0,
                            f"rank_{label}": int(((logits > logits[target]) |
                                ((logits == logits[target]) & (token_ids < target))).sum().cpu()) + 1})
        summary.update(m=summary["logp_R"]-summary["logp_M"],
                       argmax_token_id=int(logits.argmax().cpu()),
                       candidate_probability_sum=summary["p_R"]+summary["p_M"],
                       log_candidate_probability_sum=float(torch.logsumexp(logp[[a_r, a_m]], dim=0).cpu()))
        return None

    # PEFT delegates generate to the adapted base module, bypassing its wrapper's
    # forward method. Observe that actual module without removing the adapter.
    forward_owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    hook = forward_owner.register_forward_hook(capture) if condition == "U" and observe else None
    config = GenerationConfig(do_sample=False, num_beams=1, num_beam_groups=1,
        max_new_tokens=request["max_new_tokens"], min_length=0, min_new_tokens=0,
        repetition_penalty=1.0, no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
        forced_bos_token_id=None, forced_eos_token_id=None, bad_words_ids=None,
        suppress_tokens=None, begin_suppress_tokens=None, sequence_bias=None,
        token_healing=False, renormalize_logits=False, remove_invalid_values=False,
        exponential_decay_length_penalty=None, constraints=None, force_words_ids=None,
        stop_strings=None, eos_token_id=list(eos_ids), pad_token_id=pad_id, use_cache=True)
    try:
        with torch.inference_mode():
            sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      generation_config=config)[0].cpu().tolist()
    finally:
        if hook is not None:
            hook.remove()
    width = len(request["input_ids"])
    generated = sequence[width:]
    if sequence[:width] != request["input_ids"] or not generated or len(generated) > request["max_new_tokens"]:
        raise RuntimeError("BRANCH incomplete generation contract")
    eos_position = next((i for i, token in enumerate(generated) if token in eos_ids), None)
    if eos_position is None and len(generated) != request["max_new_tokens"]:
        raise RuntimeError("BRANCH generation interrupted before EOS or budget")
    if condition == "U" and observe:
        if not summary or generated[0] != summary["argmax_token_id"]:
            raise RuntimeError("BRANCH actual first token differs from raw full-vocabulary argmax")
        summary["first_token_id"] = generated[0]
        summary["first_branch"] = "R" if generated[0] == geometry["aR"] else "M" if generated[0] == geometry["aM"] else "other"
    response = sequence[geometry["prompt_tokens"]:width] + (generated if eos_position is None else generated[:eos_position])
    source = tokenizer.decode(response, clean_up_tokenization_spaces=False, skip_special_tokens=False)
    return {"source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "generated_token_ids": generated, "generated_tokens": len(generated),
            "stop_reason": "eos" if eos_position is not None else "max_new_tokens",
            "probability": summary if condition == "U" and observe else None}


def load_branch_pc_generation(path, *, state_id, pair_id, condition, design_sha256):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if any(value.get(k) != v for k, v in {"state_id": state_id, "pair_id": pair_id,
            "condition": condition, "design_sha256": design_sha256}.items()):
        raise ValueError("BRANCH generation identity differs")
    if hashlib.sha256(value["source"].encode()).hexdigest() != value["source_sha256"]:
        raise ValueError("BRANCH program hash differs")
    if value["stop_reason"] not in ("eos", "max_new_tokens") or not value["generated_token_ids"]:
        raise ValueError("BRANCH generation is incomplete")
    if condition == "U" and (not value.get("probability") or
            value["probability"]["argmax_token_id"] != value["probability"]["first_token_id"]):
        raise ValueError("BRANCH U observation is incomplete")
    return value


def execute_branch_pc_generation(*, state, geometry, model_path, execution_root, design_sha256, eos_ids, pad_id):
    """Reuse a state once; atomically publish each position to the shared judge pool."""
    output = execution_root / "runtime/generation" / state["state_id"]
    pending = []
    for index, pair in enumerate(geometry):
        for condition in ("U", "CR", "CM"):
            path = output / f"{index:03d}_{condition}" / "GENERATION.json"
            if load_branch_pc_generation(path, state_id=state["state_id"], pair_id=pair["pair_id"],
                    condition=condition, design_sha256=design_sha256) is None:
                pending.append((pair, condition, path))
    started = time.perf_counter()
    loaded = bool(pending)
    peak = None
    if pending:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from ..train.modeling import configure_determinism
        configure_determinism(FORMAL_GENERATION_SEED)
        torch.set_num_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0")
        if state["adapter_path"] is not None:
            model = PeftModel.from_pretrained(model, state["adapter_path"], is_trainable=False)
            if any(p.dtype != torch.float32 for name, p in model.named_parameters() if "lora_" in name):
                raise ValueError("BRANCH LoRA must remain FP32")
        model.eval()
        for pair, condition, path in pending:
            # A bounded queue shared by all four GPU workers; a few in-flight
            # publications may exceed16, but never the finite worker count.
            while (len(list((execution_root / "runtime/generation").glob("*/*/GENERATION.json"))) -
                   len(list((execution_root / "runtime/scores").glob("*/*.json")))) >= 16:
                time.sleep(0.5)
            record = generate_branch_pc_completion(model, tokenizer, pair, condition,
                                                    device="cuda:0", eos_ids=eos_ids, pad_id=pad_id)
            _atomic_json(path, {"state_id": state["state_id"], "pair_id": pair["pair_id"],
                "group_id": pair["group_id"], "condition": condition, "design_sha256": design_sha256,
                **record})
        peak = torch.cuda.max_memory_reserved()
    result = {"state_id": state["state_id"], "valid_positions": len(geometry)*3,
              "model_loaded": loaded, "generated_this_attempt": len(pending),
              "wall_seconds": time.perf_counter()-started, "peak_reserved_bytes": peak}
    _atomic_json(output / "COMPLETE.json", result)
    return result


def load_scope_pc_panel(path: Path, *, job_id: str, pair_ids: Sequence[str]):
    """Resume an atomically closed panel from this bound job, never a partial list."""
    if not path.exists():
        return None
    value = json.loads(path.read_text("utf-8"))
    if value.get("job_id") != job_id or [r["pair_id"] for r in value.get("records", [])] != list(pair_ids):
        raise ValueError("SCOPE existing panel identity differs")
    checksum = value.get("content_sha256")
    if checksum:
        payload = {k: v for k, v in value.items() if k != "content_sha256"}
        actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if checksum != actual:
            raise ValueError("SCOPE existing panel checksum differs")
    return value


def _write_scope_pc_panel(path: Path, payload: Mapping[str, object]):
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _atomic_json(path, {**payload, "content_sha256": digest})


def execute_scope_pc_evaluation(*, job: Mapping[str, object], pairs: Sequence[Mapping[str, object]],
                                model_path: Path, adapter_path: Path | None, output_dir: Path) -> dict[str, object]:
    """One read-only BASE/adapter process, sharing its probability/NLL/generation slot."""
    if any(pair.get("role") not in {"T64", "D"} for pair in pairs) or not pairs:
        raise ValueError("SCOPE evaluation roles exclude H and ordinary T")
    if not job.get("design_bound"):
        raise ValueError("SCOPE design must be bound before model load")
    step = job["step"]
    if step not in (0, 25, 100) or (step == 0) != (adapter_path is None):
        raise ValueError("SCOPE BASE/adapter step mismatch")
    if step == 25 and (job.get("nll") or job.get("generation_modes")):
        raise ValueError("SCOPE step25 is probability only")
    cached = load_scope_pc_panel(output_dir / "PROBABILITY.json", job_id=job["job_id"], pair_ids=[p["pair_id"] for p in pairs])
    d_pairs = [p for p in pairs if p["role"] == "D"]
    cached_generation = {mode: load_scope_pc_panel(output_dir / f"GENERATION_{mode}.json", job_id=job["job_id"],
        pair_ids=[p["pair_id"] for p in d_pairs]) for mode in job["generation_modes"]}
    if cached:
        if cached["step"] != step or any(not math.isfinite(r["m"]) or abs(r["alias_difference"]) > 1e-6 for r in cached["records"]):
            raise ValueError("SCOPE cached probability contract differs")
        if job["nll"] and any("nll" not in r for r in cached["records"]):
            raise ValueError("SCOPE cached NLL is incomplete")
    if cached and all(cached_generation.values()):
        result = {"job_id": job["job_id"], "step": step, "probability_count": len(pairs),
            "nll_count": len(pairs) if job["nll"] else 0,
            "generation_files": [f"GENERATION_{mode}.json" for mode in job["generation_modes"]],
            "wall_seconds": None, "peak_reserved_bytes": None, "restored_from_closed_panels": True}
        _atomic_json(output_dir / "EVALUATION_RESULT.json", result)
        return result
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from peft import PeftModel
    from ..train.modeling import configure_determinism
    from .codecontests import extract_scope_pc_completion, scope_pc_probability_records

    configure_determinism(int(job.get("evaluation_seed", FORMAL_GENERATION_SEED)))
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = cached["records"] if cached else []
    with torch.inference_mode():
        for pair in ([] if cached else pairs):
            k = pair["token_prefix_length"]
            ids = torch.tensor([pair["token_ids"]["R"][:k]], device="cuda:0")
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
            shared = torch.log_softmax(logits, dim=-1)
            a_r, a_m = pair["token_ids"]["R"][k], pair["token_ids"]["M"][k]
            m = float((shared[a_r] - shared[a_m]).cpu())
            alias = float((shared[a_r].cpu() - shared[a_m].cpu()))
            if not math.isfinite(m) or abs(m - alias) > 1e-6:
                raise RuntimeError("SCOPE probability finite/alias check failed")
            record = {"pair_id": pair["pair_id"], "group_id": pair["group_id"], "role": pair["role"],
                      "m": m, "alias_difference": m - alias}
            del ids, logits, shared
            if job["nll"]:
                values = {}
                for role in ("R", "M"):
                    ids = torch.tensor([pair["token_ids"][role]], device="cuda:0")
                    values[role] = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].cpu()
                    del ids
                record["nll"] = scope_pc_probability_records(pair, values)["nll"]
                del values
            records.append(record)
            _atomic_json(output_dir / "progress.json", {"phase": "probability_nll", "completed": len(records), "total": len(pairs)})
        if not cached:
            _write_scope_pc_panel(output_dir / "PROBABILITY.json", {"job_id": job["job_id"], "step": step, "records": records})
        generation_files = []
        d_pairs = [pair for pair in pairs if pair["role"] == "D"]
        eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=1024,
            eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for mode in job["generation_modes"]:
            destination = output_dir / f"GENERATION_{mode}.json"
            if cached_generation[mode]:
                generation_files.append(destination.name)
                continue
            generated = []
            for pair in d_pairs:
                prefix = pair["code_prefix"] if mode == "prefix" else ""
                input_ids = tokenizer.encode(pair["prompt"] + prefix, add_special_tokens=True)
                expected = pair["prefill_token_ids"] if mode == "prefix" else pair["full_prompt_token_ids"]
                if list(input_ids) != list(expected) or len(input_ids) > 2048:
                    raise ValueError("SCOPE prefill differs from bound whole-string encoding")
                ids = torch.tensor([input_ids], device="cuda:0")
                sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)[0].cpu().tolist()
                result = extract_scope_pc_completion(sequence, input_width=len(input_ids), eos_token_ids=eos_ids,
                    tokenizer=tokenizer, code_prefix=prefix)
                generated.append({"pair_id": pair["pair_id"], "group_id": pair["group_id"], **result})
                del ids
                _atomic_json(output_dir / "progress.json", {"phase": f"generation_{mode}", "completed": len(generated), "total": len(d_pairs)})
            _write_scope_pc_panel(destination, {"job_id": job["job_id"], "mode": mode, "records": generated})
            generation_files.append(destination.name)
    result = {"job_id": job["job_id"], "step": step, "probability_count": len(records),
              "nll_count": len(records) if job["nll"] else 0, "generation_files": generation_files,
              "wall_seconds": time.perf_counter() - started, "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    _atomic_json(output_dir / "EVALUATION_RESULT.json", result)
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("eligibility", "generate", "score"))
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--generation-dir", type=Path)
    parser.add_argument("--retain-generation", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "eligibility":
        if arguments.model is None or arguments.dataset_root is None:
            parser.error("eligibility requires --model and --dataset-root")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(arguments.model), local_files_only=True, trust_remote_code=False
        )
        rows = load_codecontests_sealed_rows(
            arguments.asset, completed_training_jobs=24
        )
        result = freeze_codecontests_endpoint_eligibility(
            rows=rows,
            tokenizer=tokenizer,
            codecontest_fingerprints=load_codecontests_description_fingerprints(
                arguments.dataset_root
            ),
            output_dir=arguments.output,
        )
        print(json.dumps({"schema": result["schema"], "eligible_count": result["eligible_count"]}, sort_keys=True))
        return 0
    if arguments.eligibility is None:
        parser.error("generate and score require --eligibility")
    eligibility = json.loads(arguments.eligibility.read_text("utf-8"))
    members = eligibility.get("members")
    if not isinstance(members, list) or eligibility.get("eligible_count") != len(members):
        raise ValueError("sealed eligibility manifest is malformed")
    rows_by_id = _load_rows_by_members(arguments.asset, members)
    if arguments.command == "generate":
        if arguments.model is None or arguments.adapter is None:
            parser.error("generate requires --model and --adapter")
        result = execute_codecontests_adapter_generation(
            members=members,
            rows_by_id=rows_by_id,
            model_path=arguments.model,
            adapter_path=arguments.adapter,
            output_dir=arguments.output,
        )
    else:
        if arguments.generation_dir is None:
            parser.error("score requires --generation-dir")
        result = execute_codecontests_scoring(
            members=members,
            rows_by_id=rows_by_id,
            generation_dir=arguments.generation_dir,
            output_dir=arguments.output,
            retain_generation=arguments.retain_generation,
        )
    print(json.dumps({key: result[key] for key in result if key.endswith("count") or key in ("pass_fraction", "schema")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
