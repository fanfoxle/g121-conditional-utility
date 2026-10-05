"""CodeContests V1 endpoint contracts without oracle-bearing persistence."""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence


FORBIDDEN_SUMMARY_KEYS = frozenset(
    {
        "output",
        "expected",
        "prompt",
        "tests",
        "inputs",
        "canonical_solution",
        "candidate_solution",
        "base_input",
        "plus_input",
        "raw_token_ids",
        "generated_text",
    }
)
_MODEL_FAILURES = frozenset(
    {"compile_failure", "runtime_failure", "timeout", "wrong_answer"}
)
CODECONTESTS_STOP_STRINGS = (
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n\"\"\"",
    "\nassert",
)


def extract_scope_pc_completion(output_ids, *, input_width: int, eos_token_ids,
                                tokenizer, code_prefix: str) -> dict[str, object]:
    """The SCOPE raw complete-program seam; never invoke historical MBPP extraction."""
    if input_width < 0 or len(output_ids) < input_width:
        raise ValueError("SCOPE padded input width is invalid")
    suffix_ids = [int(value) for value in output_ids[input_width:]]
    eos_position = next((i for i, value in enumerate(suffix_ids) if value in eos_token_ids), None)
    retained = suffix_ids if eos_position is None else suffix_ids[:eos_position]
    suffix = tokenizer.decode(retained, clean_up_tokenization_spaces=False, skip_special_tokens=False)
    return {"source": code_prefix + suffix, "suffix": suffix, "generated_tokens": len(retained),
            "eos_reached": eos_position is not None, "length_capped": len(suffix_ids) >= 1024 and eos_position is None}


def scope_pc_probability_records(pair: Mapping[str, object], logits_by_text: Mapping[str, object]) -> dict[str, object]:
    """FP32 causal log odds and fixed-text shifted-response regional NLL."""
    import torch
    k = int(pair["token_prefix_length"])
    tokens = pair["token_ids"]
    if k <= 0 or k >= min(len(tokens["R"]), len(tokens["M"])):
        raise ValueError("SCOPE next-token index is invalid")
    log_probs = {}
    for role in ("R", "M"):
        logits = logits_by_text[role]
        if logits.ndim != 2 or logits.shape[0] != len(tokens[role]) or not torch.isfinite(logits).all():
            raise ValueError("SCOPE logits shape or finite check failed")
        log_probs[role] = torch.log_softmax(logits.float(), dim=-1)
    # Both alias labels deliberately consume the same shared-context forward.
    shared = log_probs["R"][k - 1]
    odds = shared[tokens["R"][k]] - shared[tokens["M"][k]]
    alias = shared[tokens["R"][k]] - shared[tokens["M"][k]]
    result = {"m": float(odds), "alias_difference": float(odds - alias), "nll": {}}
    for role in ("R", "M"):
        ids = tokens[role]
        start = max(1, int(pair["prompt_tokens"]))
        left, right = pair["difference_envelopes"][role]
        regions = {"all": (start, len(ids)), "prefix": (start, left),
                   "difference": (max(start, left), right), "suffix": (max(start, right), len(ids))}
        values = {}
        for name, (begin, end) in regions.items():
            positions = list(range(begin, end))
            if positions:
                target = torch.tensor([ids[p] for p in positions], device=log_probs[role].device)
                value = -log_probs[role][torch.tensor(positions, device=target.device) - 1, target].sum(dtype=torch.float32).item()
                values[name] = {"mean": value / len(positions), "token_count": len(positions), "sum": value}
            else:
                values[name] = {"mean": None, "token_count": 0, "sum": 0.0}
        result["nll"][role] = values
    return result


def _has_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in FORBIDDEN_SUMMARY_KEYS or _has_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_key(item) for item in value)
    return False


def classify_codecontests_candidate_summary(
    summary: Mapping[str, Any],
) -> dict[str, object]:
    """Collapse one private scorer result into a compact non-truth-bearing record."""

    if _has_forbidden_key(summary):
        raise ValueError("truth-bearing candidate or oracle payload is forbidden")
    task_id = summary.get("task_id") if isinstance(summary, Mapping) else None
    base = {
        "schema": "codecontests-v1-endpoint-record-v1",
        "task_id": task_id,
    }
    if (
        not isinstance(task_id, str)
        or summary.get("schema") != "codecontests-v1-candidate-summary-v1"
        or summary.get("summary_status") != "well_formed"
    ):
        return {
            **base,
            "classification": "infrastructure_failure",
            "failure_subtype": str(
                summary.get("runner_error_subtype", "malformed_candidate_summary")
            ),
        }
    oracle_status = summary.get("oracle_status")
    if oracle_status != "ok":
        return {
            **base,
            "classification": "infrastructure_failure",
            "failure_subtype": f"oracle_{oracle_status}",
        }
    verdict = summary.get("candidate_verdict")
    if verdict == "pass":
        return {**base, "classification": "pass", "failure_subtype": None}
    if verdict in _MODEL_FAILURES:
        subtype = summary.get("candidate_failure_subtype") or verdict
        return {
            **base,
            "classification": "candidate_model_failure",
            "failure_subtype": str(subtype),
        }
    return {
        **base,
        "classification": "infrastructure_failure",
        "failure_subtype": "unknown_candidate_verdict",
    }


def aggregate_codecontests_endpoint(
    requested_task_ids: Sequence[str], records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Aggregate a complete endpoint batch without exposing task content."""

    requested = tuple(requested_task_ids)
    unique_requested = len(requested) == len(set(requested)) and all(
        isinstance(value, str) for value in requested
    )
    by_id: dict[str, Mapping[str, object]] = {}
    malformed = not unique_requested
    for record in records:
        task_id = record.get("task_id") if isinstance(record, Mapping) else None
        classification = record.get("classification") if isinstance(record, Mapping) else None
        if (
            not isinstance(task_id, str)
            or task_id not in requested
            or task_id in by_id
            or classification
            not in {"pass", "candidate_model_failure", "infrastructure_failure"}
            or _has_forbidden_key(record)
        ):
            malformed = True
            continue
        by_id[task_id] = record
    complete = not malformed and tuple(by_id) == requested and len(by_id) == len(requested)
    counts = Counter(
        record["classification"] for record in by_id.values()
    ) if complete else Counter()
    eligible = counts["pass"] + counts["candidate_model_failure"] if complete else 0
    infrastructure = counts["infrastructure_failure"] if complete else len(requested)
    return {
        "schema": "codecontests-v1-endpoint-aggregate-v1",
        "requested_task_count": len(requested),
        "record_count": len(by_id),
        "complete": complete,
        "eligible_count": eligible,
        "pass_count": counts["pass"] if complete else 0,
        "candidate_model_failure_count": counts["candidate_model_failure"] if complete else 0,
        "infrastructure_error_count": infrastructure,
        "endpoint_valid": complete and eligible >= 250 and infrastructure == 0,
        "truth_bearing_payload_persisted": False,
    }


def aggregate_codecontests_qualification(
    requested_task_ids: Sequence[str], records: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Verify the frozen already-open 87-task F0d qualification aggregate."""

    requested = tuple(requested_task_ids)
    by_id: dict[str, Mapping[str, object]] = {}
    malformed = (
        len(requested) != 87
        or len(set(requested)) != 87
        or not all(isinstance(value, str) for value in requested)
    )
    for record in records:
        task_id = record.get("task_id") if isinstance(record, Mapping) else None
        classification = record.get("classification") if isinstance(record, Mapping) else None
        if (
            not isinstance(task_id, str)
            or task_id not in requested
            or task_id in by_id
            or classification
            not in {"pass", "candidate_model_failure", "infrastructure_failure"}
            or _has_forbidden_key(record)
        ):
            malformed = True
            continue
        by_id[task_id] = record
    complete = not malformed and tuple(by_id) == requested and len(by_id) == 87
    counts = Counter(record["classification"] for record in by_id.values()) if complete else Counter()
    return {
        "schema": "codecontests-v1-qualification-aggregate-v1",
        "requested_task_count": len(requested),
        "record_count": len(by_id),
        "complete": complete,
        "pass_count": counts["pass"] if complete else 0,
        "candidate_model_failure_count": counts["candidate_model_failure"] if complete else 0,
        "infrastructure_error_count": counts["infrastructure_failure"] if complete else len(requested),
        "qualified": (
            complete
            and counts["pass"] == 43
            and counts["candidate_model_failure"] == 44
            and counts["infrastructure_failure"] == 0
        ),
        "truth_bearing_payload_persisted": False,
    }


def codecontests_generation_kwargs(
    *, pad_token_id: int, eos_token_id: int
) -> dict[str, int | bool]:
    if (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or pad_token_id < 0
        or isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise ValueError("CodeContests pad/EOS token identities are invalid")
    return {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": 1_024,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }


def extract_codecontests_completion(
    generated_ids: Sequence[int], padded_input_width: int, tokenizer: Any
) -> str:
    """Decode only the generated suffix and apply the frozen no-repair truncation."""

    if padded_input_width < 0 or padded_input_width > len(generated_ids):
        raise ValueError("CodeContests padded input width is invalid")
    text = tokenizer.decode(
        list(generated_ids[padded_input_width:]), skip_special_tokens=True
    )
    if not isinstance(text, str):
        raise TypeError("CodeContests tokenizer decode must return text")
    positions = [text.find(stop) for stop in CODECONTESTS_STOP_STRINGS]
    present = [position for position in positions if position >= 0]
    if present:
        text = text[: min(present)]
    return text.replace("\t", "    ")


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _effective_generated_tokens(token_ids: Sequence[int], eos_token_id: int) -> int:
    ids = [int(value) for value in token_ids]
    try:
        return ids.index(eos_token_id) + 1
    except ValueError:
        return len(ids)


def execute_codecontests_generation_diagnostic(
    *,
    tasks: Sequence[Mapping[str, object]],
    model_path: Path,
    seed: int = 20_260_829,
) -> dict[str, object]:
    """Run the exact base-model generation seam on the open 16-task slice.

    Raw completions exist only in the returned in-memory value so the caller can
    pass them directly to the qualified evaluator and then discard them.
    """

    if len(tasks) != 16:
        raise ValueError("CodeContests diagnostic generation requires 16 tasks")
    normalized: list[dict[str, object]] = []
    for task in tasks:
        task_id = task.get("task_id") if isinstance(task, Mapping) else None
        prompt = task.get("serialized_prompt") if isinstance(task, Mapping) else None
        prompt_tokens = task.get("prompt_tokens") if isinstance(task, Mapping) else None
        if (
            not isinstance(task_id, str)
            or not isinstance(prompt, str)
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
        ):
            raise ValueError("CodeContests diagnostic task is malformed")
        normalized.append(
            {
                "task_id": task_id,
                "serialized_prompt": prompt,
                "prompt_tokens": prompt_tokens,
            }
        )
    if len({task["task_id"] for task in normalized}) != 16:
        raise ValueError("CodeContests diagnostic task identities are not unique")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if (
        type(tokenizer).__name__ != "Qwen2Tokenizer"
        or tokenizer.eos_token_id != 151_643
        or tokenizer.pad_token_id != 151_643
    ):
        raise RuntimeError("CodeContests generation tokenizer identity mismatch")
    tokenizer.padding_side = "left"
    model = None
    generated_rows: list[dict[str, object]] = []
    batch_metrics: list[dict[str, object]] = []
    process_started = time.perf_counter()
    load_seconds = 0.0
    peak_reserved = 0
    try:
        load_started = time.perf_counter()
        model = Qwen2ForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda:0")
        model.eval()
        torch.cuda.synchronize(0)
        load_seconds = time.perf_counter() - load_started
        for batch_index, start in enumerate(range(0, len(normalized), 8)):
            batch = normalized[start : start + 8]
            encoded = tokenizer(
                [str(task["serialized_prompt"]) for task in batch],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            prompt_counts = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
            if prompt_counts != [task["prompt_tokens"] for task in batch]:
                raise RuntimeError("CodeContests generation prompt token count mismatch")
            padded_width = int(encoded["input_ids"].shape[1])
            encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
            torch.cuda.reset_peak_memory_stats(0)
            batch_started = time.perf_counter()
            kwargs = codecontests_generation_kwargs(
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=int(tokenizer.eos_token_id),
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    **kwargs,
                    stop_strings=list(CODECONTESTS_STOP_STRINGS),
                    tokenizer=tokenizer,
                )
            torch.cuda.synchronize(0)
            generated_cpu = generated.detach().cpu().tolist()
            for offset, sequence in enumerate(generated_cpu):
                suffix = sequence[padded_width:]
                completion = extract_codecontests_completion(
                    sequence, padded_width, tokenizer
                )
                generated_rows.append(
                    {
                        "task_id": batch[offset]["task_id"],
                        "completion": completion,
                        "completion_sha256": hashlib.sha256(
                            completion.encode("utf-8")
                        ).hexdigest(),
                        "generated_tokens": _effective_generated_tokens(
                            suffix, int(tokenizer.eos_token_id)
                        ),
                    }
                )
            batch_peak = int(torch.cuda.max_memory_reserved(0))
            peak_reserved = max(peak_reserved, batch_peak)
            batch_metrics.append(
                {
                    "batch_index": batch_index,
                    "size": len(batch),
                    "padded_input_width": padded_width,
                    "wall_seconds": time.perf_counter() - batch_started,
                    "peak_reserved_bytes": batch_peak,
                }
            )
            del encoded, generated, generated_cpu
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
    compact_rows = [
        {
            "task_id": row["task_id"],
            "completion_sha256": row["completion_sha256"],
            "generated_tokens": row["generated_tokens"],
        }
        for row in generated_rows
    ]
    return {
        "schema": "codecontests-v1-diagnostic-generation-v1",
        "formal": False,
        "seed": seed,
        "task_count": len(generated_rows),
        "tasks": generated_rows,
        "output_sha256": _canonical_json_sha256(compact_rows),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in generated_rows),
        "load_seconds": load_seconds,
        "batch_metrics": batch_metrics,
        "peak_reserved_bytes": peak_reserved,
        "process_seconds": time.perf_counter() - process_started,
        "model_unloaded_and_cache_cleared": True,
    }


def compact_codecontests_generation_result(
    result: Mapping[str, object]
) -> dict[str, object]:
    """Remove ephemeral completion text while preserving deterministic evidence."""

    tasks = result.get("tasks") if isinstance(result, Mapping) else None
    if not isinstance(tasks, list) or len(tasks) != 16:
        raise ValueError("CodeContests diagnostic generation result is incomplete")
    compact_tasks: list[dict[str, object]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("CodeContests diagnostic generation task is malformed")
        completion = task.get("completion")
        digest = task.get("completion_sha256")
        if (
            not isinstance(completion, str)
            or not isinstance(digest, str)
            or hashlib.sha256(completion.encode("utf-8")).hexdigest() != digest
        ):
            raise ValueError("CodeContests diagnostic completion hash mismatch")
        compact_tasks.append(
            {
                "task_id": task.get("task_id"),
                "completion_sha256": digest,
                "generated_tokens": task.get("generated_tokens"),
            }
        )
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"tasks"}
    }
    compact["tasks"] = compact_tasks
    compact["raw_completion_persisted"] = False
    if compact.get("output_sha256") != _canonical_json_sha256(compact_tasks):
        raise ValueError("CodeContests diagnostic output digest mismatch")
    return compact
