"""NAT-REPLICA-X164 formal freeze, authorization, finite queue, and CLI.

This module owns only the successor's formal execution boundary.  Training,
generation, scoring, and analysis remain in their stable domain modules.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from typing import Mapping, Sequence
import zipfile


PROTOCOL_ID = "codecontests-nat-replica-x164-v1"
AUTHORIZATION_PHRASE = "AUTHORIZE_NAT_REPLICA_X164_FORMAL_RUN_42H"
TARGET_WINDOW_HOURS = 20.0
ABSOLUTE_WALL_CAP_HOURS = 42.0
RESOURCE_AMENDMENT_CONFIRMATION = "确认这一项资源修订"
CONCURRENCY = {"train": 2, "generation": 4, "score": 4}
TERMINALS = frozenset(
    {
        "COMPLETED_NAT_REPLICA_X164_MEASUREMENT_OR_INFORMATION_INSUFFICIENT",
        "COMPLETED_NAT_REPLICA_X164_POSITIVE_EXTERNAL_EXPLORATORY",
        "COMPLETED_NAT_REPLICA_X164_EXTERNAL_ADVERSE",
        "COMPLETED_NAT_REPLICA_X164_BELOW_INVESTMENT_SCALE",
        "COMPLETED_NAT_REPLICA_X164_EXTERNAL_UNRESOLVED",
        "INCOMPLETE_NAT_REPLICA_X164_42H_WALL_CAP",
        "STOPPED_NAT_REPLICA_X164_SCIENTIFIC_DRIFT",
        "STOPPED_NAT_REPLICA_X164_SAFETY_OR_PERMISSION",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    os.replace(temporary, path)


def nat_replica_state_ids(roots: Sequence[int]) -> tuple[str, ...]:
    if (
        len(roots) != 8
        or len(set(roots)) != 8
        or any(type(root) is not int or not 0 < root < 2_147_483_647 for root in roots)
    ):
        raise ValueError("NAT-REPLICA requires eight unique positive roots")
    states: list[str] = []
    for index, root in enumerate(roots):
        arms = ("POL_KEEP", "POL_FILTER") if index % 2 == 0 else (
            "POL_FILTER",
            "POL_KEEP",
        )
        states.extend(f"{root}_{arm}" for arm in arms)
    return tuple(states)


def nat_replica_formal_jobs(roots: Sequence[int]) -> tuple[dict[str, str], ...]:
    states = nat_replica_state_ids(roots)
    return tuple(
        [{"kind": "train", "job_id": state} for state in states]
        + [
            {"kind": "generation", "job_id": state}
            for state in ("BASE", *states)
        ]
        + [{"kind": "score", "job_id": state} for state in ("BASE", *states)]
    )


def next_nat_replica_formal_jobs(
    roots: Sequence[int],
    *,
    completed: set[tuple[str, str]],
    active: set[tuple[str, str]],
    concurrency: Mapping[str, int] = CONCURRENCY,
) -> tuple[dict[str, str], ...]:
    """Select the next fixed jobs without consulting outcomes."""

    jobs = nat_replica_formal_jobs(roots)
    identities = {(job["kind"], job["job_id"]) for job in jobs}
    if not completed <= identities or not active <= identities:
        raise ValueError("NAT-REPLICA queue contains a foreign job")
    if set(concurrency) != set(CONCURRENCY) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= CONCURRENCY[kind]
        for kind in CONCURRENCY
    ):
        raise ValueError("NAT-REPLICA runtime concurrency exceeds 2/4/4 ceilings")
    training = jobs[:16]
    missing = [
        job
        for job in training
        if (job["kind"], job["job_id"]) not in completed | active
    ]
    if missing:
        if any(kind != "train" for kind, _job_id in active):
            return ()
        first = missing[0]
        pair_root = first["job_id"].split("_", 1)[0]
        slots = concurrency["train"] - sum(kind == "train" for kind, _ in active)
        return tuple(
            job
            for job in missing
            if job["job_id"].split("_", 1)[0] == pair_root
        )[: max(0, slots)]
    if any(kind == "train" for kind, _job_id in active):
        return ()
    selected: list[dict[str, str]] = []
    for kind in ("generation", "score"):
        slots = concurrency[kind] - sum(active_kind == kind for active_kind, _ in active)
        if slots <= 0:
            continue
        for job in jobs:
            if slots <= 0:
                break
            identity = (job["kind"], job["job_id"])
            if job["kind"] != kind or identity in completed | active:
                continue
            if kind == "score" and ("generation", job["job_id"]) not in completed:
                continue
            selected.append(job)
            slots -= 1
    return tuple(selected)


def build_nat_replica_formal_freeze(
    *,
    roots: Sequence[int],
    root_seed_sha256: str,
    historical_roots_sha256: str,
    plans: Sequence[Mapping[str, object]],
    task_ids: Sequence[str],
    bindings: Mapping[str, object],
    paths: Mapping[str, object],
    python_gpu: str,
    python_judge: str,
) -> dict[str, object]:
    """Create the immutable pre-authorization scientific contract in memory."""

    roots = tuple(roots)
    states = nat_replica_state_ids(roots)
    if not _is_sha256(root_seed_sha256) or not _is_sha256(historical_roots_sha256):
        raise ValueError("NAT-REPLICA root provenance SHA-256 is invalid")
    expected_plans = [
        (int(state.split("_", 1)[0]), state.split("_", 1)[1]) for state in states
    ]
    actual_plans = [(plan.get("root"), plan.get("arm")) for plan in plans]
    if actual_plans != expected_plans:
        raise ValueError("NAT-REPLICA plan order differs from the fixed paired queue")
    for plan in plans:
        if (
            not _is_sha256(plan.get("plan_sha256"))
            or plan.get("updates") != 100
            or plan.get("response_tokens") != 583_975
            or not isinstance(plan.get("plan_path"), str)
        ):
            raise ValueError("NAT-REPLICA plan recipe or identity differs")
    expected_tasks = [f"HumanEval/{index}" for index in range(164)]
    if list(task_ids) != expected_tasks:
        raise ValueError("NAT-REPLICA endpoint must be HumanEval/0..163 in order")
    required_bindings = {
        "model_revision",
        "humaneval_compressed_sha256",
        "humaneval_logical_sha256",
        "evaluator_revision",
    }
    if not required_bindings <= set(bindings) or any(
        not isinstance(bindings[key], str) or not bindings[key]
        for key in required_bindings
    ):
        raise ValueError("NAT-REPLICA authority bindings are incomplete")
    freeze: dict[str, object] = {
        "schema": "nat-replica-x164-formal-freeze-v1",
        "protocol_id": PROTOCOL_ID,
        "state": "prepared_not_authorized",
        "formal_training_authorized": False,
        "formal_outcomes_opened": False,
        "root_provenance": {
            "method": "one-time local cryptographic draw then SHA-256 derivation",
            "outcome_blind": True,
            "seed_sha256": root_seed_sha256,
            "historical_roots_sha256": historical_roots_sha256,
        },
        "roots": list(roots),
        "plans": [dict(plan) for plan in plans],
        "task_ids": list(task_ids),
        "training_job_count": 16,
        "state_count": 17,
        "position_count": 2_788,
        "concurrency": dict(CONCURRENCY),
        "clock": {
            "target_window_hours": TARGET_WINDOW_HOURS,
            "absolute_wall_cap_hours": ABSOLUTE_WALL_CAP_HOURS,
            "starts_at": "formal_cloud_preflight_after_authorization",
            "target_is_stop_rule": False,
            "hard_cap_resets": False,
        },
        "resource_scenarios_hours": {
            "empirical_typical": 9.13599,
            "empirical_high": 12.68181,
            "disaster_envelope": 39.66704,
        },
        "training_recipe": {
            "response_tokens": 583_975,
            "background_tokens": 467_180,
            "component_tokens": 116_795,
            "updates": 100,
            "max_length": 2_048,
            "microbatch": 1,
        },
        "bindings": dict(bindings),
        "paths": dict(paths),
        "runtime": {
            "python_gpu": str(python_gpu),
            "python_judge": str(python_judge),
            "environment": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        },
        "job_order": list(nat_replica_formal_jobs(roots)),
        "terminal_set": sorted(TERMINALS),
        "authorization_phrase_sha256": hashlib.sha256(
            AUTHORIZATION_PHRASE.encode("ascii")
        ).hexdigest(),
    }
    freeze["scientific_identity_sha256"] = _canonical_sha256(freeze)
    return validate_nat_replica_formal_freeze(freeze)


def validate_nat_replica_formal_freeze(
    freeze: Mapping[str, object],
) -> Mapping[str, object]:
    """Fail closed before dry-run, formal claim, or any model work."""

    value = dict(freeze)
    identity = value.pop("scientific_identity_sha256", None)
    roots = value.get("roots")
    plans = value.get("plans")
    task_ids = value.get("task_ids")
    bindings = value.get("bindings")
    provenance = value.get("root_provenance")
    expected_clock = {
        "target_window_hours": TARGET_WINDOW_HOURS,
        "absolute_wall_cap_hours": ABSOLUTE_WALL_CAP_HOURS,
        "starts_at": "formal_cloud_preflight_after_authorization",
        "target_is_stop_rule": False,
        "hard_cap_resets": False,
    }
    expected_resources = {
        "empirical_typical": 9.13599,
        "empirical_high": 12.68181,
        "disaster_envelope": 39.66704,
    }
    expected_training = {
        "response_tokens": 583_975,
        "background_tokens": 467_180,
        "component_tokens": 116_795,
        "updates": 100,
        "max_length": 2_048,
        "microbatch": 1,
    }
    expected_tasks = [f"HumanEval/{index}" for index in range(164)]
    core_bindings = {
        "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "humaneval_compressed_sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "humaneval_logical_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
        "evaluator_revision": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
    }
    expected_plans = (
        [
            (int(state.split("_", 1)[0]), state.split("_", 1)[1])
            for state in nat_replica_state_ids(roots or [])
        ]
        if isinstance(roots, list)
        else []
    )
    plan_items = (
        plans
        if isinstance(plans, list) and all(isinstance(plan, dict) for plan in plans)
        else []
    )
    actual_plans = (
        [(plan.get("root"), plan.get("arm")) for plan in plans]
        if isinstance(plans, list) and all(isinstance(plan, dict) for plan in plans)
        else []
    )
    valid = (
        freeze.get("schema") == "nat-replica-x164-formal-freeze-v1"
        and freeze.get("protocol_id") == PROTOCOL_ID
        and freeze.get("state") == "prepared_not_authorized"
        and freeze.get("formal_training_authorized") is False
        and freeze.get("formal_outcomes_opened") is False
        and isinstance(roots, list)
        and freeze.get("training_job_count") == 16
        and freeze.get("state_count") == 17
        and freeze.get("position_count") == 2_788
        and freeze.get("concurrency") == CONCURRENCY
        and freeze.get("clock") == expected_clock
        and freeze.get("resource_scenarios_hours") == expected_resources
        and freeze.get("training_recipe") == expected_training
        and task_ids == expected_tasks
        and actual_plans == expected_plans
        and all(
            _is_sha256(plan.get("plan_sha256"))
            and plan.get("updates") == 100
            and plan.get("response_tokens") == 583_975
            and isinstance(plan.get("plan_path"), str)
            for plan in plan_items
        )
        and isinstance(bindings, dict)
        and all(bindings.get(key) == expected for key, expected in core_bindings.items())
        and isinstance(provenance, dict)
        and provenance.get("outcome_blind") is True
        and _is_sha256(provenance.get("seed_sha256"))
        and _is_sha256(provenance.get("historical_roots_sha256"))
        and freeze.get("authorization_phrase_sha256")
        == hashlib.sha256(AUTHORIZATION_PHRASE.encode("ascii")).hexdigest()
        and freeze.get("job_order") == list(nat_replica_formal_jobs(roots or []))
        and freeze.get("terminal_set") == sorted(TERMINALS)
        and _is_sha256(identity)
        and identity == _canonical_sha256(value)
    )
    if not valid:
        raise ValueError("NAT-REPLICA prepared freeze identity is invalid")
    return freeze


def dry_run_nat_replica_formal(freeze: Mapping[str, object]) -> dict[str, object]:
    """Validate the formal command without importing CUDA or creating outputs."""

    validate_nat_replica_formal_freeze(freeze)
    return {
        "schema": "nat-replica-x164-formal-dry-run-v1",
        "valid": True,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "jobs": len(freeze["job_order"]),
        "cuda_initialized": False,
        "optimizer_initialized": False,
        "formal_outcomes_opened": False,
        "formal_training_started": False,
    }


def claim_nat_replica_formal_run(
    freeze: Mapping[str, object],
    *,
    authorization: str,
    execution_root: Path,
    started_utc: datetime | None = None,
) -> dict[str, object]:
    """Atomically claim one prepared freeze after exact user authorization."""

    validate_nat_replica_formal_freeze(freeze)
    if authorization != AUTHORIZATION_PHRASE:
        raise PermissionError("NAT-REPLICA requires the exact authorization phrase")
    execution_root = Path(execution_root)
    claim_path = execution_root / "FORMAL_CLAIM.json"
    if claim_path.is_file():
        claim = json.loads(claim_path.read_text("utf-8"))
        if claim.get("scientific_identity_sha256") != freeze["scientific_identity_sha256"]:
            raise ValueError("NAT-REPLICA existing formal claim belongs to another freeze")
        return claim
    if execution_root.exists() and any(execution_root.iterdir()):
        raise FileExistsError("NAT-REPLICA formal execution root is not empty")
    began = started_utc or datetime.now(timezone.utc)
    if began.tzinfo is None or began.utcoffset() is None:
        raise ValueError("NAT-REPLICA formal clock must be timezone-aware")
    began = began.astimezone(timezone.utc)
    claim = {
        "schema": "nat-replica-x164-formal-claim-v1",
        "protocol_id": PROTOCOL_ID,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "started_utc": began.isoformat(),
        "target_utc": (began + timedelta(hours=TARGET_WINDOW_HOURS)).isoformat(),
        "hard_deadline_utc": (
            began + timedelta(hours=ABSOLUTE_WALL_CAP_HOURS)
        ).isoformat(),
        "hard_cap_resets": False,
        "authorization_phrase_sha256": hashlib.sha256(
            authorization.encode("ascii")
        ).hexdigest(),
    }
    _atomic_json(claim_path, claim)
    return claim


def record_nat_replica_soft_cap_amendment(
    freeze: Mapping[str, object],
    *,
    execution_root: Path,
    confirmation: str,
    confirmed_utc: datetime | None = None,
) -> dict[str, object]:
    """Record the explicit post-claim resource amendment without changing science."""

    validate_nat_replica_formal_freeze(freeze)
    execution_root = Path(execution_root)
    claim = _load_claim(execution_root, freeze)
    path = execution_root / "RESOURCE_AMENDMENT.json"
    if path.is_file():
        amendment = json.loads(path.read_text("utf-8"))
        if (
            amendment.get("schema") != "nat-replica-x164-resource-amendment-v1"
            or amendment.get("scientific_identity_sha256")
            != freeze["scientific_identity_sha256"]
        ):
            raise ValueError("NAT-REPLICA resource amendment belongs to another claim")
        return amendment
    if confirmation != RESOURCE_AMENDMENT_CONFIRMATION:
        raise PermissionError("NAT-REPLICA requires the exact resource amendment confirmation")
    confirmed = confirmed_utc or datetime.now(timezone.utc)
    if confirmed.tzinfo is None or confirmed.utcoffset() is None:
        raise ValueError("NAT-REPLICA resource amendment time must be timezone-aware")
    amendment = {
        "schema": "nat-replica-x164-resource-amendment-v1",
        "protocol_id": PROTOCOL_ID,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "claim_started_utc": claim["started_utc"],
        "prior_absolute_wall_cap_hours": ABSOLUTE_WALL_CAP_HOURS,
        "soft_cap_authorized": True,
        "completion_required": True,
        "actual_wall_clock_recorded_from_claim": True,
        "confirmation": confirmation,
        "confirmed_utc": confirmed.astimezone(timezone.utc).isoformat(),
    }
    _atomic_json(path, amendment)
    return amendment


def load_nat_replica_soft_cap_amendment(
    execution_root: Path, freeze: Mapping[str, object]
) -> dict[str, object] | None:
    """Load and validate the explicit soft-cap amendment, if present."""

    path = Path(execution_root) / "RESOURCE_AMENDMENT.json"
    if not path.is_file():
        return None
    amendment = json.loads(path.read_text("utf-8"))
    claim = _load_claim(Path(execution_root), freeze)
    if not (
        amendment.get("schema") == "nat-replica-x164-resource-amendment-v1"
        and amendment.get("protocol_id") == PROTOCOL_ID
        and amendment.get("scientific_identity_sha256")
        == freeze["scientific_identity_sha256"]
        and amendment.get("claim_started_utc") == claim["started_utc"]
        and amendment.get("prior_absolute_wall_cap_hours") == ABSOLUTE_WALL_CAP_HOURS
        and amendment.get("soft_cap_authorized") is True
        and amendment.get("completion_required") is True
        and amendment.get("actual_wall_clock_recorded_from_claim") is True
        and amendment.get("confirmation") == RESOURCE_AMENDMENT_CONFIRMATION
    ):
        raise ValueError("NAT-REPLICA resource amendment is invalid")
    return amendment


def resolve_nat_replica_terminal(
    *,
    complete: bool = False,
    root_sd: float | None = None,
    mean_delta_pp: float | None = None,
    ci_lower_pp: float | None = None,
    ci_upper_pp: float | None = None,
    hard_cap_reached: bool = False,
    target_window_reached: bool = False,
    scientific_drift: bool = False,
    safety_or_permission_stop: bool = False,
) -> str:
    """Resolve the preregistered closed terminal set."""

    if safety_or_permission_stop:
        return "STOPPED_NAT_REPLICA_X164_SAFETY_OR_PERMISSION"
    if scientific_drift:
        return "STOPPED_NAT_REPLICA_X164_SCIENTIFIC_DRIFT"
    if hard_cap_reached and not complete:
        return "INCOMPLETE_NAT_REPLICA_X164_42H_WALL_CAP"
    if target_window_reached and not complete:
        raise ValueError("20h target is not a terminal or stop rule")
    if not complete:
        raise ValueError("NAT-REPLICA terminal facts are incomplete")
    if (
        root_sd is None
        or mean_delta_pp is None
        or ci_lower_pp is None
        or ci_upper_pp is None
    ):
        raise ValueError("complete NAT-REPLICA requires effect and interval")
    if root_sd == 0:
        return "COMPLETED_NAT_REPLICA_X164_MEASUREMENT_OR_INFORMATION_INSUFFICIENT"
    if ci_upper_pp < 0:
        return "COMPLETED_NAT_REPLICA_X164_EXTERNAL_ADVERSE"
    if mean_delta_pp >= 2.0 and ci_lower_pp > 0:
        return "COMPLETED_NAT_REPLICA_X164_POSITIVE_EXTERNAL_EXPLORATORY"
    if ci_upper_pp < 2.0:
        return "COMPLETED_NAT_REPLICA_X164_BELOW_INVESTMENT_SCALE"
    return "COMPLETED_NAT_REPLICA_X164_EXTERNAL_UNRESOLVED"


def _load_prepared(prepared_root: Path) -> dict[str, object]:
    path = Path(prepared_root) / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
    freeze = json.loads(path.read_text("utf-8"))
    validate_nat_replica_formal_freeze(freeze)
    validate_nat_replica_prepared_manifest(Path(prepared_root), freeze)
    return freeze


def refresh_nat_replica_prepared_manifest(
    prepared_root: Path, freeze: Mapping[str, object]
) -> dict[str, object]:
    """Bind the exact deployable staging closure without credentials or caches."""

    prepared_root = Path(prepared_root)
    validate_nat_replica_formal_freeze(freeze)
    files: dict[str, str] = {}
    forbidden_parts = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
    for path in sorted(prepared_root.rglob("*")):
        if not path.is_file() or path.name == "PREPARED_MANIFEST.json":
            continue
        relative = path.relative_to(prepared_root)
        if forbidden_parts & set(relative.parts) or path.suffix in {".pyc", ".pyo"}:
            raise ValueError("NAT-REPLICA prepared staging contains a cache")
        lowered = relative.as_posix().lower()
        if any(word in lowered for word in ("cloud.txt", "credential", ".env")):
            raise ValueError("NAT-REPLICA prepared staging contains credential material")
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "nat-replica-x164-prepared-manifest-v1",
        "state": "prepared_not_authorized",
        "formal_training_authorized": False,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "files": files,
    }
    _atomic_json(prepared_root / "PREPARED_MANIFEST.json", manifest)
    return manifest


def validate_nat_replica_prepared_manifest(
    prepared_root: Path, freeze: Mapping[str, object]
) -> Mapping[str, object]:
    manifest_path = Path(prepared_root) / "PREPARED_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if not (
        manifest.get("schema") == "nat-replica-x164-prepared-manifest-v1"
        and manifest.get("state") == "prepared_not_authorized"
        and manifest.get("formal_training_authorized") is False
        and manifest.get("scientific_identity_sha256")
        == freeze["scientific_identity_sha256"]
        and isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("NAT-REPLICA prepared manifest identity differs")
    for relative, expected in manifest["files"].items():
        path = (Path(prepared_root) / relative).resolve()
        if not path.is_relative_to(Path(prepared_root).resolve()) or not path.is_file():
            raise ValueError("NAT-REPLICA prepared manifest path is missing or escaped")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("NAT-REPLICA prepared manifest file hash differs")
    actual: set[str] = set()
    forbidden_parts = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
    for path in Path(prepared_root).rglob("*"):
        if not path.is_file() or path.name == "PREPARED_MANIFEST.json":
            continue
        relative = path.relative_to(prepared_root)
        if forbidden_parts & set(relative.parts) or path.suffix in {".pyc", ".pyo"}:
            raise ValueError("NAT-REPLICA prepared staging contains a cache")
        actual.add(relative.as_posix())
    if actual != set(manifest["files"]):
        raise ValueError("NAT-REPLICA prepared manifest membership differs")
    return manifest


def _load_claim(execution_root: Path, freeze: Mapping[str, object]) -> dict[str, object]:
    path = Path(execution_root) / "FORMAL_CLAIM.json"
    claim = json.loads(path.read_text("utf-8"))
    if not (
        claim.get("schema") == "nat-replica-x164-formal-claim-v1"
        and claim.get("scientific_identity_sha256")
        == freeze["scientific_identity_sha256"]
        and claim.get("hard_cap_resets") is False
    ):
        raise ValueError("NAT-REPLICA formal claim identity differs")
    return claim


def _training_card(
    freeze: Mapping[str, object], entry: Mapping[str, object]
) -> dict[str, object]:
    from ..protocol import CODECONTESTS_ROLE_NAMES, derive_codecontests_role_seed

    root = int(entry["root"])
    arm = str(entry["arm"])
    card: dict[str, object] = {
        "schema": "codecontests-nat-replica-x164-run-card-v1",
        "protocol_id": PROTOCOL_ID,
        "design_id": "CC_NAT_REPLICA_X164",
        "design_sha256": freeze["scientific_identity_sha256"],
        "stage": "POLICY",
        "role": "exploratory_external_replication",
        "root": root,
        "arm_ordinal": ("POL_KEEP", "POL_FILTER").index(arm),
        "scientific_arm": arm,
        "response_token_quota": 583_975,
        "updates": 100,
        "max_length": 2_048,
        "packing_path": "single_unit",
        "role_seeds": {
            role: derive_codecontests_role_seed(root, role)
            for role in CODECONTESTS_ROLE_NAMES
        },
        "outcomes_opened": False,
        "plan_sha256": entry["plan_sha256"],
    }
    card["card_sha256"] = _canonical_sha256(card)
    return card


def execute_nat_replica_formal_worker(
    *, prepared_root: Path, execution_root: Path, kind: str, job_id: str
) -> dict[str, object]:
    """Execute one bound formal job after checking the one immutable clock."""

    prepared_root = Path(prepared_root)
    execution_root = Path(execution_root)
    freeze = _load_prepared(prepared_root)
    claim = _load_claim(execution_root, freeze)
    amendment = load_nat_replica_soft_cap_amendment(execution_root, freeze)
    if amendment is None and datetime.now(timezone.utc) >= datetime.fromisoformat(
        str(claim["hard_deadline_utc"])
    ):
        raise RuntimeError("NAT-REPLICA 42h hard deadline is closed")
    if (kind, job_id) not in {
        (job["kind"], job["job_id"]) for job in freeze["job_order"]
    }:
        raise ValueError("NAT-REPLICA worker job is outside the fixed queue")

    tasks = None
    entry = None
    adapter_path = None
    if job_id != "BASE":
        entry = next(
            plan
            for plan in freeze["plans"]
            if f"{plan['root']}_{plan['arm']}" == job_id
        )
        card = _training_card(freeze, entry)
        if kind != "train":
            from ..train.run_codecontests import validate_codecontests_s12_terminal

            terminal_path = (
                execution_root
                / "runtime"
                / "training"
                / job_id
                / "TERMINAL_TRAINING_RESULT.json"
            )
            validate_codecontests_s12_terminal(
                terminal_path,
                expected_card_sha256=str(card["card_sha256"]),
                expected_plan_sha256=str(entry["plan_sha256"]),
                expected_root=int(entry["root"]),
                expected_protocol_id=PROTOCOL_ID,
                expected_arm=str(entry["arm"]),
            )
            adapter_path = terminal_path.parent / "adapter100"
    if kind == "train":
        from ..arms.paired_filtering import load_s12_training_plan
        from ..train.run_codecontests import execute_codecontests_s12_formal

        assert entry is not None
        plan_path = Path(str(entry["plan_path"]))
        if not plan_path.is_absolute():
            plan_path = prepared_root / plan_path
        output = execution_root / "runtime" / "training" / job_id
        return execute_codecontests_s12_formal(
            plan=load_s12_training_plan(plan_path),
            card=_training_card(freeze, entry),
            model_path=Path(str(freeze["paths"]["model"])),
            output_dir=output,
            resume=(output / "recovery" / "CURRENT.json").exists()
            or (output / "TERMINAL_TRAINING_RESULT.json").exists(),
        )

    from ..eval.humaneval import (
        execute_humaneval_formal_generation,
        execute_humaneval_formal_scoring,
        load_humaneval_plus_asset,
    )

    tasks = load_humaneval_plus_asset(Path(str(freeze["paths"]["asset"])))
    if [task["task_id"] for task in tasks] != freeze["task_ids"]:
        raise ValueError("NAT-REPLICA HumanEval task order differs from freeze")
    if kind == "generation":
        return execute_humaneval_formal_generation(
            state_id=job_id,
            tasks=tasks,
            model_path=Path(str(freeze["paths"]["model"])),
            adapter_path=adapter_path,
            output_dir=execution_root / "runtime" / "generation" / job_id,
            design_sha256=str(freeze["scientific_identity_sha256"]),
        )
    return execute_humaneval_formal_scoring(
        state_id=job_id,
        tasks=tasks,
        generation_dir=execution_root / "runtime" / "generation" / job_id,
        output_dir=execution_root / "runtime" / "scores" / job_id,
        design_sha256=str(freeze["scientific_identity_sha256"]),
        evaluator_source_root=Path(str(freeze["paths"]["evaluator_source"])),
        evaluator_archive=Path(str(freeze["paths"]["evaluator_archive"])),
        sandbox_base=execution_root / "runtime" / "sandboxes" / job_id,
        workers=4,
        python_executable=Path(str(freeze["runtime"]["python_judge"])),
    )


def _completed_formal_jobs(
    freeze: Mapping[str, object], execution_root: Path
) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    runtime = Path(execution_root) / "runtime"
    for job in freeze["job_order"]:
        kind, job_id = job["kind"], job["job_id"]
        if kind == "train":
            path = runtime / "training" / job_id / "TERMINAL_TRAINING_RESULT.json"
            closed = path.is_file()
        else:
            path = runtime / ("generation" if kind == "generation" else "scores") / job_id / "COMPLETE.json"
            if path.is_file():
                value = json.loads(path.read_text("utf-8"))
                closed = (
                    value.get("completed_positions") == 164
                    and value.get("design_sha256")
                    == freeze["scientific_identity_sha256"]
                    and (kind != "score" or value.get("valid_positions") == 164)
                )
            else:
                closed = False
        if closed:
            completed.add((kind, job_id))
    return completed


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_nat_replica_formal(
    *,
    prepared_root: Path,
    execution_root: Path,
    authorization: str,
) -> dict[str, object]:
    """Run the one finite formal queue; only counts are visible before closure."""

    formal_invoked_utc = datetime.now(timezone.utc)
    prepared_root = Path(prepared_root)
    execution_root = Path(execution_root)
    freeze = _load_prepared(prepared_root)
    claim = claim_nat_replica_formal_run(
        freeze,
        authorization=authorization,
        execution_root=execution_root,
        started_utc=formal_invoked_utc,
    )
    amendment = load_nat_replica_soft_cap_amendment(execution_root, freeze)
    runtime_root = execution_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    progress_path = runtime_root / "progress.json"
    previous = json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
    attempts = list(previous.get("attempts", []))
    effective = dict(previous.get("effective_concurrency", freeze["concurrency"]))
    completed = _completed_formal_jobs(freeze, execution_root)
    active: dict[tuple[str, str], tuple[subprocess.Popen, dict[str, object]]] = {}
    status = "RUNNING"
    error = None
    jobs = {(job["kind"], job["job_id"]) for job in freeze["job_order"]}
    began = datetime.fromisoformat(str(claim["started_utc"]))
    target = datetime.fromisoformat(str(claim["target_utc"]))
    deadline = (
        None
        if amendment is not None
        else datetime.fromisoformat(str(claim["hard_deadline_utc"]))
    )

    def snapshot() -> dict[str, object]:
        now = datetime.now(timezone.utc)
        return {
            "schema": "nat-replica-x164-formal-progress-v1",
            "state": status,
            "scientific_identity_sha256": freeze["scientific_identity_sha256"],
            "completed_jobs": len(completed),
            "completed_training": sum(kind == "train" for kind, _ in completed),
            "completed_generation_states": sum(
                kind == "generation" for kind, _ in completed
            ),
            "completed_score_states": sum(kind == "score" for kind, _ in completed),
            "generated_positions": sum(
                1 for _ in (runtime_root / "generation").glob("*/[0-9][0-9][0-9].json")
            ),
            "scored_positions": sum(
                1 for _ in (runtime_root / "scores").glob("*/[0-9][0-9][0-9].json")
            ),
            "active": [record for _process, record in active.values()],
            "attempts": attempts,
            "effective_concurrency": effective,
            "target_reached": now >= target,
            "hard_deadline_utc": None if deadline is None else deadline.isoformat(),
            "soft_cap_amendment": amendment is not None,
            "elapsed_seconds": (now - began).total_seconds(),
            "formal_outcomes_opened": status in {"FIXED_WORK_COMPLETE", "HARD_CAP_REACHED"},
            "error": error,
        }

    try:
        while True:
            now = datetime.now(timezone.utc)
            if deadline is not None and now >= deadline:
                status = "HARD_CAP_REACHED"
                break
            for identity, (process, record) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del active[identity]
                record["ended_utc"] = now.isoformat()
                record["exit_code"] = returncode
                if returncode:
                    log_tail = Path(str(record["log"])).read_text(
                        "utf-8", errors="replace"
                    )[-4_000:]
                    error = log_tail
                    kind = identity[0]
                    if "out of memory" in log_tail.lower() and effective[kind] > 1:
                        effective[kind] = max(1, effective[kind] // 2)
                        status = "RESOURCE_DOWNSHIFT_RESTART_REQUIRED"
                    else:
                        status = "REPAIR_REQUIRED"
                    break
                completed.add(identity)
            if status != "RUNNING":
                break
            if completed == jobs:
                status = "FIXED_WORK_COMPLETE"
                break
            for job in next_nat_replica_formal_jobs(
                freeze["roots"],
                completed=completed,
                active=set(active),
                concurrency=effective,
            ):
                kind, job_id = job["kind"], job["job_id"]
                python = freeze["runtime"][
                    "python_judge" if kind == "score" else "python_gpu"
                ]
                log = runtime_root / "logs" / f"{kind}_{job_id}_attempt{len(attempts)+1}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(python),
                    "-m",
                    "oasst1_filter.runtime.nat_replica",
                    "worker",
                    "--prepared-root",
                    str(prepared_root),
                    "--execution-root",
                    str(execution_root),
                    "--kind",
                    kind,
                    "--job-id",
                    job_id,
                ]
                handle = log.open("ab")
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, **freeze["runtime"].get("environment", {})},
                )
                handle.close()
                record: dict[str, object] = {
                    "kind": kind,
                    "job_id": job_id,
                    "pid": process.pid,
                    "log": str(log),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(record)
                active[(kind, job_id)] = (process, record)
            _atomic_json(progress_path, snapshot())
            time.sleep(2)
    finally:
        for process, _record in active.values():
            _stop_process(process)
        active.clear()
    result = snapshot()
    _atomic_json(progress_path, result)
    if status in {"FIXED_WORK_COMPLETE", "HARD_CAP_REACHED"}:
        result["publication"] = publish_nat_replica_results(
            prepared_root=prepared_root,
            execution_root=execution_root,
            hard_cap_reached=status == "HARD_CAP_REACHED",
        )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: "null" if value is None else value
                for key, value in row.items()
            }
            for row in rows
        )
    os.replace(temporary, path)


def publish_nat_replica_results(
    *, prepared_root: Path, execution_root: Path, hard_cap_reached: bool = False
) -> dict[str, object]:
    """Open all fixed outcomes once, compute root statistics, and publish minimally."""

    freeze = _load_prepared(prepared_root)
    claim = _load_claim(execution_root, freeze)
    amendment = load_nat_replica_soft_cap_amendment(execution_root, freeze)
    if amendment is not None:
        hard_cap_reached = False
    states = ("BASE", *nat_replica_state_ids(freeze["roots"]))
    position_rows: list[dict[str, object]] = []
    passes = {state: 0 for state in states}
    valid = {state: 0 for state in states}
    for state in states:
        root, arm = (
            (None, "BASE")
            if state == "BASE"
            else (int(state.split("_", 1)[0]), state.split("_", 1)[1])
        )
        for index, task_id in enumerate(freeze["task_ids"]):
            path = Path(execution_root) / "runtime" / "scores" / state / f"{index:03d}.json"
            row: dict[str, object] = {
                "state_id": state,
                "root": root,
                "arm": arm,
                "task_id": task_id,
                "Y": None,
                "verdict": "missing",
                "completion_sha256": None,
            }
            if path.is_file():
                score = json.loads(path.read_text("utf-8"))
                if not (
                    score.get("state_id") == state
                    and score.get("task_id") == task_id
                    and score.get("task_index") == index
                    and score.get("design_sha256")
                    == freeze["scientific_identity_sha256"]
                ):
                    raise ValueError("NAT-REPLICA score binding differs during publication")
                row.update(
                    Y=score.get("Y"),
                    verdict=score.get("verdict"),
                    completion_sha256=score.get("completion_sha256"),
                )
                if score.get("Y") in (0, 1):
                    valid[state] += 1
                    passes[state] += int(score["Y"])
            position_rows.append(row)
    complete = all(count == 164 for count in valid.values())
    root_rows: list[dict[str, object]] = []
    deltas: list[float] = []
    if complete:
        for root in freeze["roots"]:
            keep = passes[f"{root}_POL_KEEP"]
            filtered = passes[f"{root}_POL_FILTER"]
            delta = (filtered - keep) / 164
            deltas.append(delta)
            root_rows.append(
                {
                    "root": root,
                    "K_pass": keep,
                    "F_pass": filtered,
                    "F_minus_K": filtered - keep,
                    "delta": delta,
                    "delta_pp": 100 * delta,
                }
            )
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        half_width = 2.364624251 * sd / math.sqrt(8)
        ci = [mean - half_width, mean + half_width]
        terminal = resolve_nat_replica_terminal(
            complete=True,
            root_sd=sd,
            mean_delta_pp=100 * mean,
            ci_lower_pp=100 * ci[0],
            ci_upper_pp=100 * ci[1],
        )
    else:
        mean = sd = None
        ci = [None, None]
        terminal = resolve_nat_replica_terminal(
            complete=False, hard_cap_reached=hard_cap_reached
        )
        root_rows = [
            {
                "root": root,
                "K_pass": passes[f"{root}_POL_KEEP"],
                "F_pass": passes[f"{root}_POL_FILTER"],
                "K_valid": valid[f"{root}_POL_KEEP"],
                "F_valid": valid[f"{root}_POL_FILTER"],
                "F_minus_K": None,
                "delta": None,
                "delta_pp": None,
            }
            for root in freeze["roots"]
        ]
    execution_root = Path(execution_root)
    result: dict[str, object] = {
        "schema": "nat-replica-x164-formal-result-v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "position_count": 2_788,
        "valid_positions": sum(valid.values()),
        "complete": complete,
        "BASE": {"pass": passes["BASE"], "valid": valid["BASE"]},
        "root_rows": root_rows,
        "mean_delta": mean,
        "mean_delta_pp": None if mean is None else 100 * mean,
        "root_sd": sd,
        "ci95_df7": ci,
        "investment_scale_pp": 2.0,
        "clock": claim,
        "resource_amendment": amendment,
        "actual_wall_seconds": (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(str(claim["started_utc"]))
        ).total_seconds(),
        "resource_contract": {
            "target_hours": 20.0,
            "prior_hard_cap_hours": 42.0,
            "hard_cap_enforced": amendment is None,
            "completion_required": amendment is not None,
            **freeze["resource_scenarios_hours"],
        },
        "interpretation": (
            "Prospective exploratory external replication; this is not proof that "
            "individual samples are harmful or that deletion generally improves quality."
        ),
    }
    prefix = "NAT_REPLICA_X164"
    _write_csv(execution_root / f"{prefix}_POSITION_TABLE.csv", position_rows)
    _write_csv(execution_root / f"{prefix}_ROOT_TABLE.csv", root_rows)
    _atomic_json(execution_root / f"{prefix}_RESULT.json", result)
    report = (
        "# NAT-REPLICA-X164 formal result\n\n"
        f"Terminal: `{terminal}`. Valid positions: {sum(valid.values())}/2788.\n\n"
        f"Mean F-K: {result['mean_delta_pp']} pp; df=7 working 95% interval: {ci}.\n\n"
        f"{result['interpretation']}\n"
    )
    (execution_root / f"{prefix}_REPORT.md").write_text(report, "utf-8")
    handoff = (
        "# NAT-REPLICA-X164 handoff\n\n"
        f"Scientific identity: `{freeze['scientific_identity_sha256']}`.\n"
        f"Terminal: `{terminal}`. No automatic follow-up experiment is authorized.\n"
    )
    (execution_root / f"{prefix}_HANDOFF.md").write_text(handoff, "utf-8")
    names = [
        "FORMAL_CLAIM.json",
        f"{prefix}_POSITION_TABLE.csv",
        f"{prefix}_ROOT_TABLE.csv",
        f"{prefix}_RESULT.json",
        f"{prefix}_REPORT.md",
        f"{prefix}_HANDOFF.md",
    ]
    archive = execution_root / f"{prefix}_DELIVERY.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(
            Path(prepared_root) / "NAT_REPLICA_X164_EXECUTION_FREEZE.json",
            "NAT_REPLICA_X164_EXECUTION_FREEZE.json",
        )
        for name in names:
            handle.write(execution_root / name, name)
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("NAT-REPLICA delivery ZIP CRC failed")
    sidecar = execution_root / f"{prefix}_DELIVERY.zip.sha256"
    sidecar.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n", "utf-8"
    )
    return result


def prepare_nat_replica_formal_assets(
    *,
    workspace: Path,
    prepared_root: Path,
    seed_hex: str,
    model_path: Path,
    asset_path: Path,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    python_gpu: str,
    python_judge: str,
    local_asset_path: Path | None = None,
    local_evaluator_archive: Path | None = None,
) -> dict[str, object]:
    """Materialize the portable, outcome-blind pre-authorization asset set."""

    from ..arms.paired_filtering import (
        NAT_REPLICA_PROTOCOL_ID,
        build_nat_replace_plans,
        build_nat_replica_roots,
        dump_s12_training_plan,
    )
    from ..eval.humaneval import (
        EVALPLUS_REVISION,
        EVALPLUS_SOURCE_ARCHIVE_SHA256,
        HUMANEVAL_PLUS_LOGICAL_SHA256,
        HUMANEVAL_PLUS_SHA256,
        load_humaneval_plus_asset,
    )
    from .run_codecontests_v2 import reconstruct_nat_replace_input

    workspace = Path(workspace)
    prepared_root = Path(prepared_root)
    freeze_path = prepared_root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text("utf-8"))
        validate_nat_replica_formal_freeze(freeze)
        return freeze
    prepared_root.mkdir(parents=True, exist_ok=True)
    recovered = reconstruct_nat_replace_input(workspace)
    historical = list(recovered["historical_roots"])
    historical_sha = _canonical_sha256(historical)
    roots = build_nat_replica_roots(seed_hex=seed_hex, historical_roots=historical)
    root_draw = {
        "schema": "nat-replica-x164-outcome-blind-root-draw-v1",
        "protocol_id": PROTOCOL_ID,
        "method": "one local secrets.token_hex(32) draw; SHA-256 indexed derivation",
        "seed_hex": seed_hex,
        "historical_roots_sha256": historical_sha,
        "historical_root_count": len(historical),
        "roots": list(roots),
        "model_outcomes_read": False,
    }
    _atomic_json(prepared_root / "ROOT_DRAW.json", root_draw)
    plans_by_identity: dict[tuple[int, str], dict[str, object]] = {}
    for root in roots:
        for arm, plan in build_nat_replace_plans(
            recovered["pairs"], root=root, protocol_id=NAT_REPLICA_PROTOCOL_ID
        ).items():
            relative = Path("plans") / f"{root}_{arm}.json"
            digest = dump_s12_training_plan(prepared_root / relative, plan)
            plans_by_identity[(root, arm)] = {
                "root": root,
                "arm": arm,
                "plan_path": relative.as_posix(),
                "plan_sha256": digest,
                "updates": len(plan.update_bins),
                "response_tokens": sum(
                    exposure.retained_response_tokens for exposure in plan.exposures
                ),
                "exposures": len(plan.exposures),
            }
    ordered_plans = [
        plans_by_identity[
            (int(state.split("_", 1)[0]), state.split("_", 1)[1])
        ]
        for state in nat_replica_state_ids(roots)
    ]
    tasks = load_humaneval_plus_asset(Path(local_asset_path or asset_path))
    archive_sha = hashlib.sha256(
        Path(local_evaluator_archive or evaluator_archive).read_bytes()
    ).hexdigest()
    if archive_sha != EVALPLUS_SOURCE_ARCHIVE_SHA256:
        raise ValueError("NAT-REPLICA evaluator archive differs from qualification freeze")
    source_bindings_sha = _canonical_sha256(recovered["source_bindings"])
    plan_file_hashes = {
        str(plan["plan_path"]): hashlib.sha256(
            (prepared_root / str(plan["plan_path"])).read_bytes()
        ).hexdigest()
        for plan in ordered_plans
    }
    bindings = {
        "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "humaneval_compressed_sha256": HUMANEVAL_PLUS_SHA256,
        "humaneval_logical_sha256": HUMANEVAL_PLUS_LOGICAL_SHA256,
        "evaluator_revision": EVALPLUS_REVISION,
        "evaluator_archive_sha256": archive_sha,
        "qualified_source_bindings_sha256": source_bindings_sha,
        "qualified_authority_hashes": recovered["authority_hashes"],
        "manifest_sha256": recovered["manifest_sha256"],
        "training_token_totals": recovered["token_totals"],
        "plan_file_hashes": plan_file_hashes,
        "root_draw_sha256": hashlib.sha256(
            (prepared_root / "ROOT_DRAW.json").read_bytes()
        ).hexdigest(),
    }
    freeze = build_nat_replica_formal_freeze(
        roots=roots,
        root_seed_sha256=hashlib.sha256(seed_hex.encode("ascii")).hexdigest(),
        historical_roots_sha256=historical_sha,
        plans=ordered_plans,
        task_ids=[str(task["task_id"]) for task in tasks],
        bindings=bindings,
        paths={
            "model": str(model_path),
            "asset": str(asset_path),
            "evaluator_source": str(evaluator_source_root),
            "evaluator_archive": str(evaluator_archive),
        },
        python_gpu=python_gpu,
        python_judge=python_judge,
    )
    _atomic_json(freeze_path, freeze)
    refresh_nat_replica_prepared_manifest(prepared_root, freeze)
    return freeze


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in (
        "workspace",
        "prepared-root",
        "model",
        "asset",
        "evaluator-source",
        "evaluator-archive",
    ):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--seed-hex", required=True)
    prepare.add_argument("--python-gpu", required=True)
    prepare.add_argument("--python-judge", required=True)
    prepare.add_argument("--local-asset", type=Path)
    prepare.add_argument("--local-evaluator-archive", type=Path)
    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("--prepared-root", type=Path, required=True)
    refresh = commands.add_parser("refresh-manifest")
    refresh.add_argument("--prepared-root", type=Path, required=True)
    amend = commands.add_parser("amend-soft-cap")
    amend.add_argument("--prepared-root", type=Path, required=True)
    amend.add_argument("--execution-root", type=Path, required=True)
    amend.add_argument("--confirmation", required=True)
    run = commands.add_parser("run")
    run.add_argument("--prepared-root", type=Path, required=True)
    run.add_argument("--execution-root", type=Path, required=True)
    run.add_argument("--authorization", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--prepared-root", type=Path, required=True)
    worker.add_argument("--execution-root", type=Path, required=True)
    worker.add_argument("--kind", choices=("train", "generation", "score"), required=True)
    worker.add_argument("--job-id", required=True)
    report = commands.add_parser("report")
    report.add_argument("--prepared-root", type=Path, required=True)
    report.add_argument("--execution-root", type=Path, required=True)
    report.add_argument("--hard-cap-reached", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        freeze = prepare_nat_replica_formal_assets(
            workspace=arguments.workspace,
            prepared_root=arguments.prepared_root,
            seed_hex=arguments.seed_hex,
            model_path=arguments.model,
            asset_path=arguments.asset,
            evaluator_source_root=arguments.evaluator_source,
            evaluator_archive=arguments.evaluator_archive,
            python_gpu=arguments.python_gpu,
            python_judge=arguments.python_judge,
            local_asset_path=arguments.local_asset,
            local_evaluator_archive=arguments.local_evaluator_archive,
        )
        print(
            json.dumps(
                {
                    "state": freeze["state"],
                    "scientific_identity_sha256": freeze["scientific_identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "dry-run":
        freeze = _load_prepared(arguments.prepared_root)
        print(json.dumps(dry_run_nat_replica_formal(freeze), sort_keys=True))
        return 0
    if arguments.command == "refresh-manifest":
        root = arguments.prepared_root
        freeze = json.loads(
            (root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json").read_text("utf-8")
        )
        manifest = refresh_nat_replica_prepared_manifest(root, freeze)
        print(json.dumps({"files": len(manifest["files"]), "state": manifest["state"]}))
        return 0
    if arguments.command == "amend-soft-cap":
        freeze = _load_prepared(arguments.prepared_root)
        amendment = record_nat_replica_soft_cap_amendment(
            freeze,
            execution_root=arguments.execution_root,
            confirmation=arguments.confirmation,
        )
        print(json.dumps(amendment, sort_keys=True))
        return 0
    if arguments.command == "worker":
        result = execute_nat_replica_formal_worker(
            prepared_root=arguments.prepared_root,
            execution_root=arguments.execution_root,
            kind=arguments.kind,
            job_id=arguments.job_id,
        )
        print(json.dumps(result, sort_keys=True))
        return 75 if result.get("counts", {}).get("infrastructure_error", 0) else 0
    if arguments.command == "report":
        result = publish_nat_replica_results(
            prepared_root=arguments.prepared_root,
            execution_root=arguments.execution_root,
            hard_cap_reached=arguments.hard_cap_reached,
        )
        print(json.dumps({"terminal": result["terminal"]}, sort_keys=True))
        return 0
    result = run_nat_replica_formal(
        prepared_root=arguments.prepared_root,
        execution_root=arguments.execution_root,
        authorization=arguments.authorization,
    )
    print(json.dumps(result, sort_keys=True))
    return 75 if result["state"] in {
        "REPAIR_REQUIRED",
        "RESOURCE_DOWNSHIFT_RESTART_REQUIRED",
    } else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
