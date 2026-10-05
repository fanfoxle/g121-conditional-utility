"""Thin CPU-only execution boundary for NAT-SPEC-20 qualification."""

from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import statistics
import subprocess
import time
from typing import Mapping, Sequence
import zipfile

from ..arms.paired_filtering import (
    NAT_MSEP_PROTOCOL_ID,
    NAT_MSEP_QUALIFICATION_NAMESPACE,
    NAT_REPLICA_PROTOCOL_ID,
    NAT_SPEC_PROTOCOL_ID,
    NAT_SPEC_QUALIFICATION_NAMESPACE,
    NatMsepDeadlineExceeded,
    NatMsepQualificationError,
    NatMsepRootQualification,
    NatMsepAnonymousExposure,
    NatSpecQualificationError,
    NatSpecRootQualification,
    build_nat_msep_subset_support,
    load_s12_training_plan,
    nat_msep_qualification_rank,
    qualify_nat_msep_plan_pair,
    qualify_nat_spec_plan_pair,
    s12_training_plan_sha256,
)


NAT_SPEC_TERMINALS = {
    "QUALIFIED_NAT_SPEC_20_CPU_CONTROL",
    "NAT_SPEC_20_DESIGN_NOT_QUALIFIED",
}
NAT_SPEC_FAILURE_REASONS = {
    "source_or_plan_identity_drift",
    "budget_or_component_contract_drift",
    "exact_uniform_sampler_invalid",
    "control_overlap_above_half",
    "qualification_census_over_2h",
}
NAT_MSEP_TERMINALS = {
    "NAT_MSEP_20_ASSET_CONTRACT_NOT_READY",
    "NAT_MSEP_20_EXACT_SEPARATION_INSUFFICIENT",
    "NAT_MSEP_20_CPU_CAP_UNRESOLVED",
    "NAT_MSEP_20_CPU_CONTRAST_QUALIFIED_FOR_REVIEW",
}
NAT_MSEP_FORMAL_TERMINALS = frozenset(
    {
        "NAT_MSEP_20_MEASUREMENT_INVALID",
        "NAT_MSEP_20_STOPPED_SAFETY_OR_PERMISSION",
        "NAT_MSEP_20_FORMAL_ROOT_SET_NOT_QUALIFIED",
        "NAT_MSEP_20_INCOMPLETE_RESOURCE_CAP",
        "NAT_MSEP_20_INCOMPLETE_INFRASTRUCTURE",
        "NAT_MSEP_20_INFORMATION_INSUFFICIENT",
        "NAT_MSEP_20_ACTIVE_CONTRAST_ADVERSE",
        "NAT_MSEP_20_BELOW_INVESTMENT_SCALE",
        "NAT_MSEP_20_POSITIVE_EXPLORATORY",
        "NAT_MSEP_20_UNRESOLVED_WITHIN_BUDGET",
    }
)
NAT_MSEP_FORMAL_CONCURRENCY = {"train": 2, "generation": 4, "score": 4}


def build_nat_msep_prior_root_registry(
    *,
    design: Mapping[str, object],
    replica_freeze: Mapping[str, object],
    expected_count: int | None = 124,
) -> dict[str, object]:
    """Build the identity-only canonical registry accepted by the formal root draw."""

    collections = (
        design.get("historical_roots"), design.get("roots"), replica_freeze.get("roots")
    )
    if any(not isinstance(values, (list, tuple)) for values in collections):
        raise ValueError("NAT-MSEP prior-root authority is incomplete")
    roots: list[int] = []
    for values in collections:
        assert isinstance(values, (list, tuple))
        for value in values:
            if type(value) is not int or value <= 0:
                raise ValueError("NAT-MSEP prior-root authority contains an invalid root")
            roots.append(value)
    canonical = sorted(set(roots))
    if expected_count is not None and len(canonical) != expected_count:
        raise ValueError(
            f"NAT-MSEP prior-root registry has {len(canonical)} rather than {expected_count} roots"
        )
    encoded = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "nat-msep-20-prior-root-registry-v1",
        "roots": canonical,
        "count": len(canonical),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "outcome_fields_received": False,
    }


def claim_nat_msep_formal_randomization(
    *,
    runtime_root: Path,
    prior_roots: Sequence[int],
    started_utc: datetime | None = None,
    root_seed_hex: str | None = None,
    mask_seed_hex: str | None = None,
) -> dict[str, object]:
    """Atomically commit the non-resetting clock and both seeds before geometry."""

    from ..arms.paired_filtering import (
        NAT_MSEP_FORMAL_MASK_NAMESPACE,
        NAT_MSEP_FORMAL_ROOT_NAMESPACE,
    )

    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    claim_path = runtime_root / "FORMAL_RANDOMIZATION_CLAIM.json"
    registry = sorted(set(prior_roots))
    if list(prior_roots) != registry or any(
        type(root) is not int or root <= 0 for root in registry
    ):
        raise ValueError("NAT-MSEP claim requires the canonical prior-root registry")
    registry_sha = hashlib.sha256(
        json.dumps(registry, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claim_path.exists():
        existing = json.loads(claim_path.read_text("utf-8"))
        if existing.get("prior_root_registry_sha256") != registry_sha:
            raise ValueError("NAT-MSEP existing claim registry differs")
        return existing
    began = started_utc or datetime.now(timezone.utc)
    if began.tzinfo is None:
        raise ValueError("NAT-MSEP formal start must be timezone-aware")
    began = began.astimezone(timezone.utc)
    root_seed = root_seed_hex or secrets.token_hex(32)
    mask_seed = mask_seed_hex or secrets.token_hex(32)
    if any(
        not _is_lower_sha256(value) for value in (root_seed, mask_seed)
    ):
        raise ValueError("NAT-MSEP formal seeds must be 32-byte lowercase hex")
    claim: dict[str, object] = {
        "schema": "nat-msep-20-formal-randomization-claim-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "started_utc": began.isoformat(),
        "deadline_utc": (began + timedelta(hours=48)).isoformat(),
        "root_seed_hex": root_seed,
        "mask_seed_hex": mask_seed,
        "root_namespace": NAT_MSEP_FORMAL_ROOT_NAMESPACE,
        "mask_base_namespace": NAT_MSEP_FORMAL_MASK_NAMESPACE,
        "mask_namespace": f"{NAT_MSEP_FORMAL_MASK_NAMESPACE}/{mask_seed}",
        "prior_root_registry_sha256": registry_sha,
        "hard_cap_resets": False,
    }
    _atomic_json(claim_path, claim)
    return claim


def _load_nat_replica_freeze_input(path: Path) -> dict[str, object]:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            return json.loads(
                archive.read("NAT_REPLICA_X164_EXECUTION_FREEZE.json").decode("utf-8")
            )
    return json.loads(path.read_text("utf-8"))


def validate_nat_msep_formal_prepared(prepared_root: Path) -> dict[str, object]:
    prepared_root = Path(prepared_root)
    value = json.loads((prepared_root / "PREPARED.json").read_text("utf-8"))
    registry = json.loads(
        (prepared_root / "PRIOR_ROOT_REGISTRY.json").read_text("utf-8")
    )
    roots = registry.get("roots")
    if not (
        value.get("schema") == "nat-msep-20-formal-prepared-v1"
        and value.get("state") == "prepared_nonformal"
        and value.get("formal_randomization_created") is False
        and registry.get("schema") == "nat-msep-20-prior-root-registry-v1"
        and isinstance(roots, list)
        and len(roots) == 124
        and registry.get("sha256")
        == hashlib.sha256(
            json.dumps(roots, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        and value.get("prior_root_registry_sha256") == registry.get("sha256")
        and isinstance(value.get("paths"), dict)
        and isinstance(value.get("bindings"), dict)
    ):
        raise ValueError("NAT-MSEP prepared formal identity differs")
    paths = value["paths"]
    bindings = value["bindings"]
    expected = {
        "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "humaneval_compressed_sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "humaneval_logical_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
        "evaluator_revision": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        "evaluator_archive_sha256": "d11dc64dfee6858a52c493466aa91a7bd0e3d50deda9a586144c1ba127c0e869",
    }
    if any(bindings.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("NAT-MSEP prepared fixed binding differs")
    if (
        not Path(str(paths.get("model", ""))).is_dir()
        or not Path(str(paths.get("asset", ""))).is_file()
        or not Path(str(paths.get("evaluator_source", ""))).is_dir()
        or not Path(str(paths.get("evaluator_archive", ""))).is_file()
        or _sha256_file(Path(str(paths["evaluator_archive"])))
        != expected["evaluator_archive_sha256"]
    ):
        raise ValueError("NAT-MSEP prepared runtime asset differs")
    from .nat_replica import validate_nat_replica_formal_freeze

    validate_nat_replica_formal_freeze(
        _load_nat_replica_freeze_input(Path(str(paths["replica_freeze"])))
    )
    return value


def prepare_nat_msep_formal(
    *,
    workspace: Path,
    prepared_root: Path,
    design_path: Path,
    replica_freeze_path: Path,
    model_path: Path,
    asset_path: Path,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    python_gpu: str,
    python_judge: str,
) -> dict[str, object]:
    """Freeze only outcome-free D4 inputs; never draw a formal seed or root."""

    from ..eval.humaneval import (
        EVALPLUS_REVISION,
        EVALPLUS_SOURCE_ARCHIVE_SHA256,
        HUMANEVAL_PLUS_LOGICAL_SHA256,
        HUMANEVAL_PLUS_SHA256,
        load_humaneval_plus_asset,
    )

    prepared_root = Path(prepared_root)
    existing = prepared_root / "PREPARED.json"
    if existing.exists():
        return validate_nat_msep_formal_prepared(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)
    design = json.loads(Path(design_path).read_text("utf-8"))
    replica = _load_nat_replica_freeze_input(Path(replica_freeze_path))
    from .nat_replica import validate_nat_replica_formal_freeze

    validate_nat_replica_formal_freeze(replica)
    registry = build_nat_msep_prior_root_registry(
        design=design, replica_freeze=replica
    )
    tasks = load_humaneval_plus_asset(Path(asset_path))
    if [row["task_id"] for row in tasks] != [f"HumanEval/{i}" for i in range(164)]:
        raise ValueError("NAT-MSEP prepared HumanEval panel differs")
    if not Path(model_path).is_dir() or Path(model_path).name != (
        "8faed761d45a263340a0528343f099c05c9a4323"
    ):
        raise ValueError("NAT-MSEP prepared model revision path differs")
    if (
        not Path(evaluator_source_root).is_dir()
        or not Path(evaluator_archive).is_file()
        or _sha256_file(Path(evaluator_archive)) != EVALPLUS_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("NAT-MSEP prepared evaluator authority differs")
    bindings = {
        "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "humaneval_compressed_sha256": HUMANEVAL_PLUS_SHA256,
        "humaneval_logical_sha256": HUMANEVAL_PLUS_LOGICAL_SHA256,
        "evaluator_revision": EVALPLUS_REVISION,
        "evaluator_archive_sha256": EVALPLUS_SOURCE_ARCHIVE_SHA256,
        "design_file_sha256": _sha256_file(Path(design_path)),
    }
    paths = {
        "workspace": str(Path(workspace)),
        "design": str(Path(design_path)),
        "replica_freeze": str(Path(replica_freeze_path)),
        "model": str(Path(model_path)),
        "asset": str(Path(asset_path)),
        "evaluator_source": str(Path(evaluator_source_root)),
        "evaluator_archive": str(Path(evaluator_archive)),
    }
    value: dict[str, object] = {
        "schema": "nat-msep-20-formal-prepared-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "state": "prepared_nonformal",
        "formal_randomization_created": False,
        "prior_root_registry_sha256": registry["sha256"],
        "paths": paths,
        "bindings": bindings,
        "runtime": {"python_gpu": str(python_gpu), "python_judge": str(python_judge)},
        "task_ids": [f"HumanEval/{i}" for i in range(164)],
    }
    _atomic_json(prepared_root / "PRIOR_ROOT_REGISTRY.json", registry)
    _atomic_json(existing, value)
    return validate_nat_msep_formal_prepared(prepared_root)


def dry_run_nat_msep_formal(prepared_root: Path) -> dict[str, object]:
    """Validate the nonformal launch contract without CUDA, seeds, roots, or outcomes."""

    value = validate_nat_msep_formal_prepared(prepared_root)
    return {
        "valid": True,
        "prior_root_registry_sha256": value["prior_root_registry_sha256"],
        "cuda_initialized": False,
        "optimizer_initialized": False,
        "formal_seed_created": False,
        "formal_root_created": False,
        "formal_mask_created": False,
        "formal_outcomes_opened": False,
        "formal_training_started": False,
    }


def _materialize_nat_msep_formal_root(
    pairs: Sequence[Mapping[str, object]],
    root: int,
    mask_namespace: str,
    plans_root: str,
    deadline_epoch: float,
) -> dict[str, object]:
    from ..arms.paired_filtering import (
        build_nat_replace_plans,
        dump_s12_training_plan,
        materialize_nat_msep_plan_pair,
    )

    plans = build_nat_replace_plans(
        pairs, root=root, protocol_id=NAT_MSEP_PROTOCOL_ID
    )
    filtered = plans["POL_FILTER"]
    qualified = materialize_nat_msep_plan_pair(
        root=root,
        keep_plan=plans["POL_KEEP"],
        filtered_plan=filtered,
        rank_namespace=mask_namespace,
        deadline_epoch=deadline_epoch,
    )
    if qualified.msep_plan is None:
        raise RuntimeError("NAT-MSEP formal materialization omitted the M plan")
    entries: dict[str, dict[str, object]] = {}
    for arm, plan in (
        ("POL_FILTER", filtered), ("POL_MSEP", qualified.msep_plan)
    ):
        path = Path(plans_root) / f"{root}_{arm}.json"
        identity = dump_s12_training_plan(path, plan)
        entries[arm] = {
            "root": root,
            "arm": arm,
            "plan_path": str(path),
            "plan_sha256": identity,
            "plan_file_sha256": _sha256_file(path),
            "updates": 100,
            "response_tokens": 583_975,
        }
    masks = [
        {
            "root": root,
            "update": int(row["update"]),
            "rank": int(row["qualification_rank"]),
            "selected_ordinals": list(row["selected_ordinals"]),
            "selected_count": int(row["selected_count"]),
            "selected_response_tokens": int(row["selected_response_tokens"]),
        }
        for row in qualified.update_rows
    ]
    return {"root": root, "plans": entries, "masks": masks, "gate": qualified.root_row}


def materialize_nat_msep_formal_freeze(
    *, prepared_root: Path, task_root: Path, output_root: Path
) -> dict[str, object]:
    """Claim once, then construct all formal roots/masks/plans and one freeze."""

    from ..arms.paired_filtering import build_nat_msep_formal_roots
    from ..eval.humaneval import load_humaneval_plus_asset
    from .run_codecontests_v2 import reconstruct_nat_replace_input

    prepared = validate_nat_msep_formal_prepared(prepared_root)
    registry_payload = json.loads(
        (Path(prepared_root) / "PRIOR_ROOT_REGISTRY.json").read_text("utf-8")
    )
    prior_roots = tuple(int(root) for root in registry_payload["roots"])
    task_root = Path(task_root)
    output_root = Path(output_root)
    task_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    claim = claim_nat_msep_formal_randomization(
        runtime_root=task_root, prior_roots=prior_roots
    )
    freeze_path = output_root / "NAT_MSEP_20_EXECUTION_FREEZE.json"
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text("utf-8"))
        if freeze.get("claim_sha256") != _canonical_json_sha256(claim):
            raise ValueError("NAT-MSEP existing execution freeze claim differs")
        return freeze
    deadline = datetime.fromisoformat(str(claim["deadline_utc"]))
    roots = build_nat_msep_formal_roots(
        seed_hex=str(claim["root_seed_hex"]), prior_roots=prior_roots
    )
    # Geometry becomes visible only after the claim above is durably committed.
    recovered = reconstruct_nat_replace_input(Path(str(prepared["paths"]["workspace"])))
    pairs = tuple(recovered["pairs"])
    plans_root = task_root / "plans"
    results: dict[int, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                _materialize_nat_msep_formal_root,
                pairs,
                root,
                str(claim["mask_namespace"]),
                str(plans_root),
                deadline.timestamp(),
            ): root
            for root in roots
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    states = nat_msep_state_ids(roots)
    plan_rows = [
        results[int(state.split("_", 1)[0])]["plans"][state.split("_", 1)[1]]
        for state in states
    ]
    masks = [row for root in roots for row in results[root]["masks"]]
    gates = [results[root]["gate"] for root in roots]
    tasks = load_humaneval_plus_asset(Path(str(prepared["paths"]["asset"])))
    freeze = build_nat_msep_execution_freeze(
        claim=claim,
        prior_roots=prior_roots,
        roots=roots,
        plans=plan_rows,
        mask_rows=masks,
        root_rows=gates,
        task_ids=[str(task["task_id"]) for task in tasks],
        bindings=prepared["bindings"],
        paths={
            key: prepared["paths"][key]
            for key in ("model", "asset", "evaluator_source", "evaluator_archive")
        },
        python_gpu=str(prepared["runtime"]["python_gpu"]),
        python_judge=str(prepared["runtime"]["python_judge"]),
    )
    _atomic_json(freeze_path, freeze)
    _atomic_json(
        task_root / "E0_TIMING.json",
        {
            "started_utc": claim["started_utc"],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return freeze


def _load_nat_msep_execution_freeze(output_root: Path) -> dict[str, object]:
    path = Path(output_root) / "NAT_MSEP_20_EXECUTION_FREEZE.json"
    freeze = json.loads(path.read_text("utf-8"))
    if not (
        freeze.get("schema") == "nat-msep-20-execution-freeze-v1"
        and freeze.get("protocol_id") == NAT_MSEP_PROTOCOL_ID
        and _is_lower_sha256(freeze.get("scientific_identity_sha256"))
        and len(freeze.get("roots", ())) == 8
        and len(freeze.get("plans", ())) == 16
    ):
        raise ValueError("NAT-MSEP execution freeze identity differs")
    return freeze


def _nat_msep_training_card(
    freeze: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    from ..protocol import CODECONTESTS_ROLE_NAMES, derive_codecontests_role_seed

    root = int(plan["root"])
    arm = str(plan["arm"])
    card: dict[str, object] = {
        "schema": "codecontests-nat-msep-20-run-card-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "design_id": "CC_NAT_MSEP_20",
        "design_sha256": freeze["scientific_identity_sha256"],
        "stage": "POLICY",
        "role": "exploratory_active_contrast",
        "root": root,
        "arm_ordinal": ("POL_FILTER", "POL_MSEP").index(arm),
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
        "plan_sha256": plan["plan_sha256"],
    }
    card["card_sha256"] = _canonical_json_sha256(card)
    return card


def execute_nat_msep_formal_worker(
    *,
    output_root: Path,
    task_root: Path,
    kind: str,
    job_id: str,
    score_workers: int = 4,
) -> dict[str, object]:
    """Execute one frozen F/M training, generation, or scoring job."""

    freeze = _load_nat_msep_execution_freeze(output_root)
    if (kind, job_id) not in {
        (str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]
    }:
        raise ValueError("NAT-MSEP worker job is outside the frozen queue")
    deadline = datetime.fromisoformat(str(freeze["claim"]["deadline_utc"]))
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("NAT-MSEP 48h deadline is closed")
    plan = next(
        item
        for item in freeze["plans"]
        if f"{item['root']}_{item['arm']}" == job_id
    )
    card = _nat_msep_training_card(freeze, plan)
    task_root = Path(task_root)
    training_root = task_root / "training" / job_id
    if kind == "train":
        from ..train.run_codecontests import execute_codecontests_s12_formal

        return execute_codecontests_s12_formal(
            plan=load_s12_training_plan(Path(str(plan["plan_path"]))),
            card=card,
            model_path=Path(str(freeze["paths"]["model"])),
            output_dir=training_root,
            resume=(training_root / "recovery" / "CURRENT.json").exists()
            or (training_root / "TERMINAL_TRAINING_RESULT.json").exists(),
        )
    from ..train.run_codecontests import validate_codecontests_s12_terminal

    validate_codecontests_s12_terminal(
        training_root / "TERMINAL_TRAINING_RESULT.json",
        expected_card_sha256=str(card["card_sha256"]),
        expected_plan_sha256=str(plan["plan_sha256"]),
        expected_root=int(plan["root"]),
        expected_protocol_id=NAT_MSEP_PROTOCOL_ID,
        expected_arm=str(plan["arm"]),
    )
    from ..eval.humaneval import (
        execute_humaneval_formal_generation,
        execute_humaneval_formal_scoring,
        load_humaneval_plus_asset,
    )

    tasks = load_humaneval_plus_asset(Path(str(freeze["paths"]["asset"])))
    if [task["task_id"] for task in tasks] != freeze["task_ids"]:
        raise ValueError("NAT-MSEP HumanEval task identity differs")
    if kind == "generation":
        return execute_humaneval_formal_generation(
            state_id=job_id,
            tasks=tasks,
            model_path=Path(str(freeze["paths"]["model"])),
            adapter_path=training_root / "adapter100",
            output_dir=task_root / "generation" / job_id,
            design_sha256=str(freeze["scientific_identity_sha256"]),
        )
    return execute_humaneval_formal_scoring(
        state_id=job_id,
        tasks=tasks,
        generation_dir=task_root / "generation" / job_id,
        output_dir=task_root / "scores" / job_id,
        design_sha256=str(freeze["scientific_identity_sha256"]),
        evaluator_source_root=Path(str(freeze["paths"]["evaluator_source"])),
        evaluator_archive=Path(str(freeze["paths"]["evaluator_archive"])),
        sandbox_base=task_root / "sandboxes" / job_id,
        workers=score_workers,
        python_executable=Path(str(freeze["runtime"]["python_judge"])),
    )


def _completed_nat_msep_formal_jobs(
    freeze: Mapping[str, object], task_root: Path
) -> set[tuple[str, str]]:
    from ..train.run_codecontests import validate_codecontests_s12_terminal

    completed: set[tuple[str, str]] = set()
    for job in freeze["job_order"]:
        kind, job_id = str(job["kind"]), str(job["job_id"])
        if kind == "train":
            plan = next(
                item for item in freeze["plans"]
                if f"{item['root']}_{item['arm']}" == job_id
            )
            card = _nat_msep_training_card(freeze, plan)
            path = task_root / "training" / job_id / "TERMINAL_TRAINING_RESULT.json"
            if path.is_file():
                validate_codecontests_s12_terminal(
                    path,
                    expected_card_sha256=str(card["card_sha256"]),
                    expected_plan_sha256=str(plan["plan_sha256"]),
                    expected_root=int(plan["root"]),
                    expected_protocol_id=NAT_MSEP_PROTOCOL_ID,
                    expected_arm=str(plan["arm"]),
                )
                completed.add((kind, job_id))
        else:
            directory = "generation" if kind == "generation" else "scores"
            path = task_root / directory / job_id / "COMPLETE.json"
            if path.is_file():
                value = json.loads(path.read_text("utf-8"))
                if (
                    value.get("completed_positions") == 164
                    and value.get("design_sha256") == freeze["scientific_identity_sha256"]
                    and (kind != "score" or value.get("valid_positions") == 164)
                ):
                    completed.add((kind, job_id))
    return completed


def _nat_msep_position_rows(
    freeze: Mapping[str, object], task_root: Path, *, missing_reason: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in freeze["roots"]:
        for arm in ("POL_FILTER", "POL_MSEP"):
            state = f"{root}_{arm}"
            for index, task_id in enumerate(freeze["task_ids"]):
                path = task_root / "scores" / state / f"{index:03d}.json"
                row: dict[str, object] = {
                    "state_id": state, "root": int(root), "arm": arm,
                    "task_id": task_id, "Y": None, "verdict": "missing",
                    "completion_sha256": None, "missing_reason": missing_reason,
                }
                if path.is_file():
                    score = json.loads(path.read_text("utf-8"))
                    if not (
                        score.get("state_id") == state
                        and score.get("task_id") == task_id
                        and score.get("task_index") == index
                        and score.get("design_sha256") == freeze["scientific_identity_sha256"]
                    ):
                        raise ValueError("NAT-MSEP score binding differs")
                    row.update(
                        Y=score.get("Y"), verdict=score.get("verdict"),
                        completion_sha256=score.get("completion_sha256"),
                        missing_reason=(
                            None if score.get("Y") in (0, 1)
                            else str(score.get("detail") or "infrastructure_error")
                        ),
                    )
                rows.append(row)
    return rows


def _stop_nat_msep_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_nat_msep_formal(
    *, prepared_root: Path, task_root: Path, output_root: Path
) -> dict[str, object]:
    """Run the finite formal F/M queue and publish only at a legal terminal."""

    prepared_root, task_root, output_root = map(
        Path, (prepared_root, task_root, output_root)
    )
    result_path = output_root / "NAT_MSEP_20_RESULT.json"
    if result_path.exists():
        return json.loads(result_path.read_text("utf-8"))
    freeze = materialize_nat_msep_formal_freeze(
        prepared_root=prepared_root, task_root=task_root, output_root=output_root
    )
    e0_timing = json.loads((task_root / "E0_TIMING.json").read_text("utf-8"))
    began = datetime.fromisoformat(str(freeze["claim"]["started_utc"]))
    deadline = datetime.fromisoformat(str(freeze["claim"]["deadline_utc"]))
    e0_finished = datetime.fromisoformat(str(e0_timing["finished_utc"]))
    if not freeze["root_gate"]["passed"]:
        return publish_nat_msep_formal(
            output_root=output_root, freeze=freeze, position_rows=(),
            phase_seconds={"E0": (e0_finished - began).total_seconds(), "E1": 0, "E2": 0, "E3": 0},
            effective_concurrency=NAT_MSEP_FORMAL_CONCURRENCY, repairs=(),
            completed_training=0,
            forced_terminal="NAT_MSEP_20_FORMAL_ROOT_SET_NOT_QUALIFIED",
        )
    task_root.mkdir(parents=True, exist_ok=True)
    progress_path = task_root / "progress.json"
    previous = json.loads(progress_path.read_text("utf-8")) if progress_path.exists() else {}
    effective = dict(previous.get("effective_concurrency", freeze["concurrency"]))
    attempts: list[dict[str, object]] = list(previous.get("attempts", ()))
    repairs: list[str] = list(previous.get("repairs", ()))
    retry_counts: dict[str, int] = dict(previous.get("retry_counts", {}))
    milestones: dict[str, str] = dict(previous.get("milestones", {}))
    completed = _completed_nat_msep_formal_jobs(freeze, task_root)
    all_jobs = {
        (str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]
    }
    active: dict[
        tuple[str, str], tuple[subprocess.Popen[bytes], dict[str, object]]
    ] = {}
    status = "running"
    error: str | None = None

    def write_progress() -> None:
        generations = sum(kind == "generation" for kind, _ in completed)
        scores = sum(kind == "score" for kind, _ in completed)
        progress = build_nat_msep_progress(
            scientific_identity_sha256=str(freeze["scientific_identity_sha256"]),
            stage=(
                "terminal" if status == "terminal"
                else "training" if sum(kind == "train" for kind, _ in completed) < 16
                else "generation" if generations < 16
                else "scoring" if scores < 16
                else "publication"
            ),
            completed_training=sum(kind == "train" for kind, _ in completed),
            completed_generation_states=generations,
            completed_score_states=scores,
            generated_positions=sum(1 for _ in (task_root / "generation").glob("*/[0-9][0-9][0-9].json")),
            scored_positions=sum(1 for _ in (task_root / "scores").glob("*/[0-9][0-9][0-9].json")),
            active=[record for _process, record in active.values()],
            effective_concurrency=effective,
            started_utc=began, deadline_utc=deadline,
            repair_count=len(repairs), error=error,
        )
        progress.update(
            status=status, attempts=attempts, repairs=repairs,
            retry_counts=retry_counts, milestones=milestones,
            supervisor_pid=os.getpid(),
        )
        _atomic_json(progress_path, progress)

    try:
        while True:
            now = datetime.now(timezone.utc)
            if now >= deadline:
                status = "hard_cap_reached"
                break
            for identity, (process, record) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del active[identity]
                record["ended_utc"] = now.isoformat()
                record["exit_code"] = returncode
                if returncode == 0:
                    refreshed = _completed_nat_msep_formal_jobs(freeze, task_root)
                    if identity in refreshed:
                        completed |= refreshed
                        continue
                    returncode = 1
                    error = "worker exited zero without a valid frozen completion receipt"
                log_tail = Path(str(record["log"])).read_text(
                    "utf-8", errors="replace"
                )[-8_000:]
                error = log_tail
                key = f"{identity[0]}:{identity[1]}"
                retry_counts[key] = retry_counts.get(key, 0) + 1
                if "out of memory" in log_tail.lower():
                    try:
                        changed = downshift_nat_msep_concurrency(
                            effective, kind=identity[0]
                        )
                    except ValueError:
                        changed = None
                    if changed is not None:
                        effective = changed
                        repairs.append(
                            f"resource-only {identity[0]} concurrency downshift after OOM"
                        )
                if retry_counts[key] >= 3:
                    status = "repair_required"
                    write_progress()
                    return {
                        "status": status,
                        "job": key,
                        "progress": str(progress_path),
                        "scientific_terminal_created": False,
                    }
            if completed == all_jobs:
                status = "fixed_work_complete"
                break
            training_done = sum(kind == "train" for kind, _ in completed) == 16
            if training_done and "training_finished_utc" not in milestones:
                milestones["training_finished_utc"] = now.isoformat()
            for job in next_nat_msep_formal_jobs(
                freeze["roots"], completed=completed, active=set(active),
                concurrency=effective,
            ):
                kind, job_id = str(job["kind"]), str(job["job_id"])
                log = task_root / "logs" / f"{kind}_{job_id}_attempt{retry_counts.get(f'{kind}:{job_id}', 0) + 1}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(freeze["runtime"][
                        "python_judge" if kind == "score" else "python_gpu"
                    ]), "-m",
                    "oasst1_filter.runtime.nat_spec", "formal-worker",
                    "--output-root", str(output_root), "--task-root", str(task_root),
                    "--kind", kind, "--job-id", job_id,
                    "--score-workers", str(effective["score"]),
                ]
                with log.open("ab") as handle:
                    process = subprocess.Popen(
                        command, stdout=handle, stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                    )
                record: dict[str, object] = {
                    "kind": kind, "job_id": job_id, "pid": process.pid,
                    "log": str(log), "started_utc": now.isoformat(),
                }
                attempts.append(record)
                active[(kind, job_id)] = (process, record)
            write_progress()
            time.sleep(2)
    finally:
        for process, _record in active.values():
            _stop_nat_msep_process(process)
        active.clear()
    work_finished = datetime.now(timezone.utc)
    milestones.setdefault("work_finished_utc", work_finished.isoformat())
    completed_training = sum(kind == "train" for kind, _ in completed)
    missing_reason = (
        "formal_48h_resource_cap" if status == "hard_cap_reached" else "not_scored"
    )
    rows = _nat_msep_position_rows(freeze, task_root, missing_reason=missing_reason)
    training_finished = datetime.fromisoformat(
        milestones.get("training_finished_utc", work_finished.isoformat())
    )
    phase_seconds = {
        "E0": (e0_finished - began).total_seconds(),
        "E1": max(0.0, (training_finished - e0_finished).total_seconds()),
        "E2": max(0.0, (work_finished - training_finished).total_seconds()),
        "E3": 0.0,
    }
    result = publish_nat_msep_formal(
        output_root=output_root, freeze=freeze, position_rows=rows,
        phase_seconds=phase_seconds, effective_concurrency=effective,
        repairs=repairs, completed_training=completed_training,
        forced_terminal=(
            "NAT_MSEP_20_INCOMPLETE_RESOURCE_CAP"
            if status == "hard_cap_reached" else None
        ),
    )
    status = "terminal"
    completed = _completed_nat_msep_formal_jobs(freeze, task_root)
    write_progress()
    return result


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def nat_msep_state_ids(roots: Sequence[int]) -> tuple[str, ...]:
    """Return the frozen root-blocked alternating F/M state order."""

    if (
        len(roots) != 8
        or len(set(roots)) != 8
        or any(type(root) is not int or not 0 < root < 2_147_483_647 for root in roots)
    ):
        raise ValueError("NAT-MSEP formal execution requires eight unique 31-bit roots")
    states: list[str] = []
    for index, root in enumerate(roots):
        arms = ("POL_FILTER", "POL_MSEP") if index % 2 == 0 else (
            "POL_MSEP",
            "POL_FILTER",
        )
        states.extend(f"{root}_{arm}" for arm in arms)
    return tuple(states)


def nat_msep_formal_jobs(roots: Sequence[int]) -> tuple[dict[str, str], ...]:
    states = nat_msep_state_ids(roots)
    return tuple(
        [{"kind": "train", "job_id": state} for state in states]
        + [{"kind": "generation", "job_id": state} for state in states]
        + [{"kind": "score", "job_id": state} for state in states]
    )


def next_nat_msep_formal_jobs(
    roots: Sequence[int],
    *,
    completed: set[tuple[str, str]],
    active: set[tuple[str, str]],
    concurrency: Mapping[str, int] = NAT_MSEP_FORMAL_CONCURRENCY,
) -> tuple[dict[str, str], ...]:
    """Advance the finite queue without consulting any model outcome."""

    jobs = nat_msep_formal_jobs(roots)
    identities = {(job["kind"], job["job_id"]) for job in jobs}
    if not completed <= identities or not active <= identities:
        raise ValueError("NAT-MSEP queue contains a foreign job")
    if set(concurrency) != set(NAT_MSEP_FORMAL_CONCURRENCY) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= NAT_MSEP_FORMAL_CONCURRENCY[kind]
        for kind in NAT_MSEP_FORMAL_CONCURRENCY
    ):
        raise ValueError("NAT-MSEP runtime concurrency exceeds 2/4/4 ceilings")
    training = jobs[:16]
    missing_training = [
        job
        for job in training
        if (job["kind"], job["job_id"]) not in completed | active
    ]
    if missing_training:
        if any(kind != "train" for kind, _ in active):
            return ()
        first = missing_training[0]
        pair_root = first["job_id"].split("_", 1)[0]
        slots = concurrency["train"] - sum(kind == "train" for kind, _ in active)
        return tuple(
            job
            for job in missing_training
            if job["job_id"].split("_", 1)[0] == pair_root
        )[: max(0, slots)]
    if any(kind == "train" for kind, _ in active):
        return ()
    if not all(("train", state) in completed for state in nat_msep_state_ids(roots)):
        return ()
    selected: list[dict[str, str]] = []
    for kind in ("generation", "score"):
        # Generation concurrency counts model processes. Scoring concurrency is
        # the worker count of one global judge queue, never states x workers.
        slots = (
            concurrency[kind] - sum(active_kind == kind for active_kind, _ in active)
            if kind == "generation"
            else int(not any(active_kind == "score" for active_kind, _ in active))
        )
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


def downshift_nat_msep_concurrency(
    current: Mapping[str, int], *, kind: str
) -> dict[str, int]:
    """Apply the next predeclared resource-only concurrency step."""

    ladders = {"train": (2, 1), "generation": (4, 2, 1), "score": (4, 2, 1)}
    if set(current) != set(ladders) or kind not in ladders:
        raise ValueError("NAT-MSEP concurrency state is invalid")
    result = dict(current)
    ladder = ladders[kind]
    try:
        index = ladder.index(int(current[kind]))
    except (ValueError, TypeError) as exc:
        raise ValueError("NAT-MSEP concurrency state is outside its ladder") from exc
    if index + 1 >= len(ladder):
        raise ValueError("NAT-MSEP concurrency is already at its serial floor")
    result[kind] = ladder[index + 1]
    if any(int(result[name]) > int(current[name]) for name in ladders):
        raise AssertionError("NAT-MSEP concurrency cannot increase")
    return result


def build_nat_msep_progress(
    *,
    scientific_identity_sha256: str,
    stage: str,
    completed_training: int,
    completed_generation_states: int,
    completed_score_states: int,
    generated_positions: int,
    scored_positions: int,
    active: Sequence[Mapping[str, object]],
    effective_concurrency: Mapping[str, int],
    started_utc: datetime,
    deadline_utc: datetime,
    now_utc: datetime | None = None,
    repair_count: int = 0,
    error: str | None = None,
) -> dict[str, object]:
    """Build the compact outcome-sealed supervisor heartbeat record."""

    if not _is_lower_sha256(scientific_identity_sha256):
        raise ValueError("NAT-MSEP progress identity is invalid")
    if stage not in {
        "root_gate", "training", "generation", "scoring", "publication", "terminal"
    }:
        raise ValueError("NAT-MSEP progress stage is invalid")
    if started_utc.tzinfo is None or deadline_utc.tzinfo is None:
        raise ValueError("NAT-MSEP progress clock must be timezone-aware")
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("NAT-MSEP progress observation must be timezone-aware")
    counts = (
        completed_training,
        completed_generation_states,
        completed_score_states,
        generated_positions,
        scored_positions,
        repair_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("NAT-MSEP progress counts must be nonnegative integers")
    if not (
        completed_training <= 16
        and completed_generation_states <= 16
        and completed_score_states <= 16
        and generated_positions <= 2_624
        and scored_positions <= 2_624
    ):
        raise ValueError("NAT-MSEP progress count exceeds the frozen queue")
    # Reuse queue validation for exact keys and ceilings.
    if set(effective_concurrency) != set(NAT_MSEP_FORMAL_CONCURRENCY) or any(
        type(effective_concurrency[key]) is not int
        or not 1 <= effective_concurrency[key] <= NAT_MSEP_FORMAL_CONCURRENCY[key]
        for key in NAT_MSEP_FORMAL_CONCURRENCY
    ):
        raise ValueError("NAT-MSEP progress concurrency is invalid")
    return {
        "schema": "nat-msep-20-formal-progress-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "scientific_identity_sha256": scientific_identity_sha256,
        "stage": stage,
        "completed_training": completed_training,
        "completed_generation_states": completed_generation_states,
        "completed_score_states": completed_score_states,
        "generated_positions": generated_positions,
        "scored_positions": scored_positions,
        "active": [dict(item) for item in active],
        "effective_concurrency": dict(effective_concurrency),
        "started_utc": started_utc.astimezone(timezone.utc).isoformat(),
        "deadline_utc": deadline_utc.astimezone(timezone.utc).isoformat(),
        "updated_utc": current.astimezone(timezone.utc).isoformat(),
        "elapsed_seconds": (current - started_utc).total_seconds(),
        "deadline_reached": current >= deadline_utc,
        "repair_count": repair_count,
        "error": error,
        "outcomes_sealed": True,
    }


def resolve_nat_msep_formal_pretraining_gate(
    *,
    root_rows: Sequence[dict[str, object]],
    update_count: int,
    plan_count: int,
) -> dict[str, object]:
    """Close the one-draw numeric gate only after every support and plan is valid."""

    if (
        len(root_rows) != 8
        or update_count != 800
        or plan_count != 16
        or len({row.get("root") for row in root_rows}) != 8
        or any(row.get("updates") != 100 for row in root_rows)
    ):
        raise ValueError("NAT-MSEP formal pretraining gate is incomplete")
    passed = all(
        row.get("id_gate") is True and row.get("payload_gate") is True
        for row in root_rows
    )
    return {
        "passed": passed,
        "terminal": (
            None if passed else "NAT_MSEP_20_FORMAL_ROOT_SET_NOT_QUALIFIED"
        ),
        "training_job_count": 16 if passed else 0,
        "redraw_allowed": False,
    }


def build_nat_msep_execution_freeze(
    *,
    claim: Mapping[str, object],
    prior_roots: Sequence[int],
    roots: Sequence[int],
    plans: Sequence[Mapping[str, object]],
    mask_rows: Sequence[Mapping[str, object]],
    root_rows: Sequence[Mapping[str, object]],
    task_ids: Sequence[str],
    bindings: Mapping[str, object],
    paths: Mapping[str, object],
    python_gpu: str,
    python_judge: str,
) -> dict[str, object]:
    """Build the one immutable post-root-gate execution contract."""

    from ..arms.paired_filtering import (
        NAT_MSEP_FORMAL_MASK_NAMESPACE,
        NAT_MSEP_FORMAL_ROOT_NAMESPACE,
    )

    states = nat_msep_state_ids(roots)
    prior = tuple(sorted(set(prior_roots)))
    if any(type(root) is not int or root <= 0 for root in prior):
        raise ValueError("NAT-MSEP prior-root registry is invalid")
    registry_sha = hashlib.sha256(
        json.dumps(prior, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_mask_namespace = (
        f"{NAT_MSEP_FORMAL_MASK_NAMESPACE}/{claim.get('mask_seed_hex', '')}"
    )
    try:
        began = datetime.fromisoformat(str(claim["started_utc"]))
        deadline = datetime.fromisoformat(str(claim["deadline_utc"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("NAT-MSEP formal claim clock is invalid") from exc
    claim_valid = (
        claim.get("schema") == "nat-msep-20-formal-randomization-claim-v1"
        and claim.get("protocol_id") == NAT_MSEP_PROTOCOL_ID
        and claim.get("root_namespace") == NAT_MSEP_FORMAL_ROOT_NAMESPACE
        and claim.get("mask_base_namespace") == NAT_MSEP_FORMAL_MASK_NAMESPACE
        and claim.get("mask_namespace") == expected_mask_namespace
        and claim.get("prior_root_registry_sha256") == registry_sha
        and claim.get("hard_cap_resets") is False
        and deadline - began == timedelta(hours=48)
        and began.tzinfo is not None
        and deadline.tzinfo is not None
        and all(
            isinstance(claim.get(key), str)
            and len(str(claim[key])) == 64
            and str(claim[key]) == str(claim[key]).lower()
            and all(c in "0123456789abcdef" for c in str(claim[key]))
            for key in ("root_seed_hex", "mask_seed_hex")
        )
    )
    if not claim_valid:
        raise ValueError("NAT-MSEP formal randomization claim is invalid")
    expected_plans = [
        (int(state.split("_", 1)[0]), state.split("_", 1)[1]) for state in states
    ]
    actual_plans = [(item.get("root"), item.get("arm")) for item in plans]
    if actual_plans != expected_plans or any(
        item.get("updates") != 100
        or item.get("response_tokens") != 583_975
        or not _is_lower_sha256(item.get("plan_sha256"))
        or not _is_lower_sha256(item.get("plan_file_sha256"))
        or not isinstance(item.get("plan_path"), str)
        for item in plans
    ):
        raise ValueError("NAT-MSEP formal plan registry differs")
    mask_keys = [(row.get("root"), row.get("update")) for row in mask_rows]
    expected_mask_keys = [(root, update) for root in roots for update in range(100)]
    if mask_keys != expected_mask_keys or any(
        type(row.get("rank")) is not int
        or int(row["rank"]) < 0
        or not isinstance(row.get("selected_ordinals"), list)
        for row in mask_rows
    ):
        raise ValueError("NAT-MSEP formal mask manifest differs")
    gate = resolve_nat_msep_formal_pretraining_gate(
        root_rows=[dict(row) for row in root_rows],
        update_count=len(mask_rows),
        plan_count=len(plans),
    )
    expected_tasks = [f"HumanEval/{index}" for index in range(164)]
    if list(task_ids) != expected_tasks:
        raise ValueError("NAT-MSEP endpoint must be HumanEval/0..163")
    required_bindings = {
        "model_revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "humaneval_compressed_sha256": "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
        "humaneval_logical_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
        "evaluator_revision": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        "evaluator_archive_sha256": "d11dc64dfee6858a52c493466aa91a7bd0e3d50deda9a586144c1ba127c0e869",
    }
    if any(bindings.get(key) != value for key, value in required_bindings.items()):
        raise ValueError("NAT-MSEP model/evaluator binding differs")
    if not all(isinstance(paths.get(key), str) and paths.get(key) for key in (
        "model", "asset", "evaluator_source", "evaluator_archive"
    )) or not python_gpu or not python_judge:
        raise ValueError("NAT-MSEP runtime paths are incomplete")
    freeze: dict[str, object] = {
        "schema": "nat-msep-20-execution-freeze-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "state": (
            "formal_root_set_qualified"
            if gate["passed"]
            else "formal_root_set_not_qualified"
        ),
        "claim_sha256": _canonical_json_sha256(dict(claim)),
        "claim": dict(claim),
        "prior_root_registry": list(prior),
        "prior_root_registry_sha256": registry_sha,
        "roots": list(roots),
        "plans": [dict(plan) for plan in plans],
        "mask_rows": [dict(row) for row in mask_rows],
        "root_gates": [dict(row) for row in root_rows],
        "task_ids": list(task_ids),
        "training_job_count": int(gate["training_job_count"]),
        "state_count": 16,
        "position_count": 2_624,
        "concurrency": dict(NAT_MSEP_FORMAL_CONCURRENCY),
        "resource_ladders": {
            "train": [2, 1], "generation": [4, 2, 1], "score": [4, 2, 1]
        },
        "training_recipe": {
            "response_tokens": 583_975,
            "background_tokens": 467_180,
            "component_tokens": 116_795,
            "updates": 100,
            "max_length": 2_048,
            "microbatch": 1,
        },
        "generation_recipe": {
            "do_sample": False,
            "max_new_tokens": 1_024,
            "microbatch": 1,
            "eos_token_id": 151_643,
            "pad_token_id": 151_643,
            "suffix_only": True,
            "tabs_to_four_spaces": True,
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
        "job_order": list(nat_msep_formal_jobs(roots)),
        "terminal_set": sorted(NAT_MSEP_FORMAL_TERMINALS),
        "root_gate": gate,
        "formal_outcomes_opened": False,
    }
    freeze["scientific_identity_sha256"] = _canonical_json_sha256(freeze)
    return freeze


_MSEP_FORMAL_CONTENT_NAMES = (
    "NAT_MSEP_20_EXECUTION_FREEZE.json",
    "NAT_MSEP_20_POSITION_TABLE.csv",
    "NAT_MSEP_20_ROOT_TABLE.csv",
    "NAT_MSEP_20_RESULT.json",
    "NAT_MSEP_20_REPORT.md",
)
_MSEP_FORMAL_POSITION_FIELDS = (
    "state_id", "root", "arm", "task_id", "Y", "verdict",
    "completion_sha256", "missing_reason",
)
_MSEP_FORMAL_ROOT_FIELDS = (
    "root", "F_pass", "F_valid", "M_pass", "M_valid", "F_minus_M",
    "delta", "bound_lower", "bound_upper", "missing_F", "missing_M",
    "updates", "D_id", "delta_id_num", "delta_id_den", "id_gate",
    "D_payload", "delta_payload_num", "delta_payload_den", "payload_gate",
)


def _write_nat_msep_formal_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                field: "null" if row.get(field) is None else row.get(field)
                for field in fields
            }
            for row in rows
        )
    os.replace(temporary, path)


def publish_nat_msep_formal(
    *,
    output_root: Path,
    freeze: Mapping[str, object],
    position_rows: Sequence[Mapping[str, object]],
    phase_seconds: Mapping[str, float],
    effective_concurrency: Mapping[str, int],
    repairs: Sequence[str],
    completed_training: int = 16,
    forced_terminal: str | None = None,
    now_utc: datetime | None = None,
) -> dict[str, object]:
    """Publish and verify the protocol's one minimal five-file delivery."""

    from ..analysis.codecontests import analyze_nat_msep_positions

    publication_started = time.perf_counter()

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    archive_name = "NAT_MSEP_20_DELIVERY.zip"
    allowed = set(_MSEP_FORMAL_CONTENT_NAMES) | {
        archive_name, f"{archive_name}.sha256"
    }
    unknown = {path.name for path in output_root.iterdir()} - allowed
    if unknown:
        raise RuntimeError(f"NAT-MSEP formal output contains unknown entries: {sorted(unknown)}")
    if freeze.get("protocol_id") != NAT_MSEP_PROTOCOL_ID:
        raise ValueError("NAT-MSEP formal freeze protocol differs")
    freeze_path = output_root / _MSEP_FORMAL_CONTENT_NAMES[0]
    if freeze_path.exists():
        if json.loads(freeze_path.read_text("utf-8")) != dict(freeze):
            raise ValueError("NAT-MSEP immutable execution freeze differs")
    else:
        _atomic_json(freeze_path, dict(freeze))

    roots = tuple(int(root) for root in freeze.get("roots", ()))
    pre_freeze = freeze.get("state") == "pre_freeze_terminal"
    root_gate_failed = (
        isinstance(freeze.get("root_gate"), Mapping)
        and freeze["root_gate"].get("passed") is False  # type: ignore[index]
    )
    if pre_freeze:
        if position_rows or roots:
            raise ValueError("pre-freeze terminal cannot publish scientific rows")
        root_rows: list[dict[str, object]] = []
        analysis: dict[str, object] = {
            "complete": False, "valid_positions": 0, "mean": None,
            "root_sd": None, "working_interval": [None, None],
        }
    elif root_gate_failed:
        if position_rows or len(roots) != 8:
            raise ValueError("root-gate terminal row contract differs")
        root_rows = []
        for row in freeze.get("root_gates", ()):
            published = {
                "root": int(row["root"]),
                "F_pass": None, "F_valid": None, "M_pass": None,
                "M_valid": None, "F_minus_M": None, "delta": None,
                "bound_lower": None, "bound_upper": None,
                "missing_F": 164, "missing_M": 164,
            }
            published.update(dict(row))
            root_rows.append(published)
        if len(root_rows) != 8:
            raise ValueError("root-gate terminal requires eight gate rows")
        analysis = {
            "complete": False, "valid_positions": 0, "mean": None,
            "root_sd": None, "working_interval": [None, None],
        }
        forced_terminal = forced_terminal or "NAT_MSEP_20_FORMAL_ROOT_SET_NOT_QUALIFIED"
    else:
        analysis = analyze_nat_msep_positions(
            position_rows=position_rows, roots=roots
        )
        root_rows = list(analysis["root_rows"])  # type: ignore[arg-type]
    terminal = forced_terminal or analysis.get("terminal")
    if terminal not in NAT_MSEP_FORMAL_TERMINALS:
        raise ValueError("NAT-MSEP formal publication lacks a legal terminal")
    if analysis.get("complete") and forced_terminal is not None:
        raise ValueError("a complete effect panel cannot override its terminal")

    _write_nat_msep_formal_csv(
        output_root / _MSEP_FORMAL_CONTENT_NAMES[1],
        position_rows,
        _MSEP_FORMAL_POSITION_FIELDS,
    )
    _write_nat_msep_formal_csv(
        output_root / _MSEP_FORMAL_CONTENT_NAMES[2],
        root_rows,
        _MSEP_FORMAL_ROOT_FIELDS,
    )
    current = now_utc or datetime.now(timezone.utc)
    began_text = freeze.get("claim", {}).get("started_utc") if isinstance(
        freeze.get("claim"), Mapping
    ) else None
    began = datetime.fromisoformat(str(began_text)) if began_text else current
    wall_seconds = max(0.0, (current - began).total_seconds())
    report = (
        "# NAT-MSEP-20 formal result\n\n"
        f"Terminal: `{terminal}`. New-root gate: "
        f"`{'pass' if not root_gate_failed and not pre_freeze else 'fail' if root_gate_failed else 'not_available'}`.\n\n"
        f"Training terminals: {completed_training}/16. Valid HumanEval+ positions: "
        f"{analysis.get('valid_positions', 0)}/2624.\n\n"
        f"Mean F-M: {analysis.get('mean')}; sample SD: {analysis.get('root_sd')}; "
        f"descriptive 95% working interval: {analysis.get('working_interval')}.\n\n"
        "HumanEval+164 is an adaptively reused exploratory endpoint; the result is "
        "not confirmatory evidence and does not establish per-sample causality.\n"
    )
    _atomic_text(output_root / _MSEP_FORMAL_CONTENT_NAMES[4], report)
    if now_utc is None:
        current = datetime.now(timezone.utc)
        wall_seconds = max(0.0, (current - began).total_seconds())
    phases = {key: float(value) for key, value in phase_seconds.items()}
    phases["E3"] = phases.get("E3", 0.0) + (
        0.0 if now_utc is not None else time.perf_counter() - publication_started
    )
    result: dict[str, object] = {
        "schema": "nat-msep-20-formal-result-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "terminal": terminal,
        "scientific_identity_sha256": freeze.get("scientific_identity_sha256"),
        "new_root_gate_passed": False if root_gate_failed else None if pre_freeze else True,
        "completed_training": completed_training,
        "position_count": 2_624 if not pre_freeze and not root_gate_failed else 0,
        "valid_positions": analysis.get("valid_positions", 0),
        "complete": bool(analysis.get("complete")),
        "root_rows": root_rows,
        "mean_F_minus_M": analysis.get("mean"),
        "root_sd": analysis.get("root_sd"),
        "working_interval_df7": analysis.get("working_interval"),
        "investment_scale": 0.02,
        "actual_formal_wall_seconds": wall_seconds,
        "phase_seconds": phases,
        "initial_concurrency": dict(NAT_MSEP_FORMAL_CONCURRENCY),
        "effective_concurrency": dict(effective_concurrency),
        "engineering_repairs": list(repairs),
        "interpretation": (
            "Adaptively reused HumanEval+164 exploratory active contrast; not a "
            "confirmatory confidence statement or proof of individual-sample harm."
        ),
    }
    # RESULT is the final atomic scientific commit. Packaging follows without
    # altering any of the five scientific contents.
    _atomic_json(output_root / _MSEP_FORMAL_CONTENT_NAMES[3], result)
    archive = output_root / archive_name
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name in _MSEP_FORMAL_CONTENT_NAMES:
            handle.write(output_root / name, name)
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as handle:
        if tuple(handle.namelist()) != _MSEP_FORMAL_CONTENT_NAMES or handle.testzip() is not None:
            raise ValueError("NAT-MSEP formal delivery ZIP differs")
    _atomic_text(
        output_root / f"{archive_name}.sha256",
        f"{_sha256_file(archive)}  {archive_name}\n",
    )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_training_normalization() -> None:
    """Inspect the trainer without importing its torch-bearing module."""

    trainer = Path(__file__).resolve().parents[1] / "train" / "run_codecontests.py"
    tree = ast.parse(trainer.read_text("utf-8"), filename=str(trainer))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "complete_codecontests_update"
        ),
        None,
    )
    if function is None:
        raise NatSpecQualificationError(
            "budget_or_component_contract_drift",
            "CodeContests update function is absent",
        )
    attributes = {
        node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
    }
    has_token_weight = any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "retained_response_tokens"
        and isinstance(node.right, ast.Name)
        and node.right.id == "active_tokens"
        for node in ast.walk(function)
    )
    has_shifted_mask_check = any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "labels"
        for node in ast.walk(function)
    )
    if (
        "retained_response_tokens" not in attributes
        or not has_token_weight
        or not has_shifted_mask_check
        or {"component", "unit_id"} & attributes
    ):
        raise NatSpecQualificationError(
            "budget_or_component_contract_drift",
            "trainer normalization is not response-token-only",
        )


def _validate_prepared_inputs(
    prepared_root: Path, roots: Sequence[int]
) -> dict[int, dict[str, Path]]:
    freeze_path = prepared_root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
    draw_path = prepared_root / "ROOT_DRAW.json"
    try:
        freeze = json.loads(freeze_path.read_text("utf-8"))
        draw = json.loads(draw_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC frozen inputs are unreadable"
        ) from exc
    frozen_roots = tuple(freeze.get("roots", ()))
    if (
        freeze.get("schema") != "nat-replica-x164-formal-freeze-v1"
        or freeze.get("protocol_id") != NAT_REPLICA_PROTOCOL_ID
        or tuple(draw.get("roots", ())) != frozen_roots
        or tuple(roots) != tuple(root for root in frozen_roots if root in set(roots))
        or len(set(roots)) != len(roots)
    ):
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC root/freeze identity differs"
        )
    expected_draw_hash = freeze.get("bindings", {}).get("root_draw_sha256")
    if expected_draw_hash != _sha256_file(draw_path):
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC root draw bytes differ"
        )
    records = {
        (int(item["root"]), str(item["arm"])): item
        for item in freeze.get("plans", ())
        if isinstance(item, dict) and "root" in item and "arm" in item
    }
    file_hashes = freeze.get("bindings", {}).get("plan_file_hashes", {})
    resolved: dict[int, dict[str, Path]] = {}
    for root in roots:
        resolved[root] = {}
        for arm in ("POL_KEEP", "POL_FILTER"):
            record = records.get((root, arm))
            if record is None:
                raise NatSpecQualificationError(
                    "source_or_plan_identity_drift", "NAT-SPEC plan record is absent"
                )
            relative = str(record.get("plan_path"))
            path = prepared_root / relative
            if file_hashes.get(relative) != _sha256_file(path):
                raise NatSpecQualificationError(
                    "source_or_plan_identity_drift", "NAT-SPEC plan bytes differ"
                )
            plan = load_s12_training_plan(path)
            if (
                s12_training_plan_sha256(plan) != record.get("plan_sha256")
                or int(record.get("updates", -1)) != 100
                or int(record.get("response_tokens", -1)) != 583_975
            ):
                raise NatSpecQualificationError(
                    "source_or_plan_identity_drift", "NAT-SPEC plan semantics differ"
                )
            resolved[root][arm] = path
    _validate_training_normalization()
    return resolved


def _qualify_root_worker(
    root: int, keep_path: str, filtered_path: str
) -> NatSpecRootQualification:
    return qualify_nat_spec_plan_pair(
        root=root,
        keep_plan=load_s12_training_plan(Path(keep_path)),
        filtered_plan=load_s12_training_plan(Path(filtered_path)),
    )


def qualify_nat_spec_roots(
    prepared_root: Path,
    *,
    roots: Sequence[int],
    max_workers: int,
) -> tuple[NatSpecRootQualification, ...]:
    """Qualify roots independently, returning canonical frozen-root order."""

    root_order = tuple(roots)
    if not 1 <= max_workers <= 8:
        raise ValueError("NAT-SPEC CPU worker count must be in 1..8")
    resolved = _validate_prepared_inputs(Path(prepared_root), root_order)
    worker_count = min(max_workers, len(root_order))
    if worker_count == 1:
        values = [
            _qualify_root_worker(
                root,
                str(resolved[root]["POL_KEEP"]),
                str(resolved[root]["POL_FILTER"]),
            )
            for root in root_order
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                root: pool.submit(
                    _qualify_root_worker,
                    root,
                    str(resolved[root]["POL_KEEP"]),
                    str(resolved[root]["POL_FILTER"]),
                )
                for root in root_order
            }
            values = [futures[root].result() for root in root_order]
    return tuple(values)


def validate_nat_msep_prepared_inputs(
    prepared_root: Path, roots: Sequence[int]
) -> dict[int, dict[str, Path]]:
    """Validate only names, schemas, frozen identities, hashes, and trainer AST."""

    try:
        return _validate_prepared_inputs(Path(prepared_root), tuple(roots))
    except NatSpecQualificationError as exc:
        raise NatMsepQualificationError(exc.reason_code, str(exc)) from exc


def _synthetic_msep_probe(root: int, stress: bool = False) -> dict[str, object]:
    if stress:
        weights = (1,) * 20 + (11,) * 15
        k, q = 10, 10
    else:
        weights = (1,) * 12
        k, q = 4, 4
    exposures = tuple(
        NatMsepAnonymousExposure(
            f"r{root}:u0:e{ordinal}",
            weight,
            ordinal < 4,
            ((0, ordinal), (True,)),
        )
        for ordinal, weight in enumerate(weights)
    )
    support = build_nat_msep_subset_support(exposures, k=k, q=q)
    rank = nat_msep_qualification_rank(root=root, update=0, omega=support.omega2)
    subset = support.unrank(rank)
    identity = hashlib.sha256(
        json.dumps(
            {
                "omega": support.omega,
                "omega1": support.omega1,
                "omega2": support.omega2,
                "id": support.optimum_id_overlap,
                "payload": support.optimum_payload_overlap,
                "rank": rank,
                "subset": subset,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "root": root,
        "exposures": len(exposures),
        "omega": support.omega,
        "omega1": support.omega1,
        "omega2": support.omega2,
        "identity_sha256": identity,
    }


def run_nat_msep_preflight(
    prepared_root: Path, *, roots: Sequence[int], max_workers: int = 8
) -> dict[str, object]:
    """Run asset-only real checks and entirely synthetic scientific probes."""

    if not 1 <= max_workers <= 8:
        raise ValueError("NAT-MSEP CPU worker count must be in 1..8")
    before = time.perf_counter()
    validate_nat_msep_prepared_inputs(Path(prepared_root), tuple(roots))
    serial = tuple(_synthetic_msep_probe(root) for root in roots)
    with ProcessPoolExecutor(max_workers=min(max_workers, len(roots))) as pool:
        futures = {root: pool.submit(_synthetic_msep_probe, root) for root in roots}
        parallel = tuple(futures[root].result() for root in roots)
    if serial != parallel:
        raise RuntimeError("NAT-MSEP synthetic C1/C8 parity differs")
    stress = _synthetic_msep_probe(int(roots[0]), True)
    if int(stress["exposures"]) < 35 or int(stress["omega"]) < 60_172:
        raise RuntimeError("NAT-MSEP synthetic stress coverage is insufficient")
    return {
        "schema": "nat-msep-20-preflight-v1",
        "roots": list(roots),
        "asset_root_count": len(roots),
        "synthetic_parity": True,
        "synthetic_stress": stress,
        "requested_workers": max_workers,
        "effective_workers": min(max_workers, len(roots)),
        "wall_seconds": round(time.perf_counter() - before, 6),
        "historical_msep_updates_calculated": 0,
        "model_outcomes_read": False,
        "gpu_processes_started": 0,
        "model_weights_loaded": 0,
        "programs_generated_or_scored": 0,
    }


def _qualify_msep_root_worker(
    root: int, keep_path: str, filtered_path: str, deadline_epoch: float
) -> NatMsepRootQualification:
    return qualify_nat_msep_plan_pair(
        root=root,
        keep_plan=load_s12_training_plan(Path(keep_path)),
        filtered_plan=load_s12_training_plan(Path(filtered_path)),
        deadline_epoch=deadline_epoch,
    )


def qualify_nat_msep_roots(
    prepared_root: Path,
    *,
    roots: Sequence[int],
    max_workers: int,
    deadline_epoch: float,
    progress_path: Path | None = None,
    started_epoch: float | None = None,
    recovery_count: int = 0,
) -> tuple[NatMsepRootQualification, ...]:
    """Run one sequential worker per root and return only complete roots."""

    root_order = tuple(roots)
    if not 1 <= max_workers <= 8:
        raise ValueError("NAT-MSEP CPU worker count must be in 1..8")
    resolved = validate_nat_msep_prepared_inputs(Path(prepared_root), root_order)
    complete: dict[int, NatMsepRootQualification] = {}
    worker_count = min(max_workers, len(root_order))
    start = time.time() if started_epoch is None else started_epoch

    def publish_progress(phase: str) -> None:
        if progress_path is None:
            return
        _atomic_json(
            progress_path,
            {
                "schema": "nat-msep-20-progress-v1",
                "protocol_id": NAT_MSEP_PROTOCOL_ID,
                "phase": phase,
                "planned_roots": len(root_order),
                "completed_roots": len(complete),
                "planned_updates": 100 * len(root_order),
                "completed_updates": 100 * len(complete),
                "actual_concurrency": worker_count,
                "wall_seconds": round(time.time() - start, 6),
                "deadline_epoch": deadline_epoch,
                "supervisor_pid": __import__("os").getpid(),
                "recovery_count": recovery_count,
            },
        )

    publish_progress("formal_cpu_census")
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _qualify_msep_root_worker,
                root,
                str(resolved[root]["POL_KEEP"]),
                str(resolved[root]["POL_FILTER"]),
                deadline_epoch,
            ): root
            for root in root_order
        }
        for future in as_completed(futures):
            root = futures[future]
            try:
                value = future.result()
            except NatMsepDeadlineExceeded:
                continue
            complete[root] = value
            publish_progress("formal_cpu_census")
    publish_progress("science_rows_closed")
    return tuple(complete[root] for root in root_order if root in complete)


_CONTENT_NAMES = (
    "NAT_SPEC_20_QUALIFICATION_FREEZE.json",
    "NAT_SPEC_20_QUALIFICATION_UPDATE_TABLE.csv",
    "NAT_SPEC_20_QUALIFICATION_ROOT_TABLE.csv",
    "NAT_SPEC_20_QUALIFICATION_RESULT.json",
    "NAT_SPEC_20_QUALIFICATION_REPORT.md",
)
_MSEP_CONTENT_NAMES = (
    "NAT_MSEP_20_QUALIFICATION_FREEZE.json",
    "NAT_MSEP_20_QUALIFICATION_UPDATE_TABLE.csv",
    "NAT_MSEP_20_QUALIFICATION_ROOT_TABLE.csv",
    "NAT_MSEP_20_QUALIFICATION_RESULT.json",
    "NAT_MSEP_20_QUALIFICATION_REPORT.md",
)
_MSEP_ARCHIVE_NAME = "NAT_MSEP_20_QUALIFICATION_DELIVERY.zip"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, "utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def ensure_nat_msep_formal_freeze(
    output_root: Path,
    base_freeze: dict[str, object],
    *,
    now_epoch: float | None = None,
) -> dict[str, object]:
    """Atomically establish, or validate and reuse, the one formal clock."""

    output_root = Path(output_root)
    path = output_root / "NAT_MSEP_20_QUALIFICATION_FREEZE.json"
    if path.exists():
        existing = json.loads(path.read_text("utf-8"))
        if any(existing.get(key) != value for key, value in base_freeze.items()):
            raise RuntimeError("NAT-MSEP existing formal freeze differs from inputs")
        if (
            existing.get("formal_deadline_epoch")
            != existing.get("formal_start_epoch") + 7200
        ):
            raise RuntimeError("NAT-MSEP existing formal clock is invalid")
        return existing
    started = time.time() if now_epoch is None else float(now_epoch)
    payload = {
        **base_freeze,
        "formal_start_epoch": started,
        "formal_deadline_epoch": started + 7200,
        "formal_start_utc": datetime.fromtimestamp(
            started, timezone.utc
        ).isoformat(),
        "formal_deadline_utc": datetime.fromtimestamp(
            started + 7200, timezone.utc
        ).isoformat(),
        "wall_limit_seconds": 7200,
        "clock_resets": False,
    }
    _atomic_json(path, payload)
    return payload


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _fraction_fields(value: Fraction | None) -> dict[str, object]:
    if value is None:
        return {
            "fraction": None,
            "numerator": None,
            "denominator": None,
            "decimal": None,
        }
    return {
        "fraction": f"{value.numerator}/{value.denominator}",
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": format(float(value), ".12g"),
    }


def _qualification_freeze(prepared_root: Path, source: dict[str, object]) -> dict[str, object]:
    plans = []
    file_hashes = source["bindings"]["plan_file_hashes"]  # type: ignore[index]
    for item in source["plans"]:  # type: ignore[index]
        relative = str(item["plan_path"])
        plans.append(
            {
                "root": int(item["root"]),
                "arm": str(item["arm"]),
                "plan_path": relative,
                "plan_sha256": str(item["plan_sha256"]),
                "plan_file_sha256": str(file_hashes[relative]),
            }
        )
    return {
        "schema": "nat-spec-20-qualification-freeze-v1",
        "protocol_id": NAT_SPEC_PROTOCOL_ID,
        "source_protocol_id": NAT_REPLICA_PROTOCOL_ID,
        "source_freeze_sha256": _sha256_file(
            prepared_root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
        ),
        "roots": list(source["roots"]),
        "plans": plans,
        "planned_root_count": 8,
        "planned_update_count": 800,
        "cpu_worker_target": 8,
        "cpu_worker_max": 8,
        "qualification_namespace": NAT_SPEC_QUALIFICATION_NAMESPACE,
        "wall_limit_seconds": 7200,
        "prohibited": [
            "model_outcome_read",
            "cuda_or_gpu",
            "model_weight_load",
            "training",
            "program_generation_or_scoring",
            "formal_root_creation",
            "formal_mask_creation",
        ],
    }


def _publish_delivery(
    *,
    output_root: Path,
    freeze: dict[str, object],
    roots: Sequence[NatSpecRootQualification],
    started_utc: str,
    finished_utc: str,
    wall_seconds: float,
    actual_concurrency: int,
    failure_reason: str | None,
    repairs: Sequence[str],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    unknown = {path.name for path in output_root.iterdir()} - set(_CONTENT_NAMES) - {
        "NAT_SPEC_20_QUALIFICATION_DELIVERY.zip",
        "NAT_SPEC_20_QUALIFICATION_DELIVERY.zip.sha256",
    }
    if unknown:
        raise RuntimeError(f"NAT-SPEC output directory contains unknown entries: {sorted(unknown)}")
    _atomic_json(output_root / _CONTENT_NAMES[0], freeze)

    root_order = {int(root): index for index, root in enumerate(freeze["roots"])}
    update_rows = [dict(row) for root in roots for row in root.update_rows]
    update_rows.sort(
        key=lambda row: (root_order[int(row["root"])], int(row["update"]))
    )
    root_rows: list[dict[str, object]] = []
    for value in roots:
        root_rows.append(
            {
                "root": value.root,
                "updates": len(value.update_rows),
                "keep_tokens": value.keep_tokens,
                "filtered_tokens": value.filtered_tokens,
                "target_tokens": value.target_tokens,
                "donor_tokens": value.donor_tokens,
                "alpha_num": value.alpha.numerator,
                "alpha_den": value.alpha.denominator,
                "alpha_decimal": format(float(value.alpha), ".12g"),
                "singleton_updates": sum(
                    bool(row["omega_is_singleton"]) for row in value.update_rows
                ),
                "repeated_exposure_count": value.repeated_exposure_count,
                "sample_fully_removed_source_records": value.sample_fully_removed_source_records,
                "sample_partially_removed_source_records": value.sample_partially_removed_source_records,
                "sample_retained_source_records": value.sample_retained_source_records,
            }
        )
    root_rows.sort(key=lambda row: root_order[int(row["root"])])
    update_fields = (
        "root",
        "update",
        "exposures",
        "k",
        "q",
        "omega",
        "log10_omega",
        "omega_is_singleton",
        "qualification_rank",
        "rank_roundtrips",
        "expected_i_overlap_num",
        "expected_i_overlap_den",
        "sample_actual_i_overlap",
        "repeated_exposure_count",
        "sample_fully_removed_source_records",
        "sample_partially_removed_source_records",
        "sample_retained_source_records",
    )
    root_fields = tuple(root_rows[0].keys()) if root_rows else (
        "root",
        "updates",
        "keep_tokens",
        "filtered_tokens",
        "target_tokens",
        "donor_tokens",
        "alpha_num",
        "alpha_den",
        "alpha_decimal",
        "singleton_updates",
        "repeated_exposure_count",
        "sample_fully_removed_source_records",
        "sample_partially_removed_source_records",
        "sample_retained_source_records",
    )
    _write_csv(output_root / _CONTENT_NAMES[1], update_rows, update_fields)
    _write_csv(output_root / _CONTENT_NAMES[2], root_rows, root_fields)

    complete = len(update_rows) == 800 and len(root_rows) == 8 and failure_reason is None
    independently_recomputed: list[Fraction] = []
    if complete:
        by_root: dict[int, Fraction] = {}
        with (output_root / _CONTENT_NAMES[1]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                root = int(row["root"])
                by_root[root] = by_root.get(root, Fraction(0, 1)) + Fraction(
                    int(row["expected_i_overlap_num"]),
                    int(row["expected_i_overlap_den"]),
                )
        independently_recomputed = [
            by_root[int(root)] / 116_795 for root in freeze["roots"]
        ]
        if independently_recomputed != [value.alpha for value in roots]:
            raise RuntimeError("NAT-SPEC independent alpha closure differs")
        resolved = resolve_nat_spec_terminal(
            completed_updates=800,
            completed_roots=8,
            root_alphas=independently_recomputed,
        )
    else:
        resolved = resolve_nat_spec_terminal(
            completed_updates=len(update_rows),
            completed_roots=len(root_rows),
            root_alphas=(),
            failure_reason=failure_reason,
        )
    alpha_bar = resolved["alpha_bar"]
    result: dict[str, object] = {
        "schema": "nat-spec-20-qualification-result-v1",
        "protocol_id": NAT_SPEC_PROTOCOL_ID,
        "terminal": resolved["terminal"],
        "reason_code": resolved["reason_code"],
        "planned_update_count": 800,
        "completed_update_count": len(update_rows),
        "planned_root_count": 8,
        "completed_root_count": len(root_rows),
        "root_alphas": [
            {"root": int(root), **_fraction_fields(alpha)}
            for root, alpha in zip(freeze["roots"], independently_recomputed, strict=False)
        ],
        "alpha_bar": _fraction_fields(alpha_bar if isinstance(alpha_bar, Fraction) else None),
        "gates": {
            "source_and_plan_identity": "pass" if failure_reason is None else "not_evaluated",
            "budget_and_component_contract": "pass" if complete else "not_evaluated",
            "exact_uniform_sampler": "pass" if complete else "not_evaluated",
            "control_overlap_at_most_half": resolved["alpha_gate"],
        },
        "qualification_sample_reusable_for_formal_training": False,
        "actual_cpu_concurrency": actual_concurrency,
        "cpu_worker_max": 8,
        "formal_census_started_utc": started_utc,
        "formal_census_finished_utc": finished_utc,
        "formal_census_wall_seconds": round(wall_seconds, 6),
        "engineering_repairs": list(repairs),
        "model_outcomes_read": False,
        "gpu_processes_started": 0,
        "model_weights_loaded": 0,
        "programs_generated_or_scored": 0,
        "formal_roots_created": 0,
        "formal_masks_created": 0,
        "training_runs_started": 0,
    }
    _atomic_json(output_root / _CONTENT_NAMES[3], result)
    alpha_text = (
        result["alpha_bar"]["fraction"]  # type: ignore[index]
        if result["alpha_bar"]["fraction"] is not None  # type: ignore[index]
        else "not_evaluated"
    )
    report = (
        "# NAT-SPEC-20 CPU qualification\n\n"
        f"Terminal: `{result['terminal']}`. Reason: "
        f"`{result['reason_code'] or 'none'}`.\n\n"
        f"Completed {len(update_rows)}/800 updates and {len(root_rows)}/8 roots; "
        f"exact alpha_bar is `{alpha_text}`. The qualification-only sample is not a "
        "formal training mask.\n\n"
        f"Actual CPU concurrency was {actual_concurrency}; formal census wall time was "
        f"{wall_seconds:.6f} seconds. Engineering repairs: "
        f"{', '.join(repairs) if repairs else 'none'}.\n\n"
        "This result assesses only construction eligibility. It does not report model "
        "quality or authorize formal training. No model outcome was read; no GPU, model "
        "weight, generation, scoring, formal root, formal mask, or training run was used.\n"
    )
    _atomic_text(output_root / _CONTENT_NAMES[4], report)

    archive_path = output_root / "NAT_SPEC_20_QUALIFICATION_DELIVERY.zip"
    temporary_archive = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _CONTENT_NAMES:
            archive.write(output_root / name, arcname=name)
    temporary_archive.replace(archive_path)
    archive_hash = _sha256_file(archive_path)
    _atomic_text(
        output_root / "NAT_SPEC_20_QUALIFICATION_DELIVERY.zip.sha256",
        f"{archive_hash}  {archive_path.name}\n",
    )
    return result


def _nat_msep_base_freeze(
    prepared_root: Path, source: dict[str, object]
) -> dict[str, object]:
    plans = []
    bindings = source.get("bindings", {})
    file_hashes = bindings.get("plan_file_hashes", {}) if isinstance(bindings, dict) else {}
    for item in source.get("plans", ()):
        if not isinstance(item, dict):
            continue
        relative = str(item.get("plan_path", ""))
        plans.append(
            {
                "root": int(item["root"]),
                "arm": str(item["arm"]),
                "plan_path": relative,
                "plan_sha256": str(item.get("plan_sha256", "")),
                "plan_file_sha256": str(file_hashes.get(relative, "")),
            }
        )
    source_path = prepared_root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
    return {
        "schema": "nat-msep-20-qualification-freeze-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "source_protocol_id": NAT_REPLICA_PROTOCOL_ID,
        "source_freeze_sha256": _sha256_file(source_path),
        "roots": [int(root) for root in source.get("roots", ())],
        "plans": plans,
        "planned_root_count": 8,
        "planned_update_count": 800,
        "cpu_worker_target": 8,
        "cpu_worker_max": 8,
        "qualification_namespace": NAT_MSEP_QUALIFICATION_NAMESPACE,
        "B": 583_975,
        "Q": 116_795,
        "gate_minimum_D": 58_398,
        "gate_integer_rule": "10*D>=583975",
        "qualification_mask_reusable_for_training": False,
        "prohibited": [
            "model_outcome_read",
            "cuda_or_gpu",
            "model_weight_load",
            "training",
            "program_generation_or_execution_or_scoring",
            "formal_root_creation",
            "formal_mask_creation",
        ],
    }


def _msep_csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_msep_csv(
    path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> None:
    encoded = [
        {field: _msep_csv_value(row.get(field)) for field in fields} for row in rows
    ]
    _write_csv(path, encoded, fields)


_MSEP_UPDATE_FIELDS = (
    "root",
    "update",
    "exposures",
    "k",
    "q",
    "omega",
    "omega1",
    "omega2",
    "log10_omega",
    "omega_is_singleton",
    "qualification_rank",
    "rank_roundtrips",
    "optimum_id_overlap",
    "optimum_payload_overlap",
    "selected_ordinals",
    "selected_count",
    "selected_response_tokens",
    "materialization_gate",
)
_MSEP_ROOT_FIELDS = (
    "root",
    "updates",
    "D_id",
    "delta_id_num",
    "delta_id_den",
    "id_gate",
    "D_payload",
    "delta_payload_num",
    "delta_payload_den",
    "payload_gate",
    "singleton_updates",
    "omega_sum",
    "omega1_sum",
    "omega2_sum",
    "f_arm_response_tokens",
    "m_arm_response_tokens",
    "f_source_removal_classes",
    "m_source_removal_classes",
    "source_exposure_count_histogram",
    "cluster_exposure_count_histogram",
    "i_associated_clusters",
    "f_i_through_b_clusters",
    "m_i_through_b_clusters",
    "f_cumulative_removed_response_tokens",
    "m_cumulative_removed_response_tokens",
    "f_arm_input_tokens",
    "m_arm_input_tokens",
    "f_to_m_input_ratio_num",
    "f_to_m_input_ratio_den",
    "retained_k_input_length_cdf_max_abs_num",
    "retained_k_input_length_cdf_max_abs_den",
)


def verify_nat_msep_delivery(output_root: Path) -> dict[str, object]:
    """Verify the sidecar, archive members, schemas, row counts, and arithmetic."""

    output_root = Path(output_root)
    archive_path = output_root / _MSEP_ARCHIVE_NAME
    sidecar_path = output_root / f"{_MSEP_ARCHIVE_NAME}.sha256"
    sidecar_hash, sidecar_name = sidecar_path.read_text("utf-8").strip().split()
    if sidecar_name != _MSEP_ARCHIVE_NAME or sidecar_hash != _sha256_file(archive_path):
        raise RuntimeError("NAT-MSEP archive sidecar differs")
    with zipfile.ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != _MSEP_CONTENT_NAMES:
            raise RuntimeError("NAT-MSEP archive members differ")
        for name in _MSEP_CONTENT_NAMES:
            archive.read(name)
    freeze = json.loads((output_root / _MSEP_CONTENT_NAMES[0]).read_text("utf-8"))
    result = json.loads((output_root / _MSEP_CONTENT_NAMES[3]).read_text("utf-8"))
    if (
        freeze.get("schema") != "nat-msep-20-qualification-freeze-v1"
        or result.get("schema") != "nat-msep-20-qualification-result-v1"
        or result.get("terminal") not in NAT_MSEP_TERMINALS
    ):
        raise RuntimeError("NAT-MSEP delivery schema or terminal differs")
    with (output_root / _MSEP_CONTENT_NAMES[1]).open(
        newline="", encoding="utf-8"
    ) as handle:
        updates = list(csv.DictReader(handle))
    with (output_root / _MSEP_CONTENT_NAMES[2]).open(
        newline="", encoding="utf-8"
    ) as handle:
        roots = list(csv.DictReader(handle))
    if len(updates) != int(result["completed_update_count"]) or len(roots) != int(
        result["completed_root_count"]
    ):
        raise RuntimeError("NAT-MSEP result/table row counts differ")
    if len(updates) != 100 * len(roots):
        raise RuntimeError("NAT-MSEP delivery contains a partial root")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in updates:
        grouped.setdefault(int(row["root"]), []).append(row)
    for row in roots:
        root = int(row["root"])
        root_updates = grouped.get(root, [])
        if len(root_updates) != 100:
            raise RuntimeError("NAT-MSEP root lacks 100 updates")
        d_id = 116_795 - sum(int(item["optimum_id_overlap"]) for item in root_updates)
        d_payload = 116_795 - sum(
            int(item["optimum_payload_overlap"]) for item in root_updates
        )
        if d_id != int(row["D_id"]) or d_payload != int(row["D_payload"]):
            raise RuntimeError("NAT-MSEP independent D arithmetic differs")
        if (10 * d_id >= 583_975) != (row["id_gate"] == "True"):
            raise RuntimeError("NAT-MSEP independent ID gate differs")
        if (10 * d_payload >= 583_975) != (row["payload_gate"] == "True"):
            raise RuntimeError("NAT-MSEP independent payload gate differs")
    return {
        "verified": True,
        "terminal": result["terminal"],
        "completed_updates": len(updates),
        "completed_roots": len(roots),
        "archive_sha256": sidecar_hash,
        "archive_members": list(_MSEP_CONTENT_NAMES),
    }


def publish_nat_msep_delivery(
    *,
    output_root: Path,
    freeze: dict[str, object],
    roots: Sequence[NatMsepRootQualification],
    actual_concurrency: int,
    repairs: Sequence[str] = (),
    asset_ready: bool = True,
    deadline_exceeded: bool = False,
) -> dict[str, object]:
    """Publish the minimal five-file NAT-MSEP delivery and verified archive."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    allowed = set(_MSEP_CONTENT_NAMES) | {
        _MSEP_ARCHIVE_NAME,
        f"{_MSEP_ARCHIVE_NAME}.sha256",
    }
    unknown = {path.name for path in output_root.iterdir()} - allowed
    if unknown:
        raise RuntimeError(
            f"NAT-MSEP output directory contains unknown entries: {sorted(unknown)}"
        )
    freeze_path = output_root / _MSEP_CONTENT_NAMES[0]
    if not freeze_path.exists():
        _atomic_json(freeze_path, freeze)
    elif json.loads(freeze_path.read_text("utf-8")) != freeze:
        raise RuntimeError("NAT-MSEP published freeze differs")

    ordered_roots = tuple(int(root) for root in freeze.get("roots", ()))
    order = {root: index for index, root in enumerate(ordered_roots)}
    values = sorted(roots, key=lambda value: order[value.root])
    deadline = freeze.get("formal_deadline_epoch")
    if (
        asset_ready
        and not deadline_exceeded
        and len(values) == 8
        and isinstance(deadline, (int, float))
        and time.time() >= float(deadline) - 0.25
    ):
        values = values[:7]
        deadline_exceeded = True
    if not asset_ready:
        values = []
    update_rows = [dict(row) for value in values for row in value.update_rows]
    update_rows.sort(key=lambda row: (order[int(row["root"])], int(row["update"])))
    root_rows = [dict(value.root_row) for value in values]
    root_rows.sort(key=lambda row: order[int(row["root"])])
    _write_msep_csv(
        output_root / _MSEP_CONTENT_NAMES[1], update_rows, _MSEP_UPDATE_FIELDS
    )
    _write_msep_csv(
        output_root / _MSEP_CONTENT_NAMES[2], root_rows, _MSEP_ROOT_FIELDS
    )

    resolved = resolve_nat_msep_terminal(
        asset_ready=asset_ready,
        completed_updates=len(update_rows),
        completed_roots=len(root_rows),
        root_rows=root_rows,
        deadline_exceeded=deadline_exceeded,
    )
    finished_epoch = time.time()
    start_epoch = freeze.get("formal_start_epoch")
    wall_seconds = (
        finished_epoch - float(start_epoch)
        if isinstance(start_epoch, (int, float))
        else 0.0
    )
    result: dict[str, object] = {
        "schema": "nat-msep-20-qualification-result-v1",
        "protocol_id": NAT_MSEP_PROTOCOL_ID,
        "terminal": resolved["terminal"],
        "reason_code": resolved["reason_code"],
        "planned_update_count": 800,
        "completed_update_count": len(update_rows),
        "planned_root_count": 8,
        "completed_root_count": len(root_rows),
        "root_results": [
            {
                "root": int(row["root"]),
                "D_id": int(row["D_id"]),
                "delta_id": f"{row['delta_id_num']}/{row['delta_id_den']}",
                "id_gate": bool(row["id_gate"]),
                "D_payload": int(row["D_payload"]),
                "delta_payload": (
                    f"{row['delta_payload_num']}/{row['delta_payload_den']}"
                ),
                "payload_gate": bool(row["payload_gate"]),
            }
            for row in root_rows
        ],
        "gates": {
            "id_8_of_8": resolved["gates"]["id"],
            "payload_8_of_8": resolved["gates"]["payload"],
        },
        "actual_cpu_concurrency": actual_concurrency,
        "formal_result_commit_epoch": finished_epoch,
        "formal_result_commit_utc": datetime.fromtimestamp(
            finished_epoch, timezone.utc
        ).isoformat(),
        "formal_census_wall_seconds": round(wall_seconds, 6),
        "engineering_repairs": list(repairs),
        "qualification_mask_reusable_for_training": False,
        "model_outcomes_read": False,
        "gpu_processes_started": 0,
        "model_weights_loaded": 0,
        "programs_generated_or_executed_or_scored": 0,
        "formal_roots_created": 0,
        "formal_masks_created": 0,
        "training_runs_started": 0,
    }
    _atomic_json(output_root / _MSEP_CONTENT_NAMES[3], result)

    if root_rows:
        singleton = sum(int(row["singleton_updates"]) for row in root_rows)
        omega = sum(int(row["omega_sum"]) for row in root_rows)
        source_summary = (
            f"Across complete roots: singleton updates={singleton}; omega sum={omega}. "
            f"I-associated clusters={sum(int(row['i_associated_clusters']) for row in root_rows)}; "
            f"F/M I-through-B clusters="
            f"{sum(int(row['f_i_through_b_clusters']) for row in root_rows)}/"
            f"{sum(int(row['m_i_through_b_clusters']) for row in root_rows)}."
        )
    else:
        source_summary = "No complete root source diagnostics were eligible for tables."
    report = (
        "# NAT-MSEP-20 CPU qualification\n\n"
        f"Terminal: `{result['terminal']}`. Reason: "
        f"`{result['reason_code'] or 'none'}`.\n\n"
        f"Completed {len(update_rows)}/800 updates and {len(root_rows)}/8 roots. "
        f"{source_summary}\n\n"
        f"Actual CPU concurrency was {actual_concurrency}; formal wall time through "
        f"RESULT commit was {wall_seconds:.6f} seconds. Engineering repairs: "
        f"{', '.join(repairs) if repairs else 'none'}.\n\n"
        "The qualification mask is not reusable for training. No model outcome was "
        "read; no GPU, model weight, program generation/execution/scoring, formal "
        "root, formal mask, or training run was used.\n"
    )
    _atomic_text(output_root / _MSEP_CONTENT_NAMES[4], report)
    archive_path = output_root / _MSEP_ARCHIVE_NAME
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _MSEP_CONTENT_NAMES:
            archive.write(output_root / name, arcname=name)
    temporary.replace(archive_path)
    _atomic_text(
        output_root / f"{_MSEP_ARCHIVE_NAME}.sha256",
        f"{_sha256_file(archive_path)}  {_MSEP_ARCHIVE_NAME}\n",
    )
    verify_nat_msep_delivery(output_root)
    return result


def run_nat_msep_cpu_qualification(
    prepared_root: Path,
    *,
    output_root: Path,
    runtime_root: Path,
    max_workers: int = 8,
) -> dict[str, object]:
    """Run the formal eight-root MSEP census after establishing one freeze."""

    prepared_root = Path(prepared_root)
    output_root = Path(output_root)
    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / _MSEP_CONTENT_NAMES[3]
    if result_path.exists():
        verify_nat_msep_delivery(output_root)
        return json.loads(result_path.read_text("utf-8"))
    repair_path = runtime_root / "REPAIR_HISTORY.json"
    if repair_path.exists():
        repair_payload = json.loads(repair_path.read_text("utf-8"))
        repairs = tuple(str(item) for item in repair_payload.get("repairs", ()))
    else:
        repairs = ()
    try:
        source_path = prepared_root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
        source = json.loads(source_path.read_text("utf-8"))
        roots = tuple(int(root) for root in source.get("roots", ()))
        if len(roots) != 8 or len(set(roots)) != 8:
            raise NatMsepQualificationError(
                "source_or_plan_identity_drift",
                "NAT-MSEP requires eight distinct frozen roots",
            )
        validate_nat_msep_prepared_inputs(prepared_root, roots)
        base_freeze = _nat_msep_base_freeze(prepared_root, source)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        asset_freeze: dict[str, object] = {
            "schema": "nat-msep-20-qualification-freeze-v1",
            "protocol_id": NAT_MSEP_PROTOCOL_ID,
            "source_freeze_sha256": None,
            "roots": [],
            "planned_root_count": 8,
            "planned_update_count": 800,
            "formal_start_epoch": None,
            "formal_deadline_epoch": None,
            "asset_diagnostic": f"{type(exc).__name__}: {exc}",
        }
        return publish_nat_msep_delivery(
            output_root=output_root,
            freeze=asset_freeze,
            roots=(),
            actual_concurrency=0,
            repairs=repairs,
            asset_ready=False,
        )

    existing_freeze = output_root / _MSEP_CONTENT_NAMES[0]
    recovery_count = int(existing_freeze.exists())
    freeze = ensure_nat_msep_formal_freeze(output_root, base_freeze)
    started_epoch = float(freeze["formal_start_epoch"])
    deadline_epoch = float(freeze["formal_deadline_epoch"])
    try:
        values = qualify_nat_msep_roots(
            prepared_root,
            roots=roots,
            max_workers=max_workers,
            deadline_epoch=deadline_epoch,
            progress_path=runtime_root / "progress.json",
            started_epoch=started_epoch,
            recovery_count=recovery_count,
        )
    except NatMsepQualificationError as exc:
        asset_freeze = dict(freeze)
        asset_freeze["asset_diagnostic"] = f"{exc.reason_code}: {exc}"
        _atomic_json(output_root / _MSEP_CONTENT_NAMES[0], asset_freeze)
        return publish_nat_msep_delivery(
            output_root=output_root,
            freeze=asset_freeze,
            roots=(),
            actual_concurrency=min(max_workers, 8),
            repairs=repairs,
            asset_ready=False,
        )
    except Exception as exc:
        _atomic_json(
            runtime_root / "REPAIR_REQUIRED.json",
            {
                "schema": "nat-msep-20-engineering-recovery-v1",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "scientific_terminal_created": False,
                "formal_start_epoch": started_epoch,
                "formal_deadline_epoch": deadline_epoch,
                "model_outcomes_read": False,
                "gpu_processes_started": 0,
            },
        )
        raise
    deadline_exceeded = len(values) < 8 or time.time() >= deadline_epoch
    result = publish_nat_msep_delivery(
        output_root=output_root,
        freeze=freeze,
        roots=values,
        actual_concurrency=min(max_workers, 8),
        repairs=repairs,
        deadline_exceeded=deadline_exceeded,
    )
    _atomic_json(
        runtime_root / "progress.json",
        {
            "schema": "nat-msep-20-progress-v1",
            "protocol_id": NAT_MSEP_PROTOCOL_ID,
            "phase": "terminal_published",
            "planned_roots": 8,
            "completed_roots": result["completed_root_count"],
            "planned_updates": 800,
            "completed_updates": result["completed_update_count"],
            "actual_concurrency": min(max_workers, 8),
            "wall_seconds": result["formal_census_wall_seconds"],
            "deadline_epoch": deadline_epoch,
            "supervisor_pid": __import__("os").getpid(),
            "recovery_count": recovery_count,
        },
    )
    return result


def run_nat_spec_cpu_qualification(
    prepared_root: Path,
    *,
    output_root: Path,
    runtime_root: Path,
    max_workers: int = 8,
) -> dict[str, object]:
    """Run and publish the bounded eight-root outcome-blind CPU census."""

    prepared_root = Path(prepared_root)
    output_root = Path(output_root)
    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    repair_path = runtime_root / "REPAIR_HISTORY.json"
    if repair_path.exists():
        repair_payload = json.loads(repair_path.read_text("utf-8"))
        repairs = tuple(str(item) for item in repair_payload.get("repairs", ()))
    else:
        repairs = ()
    source = json.loads(
        (prepared_root / "NAT_REPLICA_X164_EXECUTION_FREEZE.json").read_text("utf-8")
    )
    roots = tuple(int(root) for root in source.get("roots", ()))
    freeze = _qualification_freeze(prepared_root, source)
    if len(roots) != 8 or len(set(roots)) != 8:
        now = datetime.now(timezone.utc).isoformat()
        return _publish_delivery(
            output_root=output_root,
            freeze=freeze,
            roots=(),
            started_utc=now,
            finished_utc=now,
            wall_seconds=0.0,
            actual_concurrency=0,
            failure_reason="source_or_plan_identity_drift",
            repairs=repairs,
        )

    clock_path = runtime_root / "QUALIFICATION_CLOCK.json"
    if clock_path.exists():
        clock = json.loads(clock_path.read_text("utf-8"))
        started_epoch = float(clock["started_epoch"])
        started_utc = str(clock["started_utc"])
    else:
        started_epoch = time.time()
        started_utc = datetime.now(timezone.utc).isoformat()
        _atomic_json(
            clock_path,
            {
                "schema": "nat-spec-20-qualification-clock-v1",
                "started_epoch": started_epoch,
                "started_utc": started_utc,
                "wall_limit_seconds": 7200,
                "resets": False,
            },
        )
    actual_concurrency = min(max_workers, len(roots))
    _atomic_json(
        runtime_root / "progress.json",
        {
            "phase": "formal_cpu_census",
            "planned_roots": 8,
            "completed_roots": 0,
            "planned_updates": 800,
            "completed_updates": 0,
            "started_utc": started_utc,
        },
    )
    try:
        values = qualify_nat_spec_roots(
            prepared_root, roots=roots, max_workers=max_workers
        )
    except NatSpecQualificationError as exc:
        finished_utc = datetime.now(timezone.utc).isoformat()
        return _publish_delivery(
            output_root=output_root,
            freeze=freeze,
            roots=(),
            started_utc=started_utc,
            finished_utc=finished_utc,
            wall_seconds=time.time() - started_epoch,
            actual_concurrency=actual_concurrency,
            failure_reason=exc.reason_code,
            repairs=repairs,
        )
    except Exception as exc:
        _atomic_json(
            runtime_root / "REPAIR_REQUIRED.json",
            {
                "schema": "nat-spec-20-repair-required-v1",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "scientific_terminal_created": False,
                "model_outcomes_read": False,
            },
        )
        raise

    wall_seconds = time.time() - started_epoch
    finished_utc = datetime.now(timezone.utc).isoformat()
    if wall_seconds > 7200:
        values = ()
        failure_reason = "qualification_census_over_2h"
    else:
        failure_reason = None
    _atomic_json(
        runtime_root / "progress.json",
        {
            "phase": "terminal_published",
            "planned_roots": 8,
            "completed_roots": len(values),
            "planned_updates": 800,
            "completed_updates": sum(len(value.update_rows) for value in values),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
        },
    )
    return _publish_delivery(
        output_root=output_root,
        freeze=freeze,
        roots=values,
        started_utc=started_utc,
        finished_utc=finished_utc,
        wall_seconds=wall_seconds,
        actual_concurrency=actual_concurrency,
        failure_reason=failure_reason,
        repairs=repairs,
    )


NAT_ONOFF_CONCURRENCY = {"train": 2, "generation": 4, "score": 4}
NAT_ONOFF_TERMINALS = frozenset(
    {
        "Q1_ONOFF_READOUT_DEGENERATE",
        "Q1_ONOFF_POSITIVE_EXPLORATORY",
        "Q1_ONOFF_BELOW_INVESTMENT_SCALE",
        "Q1_ONOFF_UNRESOLVED_WITHIN_BUDGET",
        "Q1_ONOFF_MEASUREMENT_INVALID",
        "Q1_ONOFF_INCOMPLETE_RESOURCE",
        "Q1_ONOFF_INCOMPLETE_INFRASTRUCTURE",
        "Q1_ONOFF_NEEDS_SCIENTIFIC_REVIEW",
        "Q1_ONOFF_BLOCKED_SAFETY_OR_PERMISSION",
    }
)
NAT_E95_DELETE_TERMINALS = frozenset(
    {
        "E95_DIRECTIONAL_SUPPORT_EXPLORATORY",
        "E95_DIRECTION_REVERSED_EXPLORATORY",
        "E95_SMALL_OR_UNRESOLVED",
        "E95_MEASUREMENT_INVALID",
        "E95_INCOMPLETE_RESOURCE",
        "E95_INCOMPLETE_INFRASTRUCTURE",
        "E95_NEEDS_SCIENTIFIC_REVIEW",
        "E95_BLOCKED_SAFETY_OR_PERMISSION",
    }
)
NAT_E95_DELETE_ROOTS = (
    1243236849,
    346604563,
    2078544004,
    2066018300,
    687218389,
    166203935,
    653279739,
    1508294500,
)
NAT_E95_DELETE_BLOCK_TABLE_SHA256 = (
    "598bd29cf7109c617c9f87a90bbb610f4e1c92adfb9df16dbdbdd15c258ae3e8"
)
NAT_E95_DELETE_S12D_TABLE_SHA256 = (
    "c534a19c28f51e9fc2b1872a39b22cd32bab733dd5f55148ee65c7cc3a98aa0b"
)
NAT_E95_DELETE_THETA0_SHA256 = (
    "407355b826579dc7463da4dd250282846084309c0a2be802371fe73be48d3f97"
)
W259_PAIRS = (
    ("P01", 94_157_648, 432_176_365, "ddeda7e93c3315b061a1176bb9fb83b68d848c73be726952e1e9b81869751a7c"),
    ("P02", 1_409_923_109, 992_380_055, "791419630307f9659ee1d61091fd54f9b47ff7b86381f9e13c6d149293bf58a8"),
    ("P03", 1_709_473_788, 1_478_250_915, "92be926135f55c381eef72e74a6970507a19f64417377738f18d92681b80b8f9"),
    ("P04", 122_961_837, 1_875_991_562, "d6a58f54512b0485fcbba69ae21996014ef0c999508f5dc1d20472be8aac7a22"),
    ("P05", 36_289_367, 1_022_018_198, "ab70f62ba25b6a3aec32bb2f29c968daed20cdaba7b340921dbf742a54c39e21"),
    ("P06", 1_761_626_600, 1_728_384_408, "a44135a2116e8a498ae46f2173c08695209e4eb876c15508dc75d3d18b729094"),
    ("P07", 1_653_610_645, 1_907_824_140, "8b75b63deda8960e60ff150a89079ebb6d9f07402c1c8b79d07d5a9f0188b984"),
    ("P08", 2_021_254_069, 1_422_873_367, "88a09ee136efefeb0a16e3a526a46c8e06e11dec4dc2e04898245777215e1a6d"),
)
W259_TERMINALS = frozenset(
    {
        "W_READOUT_DEGENERATE",
        "W_DIRECTIONAL_SUPPORT_EXPLORATORY",
        "W_DIRECTION_REVERSED_EXPLORATORY",
        "W_BELOW_INVESTMENT_SCALE",
        "W_UNRESOLVED_WITHIN_BUDGET",
        "W_MEASUREMENT_INVALID",
        "W_INCOMPLETE_RESOURCE",
        "W_INCOMPLETE_INFRASTRUCTURE",
        "W_NEEDS_SCIENTIFIC_REVIEW",
        "W_BLOCKED_SAFETY_OR_PERMISSION",
    }
)
G121_PROTOCOL_ID = "Q1-G121-FIXED-REPLICATION-16-v1"
G121_ARMS = ("B", "BG")
G121_WALL_CAP_SECONDS = 50 * 3600
G121_GPU_CAP_SECONDS = 46 * 3600
G121_TERMINALS = frozenset({"INCOMPLETE_RESOURCE_STOP", "MEASUREMENT_INVALID",
    "UNCERTAINTY_DEGENERATE", "DIRECTION_REVERSED", "POSITIVE_CONDITIONAL_REPLICATION",
    "POSITIVE_BELOW_CONTINUATION_SCALE", "NO_SUPPORT_AT_CONTINUATION_SCALE", "WITHIN_BUDGET_UNRESOLVED"})


def prepare_g121_design(*, project_root: Path, task_root: Path, run_id: str):
    """Bind the actual CF12 payload/recipe and materialize the one new fixed batch."""
    from ..arms.paired_filtering import build_w259_factorial_plans, dump_s12_training_plan
    root, task = Path(project_root), Path(task_root)
    artifacts = root / "20_q1_operational_defect_audit/experiment/artifacts"
    source_dir = artifacts / "q1_w259_cf12/formal_20260916_1"
    source = validate_w259_cf12_execution_freeze(json.loads((source_dir/"EXECUTION_FREEZE.json").read_text("utf-8")))
    old_design_dir = root / ".scratch/q1-w259-cf12/design"
    old_path = old_design_dir/"DESIGN_RECEIPT.json"
    if _sha256_file(old_path) != "43fcb24b64758f0799802dad2471f7178b561469ae1e4d120fd3eef52d69b2fe":
        raise ValueError("G121 historical design receipt differs")
    old = json.loads(old_path.read_text("utf-8"))
    unsigned = dict(old)
    digest = unsigned.pop("design_sha256")
    if digest != "866ea162f2afca6e0e76dd5c40a3bcd6cafeda04118ba9c58f88e6f3a9847a03" or digest != _canonical_json_sha256(unsigned):
        raise ValueError("G121 historical design differs")
    prior_files = {"prospective": artifacts/"q1_w259_prospective/formal_20260915_1/EXECUTION_FREEZE.json",
                   "onoff":artifacts/"q1_nat_onoff354/formal_20260913_1/EXECUTION_FREEZE.json",
                   "msep":artifacts/"codecontests_nat_msep20/formal_20260910_1/NAT_MSEP_20_EXECUTION_FREEZE.json"}
    prior = {k:json.loads(p.read_text("utf-8")) for k,p in prior_files.items()}
    historical = {2026091201}
    for p in prior["prospective"]["pairs"]:
        historical.update((int(p["init_seed"]),int(p["root"])))
    for k in ("onoff","msep"):
        historical.update(map(int,prior[k]["roots"]))
        initial = prior[k].get("training_recipe",{}).get("init_seed")
        if initial is not None:
            historical.add(int(initial))
    if historical != {int(r["value"]) for r in old["exclusions"]}:
        raise ValueError("G121 inherited exclusion list disagrees with named historical sources")
    excluded = historical | {int(c[k]) for c in source["contexts"] for k in ("init_seed","root")}
    pairs = derive_g121_pairs(excluded_values=excluded)
    group_path = artifacts/"q1_w259_factorial/static_check_20260916_1/GROUP_MEMBERS.csv"
    if not group_path.is_file():
        group_path = task/"tmp/GROUP_MEMBERS.csv"
    if _sha256_file(group_path) != old["source_bindings"]["group_members_sha256"]:
        raise ValueError("G121 original group table file differs")
    with group_path.open(newline="",encoding="utf-8") as f:
        group_rows = list(csv.DictReader(f))
    groups = {r["canonical_record_id"]:r["group"] for r in group_rows}
    gids = sorted(k for k,v in groups.items() if v=="G")
    ghash = hashlib.sha256(("\n".join(gids)+"\n").encode()).hexdigest()
    if len(gids)!=121 or ghash != "82d890e6931d88dd9a07d01e3145772661979ac8a727d6d2b2edfc95566a3e76":
        raise ValueError("G121 fixed membership differs")
    source_plan_path = old_design_dir/"plans/C01_BW.json"
    old_bw = next(p for p in source["plans"] if p["context_id"]=="C01" and p["arm"]=="BW")
    if _sha256_file(source_plan_path) != old_bw["plan_file_sha256"]:
        raise ValueError("G121 historical BW payload differs")
    source_plan = load_s12_training_plan(source_plan_path)
    def order(r):
        return sorted(range(100),key=lambda b:(hashlib.sha256(f"q1-nat-onoff-354-v1|block-order|{r}|{b}".encode("ascii")).digest(),b))
    old_order = order(old_bw["root"])
    def build(r):
        return build_w259_factorial_plans(root=r,source_w_plan=source_plan,source_block_order=old_order,
            group_by_unit=groups,arms=G121_ARMS,
            expected_block_order_sha256=hashlib.sha256(",".join(map(str,order(r))).encode()).hexdigest())
    rebuilt = build(old_bw["root"])
    for arm in G121_ARMS:
        authority = next(p for p in source["plans"] if p["context_id"]=="C01" and p["arm"]==arm)
        path = old_design_dir/"plans"/f"C01_{arm}.json"
        if _sha256_file(path) != authority["plan_file_sha256"] or s12_training_plan_sha256(rebuilt["plans"][arm]) != authority["plan_sha256"]:
            raise ValueError("G121 reconstructed B/BG differs from actual CF12")
    receipt_path = task/"design/DESIGN_RECEIPT.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text("utf-8"))
        if receipt["run_id"] != run_id or receipt["pairs"] != list(pairs):
            raise ValueError("G121 design namespace occupied")
        for p in receipt["plans"]:
            if _sha256_file(Path(p["plan_path"])) != p["plan_file_sha256"]:
                raise ValueError("G121 persisted plan drift")
        return receipt
    plans = []
    for pair in pairs:
        built = build(pair["root"])
        if built["component_exposures"] != {"B":1996,"G":121,"H":138} or built["component_tokens"] != {"B":583975,"G":39714,"H":39715}:
            raise ValueError("G121 inherited dose differs")
        for arm in G121_ARMS:
            plan = built["plans"][arm]
            if len(plan.update_bins)!=100 or any(sum(x.retained_response_tokens for x in b)<=0 for b in plan.update_bins):
                raise ValueError("G121 block denominator invalid")
            if any(sum(x.loss_mask[1:])!=x.retained_response_tokens for x in plan.exposures):
                raise ValueError("G121 shifted token denominator differs")
            path = task/"design/plans"/f'{pair["pair_id"]}_{arm}.json'
            identity = dump_s12_training_plan(path,plan)
            plans.append({**pair,"arm":arm,"plan_path":str(path),"plan_sha256":identity,
                          "plan_file_sha256":_sha256_file(path),"response_tokens":sum(x.retained_response_tokens for x in plan.exposures),
                          "updates":100,"block_order_sha256":built["block_order_sha256"]})
    receipt = {"schema":"q1-g121-design-receipt-v1","protocol_id":G121_PROTOCOL_ID,"run_id":run_id,
               "pairs":list(pairs),"contexts":list(pairs),"plans":plans,"exclusions":sorted(excluded),
               "inherited_exclusions_verified":sorted(historical),"group_sha256":ghash,
               "source_bindings":{**source["bindings"],"cf12_freeze_sha256":_sha256_file(source_dir/"EXECUTION_FREEZE.json"),
                                  **{k+"_freeze_sha256":_sha256_file(p) for k,p in prior_files.items()}},
               "paths":source["paths"],"runtime":source["runtime"],"software_versions":source["software_versions"],
               "source_actual_recipe": {"model_revision":source["bindings"]["model_revision"],
                  "lora_rank":16,"lora_alpha":32,"lora_dropout":.05,"precision":"BF16 base / FP32 adapter",
                  "optimizer":{"name":"AdamW","lr":.0002,"betas":[.9,.95],"eps":1e-8,"weight_decay":0,"foreach":False,"fused":False},
                  "lr_schedule":"u<4: peak*(u+1)/4; else peak*(.1+.9*(1+cos(pi*(u-4)/95))/2); set before optimizer.step",
                  "normalization":"sum shifted response CE / block active tokens; clip=1; B/S streams; B-before-G; no dummy forwards",
                  "generation":{"greedy":True,"max_new_tokens":1024,"microbatch":1,"seed":20260829,"eos_pad":151643},
                  "scoring":source["isolation"],"historical_onoff_freeze_text_erratum":"CF12 actual r16/betas(.9,.95), not old ONOFF freeze's r8/(.9,.999)"},
               "formal_outcomes_opened":False,"status":"STATIC_READY_RUNTIME_NOT_TESTED"}
    receipt["design_sha256"] = _canonical_json_sha256(receipt)
    _atomic_json(receipt_path,receipt)
    return receipt


def publish_g121(*, output_root, freeze, position_rows, completed_training,
                 repairs, gpu_active_seconds, cloud_wall_seconds, forced_terminal=None):
    from ..analysis.codecontests import analyze_g121_positions
    output_root=Path(output_root)
    if json.loads((output_root/"EXECUTION_FREEZE.json").read_text("utf-8"))!=freeze:
        raise ValueError("G121 immutable freeze differs")
    rows=[dict(r) for r in position_rows]
    states=set(g121_state_ids(freeze["contexts"]))
    training=set(completed_training)
    if not training<=states:
        raise ValueError("G121 publication training identity differs")
    for r in rows:
        if r["state_id"] not in training:
            r.update(Y=None,missing_reason="training_identity_not_validated",verdict="not_evaluated")
    analysis=analyze_g121_positions(rows,pairs=freeze["contexts"],task_ids=freeze["task_ids"])
    if forced_terminal not in (None,"INCOMPLETE_RESOURCE_STOP","MEASUREMENT_INVALID"):
        raise ValueError("G121 forced terminal differs")
    complete=analysis["complete"] and training==states
    terminal=analysis["terminal"] if complete else forced_terminal or "MEASUREMENT_INVALID"
    result={"schema":"q1-g121-result-v1","protocol_id":G121_PROTOCOL_ID,"run_id":freeze["run_id"],
            "terminal":terminal,"complete":complete,"analysis":analysis,"completed_training_states":len(training),
            "valid_positions":analysis["valid_positions"],"expected_positions":5248,
            "scientific_identity_sha256":freeze["scientific_identity_sha256"],
            "gpu_active_seconds":gpu_active_seconds,"cloud_wall_seconds":cloud_wall_seconds,
            "gpu_cap_seconds":G121_GPU_CAP_SECONDS,"wall_cap_seconds":G121_WALL_CAP_SECONDS,
            "preparation_cost":{k:freeze["acceptance"][k] for k in ("gpu_active_seconds","wall_seconds")},
            "repairs":list(repairs),"goal_ledger_seconds":"unknown_remote","local_transfer_seconds":"not_evaluated",
            "cloud_residency_seconds":"unknown","cpu_core_hours":"unknown"}
    _write_nat_onoff_csv(output_root/"POSITION_TABLE.csv",rows,
        ("state_id","pair_id","init_seed","root","arm","task_id","Y","verdict","completion_sha256","missing_reason"))
    _write_nat_onoff_csv(output_root/"ROOT_TABLE.csv",analysis["root_rows"],tuple(analysis["root_rows"][0]))
    _atomic_json(output_root/"RESULT.json",result)
    report=["# Q1-G121-R16 — exploratory / conditional", "",
        "Material Passport: experiment-agent / run; single-agent verification, not cross-model audit.","",
        f"Terminal: {terminal}; training {len(training)}/32; valid positions {analysis['valid_positions']}/5248.",
        f"B={analysis['B_pass']}/2624; BG={analysis['BG_pass']}/2624; net={analysis['net_pass']}; gain={analysis['gain']}; loss={analysis['loss']}; gross={analysis['gross']}.",
        f"Mean={analysis['mean_pp']} pp; sample SD={analysis['sample_sd_pp']}; primary 95% t15={analysis['ci95_pp']}; reference 97.5%={analysis['reference_97_5_ci_pp']}.",
        f"Formal cloud wall={cloud_wall_seconds}s; task GPU union={gpu_active_seconds}s (conservative process lifetime). Preparation separate: {result['preparation_cost']}.",
        f"Repairs: {list(repairs)}. Missing statistics are not_evaluated, never zero-imputed.","",
        "Only the new 16 joint initialization/root pairs enter the result; old CF12 is historical and not pooled. HE164 was previously used. No independent external confirmation, individual harmful-record attribution, or claim that G is worse than H. The 2 pp / net 53 threshold is an investment point-estimate scale, not a true-effect lower bound. No automatic successor.","",
        "Local transfer, goal ledger, paid cloud residency and CPU core-hours require separate local reporting; unknown values are not free cost.",""]
    (output_root/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    archive_path=output_root/"DELIVERY.zip"
    with zipfile.ZipFile(archive_path,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _W259_CF12_CORE:
            archive.write(output_root/name,arcname=name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("G121 ZIP verification failed")
    (output_root/"DELIVERY.zip.sha256").write_text(f"{_sha256_file(archive_path)}  DELIVERY.zip\n",encoding="utf-8")
    return result


def validate_g121_execution_freeze(value):
    unsigned=dict(value)
    digest=unsigned.pop("scientific_identity_sha256",None)
    if digest!=_canonical_json_sha256(unsigned) or value.get("protocol_id")!=G121_PROTOCOL_ID:
        raise ValueError("G121 freeze identity differs")
    validate_g121_authorization(value["authorization"],run_id=value["run_id"])
    pairs=_validate_g121_pairs(value["contexts"])
    if (value["job_order"]!=list(g121_jobs(pairs)) or len(value["plans"])!=32
        or value["effective_concurrency"]!={"train":2,"generation":4,"score":4}
        or _parse_protocol_datetime(value["deadline_utc"])-_parse_protocol_datetime(value["started_utc"])!=timedelta(seconds=G121_WALL_CAP_SECONDS)):
        raise ValueError("G121 freeze queue or budget differs")
    expected=set(g121_state_ids(pairs))
    actual=[f'{p["context_id"]}_{p["arm"]}' for p in value["plans"]]
    if set(actual)!=expected or len(actual)!=len(set(actual)):
        raise ValueError("G121 freeze plans differ")
    by_pair={p["pair_id"]:p for p in pairs}
    for plan in value["plans"]:
        pair=by_pair[plan["context_id"]]
        if (any(plan.get(k)!=pair[k] for k in ("pair_id","root","init_seed"))
            or plan.get("updates")!=100
            or plan.get("response_tokens")!={"B":583975,"BG":623689}[plan["arm"]]):
            raise ValueError("G121 freeze state identity differs")
    if value["task_ids"]!=[f"HumanEval/{i}" for i in range(164)]:
        raise ValueError("G121 freeze endpoint differs")
    return dict(value)


def prepare_g121_execution(*, task_root, output_root, authorization_path):
    """Called only at the authorized detached supervisor entry; t0 never resets."""
    from ..train.run_codecontests import validate_w259_context_initial_state_set
    from ..eval.humaneval import load_humaneval_plus_asset, EVALPLUS_SOURCE_ARCHIVE_SHA256
    task_root,output_root=Path(task_root),Path(output_root)
    freeze_path=output_root/"EXECUTION_FREEZE.json"
    if freeze_path.is_file():
        return validate_g121_execution_freeze(json.loads(freeze_path.read_text("utf-8")))
    began=datetime.now(timezone.utc)
    design=json.loads((task_root/"design/DESIGN_RECEIPT.json").read_text("utf-8"))
    authorization=validate_g121_authorization(json.loads(Path(authorization_path).read_text("utf-8")),run_id=design["run_id"])
    clock_path=task_root/"FORMAL_CLOCK.json"
    if clock_path.is_file():
        clock=json.loads(clock_path.read_text("utf-8"))
        if clock["authorization"]!=authorization:
            raise ValueError("G121 authorization changed after t0")
        began=_parse_protocol_datetime(clock["started_utc"])
    else:
        _atomic_json(clock_path,{"started_utc":began.isoformat(),"authorization":authorization})
    acceptance=json.loads((task_root/"acceptance/ACCEPTANCE.json").read_text("utf-8"))
    if not (acceptance.get("passed") is True and acceptance.get("formal_started") is False
            and acceptance.get("synthetic_only") is True and acceptance.get("isolation_passed") is True
            and acceptance["gpu_active_seconds"]<=1800 and acceptance["wall_seconds"]<=7200
            and acceptance["effective_concurrency"]=={"train":2,"generation":4,"score":4}):
        raise ValueError("G121 real machine acceptance not passed")
    unsigned=dict(design); digest=unsigned.pop("design_sha256")
    if digest!=_canonical_json_sha256(unsigned):
        raise ValueError("G121 design receipt differs")
    pairs=_validate_g121_pairs(design["pairs"])
    initial=acceptance["initial_states"]
    theta=validate_w259_context_initial_state_set(initial,expected_contexts={p["pair_id"]:p["init_seed"] for p in pairs},protocol_id=G121_PROTOCOL_ID)
    initial_by={p["context_id"]:p for p in initial}
    for initial_state in initial:
        parent=Path(initial_state["initial_adapter_path"]).parent
        for artifact in initial_state["adapter_files"]:
            if _sha256_file(parent/artifact["path"])!=artifact["sha256"]:
                raise ValueError("G121 initial adapter file differs")
    contexts=[{**p,"theta_i_sha256":theta[p["pair_id"]],"initial_adapter_path":initial_by[p["pair_id"]]["initial_adapter_path"]} for p in pairs]
    for p in design["plans"]:
        if _sha256_file(Path(p["plan_path"]))!=p["plan_file_sha256"]:
            raise ValueError("G121 plan file differs")
    paths=design["paths"]
    tasks=load_humaneval_plus_asset(Path(paths["asset"]))
    ids=[t["task_id"] for t in tasks]
    if ids!=[f"HumanEval/{i}" for i in range(164)] or _sha256_file(Path(paths["evaluator_archive"]))!=EVALPLUS_SOURCE_ARCHIVE_SHA256:
        raise ValueError("G121 endpoint differs")
    freeze={"schema":"q1-g121-execution-freeze-v1","protocol_id":G121_PROTOCOL_ID,"run_id":design["run_id"],
            "started_utc":began.isoformat(),"deadline_utc":(began+timedelta(seconds=G121_WALL_CAP_SECONDS)).isoformat(),
            "clock_resets":False,"authorization":authorization,"contexts":contexts,"plans":design["plans"],
            "design_receipt":design,"acceptance":acceptance,"paths":paths,"runtime":design["runtime"],
            "task_ids":ids,"job_order":list(g121_jobs(pairs)),"effective_concurrency":{"train":2,"generation":4,"score":4},
            "preparation_cost_separate":True,"wall_cap_seconds":G121_WALL_CAP_SECONDS,"gpu_cap_seconds":G121_GPU_CAP_SECONDS}
    freeze["scientific_identity_sha256"]=_canonical_json_sha256(freeze)
    validate_g121_execution_freeze(freeze)
    _atomic_json(freeze_path,freeze)
    return freeze


def validate_g121_authorization(value, *, run_id):
    expected = {"protocol_id":G121_PROTOCOL_ID,"run_id":run_id,
                "explicit_user_approval":True,"wall_cap_seconds":G121_WALL_CAP_SECONDS,
                "gpu_cap_seconds":G121_GPU_CAP_SECONDS,"states":32,"pairs":16,
                "current_window_available":True}
    if any(value.get(k)!=v for k,v in expected.items()) or not value.get("user_message") or not value.get("approved_utc"):
        raise ValueError("G121 requires new explicit formal authorization")
    _parse_protocol_datetime(value["approved_utc"])
    return dict(value)


def g121_budget_admission(*, wall_elapsed_seconds, gpu_active_seconds,
                          remaining_states, observed_wave_seconds):
    # High planning scenario is a reserve estimate, never a promise. Observed
    # complete GPU waves can only increase it; neither outcomes nor losses enter.
    per_state=max([42.04*3600/32]+[float(x)/4*1.15 for x in observed_wave_seconds])
    gpu_required=remaining_states*per_state
    wall_required=gpu_required*44.94/42.04+1800
    return {"admit":wall_elapsed_seconds+wall_required<=G121_WALL_CAP_SECONDS
            and gpu_active_seconds+gpu_required<=G121_GPU_CAP_SECONDS,
            "required_wall_seconds":wall_required,"required_gpu_seconds":gpu_required}


def g121_repair_scope(pairs, *, kind, state_id):
    if state_id not in g121_state_ids(pairs):
        raise ValueError("G121 repair identity differs")
    if kind=="training_computation":
        return tuple((k,state_id.split("_")[0]+"_"+a) for k in ("train","generation","score") for a in G121_ARMS)
    if kind=="interruption":
        return (("train",state_id),)
    if kind=="generation":
        return (("generation",state_id),("score",state_id))
    if kind=="score":
        return (("score",state_id),)
    raise ValueError("G121 repair requires scientific review")


def repair_g121_stage(*, task_root, output_root, kind, state_id, reason):
    """Explicit engineering repair, preserving displaced evidence and the clock."""
    task_root=Path(task_root).resolve()
    freeze=_load_w259_cf12_freeze(Path(output_root))
    if freeze["protocol_id"]!=G121_PROTOCOL_ID or not reason.strip():
        raise ValueError("G121 repair requires a bound reason")
    if _g121_owned_processes(task_root):
        raise RuntimeError("stop only this task's supervisor/workers before repair")
    if (Path(output_root)/"RESULT.json").is_file():
        raise ValueError("G121 scientific terminal cannot be reopened by stage repair")
    scope=g121_repair_scope(freeze["contexts"],kind=kind,state_id=state_id)
    progress_path=task_root/"progress.json"
    progress=json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
    now=datetime.now(timezone.utc).isoformat()
    destination=task_root/"repairs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    moves=[]
    for stage,state in scope:
        if kind=="interruption":
            continue  # Production resume validates optimizer/RNG and full identity.
        source=task_root/{"train":"training","generation":"generation","score":"scores"}[stage]/state
        if source.exists():
            source=source.resolve()
            if task_root not in source.parents:
                raise ValueError("G121 repair path escaped task root")
            target=destination/stage/state
            target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(source,target)
            moves.append({"source":str(source),"preserved_at":str(target)})
    for stage,state in scope:
        progress.setdefault("retry_counts",{}).pop(f"{stage}:{state}",None)
    for attempt in progress.get("attempts",[]):
        if not attempt.get("ended_utc"):
            attempt.update(ended_utc=now,end_conservative_after_repair=True)
    progress.setdefault("repairs",[]).append(f"{now}: {kind}/{state_id}: {reason}; preserved {len(moves)} stage directories")
    progress.update(status="engineering_repair_applied",active=[])
    _atomic_json(destination/"REPAIR.json",{"kind":kind,"state_id":state_id,"reason":reason,"moves":moves,"clock_reset":False})
    _atomic_json(progress_path,progress)
    return {"scope":scope,"preserved":moves,"clock_reset":False}


def derive_g121_pairs(*, excluded_values: set[int]) -> tuple[dict[str, object], ...]:
    """The plan's single collision-only draw, independent of outcomes and speed."""
    if any(type(v) is not int or v <= 0 for v in excluded_values):
        raise ValueError("G121 historical seed exclusion is invalid")
    used = set(excluded_values) | {2026091201}
    first_b = set(sorted(range(1,17), key=lambda i: (
        hashlib.sha256(f"{G121_PROTOCOL_ID}|{i:02d}|order".encode("ascii")).digest(), i))[:8])
    pairs = []
    for i in range(1,17):
        row = {"pair_id": f"R{i:02d}", "context_id": f"R{i:02d}",
               "first_arm": "B" if i in first_b else "BG"}
        for label in ("init", "root"):
            counter = 0
            while True:
                payload = f"{G121_PROTOCOL_ID}|{i:02d}|{label}|{counter}"
                value = 1 + int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") % (2**31-2)
                if value not in used:
                    break
                counter += 1
            used.add(value)
            row["init_seed" if label == "init" else "root"] = value
            row[f"{label}_counter"] = counter
        pairs.append(row)
    return tuple(pairs)


def _validate_g121_pairs(pairs):
    rows = tuple(dict(p) for p in pairs)
    if tuple(p.get("pair_id") for p in rows) != tuple(f"R{i:02d}" for i in range(1,17)):
        raise ValueError("G121 requires ordered R01..R16")
    values = [p.get(k) for p in rows for k in ("init_seed", "root")]
    if any(type(v) is not int or not 0 < v < 2**31-1 for v in values) or len(set(values)) != 32:
        raise ValueError("G121 pair seeds are invalid or repeated")
    expected = derive_g121_pairs(excluded_values={2026091201})
    if any(p.get("first_arm") != q["first_arm"] or p.get("context_id") != p["pair_id"] for p,q in zip(rows,expected)):
        raise ValueError("G121 arm order differs")
    return rows


def g121_state_ids(pairs):
    pairs = _validate_g121_pairs(pairs)
    return tuple(f'{p["pair_id"]}_{p["first_arm"] if round_index == 0 else ("BG" if p["first_arm"] == "B" else "B")}'
                 for start in range(0,16,2) for round_index in (0,1) for p in pairs[start:start+2])


def g121_jobs(pairs):
    return tuple({"kind":kind,"job_id":state} for kind in ("train","generation","score") for state in g121_state_ids(pairs))


def next_g121_jobs(pairs, *, completed, active):
    """Eight fixed GPU waves; scoring may overlap but has four global slots."""
    states = g121_state_ids(pairs)
    legal = {(j["kind"],j["job_id"]) for j in g121_jobs(pairs)}
    if not completed <= legal or not active <= legal or completed & active:
        raise ValueError("G121 queue identity is invalid")
    if any(sum(k == kind for k,_ in active) > limit for kind,limit in (("train",2),("generation",4),("score",4))):
        raise ValueError("G121 queue exceeds global concurrency")
    if any(k=="train" for k,_ in active) and any(k=="generation" for k,_ in active):
        raise ValueError("G121 training and generation GPU overlap")
    selected = []
    # Global CPU slots span waves and are never multiplied inside workers.
    score_slots = 4-sum(k=="score" for k,_ in active)
    for state in states:
        if score_slots and ("generation",state) in completed and ("score",state) not in completed|active:
            selected.append({"kind":"score","job_id":state})
            score_slots -= 1
    for start in range(0,32,4):
        wave = states[start:start+4]
        if all(("generation",s) in completed for s in wave):
            continue
        if not all(("train",s) in completed for s in wave):
            if any(k=="generation" for k,_ in active):
                return tuple(selected)
            # Both first arms close before the second round is admitted.
            round_states = wave[:2] if not all(("train",s) in completed for s in wave[:2]) else wave[2:]
            slots = 2-sum(k=="train" for k,_ in active)
            selected += [{"kind":"train","job_id":s} for s in round_states if ("train",s) not in completed|active][:max(0,slots)]
        elif not any(k=="train" for k,_ in active):
            slots = 4-sum(k=="generation" for k,_ in active)
            selected += [{"kind":"generation","job_id":s} for s in wave if ("generation",s) not in completed|active][:max(0,slots)]
        break
    return tuple(selected)


W259_CF12_PROTOCOL_ID = "q1-w259-cf12-v1"
W259_CF12_RANDOMIZATION_DOMAIN = "q1-w259-fixed-factorial-v1|20260916"
W259_CF12_ARMS = ("B", "BG", "BH", "BW")
W259_CF12_CONCURRENCY = {"train": 2, "generation": 4, "score": 4}
W259_CF12_WALL_CAP_SECONDS = 66 * 3600
W259_CF12_GPU_CAP_SECONDS = 54 * 3600
W259_CF12_TERMINALS = frozenset(
    {
        "CF12_SUBGROUP_SUPPORT_EXPLORATORY",
        "CF12_COMPLETE_NO_SUBGROUP_INVESTMENT_SUPPORT",
        "CF12_INCOMPLETE_RESOURCE",
        "CF12_INCOMPLETE_INFRASTRUCTURE",
        "CF12_INCOMPLETE_MEASUREMENT",
        "CF12_NEEDS_SCIENTIFIC_REVIEW",
        "CF12_BLOCKED_SAFETY_OR_PERMISSION",
    }
)


def derive_w259_cf12_contexts(
    *, excluded_values: set[int]
) -> tuple[dict[str, object], ...]:
    """Derive the twelve frozen joint seed/root contexts without outcomes."""

    if any(type(value) is not int or value < 0 for value in excluded_values):
        raise ValueError("Q1-W259-CF12 historical exclusion value is invalid")
    used = set(excluded_values) | {0}
    rows: list[dict[str, object]] = []
    for index in range(1, 13):
        context_id = f"C{index:02d}"
        values: dict[str, int] = {}
        for kind in ("init", "root"):
            k = 0
            while True:
                payload = (
                    f"{W259_CF12_RANDOMIZATION_DOMAIN}|{kind}|{context_id}|{k}"
                )
                value = int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)
                value %= 2**31 - 1
                if value not in used:
                    break
                k += 1
            used.add(value)
            values[kind] = value
            values[f"{kind}_k"] = k
        rows.append(
            {
                "context_id": context_id,
                "init_seed": values["init"],
                "init_k": values["init_k"],
                "root": values["root"],
                "root_k": values["root_k"],
            }
        )
    return tuple(rows)


def _validate_w259_cf12_contexts(
    contexts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = tuple(dict(row) for row in contexts)
    if len(rows) != 12 or tuple(row.get("context_id") for row in rows) != tuple(
        f"C{index:02d}" for index in range(1, 13)
    ):
        raise ValueError("Q1-W259-CF12 context registry differs")
    values = [
        row.get(key) for row in rows for key in ("init_seed", "root")
    ]
    if any(type(value) is not int or value <= 0 for value in values) or len(
        set(values)
    ) != 24:
        raise ValueError("Q1-W259-CF12 context values differ")
    return rows


def _w259_cf12_arm_order(context_index: int) -> tuple[str, ...]:
    base = sorted(
        W259_CF12_ARMS,
        key=lambda arm: hashlib.sha256(
            f"{W259_CF12_RANDOMIZATION_DOMAIN}|arm-order|{arm}".encode()
        ).digest(),
    )
    shift = context_index % 4
    return tuple(base[shift:] + base[:shift])


def w259_cf12_state_ids(
    contexts: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    rows = _validate_w259_cf12_contexts(contexts)
    return tuple(
        f"{row['context_id']}_{arm}"
        for index, row in enumerate(rows)
        for arm in _w259_cf12_arm_order(index)
    )


def w259_cf12_jobs(
    contexts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], ...]:
    states = w259_cf12_state_ids(contexts)
    return tuple(
        [{"kind": "train", "job_id": state} for state in states]
        + [{"kind": "generation", "job_id": state} for state in states]
        + [{"kind": "score", "job_id": state} for state in states]
    )


def next_w259_cf12_jobs(
    contexts: Sequence[Mapping[str, object]],
    *,
    completed: set[tuple[str, str]],
    active: set[tuple[str, str]],
    concurrency: Mapping[str, int],
) -> tuple[dict[str, str], ...]:
    """Advance one of six outcome-blind waves with strict GPU phase barriers."""

    if set(concurrency) != set(W259_CF12_CONCURRENCY) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= W259_CF12_CONCURRENCY[kind]
        for kind in W259_CF12_CONCURRENCY
    ):
        raise ValueError("Q1-W259-CF12 concurrency exceeds 2/4/4")
    jobs = w259_cf12_jobs(contexts)
    identities = {(job["kind"], job["job_id"]) for job in jobs}
    if not completed <= identities or not active <= identities:
        raise ValueError("Q1-W259-CF12 queue contains a foreign job")
    states = w259_cf12_state_ids(contexts)
    wave_states: tuple[str, ...] | None = None
    for start in range(0, len(states), 8):
        candidate = states[start : start + 8]
        if not all(
            (kind, state) in completed
            for state in candidate
            for kind in ("train", "generation", "score")
        ):
            wave_states = candidate
            break
    if wave_states is None:
        return ()
    if any(state not in wave_states for _kind, state in active):
        raise ValueError("Q1-W259-CF12 active job crosses wave boundary")

    if not all(("train", state) in completed for state in wave_states):
        if any(kind != "train" for kind, _state in active):
            return ()
        slots = concurrency["train"] - sum(kind == "train" for kind, _ in active)
        return tuple(
            {"kind": "train", "job_id": state}
            for state in wave_states
            if ("train", state) not in completed | active
        )[: max(0, slots)]
    if any(kind == "train" for kind, _state in active):
        return ()

    selected: list[dict[str, str]] = []
    for kind in ("generation", "score"):
        slots = concurrency[kind] - sum(
            active_kind == kind for active_kind, _state in active
        )
        for state in wave_states:
            if slots <= 0:
                break
            identity = (kind, state)
            if identity in completed | active:
                continue
            if kind == "score" and ("generation", state) not in completed:
                continue
            selected.append({"kind": kind, "job_id": state})
            slots -= 1
    return tuple(selected)


def w259_cf12_budget_admission(
    *,
    wall_elapsed_seconds: float,
    gpu_active_seconds: float | None,
    remaining_train_states: int,
    remaining_e2_states: int,
    completed_train_hours_per_state: Sequence[float],
    completed_e2_hours_per_state: Sequence[float],
    concurrency: Mapping[str, int],
) -> dict[str, object]:
    """Conservatively admit a wave against the non-resetting 66h/54h caps."""

    if gpu_active_seconds is None:
        raise ValueError("Q1-W259-CF12 GPU usage is unknown")
    numeric = (wall_elapsed_seconds, gpu_active_seconds)
    if any(not isinstance(value, (int, float)) or value < 0 for value in numeric):
        raise ValueError("Q1-W259-CF12 elapsed budget is invalid")
    if any(
        type(value) is not int or value < 0
        for value in (remaining_train_states, remaining_e2_states)
    ):
        raise ValueError("Q1-W259-CF12 remaining state count is invalid")
    if set(concurrency) != set(W259_CF12_CONCURRENCY) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= W259_CF12_CONCURRENCY[kind]
        for kind in W259_CF12_CONCURRENCY
    ):
        raise ValueError("Q1-W259-CF12 concurrency exceeds 2/4/4")
    if any(value <= 0 for value in (*completed_train_hours_per_state, *completed_e2_hours_per_state)):
        raise ValueError("Q1-W259-CF12 completed phase rate is invalid")

    historical_train = 2.532157 / 16
    historical_e2 = 9.709976 / 16
    observed_train = (
        float(statistics.median(completed_train_hours_per_state))
        if completed_train_hours_per_state
        else 0.0
    )
    recent_e2 = max(completed_e2_hours_per_state[-2:], default=0.0)
    train_rate = max(historical_train * 1.10, observed_train * 1.15)
    e2_rate = max(historical_e2 * 1.15, recent_e2 * 1.20)
    train_rate *= W259_CF12_CONCURRENCY["train"] / concurrency["train"]
    e2_scale = max(
        W259_CF12_CONCURRENCY["generation"] / concurrency["generation"],
        W259_CF12_CONCURRENCY["score"] / concurrency["score"],
    )
    e2_rate *= e2_scale
    compute_hours = (
        train_rate * remaining_train_states + e2_rate * remaining_e2_states
    )
    wall_required = compute_hours + 6.0
    gpu_required = compute_hours + 1.0
    wall_remaining = W259_CF12_WALL_CAP_SECONDS / 3600 - wall_elapsed_seconds / 3600
    gpu_remaining = W259_CF12_GPU_CAP_SECONDS / 3600 - gpu_active_seconds / 3600
    if wall_remaining <= 10 / 60:
        reason = "WALL_WIND_DOWN"
    elif gpu_remaining <= 10 / 60:
        reason = "GPU_WIND_DOWN"
    elif wall_required > wall_remaining:
        reason = "WALL_BUDGET_INSUFFICIENT"
    elif gpu_required > gpu_remaining:
        reason = "GPU_BUDGET_INSUFFICIENT"
    else:
        reason = "ADMITTED"
    return {
        "admit": reason == "ADMITTED",
        "reason": reason,
        "train_hours_per_state": train_rate,
        "e2_hours_per_state": e2_rate,
        "wall_required_hours": wall_required,
        "gpu_required_hours": gpu_required,
        "wall_remaining_hours": wall_remaining,
        "gpu_remaining_hours": gpu_remaining,
    }


def _parse_protocol_datetime(value: object) -> datetime:
    """Parse ISO-8601 clocks from Python or PowerShell without moving time."""

    text = str(value)
    if text.endswith("Z"):
        base, suffix = text[:-1], "+00:00"
    elif len(text) >= 6 and text[-6] in "+-" and text[-3] == ":":
        base, suffix = text[:-6], text[-6:]
    else:
        base, suffix = text, ""
    if "." in base:
        head, fraction = base.rsplit(".", 1)
        if fraction.isdigit() and len(fraction) > 6:
            base = f"{head}.{fraction[:6]}"
    return datetime.fromisoformat(base + suffix)


def nat_onoff_state_ids(roots: Sequence[int]) -> tuple[str, ...]:
    if len(roots) != 8 or len(set(roots)) != 8:
        raise ValueError("Q1-NAT-ONOFF needs eight unique roots")
    result = []
    for index, root in enumerate(roots):
        if type(root) is not int or root <= 0:
            raise ValueError("Q1-NAT-ONOFF root identity is invalid")
        arms = ("ON", "OFF") if index % 2 == 0 else ("OFF", "ON")
        result.extend(f"{root}_{arm}" for arm in arms)
    return tuple(result)


def nat_onoff_jobs(roots: Sequence[int]) -> tuple[dict[str, str], ...]:
    states = nat_onoff_state_ids(roots)
    return tuple(
        [{"kind": "train", "job_id": state} for state in states]
        + [{"kind": "generation", "job_id": state} for state in states]
        + [{"kind": "score", "job_id": state} for state in states]
    )


def next_nat_onoff_jobs(
    roots: Sequence[int], *, completed: set[tuple[str, str]],
    active: set[tuple[str, str]], concurrency: Mapping[str, int],
) -> tuple[dict[str, str], ...]:
    """Advance the fixed queue without observing any model outcome."""

    jobs = nat_onoff_jobs(roots)
    identities = {(job["kind"], job["job_id"]) for job in jobs}
    if not completed <= identities or not active <= identities:
        raise ValueError("Q1-NAT-ONOFF queue contains a foreign job")
    if set(concurrency) != set(NAT_ONOFF_CONCURRENCY) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= NAT_ONOFF_CONCURRENCY[kind]
        for kind in NAT_ONOFF_CONCURRENCY
    ):
        raise ValueError("Q1-NAT-ONOFF concurrency exceeds 2/4/4")
    training = jobs[:16]
    missing_training = [
        job for job in training
        if (job["kind"], job["job_id"]) not in completed | active
    ]
    if missing_training:
        if any(kind != "train" for kind, _state in active):
            return ()
        slots = concurrency["train"] - sum(kind == "train" for kind, _state in active)
        active_roots = {
            state.split("_", 1)[0] for kind, state in active if kind == "train"
        }
        if len(active_roots) > 1:
            raise ValueError("Q1-NAT-ONOFF active training crosses root pairs")
        pair_root = next(iter(active_roots), missing_training[0]["job_id"].split("_", 1)[0])
        return tuple(
            job for job in missing_training
            if job["job_id"].split("_", 1)[0] == pair_root
        )[:max(0, slots)]
    if any(kind == "train" for kind, _state in active):
        return ()
    if not all(("train", state) in completed for state in nat_onoff_state_ids(roots)):
        return ()
    selected = []
    for kind in ("generation", "score"):
        slots = (
            concurrency[kind] - sum(active_kind == kind for active_kind, _state in active)
            if kind == "generation"
            else int(not any(active_kind == "score" for active_kind, _state in active))
        )
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


def nat_e95_delete_state_ids(roots: Sequence[int]) -> tuple[str, ...]:
    if len(roots) != 8 or len(set(roots)) != 8:
        raise ValueError("Q1-NAT-E95-DELETE needs eight unique roots")
    if any(type(root) is not int or root <= 0 for root in roots):
        raise ValueError("Q1-NAT-E95-DELETE root identity is invalid")
    return tuple(f"{root}_MID" for root in roots)


def nat_e95_delete_jobs(roots: Sequence[int]) -> tuple[dict[str, str], ...]:
    states = nat_e95_delete_state_ids(roots)
    return tuple(
        [{"kind": "train", "job_id": state} for state in states]
        + [{"kind": "generation", "job_id": state} for state in states]
        + [{"kind": "score", "job_id": state} for state in states]
    )


def next_nat_e95_delete_jobs(
    roots: Sequence[int],
    *,
    completed: set[tuple[str, str]],
    active: set[tuple[str, str]],
    concurrency: Mapping[str, int],
) -> tuple[dict[str, str], ...]:
    """Advance the fixed MID queue without observing model outcomes."""

    ceilings = {"train": 2, "generation": 4, "score": 4}
    if set(concurrency) != set(ceilings) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= ceilings[kind]
        for kind in ceilings
    ):
        raise ValueError("Q1-NAT-E95-DELETE concurrency exceeds 2/4/4")
    jobs = nat_e95_delete_jobs(roots)
    identities = {(job["kind"], job["job_id"]) for job in jobs}
    if not completed <= identities or not active <= identities:
        raise ValueError("Q1-NAT-E95-DELETE queue contains a foreign job")
    states = nat_e95_delete_state_ids(roots)
    training_complete = all(("train", state) in completed for state in states)
    if not training_complete:
        if any(kind != "train" for kind, _state in active):
            return ()
        slots = concurrency["train"] - sum(kind == "train" for kind, _ in active)
        return tuple(
            job
            for job in jobs
            if job["kind"] == "train"
            and (job["kind"], job["job_id"]) not in completed | active
        )[:max(0, slots)]
    if any(kind == "train" for kind, _state in active):
        return ()
    selected: list[dict[str, str]] = []
    for kind in ("generation", "score"):
        slots = concurrency[kind] - sum(active_kind == kind for active_kind, _ in active)
        for job in jobs:
            identity = (job["kind"], job["job_id"])
            if slots <= 0:
                break
            if job["kind"] != kind or identity in completed | active:
                continue
            if kind == "score" and ("generation", job["job_id"]) not in completed:
                continue
            selected.append(job)
            slots -= 1
    return tuple(selected)


def w259_state_ids(
    pairs: Sequence[tuple[str, int, int, str]],
) -> tuple[str, ...]:
    if tuple(pairs) != W259_PAIRS:
        raise ValueError("Q1-W259-PROSPECTIVE pair registry differs")
    states: list[str] = []
    for index, (pair_id, _seed, _root, _digest) in enumerate(pairs):
        arms = ("W_USE", "W_OMIT") if index % 2 == 0 else ("W_OMIT", "W_USE")
        states.extend(f"{pair_id}_{arm}" for arm in arms)
    return tuple(states)


def w259_jobs(
    pairs: Sequence[tuple[str, int, int, str]],
) -> tuple[dict[str, str], ...]:
    states = w259_state_ids(pairs)
    return tuple(
        [{"kind": "train", "job_id": state} for state in states]
        + [{"kind": "generation", "job_id": state} for state in states]
        + [{"kind": "score", "job_id": state} for state in states]
    )


def next_w259_jobs(
    pairs: Sequence[tuple[str, int, int, str]],
    *,
    completed: set[tuple[str, str]],
    active: set[tuple[str, str]],
    concurrency: Mapping[str, int],
) -> tuple[dict[str, str], ...]:
    """Advance the fixed pair-adjacent W259 queue without reading outcomes."""

    ceilings = NAT_ONOFF_CONCURRENCY
    if set(concurrency) != set(ceilings) or any(
        type(concurrency[kind]) is not int
        or not 1 <= concurrency[kind] <= ceilings[kind]
        for kind in ceilings
    ):
        raise ValueError("Q1-W259-PROSPECTIVE concurrency exceeds 2/4/4")
    jobs = w259_jobs(pairs)
    identities = {(job["kind"], job["job_id"]) for job in jobs}
    if not completed <= identities or not active <= identities:
        raise ValueError("Q1-W259-PROSPECTIVE queue contains a foreign job")
    states = w259_state_ids(pairs)
    if not all(("train", state) in completed for state in states):
        if any(kind != "train" for kind, _state in active):
            return ()
        active_pairs = {
            state.split("_", 1)[0] for kind, state in active if kind == "train"
        }
        if len(active_pairs) > 1:
            raise ValueError("Q1-W259-PROSPECTIVE active training crosses pairs")
        pair_id = next(iter(active_pairs), None)
        if pair_id is None:
            pair_id = next(
                state.split("_", 1)[0]
                for state in states
                if ("train", state) not in completed
            )
        slots = concurrency["train"] - sum(kind == "train" for kind, _ in active)
        return tuple(
            job
            for job in jobs[:16]
            if job["job_id"].startswith(f"{pair_id}_")
            and (job["kind"], job["job_id"]) not in completed | active
        )[:max(0, slots)]
    if any(kind == "train" for kind, _state in active):
        return ()
    selected: list[dict[str, str]] = []
    for kind in ("generation", "score"):
        slots = concurrency[kind] - sum(active_kind == kind for active_kind, _ in active)
        for job in jobs:
            identity = (job["kind"], job["job_id"])
            if slots <= 0:
                break
            if job["kind"] != kind or identity in completed | active:
                continue
            if kind == "score" and ("generation", job["job_id"]) not in completed:
                continue
            selected.append(job)
            slots -= 1
    return tuple(selected)


def validate_w259_execution_freeze(
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Validate the complete prospective freeze without opening outcomes."""

    value = dict(freeze)
    identity = value.get("scientific_identity_sha256")
    unsigned = dict(value)
    unsigned.pop("scientific_identity_sha256", None)
    pairs = value.get("pairs")
    plans = value.get("plans")
    expected_specs = {
        pair_id: (seed, root) for pair_id, seed, root, _digest in W259_PAIRS
    }
    pair_rows_valid = isinstance(pairs, list) and len(pairs) == 8 and all(
        isinstance(row, Mapping)
        and expected_specs.get(str(row.get("pair_id")))
        == (row.get("init_seed"), row.get("root"))
        and _is_lower_sha256(row.get("theta_i_sha256"))
        and isinstance(row.get("initial_adapter_path"), str)
        for row in pairs
    )
    plan_keys = (
        {
            (str(row.get("pair_id")), str(row.get("arm")))
            for row in plans
            if isinstance(row, Mapping)
        }
        if isinstance(plans, list)
        else set()
    )
    expected_plan_keys = {
        (pair_id, arm)
        for pair_id, _seed, _root, _digest in W259_PAIRS
        for arm in ("W_USE", "W_OMIT")
    }
    began = _parse_protocol_datetime(value.get("started_utc"))
    deadline = _parse_protocol_datetime(value.get("deadline_utc"))
    valid = (
        value.get("schema") == "q1-w259-prospective-execution-freeze-v1"
        and value.get("protocol_id") == "q1-w259-prospective-v1"
        and identity == _canonical_json_sha256(unsigned)
        and value.get("clock_resets") is False
        and pair_rows_valid
        and len({str(row["pair_id"]) for row in pairs}) == 8
        and len({str(row["theta_i_sha256"]) for row in pairs}) == 8
        and isinstance(plans, list)
        and len(plans) == 16
        and plan_keys == expected_plan_keys
        and value.get("task_ids") == [f"HumanEval/{index}" for index in range(164)]
        and value.get("job_order") == list(w259_jobs(W259_PAIRS))
        and value.get("target_concurrency") == NAT_ONOFF_CONCURRENCY
        and value.get("effective_concurrency") == NAT_ONOFF_CONCURRENCY
        and value.get("gpu_cap_seconds") == 72_000
        and value.get("wall_cap_seconds") == 216_000
        and deadline - began == timedelta(hours=60)
    )
    if not valid:
        raise ValueError("Q1-W259-PROSPECTIVE execution freeze differs")
    return value


def validate_w259_cf12_execution_freeze(
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable CF12 execution identity without opening outcomes."""

    value = dict(freeze)
    unsigned = dict(value)
    identity = unsigned.pop("scientific_identity_sha256", None)
    contexts = value.get("contexts")
    plans = value.get("plans")
    try:
        context_rows = _validate_w259_cf12_contexts(contexts)  # type: ignore[arg-type]
        jobs = list(w259_cf12_jobs(context_rows))
    except (TypeError, ValueError):
        context_rows, jobs = (), []
    expected_plan_keys = {
        (str(row["context_id"]), arm) for row in context_rows for arm in W259_CF12_ARMS
    }
    plan_keys = (
        {
            (str(row.get("context_id")), str(row.get("arm")))
            for row in plans
            if isinstance(row, Mapping)
        }
        if isinstance(plans, list)
        else set()
    )
    began = _parse_protocol_datetime(value.get("started_utc"))
    deadline = _parse_protocol_datetime(value.get("deadline_utc"))
    valid = (
        value.get("schema") == "q1-w259-cf12-execution-freeze-v1"
        and value.get("protocol_id") == W259_CF12_PROTOCOL_ID
        and identity == _canonical_json_sha256(unsigned)
        and value.get("clock_resets") is False
        and len(context_rows) == 12
        and all(
            _is_lower_sha256(row.get("theta_i_sha256"))
            and isinstance(row.get("initial_adapter_path"), str)
            for row in context_rows
        )
        and len({row.get("theta_i_sha256") for row in context_rows}) == 12
        and isinstance(plans, list)
        and len(plans) == 48
        and plan_keys == expected_plan_keys
        and all(
            row.get("response_tokens")
            == {"B": 583975, "BG": 623689, "BH": 623690, "BW": 663404}.get(
                str(row.get("arm"))
            )
            and row.get("updates") == 100
            and _is_lower_sha256(row.get("plan_sha256"))
            for row in plans
        )
        and value.get("task_ids") == [f"HumanEval/{index}" for index in range(164)]
        and value.get("job_order") == jobs
        and value.get("target_concurrency") == W259_CF12_CONCURRENCY
        and value.get("gpu_cap_seconds") == W259_CF12_GPU_CAP_SECONDS
        and value.get("wall_cap_seconds") == W259_CF12_WALL_CAP_SECONDS
        and deadline - began == timedelta(hours=66)
    )
    if not valid:
        raise ValueError("Q1-W259-CF12 execution freeze differs")
    return value


_NAT_E95_DELETE_CORE = (
    "EXECUTION_FREEZE.json",
    "POSITION_TABLE.csv",
    "ROOT_TABLE.csv",
    "RESULT.json",
    "REPORT.md",
)


def publish_nat_e95_delete(
    *,
    output_root: Path,
    freeze: Mapping[str, object],
    historical_position_rows: Sequence[Mapping[str, object]],
    mid_position_rows: Sequence[Mapping[str, object]],
    phase_seconds: Mapping[str, float],
    effective_concurrency: Mapping[str, int],
    repairs: Sequence[str],
    gpu_active_seconds: float,
    forced_terminal: str | None = None,
) -> dict[str, object]:
    """Publish the one five-file E95 deletion result without altering old data."""

    output_root = Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text("utf-8"))
    roots = tuple(int(root) for root in freeze.get("roots", ()))
    tasks = tuple(str(task) for task in freeze.get("task_ids", ()))
    identity = freeze.get("scientific_identity_sha256")
    unsigned = dict(freeze)
    unsigned.pop("scientific_identity_sha256", None)
    if not (
        freeze.get("schema") == "q1-nat-e95-delete-execution-freeze-v1"
        and freeze.get("protocol_id") == "q1-nat-e95-delete-v1"
        and identity == _canonical_json_sha256(unsigned)
        and _is_lower_sha256(freeze.get("old_result_sha256"))
        and len(roots) == 8
        and len(tasks) == 164
    ):
        raise ValueError("Q1-NAT-E95-DELETE publication freeze is invalid")
    if forced_terminal is not None and forced_terminal not in NAT_E95_DELETE_TERMINALS:
        raise ValueError("Q1-NAT-E95-DELETE forced terminal is unknown")

    def normalized(
        row: Mapping[str, object], *, origin: str, origin_run: str
    ) -> dict[str, object]:
        value = dict(row)
        y = value.get("Y")
        if y in ("0", "1"):
            value["Y"] = int(str(y))
        elif y in ("null", "", None):
            value["Y"] = None
        missing = value.get("missing_reason")
        if missing in ("null", "", None):
            value["missing_reason"] = None
        value["origin"] = origin
        value["origin_run"] = origin_run
        return value

    historical = [
        normalized(
            row,
            origin="reused",
            origin_run="q1_nat_onoff354/formal_20260913_1",
        )
        for row in historical_position_rows
    ]
    mid = [
        normalized(
            row,
            origin="new",
            origin_run="q1_nat_e95_delete/formal_20260914_1",
        )
        for row in mid_position_rows
    ]
    if len(historical) != 2_624 or len(mid) != 1_312:
        raise ValueError("Q1-NAT-E95-DELETE publication position counts differ")
    rows = historical + mid
    from ..analysis.codecontests import analyze_nat_e95_delete_positions

    analysis = analyze_nat_e95_delete_positions(rows, roots=roots, task_ids=tasks)
    terminal = forced_terminal or analysis["terminal"]
    if terminal not in NAT_E95_DELETE_TERMINALS:
        raise ValueError("Q1-NAT-E95-DELETE publication lacks a legal terminal")
    output_root.mkdir(parents=True, exist_ok=True)
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if not freeze_path.is_file() or json.loads(freeze_path.read_text("utf-8")) != dict(freeze):
        raise ValueError("Q1-NAT-E95-DELETE immutable freeze file differs")
    position_fields = (
        "state_id", "root", "arm", "task_id", "Y", "verdict",
        "completion_sha256", "missing_reason", "origin", "origin_run",
    )
    _write_nat_onoff_csv(output_root / "POSITION_TABLE.csv", rows, position_fields)
    root_fields = (
        "root", "OFF_pass", "OFF_valid", "ON_pass", "ON_valid",
        "MID_pass", "MID_valid", "missing_MID", "dE_pp", "dW_pp",
        "historical_total_pp", "identity_error_pp", "dE_bound_lower_pp",
        "dE_bound_upper_pp", "dW_bound_lower_pp", "dW_bound_upper_pp",
    )
    _write_nat_onoff_csv(output_root / "ROOT_TABLE.csv", analysis["root_rows"], root_fields)
    result = {
        "schema": "q1-nat-e95-delete-result-v1",
        "protocol_id": "q1-nat-e95-delete-v1",
        "terminal": terminal,
        "scientific_identity_sha256": identity,
        "old_result_sha256": freeze["old_result_sha256"],
        "complete": bool(analysis["complete"]),
        "valid_new_positions": sum(row.get("Y") in (0, 1) for row in mid),
        "expected_new_positions": 1_312,
        "reused_positions": len(historical),
        "displayed_positions": len(rows),
        "analysis": analysis,
        "phase_seconds": {str(key): float(value) for key, value in phase_seconds.items()},
        "effective_concurrency": dict(effective_concurrency),
        "gpu_active_seconds": float(gpu_active_seconds),
        "gpu_cap_seconds": 86_400.0,
        "repairs": list(repairs),
    }
    _atomic_json(result_path, result)
    report = [
        "# Q1-NAT-E95-DELETE report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {datetime.now(timezone.utc).isoformat()}",
        "- Verification Status: UNVERIFIED",
        "- Version Label: q1_nat_e95_delete_result_v1",
        "",
        f"Terminal: `{terminal}`",
        "",
        f"New MID positions: {result['valid_new_positions']}/1312; reused historical ON/OFF positions: 2624.",
        "",
        "MID deletes the frozen E95 exposures from old ON while preserving B/W identities, masks, blocks and order. W remains on the original S random stream; deleted E consumes no random draws.",
        "",
        f"GPU active union: {float(gpu_active_seconds):.3f}s / 86400s. Effective concurrency: {dict(effective_concurrency)}. Repairs/downshifts: {list(repairs)}.",
        "",
        "HumanEval+ is reused and exploratory; historical endpoints and MID were executed in different periods. The 2 pp boundary is an investment scale, not a confirmed effect threshold.",
        "",
        f"Mean MID-ON: {analysis['mean_dE_pp']} pp; SD: {analysis['sd_dE_pp']} pp; descriptive t7 interval: {analysis['ci95_dE_pp']}.",
        "",
        f"Leave-one-root mean range: {analysis['leave_one_root_mean_dE_range_pp']} pp. MID extreme readout: {analysis['extreme_readout']}. Flags: {analysis['descriptive_flags']}.",
        "",
        f"ON-to-MID four-cell counts: {analysis['transition_counts']}.",
        "",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report), "utf-8", newline="\n")
    archive_path = output_root / "DELIVERY.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _NAT_E95_DELETE_CORE:
            archive.write(output_root / name, arcname=name)
    os.replace(temporary, archive_path)
    digest = _sha256_file(archive_path)
    (output_root / "DELIVERY.zip.sha256").write_text(
        f"{digest}  DELIVERY.zip\n", "utf-8"
    )
    return result


_W259_CORE = (
    "EXECUTION_FREEZE.json",
    "POSITION_TABLE.csv",
    "ROOT_TABLE.csv",
    "RESULT.json",
    "REPORT.md",
)


def publish_w259_prospective(
    *,
    output_root: Path,
    freeze: Mapping[str, object],
    position_rows: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[str, int, int, str]],
    task_ids: Sequence[str],
    phase_seconds: Mapping[str, float],
    effective_concurrency: Mapping[str, int],
    repairs: Sequence[str],
    gpu_active_seconds: float,
    cloud_wall_seconds: float,
    forced_terminal: str | None = None,
) -> dict[str, object]:
    """Publish the one finite five-file W259 result and deterministic membership."""

    from ..analysis.codecontests import analyze_w259_positions

    output_root = Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text("utf-8"))
        if existing.get("protocol_id") != "q1-w259-prospective-v1":
            raise FileExistsError("Q1-W259-PROSPECTIVE result namespace is occupied")
        return existing
    if forced_terminal is not None and forced_terminal not in W259_TERMINALS:
        raise ValueError("Q1-W259-PROSPECTIVE forced terminal is outside the protocol")
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if not freeze_path.is_file() or json.loads(freeze_path.read_text("utf-8")) != dict(freeze):
        raise ValueError("Q1-W259-PROSPECTIVE immutable freeze file differs")
    rows = [dict(row) for row in position_rows]
    analysis = analyze_w259_positions(rows, pairs=pairs, task_ids=task_ids)
    terminal = forced_terminal or analysis.get("terminal")
    if terminal is None:
        terminal = "W_INCOMPLETE_INFRASTRUCTURE"
    if terminal not in W259_TERMINALS:
        raise ValueError("Q1-W259-PROSPECTIVE analysis terminal differs")

    pair_order = {pair_id: index for index, (pair_id, *_rest) in enumerate(pairs)}
    arm_order = {"W_USE": 0, "W_OMIT": 1}
    task_order = {str(task_id): index for index, task_id in enumerate(task_ids)}
    rows.sort(
        key=lambda row: (
            pair_order[str(row["pair_id"])],
            arm_order[str(row["arm"])],
            task_order[str(row["task_id"])],
        )
    )
    position_fields = (
        "state_id", "pair_id", "root", "arm", "task_id", "Y", "verdict",
        "completion_sha256", "missing_reason",
    )
    _write_nat_onoff_csv(output_root / "POSITION_TABLE.csv", rows, position_fields)
    root_fields = (
        "pair_id", "root", "W_USE_pass", "W_USE_valid", "W_OMIT_pass",
        "W_OMIT_valid", "missing", "net_pass", "d_pp", "bound_lower_pp",
        "bound_upper_pp",
    )
    _write_nat_onoff_csv(
        output_root / "ROOT_TABLE.csv", analysis["root_rows"], root_fields
    )
    result = {
        "schema": "q1-w259-prospective-result-v1",
        "protocol_id": "q1-w259-prospective-v1",
        "terminal": terminal,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "complete": bool(analysis["complete"]),
        "valid_positions": sum(row.get("Y") in (0, 1) for row in rows),
        "expected_positions": 2_624,
        "analysis": analysis,
        "phase_seconds": {str(key): float(value) for key, value in phase_seconds.items()},
        "effective_concurrency": dict(effective_concurrency),
        "gpu_active_seconds": float(gpu_active_seconds),
        "gpu_cap_seconds": 72_000.0,
        "cloud_wall_seconds": float(cloud_wall_seconds),
        "wall_cap_seconds": 216_000.0,
        "repairs": list(repairs),
    }
    _atomic_json(result_path, result)
    report = [
        "# Q1-W259-PROSPECTIVE report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {datetime.now(timezone.utc).isoformat()}",
        "- Verification Status: UNVERIFIED",
        "- Version Label: q1_w259_prospective_result_v1",
        "",
        f"Terminal: `{terminal}`",
        "",
        f"Valid HumanEval+ positions: {result['valid_positions']}/2624.",
        "",
        "The finite estimand is the eight-pair mean of W_OMIT minus W_USE in percentage points. Each pair shares one fresh LoRA initialization; initialization and root variation are not separated.",
        "",
        f"Mean paired difference: {analysis['mean_d_pp']} pp; sample SD: {analysis['sd_d_pp']} pp; descriptive t7 interval: {analysis['ci95_pp']}.",
        "",
        f"Net pass differences: {analysis['root_net_pass']}; leave-one-pair mean range: {analysis['leave_one_pair_mean_range_pp']}.",
        "",
        f"GPU active union: {float(gpu_active_seconds):.3f}s / 72000s; cloud wall: {float(cloud_wall_seconds):.3f}s / 216000s; effective concurrency: {dict(effective_concurrency)}; repairs/downshifts: {list(repairs)}.",
        "",
        "HumanEval+ is reused and the t7 interval is exploratory. The 2 pp boundary is an investment scale, not a confirmed lower bound; this result does not identify individual harmful records or separate initialization from root variance.",
        "",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report), "utf-8", newline="\n")
    archive_path = output_root / "DELIVERY.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _W259_CORE:
            archive.write(output_root / name, arcname=name)
    os.replace(temporary, archive_path)
    digest = _sha256_file(archive_path)
    (output_root / "DELIVERY.zip.sha256").write_text(
        f"{digest}  DELIVERY.zip\n", "utf-8"
    )
    return result


_W259_CF12_CORE = (
    "EXECUTION_FREEZE.json",
    "POSITION_TABLE.csv",
    "ROOT_TABLE.csv",
    "RESULT.json",
    "REPORT.md",
)


def publish_w259_cf12(
    *,
    output_root: Path,
    freeze: Mapping[str, object],
    position_rows: Sequence[Mapping[str, object]],
    phase_seconds: Mapping[str, float],
    effective_concurrency: Mapping[str, int],
    repairs: Sequence[str],
    gpu_active_seconds: float,
    cloud_wall_seconds: float,
    forced_terminal: str | None = None,
) -> dict[str, object]:
    """Publish the finite five-file CF12 result and delivery archive."""

    from ..analysis.codecontests import analyze_w259_cf12_positions

    output_root = Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text("utf-8"))
        if result.get("protocol_id") != W259_CF12_PROTOCOL_ID:
            raise FileExistsError("Q1-W259-CF12 result namespace is occupied")
        return result
    if forced_terminal is not None and forced_terminal not in W259_CF12_TERMINALS:
        raise ValueError("Q1-W259-CF12 forced terminal differs")
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if not freeze_path.is_file() or json.loads(freeze_path.read_text("utf-8")) != dict(freeze):
        raise ValueError("Q1-W259-CF12 immutable freeze differs")
    rows = [dict(row) for row in position_rows]
    analysis = analyze_w259_cf12_positions(
        rows, contexts=freeze["contexts"], task_ids=freeze["task_ids"]
    )
    terminal = forced_terminal or analysis.get("terminal") or "CF12_INCOMPLETE_INFRASTRUCTURE"
    if terminal not in W259_CF12_TERMINALS:
        raise ValueError("Q1-W259-CF12 terminal differs")
    context_order = {
        str(row["context_id"]): index for index, row in enumerate(freeze["contexts"])
    }
    arm_order = {arm: index for index, arm in enumerate(W259_CF12_ARMS)}
    task_order = {str(task): index for index, task in enumerate(freeze["task_ids"])}
    rows.sort(
        key=lambda row: (
            context_order[str(row["context_id"])],
            arm_order[str(row["arm"])],
            task_order[str(row["task_id"])],
        )
    )
    _write_nat_onoff_csv(
        output_root / "POSITION_TABLE.csv",
        rows,
        (
            "state_id", "context_id", "root", "arm", "task_id", "Y", "verdict",
            "completion_sha256", "missing_reason",
        ),
    )
    root_fields = ["context_id", "root"]
    root_fields += [item for arm in W259_CF12_ARMS for item in (f"{arm}_pass", f"{arm}_valid")]
    root_fields.append("missing")
    for name in ("hG", "hH", "cG", "cH", "hW", "J"):
        root_fields.extend(
            (f"{name}_pp", f"{name}_bound_lower_pp", f"{name}_bound_upper_pp")
        )
    _write_nat_onoff_csv(output_root / "ROOT_TABLE.csv", analysis["root_rows"], root_fields)
    result = {
        "schema": "q1-w259-cf12-result-v1",
        "protocol_id": W259_CF12_PROTOCOL_ID,
        "terminal": terminal,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "complete": bool(analysis["complete"]),
        "valid_positions": sum(row.get("Y") in (0, 1) for row in rows),
        "expected_positions": 7872,
        "analysis": analysis,
        "phase_seconds": {str(key): float(value) for key, value in phase_seconds.items()},
        "effective_concurrency": dict(effective_concurrency),
        "gpu_active_seconds": float(gpu_active_seconds),
        "gpu_cap_seconds": float(W259_CF12_GPU_CAP_SECONDS),
        "cloud_wall_seconds": float(cloud_wall_seconds),
        "wall_cap_seconds": float(W259_CF12_WALL_CAP_SECONDS),
        "repairs": list(repairs),
    }
    _atomic_json(result_path, result)
    report = [
        "# Q1-W259-CF12 report", "", "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: run",
        f"- Origin Date: {datetime.now(timezone.utc).isoformat()}",
        "- Verification Status: UNVERIFIED",
        "- Version Label: q1_w259_cf12_result_v1", "",
        f"Terminal: `{terminal}`", "",
        f"Valid HumanEval+ positions: {result['valid_positions']}/7872.", "",
        f"Main labels: {analysis['main_labels']}.", "",
        f"Six contrasts: {analysis['contrasts']}.", "",
        f"GPU-active union: {gpu_active_seconds:.3f}s / {W259_CF12_GPU_CAP_SECONDS}s; cloud wall: {cloud_wall_seconds:.3f}s / {W259_CF12_WALL_CAP_SECONDS}s; effective concurrency: {dict(effective_concurrency)}; repairs: {list(repairs)}.",
        "",
        "HumanEval+ was reused in study selection. The intervals and subgroup labels are exploratory; initialization and root vary jointly. The 2 pp threshold is an investment scale, not a harm bound, and no result identifies individual harmful records or an independently confirmed mechanism.",
        "",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report), "utf-8", newline="\n")
    archive_path = output_root / "DELIVERY.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _W259_CF12_CORE:
            archive.write(output_root / name, arcname=name)
    os.replace(temporary, archive_path)
    digest = _sha256_file(archive_path)
    (output_root / "DELIVERY.zip.sha256").write_text(
        f"{digest}  DELIVERY.zip\n", "utf-8"
    )
    return result


def prepare_nat_e95_delete_execution(
    *,
    task_root: Path,
    output_root: Path,
    old_output_root: Path,
    acceptance_path: Path,
    e95_block_table_path: Path,
    s12d_table_path: Path,
    model_path: Path,
    asset_path: Path,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    python_gpu: str,
    python_judge: str,
) -> dict[str, object]:
    """Freeze eight MID plans by deleting E from the actual old ON plans."""

    from ..arms import paired_filtering
    from ..eval.humaneval import (
        EVALPLUS_REVISION,
        EVALPLUS_SOURCE_ARCHIVE_SHA256,
        HUMANEVAL_PLUS_LOGICAL_SHA256,
        HUMANEVAL_PLUS_SHA256,
        load_humaneval_plus_asset,
    )

    task_root = Path(task_root)
    output_root = Path(output_root)
    old_output_root = Path(old_output_root)
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if freeze_path.is_file():
        value = json.loads(freeze_path.read_text("utf-8"))
        unsigned = dict(value)
        identity = unsigned.pop("scientific_identity_sha256", None)
        if not (
            value.get("schema") == "q1-nat-e95-delete-execution-freeze-v1"
            and value.get("protocol_id") == "q1-nat-e95-delete-v1"
            and identity == _canonical_json_sha256(unsigned)
        ):
            raise ValueError("Q1-NAT-E95-DELETE existing freeze differs")
        return value

    old_freeze_path = old_output_root / "EXECUTION_FREEZE.json"
    old_result_path = old_output_root / "RESULT.json"
    old_position_path = old_output_root / "POSITION_TABLE.csv"
    old_freeze = json.loads(old_freeze_path.read_text("utf-8"))
    old_result = json.loads(old_result_path.read_text("utf-8"))
    old_identity = "5fa7b40099ea8b1476b4b432dbb5aea480e850c1eda6351e08247cbe25324266"
    roots = tuple(int(root) for root in old_freeze.get("roots", ()))
    task_ids = tuple(str(task) for task in old_freeze.get("task_ids", ()))
    if not (
        old_freeze.get("schema") == "q1-nat-onoff-354-execution-freeze-v1"
        and old_freeze.get("protocol_id") == "q1-nat-onoff-354-v1"
        and old_freeze.get("scientific_identity_sha256") == old_identity
        and roots == NAT_E95_DELETE_ROOTS
        and task_ids == tuple(f"HumanEval/{index}" for index in range(164))
        and old_result.get("protocol_id") == "q1-nat-onoff-354-v1"
        and old_result.get("scientific_identity_sha256") == old_identity
        and old_result.get("complete") is True
        and old_result.get("terminal") == "Q1_ONOFF_POSITIVE_EXPLORATORY"
        and old_position_path.is_file()
    ):
        raise ValueError("Q1-NAT-E95-DELETE historical endpoint binding differs")

    acceptance = json.loads(Path(acceptance_path).read_text("utf-8"))
    began = _parse_protocol_datetime(acceptance.get("started_utc"))
    deadline = _parse_protocol_datetime(acceptance.get("deadline_utc"))
    versions = acceptance.get("versions", {})
    if not (
        acceptance.get("schema") == "q1-nat-e95-delete-machine-acceptance-v1"
        and acceptance.get("protocol_id") == "q1-nat-e95-delete-v1"
        and acceptance.get("passed") is True
        and acceptance.get("clock_resets") is False
        and acceptance.get("isolation_passed") is True
        and acceptance.get("theta0_sha256") == NAT_E95_DELETE_THETA0_SHA256
        and began.tzinfo is not None
        and deadline - began == timedelta(hours=30)
        and isinstance(versions, Mapping)
        and str(versions.get("torch")) == "2.9.1+cu130"
        and str(versions.get("transformers")) == "5.3.0"
        and str(versions.get("peft")) == "0.18.1"
        and str(versions.get("numpy")) == "2.1.2"
    ):
        raise ValueError("Q1-NAT-E95-DELETE machine acceptance is not passed")
    if (
        _sha256_file(Path(e95_block_table_path))
        != NAT_E95_DELETE_BLOCK_TABLE_SHA256
        or _sha256_file(Path(s12d_table_path))
        != NAT_E95_DELETE_S12D_TABLE_SHA256
    ):
        raise ValueError("Q1-NAT-E95-DELETE binding tables differ")

    category_by_unit: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    with Path(s12d_table_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("natural_member") != "True":
                continue
            unit_id = f"{row['triplet_id']}/I"
            if unit_id in category_by_unit:
                raise ValueError("Q1-NAT-E95-DELETE S identity is duplicated")
            category = str(row["i_category"])
            category_by_unit[unit_id] = category
            category_counts[category] = category_counts.get(category, 0) + 1
    expected_categories = {
        "wrong_answer": 259,
        "compile_failure": 12,
        "runtime_failure": 75,
        "timeout": 8,
    }
    if category_counts != expected_categories or len(category_by_unit) != 354:
        raise ValueError("Q1-NAT-E95-DELETE W/E category ledger differs")

    bound_blocks: dict[tuple[int, int], dict[str, int]] = {}
    with Path(e95_block_table_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["root_id"]), int(row["update"]))
            if key in bound_blocks or row.get("order_bound") != "True":
                raise ValueError("Q1-NAT-E95-DELETE block binding is duplicated or unordered")
            bound_blocks[key] = {
                "update": key[1],
                "b_tokens": int(row["b_tokens"]),
                "w_tokens": int(row["w_tokens"]),
                "e_tokens": int(row["e_tokens"]),
                "mid_denominator": int(row["mid_denominator"]),
            }
    expected_block_keys = {
        (root, update) for root in roots for update in range(100)
    }
    if set(bound_blocks) != expected_block_keys:
        raise ValueError("Q1-NAT-E95-DELETE block table geometry differs")

    on_plan_rows = {
        int(row["root"]): row
        for row in old_freeze.get("plans", ())
        if row.get("arm") == "ON"
    }
    if set(on_plan_rows) != set(roots):
        raise ValueError("Q1-NAT-E95-DELETE old ON plan registry differs")
    task_root.mkdir(parents=True, exist_ok=True)
    plans_root = task_root / "plans"
    plan_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, int]] = []
    for root in roots:
        old_row = on_plan_rows[root]
        old_path = Path(str(old_row["plan_path"]))
        if (
            not old_path.is_file()
            or _sha256_file(old_path) != old_row.get("plan_file_sha256")
            or not _is_lower_sha256(old_row.get("plan_sha256"))
        ):
            raise ValueError("Q1-NAT-E95-DELETE old ON plan file differs")
        old_plan = paired_filtering.load_s12_training_plan(old_path)
        built = paired_filtering.build_nat_e95_delete_plan(
            old_on_plan=old_plan,
            supplement_category_by_unit=category_by_unit,
        )
        if not (
            built["component_exposures"]
            == {"B": 1996, "W": 259, "E_deleted": 95}
            and built["component_tokens"]
            == {"B": 583975, "W": 79429, "E_deleted": 26095}
            and len(built["block_rows"]) == 100
        ):
            raise ValueError("Q1-NAT-E95-DELETE MID dose differs")
        for row in built["block_rows"]:
            observed = {key: int(value) for key, value in dict(row).items()}
            if observed != bound_blocks[(root, int(observed["update"]))]:
                raise ValueError("Q1-NAT-E95-DELETE MID block differs from E95-BIND")
            block_rows.append({"root": root, **observed})
        plan_path = plans_root / f"{root}_MID.json"
        plan_identity = paired_filtering.dump_s12_training_plan(plan_path, built["plan"])
        plan_rows.append({
            "root": root,
            "arm": "MID",
            "plan_path": str(plan_path),
            "plan_sha256": plan_identity,
            "plan_file_sha256": _sha256_file(plan_path),
            "old_on_plan_path": str(old_path),
            "old_on_plan_sha256": old_row["plan_sha256"],
            "old_on_plan_file_sha256": old_row["plan_file_sha256"],
            "updates": 100,
            "response_tokens": 663404,
            "component_tokens": {"B": 583975, "W": 79429, "E_deleted": 26095},
        })

    tasks = load_humaneval_plus_asset(Path(asset_path))
    if [str(task["task_id"]) for task in tasks] != list(task_ids):
        raise ValueError("Q1-NAT-E95-DELETE HumanEval+ panel differs")
    if (
        Path(model_path).name != "8faed761d45a263340a0528343f099c05c9a4323"
        or _sha256_file(Path(evaluator_archive)) != EVALPLUS_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("Q1-NAT-E95-DELETE model/evaluator authority differs")
    output_root.mkdir(parents=True, exist_ok=True)
    freeze: dict[str, object] = {
        "schema": "q1-nat-e95-delete-execution-freeze-v1",
        "protocol_id": "q1-nat-e95-delete-v1",
        "started_utc": began.astimezone(timezone.utc).isoformat(),
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat(),
        "clock_resets": False,
        "roots": list(roots),
        "task_ids": list(task_ids),
        "plans": plan_rows,
        "job_order": list(nat_e95_delete_jobs(roots)),
        "old_scientific_identity_sha256": old_identity,
        "old_freeze_sha256": _sha256_file(old_freeze_path),
        "old_result_sha256": _sha256_file(old_result_path),
        "old_position_table_sha256": _sha256_file(old_position_path),
        "paths": {
            "model": str(model_path),
            "asset": str(asset_path),
            "evaluator_source": str(evaluator_source_root),
            "evaluator_archive": str(evaluator_archive),
            "old_output": str(old_output_root),
            "old_position_table": str(old_position_path),
            "e95_block_table": str(e95_block_table_path),
            "s12d_table": str(s12d_table_path),
        },
        "runtime": {"python_gpu": python_gpu, "python_judge": python_judge},
        "bindings": {
            "model_revision": Path(model_path).name,
            "humaneval_compressed_sha256": HUMANEVAL_PLUS_SHA256,
            "humaneval_logical_sha256": HUMANEVAL_PLUS_LOGICAL_SHA256,
            "evaluator_revision": EVALPLUS_REVISION,
            "evaluator_archive_sha256": EVALPLUS_SOURCE_ARCHIVE_SHA256,
            "e95_block_table_sha256": NAT_E95_DELETE_BLOCK_TABLE_SHA256,
            "s12d_table_sha256": NAT_E95_DELETE_S12D_TABLE_SHA256,
        },
        "data_contract": {
            "B_exposures": 1996,
            "W_exposures": 259,
            "E_deleted_exposures": 95,
            "B_tokens": 583975,
            "W_tokens": 79429,
            "E_deleted_tokens": 26095,
            "MID_tokens": 663404,
            "blocks": 100,
            "block_rows_sha256": _canonical_json_sha256(block_rows),
        },
        "training_recipe": {
            "init_seed": 2026091201,
            "expected_theta0_sha256": NAT_E95_DELETE_THETA0_SHA256,
            "updates": 100,
            "max_length": 2048,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "adamw_betas": [0.9, 0.95],
            "optimizer": "AdamW lr=2e-4 eps=1e-8 wd=0 foreach=false fused=false",
            "rng_streams": "historical q1-nat-onoff-354 root-derived B/S; E deletion does not advance S",
        },
        "generation_recipe": {
            "do_sample": False,
            "max_new_tokens": 1024,
            "stop_strings": 8,
            "endpoint": "HumanEval+164 exact",
        },
        "acceptance": acceptance,
        "isolation": acceptance["isolation"],
        "software_versions": dict(versions),
        "target_concurrency": {"train": 2, "generation": 4, "score": 4},
        "effective_concurrency": dict(acceptance["effective_concurrency"]),
        "resource_ladders": {
            "train": [2, 1], "generation": [4, 2, 1], "score": [4, 2, 1],
        },
        "gpu_cap_seconds": 86400,
        "wall_cap_seconds": 108000,
        "formal_outcomes_opened": False,
        "terminal_set": sorted(NAT_E95_DELETE_TERMINALS),
    }
    freeze["scientific_identity_sha256"] = _canonical_json_sha256(freeze)
    _atomic_json(freeze_path, freeze)
    _atomic_json(task_root / "E0_TIMING.json", {
        "started_utc": acceptance["finished_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    return freeze


_NAT_ONOFF_CORE = (
    "EXECUTION_FREEZE.json",
    "POSITION_TABLE.csv",
    "ROOT_TABLE.csv",
    "RESULT.json",
    "REPORT.md",
)


def _write_nat_onoff_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "null" if row.get(key) is None else row.get(key) for key in fields})
    os.replace(temporary, path)


def publish_nat_onoff(
    *, output_root: Path, freeze: Mapping[str, object],
    position_rows: Sequence[Mapping[str, object]], phase_seconds: Mapping[str, float],
    effective_concurrency: Mapping[str, int], repairs: Sequence[str],
    gpu_active_seconds: float, forced_terminal: str | None = None,
) -> dict[str, object]:
    """Publish the exact five-file Q1-NAT-ONOFF closure once."""

    output_root = Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text("utf-8"))
    roots = tuple(int(root) for root in freeze.get("roots", ()))
    tasks = tuple(str(task) for task in freeze.get("task_ids", ()))
    if not (
        freeze.get("schema") == "q1-nat-onoff-354-execution-freeze-v1"
        and freeze.get("protocol_id") == "q1-nat-onoff-354-v1"
        and _is_lower_sha256(freeze.get("scientific_identity_sha256"))
        and len(roots) == 8 and len(tasks) == 164
    ):
        raise ValueError("Q1-NAT-ONOFF publication freeze is invalid")
    if forced_terminal is not None and forced_terminal not in NAT_ONOFF_TERMINALS:
        raise ValueError("Q1-NAT-ONOFF forced terminal is unknown")
    from ..analysis.codecontests import analyze_nat_onoff_positions

    rows = [dict(row) for row in position_rows]
    analysis = None
    if len(rows) == 2_624:
        analysis = analyze_nat_onoff_positions(rows, roots=roots, task_ids=tasks)
    terminal = forced_terminal or (analysis and analysis["terminal"])
    if terminal not in NAT_ONOFF_TERMINALS:
        raise ValueError("Q1-NAT-ONOFF publication lacks a legal terminal")
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "EXECUTION_FREEZE.json", dict(freeze))
    position_fields = (
        "state_id", "root", "arm", "task_id", "Y", "verdict",
        "completion_sha256", "missing_reason",
    )
    _write_nat_onoff_csv(output_root / "POSITION_TABLE.csv", rows, position_fields)
    root_rows = analysis["root_rows"] if analysis else []
    root_fields = (
        "root", "OFF_pass", "OFF_valid", "ON_pass", "ON_valid", "d_pp",
        "bound_lower_pp", "bound_upper_pp", "missing_OFF", "missing_ON",
    )
    _write_nat_onoff_csv(output_root / "ROOT_TABLE.csv", root_rows, root_fields)
    result = {
        "schema": "q1-nat-onoff-354-result-v1",
        "protocol_id": "q1-nat-onoff-354-v1",
        "terminal": terminal,
        "scientific_identity_sha256": freeze["scientific_identity_sha256"],
        "complete": bool(analysis and analysis["complete"]),
        "valid_positions": sum(row.get("Y") in (0, 1) for row in rows),
        "expected_positions": 2_624,
        "analysis": analysis,
        "phase_seconds": {str(key): float(value) for key, value in phase_seconds.items()},
        "effective_concurrency": dict(effective_concurrency),
        "gpu_active_seconds": float(gpu_active_seconds),
        "gpu_cap_seconds": 72_000.0,
        "repairs": list(repairs),
    }
    _atomic_json(result_path, result)
    report = [
        "# Q1-NAT-ONOFF-354 report", "", f"Terminal: `{terminal}`", "",
        "The corrected identity is used throughout: historical N230 problem IDs are disjoint from S354; pair/triplet indices are not problem IDs.", "",
        f"Valid positions: {result['valid_positions']}/2624. GPU active union: {float(gpu_active_seconds):.3f}s / 72000s.", "",
        "S is present exactly once per ON root and absent from OFF. B uses common finite deletion without replacement.", "",
        "HumanEval+ is exploratory. The interval is root-level and the 2 pp boundary is an investment scale, not proof of a true effect of at least 2 pp.", "",
        f"Repairs/resource-only downshifts: {list(repairs)}", "",
    ]
    if analysis:
        report.extend([
            f"Mean OFF-ON: {analysis['mean_d_pp']} pp; SD: {analysis['sd_d_pp']} pp; 95% t7 interval: {analysis['ci95_pp']}.", "",
            f"Leave-one-root mean range: {analysis['leave_one_root_mean_range_pp']} pp.", "",
        ])
    (output_root / "REPORT.md").write_text("\n".join(report), "utf-8", newline="\n")
    archive_path = output_root / "DELIVERY.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _NAT_ONOFF_CORE:
            archive.write(output_root / name, arcname=name)
    os.replace(temporary, archive_path)
    digest = _sha256_file(archive_path)
    (output_root / "DELIVERY.zip.sha256").write_text(f"{digest}  DELIVERY.zip\n", "utf-8")
    return result


def prepare_nat_onoff_execution(
    *, workspace: Path, task_root: Path, output_root: Path,
    background_plan_path: Path, prior_root_registry_path: Path,
    acceptance_path: Path, model_path: Path, asset_path: Path,
    evaluator_source_root: Path, evaluator_archive: Path,
    python_gpu: str, python_judge: str,
) -> dict[str, object]:
    """Draw roots once and freeze all outcome-blind Q1-NAT-ONOFF inputs."""

    from ..arms.paired_filtering import (
        NAT_ONOFF_PROTOCOL_ID, NAT_ONOFF_ROOT_NAMESPACE,
        build_nat_onoff_plans, build_nat_onoff_roots, dump_s12_training_plan,
    )
    from ..eval.humaneval import (
        EVALPLUS_REVISION, EVALPLUS_SOURCE_ARCHIVE_SHA256,
        HUMANEVAL_PLUS_LOGICAL_SHA256, HUMANEVAL_PLUS_SHA256,
        load_humaneval_plus_asset,
    )
    from .run_codecontests_v2 import reconstruct_nat_onoff_inputs

    task_root, output_root = Path(task_root), Path(output_root)
    task_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text("utf-8"))
        if not (
            freeze.get("schema") == "q1-nat-onoff-354-execution-freeze-v1"
            and freeze.get("protocol_id") == NAT_ONOFF_PROTOCOL_ID
            and _is_lower_sha256(freeze.get("scientific_identity_sha256"))
        ):
            raise ValueError("Q1-NAT-ONOFF existing execution freeze differs")
        return freeze

    acceptance = json.loads(Path(acceptance_path).read_text("utf-8"))
    if not (
        acceptance.get("schema") == "q1-nat-onoff-354-machine-acceptance-v1"
        and acceptance.get("passed") is True
        and acceptance.get("clock_resets") is False
        and acceptance.get("isolation_passed") is True
        and acceptance.get("theta0_shared") is True
    ):
        raise ValueError("Q1-NAT-ONOFF machine acceptance is not passed")
    began = datetime.fromisoformat(str(acceptance["started_utc"]))
    deadline = datetime.fromisoformat(str(acceptance["deadline_utc"]))
    if began.tzinfo is None or deadline - began != timedelta(hours=36):
        raise ValueError("Q1-NAT-ONOFF non-resetting 36h clock differs")

    registry = json.loads(Path(prior_root_registry_path).read_text("utf-8"))
    prior_roots = tuple(sorted({int(root) for root in registry["roots"]}))
    if not prior_roots or any(root <= 0 for root in prior_roots):
        raise ValueError("Q1-NAT-ONOFF prior root registry differs")
    prior_sha = hashlib.sha256(
        json.dumps(prior_roots, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    claim_path = task_root / "RANDOMIZATION_CLAIM.json"
    if claim_path.is_file():
        claim = json.loads(claim_path.read_text("utf-8"))
        if claim.get("prior_root_registry_sha256") != prior_sha:
            raise ValueError("Q1-NAT-ONOFF randomization claim registry differs")
    else:
        claim = {
            "schema": "q1-nat-onoff-354-randomization-claim-v1",
            "protocol_id": NAT_ONOFF_PROTOCOL_ID,
            "root_seed_hex": secrets.token_hex(32),
            "root_namespace": NAT_ONOFF_ROOT_NAMESPACE,
            "prior_root_registry_sha256": prior_sha,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "outcomes_opened": False,
            "redraw_allowed": False,
        }
        _atomic_json(claim_path, claim)
    roots = build_nat_onoff_roots(
        seed_hex=str(claim["root_seed_hex"]), prior_roots=prior_roots
    )
    recovered = reconstruct_nat_onoff_inputs(
        workspace=Path(workspace), background_plan_path=Path(background_plan_path)
    )
    plans_root = task_root / "plans"
    plan_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for root in roots:
        built = build_nat_onoff_plans(
            root=root,
            background_plan=recovered["background_plan"],
            supplement_units=recovered["supplement_units"],
            background_metadata=recovered["background_metadata"],
            supplement_metadata=recovered["supplement_metadata"],
        )
        geometry_rows.append({
            "root": root,
            "block_order": list(built["block_order"]),
            "supplement_block_by_unit": dict(built["supplement_block_by_unit"]),
            "excluded_background_exposure_ordinals": list(
                built["excluded_background_exposure_ordinals"]
            ),
            "exclusion_counts": dict(built["exclusion_counts"]),
        })
        by_arm: dict[str, dict[str, object]] = {}
        for arm in ("ON", "OFF"):
            plan = built["plans"][arm]
            path = plans_root / f"{root}_{arm}.json"
            identity = dump_s12_training_plan(path, plan)
            row = {
                "root": root, "arm": arm, "plan_path": str(path),
                "plan_sha256": identity, "plan_file_sha256": _sha256_file(path),
                "updates": 100,
                "response_tokens": sum(
                    item.retained_response_tokens for item in plan.exposures
                ),
                "component_tokens": {
                    component: sum(
                        item.retained_response_tokens for item in plan.exposures
                        if item.component == component
                    )
                    for component in ("B", "S")
                },
            }
            by_arm[arm] = row
        plan_rows.extend((by_arm["ON"], by_arm["OFF"]))
    # Rebuild exact global alternating state order without relying on loop position.
    indexed = {(int(row["root"]), str(row["arm"])): row for row in plan_rows}
    plan_rows = [
        indexed[(int(state.rsplit("_", 1)[0]), state.rsplit("_", 1)[1])]
        for state in nat_onoff_state_ids(roots)
    ]
    tasks = load_humaneval_plus_asset(Path(asset_path))
    task_ids = [str(task["task_id"]) for task in tasks]
    if task_ids != [f"HumanEval/{index}" for index in range(164)]:
        raise ValueError("Q1-NAT-ONOFF HumanEval+ panel differs")
    if (
        Path(model_path).name != "8faed761d45a263340a0528343f099c05c9a4323"
        or _sha256_file(Path(evaluator_archive)) != EVALPLUS_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("Q1-NAT-ONOFF model/evaluator authority differs")
    freeze: dict[str, object] = {
        "schema": "q1-nat-onoff-354-execution-freeze-v1",
        "protocol_id": NAT_ONOFF_PROTOCOL_ID,
        "started_utc": began.astimezone(timezone.utc).isoformat(),
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat(),
        "clock_resets": False,
        "claim": claim,
        "prior_root_registry_sha256": prior_sha,
        "roots": list(roots), "task_ids": task_ids,
        "plans": plan_rows, "geometry": geometry_rows,
        "job_order": list(nat_onoff_jobs(roots)),
        "paths": {
            "model": str(model_path), "asset": str(asset_path),
            "evaluator_source": str(evaluator_source_root),
            "evaluator_archive": str(evaluator_archive),
            "background_plan": str(background_plan_path),
        },
        "runtime": {"python_gpu": python_gpu, "python_judge": python_judge},
        "bindings": {
            "model_revision": Path(model_path).name,
            "humaneval_compressed_sha256": HUMANEVAL_PLUS_SHA256,
            "humaneval_logical_sha256": HUMANEVAL_PLUS_LOGICAL_SHA256,
            "evaluator_revision": EVALPLUS_REVISION,
            "evaluator_archive_sha256": EVALPLUS_SOURCE_ARCHIVE_SHA256,
            "background_plan_file_sha256": recovered["background_plan_file_sha256"],
            "manifest_sha256": recovered["manifest_sha256"],
            "authority_hashes": recovered["authority_hashes"],
        },
        "data_contract": {
            key: recovered[key] for key in (
                "category_counts", "actual_s_shifted_response_tokens",
                "historical_s_saved_response_tokens", "background_response_tokens",
                "background_exposures", "background_unique_units",
                "s_problem_count", "b_problem_count", "cross_problem_count",
            )
        },
        "training_recipe": {
            "init_seed": 2026091201, "updates": 100, "max_length": 2048,
            "optimizer": "AdamW lr=2e-4 betas=(0.9,0.95) eps=1e-8 wd=0",
            "precision": "bf16 base + fp32 LoRA", "lora_rank": 16,
            "dropout": 0.05, "gradient_checkpointing": "nonreentrant preserve_rng_state",
            "rng_streams": "independent root-derived B/S full forward+backward",
        },
        "generation_recipe": {
            "do_sample": False, "max_new_tokens": 1024,
            "stop_strings": 8, "endpoint": "HumanEval+164 exact",
        },
        "isolation": acceptance["isolation"],
        "acceptance": acceptance,
        "target_concurrency": dict(NAT_ONOFF_CONCURRENCY),
        "effective_concurrency": dict(acceptance["effective_concurrency"]),
        "resource_ladders": {
            "train": [2, 1], "generation": [4, 2, 1], "score": [4, 2, 1]
        },
        "gpu_cap_seconds": 72_000, "wall_cap_seconds": 129_600,
        "formal_outcomes_opened": False,
        "terminal_set": sorted(NAT_ONOFF_TERMINALS),
    }
    freeze["scientific_identity_sha256"] = _canonical_json_sha256(freeze)
    _atomic_json(freeze_path, freeze)
    _atomic_json(task_root / "E0_TIMING.json", {
        "started_utc": acceptance["finished_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    return freeze


def _load_nat_onoff_freeze(output_root: Path) -> dict[str, object]:
    freeze = json.loads((Path(output_root) / "EXECUTION_FREEZE.json").read_text("utf-8"))
    identity = freeze.get("scientific_identity_sha256")
    unsigned = dict(freeze)
    unsigned.pop("scientific_identity_sha256", None)
    if not (
        freeze.get("schema") == "q1-nat-onoff-354-execution-freeze-v1"
        and freeze.get("protocol_id") == "q1-nat-onoff-354-v1"
        and identity == _canonical_json_sha256(unsigned)
        and len(freeze.get("roots", ())) == 8
        and len(freeze.get("plans", ())) == 16
    ):
        raise ValueError("Q1-NAT-ONOFF execution freeze identity differs")
    return freeze


def _nat_onoff_training_card(
    freeze: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    card: dict[str, object] = {
        "schema": "q1-nat-onoff-354-run-card-v1",
        "protocol_id": "q1-nat-onoff-354-v1",
        "root": int(plan["root"]), "arm": str(plan["arm"]),
        "init_seed": 2026091201, "updates": 100, "max_length": 2048,
        "plan_sha256": str(plan["plan_sha256"]),
        "response_tokens": int(plan["response_tokens"]),
        "outcomes_opened": False,
    }
    card["card_sha256"] = _canonical_json_sha256(card)
    return card


def execute_nat_onoff_worker(
    *, output_root: Path, task_root: Path, kind: str, job_id: str,
    score_workers: int = 4,
) -> dict[str, object]:
    """Execute one fixed training, generation, or isolated scoring state."""

    freeze = _load_nat_onoff_freeze(output_root)
    if (kind, job_id) not in {
        (str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]
    }:
        raise ValueError("Q1-NAT-ONOFF worker is outside the frozen queue")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(str(freeze["deadline_utc"])):
        raise RuntimeError("Q1-NAT-ONOFF 36h deadline is closed")
    plan_row = next(
        row for row in freeze["plans"]
        if f"{row['root']}_{row['arm']}" == job_id
    )
    card = _nat_onoff_training_card(freeze, plan_row)
    task_root = Path(task_root)
    training_root = task_root / "training" / job_id
    if kind == "train":
        from ..train.run_codecontests import execute_nat_onoff_training
        return execute_nat_onoff_training(
            plan=load_s12_training_plan(Path(str(plan_row["plan_path"]))),
            card=card, model_path=Path(str(freeze["paths"]["model"])),
            output_dir=training_root,
            resume=(training_root / "recovery" / "CURRENT.json").is_file()
            or (training_root / "TERMINAL_TRAINING_RESULT.json").is_file(),
        )
    from ..train.run_codecontests import validate_nat_onoff_training_terminal
    validate_nat_onoff_training_terminal(
        training_root / "TERMINAL_TRAINING_RESULT.json",
        expected_card_sha256=str(card["card_sha256"]),
        expected_plan_sha256=str(plan_row["plan_sha256"]),
    )
    from ..eval.humaneval import (
        execute_humaneval_formal_generation, execute_humaneval_formal_scoring,
        load_humaneval_plus_asset,
    )
    tasks = load_humaneval_plus_asset(Path(str(freeze["paths"]["asset"])))
    if [task["task_id"] for task in tasks] != freeze["task_ids"]:
        raise ValueError("Q1-NAT-ONOFF endpoint identity differs")
    if kind == "generation":
        return execute_humaneval_formal_generation(
            state_id=job_id, tasks=tasks,
            model_path=Path(str(freeze["paths"]["model"])),
            adapter_path=training_root / "adapter100",
            output_dir=task_root / "generation" / job_id,
            design_sha256=str(freeze["scientific_identity_sha256"]),
        )
    return execute_humaneval_formal_scoring(
        state_id=job_id, tasks=tasks,
        generation_dir=task_root / "generation" / job_id,
        output_dir=task_root / "scores" / job_id,
        design_sha256=str(freeze["scientific_identity_sha256"]),
        evaluator_source_root=Path(str(freeze["paths"]["evaluator_source"])),
        evaluator_archive=Path(str(freeze["paths"]["evaluator_archive"])),
        sandbox_base=task_root / "sandboxes" / job_id,
        workers=score_workers,
        python_executable=Path(str(freeze["runtime"]["python_judge"])),
        require_isolation=True,
    )


def _completed_nat_onoff_jobs(
    freeze: Mapping[str, object], task_root: Path
) -> set[tuple[str, str]]:
    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    completed: set[tuple[str, str]] = set()
    for job in freeze["job_order"]:
        kind, job_id = str(job["kind"]), str(job["job_id"])
        if kind == "train":
            plan = next(
                row for row in freeze["plans"]
                if f"{row['root']}_{row['arm']}" == job_id
            )
            card = _nat_onoff_training_card(freeze, plan)
            path = task_root / "training" / job_id / "TERMINAL_TRAINING_RESULT.json"
            if path.is_file():
                validate_nat_onoff_training_terminal(
                    path, expected_card_sha256=str(card["card_sha256"]),
                    expected_plan_sha256=str(plan["plan_sha256"]),
                )
                completed.add((kind, job_id))
            continue
        directory = "generation" if kind == "generation" else "scores"
        path = task_root / directory / job_id / "COMPLETE.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text("utf-8"))
        valid = (
            value.get("state_id") == job_id
            and value.get("completed_positions") == 164
            and value.get("design_sha256") == freeze["scientific_identity_sha256"]
        )
        if kind == "score":
            valid = valid and value.get("valid_positions") == 164 and (
                value.get("all_sandboxes_released") is True
            )
        if valid:
            completed.add((kind, job_id))
    return completed


def _nat_onoff_position_rows(
    freeze: Mapping[str, object], task_root: Path, *, missing_reason: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in freeze["roots"]:
        for arm in ("OFF", "ON"):
            state = f"{root}_{arm}"
            for index, task_id in enumerate(freeze["task_ids"]):
                row: dict[str, object] = {
                    "state_id": state, "root": int(root), "arm": arm,
                    "task_id": task_id, "Y": None, "verdict": "not_evaluated",
                    "completion_sha256": None, "missing_reason": missing_reason,
                }
                path = task_root / "scores" / state / f"{index:03d}.json"
                if path.is_file():
                    score = json.loads(path.read_text("utf-8"))
                    if not (
                        score.get("state_id") == state
                        and score.get("task_id") == task_id
                        and score.get("task_index") == index
                        and score.get("design_sha256")
                        == freeze["scientific_identity_sha256"]
                    ):
                        raise ValueError("Q1-NAT-ONOFF score binding differs")
                    row.update(
                        Y=score.get("Y"), verdict=score.get("verdict"),
                        completion_sha256=score.get("completion_sha256"),
                        missing_reason=(
                            None if score.get("Y") in (0, 1)
                            else str(score.get("detail") or "infrastructure_error")
                        ),
                    )
                rows.append(row)
    return rows


def run_nat_onoff_formal(
    *, task_root: Path, output_root: Path
) -> dict[str, object]:
    """Run the frozen finite queue under the non-resetting wall/GPU caps."""

    task_root, output_root = Path(task_root), Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text("utf-8"))
    freeze = _load_nat_onoff_freeze(output_root)
    began = datetime.fromisoformat(str(freeze["started_utc"]))
    deadline = datetime.fromisoformat(str(freeze["deadline_utc"]))
    progress_path = task_root / "progress.json"
    previous = json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
    effective = dict(previous.get("effective_concurrency", freeze["effective_concurrency"]))
    attempts: list[dict[str, object]] = list(previous.get("attempts", ()))
    repairs: list[str] = list(previous.get("repairs", ()))
    retry_counts: dict[str, int] = dict(previous.get("retry_counts", {}))
    milestones: dict[str, str] = dict(previous.get("milestones", {}))
    completed = _completed_nat_onoff_jobs(freeze, task_root)
    all_jobs = {
        (str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]
    }
    active: dict[tuple[str, str], tuple[subprocess.Popen[bytes], dict[str, object]]] = {}
    acceptance_gpu = float(freeze["acceptance"].get("gpu_active_seconds", 0.0))
    status, error = "running", None

    def attempt_seconds(now: datetime) -> float:
        intervals: list[tuple[datetime, datetime]] = []
        for record in attempts:
            if record.get("kind") not in {"train", "generation"}:
                continue
            start = datetime.fromisoformat(str(record["started_utc"]))
            end_text = record.get("ended_utc")
            end = datetime.fromisoformat(str(end_text)) if end_text else now
            if end > start:
                intervals.append((start, end))
        merged: list[list[datetime]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return acceptance_gpu + sum(
            (end - start).total_seconds() for start, end in merged
        )

    def write_progress(now: datetime) -> None:
        generated = sum(kind == "generation" for kind, _ in completed)
        scored = sum(kind == "score" for kind, _ in completed)
        payload = {
            "schema": "q1-nat-onoff-354-progress-v1",
            "protocol_id": "q1-nat-onoff-354-v1",
            "scientific_identity_sha256": freeze["scientific_identity_sha256"],
            "status": status,
            "stage": (
                "training" if sum(kind == "train" for kind, _ in completed) < 16
                else "generation_scoring" if scored < 16 else "publication"
            ),
            "completed_training": sum(kind == "train" for kind, _ in completed),
            "completed_generation_states": generated,
            "completed_score_states": scored,
            "generated_positions": sum(1 for _ in (task_root / "generation").glob("*/[0-9][0-9][0-9].json")),
            "scored_positions": sum(1 for _ in (task_root / "scores").glob("*/[0-9][0-9][0-9].json")),
            "active": [record for _process, record in active.values()],
            "effective_concurrency": effective,
            "started_utc": began.isoformat(), "deadline_utc": deadline.isoformat(),
            "updated_utc": now.isoformat(),
            "elapsed_seconds": max(0.0, (now - began).total_seconds()),
            "remaining_wall_seconds": max(0.0, (deadline - now).total_seconds()),
            "gpu_active_seconds": attempt_seconds(now),
            "remaining_gpu_seconds": max(0.0, 72_000 - attempt_seconds(now)),
            "attempts": attempts, "repairs": repairs,
            "retry_counts": retry_counts, "milestones": milestones,
            "error": error, "supervisor_pid": os.getpid(), "outcomes_sealed": True,
        }
        _atomic_json(progress_path, payload)

    try:
        while True:
            now = datetime.now(timezone.utc)
            if now >= deadline:
                status = "wall_cap_reached"
                break
            if attempt_seconds(now) >= 72_000:
                status = "gpu_cap_reached"
                break
            for identity, (process, record) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del active[identity]
                record["ended_utc"] = now.isoformat()
                record["exit_code"] = returncode
                if returncode == 0:
                    refreshed = _completed_nat_onoff_jobs(freeze, task_root)
                    if identity in refreshed:
                        completed |= refreshed
                        continue
                    error = "worker exited zero without a valid frozen completion receipt"
                else:
                    log_path = Path(str(record["log"]))
                    error = log_path.read_text("utf-8", errors="replace")[-8_000:]
                key = f"{identity[0]}:{identity[1]}"
                retry_counts[key] = retry_counts.get(key, 0) + 1
                if "out of memory" in str(error).lower():
                    try:
                        effective = downshift_nat_msep_concurrency(
                            effective, kind=identity[0]
                        )
                        repairs.append(
                            f"resource-only {identity[0]} concurrency downshift after OOM"
                        )
                    except ValueError:
                        pass
                if retry_counts[key] >= 3:
                    status = "repair_required"
                    write_progress(now)
                    return {
                        "status": status, "job": key,
                        "progress": str(progress_path),
                        "scientific_terminal_created": False,
                    }
            if completed == all_jobs:
                status = "fixed_work_complete"
                break
            if sum(kind == "train" for kind, _ in completed) == 16:
                milestones.setdefault("training_finished_utc", now.isoformat())
                theta0_values = {
                    json.loads(path.read_text("utf-8"))["theta0_sha256"]
                    for path in (task_root / "training").glob(
                        "*/TERMINAL_TRAINING_RESULT.json"
                    )
                }
                if len(theta0_values) != 1:
                    raise RuntimeError("Q1-NAT-ONOFF states do not share theta0")
            for job in next_nat_onoff_jobs(
                freeze["roots"], completed=completed, active=set(active),
                concurrency=effective,
            ):
                kind, job_id = str(job["kind"]), str(job["job_id"])
                key = f"{kind}:{job_id}"
                attempt = retry_counts.get(key, 0) + 1
                log = task_root / "logs" / f"{kind}_{job_id}_attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(freeze["runtime"]["python_judge" if kind == "score" else "python_gpu"]),
                    "-m", "oasst1_filter.runtime.nat_spec", "onoff-worker",
                    "--output-root", str(output_root), "--task-root", str(task_root),
                    "--kind", kind, "--job-id", job_id,
                    "--score-workers", str(effective["score"]),
                ]
                with log.open("ab") as handle:
                    process = subprocess.Popen(
                        command, stdout=handle, stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                    )
                record: dict[str, object] = {
                    "kind": kind, "job_id": job_id, "pid": process.pid,
                    "log": str(log), "started_utc": now.isoformat(),
                }
                attempts.append(record)
                active[(kind, job_id)] = (process, record)
            write_progress(now)
            time.sleep(2)
    finally:
        for process, _record in active.values():
            _stop_nat_msep_process(process)
        active.clear()

    finished = datetime.now(timezone.utc)
    milestones.setdefault("work_finished_utc", finished.isoformat())
    reason = (
        "formal_36h_resource_cap" if status == "wall_cap_reached"
        else "formal_20h_gpu_cap" if status == "gpu_cap_reached"
        else "not_evaluated"
    )
    rows = _nat_onoff_position_rows(freeze, task_root, missing_reason=reason)
    acceptance_finished = datetime.fromisoformat(str(freeze["acceptance"]["finished_utc"]))
    e0 = json.loads((task_root / "E0_TIMING.json").read_text("utf-8"))
    e0_finished = datetime.fromisoformat(str(e0["finished_utc"]))
    training_finished = datetime.fromisoformat(
        milestones.get("training_finished_utc", finished.isoformat())
    )
    phase_seconds = {
        "D3": max(0.0, (acceptance_finished - began).total_seconds()),
        "E0": max(0.0, (e0_finished - acceptance_finished).total_seconds()),
        "E1": max(0.0, (training_finished - e0_finished).total_seconds()),
        "E2": max(0.0, (finished - training_finished).total_seconds()),
        "E3": 0.0,
    }
    result = publish_nat_onoff(
        output_root=output_root, freeze=freeze, position_rows=rows,
        phase_seconds=phase_seconds, effective_concurrency=effective,
        repairs=repairs, gpu_active_seconds=attempt_seconds(finished),
        forced_terminal=(
            "Q1_ONOFF_INCOMPLETE_RESOURCE"
            if status in {"wall_cap_reached", "gpu_cap_reached"} else None
        ),
    )
    status = "terminal"
    completed = _completed_nat_onoff_jobs(freeze, task_root)
    write_progress(datetime.now(timezone.utc))
    return result


def _load_nat_e95_delete_freeze(output_root: Path) -> dict[str, object]:
    freeze = json.loads(
        (Path(output_root) / "EXECUTION_FREEZE.json").read_text("utf-8")
    )
    identity = freeze.get("scientific_identity_sha256")
    unsigned = dict(freeze)
    unsigned.pop("scientific_identity_sha256", None)
    if not (
        freeze.get("schema") == "q1-nat-e95-delete-execution-freeze-v1"
        and freeze.get("protocol_id") == "q1-nat-e95-delete-v1"
        and identity == _canonical_json_sha256(unsigned)
        and tuple(freeze.get("roots", ())) == NAT_E95_DELETE_ROOTS
        and len(freeze.get("plans", ())) == 8
        and len(freeze.get("task_ids", ())) == 164
        and freeze.get("gpu_cap_seconds") == 86_400
        and freeze.get("wall_cap_seconds") == 108_000
    ):
        raise ValueError("Q1-NAT-E95-DELETE execution freeze identity differs")
    return freeze


def _nat_e95_delete_training_card(
    freeze: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    card: dict[str, object] = {
        "schema": "q1-nat-e95-delete-run-card-v1",
        "protocol_id": "q1-nat-e95-delete-v1",
        "root": int(plan["root"]),
        "arm": "MID",
        "init_seed": 2026091201,
        "updates": 100,
        "max_length": 2048,
        "plan_sha256": str(plan["plan_sha256"]),
        "response_tokens": int(plan["response_tokens"]),
        "expected_theta0_sha256": str(
            freeze["training_recipe"]["expected_theta0_sha256"]
        ),
        "outcomes_opened": False,
    }
    card["card_sha256"] = _canonical_json_sha256(card)
    return card


def execute_nat_e95_delete_worker(
    *, output_root: Path, task_root: Path, kind: str, job_id: str,
    score_workers: int = 1,
) -> dict[str, object]:
    """Execute one frozen MID train, generation, or isolated score state."""

    freeze = _load_nat_e95_delete_freeze(output_root)
    legal = {
        (str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]
    }
    if (kind, job_id) not in legal:
        raise ValueError("Q1-NAT-E95-DELETE worker is outside the frozen queue")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(str(freeze["deadline_utc"])):
        raise RuntimeError("Q1-NAT-E95-DELETE 30h deadline is closed")
    plan_row = next(
        row for row in freeze["plans"] if f"{row['root']}_MID" == job_id
    )
    card = _nat_e95_delete_training_card(freeze, plan_row)
    task_root = Path(task_root)
    training_root = task_root / "training" / job_id
    if kind == "train":
        from ..train.run_codecontests import execute_nat_onoff_training

        return execute_nat_onoff_training(
            plan=load_s12_training_plan(Path(str(plan_row["plan_path"]))),
            card=card,
            model_path=Path(str(freeze["paths"]["model"])),
            output_dir=training_root,
            resume=(training_root / "recovery" / "CURRENT.json").is_file()
            or (training_root / "TERMINAL_TRAINING_RESULT.json").is_file(),
        )

    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    validate_nat_onoff_training_terminal(
        training_root / "TERMINAL_TRAINING_RESULT.json",
        expected_card_sha256=str(card["card_sha256"]),
        expected_plan_sha256=str(plan_row["plan_sha256"]),
        expected_protocol_id="q1-nat-e95-delete-v1",
        expected_theta0_sha256=NAT_E95_DELETE_THETA0_SHA256,
    )
    from ..eval.humaneval import (
        execute_humaneval_formal_generation,
        execute_humaneval_formal_scoring,
        load_humaneval_plus_asset,
    )

    tasks = load_humaneval_plus_asset(Path(str(freeze["paths"]["asset"])))
    if [task["task_id"] for task in tasks] != freeze["task_ids"]:
        raise ValueError("Q1-NAT-E95-DELETE endpoint identity differs")
    if kind == "generation":
        return execute_humaneval_formal_generation(
            state_id=job_id,
            tasks=tasks,
            model_path=Path(str(freeze["paths"]["model"])),
            adapter_path=training_root / "adapter100",
            output_dir=task_root / "generation" / job_id,
            design_sha256=str(freeze["scientific_identity_sha256"]),
        )
    if score_workers != 1:
        raise ValueError("each E95 score-state worker owns exactly one global slot")
    return execute_humaneval_formal_scoring(
        state_id=job_id,
        tasks=tasks,
        generation_dir=task_root / "generation" / job_id,
        output_dir=task_root / "scores" / job_id,
        design_sha256=str(freeze["scientific_identity_sha256"]),
        evaluator_source_root=Path(str(freeze["paths"]["evaluator_source"])),
        evaluator_archive=Path(str(freeze["paths"]["evaluator_archive"])),
        sandbox_base=task_root / "sandboxes" / job_id,
        workers=1,
        python_executable=Path(str(freeze["runtime"]["python_judge"])),
        require_isolation=True,
    )


def _completed_nat_e95_delete_jobs(
    freeze: Mapping[str, object], task_root: Path
) -> set[tuple[str, str]]:
    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    completed: set[tuple[str, str]] = set()
    for job in freeze["job_order"]:
        kind, job_id = str(job["kind"]), str(job["job_id"])
        if kind == "train":
            plan = next(
                row for row in freeze["plans"] if f"{row['root']}_MID" == job_id
            )
            card = _nat_e95_delete_training_card(freeze, plan)
            terminal = task_root / "training" / job_id / "TERMINAL_TRAINING_RESULT.json"
            if terminal.is_file():
                validate_nat_onoff_training_terminal(
                    terminal,
                    expected_card_sha256=str(card["card_sha256"]),
                    expected_plan_sha256=str(plan["plan_sha256"]),
                    expected_protocol_id="q1-nat-e95-delete-v1",
                    expected_theta0_sha256=NAT_E95_DELETE_THETA0_SHA256,
                )
                completed.add((kind, job_id))
            continue
        directory = "generation" if kind == "generation" else "scores"
        receipt = task_root / directory / job_id / "COMPLETE.json"
        if not receipt.is_file():
            continue
        value = json.loads(receipt.read_text("utf-8"))
        valid = (
            value.get("state_id") == job_id
            and value.get("completed_positions") == 164
            and value.get("design_sha256") == freeze["scientific_identity_sha256"]
        )
        if kind == "score":
            valid = valid and value.get("valid_positions") == 164 and (
                value.get("all_sandboxes_released") is True
            )
        if valid:
            completed.add((kind, job_id))
    return completed


def _nat_e95_delete_position_rows(
    freeze: Mapping[str, object], task_root: Path, *, missing_reason: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in freeze["roots"]:
        state = f"{root}_MID"
        for index, task_id in enumerate(freeze["task_ids"]):
            row: dict[str, object] = {
                "state_id": state,
                "root": int(root),
                "arm": "MID",
                "task_id": task_id,
                "Y": None,
                "verdict": "not_evaluated",
                "completion_sha256": None,
                "missing_reason": missing_reason,
            }
            score_path = task_root / "scores" / state / f"{index:03d}.json"
            if score_path.is_file():
                score = json.loads(score_path.read_text("utf-8"))
                if not (
                    score.get("state_id") == state
                    and score.get("task_id") == task_id
                    and score.get("task_index") == index
                    and score.get("design_sha256")
                    == freeze["scientific_identity_sha256"]
                ):
                    raise ValueError("Q1-NAT-E95-DELETE score binding differs")
                row.update(
                    Y=score.get("Y"),
                    verdict=score.get("verdict"),
                    completion_sha256=score.get("completion_sha256"),
                    missing_reason=(
                        None if score.get("Y") in (0, 1)
                        else str(score.get("detail") or "infrastructure_error")
                    ),
                )
            rows.append(row)
    return rows


def run_nat_e95_delete_formal(
    *, task_root: Path, output_root: Path
) -> dict[str, object]:
    """Run the finite eight-state MID queue under non-resetting caps."""

    task_root, output_root = Path(task_root), Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text("utf-8"))
    freeze = _load_nat_e95_delete_freeze(output_root)
    began = datetime.fromisoformat(str(freeze["started_utc"]))
    deadline = datetime.fromisoformat(str(freeze["deadline_utc"]))
    progress_path = task_root / "progress.json"
    previous = json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
    effective = dict(previous.get("effective_concurrency", freeze["effective_concurrency"]))
    attempts: list[dict[str, object]] = list(previous.get("attempts", ()))
    repairs: list[str] = list(previous.get("repairs", ()))
    retry_counts: dict[str, int] = dict(previous.get("retry_counts", {}))
    milestones: dict[str, str] = dict(previous.get("milestones", {}))
    completed = _completed_nat_e95_delete_jobs(freeze, task_root)
    all_jobs = {
        (str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]
    }
    active: dict[tuple[str, str], tuple[subprocess.Popen[bytes], dict[str, object]]] = {}
    acceptance_gpu = float(freeze["acceptance"].get("gpu_active_seconds", 0.0))
    gpu_cap = float(freeze["gpu_cap_seconds"])
    status, error = "running", None
    failure_terminal: str | None = None

    def gpu_seconds(now: datetime) -> float:
        intervals: list[tuple[datetime, datetime]] = []
        for record in attempts:
            if record.get("kind") not in {"train", "generation"}:
                continue
            start = datetime.fromisoformat(str(record["started_utc"]))
            end = (
                datetime.fromisoformat(str(record["ended_utc"]))
                if record.get("ended_utc") else now
            )
            if end > start:
                intervals.append((start, end))
        merged: list[list[datetime]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return acceptance_gpu + sum((end - start).total_seconds() for start, end in merged)

    def write_progress(now: datetime) -> None:
        trained = sum(kind == "train" for kind, _ in completed)
        generated = sum(kind == "generation" for kind, _ in completed)
        scored = sum(kind == "score" for kind, _ in completed)
        _atomic_json(progress_path, {
            "schema": "q1-nat-e95-delete-progress-v1",
            "protocol_id": "q1-nat-e95-delete-v1",
            "scientific_identity_sha256": freeze["scientific_identity_sha256"],
            "status": status,
            "stage": "training" if trained < 8 else "generation_scoring" if scored < 8 else "publication",
            "completed_training": trained,
            "completed_generation_states": generated,
            "completed_score_states": scored,
            "generated_positions": sum(1 for _ in (task_root / "generation").glob("*/[0-9][0-9][0-9].json")),
            "scored_positions": sum(1 for _ in (task_root / "scores").glob("*/[0-9][0-9][0-9].json")),
            "active": [record for _process, record in active.values()],
            "effective_concurrency": effective,
            "started_utc": began.isoformat(),
            "deadline_utc": deadline.isoformat(),
            "updated_utc": now.isoformat(),
            "elapsed_seconds": max(0.0, (now - began).total_seconds()),
            "remaining_wall_seconds": max(0.0, (deadline - now).total_seconds()),
            "gpu_active_seconds": gpu_seconds(now),
            "remaining_gpu_seconds": max(0.0, gpu_cap - gpu_seconds(now)),
            "attempts": attempts,
            "repairs": repairs,
            "retry_counts": retry_counts,
            "milestones": milestones,
            "error": error,
            "supervisor_pid": os.getpid(),
            "outcomes_sealed": True,
        })

    try:
        while True:
            now = datetime.now(timezone.utc)
            if now >= deadline:
                status = "wall_cap_reached"
                break
            if gpu_seconds(now) >= gpu_cap:
                status = "gpu_cap_reached"
                break
            for identity, (process, record) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del active[identity]
                record["ended_utc"] = now.isoformat()
                record["exit_code"] = returncode
                if returncode == 0:
                    refreshed = _completed_nat_e95_delete_jobs(freeze, task_root)
                    if identity in refreshed:
                        completed |= refreshed
                        continue
                    error = "worker exited zero without a valid frozen completion receipt"
                else:
                    log_path = Path(str(record["log"]))
                    error = log_path.read_text("utf-8", errors="replace")[-8_000:]
                key = f"{identity[0]}:{identity[1]}"
                retry_counts[key] = retry_counts.get(key, 0) + 1
                if "out of memory" in str(error).lower():
                    try:
                        effective = downshift_nat_msep_concurrency(effective, kind=identity[0])
                        repairs.append(f"resource-only {identity[0]} concurrency downshift after OOM")
                    except ValueError:
                        pass
                if retry_counts[key] >= 3:
                    status = "repair_required"
                    break
            if status == "repair_required":
                break
            if completed == all_jobs:
                status = "fixed_work_complete"
                break
            if sum(kind == "train" for kind, _ in completed) == 8:
                milestones.setdefault("training_finished_utc", now.isoformat())
                theta_values = {
                    json.loads(path.read_text("utf-8"))["theta0_sha256"]
                    for path in (task_root / "training").glob("*/TERMINAL_TRAINING_RESULT.json")
                }
                if theta_values != {NAT_E95_DELETE_THETA0_SHA256}:
                    raise RuntimeError("Q1-NAT-E95-DELETE theta0 closure differs")
            for job in next_nat_e95_delete_jobs(
                freeze["roots"], completed=completed, active=set(active), concurrency=effective
            ):
                kind, job_id = str(job["kind"]), str(job["job_id"])
                key = f"{kind}:{job_id}"
                attempt = retry_counts.get(key, 0) + 1
                log = task_root / "logs" / f"{kind}_{job_id}_attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(freeze["runtime"]["python_judge" if kind == "score" else "python_gpu"]),
                    "-m", "oasst1_filter.runtime.nat_spec", "e95-delete-worker",
                    "--output-root", str(output_root), "--task-root", str(task_root),
                    "--kind", kind, "--job-id", job_id, "--score-workers", "1",
                ]
                with log.open("ab") as handle:
                    process = subprocess.Popen(
                        command, stdout=handle, stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                    )
                record: dict[str, object] = {
                    "kind": kind, "job_id": job_id, "pid": process.pid,
                    "log": str(log), "started_utc": now.isoformat(),
                }
                attempts.append(record)
                active[(kind, job_id)] = (process, record)
            write_progress(now)
            time.sleep(2)
    except PermissionError as exc:
        status = "blocked_safety_or_permission"
        error = f"{type(exc).__name__}: {exc}"
        failure_terminal = "E95_BLOCKED_SAFETY_OR_PERMISSION"
    except (ValueError, RuntimeError) as exc:
        status = "measurement_or_binding_failure"
        error = f"{type(exc).__name__}: {exc}"
        message = str(exc).lower()
        failure_terminal = (
            "E95_NEEDS_SCIENTIFIC_REVIEW"
            if "historical" in message or "freeze" in message
            else "E95_MEASUREMENT_INVALID"
            if any(word in message for word in ("theta0", "score binding", "endpoint identity"))
            else "E95_INCOMPLETE_INFRASTRUCTURE"
        )
    except Exception as exc:
        status = "unexpected_infrastructure_failure"
        error = f"{type(exc).__name__}: {exc}"
        failure_terminal = "E95_INCOMPLETE_INFRASTRUCTURE"
    finally:
        # Resource terminals are hard stops: every task-owned process group is
        # reaped before any CPU-only missingness publication begins.
        for process, record in active.values():
            _stop_nat_msep_process(process)
            record.setdefault("ended_utc", datetime.now(timezone.utc).isoformat())
            record.setdefault("stopped_by_supervisor", status)
        active.clear()

    finished = datetime.now(timezone.utc)
    milestones.setdefault("work_finished_utc", finished.isoformat())
    missing_reason = {
        "wall_cap_reached": "formal_30h_resource_cap",
        "gpu_cap_reached": "formal_24h_gpu_cap",
        "repair_required": "unrecoverable_infrastructure_after_three_attempts",
    }.get(status, error or "not_evaluated")
    mid_rows = _nat_e95_delete_position_rows(
        freeze, task_root, missing_reason=missing_reason
    )
    with Path(str(freeze["paths"]["old_position_table"])).open(
        newline="", encoding="utf-8"
    ) as handle:
        historical_rows = list(csv.DictReader(handle))
    acceptance_finished = datetime.fromisoformat(str(freeze["acceptance"]["finished_utc"]))
    e0 = json.loads((task_root / "E0_TIMING.json").read_text("utf-8"))
    e0_finished = datetime.fromisoformat(str(e0["finished_utc"]))
    training_finished = datetime.fromisoformat(
        milestones.get("training_finished_utc", finished.isoformat())
    )
    forced = (
        "E95_INCOMPLETE_RESOURCE" if status in {"wall_cap_reached", "gpu_cap_reached"}
        else "E95_INCOMPLETE_INFRASTRUCTURE" if status == "repair_required"
        else failure_terminal
    )
    result = publish_nat_e95_delete(
        output_root=output_root,
        freeze=freeze,
        historical_position_rows=historical_rows,
        mid_position_rows=mid_rows,
        phase_seconds={
            "D3": max(0.0, (acceptance_finished - began).total_seconds()),
            "E0": max(0.0, (e0_finished - acceptance_finished).total_seconds()),
            "E1": max(0.0, (training_finished - e0_finished).total_seconds()),
            "E2": max(0.0, (finished - training_finished).total_seconds()),
            "E3": 0.0,
        },
        effective_concurrency=effective,
        repairs=repairs,
        gpu_active_seconds=gpu_seconds(finished),
        forced_terminal=forced,
    )
    status = "terminal"
    completed = _completed_nat_e95_delete_jobs(freeze, task_root)
    write_progress(datetime.now(timezone.utc))
    return result


def prepare_w259_cf12_design(
    *,
    design_root: Path,
    source_plan_path: Path,
    prospective_freeze_path: Path,
    onoff_freeze_path: Path,
    msep_freeze_path: Path,
    group_members_path: Path,
    block_binding_path: Path,
) -> dict[str, object]:
    """Create the outcome-blind CF12 context registry and forty-eight plans."""

    from ..arms import paired_filtering

    design_root = Path(design_root)
    receipt_path = design_root / "DESIGN_RECEIPT.json"
    if receipt_path.is_file():
        value = json.loads(receipt_path.read_text("utf-8"))
        if value.get("protocol_id") != W259_CF12_PROTOCOL_ID:
            raise FileExistsError("Q1-W259-CF12 design namespace is occupied")
        if any(
            not Path(str(row["plan_path"])).is_file()
            or _sha256_file(Path(str(row["plan_path"]))) != row["plan_file_sha256"]
            for row in value.get("plans", ())
        ):
            raise ValueError("Q1-W259-CF12 persisted design plan differs")
        return value

    prospective = json.loads(Path(prospective_freeze_path).read_text("utf-8"))
    onoff = json.loads(Path(onoff_freeze_path).read_text("utf-8"))
    msep = json.loads(Path(msep_freeze_path).read_text("utf-8"))
    if prospective.get("protocol_id") != "q1-w259-prospective-v1":
        raise ValueError("Q1-W259-CF12 prospective exclusion source differs")
    if onoff.get("protocol_id") != "q1-nat-onoff-354-v1":
        raise ValueError("Q1-W259-CF12 on/off exclusion source differs")
    if msep.get("protocol_id") != "codecontests-nat-msep-20-v1":
        raise ValueError("Q1-W259-CF12 MSEP exclusion source differs")
    exclusions = {2_026_091_201}
    exclusion_rows: list[dict[str, object]] = [
        {"source": "old_theta_initialization", "field": "init_seed", "value": 2_026_091_201}
    ]
    for row in prospective.get("pairs", ()):
        for field in ("init_seed", "root"):
            value = int(row[field])
            exclusions.add(value)
            exclusion_rows.append(
                {"source": "q1_w259_prospective", "field": field, "value": value}
            )
    for source, value in (
        [("q1_nat_onoff354", int(item)) for item in onoff.get("roots", ())]
        + [("codecontests_nat_msep20", int(item)) for item in msep.get("roots", ())]
    ):
        exclusions.add(value)
        exclusion_rows.append({"source": source, "field": "root", "value": value})
    contexts = derive_w259_cf12_contexts(excluded_values=exclusions)

    with Path(group_members_path).open(newline="", encoding="utf-8") as handle:
        group_rows = list(csv.DictReader(handle))
    group_by_unit = {
        str(row["canonical_record_id"]): str(row["group"]) for row in group_rows
    }
    if len(group_by_unit) != 259:
        raise ValueError("Q1-W259-CF12 static group membership count differs")
    group_hashes = {
        group: hashlib.sha256(
            ("".join(f"{unit_id}\n" for unit_id in sorted(
                unit for unit, observed in group_by_unit.items() if observed == group
            ))).encode("utf-8")
        ).hexdigest()
        for group in ("G", "H")
    }
    if group_hashes != {
        "G": "82d890e6931d88dd9a07d01e3145772661979ac8a727d6d2b2edfc95566a3e76",
        "H": "993f9b992d5e983ce2999234d118ae6304a708009b5a3e3f6e9bafbd8cf0ef7c",
    }:
        raise ValueError("Q1-W259-CF12 locked member hashes differ")
    with Path(block_binding_path).open(newline="", encoding="utf-8") as handle:
        binding_rows = list(csv.DictReader(handle))
    block_by_label = {
        str(row["static_block_id"]): int(row["original_canonical_block"])
        for row in binding_rows
    }
    if len(block_by_label) != 100 or set(block_by_label.values()) != set(range(100)):
        raise ValueError("Q1-W259-CF12 static block binding differs")
    if any(
        block_by_label.get(str(row["static_block_id"]))
        != int(row["original_canonical_block"])
        for row in group_rows
    ):
        raise ValueError("Q1-W259-CF12 member/block binding differs")

    source_plan_path = Path(source_plan_path)
    if _sha256_file(source_plan_path) != "4bea515785847cb537ad83681a36ae5a8365bc8e963df2e94b61cbe695a1963f":
        raise ValueError("Q1-W259-CF12 source P01 plan file differs")
    source_plan = paired_filtering.load_s12_training_plan(source_plan_path)
    source_root = 432_176_365
    source_order = tuple(
        sorted(
            range(100),
            key=lambda key: (
                hashlib.sha256(
                    f"q1-nat-onoff-354-v1|block-order|{source_root}|{key}".encode("ascii")
                ).digest(),
                key,
            ),
        )
    )
    plan_rows: list[dict[str, object]] = []
    block_summaries: list[dict[str, object]] = []
    plans_root = design_root / "plans"
    for context in contexts:
        root = int(context["root"])
        order = tuple(
            sorted(
                range(100),
                key=lambda key: (
                    hashlib.sha256(
                        f"q1-nat-onoff-354-v1|block-order|{root}|{key}".encode("ascii")
                    ).digest(),
                    key,
                ),
            )
        )
        order_digest = hashlib.sha256(",".join(map(str, order)).encode("ascii")).hexdigest()
        built = paired_filtering.build_w259_factorial_plans(
            root=root,
            source_w_plan=source_plan,
            source_block_order=source_order,
            group_by_unit=group_by_unit,
            expected_block_order_sha256=order_digest,
        )
        if built["component_exposures"] != {"B": 1996, "G": 121, "H": 138} or built[
            "component_tokens"
        ] != {"B": 583975, "G": 39714, "H": 39715}:
            raise ValueError("Q1-W259-CF12 component dose differs")
        block_summaries.extend(
            {"context_id": context["context_id"], "root": root, **dict(row)}
            for row in built["block_rows"]
        )
        for arm, expected_tokens in {
            "B": 583975, "BG": 623689, "BH": 623690, "BW": 663404
        }.items():
            plan_path = plans_root / f"{context['context_id']}_{arm}.json"
            plan_sha = paired_filtering.dump_s12_training_plan(
                plan_path, built["plans"][arm]
            )
            plan_rows.append(
                {
                    "context_id": context["context_id"],
                    "init_seed": context["init_seed"],
                    "root": root,
                    "arm": arm,
                    "block_order_sha256": order_digest,
                    "plan_path": str(plan_path),
                    "plan_sha256": plan_sha,
                    "plan_file_sha256": _sha256_file(plan_path),
                    "updates": 100,
                    "response_tokens": expected_tokens,
                }
            )
    design_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "schema": "q1-w259-cf12-design-receipt-v1",
        "protocol_id": W259_CF12_PROTOCOL_ID,
        "randomization_domain": W259_CF12_RANDOMIZATION_DOMAIN,
        "contexts": list(contexts),
        "exclusions": exclusion_rows,
        "plans": plan_rows,
        "job_order": list(w259_cf12_jobs(contexts)),
        "component_exposures": {"B": 1996, "G": 121, "H": 138},
        "component_tokens": {"B": 583975, "G": 39714, "H": 39715},
        "group_sha256": group_hashes,
        "source_bindings": {
            "source_plan_path": str(source_plan_path),
            "source_plan_file_sha256": _sha256_file(source_plan_path),
            "prospective_freeze_sha256": _sha256_file(Path(prospective_freeze_path)),
            "onoff_freeze_sha256": _sha256_file(Path(onoff_freeze_path)),
            "msep_freeze_sha256": _sha256_file(Path(msep_freeze_path)),
            "group_members_sha256": _sha256_file(Path(group_members_path)),
            "block_binding_sha256": _sha256_file(Path(block_binding_path)),
        },
        "block_rows_sha256": _canonical_json_sha256(block_summaries),
        "formal_outcomes_opened": False,
    }
    receipt["design_sha256"] = _canonical_json_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return receipt


def prepare_w259_cf12_execution(
    *,
    task_root: Path,
    output_root: Path,
    design_root: Path,
    acceptance_path: Path,
    model_path: Path,
    asset_path: Path,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    python_gpu: str,
    python_judge: str,
) -> dict[str, object]:
    """Freeze the accepted CF12 machine, contexts, plans, and endpoint."""

    from ..eval.humaneval import (
        EVALPLUS_REVISION,
        EVALPLUS_SOURCE_ARCHIVE_SHA256,
        HUMANEVAL_PLUS_LOGICAL_SHA256,
        HUMANEVAL_PLUS_SHA256,
        load_humaneval_plus_asset,
    )
    from ..train.run_codecontests import validate_w259_context_initial_state_set

    task_root, output_root, design_root = map(Path, (task_root, output_root, design_root))
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if freeze_path.is_file():
        return validate_w259_cf12_execution_freeze(
            json.loads(freeze_path.read_text("utf-8"))
        )
    design_path = design_root / "DESIGN_RECEIPT.json"
    design = json.loads(design_path.read_text("utf-8"))
    if (
        design.get("schema") != "q1-w259-cf12-design-receipt-v1"
        or design.get("protocol_id") != W259_CF12_PROTOCOL_ID
        or design.get("formal_outcomes_opened") is not False
    ):
        raise ValueError("Q1-W259-CF12 design receipt differs")
    contexts = _validate_w259_cf12_contexts(design["contexts"])
    plan_rows = []
    for source in design["plans"]:
        row = dict(source)
        plan_path = design_root / "plans" / f"{row['context_id']}_{row['arm']}.json"
        if not plan_path.is_file() or _sha256_file(plan_path) != row["plan_file_sha256"]:
            raise ValueError("Q1-W259-CF12 formal plan file differs")
        row["plan_path"] = str(plan_path)
        plan_rows.append(row)
    acceptance = json.loads(Path(acceptance_path).read_text("utf-8"))
    began = _parse_protocol_datetime(acceptance.get("started_utc"))
    deadline = _parse_protocol_datetime(acceptance.get("deadline_utc"))
    versions = acceptance.get("versions", {})
    initial_states = acceptance.get("initial_states")
    if not (
        acceptance.get("schema") == "q1-w259-cf12-machine-acceptance-v1"
        and acceptance.get("protocol_id") == W259_CF12_PROTOCOL_ID
        and acceptance.get("passed") is True
        and acceptance.get("clock_resets") is False
        and acceptance.get("isolation_passed") is True
        and deadline - began == timedelta(hours=66)
        and isinstance(initial_states, list)
        and isinstance(versions, Mapping)
        and str(versions.get("torch")) == "2.9.1+cu130"
        and str(versions.get("transformers")) == "5.3.0"
        and str(versions.get("peft")) == "0.18.1"
        and str(versions.get("numpy")) == "2.1.2"
    ):
        raise ValueError("Q1-W259-CF12 machine acceptance is not passed")
    expected_contexts = {
        str(row["context_id"]): int(row["init_seed"]) for row in contexts
    }
    theta_by_context = validate_w259_context_initial_state_set(
        initial_states, expected_contexts=expected_contexts
    )
    initial_by_context = {str(row["context_id"]): row for row in initial_states}
    tasks = load_humaneval_plus_asset(Path(asset_path))
    task_ids = [str(task["task_id"]) for task in tasks]
    if task_ids != [f"HumanEval/{index}" for index in range(164)]:
        raise ValueError("Q1-W259-CF12 HumanEval+ endpoint differs")
    if Path(model_path).name != "8faed761d45a263340a0528343f099c05c9a4323" or _sha256_file(
        Path(evaluator_archive)
    ) != EVALPLUS_SOURCE_ARCHIVE_SHA256:
        raise ValueError("Q1-W259-CF12 model/evaluator authority differs")
    frozen_contexts = [
        {
            **dict(row),
            "theta_i_sha256": theta_by_context[str(row["context_id"])],
            "initial_adapter_path": str(
                initial_by_context[str(row["context_id"])]["initial_adapter_path"]
            ),
            "initial_adapter_files": initial_by_context[str(row["context_id"])][
                "adapter_files"
            ],
        }
        for row in contexts
    ]
    output_root.mkdir(parents=True, exist_ok=False)
    freeze: dict[str, object] = {
        "schema": "q1-w259-cf12-execution-freeze-v1",
        "protocol_id": W259_CF12_PROTOCOL_ID,
        "started_utc": began.astimezone(timezone.utc).isoformat(),
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat(),
        "clock_resets": False,
        "contexts": frozen_contexts,
        "plans": plan_rows,
        "task_ids": task_ids,
        "job_order": list(w259_cf12_jobs(frozen_contexts)),
        "paths": {
            "model": str(model_path),
            "asset": str(asset_path),
            "evaluator_source": str(evaluator_source_root),
            "evaluator_archive": str(evaluator_archive),
            "design_root": str(design_root),
        },
        "runtime": {"python_gpu": python_gpu, "python_judge": python_judge},
        "bindings": {
            "design_sha256": design["design_sha256"],
            "design_receipt_sha256": _sha256_file(design_path),
            "model_revision": Path(model_path).name,
            "humaneval_compressed_sha256": HUMANEVAL_PLUS_SHA256,
            "humaneval_logical_sha256": HUMANEVAL_PLUS_LOGICAL_SHA256,
            "evaluator_revision": EVALPLUS_REVISION,
            "evaluator_archive_sha256": EVALPLUS_SOURCE_ARCHIVE_SHA256,
        },
        "data_contract": {
            "B_exposures": 1996, "G_exposures": 121, "H_exposures": 138,
            "B_tokens": 583975, "G_tokens": 39714, "H_tokens": 39715,
            "blocks": 100,
        },
        "training_recipe": {
            "updates": 100, "max_length": 2048, "lora_rank": 16,
            "lora_alpha": 32, "lora_dropout": 0.05,
            "context_initialization": "one persisted fresh LoRA state shared by four arms",
            "optimizer": "fresh empty optimizer per arm",
        },
        "generation_recipe": {
            "do_sample": False, "max_new_tokens": 1024,
            "endpoint": "HumanEval+164 exact",
        },
        "acceptance": acceptance,
        "isolation": acceptance["isolation"],
        "software_versions": dict(versions),
        "target_concurrency": dict(W259_CF12_CONCURRENCY),
        "effective_concurrency": dict(W259_CF12_CONCURRENCY),
        "resource_ladders": {"train": [2, 1], "generation": [4, 2, 1], "score": [4, 2, 1]},
        "gpu_cap_seconds": W259_CF12_GPU_CAP_SECONDS,
        "wall_cap_seconds": W259_CF12_WALL_CAP_SECONDS,
        "formal_outcomes_opened": False,
        "terminal_set": sorted(W259_CF12_TERMINALS),
    }
    freeze["scientific_identity_sha256"] = _canonical_json_sha256(freeze)
    validate_w259_cf12_execution_freeze(freeze)
    _atomic_json(freeze_path, freeze)
    _atomic_json(
        task_root / "E0_TIMING.json",
        {"started_utc": acceptance["finished_utc"], "finished_utc": datetime.now(timezone.utc).isoformat()},
    )
    return freeze


def prepare_w259_execution(
    *,
    task_root: Path,
    output_root: Path,
    old_output_root: Path,
    acceptance_path: Path,
    pair_plan_path: Path,
    s12d_table_path: Path,
    model_path: Path,
    asset_path: Path,
    evaluator_source_root: Path,
    evaluator_archive: Path,
    python_gpu: str,
    python_judge: str,
) -> dict[str, object]:
    """Freeze eight fresh pairs and sixteen exact B+W/B plans."""

    from ..arms import paired_filtering
    from ..eval.humaneval import (
        EVALPLUS_REVISION,
        EVALPLUS_SOURCE_ARCHIVE_SHA256,
        HUMANEVAL_PLUS_LOGICAL_SHA256,
        HUMANEVAL_PLUS_SHA256,
        load_humaneval_plus_asset,
    )
    from ..train.run_codecontests import validate_w259_initial_state_set

    task_root, output_root = Path(task_root), Path(output_root)
    freeze_path = output_root / "EXECUTION_FREEZE.json"
    if freeze_path.is_file():
        return validate_w259_execution_freeze(
            json.loads(freeze_path.read_text("utf-8"))
        )
    old_output_root = Path(old_output_root)
    old_freeze_path = old_output_root / "EXECUTION_FREEZE.json"
    old_result_path = old_output_root / "RESULT.json"
    old_freeze = json.loads(old_freeze_path.read_text("utf-8"))
    old_result = json.loads(old_result_path.read_text("utf-8"))
    old_identity = "5fa7b40099ea8b1476b4b432dbb5aea480e850c1eda6351e08247cbe25324266"
    task_ids = tuple(str(task) for task in old_freeze.get("task_ids", ()))
    if not (
        old_freeze.get("schema") == "q1-nat-onoff-354-execution-freeze-v1"
        and old_freeze.get("protocol_id") == "q1-nat-onoff-354-v1"
        and old_freeze.get("scientific_identity_sha256") == old_identity
        and old_result.get("protocol_id") == "q1-nat-onoff-354-v1"
        and old_result.get("scientific_identity_sha256") == old_identity
        and old_result.get("complete") is True
        and old_result.get("terminal") == "Q1_ONOFF_POSITIVE_EXPLORATORY"
        and task_ids == tuple(f"HumanEval/{index}" for index in range(164))
    ):
        raise ValueError("Q1-W259-PROSPECTIVE historical endpoint binding differs")

    pair_plan_path, s12d_table_path = Path(pair_plan_path), Path(s12d_table_path)
    if _sha256_file(pair_plan_path) != "358d7a0c5462dcebdb2bf679206c2d3409fd1e139ee82fcd425f8c12b11777b3":
        raise ValueError("Q1-W259-PROSPECTIVE PAIR_PLAN differs")
    if _sha256_file(s12d_table_path) != NAT_E95_DELETE_S12D_TABLE_SHA256:
        raise ValueError("Q1-W259-PROSPECTIVE S12d category table differs")
    with pair_plan_path.open(newline="", encoding="utf-8") as handle:
        pair_plan_rows = list(csv.DictReader(handle))
    observed_pairs = tuple(
        (
            str(row["pair_id"]), int(row["init_seed"]), int(row["root"]),
            str(row["block_order_sha256"]),
        )
        for row in pair_plan_rows
    )
    if observed_pairs != W259_PAIRS or any(
        int(row["w_use_tokens"]) != 663_404
        or int(row["w_omit_tokens"]) != 583_975
        or int(row["updates"]) != 100
        for row in pair_plan_rows
    ):
        raise ValueError("Q1-W259-PROSPECTIVE pair plan values differ")

    categories: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    with s12d_table_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("natural_member") != "True":
                continue
            unit_id = f"{row['triplet_id']}/I"
            category = str(row["i_category"])
            if unit_id in categories:
                raise ValueError("Q1-W259-PROSPECTIVE supplement identity is duplicated")
            categories[unit_id] = category
            category_counts[category] = category_counts.get(category, 0) + 1
    if category_counts != {
        "wrong_answer": 259,
        "compile_failure": 12,
        "runtime_failure": 75,
        "timeout": 8,
    }:
        raise ValueError("Q1-W259-PROSPECTIVE W/E category ledger differs")

    geometry = old_freeze.get("geometry")
    if not isinstance(geometry, list) or len(geometry) != 8:
        raise ValueError("Q1-W259-PROSPECTIVE historical geometry is absent")
    supplement_maps = [dict(row["supplement_block_by_unit"]) for row in geometry]
    if any(value != supplement_maps[0] for value in supplement_maps[1:]):
        raise ValueError("Q1-W259-PROSPECTIVE historical supplement block maps differ")
    source_geometry = geometry[0]
    source_root = int(source_geometry["root"])
    source_plan_row = next(
        row for row in old_freeze["plans"]
        if int(row["root"]) == source_root and row["arm"] == "ON"
    )
    source_plan_path = Path(str(source_plan_row["plan_path"]))
    if (
        not source_plan_path.is_file()
        or _sha256_file(source_plan_path) != source_plan_row["plan_file_sha256"]
    ):
        raise ValueError("Q1-W259-PROSPECTIVE historical source plan differs")
    source_plan = paired_filtering.load_s12_training_plan(source_plan_path)

    acceptance = json.loads(Path(acceptance_path).read_text("utf-8"))
    began = _parse_protocol_datetime(acceptance.get("started_utc"))
    deadline = _parse_protocol_datetime(acceptance.get("deadline_utc"))
    versions = acceptance.get("versions", {})
    initial_states = acceptance.get("initial_states")
    if not (
        acceptance.get("schema") == "q1-w259-prospective-machine-acceptance-v1"
        and acceptance.get("protocol_id") == "q1-w259-prospective-v1"
        and acceptance.get("passed") is True
        and acceptance.get("clock_resets") is False
        and acceptance.get("isolation_passed") is True
        and deadline - began == timedelta(hours=60)
        and isinstance(versions, Mapping)
        and str(versions.get("torch")) == "2.9.1+cu130"
        and str(versions.get("transformers")) == "5.3.0"
        and str(versions.get("peft")) == "0.18.1"
        and str(versions.get("numpy")) == "2.1.2"
        and isinstance(initial_states, list)
    ):
        raise ValueError("Q1-W259-PROSPECTIVE machine acceptance is not passed")
    theta_by_pair = validate_w259_initial_state_set(
        initial_states,
        expected_pairs={pair_id: seed for pair_id, seed, _root, _digest in W259_PAIRS},
    )
    initial_by_pair = {str(row["pair_id"]): row for row in initial_states}

    task_root.mkdir(parents=True, exist_ok=True)
    plans_root = task_root / "plans"
    plan_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for pair_id, init_seed, root, order_digest in W259_PAIRS:
        built = paired_filtering.build_w259_prospective_plans(
            root=root,
            old_on_plan=source_plan,
            old_block_order=tuple(int(value) for value in source_geometry["block_order"]),
            supplement_category_by_unit=categories,
            expected_supplement_block_by_unit=supplement_maps[0],
            expected_block_order_sha256=order_digest,
        )
        if not (
            built["component_exposures"] == {"B": 1996, "W": 259, "E_deleted": 95}
            and built["component_tokens"] == {"B": 583975, "W": 79429, "E_deleted": 26095}
            and len(built["block_rows"]) == 100
        ):
            raise ValueError("Q1-W259-PROSPECTIVE B/W dose differs")
        block_rows.extend({"pair_id": pair_id, "root": root, **dict(row)} for row in built["block_rows"])
        for arm in ("W_USE", "W_OMIT"):
            plan = built["plans"][arm]
            plan_path = plans_root / f"{pair_id}_{arm}.json"
            plan_sha = paired_filtering.dump_s12_training_plan(plan_path, plan)
            plan_rows.append(
                {
                    "pair_id": pair_id,
                    "init_seed": init_seed,
                    "root": root,
                    "arm": arm,
                    "plan_path": str(plan_path),
                    "plan_sha256": plan_sha,
                    "plan_file_sha256": _sha256_file(plan_path),
                    "updates": 100,
                    "response_tokens": 663_404 if arm == "W_USE" else 583_975,
                    "component_tokens": {
                        "B": 583_975,
                        "W": 79_429 if arm == "W_USE" else 0,
                    },
                }
            )

    tasks = load_humaneval_plus_asset(Path(asset_path))
    if [str(task["task_id"]) for task in tasks] != list(task_ids):
        raise ValueError("Q1-W259-PROSPECTIVE HumanEval+ panel differs")
    if (
        Path(model_path).name != "8faed761d45a263340a0528343f099c05c9a4323"
        or _sha256_file(Path(evaluator_archive)) != EVALPLUS_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("Q1-W259-PROSPECTIVE model/evaluator authority differs")

    pair_rows = [
        {
            "pair_id": pair_id,
            "init_seed": seed,
            "root": root,
            "block_order_sha256": order_digest,
            "theta_i_sha256": theta_by_pair[pair_id],
            "initial_adapter_path": str(initial_by_pair[pair_id]["initial_adapter_path"]),
            "initial_adapter_files": initial_by_pair[pair_id]["adapter_files"],
        }
        for pair_id, seed, root, order_digest in W259_PAIRS
    ]
    output_root.mkdir(parents=True, exist_ok=False)
    freeze: dict[str, object] = {
        "schema": "q1-w259-prospective-execution-freeze-v1",
        "protocol_id": "q1-w259-prospective-v1",
        "started_utc": began.astimezone(timezone.utc).isoformat(),
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat(),
        "clock_resets": False,
        "pairs": pair_rows,
        "plans": plan_rows,
        "task_ids": list(task_ids),
        "job_order": list(w259_jobs(W259_PAIRS)),
        "paths": {
            "model": str(model_path),
            "asset": str(asset_path),
            "evaluator_source": str(evaluator_source_root),
            "evaluator_archive": str(evaluator_archive),
            "old_output": str(old_output_root),
            "pair_plan": str(pair_plan_path),
            "s12d_table": str(s12d_table_path),
            "source_old_on_plan": str(source_plan_path),
        },
        "runtime": {"python_gpu": python_gpu, "python_judge": python_judge},
        "bindings": {
            "old_scientific_identity_sha256": old_identity,
            "old_freeze_sha256": _sha256_file(old_freeze_path),
            "old_result_sha256": _sha256_file(old_result_path),
            "pair_plan_sha256": _sha256_file(pair_plan_path),
            "s12d_table_sha256": _sha256_file(s12d_table_path),
            "model_revision": Path(model_path).name,
            "humaneval_compressed_sha256": HUMANEVAL_PLUS_SHA256,
            "humaneval_logical_sha256": HUMANEVAL_PLUS_LOGICAL_SHA256,
            "evaluator_revision": EVALPLUS_REVISION,
            "evaluator_archive_sha256": EVALPLUS_SOURCE_ARCHIVE_SHA256,
        },
        "data_contract": {
            "B_exposures": 1996,
            "W_exposures": 259,
            "E_deleted_exposures": 95,
            "B_tokens": 583975,
            "W_tokens": 79429,
            "E_deleted_tokens": 26095,
            "W_USE_tokens": 663404,
            "W_OMIT_tokens": 583975,
            "blocks": 100,
            "block_rows_sha256": _canonical_json_sha256(block_rows),
        },
        "training_recipe": {
            "updates": 100,
            "max_length": 2048,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "optimizer": "AdamW lr=2e-4 betas=(0.9,0.95) eps=1e-8 wd=0 foreach=false fused=false",
            "pair_initialization": "one persisted fresh LoRA state shared only within each pair",
            "rng_streams": "root-derived B/S streams reset after pair initialization",
        },
        "generation_recipe": {
            "do_sample": False,
            "max_new_tokens": 1024,
            "stop_strings": 8,
            "endpoint": "HumanEval+164 exact",
        },
        "acceptance": acceptance,
        "isolation": acceptance["isolation"],
        "software_versions": dict(versions),
        "target_concurrency": dict(NAT_ONOFF_CONCURRENCY),
        "effective_concurrency": dict(NAT_ONOFF_CONCURRENCY),
        "resource_ladders": {"train": [2, 1], "generation": [4, 2, 1], "score": [4, 2, 1]},
        "gpu_cap_seconds": 72_000,
        "wall_cap_seconds": 216_000,
        "formal_outcomes_opened": False,
        "terminal_set": sorted(W259_TERMINALS),
    }
    freeze["scientific_identity_sha256"] = _canonical_json_sha256(freeze)
    validate_w259_execution_freeze(freeze)
    _atomic_json(freeze_path, freeze)
    _atomic_json(
        task_root / "E0_TIMING.json",
        {
            "started_utc": acceptance["finished_utc"],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return freeze


def _load_w259_freeze(output_root: Path) -> dict[str, object]:
    freeze = json.loads(
        (Path(output_root) / "EXECUTION_FREEZE.json").read_text("utf-8")
    )
    return validate_w259_execution_freeze(freeze)


def _w259_training_card(
    freeze: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    pair = next(
        row for row in freeze["pairs"] if row["pair_id"] == plan["pair_id"]
    )
    card: dict[str, object] = {
        "schema": "q1-w259-prospective-run-card-v1",
        "protocol_id": "q1-w259-prospective-v1",
        "pair_id": str(plan["pair_id"]),
        "root": int(plan["root"]),
        "arm": str(plan["arm"]),
        "init_seed": int(pair["init_seed"]),
        "updates": 100,
        "max_length": 2048,
        "plan_sha256": str(plan["plan_sha256"]),
        "response_tokens": int(plan["response_tokens"]),
        "expected_theta_i_sha256": str(pair["theta_i_sha256"]),
        "initial_adapter_path": str(pair["initial_adapter_path"]),
        "outcomes_opened": False,
    }
    card["card_sha256"] = _canonical_json_sha256(card)
    return card


def execute_w259_worker(
    *,
    output_root: Path,
    task_root: Path,
    kind: str,
    job_id: str,
    score_workers: int = 1,
) -> dict[str, object]:
    """Execute one frozen W259 training, generation, or score state."""

    freeze = _load_w259_freeze(output_root)
    legal = {(str(row["kind"]), str(row["job_id"])) for row in freeze["job_order"]}
    if (kind, job_id) not in legal:
        raise ValueError("Q1-W259-PROSPECTIVE worker is outside the frozen queue")
    if datetime.now(timezone.utc) >= _parse_protocol_datetime(freeze["deadline_utc"]):
        raise RuntimeError("Q1-W259-PROSPECTIVE 60h deadline is closed")
    plan_row = next(
        row for row in freeze["plans"]
        if f"{row['pair_id']}_{row['arm']}" == job_id
    )
    card = _w259_training_card(freeze, plan_row)
    task_root = Path(task_root)
    training_root = task_root / "training" / job_id
    if kind == "train":
        from ..train.run_codecontests import execute_nat_onoff_training

        return execute_nat_onoff_training(
            plan=load_s12_training_plan(Path(str(plan_row["plan_path"]))),
            card=card,
            model_path=Path(str(freeze["paths"]["model"])),
            output_dir=training_root,
            resume=(training_root / "recovery" / "CURRENT.json").is_file()
            or (training_root / "TERMINAL_TRAINING_RESULT.json").is_file(),
        )
    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    validate_nat_onoff_training_terminal(
        training_root / "TERMINAL_TRAINING_RESULT.json",
        expected_card_sha256=str(card["card_sha256"]),
        expected_plan_sha256=str(plan_row["plan_sha256"]),
        expected_protocol_id="q1-w259-prospective-v1",
        expected_theta0_sha256=str(card["expected_theta_i_sha256"]),
    )
    from ..eval.humaneval import (
        execute_humaneval_formal_generation,
        execute_humaneval_formal_scoring,
        load_humaneval_plus_asset,
    )

    tasks = load_humaneval_plus_asset(Path(str(freeze["paths"]["asset"])))
    if [task["task_id"] for task in tasks] != freeze["task_ids"]:
        raise ValueError("Q1-W259-PROSPECTIVE endpoint identity differs")
    if kind == "generation":
        return execute_humaneval_formal_generation(
            state_id=job_id,
            tasks=tasks,
            model_path=Path(str(freeze["paths"]["model"])),
            adapter_path=training_root / "adapter100",
            output_dir=task_root / "generation" / job_id,
            design_sha256=str(freeze["scientific_identity_sha256"]),
        )
    if score_workers != 1:
        raise ValueError("each W259 score-state worker owns exactly one global slot")
    return execute_humaneval_formal_scoring(
        state_id=job_id,
        tasks=tasks,
        generation_dir=task_root / "generation" / job_id,
        output_dir=task_root / "scores" / job_id,
        design_sha256=str(freeze["scientific_identity_sha256"]),
        evaluator_source_root=Path(str(freeze["paths"]["evaluator_source"])),
        evaluator_archive=Path(str(freeze["paths"]["evaluator_archive"])),
        sandbox_base=task_root / "sandboxes" / job_id,
        workers=1,
        python_executable=Path(str(freeze["runtime"]["python_judge"])),
        require_isolation=True,
    )


def _completed_w259_jobs(
    freeze: Mapping[str, object], task_root: Path
) -> set[tuple[str, str]]:
    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    completed: set[tuple[str, str]] = set()
    for job in freeze["job_order"]:
        kind, job_id = str(job["kind"]), str(job["job_id"])
        if kind == "train":
            plan = next(
                row for row in freeze["plans"]
                if f"{row['pair_id']}_{row['arm']}" == job_id
            )
            card = _w259_training_card(freeze, plan)
            terminal = task_root / "training" / job_id / "TERMINAL_TRAINING_RESULT.json"
            if terminal.is_file():
                validate_nat_onoff_training_terminal(
                    terminal,
                    expected_card_sha256=str(card["card_sha256"]),
                    expected_plan_sha256=str(plan["plan_sha256"]),
                    expected_protocol_id="q1-w259-prospective-v1",
                    expected_theta0_sha256=str(card["expected_theta_i_sha256"]),
                )
                completed.add((kind, job_id))
            continue
        directory = "generation" if kind == "generation" else "scores"
        receipt = task_root / directory / job_id / "COMPLETE.json"
        if not receipt.is_file():
            continue
        value = json.loads(receipt.read_text("utf-8"))
        valid = (
            value.get("state_id") == job_id
            and value.get("completed_positions") == 164
            and value.get("design_sha256") == freeze["scientific_identity_sha256"]
        )
        if kind == "score":
            valid = valid and value.get("valid_positions") == 164 and (
                value.get("all_sandboxes_released") is True
            )
        if valid:
            completed.add((kind, job_id))
    return completed


def _w259_position_rows(
    freeze: Mapping[str, object], task_root: Path, *, missing_reason: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in freeze["pairs"]:
        pair_id, root = str(pair["pair_id"]), int(pair["root"])
        for arm in ("W_USE", "W_OMIT"):
            state = f"{pair_id}_{arm}"
            for index, task_id in enumerate(freeze["task_ids"]):
                row: dict[str, object] = {
                    "state_id": state,
                    "pair_id": pair_id,
                    "root": root,
                    "arm": arm,
                    "task_id": task_id,
                    "Y": None,
                    "verdict": "not_evaluated",
                    "completion_sha256": None,
                    "missing_reason": missing_reason,
                }
                score_path = task_root / "scores" / state / f"{index:03d}.json"
                if score_path.is_file():
                    score = json.loads(score_path.read_text("utf-8"))
                    if not (
                        score.get("state_id") == state
                        and score.get("task_id") == task_id
                        and score.get("task_index") == index
                        and score.get("design_sha256") == freeze["scientific_identity_sha256"]
                    ):
                        raise ValueError("Q1-W259-PROSPECTIVE score binding differs")
                    row.update(
                        Y=score.get("Y"),
                        verdict=score.get("verdict"),
                        completion_sha256=score.get("completion_sha256"),
                        missing_reason=(
                            None
                            if score.get("Y") in (0, 1)
                            else str(score.get("detail") or "infrastructure_error")
                        ),
                    )
                rows.append(row)
    return rows


def run_w259_formal(*, task_root: Path, output_root: Path) -> dict[str, object]:
    """Run the finite sixteen-state W259 queue under absolute caps."""

    task_root, output_root = Path(task_root), Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text("utf-8"))
    freeze = _load_w259_freeze(output_root)
    began = _parse_protocol_datetime(freeze["started_utc"])
    deadline = _parse_protocol_datetime(freeze["deadline_utc"])
    progress_path = task_root / "progress.json"
    previous = json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
    effective = dict(previous.get("effective_concurrency", freeze["effective_concurrency"]))
    attempts: list[dict[str, object]] = list(previous.get("attempts", ()))
    repairs: list[str] = list(previous.get("repairs", ()))
    retry_counts: dict[str, int] = dict(previous.get("retry_counts", {}))
    milestones: dict[str, str] = dict(previous.get("milestones", {}))
    completed = _completed_w259_jobs(freeze, task_root)
    all_jobs = {(str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]}
    active: dict[tuple[str, str], tuple[subprocess.Popen[bytes], dict[str, object]]] = {}
    acceptance_gpu = float(freeze["acceptance"].get("gpu_active_seconds", 0.0))
    gpu_cap = float(freeze["gpu_cap_seconds"])
    status, error = "running", None
    failure_terminal: str | None = None

    def gpu_seconds(now: datetime) -> float:
        intervals: list[tuple[datetime, datetime]] = []
        for record in attempts:
            if record.get("kind") not in {"train", "generation"}:
                continue
            start = _parse_protocol_datetime(record["started_utc"])
            end = _parse_protocol_datetime(record["ended_utc"]) if record.get("ended_utc") else now
            if end > start:
                intervals.append((start, end))
        merged: list[list[datetime]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return acceptance_gpu + sum((end - start).total_seconds() for start, end in merged)

    def write_progress(now: datetime) -> None:
        trained = sum(kind == "train" for kind, _ in completed)
        generated = sum(kind == "generation" for kind, _ in completed)
        scored = sum(kind == "score" for kind, _ in completed)
        _atomic_json(
            progress_path,
            {
                "schema": "q1-w259-prospective-progress-v1",
                "protocol_id": "q1-w259-prospective-v1",
                "scientific_identity_sha256": freeze["scientific_identity_sha256"],
                "status": status,
                "stage": "training" if trained < 16 else "generation_scoring" if scored < 16 else "publication",
                "completed_training": trained,
                "completed_generation_states": generated,
                "completed_score_states": scored,
                "generated_positions": sum(1 for _ in (task_root / "generation").glob("*/[0-9][0-9][0-9].json")),
                "scored_positions": sum(1 for _ in (task_root / "scores").glob("*/[0-9][0-9][0-9].json")),
                "active": [record for _process, record in active.values()],
                "effective_concurrency": effective,
                "started_utc": began.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "updated_utc": now.isoformat(),
                "elapsed_seconds": max(0.0, (now - began).total_seconds()),
                "remaining_wall_seconds": max(0.0, (deadline - now).total_seconds()),
                "gpu_active_seconds": gpu_seconds(now),
                "remaining_gpu_seconds": max(0.0, gpu_cap - gpu_seconds(now)),
                "attempts": attempts,
                "repairs": repairs,
                "retry_counts": retry_counts,
                "milestones": milestones,
                "error": error,
                "supervisor_pid": os.getpid(),
                "outcomes_sealed": True,
            },
        )

    try:
        while True:
            now = datetime.now(timezone.utc)
            if now >= deadline:
                status = "wall_cap_reached"
                break
            if gpu_seconds(now) >= gpu_cap:
                status = "gpu_cap_reached"
                break
            for identity, (process, record) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del active[identity]
                record["ended_utc"] = now.isoformat()
                record["exit_code"] = returncode
                if returncode == 0:
                    refreshed = _completed_w259_jobs(freeze, task_root)
                    if identity in refreshed:
                        completed |= refreshed
                        continue
                    error = "worker exited zero without a valid frozen completion receipt"
                else:
                    log_path = Path(str(record["log"]))
                    error = log_path.read_text("utf-8", errors="replace")[-8_000:]
                key = f"{identity[0]}:{identity[1]}"
                retry_counts[key] = retry_counts.get(key, 0) + 1
                if "out of memory" in str(error).lower():
                    try:
                        effective = downshift_nat_msep_concurrency(effective, kind=identity[0])
                        repairs.append(f"resource-only {identity[0]} concurrency downshift after OOM")
                    except ValueError:
                        pass
                if retry_counts[key] >= 3:
                    status = "repair_required"
                    break
            if status == "repair_required":
                break
            if completed == all_jobs:
                status = "fixed_work_complete"
                break
            if sum(kind == "train" for kind, _ in completed) == 16:
                milestones.setdefault("training_finished_utc", now.isoformat())
                theta_by_pair: dict[str, set[str]] = {}
                for path in (task_root / "training").glob("*/TERMINAL_TRAINING_RESULT.json"):
                    value = json.loads(path.read_text("utf-8"))
                    theta_by_pair.setdefault(str(value.get("pair_id")), set()).add(str(value.get("theta_i_sha256")))
                expected_theta = {str(row["pair_id"]): str(row["theta_i_sha256"]) for row in freeze["pairs"]}
                if set(theta_by_pair) != set(expected_theta) or any(
                    values != {expected_theta[pair_id]} for pair_id, values in theta_by_pair.items()
                ):
                    raise RuntimeError("Q1-W259-PROSPECTIVE theta_i pair closure differs")
            for job in next_w259_jobs(
                W259_PAIRS, completed=completed, active=set(active), concurrency=effective
            ):
                kind, job_id = str(job["kind"]), str(job["job_id"])
                key = f"{kind}:{job_id}"
                attempt = retry_counts.get(key, 0) + 1
                log = task_root / "logs" / f"{kind}_{job_id}_attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(freeze["runtime"]["python_judge" if kind == "score" else "python_gpu"]),
                    "-m", "oasst1_filter.runtime.nat_spec", "w259-worker",
                    "--output-root", str(output_root), "--task-root", str(task_root),
                    "--kind", kind, "--job-id", job_id, "--score-workers", "1",
                ]
                with log.open("ab") as handle:
                    process = subprocess.Popen(
                        command,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                    )
                record = {
                    "kind": kind,
                    "job_id": job_id,
                    "pid": process.pid,
                    "log": str(log),
                    "started_utc": now.isoformat(),
                }
                attempts.append(record)
                active[(kind, job_id)] = (process, record)
            write_progress(now)
            time.sleep(2)
    except PermissionError as exc:
        status, error = "blocked_safety_or_permission", f"{type(exc).__name__}: {exc}"
        failure_terminal = "W_BLOCKED_SAFETY_OR_PERMISSION"
    except (ValueError, RuntimeError) as exc:
        status, error = "measurement_or_binding_failure", f"{type(exc).__name__}: {exc}"
        message = str(exc).lower()
        failure_terminal = (
            "W_NEEDS_SCIENTIFIC_REVIEW"
            if "historical" in message or "freeze" in message
            else "W_MEASUREMENT_INVALID"
            if any(word in message for word in ("theta_i", "score binding", "endpoint identity"))
            else "W_INCOMPLETE_INFRASTRUCTURE"
        )
    except Exception as exc:
        status, error = "unexpected_infrastructure_failure", f"{type(exc).__name__}: {exc}"
        failure_terminal = "W_INCOMPLETE_INFRASTRUCTURE"
    finally:
        for process, record in active.values():
            _stop_nat_msep_process(process)
            record.setdefault("ended_utc", datetime.now(timezone.utc).isoformat())
            record.setdefault("stopped_by_supervisor", status)
        active.clear()

    finished = datetime.now(timezone.utc)
    milestones.setdefault("work_finished_utc", finished.isoformat())
    missing_reason = {
        "wall_cap_reached": "formal_60h_resource_cap",
        "gpu_cap_reached": "formal_20h_gpu_cap",
        "repair_required": "unrecoverable_infrastructure_after_three_attempts",
    }.get(status, error or "not_evaluated")
    rows = _w259_position_rows(freeze, task_root, missing_reason=missing_reason)
    acceptance_finished = _parse_protocol_datetime(freeze["acceptance"]["finished_utc"])
    e0 = json.loads((task_root / "E0_TIMING.json").read_text("utf-8"))
    e0_finished = _parse_protocol_datetime(e0["finished_utc"])
    training_finished = _parse_protocol_datetime(
        milestones.get("training_finished_utc", finished.isoformat())
    )
    forced = (
        "W_INCOMPLETE_RESOURCE"
        if status in {"wall_cap_reached", "gpu_cap_reached"}
        else "W_INCOMPLETE_INFRASTRUCTURE"
        if status == "repair_required"
        else failure_terminal
    )
    result = publish_w259_prospective(
        output_root=output_root,
        freeze=freeze,
        position_rows=rows,
        pairs=W259_PAIRS,
        task_ids=tuple(str(value) for value in freeze["task_ids"]),
        phase_seconds={
            "D0_D3": max(0.0, (acceptance_finished - began).total_seconds()),
            "E0": max(0.0, (e0_finished - acceptance_finished).total_seconds()),
            "E1": max(0.0, (training_finished - e0_finished).total_seconds()),
            "E2": max(0.0, (finished - training_finished).total_seconds()),
            "E3": 0.0,
        },
        effective_concurrency=effective,
        repairs=repairs,
        gpu_active_seconds=gpu_seconds(finished),
        cloud_wall_seconds=max(0.0, (finished - began).total_seconds()),
        forced_terminal=forced,
    )
    status = "terminal"
    completed = _completed_w259_jobs(freeze, task_root)
    write_progress(datetime.now(timezone.utc))
    return result


def _load_w259_cf12_freeze(output_root: Path) -> dict[str, object]:
    value = json.loads((Path(output_root) / "EXECUTION_FREEZE.json").read_text("utf-8"))
    return (validate_g121_execution_freeze(value) if value.get("protocol_id")==G121_PROTOCOL_ID
            else validate_w259_cf12_execution_freeze(value))


def _w259_cf12_training_card(
    freeze: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    context = next(
        row for row in freeze["contexts"] if row["context_id"] == plan["context_id"]
    )
    card: dict[str, object] = {
        "schema": "q1-g121-run-card-v1" if freeze["protocol_id"]==G121_PROTOCOL_ID else "q1-w259-cf12-run-card-v1",
        "protocol_id": freeze["protocol_id"],
        "context_id": str(plan["context_id"]),
        "root": int(plan["root"]),
        "arm": str(plan["arm"]),
        "init_seed": int(context["init_seed"]),
        "updates": 100,
        "max_length": 2048,
        "plan_sha256": str(plan["plan_sha256"]),
        "response_tokens": int(plan["response_tokens"]),
        "expected_theta_i_sha256": str(context["theta_i_sha256"]),
        "initial_adapter_path": str(context["initial_adapter_path"]),
        "outcomes_opened": False,
    }
    if freeze["protocol_id"]==G121_PROTOCOL_ID:
        card.update(run_id=freeze["run_id"],pair_id=plan["context_id"])
    card["card_sha256"] = _canonical_json_sha256(card)
    return card


def execute_w259_cf12_worker(
    *, output_root: Path, task_root: Path, kind: str, job_id: str,
    score_workers: int = 1,
) -> dict[str, object]:
    """Execute one frozen CF12 train, generation, or score state."""

    freeze = _load_w259_cf12_freeze(output_root)
    legal = {(str(row["kind"]), str(row["job_id"])) for row in freeze["job_order"]}
    if (kind, job_id) not in legal:
        raise ValueError("Q1-W259-CF12 worker is outside the frozen queue")
    if datetime.now(timezone.utc) >= _parse_protocol_datetime(freeze["deadline_utc"]):
        raise RuntimeError("Q1-W259-CF12 66h deadline is closed")
    plan = next(
        row for row in freeze["plans"]
        if f"{row['context_id']}_{row['arm']}" == job_id
    )
    card = _w259_cf12_training_card(freeze, plan)
    task_root = Path(task_root)
    training_root = task_root / "training" / job_id
    if kind == "train":
        from ..train.run_codecontests import execute_nat_onoff_training

        return execute_nat_onoff_training(
            plan=load_s12_training_plan(Path(str(plan["plan_path"]))),
            card=card,
            model_path=Path(str(freeze["paths"]["model"])),
            output_dir=training_root,
            resume=(training_root / "recovery" / "CURRENT.json").is_file()
            or (training_root / "TERMINAL_TRAINING_RESULT.json").is_file(),
        )
    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    validate_nat_onoff_training_terminal(
        training_root / "TERMINAL_TRAINING_RESULT.json",
        expected_card_sha256=str(card["card_sha256"]),
        expected_plan_sha256=str(plan["plan_sha256"]),
        expected_protocol_id=str(freeze["protocol_id"]),
        expected_theta0_sha256=str(card["expected_theta_i_sha256"]),
    )
    from ..eval.humaneval import (
        execute_humaneval_formal_generation,
        execute_humaneval_formal_scoring,
        load_humaneval_plus_asset,
    )

    tasks = load_humaneval_plus_asset(Path(str(freeze["paths"]["asset"])))
    if [task["task_id"] for task in tasks] != freeze["task_ids"]:
        raise ValueError("Q1-W259-CF12 endpoint identity differs")
    if kind == "generation":
        return execute_humaneval_formal_generation(
            state_id=job_id, tasks=tasks,
            model_path=Path(str(freeze["paths"]["model"])),
            adapter_path=training_root / "adapter100",
            output_dir=task_root / "generation" / job_id,
            design_sha256=str(freeze["scientific_identity_sha256"]),
        )
    if score_workers != 1:
        raise ValueError("each CF12 score-state worker owns one global slot")
    return execute_humaneval_formal_scoring(
        state_id=job_id, tasks=tasks,
        generation_dir=task_root / "generation" / job_id,
        output_dir=task_root / "scores" / job_id,
        design_sha256=str(freeze["scientific_identity_sha256"]),
        evaluator_source_root=Path(str(freeze["paths"]["evaluator_source"])),
        evaluator_archive=Path(str(freeze["paths"]["evaluator_archive"])),
        sandbox_base=task_root / "sandboxes" / job_id,
        workers=1,
        python_executable=Path(str(freeze["runtime"]["python_judge"])),
        require_isolation=True,
    )


def _completed_w259_cf12_jobs(
    freeze: Mapping[str, object], task_root: Path
) -> set[tuple[str, str]]:
    from ..train.run_codecontests import validate_nat_onoff_training_terminal

    completed: set[tuple[str, str]] = set()
    for job in freeze["job_order"]:
        kind, job_id = str(job["kind"]), str(job["job_id"])
        if kind == "train":
            plan = next(
                row for row in freeze["plans"]
                if f"{row['context_id']}_{row['arm']}" == job_id
            )
            card = _w259_cf12_training_card(freeze, plan)
            terminal = task_root / "training" / job_id / "TERMINAL_TRAINING_RESULT.json"
            if terminal.is_file():
                validate_nat_onoff_training_terminal(
                    terminal,
                    expected_card_sha256=str(card["card_sha256"]),
                    expected_plan_sha256=str(plan["plan_sha256"]),
                    expected_protocol_id=str(freeze["protocol_id"]),
                    expected_theta0_sha256=str(card["expected_theta_i_sha256"]),
                )
                completed.add((kind, job_id))
            continue
        directory = "generation" if kind == "generation" else "scores"
        receipt = task_root / directory / job_id / "COMPLETE.json"
        if not receipt.is_file():
            continue
        value = json.loads(receipt.read_text("utf-8"))
        valid = (
            value.get("state_id") == job_id
            and value.get("completed_positions") == 164
            and value.get("design_sha256") == freeze["scientific_identity_sha256"]
        )
        if kind == "score":
            valid = valid and value.get("valid_positions") == 164 and value.get(
                "all_sandboxes_released"
            ) is True
        if valid:
            completed.add((kind, job_id))
    return completed


def _w259_cf12_position_rows(
    freeze: Mapping[str, object], task_root: Path, *, missing_reason: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in freeze["contexts"]:
        context_id, root = str(context["context_id"]), int(context["root"])
        for arm in (G121_ARMS if freeze["protocol_id"]==G121_PROTOCOL_ID else W259_CF12_ARMS):
            state = f"{context_id}_{arm}"
            for index, task_id in enumerate(freeze["task_ids"]):
                row: dict[str, object] = {
                    "state_id": state, "context_id": context_id, "root": root,
                    "arm": arm, "task_id": task_id, "Y": None,
                    "verdict": "not_evaluated", "completion_sha256": None,
                    "missing_reason": missing_reason,
                }
                if freeze["protocol_id"]==G121_PROTOCOL_ID:
                    row.update(pair_id=context_id,init_seed=context["init_seed"])
                score_path = task_root / "scores" / state / f"{index:03d}.json"
                if score_path.is_file():
                    score = json.loads(score_path.read_text("utf-8"))
                    if not (
                        score.get("state_id") == state
                        and score.get("task_id") == task_id
                        and score.get("task_index") == index
                        and score.get("design_sha256") == freeze["scientific_identity_sha256"]
                    ):
                        raise ValueError("Q1-W259-CF12 score binding differs")
                    row.update(
                        Y=score.get("Y"), verdict=score.get("verdict"),
                        completion_sha256=score.get("completion_sha256"),
                        missing_reason=None if score.get("Y") in (0, 1)
                        else str(score.get("detail") or "infrastructure_error"),
                    )
                rows.append(row)
    return rows


def run_g121_detached(*, task_root, output_root, authorization_path):
    """Task-owned guardian: keep the hard deadline alive if the worker supervisor fails."""
    import fcntl
    task_root,output_root=Path(task_root).resolve(),Path(output_root).resolve()
    task_root.mkdir(parents=True,exist_ok=True)
    with (task_root/"guardian.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze=prepare_g121_execution(task_root=task_root,output_root=output_root,authorization_path=authorization_path)
        if (output_root/"RESULT.json").is_file():
            return json.loads((output_root/"RESULT.json").read_text("utf-8"))
        # No new process may overlap a surviving child of the previous guardian.
        stale=_g121_owned_processes(task_root)
        if stale:
            raise RuntimeError(f"G121 surviving owned processes require cleanup: {stale}")
        command=[freeze["runtime"]["python_judge"],"-m","oasst1_filter.runtime.nat_spec",
                 "w259-cf12-run","--task-root",str(task_root),"--output-root",str(output_root)]
        log=task_root/"supervisor.log"
        with log.open("ab") as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=_parse_protocol_datetime(freeze["deadline_utc"])
        last_gpu=0.0
        while True:
            now=datetime.now(timezone.utc)
            progress_path=task_root/"progress.json"
            progress=json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
            # Stale progress cannot hide GPU usage after a supervisor crash.
            last_gpu=float(progress.get("gpu_active_seconds",last_gpu))
            if any(r.get("kind") in {"train","generation"} and not r.get("ended_utc") for r in progress.get("attempts",[])):
                last_gpu+=max(0.0,(now-_parse_protocol_datetime(progress.get("updated_utc",freeze["started_utc"]))).total_seconds())
            if (output_root/"RESULT.json").is_file() and (output_root/"DELIVERY.zip.sha256").is_file():
                process.wait(timeout=30)
                return json.loads((output_root/"RESULT.json").read_text("utf-8"))
            if now>=deadline-timedelta(seconds=120) or last_gpu>=G121_GPU_CAP_SECONDS-120:
                _stop_nat_msep_process(process)
                for pid in _g121_owned_processes(task_root):
                    try: os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                time.sleep(5)
                for pid in _g121_owned_processes(task_root):
                    try: os.killpg(pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                completed=_completed_w259_cf12_jobs(freeze,task_root)
                rows=_w259_cf12_position_rows(freeze,task_root,missing_reason="guardian_resource_stop")
                return publish_g121(output_root=output_root,freeze=freeze,position_rows=rows,
                    completed_training={s for k,s in completed if k=="train"},
                    repairs=progress.get("repairs",[])+["remote guardian resource cutoff"],
                    gpu_active_seconds=last_gpu,cloud_wall_seconds=(datetime.now(timezone.utc)-_parse_protocol_datetime(freeze["started_utc"])).total_seconds(),
                    forced_terminal="INCOMPLETE_RESOURCE_STOP")
            if process.poll() is not None:
                # Preserve evidence and upstream assets. No blind restart after an
                # unknown exception; an online engineer can repair/restart this
                # same guardian and clock, or the guard publishes at the cap.
                _atomic_json(task_root/"NEEDS_ENGINEERING_REPAIR.json",{
                    "supervisor_exit":process.returncode,"deadline_utc":freeze["deadline_utc"],
                    "message":"supervisor stopped; task budget continues; inspect supervisor.log"})
            time.sleep(10)


def _g121_owned_processes(task_root):
    """Resolve exact task worker/supervisor command tokens, never a broad pkill."""
    owned=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try: args=(entry/"cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError,ProcessLookupError,PermissionError,UnicodeError): continue
        if ("oasst1_filter.runtime.nat_spec" in args and "--task-root" in args
            and args[args.index("--task-root")+1]==str(task_root)
            and any(mode in args for mode in ("w259-cf12-worker","w259-cf12-run"))
            and "--authorization-path" not in args):
            owned.append(int(entry.name))
    return owned


def run_w259_cf12_formal(*, task_root: Path, output_root: Path) -> dict[str, object]:
    """Run six fixed CF12 waves under non-resetting wall/GPU caps."""

    task_root, output_root = Path(task_root), Path(output_root)
    result_path = output_root / "RESULT.json"
    if result_path.is_file():
        existing=json.loads(result_path.read_text("utf-8"))
        if existing.get("protocol_id")!=G121_PROTOCOL_ID or (output_root/"DELIVERY.zip.sha256").is_file():
            return existing
    freeze = _load_w259_cf12_freeze(output_root)
    is_g121=freeze["protocol_id"]==G121_PROTOCOL_ID
    states_fn=g121_state_ids if is_g121 else w259_cf12_state_ids
    total_states,wave_size,waves=(32,4,8) if is_g121 else (48,8,6)
    gpu_cap=G121_GPU_CAP_SECONDS if is_g121 else W259_CF12_GPU_CAP_SECONDS
    began = _parse_protocol_datetime(freeze["started_utc"])
    deadline = _parse_protocol_datetime(freeze["deadline_utc"])
    progress_path = task_root / "progress.json"
    previous = json.loads(progress_path.read_text("utf-8")) if progress_path.is_file() else {}
    effective = dict(previous.get("effective_concurrency", freeze["effective_concurrency"]))
    attempts: list[dict[str, object]] = list(previous.get("attempts", ()))
    repairs: list[str] = list(previous.get("repairs", ()))
    retry_counts: dict[str, int] = dict(previous.get("retry_counts", {}))
    completed = _completed_w259_cf12_jobs(freeze, task_root)
    all_jobs = {(str(job["kind"]), str(job["job_id"])) for job in freeze["job_order"]}
    active: dict[tuple[str, str], tuple[subprocess.Popen[bytes], dict[str, object]]] = {}
    acceptance_gpu = 0.0 if is_g121 else float(freeze["acceptance"].get("gpu_active_seconds", 0.0))
    status, error, forced_terminal = "running", None, None

    def gpu_seconds(now: datetime) -> float:
        intervals: list[tuple[datetime, datetime]] = []
        for record in attempts:
            if record.get("kind") not in {"train", "generation"}:
                continue
            start = _parse_protocol_datetime(record["started_utc"])
            end = _parse_protocol_datetime(record["ended_utc"]) if record.get("ended_utc") else now
            if end > start:
                intervals.append((start, end))
        merged: list[list[datetime]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return acceptance_gpu + sum((end - start).total_seconds() for start, end in merged)

    def observed_rates() -> tuple[list[float], list[float]]:
        states = states_fn(freeze["contexts"])
        train_rates, e2_rates = [], []
        for start in range(0, total_states, wave_size):
            wave = set(states[start : start + wave_size])
            if not all((kind, state) in completed for state in wave for kind in ("train", "generation", "score")):
                break
            train_records = [r for r in attempts if r.get("job_id") in wave and r.get("kind") == "train" and r.get("ended_utc")]
            e2_records = [r for r in attempts if r.get("job_id") in wave and r.get("kind") in {"generation", "score"} and r.get("ended_utc")]
            if train_records:
                train_rates.append((max(_parse_protocol_datetime(r["ended_utc"]) for r in train_records) - min(_parse_protocol_datetime(r["started_utc"]) for r in train_records)).total_seconds() / 3600 / wave_size)
            if e2_records:
                e2_rates.append((max(_parse_protocol_datetime(r["ended_utc"]) for r in e2_records) - min(_parse_protocol_datetime(r["started_utc"]) for r in e2_records)).total_seconds() / 3600 / wave_size)
        return train_rates, e2_rates

    def write_progress(now: datetime) -> None:
        states = states_fn(freeze["contexts"])
        scored = sum(kind == "score" for kind, _ in completed)
        wave = min(waves, scored // wave_size + (scored < total_states))
        elapsed = max(0.0, (now - began).total_seconds())
        remaining = total_states - scored
        eta = None if scored == 0 else elapsed / scored * remaining
        _atomic_json(progress_path, {
            "schema": "q1-g121-progress-v1" if is_g121 else "q1-w259-cf12-progress-v1", "protocol_id": freeze["protocol_id"],
            "scientific_identity_sha256": freeze["scientific_identity_sha256"],
            "status": status, "wave": wave,
            "stage": "training" if sum(kind == "train" for kind, _ in completed) < min(wave * wave_size, total_states) else "generation_scoring" if scored < total_states else "publication",
            "completed_training": sum(kind == "train" for kind, _ in completed),
            "completed_generation_states": sum(kind == "generation" for kind, _ in completed),
            "completed_score_states": scored,
            "missing_states": remaining,
            "generated_positions": sum(1 for _ in (task_root / "generation").glob("*/[0-9][0-9][0-9].json")),
            "scored_positions": sum(1 for _ in (task_root / "scores").glob("*/[0-9][0-9][0-9].json")),
            "active": [record for _process, record in active.values()],
            "effective_concurrency": effective,
            "started_utc": began.isoformat(), "deadline_utc": deadline.isoformat(),
            "updated_utc": now.isoformat(), "elapsed_seconds": elapsed,
            "remaining_wall_seconds": max(0.0, (deadline - now).total_seconds()),
            "gpu_active_seconds": gpu_seconds(now),
            "remaining_gpu_seconds": max(0.0, gpu_cap - gpu_seconds(now)),
            "eta_seconds": eta, "last_activity_utc": max((str(r.get("ended_utc") or r.get("started_utc")) for r in attempts), default=None),
            "attempts": attempts, "repairs": repairs, "retry_counts": retry_counts,
            "error": error, "supervisor_pid": os.getpid(), "outcomes_sealed": True,
        })

    try:
        while True:
            now = datetime.now(timezone.utc)
            wall_remaining = (deadline - now).total_seconds()
            gpu_remaining = gpu_cap - gpu_seconds(now)
            if wall_remaining <= 600 or gpu_remaining <= 600:
                status = "resource_wind_down"
                break
            for identity, (process, record) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del active[identity]
                record["ended_utc"] = now.isoformat()
                record["exit_code"] = returncode
                refreshed = _completed_w259_cf12_jobs(freeze, task_root) if returncode == 0 else set()
                if identity in refreshed:
                    completed |= refreshed
                    continue
                error = (Path(str(record["log"])).read_text("utf-8", errors="replace")[-8000:] if Path(str(record["log"])).is_file() else f"worker exit {returncode}")
                key = f"{identity[0]}:{identity[1]}"
                retry_counts[key] = retry_counts.get(key, 0) + 1
                if not is_g121 and "out of memory" in str(error).lower():
                    try:
                        effective = downshift_nat_msep_concurrency(effective, kind=identity[0])
                        repairs.append(f"resource-only {identity[0]} concurrency downshift after OOM")
                    except ValueError:
                        pass
                if (is_g121 and "out of memory" in str(error).lower()) or retry_counts[key] >= 3:
                    status = "paused_engineering_repair"
            if completed == all_jobs:
                status = "fixed_work_complete"
                break
            if status == "paused_engineering_repair":
                write_progress(now)
                time.sleep(30)
                continue
            states = states_fn(freeze["contexts"])
            closed = sum(
                all((kind, state) in completed for state in states[start : start + wave_size] for kind in (("train", "generation") if is_g121 else ("train", "generation", "score")))
                for start in range(0, total_states, wave_size)
            )
            current = set(states[closed * wave_size : closed * wave_size + wave_size]) if closed < waves else set()
            if current and not any(state in current for _kind, state in completed | set(active)):
                train_rates, e2_rates = observed_rates()
                admission = w259_cf12_budget_admission(
                    wall_elapsed_seconds=max(0.0, (now - began).total_seconds()),
                    gpu_active_seconds=gpu_seconds(now),
                    remaining_train_states=sum(("train", state) not in completed for state in states),
                    remaining_e2_states=sum(("score", state) not in completed for state in states),
                    completed_train_hours_per_state=train_rates,
                    completed_e2_hours_per_state=e2_rates,
                    concurrency=effective,
                )
                if is_g121:
                    admission=g121_budget_admission(
                        wall_elapsed_seconds=max(0.0,(now-began).total_seconds()),
                        gpu_active_seconds=gpu_seconds(now),
                        remaining_states=sum(("generation",state) not in completed for state in states),
                        observed_wave_seconds=[(a+b)*wave_size*3600 for a,b in zip(train_rates,e2_rates)])
                if not admission["admit"]:
                    status = "resource_admission_closed"
                    error = str(admission)
                    break
            chosen=(next_g121_jobs(freeze["contexts"],completed=completed,active=set(active)) if is_g121
                    else next_w259_cf12_jobs(freeze["contexts"],completed=completed,active=set(active),concurrency=effective))
            for job in chosen:
                kind, job_id = str(job["kind"]), str(job["job_id"])
                key = f"{kind}:{job_id}"
                attempt = retry_counts.get(key, 0) + 1
                log = task_root / "logs" / f"{kind}_{job_id}_attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(freeze["runtime"]["python_judge" if kind == "score" else "python_gpu"]),
                    "-m", "oasst1_filter.runtime.nat_spec", "w259-cf12-worker",
                    "--output-root", str(output_root), "--task-root", str(task_root),
                    "--kind", kind, "--job-id", job_id, "--score-workers", "1",
                ]
                with log.open("ab") as handle:
                    process = subprocess.Popen(
                        command, stdout=handle, stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"},
                    )
                record = {"kind": kind, "job_id": job_id, "pid": process.pid, "log": str(log), "started_utc": now.isoformat()}
                attempts.append(record)
                active[(kind, job_id)] = (process, record)
            status = "running"
            write_progress(now)
            time.sleep(2)
    except PermissionError as exc:
        status, error, forced_terminal = "blocked_safety_or_permission", f"{type(exc).__name__}: {exc}", "CF12_BLOCKED_SAFETY_OR_PERMISSION"
    except (ValueError, RuntimeError) as exc:
        status, error = "measurement_or_binding_failure", f"{type(exc).__name__}: {exc}"
        forced_terminal = "CF12_NEEDS_SCIENTIFIC_REVIEW" if "freeze" in str(exc).lower() else "CF12_INCOMPLETE_MEASUREMENT"
        if is_g121:
            raise
    except Exception as exc:
        status, error, forced_terminal = "unexpected_infrastructure_failure", f"{type(exc).__name__}: {exc}", "CF12_INCOMPLETE_INFRASTRUCTURE"
        if is_g121:
            raise
    finally:
        for process, record in active.values():
            _stop_nat_msep_process(process)
            record.setdefault("ended_utc", datetime.now(timezone.utc).isoformat())
            record.setdefault("stopped_by_supervisor", status)
        active.clear()
    finished = datetime.now(timezone.utc)
    if status in {"resource_wind_down", "resource_admission_closed"}:
        forced_terminal = "CF12_INCOMPLETE_RESOURCE"
    elif status == "paused_engineering_repair":
        forced_terminal = "CF12_INCOMPLETE_INFRASTRUCTURE"
    if is_g121:
        rows=_w259_cf12_position_rows(freeze,task_root,missing_reason=error or status)
        result=publish_g121(output_root=output_root,freeze=freeze,position_rows=rows,
            completed_training={state for kind,state in completed if kind=="train"},
            repairs=repairs+([str(error)] if error else []),
            gpu_active_seconds=gpu_seconds(finished),
            cloud_wall_seconds=max(0.0,(finished-began).total_seconds()),
            forced_terminal="INCOMPLETE_RESOURCE_STOP" if status in {"resource_wind_down","resource_admission_closed"} else "MEASUREMENT_INVALID" if forced_terminal else None)
        status="terminal"
        write_progress(datetime.now(timezone.utc))
        return result
    rows = _w259_cf12_position_rows(freeze, task_root, missing_reason=error or status)
    acceptance_finished = _parse_protocol_datetime(freeze["acceptance"]["finished_utc"])
    result = publish_w259_cf12(
        output_root=output_root, freeze=freeze, position_rows=rows,
        phase_seconds={"D0_D3": max(0.0, (acceptance_finished - began).total_seconds()), "E0_E3": max(0.0, (finished - acceptance_finished).total_seconds())},
        effective_concurrency=effective, repairs=repairs,
        gpu_active_seconds=gpu_seconds(finished),
        cloud_wall_seconds=max(0.0, (finished - began).total_seconds()),
        forced_terminal=forced_terminal,
    )
    status = "terminal"
    completed = _completed_w259_cf12_jobs(freeze, task_root)
    write_progress(datetime.now(timezone.utc))
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--prepared-root", type=Path, required=True)
    preflight.add_argument("--root", type=int)
    preflight.add_argument("--workers", type=int, default=8)
    run = subparsers.add_parser("run")
    run.add_argument("--prepared-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runtime-root", type=Path, required=True)
    run.add_argument("--workers", type=int, default=8)
    msep_preflight = subparsers.add_parser("msep-preflight")
    msep_preflight.add_argument("--prepared-root", type=Path, required=True)
    msep_preflight.add_argument("--runtime-root", type=Path)
    msep_preflight.add_argument("--workers", type=int, default=8)
    msep_run = subparsers.add_parser("msep-run")
    msep_run.add_argument("--prepared-root", type=Path, required=True)
    msep_run.add_argument("--output-root", type=Path, required=True)
    msep_run.add_argument("--runtime-root", type=Path, required=True)
    msep_run.add_argument("--workers", type=int, default=8)
    formal_prepare = subparsers.add_parser("formal-prepare")
    for name in (
        "workspace", "prepared-root", "design-path", "replica-freeze-path",
        "model-path", "asset-path", "evaluator-source-root", "evaluator-archive",
    ):
        formal_prepare.add_argument(f"--{name}", type=Path, required=True)
    formal_prepare.add_argument("--python-gpu", required=True)
    formal_prepare.add_argument("--python-judge", required=True)
    formal_dry = subparsers.add_parser("formal-dry-run")
    formal_dry.add_argument("--prepared-root", type=Path, required=True)
    formal_run = subparsers.add_parser("formal-run")
    formal_run.add_argument("--prepared-root", type=Path, required=True)
    formal_run.add_argument("--task-root", type=Path, required=True)
    formal_run.add_argument("--output-root", type=Path, required=True)
    formal_worker = subparsers.add_parser("formal-worker")
    formal_worker.add_argument("--output-root", type=Path, required=True)
    formal_worker.add_argument("--task-root", type=Path, required=True)
    formal_worker.add_argument("--kind", choices=("train", "generation", "score"), required=True)
    formal_worker.add_argument("--job-id", required=True)
    formal_worker.add_argument("--score-workers", type=int, choices=(1, 2, 4), default=4)
    onoff_prepare = subparsers.add_parser("onoff-prepare")
    for name in (
        "workspace", "task-root", "output-root", "background-plan-path",
        "prior-root-registry-path", "acceptance-path", "model-path",
        "asset-path", "evaluator-source-root", "evaluator-archive",
    ):
        onoff_prepare.add_argument(f"--{name}", type=Path, required=True)
    onoff_prepare.add_argument("--python-gpu", required=True)
    onoff_prepare.add_argument("--python-judge", required=True)
    onoff_run = subparsers.add_parser("onoff-run")
    onoff_run.add_argument("--task-root", type=Path, required=True)
    onoff_run.add_argument("--output-root", type=Path, required=True)
    onoff_worker = subparsers.add_parser("onoff-worker")
    onoff_worker.add_argument("--output-root", type=Path, required=True)
    onoff_worker.add_argument("--task-root", type=Path, required=True)
    onoff_worker.add_argument(
        "--kind", choices=("train", "generation", "score"), required=True
    )
    onoff_worker.add_argument("--job-id", required=True)
    onoff_worker.add_argument("--score-workers", type=int, choices=(1, 2, 4), default=4)
    e95_prepare = subparsers.add_parser("e95-delete-prepare")
    for name in (
        "task-root", "output-root", "old-output-root", "acceptance-path",
        "e95-block-table-path", "s12d-table-path", "model-path", "asset-path",
        "evaluator-source-root", "evaluator-archive",
    ):
        e95_prepare.add_argument(f"--{name}", type=Path, required=True)
    e95_prepare.add_argument("--python-gpu", required=True)
    e95_prepare.add_argument("--python-judge", required=True)
    e95_run = subparsers.add_parser("e95-delete-run")
    e95_run.add_argument("--task-root", type=Path, required=True)
    e95_run.add_argument("--output-root", type=Path, required=True)
    e95_worker = subparsers.add_parser("e95-delete-worker")
    e95_worker.add_argument("--output-root", type=Path, required=True)
    e95_worker.add_argument("--task-root", type=Path, required=True)
    e95_worker.add_argument(
        "--kind", choices=("train", "generation", "score"), required=True
    )
    e95_worker.add_argument("--job-id", required=True)
    e95_worker.add_argument("--score-workers", type=int, choices=(1,), default=1)
    w259_prepare = subparsers.add_parser("w259-prepare")
    for name in (
        "task-root", "output-root", "old-output-root", "acceptance-path",
        "pair-plan-path", "s12d-table-path", "model-path", "asset-path",
        "evaluator-source-root", "evaluator-archive",
    ):
        w259_prepare.add_argument(f"--{name}", type=Path, required=True)
    w259_prepare.add_argument("--python-gpu", required=True)
    w259_prepare.add_argument("--python-judge", required=True)
    w259_run = subparsers.add_parser("w259-run")
    w259_run.add_argument("--task-root", type=Path, required=True)
    w259_run.add_argument("--output-root", type=Path, required=True)
    w259_worker = subparsers.add_parser("w259-worker")
    w259_worker.add_argument("--output-root", type=Path, required=True)
    w259_worker.add_argument("--task-root", type=Path, required=True)
    w259_worker.add_argument(
        "--kind", choices=("train", "generation", "score"), required=True
    )
    w259_worker.add_argument("--job-id", required=True)
    w259_worker.add_argument("--score-workers", type=int, choices=(1,), default=1)
    cf12_prepare = subparsers.add_parser("w259-cf12-prepare")
    for name in (
        "task-root", "output-root", "design-root", "acceptance-path", "model-path",
        "asset-path", "evaluator-source-root", "evaluator-archive",
    ):
        cf12_prepare.add_argument(f"--{name}", type=Path, required=True)
    cf12_prepare.add_argument("--python-gpu", required=True)
    cf12_prepare.add_argument("--python-judge", required=True)
    cf12_run = subparsers.add_parser("w259-cf12-run")
    cf12_run.add_argument("--authorization-path", type=Path)
    cf12_run.add_argument("--task-root", type=Path, required=True)
    cf12_run.add_argument("--output-root", type=Path, required=True)
    cf12_worker = subparsers.add_parser("w259-cf12-worker")
    cf12_worker.add_argument("--output-root", type=Path, required=True)
    cf12_worker.add_argument("--task-root", type=Path, required=True)
    cf12_worker.add_argument(
        "--kind", choices=("train", "generation", "score"), required=True
    )
    cf12_worker.add_argument("--job-id", required=True)
    cf12_worker.add_argument("--score-workers", type=int, choices=(1,), default=1)
    arguments = parser.parse_args()
    if arguments.mode == "w259-cf12-prepare":
        value = prepare_w259_cf12_execution(
            task_root=arguments.task_root,
            output_root=arguments.output_root,
            design_root=arguments.design_root,
            acceptance_path=arguments.acceptance_path,
            model_path=arguments.model_path,
            asset_path=arguments.asset_path,
            evaluator_source_root=arguments.evaluator_source_root,
            evaluator_archive=arguments.evaluator_archive,
            python_gpu=arguments.python_gpu,
            python_judge=arguments.python_judge,
        )
        print(json.dumps({
            "scientific_identity_sha256": value["scientific_identity_sha256"],
            "contexts": [row["context_id"] for row in value["contexts"]],
        }, sort_keys=True))
        return 0
    if arguments.mode == "w259-cf12-worker":
        value = execute_w259_cf12_worker(
            output_root=arguments.output_root, task_root=arguments.task_root,
            kind=arguments.kind, job_id=arguments.job_id,
            score_workers=arguments.score_workers,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "w259-cf12-run":
        if arguments.authorization_path is not None:
            value=run_g121_detached(task_root=arguments.task_root,output_root=arguments.output_root,
                                    authorization_path=arguments.authorization_path)
            print(json.dumps(value,sort_keys=True))
            return 0
        value = run_w259_cf12_formal(
            task_root=arguments.task_root, output_root=arguments.output_root
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "w259-prepare":
        value = prepare_w259_execution(
            task_root=arguments.task_root,
            output_root=arguments.output_root,
            old_output_root=arguments.old_output_root,
            acceptance_path=arguments.acceptance_path,
            pair_plan_path=arguments.pair_plan_path,
            s12d_table_path=arguments.s12d_table_path,
            model_path=arguments.model_path,
            asset_path=arguments.asset_path,
            evaluator_source_root=arguments.evaluator_source_root,
            evaluator_archive=arguments.evaluator_archive,
            python_gpu=arguments.python_gpu,
            python_judge=arguments.python_judge,
        )
        print(json.dumps({
            "scientific_identity_sha256": value["scientific_identity_sha256"],
            "pairs": [row["pair_id"] for row in value["pairs"]],
        }, sort_keys=True))
        return 0
    if arguments.mode == "w259-worker":
        value = execute_w259_worker(
            output_root=arguments.output_root,
            task_root=arguments.task_root,
            kind=arguments.kind,
            job_id=arguments.job_id,
            score_workers=arguments.score_workers,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "w259-run":
        value = run_w259_formal(
            task_root=arguments.task_root, output_root=arguments.output_root
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "e95-delete-prepare":
        value = prepare_nat_e95_delete_execution(
            task_root=arguments.task_root,
            output_root=arguments.output_root,
            old_output_root=arguments.old_output_root,
            acceptance_path=arguments.acceptance_path,
            e95_block_table_path=arguments.e95_block_table_path,
            s12d_table_path=arguments.s12d_table_path,
            model_path=arguments.model_path,
            asset_path=arguments.asset_path,
            evaluator_source_root=arguments.evaluator_source_root,
            evaluator_archive=arguments.evaluator_archive,
            python_gpu=arguments.python_gpu,
            python_judge=arguments.python_judge,
        )
        print(json.dumps({
            "scientific_identity_sha256": value["scientific_identity_sha256"],
            "roots": value["roots"],
        }, sort_keys=True))
        return 0
    if arguments.mode == "e95-delete-worker":
        value = execute_nat_e95_delete_worker(
            output_root=arguments.output_root,
            task_root=arguments.task_root,
            kind=arguments.kind,
            job_id=arguments.job_id,
            score_workers=arguments.score_workers,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "e95-delete-run":
        value = run_nat_e95_delete_formal(
            task_root=arguments.task_root, output_root=arguments.output_root
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "onoff-prepare":
        value = prepare_nat_onoff_execution(
            workspace=arguments.workspace, task_root=arguments.task_root,
            output_root=arguments.output_root,
            background_plan_path=arguments.background_plan_path,
            prior_root_registry_path=arguments.prior_root_registry_path,
            acceptance_path=arguments.acceptance_path,
            model_path=arguments.model_path, asset_path=arguments.asset_path,
            evaluator_source_root=arguments.evaluator_source_root,
            evaluator_archive=arguments.evaluator_archive,
            python_gpu=arguments.python_gpu, python_judge=arguments.python_judge,
        )
        print(json.dumps({
            "scientific_identity_sha256": value["scientific_identity_sha256"],
            "roots": value["roots"],
        }, sort_keys=True))
        return 0
    if arguments.mode == "onoff-worker":
        value = execute_nat_onoff_worker(
            output_root=arguments.output_root, task_root=arguments.task_root,
            kind=arguments.kind, job_id=arguments.job_id,
            score_workers=arguments.score_workers,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "onoff-run":
        value = run_nat_onoff_formal(
            task_root=arguments.task_root, output_root=arguments.output_root
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "formal-prepare":
        value = prepare_nat_msep_formal(
            workspace=arguments.workspace,
            prepared_root=arguments.prepared_root,
            design_path=arguments.design_path,
            replica_freeze_path=arguments.replica_freeze_path,
            model_path=arguments.model_path,
            asset_path=arguments.asset_path,
            evaluator_source_root=arguments.evaluator_source_root,
            evaluator_archive=arguments.evaluator_archive,
            python_gpu=arguments.python_gpu,
            python_judge=arguments.python_judge,
        )
        print(json.dumps({"state": value["state"], "prior_root_registry_sha256": value["prior_root_registry_sha256"]}, sort_keys=True))
        return 0
    if arguments.mode == "formal-dry-run":
        print(json.dumps(dry_run_nat_msep_formal(arguments.prepared_root), sort_keys=True))
        return 0
    if arguments.mode == "formal-worker":
        value = execute_nat_msep_formal_worker(
            output_root=arguments.output_root, task_root=arguments.task_root,
            kind=arguments.kind, job_id=arguments.job_id,
            score_workers=arguments.score_workers,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "formal-run":
        value = run_nat_msep_formal(
            prepared_root=arguments.prepared_root,
            task_root=arguments.task_root,
            output_root=arguments.output_root,
        )
        print(json.dumps(value, sort_keys=True))
        return 0
    if arguments.mode == "msep-preflight":
        source = json.loads(
            (
                arguments.prepared_root
                / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
            ).read_text("utf-8")
        )
        roots = tuple(int(root) for root in source["roots"])
        result = run_nat_msep_preflight(
            arguments.prepared_root,
            roots=roots,
            max_workers=arguments.workers,
        )
        if arguments.runtime_root is not None:
            _atomic_json(arguments.runtime_root / "PREFLIGHT.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.mode == "msep-run":
        result = run_nat_msep_cpu_qualification(
            arguments.prepared_root,
            output_root=arguments.output_root,
            runtime_root=arguments.runtime_root,
            max_workers=arguments.workers,
        )
        print(
            json.dumps(
                {
                    "terminal": result["terminal"],
                    "reason_code": result["reason_code"],
                    "completed_updates": result["completed_update_count"],
                    "completed_roots": result["completed_root_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.mode == "preflight":
        freeze = json.loads(
            (
                arguments.prepared_root
                / "NAT_REPLICA_X164_EXECUTION_FREEZE.json"
            ).read_text("utf-8")
        )
        root = arguments.root or int(freeze["roots"][0])
        before = time.perf_counter()
        values = qualify_nat_spec_roots(
            arguments.prepared_root,
            roots=(root,),
            max_workers=arguments.workers,
        )
        identity = hashlib.sha256(repr(values).encode("utf-8")).hexdigest()
        print(
            json.dumps(
                {
                    "schema": "nat-spec-20-preflight-v1",
                    "root": root,
                    "updates": len(values[0].update_rows),
                    "requested_workers": arguments.workers,
                    "effective_workers": 1,
                    "canonical_identity_sha256": identity,
                    "wall_seconds": round(time.perf_counter() - before, 6),
                    "model_outcomes_read": False,
                    "gpu_processes_started": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_nat_spec_cpu_qualification(
        arguments.prepared_root,
        output_root=arguments.output_root,
        runtime_root=arguments.runtime_root,
        max_workers=arguments.workers,
    )
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "reason_code": result["reason_code"],
                "completed_updates": result["completed_update_count"],
                "completed_roots": result["completed_root_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def resolve_nat_spec_terminal(
    *,
    completed_updates: int,
    completed_roots: int,
    root_alphas: Sequence[Fraction],
    failure_reason: str | None = None,
) -> dict[str, object]:
    """Resolve only the two frozen scientific terminals using exact arithmetic."""

    if failure_reason is not None and failure_reason not in NAT_SPEC_FAILURE_REASONS:
        raise ValueError("NAT-SPEC failure reason code is not frozen")
    complete = completed_updates == 800 and completed_roots == 8
    if not complete:
        if failure_reason is None:
            raise ValueError("NAT-SPEC incomplete census requires a scientific reason code")
        return {
            "terminal": "NAT_SPEC_20_DESIGN_NOT_QUALIFIED",
            "reason_code": failure_reason,
            "alpha_bar": None,
            "alpha_gate": "not_evaluated",
        }
    if len(root_alphas) != 8:
        raise ValueError("NAT-SPEC complete census requires eight root alphas")
    alpha_bar = sum(root_alphas, Fraction(0, 1)) / 8
    if 2 * alpha_bar.numerator <= alpha_bar.denominator:
        terminal = "QUALIFIED_NAT_SPEC_20_CPU_CONTROL"
        reason = None
        gate = "pass"
    else:
        terminal = "NAT_SPEC_20_DESIGN_NOT_QUALIFIED"
        reason = "control_overlap_above_half"
        gate = "fail"
    if failure_reason not in {None, reason}:
        raise ValueError("NAT-SPEC complete census failure reason is inconsistent")
    return {
        "terminal": terminal,
        "reason_code": reason,
        "alpha_bar": alpha_bar,
        "alpha_gate": gate,
    }


def resolve_nat_msep_terminal(
    *,
    asset_ready: bool,
    completed_updates: int,
    completed_roots: int,
    root_rows: Sequence[dict[str, object]],
    deadline_exceeded: bool = False,
) -> dict[str, object]:
    """Resolve exactly one of the four frozen NAT-MSEP terminals."""

    if type(asset_ready) is not bool or type(deadline_exceeded) is not bool:
        raise TypeError("NAT-MSEP terminal flags must be boolean")
    if completed_updates < 0 or completed_roots < 0:
        raise ValueError("NAT-MSEP completion counts must be nonnegative")
    if completed_updates != 100 * completed_roots or completed_roots != len(root_rows):
        raise ValueError("NAT-MSEP publishes complete roots only")
    if not asset_ready:
        if completed_updates or completed_roots or root_rows:
            raise ValueError("NAT-MSEP asset terminal must contain zero rows")
        terminal = "NAT_MSEP_20_ASSET_CONTRACT_NOT_READY"
        reason = "asset_contract_not_ready"
        gates = {"id": "not_evaluated", "payload": "not_evaluated"}
    elif deadline_exceeded:
        if completed_roots > 7:
            raise ValueError("NAT-MSEP CPU-cap terminal may contain at most seven roots")
        terminal = "NAT_MSEP_20_CPU_CAP_UNRESOLVED"
        reason = "formal_cpu_deadline_reached"
        gates = {"id": "not_evaluated", "payload": "not_evaluated"}
    elif completed_updates != 800 or completed_roots != 8:
        raise ValueError("NAT-MSEP incomplete census requires deadline exhaustion")
    else:
        id_pass = all(row.get("id_gate") is True for row in root_rows)
        payload_pass = all(row.get("payload_gate") is True for row in root_rows)
        gates = {
            "id": "pass" if id_pass else "fail",
            "payload": "pass" if payload_pass else "fail",
        }
        if id_pass and payload_pass:
            terminal = "NAT_MSEP_20_CPU_CONTRAST_QUALIFIED_FOR_REVIEW"
            reason = None
        else:
            terminal = "NAT_MSEP_20_EXACT_SEPARATION_INSUFFICIENT"
            reason = "exact_separation_gate_failed"
    if terminal not in NAT_MSEP_TERMINALS:
        raise AssertionError("NAT-MSEP terminal escaped its frozen vocabulary")
    return {"terminal": terminal, "reason_code": reason, "gates": gates}


if __name__ == "__main__":
    raise SystemExit(main())
