"""Detached, checkpointed CPU owner for the frozen CodeContests V2 census."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Iterable, Mapping, Sequence
import unicodedata

from ..arms.codecontests import (
    CODECONTESTS_PROMPT_PREFIX,
    CODECONTESTS_PROMPT_SUFFIX,
    read_codecontests_manifest,
)
from ..data.provenance import sha256_file
from ..eval.codecontests_program import build_judged_payload, judge_codecontests_program
from ..census.codecontests import (
    CodeContestsS12DInputInvalid,
    adjudicate_codecontests_s12c_triplet,
    adjudicate_codecontests_s12d_triplet,
    align_codecontests_natural_support,
    adjudicate_codecontests_pair,
    freeze_codecontests_tests,
    generate_codecontests_mutants,
    ineligible_codecontests_pair,
    intersect_codecontests_mutations,
    select_first_killed_mutant,
    select_codecontests_f2,
    build_codecontests_v3_candidate_pool,
    derive_codecontests_s12d_membership,
    project_codecontests_s12d_primitive_row,
    classify_codecontests_s12d_problem,
)


V1_ENDPOINT_SHA256 = "33980db8b8131221e38a0b1294b707c0e0a9eac96e279a84afce5a375be90f36"
SCHEMA = "codecontests-v2-cpu-census-v1"
S12C_CSR_SCHEMA = "codecontests-s12c-csr-execution-v1"
_S12C_TOKENIZER = None


class S12InputInvalid(ValueError):
    """An authoritative S12 input cannot be reconstructed exactly."""


class S12InfrastructureInvalid(RuntimeError):
    """The frozen evaluator did not return a scientific verdict."""


class CodeContestsS12DWorkerFailure(RuntimeError):
    """A typed fatal S12d envelope prevents relation closure."""

    def __init__(self, envelope: Mapping[str, object]) -> None:
        self.envelope = dict(envelope)
        super().__init__(str(envelope.get("reason") or "S12d worker failed"))


_S12_BRIDGE_COUNTS = {
    "R": 354,
    "I": 354,
    "M": 350,
    "N": 230,
    "endpoint_reference": 285,
}


def validate_s12_fixed_object_bridge(
    *,
    objects: Sequence[Mapping[str, object]],
    endpoint_objects: Sequence[Mapping[str, object]],
    role_judge: Callable[[str, Sequence[Mapping[str, object]]], Mapping[str, object]],
    endpoint_judge: Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Check fixed S12 predicates without selecting or substituting objects."""

    expected = dict(_S12_BRIDGE_COUNTS if expected_counts is None else expected_counts)
    if set(expected) != set(_S12_BRIDGE_COUNTS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expected.values()
    ):
        raise ValueError("S12 fixed-object bridge expected counts are invalid")
    coverage = {key: 0 for key in expected}
    disagreement_ids: list[str] = []
    infrastructure_ids: list[str] = []
    missing_ids: list[str] = []
    seen: set[str] = set()
    observed_counts = Counter(str(value.get("role")) for value in objects)
    for role in ("R", "I", "M", "N"):
        difference = expected[role] - observed_counts.get(role, 0)
        if difference:
            missing_ids.extend([f"count:{role}"] * abs(difference))

    for value in objects:
        role = value.get("role")
        object_id = value.get("object_id")
        source = value.get("source")
        source_sha256 = value.get("source_sha256")
        tests = value.get("tests")
        if (
            role not in ("R", "I", "M", "N")
            or not isinstance(object_id, str)
            or not object_id
            or object_id in seen
            or not isinstance(source, str)
            or not source
            or not isinstance(source_sha256, str)
            or _fingerprint(source) != source_sha256
            or not isinstance(tests, Sequence)
            or isinstance(tests, (str, bytes))
            or not tests
            or any(not isinstance(test, Mapping) for test in tests)
        ):
            missing_ids.append(str(object_id or "unknown-role-object"))
            continue
        seen.add(object_id)
        if role == "M":
            try:
                compile(source, "<s12-fixed-object>", "exec")
            except SyntaxError:
                coverage[role] += 1
                disagreement_ids.append(object_id)
                continue
        try:
            verdict = role_judge(source, tests)
        except Exception:
            infrastructure_ids.append(object_id)
            continue
        category = verdict.get("category") if isinstance(verdict, Mapping) else None
        if category == "runner_infrastructure_error" or category not in {
            "pass", "wrong_answer", "compile_failure", "runtime_failure", "timeout"
        }:
            infrastructure_ids.append(object_id)
            continue
        coverage[role] += 1
        compatible = False
        if role in ("R", "N"):
            compatible = (
                category == "pass"
                and verdict.get("complete_contract") is True
                and verdict.get("test_executions") == len(tests)
            )
        elif role == "M":
            compatible = category in {"wrong_answer", "runtime_failure", "timeout"}
        else:
            canonical = tuple(
                f"{test.get('family')}:{test.get('ordinal')}" for test in tests
            )
            compact: dict[str, object] = {"category": category}
            if category != "compile_failure":
                verdicts = verdict.get("test_verdicts")
                first_nonpass = next(
                    (
                        row
                        for row in verdicts
                        if isinstance(row, Mapping) and row.get("category") != "pass"
                    ),
                    None,
                ) if isinstance(verdicts, Sequence) else None
                if isinstance(first_nonpass, Mapping):
                    compact["first_nonpass_witness"] = {
                        "test_id": f"{first_nonpass.get('family')}:{first_nonpass.get('ordinal')}"
                    }
            try:
                classified = classify_s12_failure_family(
                    compact, canonical_panel=canonical
                )
            except S12InfrastructureInvalid:
                infrastructure_ids.append(object_id)
                coverage[role] -= 1
                continue
            except S12InputInvalid:
                classified = {}
            compatible = (
                classified.get("failure_family") == value.get("expected_failure_family")
                and classified.get("witness_test_id") == value.get("expected_witness_test_id")
            )
        if not compatible:
            disagreement_ids.append(object_id)

    endpoint_difference = expected["endpoint_reference"] - len(endpoint_objects)
    if endpoint_difference:
        missing_ids.extend(["count:endpoint_reference"] * abs(endpoint_difference))
    for value in endpoint_objects:
        object_id = value.get("object_id")
        member = value.get("member")
        row = value.get("row")
        if (
            not isinstance(object_id, str)
            or not object_id
            or object_id in seen
            or not isinstance(member, Mapping)
            or not isinstance(row, Mapping)
            or member.get("task_id") != object_id
            or row.get("task_id") != object_id
        ):
            missing_ids.append(str(object_id or "unknown-endpoint-object"))
            continue
        seen.add(object_id)
        try:
            verdict = endpoint_judge(member, row)
        except Exception:
            infrastructure_ids.append(object_id)
            continue
        category = verdict.get("category") if isinstance(verdict, Mapping) else None
        if category == "sandbox_infrastructure_error" or category == "runner_infrastructure_error" or category is None:
            infrastructure_ids.append(object_id)
            continue
        coverage["endpoint_reference"] += 1
        if category != "pass" or verdict.get("all_tests_pass") is not True:
            disagreement_ids.append(object_id)

    passed = (
        coverage == expected
        and not disagreement_ids
        and not infrastructure_ids
        and not missing_ids
    )
    return {
        "schema": "codecontests-s12b-fixed-object-bridge-v1",
        "protocol_id": "codecontests-s12-apcf-v1",
        "passed": passed,
        "coverage": coverage,
        "expected_counts": expected,
        "disagreement_count": len(disagreement_ids),
        "infrastructure_error_count": len(infrastructure_ids),
        "missing_input_count": len(missing_ids),
        "disagreement_object_ids": disagreement_ids[:10],
        "infrastructure_object_ids": infrastructure_ids[:10],
        "missing_object_ids": missing_ids[:10],
        "selection_performed": False,
    }


def classify_s12_failure_family(
    verdict: Mapping[str, object], *, canonical_panel: Sequence[str]
) -> dict[str, str]:
    """Map the canonical first V3b failure to the closed S12 block family."""

    if not canonical_panel or any(not isinstance(item, str) or not item for item in canonical_panel):
        raise S12InputInvalid("S12 canonical panel is missing")
    category = verdict.get("category")
    if category == "runner_infrastructure_error":
        raise S12InfrastructureInvalid("S12 replay returned runner infrastructure error")
    family = {
        "wrong_answer": "output_mismatch",
        "compile_failure": "execution_failure",
        "runtime_failure": "execution_failure",
        "timeout": "timeout",
    }.get(category)
    if family is None:
        raise S12InputInvalid("S12 replay category is not a frozen failure")
    if category == "compile_failure":
        witness = canonical_panel[0]
    else:
        compact = verdict.get("first_nonpass_witness")
        witness = compact.get("test_id") if isinstance(compact, Mapping) else None
        if not isinstance(witness, str) or witness not in canonical_panel:
            raise S12InputInvalid("S12 canonical first-nonpass witness is missing")
    return {"failure_family": family, "witness_test_id": witness}


def build_s12_mutation_panel(
    tests: Sequence[Mapping[str, object]],
    witness: Mapping[str, object] | None,
    *,
    compile_failure: bool = False,
) -> tuple[Mapping[str, object], ...]:
    """Freeze witness-first, canonical-order S12 mutation tests, capped at eight."""

    if not tests:
        raise S12InputInvalid("S12 canonical panel is missing")
    head = tests[0] if compile_failure else None
    if not compile_failure:
        if not isinstance(witness, Mapping):
            raise S12InputInvalid("S12 canonical first-nonpass witness is missing")
        identity = (witness.get("family"), witness.get("ordinal"))
        head = next(
            (
                test
                for test in tests
                if (test.get("family"), test.get("ordinal")) == identity
            ),
            None,
        )
        if head is None:
            raise S12InputInvalid("S12 witness is not in the canonical panel")
    assert head is not None
    head_identity = (head.get("family"), head.get("ordinal"))
    ordered = [head] + [
        test
        for test in tests
        if (test.get("family"), test.get("ordinal")) != head_identity
    ]
    return tuple(ordered[: min(8, len(tests))])


def reconstruct_s12_nat_membership(
    pair_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Select exactly the immutable V3b S1 certificates without reclassification."""

    selected: list[dict[str, object]] = []
    for row in pair_rows:
        if str(row.get("s1_i_discrepant")).lower() != "true":
            continue
        disposition = row.get("s1_disposition")
        provenance = {
            "original_r_recovered_certificate": "original",
            "alternative_r_recovered_certificate": "alternative",
        }.get(disposition)
        if provenance is None:
            raise S12InputInvalid("S12 certificate disposition is invalid")
        expected_original = str(row.get("s1_selected_r_is_original")).lower() == "true"
        if expected_original != (provenance == "original"):
            raise S12InputInvalid("S12 selected-R provenance is inconsistent")
        pair_id = row.get("triplet_id")
        problem_id = row.get("candidate_problem_id")
        fingerprint = row.get("s1_selected_r_fingerprint")
        if (
            not isinstance(pair_id, str)
            or not isinstance(problem_id, str)
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(value not in "0123456789abcdef" for value in fingerprint)
        ):
            raise S12InputInvalid("S12 immutable pair identity is malformed")
        selected.append(
            {
                "pair_id": pair_id,
                "problem_id": problem_id,
                "selected_r_fingerprint": fingerprint,
                "r_provenance": provenance,
            }
        )
    if len(selected) != 354:
        raise S12InputInvalid("S12 S_nat must contain exactly 354 pairs")
    if len({row["pair_id"] for row in selected}) != 354 or len(
        {row["problem_id"] for row in selected}
    ) != 354:
        raise S12InputInvalid("S12 S_nat pair/problem identities must be unique")
    provenance_counts = Counter(row["r_provenance"] for row in selected)
    if provenance_counts != Counter({"original": 281, "alternative": 73}):
        raise S12InputInvalid("S12 S_nat provenance counts differ from V3b")
    return tuple(selected)


def reconstruct_s12_selected_pair(
    member: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    row: Mapping[str, object],
    measure_tokens: Callable[[str], Mapping[str, object]],
) -> dict[str, object]:
    """Recover one selected V3b R and its fixed I from the frozen snapshot."""

    if (
        member.get("pair_id") != manifest.get("triplet_id")
        or member.get("problem_id") != manifest.get("candidate_problem_id")
    ):
        raise S12InputInvalid("S12 pair/manifest identity differs")
    pool = build_codecontests_v3_candidate_pool(
        manifest, row, measure_tokens=measure_tokens
    )
    selected = next(
        (
            candidate
            for candidate in pool["candidates"]
            if candidate.get("fingerprint") == member.get("selected_r_fingerprint")
        ),
        None,
    )
    if not isinstance(selected, Mapping) or selected.get("selectable") is not True:
        raise S12InputInvalid("S12 selected R is absent from the frozen candidate pool")
    expected_original = member.get("r_provenance") == "original"
    if selected.get("original") is not expected_original:
        raise S12InputInvalid("S12 selected R provenance differs from V3b")
    fixed_i = pool.get("fixed_i")
    if not isinstance(fixed_i, Mapping):
        raise S12InputInvalid("S12 fixed I is absent from the frozen candidate pool")
    return {
        "pair_id": member["pair_id"],
        "problem_id": member["problem_id"],
        "r_provenance": member["r_provenance"],
        "R": dict(selected),
        "I": dict(fixed_i),
    }


def reconstruct_s12_nat_from_files(
    *,
    pair_table_path: Path,
    v3b_result_path: Path,
    v3b_freeze_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
) -> dict[str, object]:
    """Run the outcome-free exact 354-identity local S12 reconstruction."""

    authorities = (
        (
            pair_table_path,
            "892124bdc3015330a70bdae9da69c88c70e7daf8f1f581d7f053642213594ba2",
        ),
        (
            v3b_result_path,
            "0292b1f83a9550339499dde162c4228e78fd3d4f43d3f61d567d6d675425b4e7",
        ),
        (
            v3b_freeze_path,
            "bffb6ebb76cb91d9d428d7e0ffe8fcb7270da47e49de1d7930285e289718f3ff",
        ),
    )
    for path, expected in authorities:
        if not path.is_file() or sha256_file(path) != expected:
            raise S12InputInvalid(f"S12 authority hash mismatch: {path.name}")
    if dataset_root.name != "802411c3010cb00d1b05bad57ca77365a3c699d6":
        raise S12InputInvalid("S12 CodeContests revision path differs")
    if model_path.name != "8faed761d45a263340a0528343f099c05c9a4323":
        raise S12InputInvalid("S12 tokenizer revision path differs")
    result = json.loads(v3b_result_path.read_text("utf-8"))
    freeze = json.loads(v3b_freeze_path.read_text("utf-8"))
    device_id = "5a542fd7b297ac5b5fc1567684670e8d799d01ec06a98834e0b5212d3a0ece78"
    if (
        result.get("measurement_device_id") != device_id
        or freeze.get("measurement_device", {}).get("measurement_device_id") != device_id
    ):
        raise S12InputInvalid("S12 V3b measurement-device identity differs")
    with pair_table_path.open("r", encoding="utf-8", newline="") as handle:
        members = reconstruct_s12_nat_membership(tuple(csv.DictReader(handle)))
    manifest = read_codecontests_manifest(manifest_path)
    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    rows = load_census_rows(dataset_root, manifest)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )

    def measurement(problem_row: Mapping[str, object], source: str) -> dict[str, int]:
        description = problem_row.get("description")
        if not isinstance(description, str):
            raise S12InputInvalid("S12 problem description is malformed")
        prompt = (
            CODECONTESTS_PROMPT_PREFIX
            + description.replace("\r\n", "\n").replace("\r", "\n")
            + CODECONTESTS_PROMPT_SUFFIX
        )
        prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
        response_ids = tuple(tokenizer.encode(source, add_special_tokens=False)) + (
            int(tokenizer.eos_token_id),
        )
        return {
            "serialized_tokens": len(prompt_ids) + len(response_ids),
            "response_tokens": len(response_ids),
        }

    compact: list[dict[str, object]] = []
    for member in members:
        triplet = by_triplet[str(member["pair_id"])]
        candidate_row = rows[_index(triplet["candidate_problem_id"])]
        reconstructed = reconstruct_s12_selected_pair(
            member,
            manifest=triplet,
            row=candidate_row,
            measure_tokens=lambda source, row=candidate_row: measurement(row, source),
        )
        neutral_row = rows[_index(triplet["neutral_problem_id"])]
        neutral_source = _solution(
            neutral_row,
            triplet["neutral_solution_id"],
            label="correct",
            fingerprint=triplet["neutral_solution_fingerprint"],
        )
        neutral_measurement = measurement(neutral_row, neutral_source)
        if (
            neutral_measurement["response_tokens"] != triplet["neutral_response_tokens"]
            or neutral_measurement["serialized_tokens"] > 2_048
        ):
            raise S12InputInvalid("S12 aligned N serialization differs from freeze")
        compact.append(
            {
                "pair_id": member["pair_id"],
                "problem_id": member["problem_id"],
                "r_provenance": member["r_provenance"],
                "selected_r_fingerprint": reconstructed["R"]["fingerprint"],
                "i_fingerprint": reconstructed["I"]["fingerprint"],
                "n_fingerprint": triplet["neutral_solution_fingerprint"],
                "response_tokens": {
                    "I": reconstructed["I"]["response_tokens"],
                    "R": reconstructed["R"]["response_tokens"],
                    "N": neutral_measurement["response_tokens"],
                },
                "serialized_tokens": {
                    "I": reconstructed["I"]["serialized_tokens"],
                    "R": reconstructed["R"]["serialized_tokens"],
                    "N": neutral_measurement["serialized_tokens"],
                },
            }
        )
    identities_sha256 = hashlib.sha256(
        _json_bytes(
            [
                [
                    row["pair_id"],
                    row["selected_r_fingerprint"],
                    row["i_fingerprint"],
                    row["n_fingerprint"],
                ]
                for row in compact
            ]
        )
    ).hexdigest()
    return {
        "schema": "codecontests-s12-local-nat-reconstruction-v1",
        "protocol_id": "codecontests-s12-apcf-v1",
        "measurement_device_id": device_id,
        "pair_count": len(compact),
        "original_r_count": sum(row["r_provenance"] == "original" for row in compact),
        "alternative_r_count": sum(
            row["r_provenance"] == "alternative" for row in compact
        ),
        "identity_sha256": identities_sha256,
        "records": compact,
        "raw_program_persisted": False,
        "formal_outcomes_opened": False,
    }


def reconstruct_s12_fixed_bridge_objects(
    *,
    pair_table_path: Path,
    v3b_result_path: Path,
    v3b_freeze_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
    predecessor_preoutcome_dir: Path,
) -> tuple[dict[str, object], ...]:
    """Recover only the fixed S12 bridge objects from frozen identity locators."""

    compact = reconstruct_s12_nat_from_files(
        pair_table_path=pair_table_path,
        v3b_result_path=v3b_result_path,
        v3b_freeze_path=v3b_freeze_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        model_path=model_path,
    )
    with pair_table_path.open("r", encoding="utf-8", newline="") as handle:
        members = reconstruct_s12_nat_membership(tuple(csv.DictReader(handle)))
    manifest = read_codecontests_manifest(manifest_path)
    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    rows = load_census_rows(dataset_root, manifest)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )

    def measurement(problem_row: Mapping[str, object], source: str) -> dict[str, int]:
        description = problem_row.get("description")
        if not isinstance(description, str):
            raise S12InputInvalid("S12 bridge problem description is malformed")
        prompt = (
            CODECONTESTS_PROMPT_PREFIX
            + description.replace("\r\n", "\n").replace("\r", "\n")
            + CODECONTESTS_PROMPT_SUFFIX
        )
        prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
        response_ids = tuple(tokenizer.encode(source, add_special_tokens=False)) + (
            int(tokenizer.eos_token_id),
        )
        return {
            "serialized_tokens": len(prompt_ids) + len(response_ids),
            "response_tokens": len(response_ids),
        }

    work_rows: dict[str, Mapping[str, object]] = {}
    paths = tuple(sorted(predecessor_preoutcome_dir.glob("*.json")))
    if len(paths) != 354:
        raise S12InputInvalid("S12 bridge requires exactly 354 pre-outcome records")
    for path in paths:
        value = json.loads(path.read_text("utf-8"))
        pair_id = value.get("pair_id") if isinstance(value, Mapping) else None
        if not isinstance(pair_id, str) or pair_id in work_rows:
            raise S12InputInvalid("S12 bridge pre-outcome identity is invalid")
        work_rows[pair_id] = value
    if set(work_rows) != {str(member["pair_id"]) for member in members}:
        raise S12InputInvalid("S12 bridge pre-outcome membership differs")

    compact_by_pair = {str(row["pair_id"]): row for row in compact["records"]}
    objects: list[dict[str, object]] = []
    for member in members:
        pair_id = str(member["pair_id"])
        triplet = by_triplet[pair_id]
        candidate_row = rows[_index(triplet["candidate_problem_id"])]
        neutral_row = rows[_index(triplet["neutral_problem_id"])]
        selected = reconstruct_s12_selected_pair(
            member,
            manifest=triplet,
            row=candidate_row,
            measure_tokens=lambda source, row=candidate_row: measurement(row, source),
        )
        record = compact_by_pair[pair_id]
        if (
            selected["R"]["fingerprint"] != record["selected_r_fingerprint"]
            or selected["I"]["fingerprint"] != record["i_fingerprint"]
        ):
            raise S12InputInvalid("S12 bridge selected source identity differs")
        candidate_tests = freeze_codecontests_tests(
            str(triplet["candidate_problem_fingerprint"]), candidate_row
        )
        work = work_rows[pair_id]
        witness_id = work.get("witness_test_id")
        canonical_ids = tuple(
            f"{test['family']}:{test['ordinal']}" for test in candidate_tests
        )
        if (
            work.get("failure_family") not in {"output_mismatch", "execution_failure", "timeout"}
            or not isinstance(witness_id, str)
            or witness_id not in canonical_ids
        ):
            raise S12InputInvalid("S12 bridge fixed I predicate is incomplete")
        r_source = str(selected["R"]["source"])
        i_source = str(selected["I"]["source"])
        objects.extend(
            (
                {
                    "object_id": f"{pair_id}:R",
                    "role": "R",
                    "source": r_source,
                    "source_sha256": str(selected["R"]["fingerprint"]),
                    "tests": candidate_tests,
                },
                {
                    "object_id": f"{pair_id}:I",
                    "role": "I",
                    "source": i_source,
                    "source_sha256": str(selected["I"]["fingerprint"]),
                    "tests": candidate_tests,
                    "expected_failure_family": str(work["failure_family"]),
                    "expected_witness_test_id": witness_id,
                },
            )
        )
        mutation = work.get("mutation")
        if isinstance(mutation, Mapping) and mutation.get("available") is True:
            mutant_source = _mutated_source(r_source, mutation)
            witness_position = canonical_ids.index(witness_id)
            hidden_panel = (candidate_tests[witness_position],) + tuple(
                test for index, test in enumerate(candidate_tests) if index != witness_position
            )
            objects.append(
                {
                    "object_id": f"{pair_id}:M",
                    "role": "M",
                    "source": mutant_source,
                    "source_sha256": str(mutation["source_sha256"]),
                    "tests": hidden_panel[: min(8, len(candidate_tests))],
                }
            )
        if work.get("n_qualified") is True:
            n_source = _solution(
                neutral_row,
                triplet["neutral_solution_id"],
                label="correct",
                fingerprint=triplet["neutral_solution_fingerprint"],
            )
            neutral_tests = freeze_codecontests_tests(
                str(triplet["neutral_problem_fingerprint"]), neutral_row
            )
            objects.append(
                {
                    "object_id": f"{pair_id}:N",
                    "role": "N",
                    "source": n_source,
                    "source_sha256": str(triplet["neutral_solution_fingerprint"]),
                    "tests": neutral_tests,
                }
            )
    counts = Counter(str(value["role"]) for value in objects)
    if counts != Counter({"R": 354, "I": 354, "M": 350, "N": 230}):
        raise S12InputInvalid("S12 bridge fixed role counts differ")
    return tuple(objects)


def _s12_fixed_bridge_role_worker(payload: Mapping[str, object]) -> dict[str, object]:
    """Judge one fixed role in an independent umask-0022 CPU worker."""

    os.umask(0o022)
    sandbox_parent = Path(str(payload["sandbox_parent"]))
    sandbox_base = sandbox_parent / f"worker_{os.getpid()}"
    sandbox_base.mkdir(parents=True, exist_ok=True)
    verdict = judge_codecontests_program(
        str(payload["source"]),
        payload["tests"],
        sandbox_base=sandbox_base,
        stop_on_first_failure=payload["role"] in {"I", "M"},
    )
    return {"object_id": payload["object_id"], "verdict": verdict}


def run_s12_fixed_object_bridge(
    *,
    pair_table_path: Path,
    v3b_result_path: Path,
    v3b_freeze_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
    predecessor_preoutcome_dir: Path,
    endpoint_path: Path,
    evalplus_asset_path: Path,
    output_dir: Path,
    workers: int = 4,
) -> dict[str, object]:
    """Execute one fixed-object bridge and persist only compact verdict metadata."""

    if workers != 4:
        raise ValueError("S12 fixed-object bridge concurrency is fixed at four")
    if output_dir.exists():
        raise FileExistsError("S12 fixed-object bridge is one-shot")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    objects = reconstruct_s12_fixed_bridge_objects(
        pair_table_path=pair_table_path,
        v3b_result_path=v3b_result_path,
        v3b_freeze_path=v3b_freeze_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        model_path=model_path,
        predecessor_preoutcome_dir=predecessor_preoutcome_dir,
    )
    sandbox_parent = output_dir / "_role_sandboxes"
    sandbox_parent.mkdir()
    verdicts: dict[str, Mapping[str, object]] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _s12_fixed_bridge_role_worker,
                    {**value, "sandbox_parent": str(sandbox_parent)},
                ): str(value["object_id"])
                for value in objects
            }
            for future in as_completed(futures):
                object_id = futures[future]
                try:
                    returned = future.result()
                    verdicts[object_id] = returned["verdict"]
                except Exception:
                    verdicts[object_id] = {
                        "category": "runner_infrastructure_error",
                        "complete_contract": False,
                        "test_executions": 0,
                        "test_verdicts": (),
                    }
    finally:
        shutil.rmtree(sandbox_parent, ignore_errors=True)

    from ..eval.run_codecontests import (
        execute_codecontests_reference_bridge,
        load_codecontests_endpoint_member_rows,
    )

    endpoint = json.loads(endpoint_path.read_text("utf-8"))
    members = endpoint.get("members")
    if (
        endpoint.get("schema") != "codecontests-v1-sealed-eligibility-v1"
        or endpoint.get("eligible_count") != 285
        or not isinstance(members, list)
        or len(members) != 285
    ):
        raise S12InputInvalid("S12 bridge endpoint authority is invalid")
    rows_by_id = load_codecontests_endpoint_member_rows(evalplus_asset_path, members)
    endpoint_result = execute_codecontests_reference_bridge(
        members=members,
        rows_by_id=rows_by_id,
        output_dir=output_dir / "endpoint_reference",
    )
    endpoint_objects = tuple(
        {"object_id": member["task_id"], "member": member, "row": rows_by_id[member["task_id"]]}
        for member in members
    )
    role_sequence = iter(verdicts[str(value["object_id"])] for value in objects)
    endpoint_by_id = {row["task_id"]: row for row in endpoint_result["tasks"]}
    result = validate_s12_fixed_object_bridge(
        objects=objects,
        endpoint_objects=endpoint_objects,
        role_judge=lambda _source, _tests: next(role_sequence),
        endpoint_judge=lambda member, _row: endpoint_by_id[str(member["task_id"])],
    )
    result.update(
        wall_seconds=time.perf_counter() - started,
        role_object_count=len(objects),
        endpoint_result_sha256=sha256_file(
            output_dir / "endpoint_reference/REFERENCE_BRIDGE_RESULT.json"
        ),
    )
    _atomic_json(output_dir / "BRIDGE_RESULT.json", result)
    return result


def _s12_preoutcome_worker(payload: Mapping[str, object]) -> dict[str, object]:
    """Replay I/N and exhaust one selected-R finite mutant work item."""

    sandbox_base = Path(str(payload["sandbox_parent"])) / f"worker_{os.getpid()}"
    sandbox_base.mkdir(parents=True, exist_ok=True)
    triplet = payload["triplet"]
    row = payload["candidate_row"]
    tests = freeze_codecontests_tests(str(triplet["candidate_problem_fingerprint"]), row)
    verdict = judge_codecontests_program(
        str(payload["i_source"]),
        tests,
        sandbox_base=sandbox_base,
        stop_on_first_failure=True,
    )
    category = verdict.get("category")
    if category == "runner_infrastructure_error":
        raise S12InfrastructureInvalid("S12 I replay returned runner infrastructure error")
    compact_verdict: dict[str, object] = {"category": category}
    witness = None
    if category != "compile_failure":
        values = verdict.get("test_verdicts")
        witness = values[0] if isinstance(values, list) and values else None
        if isinstance(witness, Mapping):
            compact_verdict["first_nonpass_witness"] = {
                "test_id": f"{witness.get('family')}:{witness.get('ordinal')}"
            }
    canonical = tuple(f"{test['family']}:{test['ordinal']}" for test in tests)
    family = classify_s12_failure_family(compact_verdict, canonical_panel=canonical)
    mutation_panel = build_s12_mutation_panel(
        tests, witness, compile_failure=category == "compile_failure"
    )
    mutation = select_first_killed_mutant(
        str(payload["r_source"]),
        mutation_panel,
        judge=lambda source, panel: judge_codecontests_program(
            str(source),
            panel,
            sandbox_base=sandbox_base,
            stop_on_first_failure=True,
        ),
    )
    neutral_tests = freeze_codecontests_tests(
        str(triplet["neutral_problem_fingerprint"]), payload["neutral_row"]
    )
    neutral = judge_codecontests_program(
        str(payload["n_source"]), neutral_tests, sandbox_base=sandbox_base
    )
    if neutral.get("category") == "runner_infrastructure_error":
        raise S12InfrastructureInvalid("S12 N replay returned runner infrastructure error")
    return {
        "pair_id": payload["pair_id"],
        "failure_family": family["failure_family"],
        "witness_test_id": family["witness_test_id"],
        "n_qualified": neutral.get("category") == "pass"
        and neutral.get("test_executions") == len(neutral_tests),
        "mutation": {key: value for key, value in mutation.items() if key != "source"},
    }


def materialize_s12_preoutcome_closure(
    *,
    pair_table_path: Path,
    v3b_result_path: Path,
    v3b_freeze_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
    checkpoint_dir: Path,
    workers: int,
) -> dict[str, object]:
    """Close S12 membership, roles, policy qualification, and finite PC mutants."""

    from transformers import AutoTokenizer

    from ..arms.paired_filtering import build_s12_stage_plans, load_s12_config

    if workers <= 0:
        raise ValueError("S12 pre-outcome worker count must be positive")
    compact = reconstruct_s12_nat_from_files(
        pair_table_path=pair_table_path,
        v3b_result_path=v3b_result_path,
        v3b_freeze_path=v3b_freeze_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        model_path=model_path,
    )
    with pair_table_path.open("r", encoding="utf-8", newline="") as handle:
        members = reconstruct_s12_nat_membership(tuple(csv.DictReader(handle)))
    manifest = read_codecontests_manifest(manifest_path)
    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    rows = load_census_rows(dataset_root, manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )

    def tokenized(triplet, row, source: str, arm: str, prompt_role: str):
        return _tokenized_source_unit(
            triplet=triplet,
            row=row,
            source=source,
            arm=arm,
            prompt_role=prompt_role,
            tokenizer=tokenizer,
        )

    source_rows: dict[str, dict[str, object]] = {}
    payloads = []
    sandbox_parent = checkpoint_dir / "_sandbox_workers"
    for member in members:
        triplet = by_triplet[str(member["pair_id"])]
        candidate_row = rows[_index(triplet["candidate_problem_id"])]
        neutral_row = rows[_index(triplet["neutral_problem_id"])]

        def measure(source: str, row=candidate_row) -> dict[str, int]:
            description = row.get("description")
            if not isinstance(description, str):
                raise S12InputInvalid("S12 problem description is malformed")
            prompt = (
                CODECONTESTS_PROMPT_PREFIX
                + description.replace("\r\n", "\n").replace("\r", "\n")
                + CODECONTESTS_PROMPT_SUFFIX
            )
            prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
            response_ids = tuple(tokenizer.encode(source, add_special_tokens=False)) + (
                int(tokenizer.eos_token_id),
            )
            return {
                "serialized_tokens": len(prompt_ids) + len(response_ids),
                "response_tokens": len(response_ids),
            }

        selected = reconstruct_s12_selected_pair(
            member, manifest=triplet, row=candidate_row, measure_tokens=measure
        )
        n_source = _solution(
            neutral_row,
            triplet["neutral_solution_id"],
            label="correct",
            fingerprint=triplet["neutral_solution_fingerprint"],
        )
        source_rows[str(member["pair_id"])] = {
            "member": member,
            "triplet": triplet,
            "candidate_row": candidate_row,
            "neutral_row": neutral_row,
            "R": selected["R"]["source"],
            "I": selected["I"]["source"],
            "N": n_source,
        }
        payloads.append(
            {
                "pair_id": member["pair_id"],
                "triplet": triplet,
                "candidate_row": candidate_row,
                "neutral_row": neutral_row,
                "r_source": selected["R"]["source"],
                "i_source": selected["I"]["source"],
                "n_source": n_source,
                "sandbox_parent": str(sandbox_parent),
            }
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, object]] = {}
    for path in checkpoint_dir.glob("*.json"):
        value = json.loads(path.read_text("utf-8"))
        completed[str(value["pair_id"])] = value
    pending = [payload for payload in payloads if payload["pair_id"] not in completed]
    if pending:
        sandbox_parent.mkdir(parents=True, exist_ok=True)
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_s12_preoutcome_worker, value): value
                    for value in pending
                }
                for future in as_completed(futures):
                    value = future.result()
                    completed[str(value["pair_id"])] = value
                    ordinal = str(value["pair_id"]).split(":", 1)[1]
                    _atomic_json(checkpoint_dir / f"{int(ordinal):06d}.json", value)
        finally:
            shutil.rmtree(sandbox_parent, ignore_errors=True)
    if len(completed) != 354:
        raise S12InfrastructureInvalid("S12 finite pre-outcome worklist is incomplete")

    nat_pairs = []
    pc_pairs = []
    policy_pairs = []
    support_rows = []
    for record in compact["records"]:
        pair_id = str(record["pair_id"])
        source = source_rows[pair_id]
        work = completed[pair_id]
        triplet = source["triplet"]
        units = {
            role: tokenized(
                triplet,
                source["candidate_row"] if role != "N" else source["neutral_row"],
                str(source[role]),
                role,
                "candidate" if role != "N" else "neutral",
            )
            for role in ("I", "R", "N")
        }
        base = {
            "pair_id": pair_id,
            "problem_id": record["problem_id"],
            "provenance": record["r_provenance"],
            "failure_family": work["failure_family"],
            "units": units,
        }
        nat_pairs.append(base)
        policy_member = bool(work["n_qualified"])
        if policy_member:
            policy_pairs.append(base)
        mutation = work["mutation"]
        mutant_qualified = bool(mutation.get("available"))
        if mutant_qualified:
            mutant_source = _mutated_source(str(source["R"]), mutation)
            mutant = tokenized(
                triplet, source["candidate_row"], mutant_source, "MUTANT", "candidate"
            )
            if len(mutant.token_ids) <= 2_048:
                pc_pairs.append(
                    {
                        **base,
                        "units": {
                            "PC_CLEAN": tokenized(
                                triplet,
                                source["candidate_row"],
                                str(source["R"]),
                                "CLEAN",
                                "candidate",
                            ),
                            "PC_MUTANT": mutant,
                        },
                    }
                )
            else:
                mutant_qualified = False
        support_rows.append(
            {
                "pair_id": pair_id,
                "problem_id": record["problem_id"],
                "r_provenance": record["r_provenance"],
                "failure_family": work["failure_family"],
                "i_response_tokens": record["response_tokens"]["I"],
                "r_response_tokens": record["response_tokens"]["R"],
                "n_qualified": policy_member,
                "policy_member": policy_member,
                "mutant_qualified": mutant_qualified,
            }
        )
    config_path = Path(__file__).parents[3] / "configs" / "codecontests_s12_apcf_v1.json"
    config = load_s12_config(config_path)
    probe_root = 1
    nat_closes = bool(build_s12_stage_plans(config, stage="NAT", root=probe_root, pairs=nat_pairs))
    policy_closes = bool(policy_pairs) and bool(
        build_s12_stage_plans(config, stage="POLICY", root=probe_root, pairs=policy_pairs)
    )
    pc_closes = bool(pc_pairs) and bool(
        build_s12_stage_plans(config, stage="PC", root=probe_root, pairs=pc_pairs)
    )
    return {
        "nat_pairs": tuple(nat_pairs),
        "policy_pairs": tuple(policy_pairs),
        "pc_pairs": tuple(pc_pairs),
        "support_rows": tuple(support_rows),
        "nat_support": evaluate_s12_role_support(
            compact["records"], roles=("I", "R"), exposure_plan_closes=nat_closes
        ),
        "policy_available": policy_closes
        and evaluate_s12_role_support(
            tuple(
                record
                for record in compact["records"]
                if completed[str(record["pair_id"])]["n_qualified"] is True
            ),
            roles=("I", "R", "N"),
            exposure_plan_closes=True,
        )["passed"],
        "pc_support": evaluate_s12_role_support(
            tuple(
                {
                    "pair_id": pair["pair_id"],
                    "problem_id": pair["problem_id"],
                    "response_tokens": {
                        "PC_CLEAN": pair["units"]["PC_CLEAN"].response_tokens,
                        "PC_MUTANT": pair["units"]["PC_MUTANT"].response_tokens,
                    },
                    "serialized_tokens": {
                        "PC_CLEAN": len(pair["units"]["PC_CLEAN"].token_ids),
                        "PC_MUTANT": len(pair["units"]["PC_MUTANT"].token_ids),
                    },
                }
                for pair in pc_pairs
            ),
            roles=("PC_CLEAN", "PC_MUTANT"),
            exposure_plan_closes=pc_closes,
        ),
        "identity_sha256": compact["identity_sha256"],
    }


def evaluate_s12_role_support(
    records: Sequence[Mapping[str, object]],
    *,
    roles: Sequence[str],
    exposure_plan_closes: bool,
) -> dict[str, object]:
    """Evaluate the frozen count/ESS/share/serialization/exposure support gate."""

    pair_ids = [record.get("pair_id") for record in records]
    problem_ids = [record.get("problem_id") for record in records]
    identity_pass = (
        len(records) == len(set(pair_ids)) == len(set(problem_ids))
        and all(isinstance(value, str) and value for value in (*pair_ids, *problem_ids))
    )
    role_metrics: dict[str, dict[str, object]] = {}
    serialized_fit = True
    for role in roles:
        weights: list[int] = []
        for record in records:
            response = record.get("response_tokens")
            serialized = record.get("serialized_tokens")
            weight = response.get(role) if isinstance(response, Mapping) else None
            length = serialized.get(role) if isinstance(serialized, Mapping) else None
            if (
                isinstance(weight, bool)
                or not isinstance(weight, int)
                or weight <= 0
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
            ):
                raise S12InputInvalid("S12 role serialization metadata is malformed")
            weights.append(weight)
            serialized_fit = serialized_fit and length <= 2_048
        total = sum(weights)
        ess = total * total / sum(weight * weight for weight in weights) if weights else 0.0
        maximum_share = max(weights, default=0) / total if total else math.inf
        role_metrics[role] = {
            "response_token_mass": total,
            "kish_ess": ess,
            "max_pair_token_share": maximum_share,
            "ess_pass": ess >= 80.0,
            "max_share_pass": maximum_share <= 0.02,
        }
    count_pass = len(records) >= 100
    passed = (
        count_pass
        and identity_pass
        and serialized_fit
        and exposure_plan_closes
        and all(
            metric["ess_pass"] and metric["max_share_pass"]
            for metric in role_metrics.values()
        )
    )
    return {
        "schema": "codecontests-s12-support-gate-v1",
        "pair_count": len(records),
        "unique_problem_count": len(set(problem_ids)),
        "roles": tuple(roles),
        "role_metrics": role_metrics,
        "count_pass": count_pass,
        "identity_pass": identity_pass,
        "serialized_fit_pass": serialized_fit,
        "exposure_plan_closure_pass": bool(exposure_plan_closes),
        "passed": passed,
    }


def _initialize_s12c_csr_worker(model_path: str) -> None:
    global _S12C_TOKENIZER
    os.umask(0o022)
    from transformers import AutoTokenizer

    _S12C_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )


_S12D_FATAL_CLASSES = frozenset(
    {"input_invalid", "contract_error", "infrastructure_error", "resource_limit"}
)
_S12D_RUNTIME_ONLY_FIELDS = frozenset(
    {"wall_time_seconds", "worker_pid", "sandbox_path", "completion_order", "log_bytes"}
)


def build_codecontests_s12d_worker_envelope(
    *,
    manifest_ordinal: int,
    triplet_id: str,
    primitive_row: Mapping[str, object] | None = None,
    fatal_class: str | None = None,
    affected_key: Sequence[str] | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    """Build exactly one typed S12d worker return."""

    if (
        isinstance(manifest_ordinal, bool)
        or not isinstance(manifest_ordinal, int)
        or manifest_ordinal < 0
        or not isinstance(triplet_id, str)
        or not triplet_id
    ):
        raise ValueError("S12d envelope identity is invalid")
    if (primitive_row is None) == (fatal_class is None):
        raise ValueError("S12d envelope must contain exactly one result disposition")
    if primitive_row is not None:
        row = dict(primitive_row)
        if row.get("triplet_id") != triplet_id:
            raise ValueError("S12d envelope primitive triplet is mismatched")
        row["manifest_ordinal"] = manifest_ordinal
        return {
            "schema": "codecontests-s12d-worker-envelope-v1",
            "manifest_ordinal": manifest_ordinal,
            "triplet_id": triplet_id,
            "envelope_disposition": "closed_triplet",
            "primitive_row": row,
            "fatal_class": None,
            "affected_key": None,
            "reason": None,
        }
    if fatal_class not in _S12D_FATAL_CLASSES or not isinstance(reason, str) or not reason:
        raise ValueError("S12d fatal envelope is invalid")
    key = tuple(affected_key) if affected_key is not None else None
    if key is not None and (
        len(key) != 4 or any(not isinstance(value, str) or not value for value in key)
    ):
        raise ValueError("S12d fatal affected key is invalid")
    return {
        "schema": "codecontests-s12d-worker-envelope-v1",
        "manifest_ordinal": manifest_ordinal,
        "triplet_id": triplet_id,
        "envelope_disposition": "fatal_error",
        "primitive_row": None,
        "fatal_class": fatal_class,
        "affected_key": key,
        "reason": reason[:512],
    }


def join_codecontests_s12d_envelopes(
    manifest_triplet_ids: Sequence[str],
    envelopes: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Join worker returns by triplet key and emit only manifest order."""

    expected = tuple(manifest_triplet_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("S12d manifest contains a duplicate triplet key")
    by_key: dict[str, Mapping[str, object]] = {}
    for envelope in envelopes:
        if envelope.get("schema") != "codecontests-s12d-worker-envelope-v1":
            raise ValueError("S12d worker envelope schema is invalid")
        key = envelope.get("triplet_id")
        if not isinstance(key, str) or not key:
            raise ValueError("S12d worker envelope key is invalid")
        if key in by_key:
            raise ValueError("S12d worker envelope contains a duplicate triplet key")
        by_key[key] = envelope
    orphan = sorted(set(by_key) - set(expected))
    if orphan:
        raise ValueError(f"S12d worker envelope has orphan key {orphan[0]}")
    missing = [key for key in expected if key not in by_key]
    if missing:
        raise ValueError(f"S12d worker envelope is missing key {missing[0]}")
    rows: list[dict[str, object]] = []
    for ordinal, key in enumerate(expected):
        envelope = by_key[key]
        if envelope.get("manifest_ordinal") != ordinal:
            raise ValueError("S12d worker envelope manifest ordinal is mismatched")
        if envelope.get("envelope_disposition") == "fatal_error":
            raise CodeContestsS12DWorkerFailure(envelope)
        if envelope.get("envelope_disposition") != "closed_triplet":
            raise ValueError("S12d worker envelope disposition is invalid")
        row = envelope.get("primitive_row")
        if not isinstance(row, Mapping) or row.get("triplet_id") != key:
            raise ValueError("S12d worker primitive row is mismatched")
        rows.append(dict(row))
    return tuple(rows)


def _s12d_scientific_projection(row: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in _S12D_RUNTIME_ONLY_FIELDS}


def compare_codecontests_s12d_primitive_relations(
    serial_rows: Sequence[Mapping[str, object]],
    concurrent_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare every scientific primitive while ignoring fixed runtime metadata."""

    serial = tuple(_s12d_scientific_projection(row) for row in serial_rows)
    concurrent = tuple(_s12d_scientific_projection(row) for row in concurrent_rows)
    mismatch = next(
        (
            index
            for index, (left, right) in enumerate(zip(serial, concurrent))
            if left != right
        ),
        None,
    )
    if mismatch is None and len(serial) != len(concurrent):
        mismatch = min(len(serial), len(concurrent))
    serial_bytes = json.dumps(
        serial, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list
    ).encode("utf-8")
    concurrent_bytes = json.dumps(
        concurrent, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list
    ).encode("utf-8")
    return {
        "schema": "codecontests-s12d-primitive-parity-v1",
        "serial_sha256": hashlib.sha256(serial_bytes).hexdigest(),
        "concurrent_sha256": hashlib.sha256(concurrent_bytes).hexdigest(),
        "serial_count": len(serial),
        "concurrent_count": len(concurrent),
        "first_mismatch_ordinal": mismatch,
        "disagreement_count": sum(
            left != right for left, right in zip(serial, concurrent)
        ) + abs(len(serial) - len(concurrent)),
        "passed": serial == concurrent,
    }


def promote_codecontests_s12d_relation(
    rows: Sequence[Mapping[str, object]], *, parity: Mapping[str, object]
) -> dict[str, object]:
    """Reveal memberships exactly once after an exact parity receipt."""

    if not (
        parity.get("schema") == "codecontests-s12d-primitive-parity-v1"
        and parity.get("passed") is True
        and parity.get("disagreement_count") == 0
    ):
        raise RuntimeError("S12d support promotion requires passed exact parity")
    promoted = tuple(
        {**dict(row), **derive_codecontests_s12d_membership(row)} for row in rows
    )
    counts = {
        "natural": sum(row["natural_member"] for row in promoted),
        "positive_control": sum(row["mutation_member"] for row in promoted),
        "policy": sum(row["policy_member"] for row in promoted),
    }
    identities = [
        (
            row.get("triplet_id"),
            row["natural_member"],
            row["mutation_member"],
            row["policy_member"],
        )
        for row in promoted
    ]
    return {
        "schema": "codecontests-s12d-promoted-relation-v1",
        "rows": promoted,
        "counts": counts,
        "parity_sha256": parity["concurrent_sha256"],
        "membership_identity_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def summarize_codecontests_s12d_promoted_reference(
    primitive_rows: Sequence[Mapping[str, object]],
    *,
    parity: Mapping[str, object],
) -> dict[str, object]:
    """Independently reconstruct the compact post-parity membership summary."""

    if not (
        parity.get("schema") == "codecontests-s12d-primitive-parity-v1"
        and parity.get("passed") is True
    ):
        raise RuntimeError("S12d reference summary requires passed parity")
    counts = {"natural": 0, "positive_control": 0, "policy": 0}
    identities: list[tuple[object, bool, bool, bool]] = []
    for primitive in primitive_rows:
        membership = derive_codecontests_s12d_membership(primitive)
        counts["natural"] += membership["natural_member"]
        counts["positive_control"] += membership["mutation_member"]
        counts["policy"] += membership["policy_member"]
        identities.append(
            (
                primitive.get("triplet_id"),
                membership["natural_member"],
                membership["mutation_member"],
                membership["policy_member"],
            )
        )
    return {
        "counts": counts,
        "membership_identity_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _compact_s12c_csr_row(value: Mapping[str, object]) -> dict[str, object]:
    objects = value.get("selection_execution_objects")
    relation = value.get("relation_summary")
    if not isinstance(objects, Sequence) or not isinstance(relation, Mapping):
        raise S12InfrastructureInvalid("S12c worker relation is malformed")
    compact_objects = [
        {
            key: item.get(key)
            for key in (
                "logical_object_id", "judge_request_id", "source_locator", "role",
                "selection_id", "judged_payload_sha256", "panel_id",
                "measurement_device_id",
            )
        }
        for item in objects
        if isinstance(item, Mapping)
    ]
    compact_relation = {
        "objects": compact_objects,
        "records": relation.get("records"),
        "semantic_counts": relation.get("semantic_counts"),
        "missing_object_ids": relation.get("missing_object_ids"),
        "orphan_object_ids": relation.get("orphan_object_ids"),
        "infrastructure_object_ids": relation.get("infrastructure_object_ids"),
    }
    relation_identity = hashlib.sha256(
        json.dumps(compact_relation, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    ).hexdigest()
    role_objects: dict[str, list[dict[str, object]]] = {}
    for item in compact_objects:
        role_objects.setdefault(str(item["role"]), []).append(item)
    response_tokens = {
        "R": value.get("selected_r_response_tokens"),
        "I": value.get("i_response_tokens"),
        "M": value.get("selected_m_response_tokens"),
        "N": value.get("n_response_tokens"),
    }
    serialized_tokens = {
        "R": value.get("selected_r_serialized_tokens"),
        "I": value.get("i_serialized_tokens"),
        "M": value.get("selected_m_serialized_tokens"),
        "N": value.get("n_serialized_tokens"),
    }
    return {
        "schema": "codecontests-s12c-csr-row-v1",
        "triplet_id": value.get("triplet_id"),
        "candidate_problem_id": value.get("candidate_problem_id"),
        "r_attempt_order": value.get("r_attempt_order"),
        "r_attempt_count": value.get("r_attempt_count"),
        "r_search_complete": value.get("r_search_complete"),
        "selected_r_id": value.get("selected_r_id"),
        "selected_r_fingerprint": value.get("selected_r_fingerprint"),
        "r_provenance": value.get("r_provenance"),
        "i_id": value.get("i_id"),
        "i_category": value.get("i_category"),
        "i_witness_test_id": value.get("i_witness_test_id"),
        "mutation_panel_anchor_test_id": value.get("mutation_panel_anchor_test_id"),
        "failure_family": value.get("failure_family"),
        "m_attempt_order": value.get("m_attempt_order"),
        "m_attempt_count": value.get("m_attempt_count"),
        "m_search_complete": value.get("m_search_complete"),
        "selected_m_id": value.get("selected_m_id"),
        "m_category": value.get("m_category"),
        "n_id": value.get("n_id"),
        "n_category": value.get("n_category"),
        "natural_member": value.get("natural_member") is True,
        "mutation_member": value.get("mutation_member") is True,
        "policy_member": value.get("policy_member") is True,
        "response_tokens": response_tokens,
        "serialized_tokens": serialized_tokens,
        "role_objects": role_objects,
        "relation_identity_sha256": relation_identity,
        "relation_object_count": len(compact_objects),
        "infrastructure_error_count": len(relation.get("infrastructure_object_ids", ())),
    }


def evaluate_codecontests_s12d_worker(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a typed envelope for one production S12d triplet."""

    triplet = payload["triplet"]
    triplet_id = str(triplet.get("triplet_id"))
    ordinal = int(payload["manifest_ordinal"])
    deadline_unix = payload.get("deadline_unix")
    if isinstance(deadline_unix, (int, float)) and time.time() >= deadline_unix:
        return build_codecontests_s12d_worker_envelope(
            manifest_ordinal=ordinal,
            triplet_id=triplet_id,
            fatal_class="resource_limit",
            reason="S12d qualification deadline expired before dispatch",
        )
    if _S12C_TOKENIZER is None:
        return build_codecontests_s12d_worker_envelope(
            manifest_ordinal=ordinal,
            triplet_id=triplet_id,
            fatal_class="infrastructure_error",
            reason="S12d tokenizer worker was not initialized",
        )
    candidate_row = payload["candidate_row"]
    neutral_row = payload["neutral_row"]

    def measurement(row, source: str) -> dict[str, int]:
        description = row.get("description")
        if not isinstance(description, str):
            raise CodeContestsS12DInputInvalid("S12d problem description is malformed")
        prompt = CODECONTESTS_PROMPT_PREFIX + _lf_prompt(description) + CODECONTESTS_PROMPT_SUFFIX
        prompt_ids = tuple(_S12C_TOKENIZER.encode(prompt, add_special_tokens=True))
        response_ids = tuple(_S12C_TOKENIZER.encode(source, add_special_tokens=False)) + (
            int(_S12C_TOKENIZER.eos_token_id),
        )
        return {
            "serialized_tokens": len(prompt_ids) + len(response_ids),
            "response_tokens": len(response_ids),
        }

    sandbox_base = (
        Path(str(payload["sandbox_parent"]))
        / f"worker_{os.getpid()}_triplet_{ordinal:06d}"
    )
    affected_key = (
        triplet_id,
        "R",
        str(triplet.get("candidate_problem_id")),
        str(triplet.get("repair_solution_id") or "original-r"),
    )
    envelope: dict[str, object]
    try:
        sandbox_base.mkdir(parents=True, exist_ok=False)
        value = adjudicate_codecontests_s12d_triplet(
            triplet,
            candidate_row,
            neutral_row,
            measure_tokens=lambda source: measurement(candidate_row, source),
            neutral_measure_tokens=lambda source: measurement(neutral_row, source),
            payload_builder=build_judged_payload,
            measurement_device_id=str(payload["measurement_device_id"]),
            judge=lambda source, tests, stop_on_first_failure=True: judge_codecontests_program(
                source,
                tests,
                sandbox_base=sandbox_base,
                stop_on_first_failure=stop_on_first_failure,
            ),
        )
        compact = _compact_s12c_csr_row(value)
        compact.update(
            {
                "manifest_ordinal": ordinal,
                "eligibility_dispositions": value.get("eligibility_dispositions", []),
                "execution_dispositions": value.get("execution_dispositions", []),
                "candidate_skip_counts": value.get("candidate_skip_counts", {}),
                "contract_error_count": 0,
            }
        )
        primitive = project_codecontests_s12d_primitive_row(compact)
        if primitive["infrastructure_error_count"]:
            envelope = build_codecontests_s12d_worker_envelope(
                manifest_ordinal=ordinal,
                triplet_id=triplet_id,
                fatal_class="infrastructure_error",
                affected_key=affected_key,
                reason="judge returned infrastructure_error",
            )
        else:
            envelope = build_codecontests_s12d_worker_envelope(
                manifest_ordinal=ordinal,
                triplet_id=triplet_id,
                primitive_row=primitive,
            )
    except CodeContestsS12DInputInvalid as exc:
        envelope = build_codecontests_s12d_worker_envelope(
            manifest_ordinal=ordinal,
            triplet_id=triplet_id,
            fatal_class="input_invalid",
            affected_key=affected_key,
            reason=str(exc),
        )
    except TimeoutError as exc:
        envelope = build_codecontests_s12d_worker_envelope(
            manifest_ordinal=ordinal,
            triplet_id=triplet_id,
            fatal_class="resource_limit",
            affected_key=affected_key,
            reason=str(exc),
        )
    except (S12InfrastructureInvalid, OSError) as exc:
        envelope = build_codecontests_s12d_worker_envelope(
            manifest_ordinal=ordinal,
            triplet_id=triplet_id,
            fatal_class="infrastructure_error",
            affected_key=affected_key,
            reason=str(exc),
        )
    except Exception as exc:
        envelope = build_codecontests_s12d_worker_envelope(
            manifest_ordinal=ordinal,
            triplet_id=triplet_id,
            fatal_class="contract_error",
            affected_key=affected_key,
            reason=str(exc),
        )
    try:
        shutil.rmtree(sandbox_base)
    except OSError as exc:
        if envelope.get("envelope_disposition") != "fatal_error":
            envelope = build_codecontests_s12d_worker_envelope(
                manifest_ordinal=ordinal,
                triplet_id=triplet_id,
                fatal_class="infrastructure_error",
                affected_key=affected_key,
                reason=f"sandbox cleanup failed: {exc}",
            )
    return envelope


def classify_codecontests_s12d_universe(
    *, manifest_path: Path, dataset_root: Path
) -> dict[str, object]:
    """Classify every referenced problem without executing any program."""

    manifest = read_codecontests_manifest(manifest_path)
    registry = validate_triplet_registry(manifest)
    if registry.get("triplet_count") != 590:
        raise CodeContestsS12DInputInvalid("S12d manifest denominator is not 590")
    rows = load_census_rows(dataset_root, manifest)
    referenced: dict[str, tuple[str, Mapping[str, object]]] = {}
    for triplet in manifest:
        for prefix in ("candidate", "neutral"):
            problem_id = triplet.get(f"{prefix}_problem_id")
            fingerprint = triplet.get(f"{prefix}_problem_fingerprint")
            if not isinstance(problem_id, str) or not isinstance(fingerprint, str):
                raise CodeContestsS12DInputInvalid("S12d referenced problem locator is invalid")
            current = referenced.get(problem_id)
            if current is not None and current[0] != fingerprint:
                raise CodeContestsS12DInputInvalid("S12d referenced problem fingerprint is inconsistent")
            referenced[problem_id] = (fingerprint, rows[_index(problem_id)])
    classifications = tuple(
        classify_codecontests_s12d_problem(
            problem_id=problem_id, problem_fingerprint=fingerprint, row=row
        )
        for problem_id, (fingerprint, row) in sorted(referenced.items())
    )
    train_10136 = next(
        (value for value in classifications if value["problem_id"] == "train:10136"), None
    )
    if not (
        isinstance(train_10136, Mapping)
        and train_10136.get("eligibility_disposition") == "expected_exclusion"
        and train_10136.get("exclusion_reason") == "unsupported_file_io_problem"
    ):
        raise CodeContestsS12DInputInvalid("S12d train:10136 classification differs")
    return {
        "schema": "codecontests-s12d-static-classification-v1",
        "referenced_problem_count": len(referenced),
        "classified_problem_count": len(classifications),
        "coverage": 1.0,
        "classifications": classifications,
    }


def run_codecontests_s12d_uqcc_pass(
    *,
    manifest_path: Path,
    dataset_root: Path,
    tokenizer_path: Path,
    scratch_dir: Path,
    measurement_device_id: str,
    workers: int,
    deadline_unix: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run one non-resumable full S12d pass at frozen C1 or C4."""

    if workers not in {1, 4}:
        raise ValueError("S12d UQCC workers must be one or four")
    if scratch_dir.exists() and any(scratch_dir.iterdir()):
        raise ValueError("S12d UQCC pass cannot resume or splice prior work")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_codecontests_manifest(manifest_path)
    registry = validate_triplet_registry(manifest)
    if registry.get("triplet_count") != 590:
        raise CodeContestsS12DInputInvalid("S12d manifest denominator is not 590")
    rows = load_census_rows(dataset_root, manifest)
    sandbox_parent = scratch_dir / "sandboxes"
    sandbox_parent.mkdir()
    payloads = tuple(
        {
            "manifest_ordinal": ordinal,
            "triplet": dict(triplet),
            "candidate_row": rows[_index(triplet["candidate_problem_id"])],
            "neutral_row": rows[_index(triplet["neutral_problem_id"])],
            "measurement_device_id": measurement_device_id,
            "sandbox_parent": str(sandbox_parent),
            "deadline_unix": deadline_unix,
        }
        for ordinal, triplet in enumerate(manifest)
    )
    envelopes: list[dict[str, object]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_s12c_csr_worker,
            initargs=(str(tokenizer_path),),
        ) as pool:
            futures = [pool.submit(evaluate_codecontests_s12d_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                envelopes.append(future.result())
                if progress_callback is not None:
                    progress_callback(len(envelopes), len(payloads))
    finally:
        try:
            shutil.rmtree(sandbox_parent)
        except FileNotFoundError:
            pass
    ordered = join_codecontests_s12d_envelopes(
        tuple(str(row["triplet_id"]) for row in manifest), envelopes
    )
    return {
        "schema": "codecontests-s12d-uqcc-pass-v1",
        "workers": workers,
        "completed": len(ordered),
        "expected": len(manifest),
        "rows": ordered,
    }


def run_codecontests_s12d_uqcc_sentinels(
    *,
    manifest_path: Path,
    dataset_root: Path,
    tokenizer_path: Path,
    scratch_dir: Path,
    measurement_device_id: str,
    sentinel_triplet_ids: Sequence[str],
    workers: int,
    deadline_unix: float | None = None,
) -> dict[str, object]:
    """Run the frozen real sentinel subset through the production worker seam."""

    if workers not in {1, 4}:
        raise ValueError("S12d sentinel workers must be one or four")
    if not 4 <= len(sentinel_triplet_ids) <= 6 or len(set(sentinel_triplet_ids)) != len(
        sentinel_triplet_ids
    ):
        raise ValueError("S12d sentinel registry must contain four to six unique keys")
    if scratch_dir.exists() and any(scratch_dir.iterdir()):
        raise ValueError("S12d sentinel pass cannot resume")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_codecontests_manifest(manifest_path)
    validate_triplet_registry(manifest)
    by_id = {str(row["triplet_id"]): row for row in manifest}
    if any(key not in by_id for key in sentinel_triplet_ids):
        raise CodeContestsS12DInputInvalid("S12d sentinel key is outside the manifest")
    rows = load_census_rows(dataset_root, manifest)
    sandbox_parent = scratch_dir / "sandboxes"
    sandbox_parent.mkdir()
    payloads = tuple(
        {
            "manifest_ordinal": ordinal,
            "triplet": dict(by_id[key]),
            "candidate_row": rows[_index(by_id[key]["candidate_problem_id"])],
            "neutral_row": rows[_index(by_id[key]["neutral_problem_id"])],
            "measurement_device_id": measurement_device_id,
            "sandbox_parent": str(sandbox_parent),
            "deadline_unix": deadline_unix,
        }
        for ordinal, key in enumerate(sentinel_triplet_ids)
    )
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_s12c_csr_worker,
            initargs=(str(tokenizer_path),),
        ) as pool:
            envelopes = tuple(pool.map(evaluate_codecontests_s12d_worker, payloads))
    finally:
        try:
            shutil.rmtree(sandbox_parent)
        except FileNotFoundError:
            pass
    ordered = join_codecontests_s12d_envelopes(tuple(sentinel_triplet_ids), envelopes)
    return {
        "schema": "codecontests-s12d-sentinel-pass-v1",
        "workers": workers,
        "rows": ordered,
    }


def _s12c_csr_worker(payload: Mapping[str, object]) -> dict[str, object]:
    if _S12C_TOKENIZER is None:
        raise RuntimeError("S12c tokenizer worker was not initialized")
    triplet = payload["triplet"]
    candidate_row = payload["candidate_row"]

    def measurement(row, source: str) -> dict[str, int]:
        description = row.get("description")
        if not isinstance(description, str):
            raise S12InputInvalid("S12c problem description is malformed")
        prompt = CODECONTESTS_PROMPT_PREFIX + _lf_prompt(description) + CODECONTESTS_PROMPT_SUFFIX
        prompt_ids = tuple(_S12C_TOKENIZER.encode(prompt, add_special_tokens=True))
        response_ids = tuple(_S12C_TOKENIZER.encode(source, add_special_tokens=False)) + (
            int(_S12C_TOKENIZER.eos_token_id),
        )
        return {"serialized_tokens": len(prompt_ids) + len(response_ids), "response_tokens": len(response_ids)}

    def measure(source: str) -> dict[str, int]:
        return measurement(candidate_row, source)

    sandbox_base = Path(str(payload["sandbox_parent"])) / f"worker_{os.getpid()}"
    sandbox_base.mkdir(parents=True, exist_ok=True)
    try:
        result = adjudicate_codecontests_s12c_triplet(
            triplet,
            candidate_row,
            payload["neutral_row"],
            measure_tokens=measure,
            neutral_measure_tokens=lambda source: measurement(payload["neutral_row"], source),
            payload_builder=build_judged_payload,
            measurement_device_id=str(payload["measurement_device_id"]),
            judge=lambda source, tests, stop_on_first_failure=True: judge_codecontests_program(
                source, tests, sandbox_base=sandbox_base,
                stop_on_first_failure=stop_on_first_failure,
            ),
        )
        return _compact_s12c_csr_row(result)
    finally:
        shutil.rmtree(sandbox_base, ignore_errors=True)


def _lf_prompt(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def run_codecontests_s12c_csr(
    *,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
    output_dir: Path,
    measurement_device_id: str,
    workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Execute/resume one complete canonical-C4 or serial-C1 S12c CSR."""

    if workers not in {1, 4}:
        raise ValueError("S12c CSR workers must be canonical four or serial one")
    if not isinstance(measurement_device_id, str) or not measurement_device_id:
        raise ValueError("S12c measurement device identity is invalid")
    manifest = read_codecontests_manifest(manifest_path)
    validate_triplet_registry(manifest)
    rows = load_census_rows(dataset_root, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    freeze = {
        "schema": S12C_CSR_SCHEMA,
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "measurement_device_id": measurement_device_id,
        "execution_mode": "canonical_c4" if workers == 4 else "serial_c1",
        "workers": workers,
    }
    freeze_path = output_dir / "execution_freeze.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text("utf-8")) != freeze:
            raise ValueError("S12c CSR resume freeze differs")
    else:
        _atomic_json(freeze_path, freeze)
    completed: dict[str, dict[str, object]] = {}
    for path in sorted(checkpoints.glob("*.json")):
        row = json.loads(path.read_text("utf-8"))
        triplet_id = row.get("triplet_id")
        if not isinstance(triplet_id, str) or triplet_id in completed:
            raise ValueError("S12c CSR checkpoint identity is invalid")
        completed[triplet_id] = row
    sandbox_parent = output_dir / "_sandboxes"
    payloads = [
        {
            "triplet": dict(triplet),
            "candidate_row": rows[_index(triplet["candidate_problem_id"])],
            "neutral_row": rows[_index(triplet["neutral_problem_id"])],
            "measurement_device_id": measurement_device_id,
            "sandbox_parent": str(sandbox_parent),
        }
        for triplet in manifest
        if triplet["triplet_id"] not in completed
    ]
    if payloads:
        sandbox_parent.mkdir(exist_ok=True)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_s12c_csr_worker,
                initargs=(str(model_path),),
            ) as pool:
                futures = {pool.submit(_s12c_csr_worker, payload): payload["triplet"]["triplet_id"] for payload in payloads}
                for future in as_completed(futures):
                    triplet_id = str(futures[future])
                    row = future.result()
                    if row.get("triplet_id") != triplet_id:
                        raise S12InfrastructureInvalid("S12c worker returned a shifted object")
                    completed[triplet_id] = row
                    _atomic_json(checkpoints / f"{int(triplet_id.split(':')[1]):06d}.json", row)
                    if progress_callback is not None:
                        progress_callback(len(completed), 590)
        finally:
            shutil.rmtree(sandbox_parent, ignore_errors=True)
    ordered = tuple(completed[f"triplet:{index}"] for index in range(590))
    summary = summarize_codecontests_s12c_csr(ordered)
    table_path = output_dir / "CODECONTESTS_S12C_CSR_TABLE.csv"
    fields = list(ordered[0])
    with table_path.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })
    os.replace(table_path.with_suffix(".csv.tmp"), table_path)
    result = {
        **summary,
        "execution_mode": freeze["execution_mode"],
        "workers": workers,
        "csr_table_sha256": sha256_file(table_path),
        "rows": ordered,
    }
    _atomic_json(output_dir / "CSR_RESULT.json", {key: value for key, value in result.items() if key != "rows"})
    return result


def summarize_codecontests_s12c_csr(
    rows: Sequence[Mapping[str, object]], *, expected_total: int = 590
) -> dict[str, object]:
    """Validate frozen-order CSR coverage before interpreting any support."""

    if isinstance(expected_total, bool) or not isinstance(expected_total, int) or expected_total <= 0:
        raise ValueError("S12c expected CSR denominator is invalid")
    expected_ids = [f"triplet:{index}" for index in range(expected_total)]
    observed_ids = [row.get("triplet_id") for row in rows]
    if observed_ids != expected_ids:
        raise ValueError("S12c CSR order or denominator differs from freeze")
    problem_ids = [row.get("candidate_problem_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in problem_ids) or len(set(problem_ids)) != expected_total:
        raise ValueError("S12c CSR problem identity is invalid")
    relation_ids = [row.get("relation_identity_sha256") for row in rows]
    if any(not isinstance(value, str) or len(value) != 64 for value in relation_ids):
        raise ValueError("S12c CSR relation identity is invalid")
    infrastructure = sum(int(row.get("infrastructure_error_count", 0)) for row in rows)
    counts = {
        "natural": sum(row.get("natural_member") is True for row in rows),
        "positive_control": sum(row.get("mutation_member") is True for row in rows),
        "policy": sum(row.get("policy_member") is True for row in rows),
    }
    identity = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "codecontests-s12c-csr-summary-v1",
        "coverage": {"completed": len(rows), "expected": expected_total},
        "counts": counts,
        "infrastructure_error_count": infrastructure,
        "selection_execution_identity_sha256": identity,
        "process_qualified": infrastructure == 0,
    }


def compare_codecontests_s12c_csr(
    canonical_rows: Sequence[Mapping[str, object]],
    serial_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare the complete C4 and C1 CSR relations without projection."""

    canonical = json.dumps(
        canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list
    ).encode("utf-8")
    serial = json.dumps(
        serial_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list
    ).encode("utf-8")
    mismatch = next(
        (
            index for index, (left, right) in enumerate(zip(canonical_rows, serial_rows))
            if left != right
        ),
        None,
    )
    if mismatch is None and len(canonical_rows) != len(serial_rows):
        mismatch = min(len(canonical_rows), len(serial_rows))
    return {
        "schema": "codecontests-s12c-csr-exact-parity-v1",
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "serial_sha256": hashlib.sha256(serial).hexdigest(),
        "canonical_count": len(canonical_rows),
        "serial_count": len(serial_rows),
        "first_mismatch_ordinal": mismatch,
        "passed": canonical == serial,
    }


def read_codecontests_s12c_csr_table(path: Path) -> tuple[dict[str, object], ...]:
    """Load the compact CSR table while restoring its explicitly typed fields."""

    json_fields = {
        "r_attempt_order", "m_attempt_order", "response_tokens",
        "serialized_tokens", "role_objects",
    }
    bool_fields = {
        "r_search_complete", "m_search_complete", "natural_member",
        "mutation_member", "policy_member",
    }
    int_fields = {
        "r_attempt_count", "m_attempt_count", "relation_object_count",
        "infrastructure_error_count",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in json_fields:
                row[key] = json.loads(str(row[key]))
            for key in bool_fields:
                if row[key] not in {"True", "False"}:
                    raise ValueError("S12c CSR boolean field is invalid")
                row[key] = row[key] == "True"
            for key in int_fields:
                row[key] = int(str(row[key]))
            for key, value in tuple(row.items()):
                if value == "":
                    row[key] = None
            rows.append(row)
    summarize_codecontests_s12c_csr(rows)
    return tuple(rows)


def read_codecontests_s12_qualified_input(directory: Path) -> dict[str, object]:
    """Read ADR0118's sole qualification authority without dispatching a judge."""

    expected = {
        "TABLE.csv": "c534a19c28f51e9fc2b1872a39b22cd32bab733dd5f55148ee65c7cc3a98aa0b",
        "RESULT.json": "ed6e35bb462348280db56e0994cbeb6a4aec5ed382a87df222922e5d0e6f12f9",
        "FREEZE.json": "e41d3ac189db850aeedb36505d52dffd7f8ac3a5da00ba4e138a88e73758dcbd",
    }
    paths = {name: directory / f"CODECONTESTS_S12D_{name}" for name in expected}
    for name, digest in expected.items():
        if sha256_file(paths[name]) != digest:
            raise ValueError(f"S12 qualified input hash differs: {name}")
    result = json.loads(paths["RESULT.json"].read_text("utf-8"))
    freeze = json.loads(paths["FREEZE.json"].read_text("utf-8"))
    if not (
        result.get("terminal") == "COMPLETED_S12D_PREFORMAL_QUALIFIED"
        and result.get("endpoint_state") == result.get("dry_consumer_state") == "passed"
        and freeze.get("canonical_table_sha256") == expected["TABLE.csv"]
        and freeze.get("endpoint_qualification") == {"c1": 285, "c4": 285, "exact_parity": True}
        and result.get("support_exposure") == "known_from_prior_run"
    ):
        raise ValueError("S12 qualified input state is not valid for GPU promotion")
    rows = read_codecontests_s12c_csr_table(paths["TABLE.csv"])
    if len(rows) != 590 or len({row["triplet_id"] for row in rows}) != 590:
        raise ValueError("S12 qualified input must contain 590 unique triplets")
    return {"rows": rows, "table_path": paths["TABLE.csv"], "result": result,
            "freeze": freeze, "hashes": expected}


def reconstruct_nat_replace_input(workspace: Path) -> dict[str, object]:
    """Recover exact certified S230 I/N, without invoking model, judge or old stages."""
    from transformers import AutoTokenizer
    from ..arms.paired_filtering import materialize_nat_replace_pairs, build_nat_replace_roots
    from ..protocol import CODECONTESTS_FORMAL_ROOTS, CODECONTESTS_V2_FORMAL_ROOTS

    artifacts = workspace / "20_q1_operational_defect_audit/experiment/artifacts"
    old = json.loads((artifacts / "codecontests_s12/formal_20260905_1/CODECONTESTS_S12_EXECUTION_FREEZE.json").read_text("utf-8"))
    qualified = read_codecontests_s12_qualified_input(artifacts / "codecontests_s12d/formal_20260905_3")
    selected = [row for row in qualified["rows"] if row["policy_member"]]
    manifest_path = workspace / "v9/codecontests_f0a/probes/phase2_census/triplet_manifest.jsonl"
    manifest = read_codecontests_manifest(manifest_path)
    by_id = {row["triplet_id"]: row for row in manifest}
    chosen_manifest = [by_id[row["triplet_id"]] for row in selected]
    dataset_root = Path(old["dataset_root"])
    if not dataset_root.is_dir():
        dataset_root = workspace / "datasets/code_contests/802411c3010cb00d1b05bad57ca77365a3c699d6"
    model_path = Path(old["model_path"])
    if not model_path.is_dir():
        model_path = workspace / "models/Qwen2.5-1.5B/8faed761d45a263340a0528343f099c05c9a4323"
    data = load_census_rows(dataset_root, chosen_manifest)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    source_bindings = []
    overlap_records = []

    def resolve(row, role):
        triplet = by_id[row["triplet_id"]]
        prefix = "candidate" if role == "I" else "neutral"
        problem = data[_index(triplet[f"{prefix}_problem_id"])]
        source = _solution(problem, row["i_id" if role == "I" else "n_id"],
                           label="incorrect" if role == "I" else "correct",
                           fingerprint=triplet[f"{prefix}_solution_fingerprint"])
        matches = [obj for obj in row["role_objects"][role]
                   if obj["selection_id"] == triplet[f"{prefix}_solution_fingerprint"]]
        digest = hashlib.sha256(build_judged_payload(source)).hexdigest()
        if len(matches) != 1 or matches[0]["judged_payload_sha256"] != digest:
            raise ValueError("NAT-REPLACE source differs from qualified CSR")
        unit = _tokenized_source_unit(triplet=triplet, row=problem, source=source, arm=role,
                     prompt_role=prefix, tokenizer=tokenizer,
                     expected_response_tokens=row["response_tokens"][role])
        source_bindings.append(dict(pair_id=row["triplet_id"], role=role,
                        problem_id=triplet[f"{prefix}_problem_id"], payload_sha256=digest,
                        response_tokens=unit.response_tokens))
        overlap_records.append(dict(
            pair_id=row["triplet_id"], role=role,
            problem_id=triplet[f"{prefix}_problem_id"],
            prompt=problem["description"], program=source,
        ))
        return unit

    pairs = materialize_nat_replace_pairs(qualified["rows"], unit_resolver=resolve)
    totals = {role: sum(p["units"][role].response_tokens for p in pairs) for role in ("I", "N")}
    if totals != {"I": 66297, "N": 67198}:
        raise ValueError("NAT-REPLACE frozen token support differs")
    historical = set(CODECONTESTS_FORMAL_ROOTS)
    for values in CODECONTESTS_V2_FORMAL_ROOTS.values():
        historical.update(values)
    historical.update(int(job["root"]) for job in old["conditional_job_plan"])
    scope = json.loads((artifacts / "codecontests_scope_pc/diagnostic_20260906_1/SCOPE_PC_DESIGN.json").read_text("utf-8"))
    historical.update(scope["historical_roots"])
    historical.update(scope["roots"])
    return dict(pairs=pairs, source_bindings=source_bindings, overlap_records=overlap_records,
                token_totals=totals,
                roots=build_nat_replace_roots(historical_roots=sorted(historical)),
                historical_roots=sorted(historical), authority_hashes=qualified["hashes"],
                manifest_sha256=sha256_file(manifest_path),
                tokenizer=dict(eos=tokenizer.eos_token_id, pad=tokenizer.pad_token_id, bos=tokenizer.bos_token_id),
                paths={key: old[key] for key in ("model_path", "dataset_root", "endpoint_path", "evalplus_asset_path")})


def reconstruct_nat_onoff_inputs(
    *, workspace: Path, background_plan_path: Path
) -> dict[str, object]:
    """Reconstruct the frozen S354 units and complete historical B0 ledger."""

    from transformers import AutoTokenizer
    from ..arms.paired_filtering import load_s12_training_plan

    workspace = Path(workspace)
    artifacts = workspace / "20_q1_operational_defect_audit/experiment/artifacts"
    qualified = read_codecontests_s12_qualified_input(
        artifacts / "codecontests_s12d/formal_20260905_3"
    )
    s_rows = [row for row in qualified["rows"] if row["natural_member"] is True]
    if len(s_rows) != 354 or len({row["i_id"] for row in s_rows}) != 354:
        raise ValueError("Q1-NAT-ONOFF requires exactly 354 unique natural I records")
    category_counts = Counter(str(row["i_category"]) for row in s_rows)
    if category_counts != Counter(
        {"wrong_answer": 259, "compile_failure": 12, "runtime_failure": 75, "timeout": 8}
    ):
        raise ValueError("Q1-NAT-ONOFF S354 category composition differs")
    background_plan_path = Path(background_plan_path)
    if sha256_file(background_plan_path) != "2bbc4f65d6500914dea253feea20f87a3de9978b2dec9f9937062b0c8f47bd65":
        raise ValueError("Q1-NAT-ONOFF B0 ledger file identity differs")
    background = load_s12_training_plan(background_plan_path)
    if (
        len(background.update_bins) != 100
        or len(background.exposures) != 1_996
        or sum(item.retained_response_tokens for item in background.exposures) != 583_975
    ):
        raise ValueError("Q1-NAT-ONOFF B0 ledger geometry differs")

    manifest_path = workspace / "v9/codecontests_f0a/probes/phase2_census/triplet_manifest.jsonl"
    manifest = read_codecontests_manifest(manifest_path)
    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    by_csr = {str(row["triplet_id"]): row for row in qualified["rows"]}
    old = json.loads(
        (artifacts / "codecontests_s12/formal_20260905_1/CODECONTESTS_S12_EXECUTION_FREEZE.json").read_text("utf-8")
    )
    dataset_root = Path(str(old["dataset_root"]))
    if not dataset_root.is_dir():
        dataset_root = workspace / "datasets/code_contests/802411c3010cb00d1b05bad57ca77365a3c699d6"
    model_path = Path(str(old["model_path"]))
    if not model_path.is_dir():
        model_path = workspace / "models/Qwen2.5-1.5B/8faed761d45a263340a0528343f099c05c9a4323"
    used_triplets = {str(row["triplet_id"]) for row in s_rows}
    used_triplets.update(item.unit_id.split("/", 1)[0] for item in background.exposures)
    chosen_manifest = [by_triplet[identity] for identity in sorted(used_triplets)]
    data = load_census_rows(dataset_root, chosen_manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )

    def source_and_unit(row: Mapping[str, object], role: str):
        triplet = by_triplet[str(row["triplet_id"])]
        prefix = "candidate" if role == "I" else "neutral"
        problem_id = str(triplet[f"{prefix}_problem_id"])
        problem = data[_index(problem_id)]
        record_id = str(row["i_id" if role == "I" else "n_id"])
        fingerprint = triplet[f"{prefix}_solution_fingerprint"]
        source = _solution(
            problem, record_id,
            label="incorrect" if role == "I" else "correct",
            fingerprint=fingerprint,
        )
        objects = [item for item in row["role_objects"][role] if item["selection_id"] == fingerprint]
        if len(objects) != 1 or objects[0]["judged_payload_sha256"] != hashlib.sha256(
            build_judged_payload(source)
        ).hexdigest():
            raise ValueError("Q1-NAT-ONOFF source differs from qualified S12d object")
        unit = _tokenized_source_unit(
            triplet=triplet, row=problem, source=source, arm=role,
            prompt_role=prefix, tokenizer=tokenizer,
            expected_response_tokens=int(row["response_tokens"][role]),
        )
        return source, unit, {
            "unit_id": unit.unit_id,
            "record_id": record_id,
            "pair_id": str(row["triplet_id"]),
            "problem_id": problem_id,
            "source_lf": source.replace("\r\n", "\n").replace("\r", "\n"),
            "payload": (unit.token_ids, unit.loss_mask),
        }

    supplement_units, supplement_metadata = [], []
    for row in s_rows:
        _source, unit, metadata = source_and_unit(row, "I")
        supplement_units.append(unit)
        supplement_metadata.append(metadata)
    background_sources = {}
    for unit_id in sorted({item.unit_id for item in background.exposures}):
        triplet_id, role = unit_id.split("/", 1)
        if role != "N" or triplet_id not in by_csr:
            raise ValueError("Q1-NAT-ONOFF B0 contains an unknown non-N unit")
        background_sources[unit_id] = source_and_unit(by_csr[triplet_id], "N")
    background_metadata = []
    for exposure in background.exposures:
        _source, _unit, metadata = background_sources[exposure.unit_id]
        background_metadata.append({**metadata, "payload": (exposure.token_ids, exposure.loss_mask)})
    s_problem_ids = {str(row["problem_id"]) for row in supplement_metadata}
    b_problem_ids = {str(row["problem_id"]) for row in background_metadata}
    if len(background_sources) != 230 or s_problem_ids & b_problem_ids:
        raise ValueError("Q1-NAT-ONOFF corrected S/B problem identity contract differs")
    return {
        "background_plan": background,
        "background_metadata": tuple(background_metadata),
        "supplement_units": tuple(supplement_units),
        "supplement_metadata": tuple(supplement_metadata),
        "category_counts": dict(sorted(category_counts.items())),
        "actual_s_shifted_response_tokens": sum(sum(unit.loss_mask[1:]) for unit in supplement_units),
        "historical_s_saved_response_tokens": sum(int(row["response_tokens"]["I"]) for row in s_rows),
        "background_response_tokens": 583_975,
        "background_exposures": 1_996,
        "background_unique_units": 230,
        "s_problem_count": len(s_problem_ids),
        "b_problem_count": len(b_problem_ids),
        "cross_problem_count": 0,
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "authority_hashes": qualified["hashes"],
        "background_plan_file_sha256": sha256_file(background_plan_path),
    }


def reconstruct_scope_pc_input(workspace: Path) -> dict[str, object]:
    """Replay only selected PC identities and static eligibility; never dispatch a judge/model."""
    from transformers import AutoTokenizer
    from ..arms.codecontests import qualify_scope_pc_pair
    from ..arms.paired_filtering import group_scope_pc_pairs
    from ..census.codecontests import _panel_identity

    artifact_root = workspace / "20_q1_operational_defect_audit/experiment/artifacts"
    qualified = read_codecontests_s12_qualified_input(artifact_root / "codecontests_s12d/formal_20260905_3")
    selected = [row for row in qualified["rows"] if row["mutation_member"]]
    previous = artifact_root / "codecontests_s12/formal_20260905_1"
    support_path = previous / "CODECONTESTS_S12_SUPPORT_TABLE.csv"
    if not support_path.exists():
        support_path = previous / "delivery/CODECONTESTS_S12_SUPPORT_TABLE.csv"
    with support_path.open(encoding="utf-8", newline="") as handle:
        old_pc = {row["pair_id"] for row in csv.DictReader(handle) if row["mutant_qualified"] == "True"}
    if len(selected) != 350 or {row["triplet_id"] for row in selected} != old_pc:
        raise ValueError("SCOPE PC membership differs from the frozen 350")
    old_freeze = json.loads((previous / "CODECONTESTS_S12_EXECUTION_FREEZE.json").read_text("utf-8"))
    manifest_path = workspace / "v9/codecontests_f0a/probes/phase2_census/triplet_manifest.jsonl"
    manifest = read_codecontests_manifest(manifest_path)
    validate_triplet_registry(manifest)
    manifest = [row for row in manifest if row["triplet_id"] in old_pc]
    by_triplet = {row["triplet_id"]: row for row in manifest}
    data_rows = load_census_rows(workspace / "datasets/code_contests/802411c3010cb00d1b05bad57ca77365a3c699d6", manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        str(workspace / "models/Qwen2.5-1.5B/8faed761d45a263340a0528343f099c05c9a4323"),
        local_files_only=True, trust_remote_code=False)
    pairs = []
    for row in selected:
        triplet = by_triplet[row["triplet_id"]]
        problem = data_rows[_index(row["candidate_problem_id"])]
        description = problem["description"]
        if _fingerprint(description) != triplet["candidate_problem_fingerprint"]:
            raise ValueError("SCOPE statement identity differs")
        source_r = _solution(problem, row["selected_r_id"], label="correct", fingerprint=row["selected_r_fingerprint"])
        parts = row["selected_m_id"].split(":", 2)
        if len(parts) != 3 or parts[0] != "mutation":
            raise ValueError("SCOPE mutant locator is malformed")
        source_m = _mutated_source(source_r, {"mutable_ordinal": int(parts[1]), "source_sha256": parts[2]})
        mutation = generate_codecontests_mutants(source_r)[int(parts[1])]
        canonical_tests = freeze_codecontests_tests(triplet["candidate_problem_fingerprint"], problem)
        anchor = row["mutation_panel_anchor_test_id"] or row["i_witness_test_id"]
        if not isinstance(anchor, str):
            raise ValueError("SCOPE mutation panel anchor missing")
        family, ordinal = anchor.split(":")
        tests = build_s12_mutation_panel(canonical_tests, {"family": family, "ordinal": int(ordinal)})
        full_panel_id, full_test_ids = _panel_identity(canonical_tests)
        panel_id, test_ids = _panel_identity(tests)
        records = {}
        for role, selection_id, source, expected_panel in (
            ("R", row["selected_r_fingerprint"], source_r, full_panel_id),
            ("M", row["selected_m_id"], source_m, panel_id),
        ):
            matches = [item for item in row["role_objects"][role] if item["selection_id"] == selection_id]
            if len(matches) != 1:
                raise ValueError("SCOPE selected role evidence is ambiguous")
            item = matches[0]
            if item["panel_id"] != expected_panel or item["judged_payload_sha256"] != hashlib.sha256(build_judged_payload(source)).hexdigest():
                raise ValueError("SCOPE selected source/panel identity differs")
            dispositions = json.loads(row["execution_dispositions"])
            verdicts = [v for v in dispositions if v["judge_request_id"] == item["judge_request_id"]]
            expected_verdict = {"pass"} if role == "R" else {"wrong_answer", "runtime_failure", "timeout"}
            if len(verdicts) != 1 or verdicts[0]["execution_disposition"] != "semantic_verdict" or verdicts[0]["semantic_verdict"] not in expected_verdict:
                raise ValueError("SCOPE selected panel lacks valid semantic evidence")
            records[role] = item
        if not set(test_ids) <= set(full_test_ids):
            raise ValueError("SCOPE kill panel is not covered by R full-pass evidence")
        prompt = CODECONTESTS_PROMPT_PREFIX + description.replace("\r\n", "\n").replace("\r", "\n") + CODECONTESTS_PROMPT_SUFFIX
        static = qualify_scope_pc_pair(prompt, source_r, source_m, tokenizer)
        for role in ("R", "M"):
            ids = static["token_ids"][role]
            if len(ids) != row["serialized_tokens"][role] or len(ids) - static["prompt_tokens"] != row["response_tokens"][role]:
                raise ValueError("SCOPE token serialization differs from qualified source")
        pairs.append({"pair_id": row["triplet_id"], "problem_id": row["candidate_problem_id"],
                      "description": description, "prompt": prompt, "source_r": source_r, "source_m": source_m,
                      "provenance": row["r_provenance"], "failure_family": row["failure_family"],
                      "selected_r_id": row["selected_r_id"], "selected_m_id": row["selected_m_id"],
                      "mutation_family": f"{mutation['from_token']}->{mutation['to_token']}",
                      "tests": list(tests), "panel_id": panel_id, "panel_test_ids": list(test_ids),
                      "panel_evidence_verified": True, "r_evidence": "complete_pass_on_exact_superset_panel",
                      "source_evidence": records, **static})
    groups = group_scope_pc_pairs(pairs)
    from ..protocol import CODECONTESTS_FORMAL_ROOTS, CODECONTESTS_V2_FORMAL_ROOTS
    historical_roots = {int(job["root"]) for job in old_freeze["conditional_job_plan"]}
    historical_roots.update(CODECONTESTS_FORMAL_ROOTS)
    for registered in CODECONTESTS_V2_FORMAL_ROOTS.values():
        historical_roots.update(registered)
    return {"pairs": pairs, "groups": groups, "qualified_group_count": sum(g["representative_pair_id"] is not None for g in groups),
            "duplicate_mapping": "absent_in_declared_inputs", "authority_hashes": qualified["hashes"],
            "previous_support_sha256": sha256_file(support_path),
            "historical_roots": sorted(historical_roots),
            "manifest_sha256": sha256_file(manifest_path)}


def materialize_codecontests_s12c_closure(
    *,
    csr_table_path: Path,
    config_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    model_path: Path,
    historical_roots: Sequence[int] = (),
) -> dict[str, object]:
    """Exact-promote a frozen CSR table and dry-dispatch every 12/16 plan branch."""

    from transformers import AutoTokenizer
    from ..protocol import CODECONTESTS_FORMAL_JOB_ORDER, CODECONTESTS_V2_FORMAL_JOB_ORDERS
    from ..arms.paired_filtering import (
        build_s12_root_registry,
        build_s12_stage_plans,
        load_s12_config,
        materialize_s12c_promoted_pairs,
        s12_training_plan_sha256,
    )

    csr_rows = read_codecontests_s12c_csr_table(csr_table_path)
    manifest = read_codecontests_manifest(manifest_path)
    validate_triplet_registry(manifest)
    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    data_rows = load_census_rows(dataset_root, manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )

    def selected_source(row: Mapping[str, object], role: str) -> tuple[str, Mapping[str, object], Mapping[str, object], str]:
        triplet = by_triplet[str(row["triplet_id"])]
        candidate_row = data_rows[_index(triplet["candidate_problem_id"])]
        if role in {"I"}:
            identity = str(row["i_id"])
            source = _solution(
                candidate_row, identity, label="incorrect",
                fingerprint=triplet["candidate_solution_fingerprint"],
            )
            return source, triplet, candidate_row, "I"
        if role in {"R", "PC_CLEAN", "PC_MUTANT"}:
            identity = str(row["selected_r_id"])
            source = _solution(
                candidate_row, identity, label="correct",
                fingerprint=row["selected_r_fingerprint"],
            )
            if role == "PC_MUTANT":
                selected_m = str(row["selected_m_id"])
                parts = selected_m.split(":", 2)
                if len(parts) != 3 or parts[0] != "mutation" or not parts[1].isdigit():
                    raise ValueError("S12c promoted mutant identity is invalid")
                source = _mutated_source(
                    source,
                    {"mutable_ordinal": int(parts[1]), "source_sha256": parts[2]},
                )
            return source, triplet, candidate_row, role
        if role == "N":
            neutral_row = data_rows[_index(triplet["neutral_problem_id"])]
            source = _solution(
                neutral_row, row["n_id"], label="correct",
                fingerprint=triplet["neutral_solution_fingerprint"],
            )
            return source, triplet, neutral_row, "N"
        raise ValueError("S12c promoted role is invalid")

    def resolve(row: Mapping[str, object], role: str):
        source, triplet, problem_row, unit_role = selected_source(row, role)
        object_role = "M" if role == "PC_MUTANT" else "R" if role == "PC_CLEAN" else role
        expected_selection = row["selected_m_id"] if object_role == "M" else (
            row["selected_r_fingerprint"] if object_role == "R" else
            triplet["candidate_solution_fingerprint"] if object_role == "I" else
            triplet["neutral_solution_fingerprint"]
        )
        candidates = row["role_objects"].get(object_role, [])
        matched = [item for item in candidates if item.get("selection_id") == expected_selection]
        if len(matched) != 1 or matched[0].get("judged_payload_sha256") != hashlib.sha256(
            build_judged_payload(source)
        ).hexdigest():
            raise ValueError("S12c promoted role identity differs from CSR")
        expected_tokens = row["response_tokens"][object_role]
        return _tokenized_source_unit(
            triplet=triplet,
            row=problem_row,
            source=source,
            arm={"PC_CLEAN": "CLEAN", "PC_MUTANT": "MUTANT"}.get(unit_role, unit_role),
            prompt_role="neutral" if object_role == "N" else "candidate",
            tokenizer=tokenizer,
            expected_response_tokens=int(expected_tokens),
        )

    promoted = materialize_s12c_promoted_pairs(csr_rows, unit_resolver=resolve)
    config = load_s12_config(config_path)
    historical_roots = tuple(historical_roots) + tuple(root for root, _arm in CODECONTESTS_FORMAL_JOB_ORDER)
    for order in CODECONTESTS_V2_FORMAL_JOB_ORDERS.values():
        historical_roots += tuple(root for root, _arm in order)
    roots = build_s12_root_registry(config, historical_roots=historical_roots)
    dry_dispatch: dict[str, object] = {}
    for stage, pair_key in (("PC", "pc_pairs"), ("NAT", "nat_pairs"), ("POLICY", "policy_pairs")):
        pairs = promoted[pair_key]
        if not pairs:
            dry_dispatch[stage] = "not_evaluated"
            continue
        stage_records: dict[str, object] = {"planning": {}, "confirmatory_12": {}, "confirmatory_16": {}}
        for root in roots[stage]["planning"]:
            plans = build_s12_stage_plans(config, stage=stage, root=root, pairs=pairs)
            stage_records["planning"][str(root)] = {
                arm: s12_training_plan_sha256(plan) for arm, plan in plans.items()
            }
        confirmatory: dict[str, object] = {}
        for root in roots[stage]["confirmatory"]:
            plans = build_s12_stage_plans(config, stage=stage, root=root, pairs=pairs)
            confirmatory[str(root)] = {
                arm: s12_training_plan_sha256(plan) for arm, plan in plans.items()
            }
        stage_records["confirmatory_12"] = {
            str(root): confirmatory[str(root)] for root in roots[stage]["confirmatory"][:12]
        }
        stage_records["confirmatory_16"] = confirmatory
        dry_dispatch[stage] = stage_records
    support_rows = tuple(
        {
            "pair_id": row["triplet_id"],
            "problem_id": row["candidate_problem_id"],
            "response_tokens": row["response_tokens"],
            "serialized_tokens": row["serialized_tokens"],
        }
        for row in csr_rows
    )
    natural_rows = tuple(row for row, source in zip(support_rows, csr_rows) if source["natural_member"])
    pc_rows = tuple(row for row, source in zip(support_rows, csr_rows) if source["mutation_member"])
    policy_rows = tuple(row for row, source in zip(support_rows, csr_rows) if source["policy_member"])
    return {
        **promoted,
        "support_rows": support_rows,
        "root_registry": roots,
        "dry_dispatch": dry_dispatch,
        "nat_support": evaluate_s12_role_support(
            natural_rows, roles=("I", "R"), exposure_plan_closes=bool(promoted["nat_pairs"])
        ),
        "pc_support": evaluate_s12_role_support(
            pc_rows, roles=("R", "M"), exposure_plan_closes=bool(promoted["pc_pairs"])
        ),
        "policy_support": evaluate_s12_role_support(
            policy_rows, roles=("I", "R", "N"), exposure_plan_closes=bool(promoted["policy_pairs"])
        ),
        "csr_table_sha256": sha256_file(csr_table_path),
        "dry_dispatch_sha256": hashlib.sha256(
            json.dumps(dry_dispatch, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _fingerprint(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _index(identity: object, *, suffix: str | None = None) -> int:
    if not isinstance(identity, str):
        raise ValueError("CodeContests identity must be text")
    pieces = identity.split(":")
    if suffix is None:
        if len(pieces) != 2 or pieces[0] != "train":
            raise ValueError("CodeContests problem identity is invalid")
    elif len(pieces) != 4 or pieces[:1] != ["train"] or pieces[2] != suffix:
        raise ValueError("CodeContests solution identity is invalid")
    try:
        return int(pieces[-1] if suffix is not None else pieces[1])
    except ValueError as exc:
        raise ValueError("CodeContests identity ordinal is invalid") from exc


def validate_triplet_registry(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate exact focal/neutral identity closure without opening outcomes."""

    if len(rows) != 590:
        raise ValueError("CodeContests V2 requires exactly 590 triplets")
    triplets = [row.get("triplet_id") for row in rows]
    focal = [row.get("candidate_problem_id") for row in rows]
    neutral = [row.get("neutral_problem_id") for row in rows]
    if triplets != [f"triplet:{index}" for index in range(590)]:
        raise ValueError("CodeContests triplet order differs from freeze")
    if len(set(focal)) != 590 or len(set(neutral)) != 590:
        raise ValueError("CodeContests focal or neutral problem identity is reused")
    if set(focal) & set(neutral):
        raise ValueError("CodeContests focal and neutral sets overlap")
    if any(row.get("repair_problem_id") != row.get("candidate_problem_id") for row in rows):
        raise ValueError("CodeContests repair problem differs from focal problem")
    return {
        "triplet_count": 590,
        "unique_focal_problem_count": len(set(focal)),
        "unique_neutral_problem_count": len(set(neutral)),
        "cross_set_overlap_count": 0,
        "all_triplets_accounted_once": True,
    }


def load_census_rows(
    dataset_root: Path, manifest: Sequence[Mapping[str, object]]
) -> dict[int, dict[str, object]]:
    """Read only the focal/neutral rows and execution-contract columns."""

    import pyarrow.parquet as pq

    wanted = {
        _index(row[f"{role}_problem_id"])
        for row in manifest
        for role in ("candidate", "neutral")
    }
    paths = tuple(sorted((dataset_root / "data").glob("train-*.parquet")))
    if len(paths) != 39:
        raise ValueError("CodeContests train shard count differs from freeze")
    columns = [
        "description",
        "public_tests",
        "private_tests",
        "generated_tests",
        "solutions",
        "incorrect_solutions",
        "input_file",
        "output_file",
    ]
    result: dict[int, dict[str, object]] = {}
    offset = 0
    for path in paths:
        count = pq.ParquetFile(path).metadata.num_rows
        selected = tuple(index for index in wanted if offset <= index < offset + count)
        if selected:
            table = pq.read_table(path, columns=columns)
            for index in selected:
                result[index] = table.slice(index - offset, 1).to_pylist()[0]
        offset += count
    if offset != 13_328 or set(result) != wanted:
        raise ValueError("CodeContests V2 referenced row load is incomplete")
    return result


def _solution(
    row: Mapping[str, object], identity: object, *, label: str, fingerprint: object
) -> str:
    ordinal = _index(identity, suffix=label)
    field = "solutions" if label == "correct" else "incorrect_solutions"
    values = row.get(field)
    if not isinstance(values, Mapping):
        raise ValueError("CodeContests stored solution field is malformed")
    languages = values.get("language")
    solutions = values.get("solution")
    if (
        not isinstance(languages, (list, tuple))
        or not isinstance(solutions, (list, tuple))
        or ordinal >= len(languages)
        or ordinal >= len(solutions)
        or languages[ordinal] != 3
        or not isinstance(solutions[ordinal], str)
    ):
        raise ValueError("CodeContests stored Python solution is unavailable")
    source = solutions[ordinal]
    if _fingerprint(source) != fingerprint:
        raise ValueError("CodeContests stored solution fingerprint differs from freeze")
    return source


def _pair_task(payload: tuple[dict[str, object], dict[str, object]]) -> dict[str, object]:
    triplet, row = payload
    started = time.perf_counter()
    try:
        tests = freeze_codecontests_tests(
            str(triplet["candidate_problem_fingerprint"]), row
        )
    except ValueError as exc:
        reason_by_error = {
            "unsupported file-I/O problem": "unsupported_file_io_problem",
            "problem has no usable paired test": "no_usable_paired_test",
        }
        reason = reason_by_error.get(str(exc))
        if reason is None:
            raise
        record = ineligible_codecontests_pair(triplet, reason)
        record["serialized_fits"] = bool(
            int(triplet["candidate_prompt_tokens"])
            + int(triplet["candidate_response_tokens"])
            <= 2_048
            and int(triplet["repair_prompt_tokens"])
            + int(triplet["repair_response_tokens"])
            <= 2_048
        )
        record["test_contract"] = {"total": 0, "public": 0, "hidden": 0}
        record["worker_wall_seconds"] = time.perf_counter() - started
        record["raw_program_persisted"] = False
        record["raw_test_persisted"] = False
        return record
    repair_source = _solution(
        row,
        triplet["repair_solution_id"],
        label="correct",
        fingerprint=triplet["repair_solution_fingerprint"],
    )
    incorrect_source = _solution(
        row,
        triplet["candidate_solution_id"],
        label="incorrect",
        fingerprint=triplet["candidate_solution_fingerprint"],
    )
    for attempt in range(2):
        base = Path("/tmp") / f"ccv2_census_{os.getpid()}"
        repair = judge_codecontests_program(repair_source, tests, sandbox_base=base)
        incorrect = judge_codecontests_program(incorrect_source, tests, sandbox_base=base)
        if "runner_infrastructure_error" not in {
            repair.get("category"),
            incorrect.get("category"),
        }:
            break
    else:
        raise RuntimeError("repeated_runner_infrastructure_error")
    record = adjudicate_codecontests_pair(triplet, tests, repair, incorrect)
    record["serialized_fits"] = bool(
        int(triplet["candidate_prompt_tokens"])
        + int(triplet["candidate_response_tokens"])
        <= 2_048
        and int(triplet["repair_prompt_tokens"])
        + int(triplet["repair_response_tokens"])
        <= 2_048
    )
    record["test_contract"] = {
        "total": len(tests),
        "public": sum(test["family"] == "public" for test in tests),
        "hidden": sum(test["family"] in {"private", "generated"} for test in tests),
    }
    record["worker_wall_seconds"] = time.perf_counter() - started
    record["raw_program_persisted"] = False
    record["raw_test_persisted"] = False
    return record


def _mutation_panel(
    tests: Sequence[Mapping[str, object]], pair: Mapping[str, object]
) -> tuple[Mapping[str, object], ...]:
    hidden = [test for test in tests if test["family"] in {"private", "generated"}]
    witness = pair.get("hidden_witness")
    if isinstance(witness, Mapping):
        identity = (witness.get("test_family"), witness.get("test_ordinal"))
        for index, test in enumerate(hidden):
            if (test["family"], test["ordinal"]) == identity:
                hidden.insert(0, hidden.pop(index))
                break
    return tuple(hidden[:8])


def _background_mutation_task(
    payload: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]
) -> dict[str, object]:
    triplet, pair, candidate_row, neutral_row = payload
    started = time.perf_counter()
    neutral_tests = freeze_codecontests_tests(
        str(triplet["neutral_problem_fingerprint"]), neutral_row
    )
    neutral_source = _solution(
        neutral_row,
        triplet["neutral_solution_id"],
        label="correct",
        fingerprint=triplet["neutral_solution_fingerprint"],
    )
    candidate_tests = freeze_codecontests_tests(
        str(triplet["candidate_problem_fingerprint"]), candidate_row
    )
    panel = _mutation_panel(candidate_tests, pair)
    repair_source = _solution(
        candidate_row,
        triplet["repair_solution_id"],
        label="correct",
        fingerprint=triplet["repair_solution_fingerprint"],
    )
    for attempt in range(2):
        base = Path("/tmp") / f"ccv2_background_{os.getpid()}"
        neutral = judge_codecontests_program(
            neutral_source, neutral_tests, sandbox_base=base
        )
        if neutral.get("category") == "runner_infrastructure_error":
            continue
        if panel:
            try:
                mutation = select_first_killed_mutant(
                    repair_source,
                    panel,
                    judge=lambda source, tests: judge_codecontests_program(
                        str(source),
                        tests,
                        sandbox_base=base,
                        stop_on_first_failure=True,
                    ),
                )
            except RuntimeError:
                continue
        else:
            mutation = {
                "available": False,
                "attempted_mutants": 0,
                "compile_valid_mutants": 0,
            }
        break
    else:
        raise RuntimeError("repeated_runner_infrastructure_error")
    compact_mutation = {key: value for key, value in mutation.items() if key != "source"}
    return {
        "schema": "codecontests-v2-background-mutation-record-v1",
        "triplet_id": triplet["triplet_id"],
        "candidate_problem_id": triplet["candidate_problem_id"],
        "neutral_problem_id": triplet["neutral_problem_id"],
        "neutral_qualified": neutral.get("category") == "pass"
        and neutral.get("complete_contract") is True,
        "neutral_category": neutral.get("category"),
        "neutral_test_count": len(neutral_tests),
        "mutation_panel_count": len(panel),
        "mutation": compact_mutation,
        "worker_wall_seconds": time.perf_counter() - started,
        "raw_program_persisted": False,
        "raw_test_persisted": False,
    }


def _plan_closes(pairs: Sequence[Mapping[str, object]], *, branch: str) -> bool:
    from ..protocol import CODECONTESTS_V2_FORMAL_ROOTS
    from ..train.stream import TokenizedUnit, build_codecontests_v2_training_plan

    if not pairs:
        return False
    for arm in ("I", "R"):
        units = tuple(
            TokenizedUnit(
                unit_id=f"{pair['triplet_id']}/{arm}",
                token_ids=tuple(range(int(pair["response_tokens"][arm]) + 1)),
                loss_mask=(False,) + (True,) * int(pair["response_tokens"][arm]),
                sampling_id=str(pair["triplet_id"]),
            )
            for pair in pairs
        )
        for root in CODECONTESTS_V2_FORMAL_ROOTS["natural-certificate"]:
            plan = build_codecontests_v2_training_plan(
                units, stage="natural-certificate", root=root
            )
            if (
                len(plan.update_bins) != 100
                or sum(item.retained_response_tokens for item in plan.exposures)
                != 583_975
            ):
                return False
    return True


def _aligned_plan_closes(
    pairs: Sequence[Mapping[str, object]], *, arms: Sequence[str], stage: str
) -> bool:
    from ..protocol import CODECONTESTS_V2_FORMAL_ROOTS
    from ..train.stream import TokenizedUnit, build_codecontests_v2_training_plan

    if not pairs:
        return False
    for arm in arms:
        units = tuple(
            TokenizedUnit(
                unit_id=f"{pair['triplet_id']}/{arm}",
                token_ids=tuple(range(int(pair["response_tokens"][arm]) + 1)),
                loss_mask=(False,) + (True,) * int(pair["response_tokens"][arm]),
                sampling_id=str(pair["triplet_id"]),
            )
            for pair in pairs
        )
        for root in CODECONTESTS_V2_FORMAL_ROOTS[stage]:
            plan = build_codecontests_v2_training_plan(units, stage=stage, root=root)
            if len(plan.update_bins) != 100:
                return False
    return True


def _mutated_source(
    repair_source: str, mutation_record: Mapping[str, object]
) -> str:
    ordinal = mutation_record.get("mutable_ordinal")
    source_sha256 = mutation_record.get("source_sha256")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise ValueError("killed mutant ordinal is invalid")
    mutants = generate_codecontests_mutants(repair_source)
    if ordinal >= len(mutants) or mutants[ordinal].get("source_sha256") != source_sha256:
        raise ValueError("killed mutant replay identity differs")
    return str(mutants[ordinal]["source"])


def _support_sha256(identities: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()


def _tokenized_source_unit(
    *,
    triplet: Mapping[str, object],
    row: Mapping[str, object],
    source: str,
    arm: str,
    prompt_role: str,
    tokenizer,
    expected_response_tokens: int | None = None,
):
    from ..arms.codecontests import CODECONTESTS_PROMPT_PREFIX, CODECONTESTS_PROMPT_SUFFIX
    from ..train.stream import TokenizedUnit

    description = row.get("description")
    if not isinstance(description, str):
        raise ValueError("CodeContests description is malformed")
    problem_fingerprint = triplet[f"{prompt_role}_problem_fingerprint"]
    if _fingerprint(description) != problem_fingerprint:
        raise ValueError("CodeContests prompt fingerprint differs from freeze")
    prompt = (
        CODECONTESTS_PROMPT_PREFIX
        + description.replace("\r\n", "\n").replace("\r", "\n")
        + CODECONTESTS_PROMPT_SUFFIX
    )
    prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
    response_ids = tuple(tokenizer.encode(source, add_special_tokens=False)) + (
        int(tokenizer.eos_token_id),
    )
    if len(prompt_ids) != int(triplet[f"{prompt_role}_prompt_tokens"]):
        raise ValueError("CodeContests prompt serialization differs from freeze")
    if expected_response_tokens is not None and len(response_ids) != expected_response_tokens:
        raise ValueError("CodeContests response serialization differs from freeze")
    if len(prompt_ids) + len(response_ids) > 2_048:
        raise ValueError("CodeContests V2 serialized unit exceeds maximum length")
    return TokenizedUnit(
        unit_id=f"{triplet['triplet_id']}/{arm}",
        token_ids=prompt_ids + response_ids,
        loss_mask=(False,) * len(prompt_ids) + (True,) * len(response_ids),
        sampling_id=str(triplet["triplet_id"]),
    )


def materialize_v2_closure(
    *,
    manifest: Sequence[Mapping[str, object]],
    rows: Mapping[int, Mapping[str, object]],
    selection: Mapping[str, object],
    model_path: Path,
    output_dir: Path,
    v1_result_path: Path,
    d4_elapsed_seconds: float,
) -> dict[str, object]:
    """Freeze available run cards, selection, projection, and the D5 pause."""

    from transformers import AutoTokenizer

    from ..arms.codecontests import build_codecontests_v2_run_cards
    from ..protocol import CODECONTESTS_V2_FORMAL_ROOTS, CODECONTESTS_V2_STAGE_ARMS
    from ..train.stream import (
        build_codecontests_v2_policy_plans,
        build_codecontests_v2_training_plan,
        summarize_codecontests_v2_plan,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    mutation_by_id = {
        str(row["triplet_id"]): row
        for row in selection.get("mutation_records", [])
        if isinstance(row, Mapping)
    }
    stage_ids = {
        "positive-control": tuple(selection["mutation_support_triplet_ids"]),
        "natural-certificate": tuple(selection["natural_support_triplet_ids"]),
        "policy-filter": tuple(selection["policy_support_triplet_ids"]),
    }
    cards_root = output_dir / "run_cards"
    cards_root.mkdir(exist_ok=True)
    card_registry: dict[str, list[dict[str, object]]] = {}

    for stage in ("positive-control", "natural-certificate"):
        identities = stage_ids[stage]
        support_sha256 = _support_sha256(identities)
        units_by_arm: dict[str, tuple[object, ...]] = {}
        for formal_arm in CODECONTESTS_V2_STAGE_ARMS[stage]:
            units = []
            for triplet_id in identities:
                triplet = by_triplet[triplet_id]
                row = rows[_index(triplet["candidate_problem_id"])]
                if stage == "positive-control":
                    repair = _solution(
                        row,
                        triplet["repair_solution_id"],
                        label="correct",
                        fingerprint=triplet["repair_solution_fingerprint"],
                    )
                    source = (
                        repair
                        if formal_arm == "PC_CLEAN"
                        else _mutated_source(repair, mutation_by_id[triplet_id])
                    )
                    arm = "CLEAN" if formal_arm == "PC_CLEAN" else "MUTANT"
                    expected = (
                        int(triplet["repair_response_tokens"])
                        if formal_arm == "PC_CLEAN"
                        else int(mutation_by_id[triplet_id]["mutant_response_tokens"])
                    )
                else:
                    if formal_arm == "NAT_R":
                        source = _solution(
                            row,
                            triplet["repair_solution_id"],
                            label="correct",
                            fingerprint=triplet["repair_solution_fingerprint"],
                        )
                        arm = "R"
                        expected = int(triplet["repair_response_tokens"])
                    else:
                        source = _solution(
                            row,
                            triplet["candidate_solution_id"],
                            label="incorrect",
                            fingerprint=triplet["candidate_solution_fingerprint"],
                        )
                        arm = "I"
                        expected = int(triplet["candidate_response_tokens"])
                units.append(
                    _tokenized_source_unit(
                        triplet=triplet,
                        row=row,
                        source=source,
                        arm=arm,
                        prompt_role="candidate",
                        tokenizer=tokenizer,
                        expected_response_tokens=expected,
                    )
                )
            units_by_arm[formal_arm] = tuple(units)
        closures = {}
        for root in CODECONTESTS_V2_FORMAL_ROOTS[stage]:
            for formal_arm in CODECONTESTS_V2_STAGE_ARMS[stage]:
                plan = build_codecontests_v2_training_plan(
                    units_by_arm[formal_arm], stage=stage, root=root
                )
                closures[(root, formal_arm)] = summarize_codecontests_v2_plan(
                    plan, support_sha256=support_sha256
                )
        cards = build_codecontests_v2_run_cards(stage, closures)
        stage_dir = cards_root / stage
        stage_dir.mkdir(exist_ok=True)
        card_registry[stage] = []
        for card in cards:
            path = stage_dir / (
                f"{int(card['training_ordinal']):02d}_{str(card['arm']).lower()}_"
                f"{card['root']}.json"
            )
            _atomic_json(path, card)
            card_registry[stage].append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    if selection.get("policy_available") is True:
        stage = "policy-filter"
        identities = stage_ids[stage]
        support_sha256 = _support_sha256(identities)
        background = []
        incorrect = []
        repair = []
        for triplet_id in identities:
            triplet = by_triplet[triplet_id]
            candidate_row = rows[_index(triplet["candidate_problem_id"])]
            neutral_row = rows[_index(triplet["neutral_problem_id"])]
            incorrect_source = _solution(
                candidate_row,
                triplet["candidate_solution_id"],
                label="incorrect",
                fingerprint=triplet["candidate_solution_fingerprint"],
            )
            repair_source = _solution(
                candidate_row,
                triplet["repair_solution_id"],
                label="correct",
                fingerprint=triplet["repair_solution_fingerprint"],
            )
            neutral_source = _solution(
                neutral_row,
                triplet["neutral_solution_id"],
                label="correct",
                fingerprint=triplet["neutral_solution_fingerprint"],
            )
            incorrect.append(
                _tokenized_source_unit(
                    triplet=triplet,
                    row=candidate_row,
                    source=incorrect_source,
                    arm="I",
                    prompt_role="candidate",
                    tokenizer=tokenizer,
                    expected_response_tokens=int(triplet["candidate_response_tokens"]),
                )
            )
            repair.append(
                _tokenized_source_unit(
                    triplet=triplet,
                    row=candidate_row,
                    source=repair_source,
                    arm="R",
                    prompt_role="candidate",
                    tokenizer=tokenizer,
                    expected_response_tokens=int(triplet["repair_response_tokens"]),
                )
            )
            background.append(
                _tokenized_source_unit(
                    triplet=triplet,
                    row=neutral_row,
                    source=neutral_source,
                    arm="N",
                    prompt_role="neutral",
                    tokenizer=tokenizer,
                    expected_response_tokens=int(triplet["neutral_response_tokens"]),
                )
            )
        closures = {}
        for root in CODECONTESTS_V2_FORMAL_ROOTS[stage]:
            plans = build_codecontests_v2_policy_plans(
                tuple(background), tuple(incorrect), tuple(repair), root=root
            )
            for arm, plan in plans.items():
                closures[(root, arm)] = summarize_codecontests_v2_plan(
                    plan, support_sha256=support_sha256
                )
        cards = build_codecontests_v2_run_cards(stage, closures)
        stage_dir = cards_root / stage
        stage_dir.mkdir(exist_ok=True)
        card_registry[stage] = []
        for card in cards:
            path = stage_dir / (
                f"{int(card['training_ordinal']):02d}_{str(card['arm']).lower()}_"
                f"{card['root']}.json"
            )
            _atomic_json(path, card)
            card_registry[stage].append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    v1 = json.loads(v1_result_path.read_text("utf-8"))
    start = datetime.fromisoformat(str(v1["formal_started_utc"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(v1["completed_utc"]).replace("Z", "+00:00"))
    v1_hours = (end - start).total_seconds() / 3600
    jobs = sum(len(values) for values in card_registry.values())
    projection_hours = v1_hours * jobs / 24 * 1.5
    admission_hours = 12.0
    outer_hours = d4_elapsed_seconds / 3600 + projection_hours + admission_hours
    projection = {
        "schema": "codecontests-v2-worst-case-projection-v1",
        "v1_observed_chain_hours": v1_hours,
        "v1_job_count": 24,
        "v2_available_job_count": jobs,
        "reserve_multiplier": 1.5,
        "formal_projection_hours": projection_hours,
        "d4_active_elapsed_hours": d4_elapsed_seconds / 3600,
        "post_authorization_admission_allowance_hours": admission_hours,
        "outer_projected_hours": outer_hours,
        "outer_maximum_hours": 160.0,
        "passed": outer_hours <= 160.0,
    }
    if not projection["passed"]:
        raise RuntimeError("CodeContests V2 worst-case wall projection exceeds 160 hours")
    selection_freeze = {
        "schema": "codecontests-v2-selection-freeze-v1",
        "created_utc": _utc(),
        "positive_control_triplet_ids": list(stage_ids["positive-control"]),
        "natural_support_triplet_ids": list(stage_ids["natural-certificate"]),
        "policy_support_triplet_ids": list(stage_ids["policy-filter"]),
        "policy_available": selection["policy_available"],
        "mutation_records": selection["mutation_records"],
        "run_cards": card_registry,
        "projection": projection,
        "outcomes_opened": False,
        "raw_programs_persisted": False,
        "raw_tests_persisted": False,
    }
    selection_path = output_dir / "CODECONTESTS_V2_SELECTION_FREEZE.json"
    _atomic_json(selection_path, selection_freeze)
    readiness = {
        "schema": "codecontests-v2-readiness-v1",
        "created_utc": _utc(),
        "state": "AWAITING_USER_CODECONTESTS_V2_TRAINING_AUTHORIZATION",
        "selection_freeze_sha256": sha256_file(selection_path),
        "policy_available": selection["policy_available"],
        "c_train": 2,
        "c_gen": 4,
        "c_val": 4,
        "launch_headroom_fraction": 0.03,
        "available_stage_job_counts": {
            stage: len(values) for stage, values in card_registry.items()
        },
        "projection": projection,
        "formal_gpu_started": False,
        "outcomes_opened": False,
    }
    readiness["identity_sha256"] = hashlib.sha256(
        json.dumps(readiness, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    readiness_path = output_dir / "CODECONTESTS_V2_READINESS.json"
    _atomic_json(readiness_path, readiness)
    return {
        "selection_freeze": {
            "path": str(selection_path),
            "bytes": selection_path.stat().st_size,
            "sha256": sha256_file(selection_path),
        },
        "readiness": {
            "path": str(readiness_path),
            "bytes": readiness_path.stat().st_size,
            "sha256": sha256_file(readiness_path),
        },
        "card_registry": card_registry,
        "projection": projection,
    }


def run_background_mutation(
    *,
    manifest: Sequence[Mapping[str, object]],
    rows: Mapping[int, Mapping[str, object]],
    ir_records: Sequence[Mapping[str, object]],
    selected_triplet_ids: Sequence[str],
    model_path: Path,
    output_dir: Path,
    workers: int,
    started: float,
    wall_limit_seconds: float,
) -> dict[str, object]:
    """Execute/resume selected N qualification and the finite mutation worklist."""

    from transformers import AutoTokenizer

    by_triplet = {str(row["triplet_id"]): row for row in manifest}
    by_pair = {str(row["triplet_id"]): row for row in ir_records}
    worklist = tuple(
        sorted(
            selected_triplet_ids,
            key=lambda triplet_id: hashlib.sha256(
                (
                    "codecontests-certificate-calibrated-filtering-v2|"
                    f"positive-control-worklist|{triplet_id}"
                ).encode("utf-8")
            ).digest(),
        )
    )
    checkpoints = output_dir / "checkpoints_background_mutation"
    checkpoints.mkdir(exist_ok=True)
    completed: dict[str, dict[str, object]] = {}
    for path in sorted(checkpoints.glob("triplet_*.json")):
        value = json.loads(path.read_text("utf-8"))
        triplet_id = value.get("triplet_id")
        if triplet_id not in worklist or triplet_id in completed:
            raise ValueError("background/mutation checkpoint identity is invalid")
        completed[str(triplet_id)] = value
    pending = []
    for triplet_id in worklist:
        if triplet_id in completed:
            continue
        triplet = by_triplet[triplet_id]
        pending.append(
            (
                dict(triplet),
                dict(by_pair[triplet_id]),
                dict(rows[_index(triplet["candidate_problem_id"])]),
                dict(rows[_index(triplet["neutral_problem_id"])]),
            )
        )

    def progress(status: str) -> None:
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema": "codecontests-v2-census-progress-v1",
                "updated_utc": _utc(),
                "status": status,
                "phase": "background_mutation",
                "completed": len(completed),
                "total": len(worklist),
                "workers": workers,
                "active_elapsed_seconds": time.monotonic() - started,
            },
        )

    progress("RUNNING_CODECONTESTS_V2_BACKGROUND_MUTATION")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_background_mutation_task, payload): payload[0]["triplet_id"]
                for payload in pending
            }
            for future in as_completed(futures):
                if time.monotonic() - started > wall_limit_seconds:
                    raise TimeoutError("CodeContests V2 census wall limit exceeded")
                triplet_id = str(futures[future])
                record = future.result()
                completed[triplet_id] = record
                ordinal = int(triplet_id.split(":", 1)[1])
                _atomic_json(checkpoints / f"triplet_{ordinal:03d}.json", record)
                progress("RUNNING_CODECONTESTS_V2_BACKGROUND_MUTATION")
    ordered = tuple(completed[triplet_id] for triplet_id in worklist)
    _write_jsonl(output_dir / "BACKGROUND_MUTATION_RECORDS.jsonl", ordered)

    selected_pairs = tuple(by_pair[triplet_id] for triplet_id in worklist)
    neutral_pairs = tuple(
        {
            **by_pair[triplet_id],
            "response_tokens": {
                **dict(by_pair[triplet_id]["response_tokens"]),
                "N": by_triplet[triplet_id]["neutral_response_tokens"],
            },
            "serialized_fits": bool(by_pair[triplet_id]["serialized_fits"])
            and int(by_triplet[triplet_id]["neutral_prompt_tokens"])
            + int(by_triplet[triplet_id]["neutral_response_tokens"])
            <= 2_048,
        }
        for triplet_id in worklist
        if completed[triplet_id]["neutral_qualified"] is True
    )
    neutral_plan_closes = _aligned_plan_closes(
        neutral_pairs,
        arms=("I", "R", "N"),
        stage="natural-certificate",
    )
    alignment = align_codecontests_natural_support(
        selected_pairs, neutral_pairs, plan_closes=neutral_plan_closes
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    mutation_records = []
    for triplet_id in worklist:
        work = completed[triplet_id]
        mutation = work["mutation"]
        if not isinstance(mutation, Mapping) or mutation.get("available") is not True:
            continue
        triplet = by_triplet[triplet_id]
        candidate_row = rows[_index(triplet["candidate_problem_id"])]
        repair_source = _solution(
            candidate_row,
            triplet["repair_solution_id"],
            label="correct",
            fingerprint=triplet["repair_solution_fingerprint"],
        )
        mutant_source = _mutated_source(repair_source, mutation)
        mutant_tokens = len(tokenizer.encode(mutant_source, add_special_tokens=False)) + 1
        mutation_records.append(
            {
                "triplet_id": triplet_id,
                "candidate_problem_id": triplet["candidate_problem_id"],
                "clean_response_tokens": triplet["repair_response_tokens"],
                "mutant_response_tokens": mutant_tokens,
                "serialized_fits": int(triplet["candidate_prompt_tokens"])
                + mutant_tokens
                <= 2_048,
                **dict(mutation),
            }
        )
    natural_support_ids = {
        str(pair["triplet_id"]) for pair in alignment["natural_support"]
    }
    mutation_intersection_rows = tuple(
        row for row in mutation_records if row["triplet_id"] in natural_support_ids
    )
    pc_plan_closes = _aligned_plan_closes(
        tuple(
            {
                "triplet_id": row["triplet_id"],
                "candidate_problem_id": row["candidate_problem_id"],
                "response_tokens": {
                    "CLEAN": row["clean_response_tokens"],
                    "MUTANT": row["mutant_response_tokens"],
                },
            }
            for row in mutation_intersection_rows
        ),
        arms=("CLEAN", "MUTANT"),
        stage="positive-control",
    )
    mutation_support = intersect_codecontests_mutations(
        alignment["natural_support"],
        mutation_records,
        worklist_complete=True,
        plan_closes=pc_plan_closes,
    )
    result = {
        "schema": "codecontests-v2-background-mutation-result-v1",
        "created_utc": _utc(),
        "status": mutation_support["terminal"]
        or "COMPLETED_CODECONTESTS_V2_SELECTION",
        "worklist_order": list(worklist),
        "worklist_count": len(worklist),
        "neutral_qualified_count": len(neutral_pairs),
        "killed_mutant_count": len(mutation_records),
        "natural_support_triplet_ids": [
            pair["triplet_id"] for pair in alignment["natural_support"]
        ],
        "policy_support_triplet_ids": [
            pair["triplet_id"] for pair in alignment["policy_support"]
        ],
        "policy_available": alignment["policy_available"],
        "neutral_support_gate": alignment["neutral_support_gate"],
        "mutation_support_triplet_ids": [
            row["triplet_id"]
            for row in mutation_support["mutation_support"]
        ]
        if mutation_support["mutation_support"] != "not_evaluated"
        else "not_evaluated",
        "mutation_support_gate": mutation_support["support"],
        "mutation_records": mutation_records,
        "active_elapsed_seconds": time.monotonic() - started,
        "raw_programs_persisted": False,
        "raw_tests_persisted": False,
    }
    _atomic_json(output_dir / "BACKGROUND_MUTATION_RESULT.json", result)
    progress(str(result["status"]))
    return result


def run_ir_census(
    *,
    manifest_path: Path,
    endpoint_path: Path,
    dataset_root: Path,
    model_path: Path,
    v1_result_path: Path,
    output_dir: Path,
    workers: int = 4,
    wall_limit_seconds: float = 24 * 60 * 60,
    d4_started_utc: str | None = None,
) -> dict[str, object]:
    """Execute or resume all 590 I/R contracts and freeze the F2 branch."""

    invocation_created_utc = d4_started_utc or _utc()
    if d4_started_utc is not None:
        if not d4_started_utc.endswith("Z"):
            raise ValueError("D4 cumulative start must be UTC")
        parsed_d4_start = datetime.fromisoformat(d4_started_utc.replace("Z", "+00:00"))
        if parsed_d4_start > datetime.now(timezone.utc):
            raise ValueError("D4 cumulative start cannot be in the future")
    invocation_started = time.monotonic()
    if workers != 4:
        raise ValueError("formal CodeContests V2 census concurrency is fixed at four")
    if sha256_file(endpoint_path) != V1_ENDPOINT_SHA256:
        raise ValueError("sealed V1 endpoint SHA-256 mismatch")
    manifest = read_codecontests_manifest(manifest_path)
    registry = validate_triplet_registry(manifest)
    rows = load_census_rows(dataset_root, manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints_ir"
    checkpoints.mkdir(exist_ok=True)
    freeze = {
        "schema": SCHEMA,
        "created_utc": invocation_created_utc,
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "sealed_endpoint": {
            "path": str(endpoint_path),
            "bytes": endpoint_path.stat().st_size,
            "sha256": sha256_file(endpoint_path),
        },
        "dataset_root": str(dataset_root),
        "tokenizer_root": str(model_path),
        "v1_timing_source": {
            "path": str(v1_result_path),
            "bytes": v1_result_path.stat().st_size,
            "sha256": sha256_file(v1_result_path),
        },
        "workers": workers,
        "wall_limit_seconds": wall_limit_seconds,
        "registry": registry,
        "raw_programs_persisted": False,
        "raw_tests_persisted": False,
    }
    freeze_path = output_dir / "CENSUS_EXECUTION_FREEZE.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text("utf-8"))
        comparable = {key: value for key, value in freeze.items() if key != "created_utc"}
        prior_comparable = {key: value for key, value in previous.items() if key != "created_utc"}
        if comparable != prior_comparable:
            raise ValueError("CodeContests V2 census resume freeze differs")
        freeze = previous
    else:
        _atomic_json(freeze_path, freeze)
    frozen_started = datetime.fromisoformat(
        str(freeze["created_utc"]).replace("Z", "+00:00")
    )
    carried_elapsed = max(
        0.0, (datetime.now(timezone.utc) - frozen_started).total_seconds()
    )
    # Preserve one cumulative D4 clock across exact checkpoint resumes.
    started = invocation_started - carried_elapsed

    completed: dict[str, dict[str, object]] = {}
    for path in sorted(checkpoints.glob("triplet_*.json")):
        value = json.loads(path.read_text("utf-8"))
        triplet_id = value.get("triplet_id")
        if not isinstance(triplet_id, str) or triplet_id in completed:
            raise ValueError("CodeContests V2 census checkpoint identity is invalid")
        completed[triplet_id] = value
    pending = [
        (dict(triplet), rows[_index(triplet["candidate_problem_id"])])
        for triplet in manifest
        if triplet["triplet_id"] not in completed
    ]

    def progress(status: str) -> None:
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema": "codecontests-v2-census-progress-v1",
                "updated_utc": _utc(),
                "status": status,
                "phase": "ir_census",
                "completed": len(completed),
                "total": 590,
                "workers": workers,
                "active_elapsed_seconds": time.monotonic() - started,
            },
        )

    progress("RUNNING_CODECONTESTS_V2_CENSUS")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_pair_task, payload): payload[0]["triplet_id"] for payload in pending}
            for future in as_completed(futures):
                if time.monotonic() - started > wall_limit_seconds:
                    raise TimeoutError("CodeContests V2 census wall limit exceeded")
                triplet_id = str(futures[future])
                record = future.result()
                if record.get("triplet_id") != triplet_id:
                    raise ValueError("CodeContests V2 worker returned the wrong triplet")
                completed[triplet_id] = record
                ordinal = int(triplet_id.split(":", 1)[1])
                _atomic_json(checkpoints / f"triplet_{ordinal:03d}.json", record)
                progress("RUNNING_CODECONTESTS_V2_CENSUS")
    ordered = tuple(completed[f"triplet:{index}"] for index in range(590))
    _write_jsonl(output_dir / "IR_CENSUS_RECORDS.jsonl", ordered)
    certified = tuple(row for row in ordered if row.get("certificate_discrepant") is True)
    high = tuple(row for row in certified if row.get("high_severity") is True)
    selection = select_codecontests_f2(
        high,
        certified,
        plan_closes={
            "high_severity": _plan_closes(high, branch="high_severity"),
            "pooled_certified": _plan_closes(certified, branch="pooled_certified"),
        },
    )
    result = {
        "schema": "codecontests-v2-ir-census-result-v1",
        "created_utc": _utc(),
        "status": selection["terminal"] or "COMPLETED_CODECONTESTS_V2_IR_CENSUS",
        "registry": registry,
        "counts": {
            "total": 590,
            "evaluator_qualified": sum(row.get("evaluator_qualified") is True for row in ordered),
            "certificate_discrepant": len(certified),
            "high_severity": len(high),
            "classifications": dict(Counter(str(row.get("classification")) for row in ordered)),
            "ineligible_reasons": dict(
                Counter(
                    str(row.get("ineligible_reason"))
                    for row in ordered
                    if row.get("ineligible_reason") is not None
                )
            ),
        },
        "test_contract": {
            "total_executions": sum(int(row["test_contract"]["total"]) * 2 for row in ordered),
            "public_tests": sum(int(row["test_contract"]["public"]) for row in ordered),
            "hidden_tests": sum(int(row["test_contract"]["hidden"]) for row in ordered),
        },
        "selected_branch": selection["selected_branch"],
        "selected_triplet_ids": [row["triplet_id"] for row in selection["selected_pairs"]],
        "high_severity_support": selection["high_severity_support"],
        "pooled_certified_support": selection["pooled_certified_support"],
        "active_elapsed_seconds": time.monotonic() - started,
        "workers": workers,
        "raw_programs_persisted": False,
        "raw_tests_persisted": False,
    }
    if selection["terminal"] is not None:
        final = {
            **result,
            "schema": "codecontests-v2-cpu-census-result-v1",
            "selection_freeze": "not_evaluated",
            "background_mutation": "not_evaluated",
            "formal_execution": "not_evaluated",
        }
        _atomic_json(output_dir / "CODECONTESTS_V2_CENSUS_RESULT.json", final)
        (output_dir / "IR_CENSUS_RESULT.json").unlink(missing_ok=True)
        progress(str(final["status"]))
        return final
    _atomic_json(output_dir / "IR_CENSUS_RESULT.json", result)
    background = run_background_mutation(
        manifest=manifest,
        rows=rows,
        ir_records=ordered,
        selected_triplet_ids=result["selected_triplet_ids"],
        model_path=model_path,
        output_dir=output_dir,
        workers=workers,
        started=started,
        wall_limit_seconds=wall_limit_seconds,
    )
    final = {
        **result,
        "schema": "codecontests-v2-cpu-census-result-v1",
        "status": background["status"],
        "background_mutation": background,
        "active_elapsed_seconds": time.monotonic() - started,
    }
    if background["status"] == "COMPLETED_CODECONTESTS_V2_SELECTION":
        final["closure"] = materialize_v2_closure(
            manifest=manifest,
            rows=rows,
            selection=background,
            model_path=model_path,
            output_dir=output_dir,
            v1_result_path=v1_result_path,
            d4_elapsed_seconds=time.monotonic() - started,
        )
        final["status"] = "AWAITING_USER_CODECONTESTS_V2_TRAINING_AUTHORIZATION"
    _atomic_json(output_dir / "CODECONTESTS_V2_CENSUS_RESULT.json", final)
    return final


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wall-limit-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument("--d4-started-utc")
    arguments = parser.parse_args()
    result = run_ir_census(
        manifest_path=arguments.manifest,
        endpoint_path=arguments.endpoint,
        dataset_root=arguments.dataset_root,
        model_path=arguments.model,
        v1_result_path=arguments.v1_result,
        output_dir=arguments.output,
        workers=arguments.workers,
        wall_limit_seconds=arguments.wall_limit_seconds,
        d4_started_utc=arguments.d4_started_utc,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "selected_branch": result["selected_branch"],
                "selected_count": len(result["selected_triplet_ids"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
