"""CodeContests V2 certificate, selection, mutation, and background census."""

from __future__ import annotations

import hashlib
import io
import json
import token
import tokenize
import unicodedata
from typing import Callable, Mapping, Sequence

from ..protocol import (
    CODECONTESTS_V3_MAX_CANDIDATES,
    CODECONTESTS_V3_MAX_SERIALIZED_TOKENS,
    CODECONTESTS_V3A_CANDIDATE_PRIORITY_NAMESPACE,
)


class CodeContestsS12DInputInvalid(ValueError):
    """A frozen S12d row cannot be reconstructed under the input contract."""


class CodeContestsS12DContractInvalid(RuntimeError):
    """Valid input reached an unknown S12d implementation state."""


CODECONTESTS_S12D_EXPECTED_EXCLUSIONS = frozenset(
    {"unsupported_file_io_problem", "no_usable_paired_test"}
)


def _identity_hash(namespace: str, value: object) -> str:
    payload = json.dumps(
        {"namespace": namespace, "value": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _panel_identity(panel: Sequence[Mapping[str, object]]) -> tuple[str, tuple[str, ...]]:
    if not panel or isinstance(panel, (str, bytes)):
        raise ValueError("CodeContests executable panel must be nonempty")
    frozen: list[dict[str, object]] = []
    test_ids: list[str] = []
    seen: set[str] = set()
    for test in panel:
        if not isinstance(test, Mapping):
            raise ValueError("CodeContests executable panel test is malformed")
        family = test.get("family")
        ordinal = test.get("ordinal")
        stdin = test.get("input")
        expected = test.get("expected_output")
        if (
            family not in _TEST_FAMILIES
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or not isinstance(stdin, str)
            or not isinstance(expected, str)
        ):
            raise ValueError("CodeContests executable panel test is malformed")
        test_id = f"{family}:{ordinal}"
        if test_id in seen:
            raise ValueError("CodeContests executable panel has a duplicate test")
        seen.add(test_id)
        test_ids.append(test_id)
        frozen.append(
            {
                "test_id": test_id,
                "input_sha256": hashlib.sha256(
                    stdin.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
                ).hexdigest(),
                "expected_output_sha256": hashlib.sha256(
                    expected.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
                ).hexdigest(),
            }
        )
    return _identity_hash("codecontests-s12c-panel-v1", frozen), tuple(test_ids)


def build_codecontests_executable_object(
    *,
    judged_payload: bytes,
    source_locator: str,
    role: str,
    selection_id: str,
    panel: Sequence[Mapping[str, object]],
    measurement_device_id: str,
) -> dict[str, object]:
    """Bind one stored source locator to its exact executable judge request."""

    if role not in {"I", "R", "M", "N", "endpoint_reference"}:
        raise ValueError("CodeContests executable role is invalid")
    if any(not isinstance(value, str) or not value for value in (
        source_locator, selection_id, measurement_device_id
    )):
        raise ValueError("CodeContests executable identity component is invalid")
    if not isinstance(judged_payload, bytes) or not judged_payload:
        raise ValueError("CodeContests judged payload must be nonempty bytes")
    payload_sha256 = hashlib.sha256(judged_payload).hexdigest()
    panel_id, panel_test_ids = _panel_identity(panel)
    logical_object_id = _identity_hash(
        "codecontests-s12c-logical-object-v1",
        {
            "source_locator": source_locator,
            "role": role,
            "selection_id": selection_id,
            "panel_id": panel_id,
        },
    )
    request_id = _identity_hash(
        "codecontests-s12c-judge-request-v1",
        {
            "logical_object_id": logical_object_id,
            "judged_payload_sha256": payload_sha256,
            "panel_id": panel_id,
            "measurement_device_id": measurement_device_id,
        },
    )
    return {
        "logical_object_id": logical_object_id,
        "judge_request_id": request_id,
        "source_locator": source_locator,
        "role": role,
        "selection_id": selection_id,
        "judged_payload": judged_payload,
        "judged_payload_sha256": payload_sha256,
        "panel_id": panel_id,
        "panel_test_ids": panel_test_ids,
        "measurement_device_id": measurement_device_id,
    }


def summarize_codecontests_keyed_relation(
    objects: Sequence[Mapping[str, object]],
    verdicts: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate and summarize verdicts strictly by logical object ID."""

    if isinstance(verdicts, Mapping):
        verdict_map = dict(verdicts)
    else:
        verdict_map: dict[str, Mapping[str, object]] = {}
        for item in verdicts:
            object_id = item.get("logical_object_id") if isinstance(item, Mapping) else None
            verdict = item.get("verdict") if isinstance(item, Mapping) else None
            if not isinstance(object_id, str) or not isinstance(verdict, Mapping):
                raise ValueError("CodeContests keyed verdict is malformed")
            if object_id in verdict_map:
                raise ValueError("CodeContests relation has a duplicate verdict key")
            verdict_map[object_id] = verdict
    object_by_id: dict[str, Mapping[str, object]] = {}
    duplicate_ids: list[str] = []
    for value in objects:
        object_id = value.get("logical_object_id") if isinstance(value, Mapping) else None
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("CodeContests relation object ID is invalid")
        if object_id in object_by_id:
            duplicate_ids.append(object_id)
        else:
            object_by_id[object_id] = value
    if duplicate_ids:
        raise ValueError("CodeContests relation has duplicate object IDs")
    orphan_ids = sorted(set(verdict_map) - set(object_by_id))
    missing_ids = [value["logical_object_id"] for value in objects if value["logical_object_id"] not in verdict_map]
    records: list[dict[str, object]] = []
    semantic_counts = {key: 0 for key in ("pass", "wrong_answer", "compile_failure", "runtime_failure", "timeout")}
    infrastructure_ids: list[str] = []
    for value in objects:
        object_id = str(value["logical_object_id"])
        verdict = verdict_map.get(object_id)
        category = verdict.get("category") if isinstance(verdict, Mapping) else None
        witness_test_id: str | None = None
        anchor_test_id: str | None = None
        failure_family: str | None = None
        if category == "runner_infrastructure_error":
            infrastructure_ids.append(object_id)
        elif category in semantic_counts:
            semantic_counts[str(category)] += 1
            if category in {"wrong_answer", "runtime_failure", "timeout"}:
                rows = verdict.get("test_verdicts")
                first = next(
                    (
                        row for row in rows
                        if isinstance(row, Mapping) and row.get("category") != "pass"
                    ),
                    None,
                ) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else None
                if not isinstance(first, Mapping):
                    raise ValueError("CodeContests semantic nonpass lacks a typed witness")
                witness_test_id = f"{first.get('family')}:{first.get('ordinal')}"
                if witness_test_id not in value.get("panel_test_ids", ()) or first.get("category") != category:
                    raise ValueError("CodeContests typed witness differs from panel verdict")
            elif category == "compile_failure":
                panel_ids = value.get("panel_test_ids")
                anchor_test_id = panel_ids[0] if isinstance(panel_ids, Sequence) and panel_ids else None
            failure_family = {
                "wrong_answer": "output_mismatch",
                "compile_failure": "execution_failure",
                "runtime_failure": "execution_failure",
                "timeout": "timeout",
            }.get(str(category))
        elif category is not None:
            raise ValueError("CodeContests verdict category is invalid")
        records.append(
            {
                "logical_object_id": object_id,
                "role": value.get("role"),
                "category": category,
                "witness_test_id": witness_test_id,
                "mutation_panel_anchor_test_id": anchor_test_id,
                "failure_family": failure_family,
            }
        )
    return {
        "schema": "codecontests-s12c-keyed-relation-summary-v1",
        "records": records,
        "reached_object_count": len(objects),
        "verdict_count": len(verdict_map),
        "semantic_counts": semantic_counts,
        "missing_object_ids": missing_ids,
        "orphan_object_ids": orphan_ids,
        "infrastructure_object_ids": infrastructure_ids,
        "passed": not missing_ids and not orphan_ids and not infrastructure_ids,
    }


_TEST_FAMILIES = ("public", "private", "generated")
_HIDDEN_FAMILIES = frozenset({"private", "generated"})
_JUDGE_CATEGORIES = frozenset(
    {
        "pass",
        "wrong_answer",
        "compile_failure",
        "runtime_failure",
        "timeout",
        "runner_infrastructure_error",
    }
)


def _lf(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("CodeContests test payload must be text")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_codecontests_source_fingerprint(value: str) -> str:
    """Return the exact V2 source identity reused by V3."""

    if not isinstance(value, str):
        raise ValueError("CodeContests source must be text")
    normalized = unicodedata.normalize("NFC", _lf(value)).strip()
    return _sha256(normalized)


def codecontests_v3_candidate_priority(
    candidate_problem_id: str, solution_fingerprint: str
) -> str:
    """Return the frozen outcome-blind alternative-R ordering key."""

    if not isinstance(candidate_problem_id, str) or not candidate_problem_id:
        raise ValueError("candidate problem ID is invalid")
    if not isinstance(solution_fingerprint, str) or len(solution_fingerprint) != 64:
        raise ValueError("solution fingerprint is invalid")
    return _sha256(
        f"{CODECONTESTS_V3A_CANDIDATE_PRIORITY_NAMESPACE}|{candidate_problem_id}|{solution_fingerprint}"
    )


def codecontests_v3a_candidate_priority(
    candidate_problem_id: str, solution_fingerprint: str
) -> str:
    """Expose V3a's deliberate reuse of the frozen V3 ordering namespace."""

    return codecontests_v3_candidate_priority(
        candidate_problem_id, solution_fingerprint
    )


def _stored_solution_ordinal(identity: object, *, label: str) -> int:
    if not isinstance(identity, str):
        raise ValueError("CodeContests stored solution identity is invalid")
    parts = identity.rsplit(":", 2)
    if len(parts) != 3 or parts[1] != label or not parts[2].isdigit():
        raise ValueError("CodeContests stored solution identity is invalid")
    return int(parts[2])


def _measured_tokens(
    source: str, measure_tokens: Callable[[str], Mapping[str, object]]
) -> tuple[int, int]:
    measured = measure_tokens(source)
    serialized = measured.get("serialized_tokens") if isinstance(measured, Mapping) else None
    response = measured.get("response_tokens") if isinstance(measured, Mapping) else None
    if (
        isinstance(serialized, bool)
        or not isinstance(serialized, int)
        or serialized <= 0
        or isinstance(response, bool)
        or not isinstance(response, int)
        or response <= 0
    ):
        raise ValueError("CodeContests tokenizer measurement is invalid")
    return serialized, response


def build_codecontests_v3_candidate_pool(
    manifest: Mapping[str, object],
    row: Mapping[str, object],
    *,
    measure_tokens: Callable[[str], Mapping[str, object]],
) -> dict[str, object]:
    """Build the bounded original-first publisher-labelled Python 3 R pool."""

    problem_id = manifest.get("candidate_problem_id")
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("CodeContests candidate problem identity is invalid")
    correct = row.get("solutions")
    incorrect = row.get("incorrect_solutions")
    if not isinstance(correct, Mapping) or not isinstance(incorrect, Mapping):
        raise ValueError("CodeContests stored solution field is malformed")
    correct_languages = correct.get("language")
    correct_sources = correct.get("solution")
    incorrect_languages = incorrect.get("language")
    incorrect_sources = incorrect.get("solution")
    if not all(
        isinstance(value, (list, tuple))
        for value in (
            correct_languages,
            correct_sources,
            incorrect_languages,
            incorrect_sources,
        )
    ) or len(correct_languages) != len(correct_sources) or len(incorrect_languages) != len(
        incorrect_sources
    ):
        raise ValueError("CodeContests stored solution field is malformed")

    original_ordinal = _stored_solution_ordinal(
        manifest.get("repair_solution_id"), label="correct"
    )
    incorrect_ordinal = _stored_solution_ordinal(
        manifest.get("candidate_solution_id"), label="incorrect"
    )
    if original_ordinal >= len(correct_sources) or incorrect_ordinal >= len(incorrect_sources):
        raise ValueError("CodeContests frozen source is unavailable")
    original_source = correct_sources[original_ordinal]
    fixed_i_source = incorrect_sources[incorrect_ordinal]
    if not isinstance(original_source, str) or not isinstance(fixed_i_source, str):
        raise ValueError("CodeContests frozen source is unavailable")
    original_fingerprint = canonical_codecontests_source_fingerprint(original_source)
    fixed_i_fingerprint = canonical_codecontests_source_fingerprint(fixed_i_source)
    if original_fingerprint != manifest.get("repair_solution_fingerprint"):
        raise ValueError("CodeContests original R fingerprint differs from freeze")
    if fixed_i_fingerprint != manifest.get("candidate_solution_fingerprint"):
        raise ValueError("CodeContests fixed I fingerprint differs from freeze")

    fixed_i_serialized, fixed_i_response = _measured_tokens(fixed_i_source, measure_tokens)
    if fixed_i_response != manifest.get("candidate_response_tokens"):
        raise ValueError("CodeContests fixed I response-token count differs from freeze")
    if fixed_i_serialized > CODECONTESTS_V3_MAX_SERIALIZED_TOKENS:
        raise ValueError("CodeContests fixed I serialization differs from freeze")

    cross_label_fingerprints = {
        canonical_codecontests_source_fingerprint(source)
        for source in incorrect_sources
        if isinstance(source, str) and source.strip()
    }
    exclusion_names = (
        "duplicate_exclusion",
        "cross_label_exclusion",
        "non_python_exclusion",
        "nontext_or_empty_exclusion",
        "serialization_exclusion",
        "beyond_bound",
        "parity_only_excluded",
    )
    exclusions = {name: 0 for name in exclusion_names}
    seen: set[str] = {original_fingerprint}
    original_serialized, original_response = _measured_tokens(original_source, measure_tokens)
    if original_response != manifest.get("repair_response_tokens"):
        raise ValueError("CodeContests original R response-token count differs from freeze")
    original_selectable = bool(
        correct_languages[original_ordinal] == 3
        and original_source.strip()
        and original_fingerprint not in cross_label_fingerprints
        and original_serialized <= CODECONTESTS_V3_MAX_SERIALIZED_TOKENS
    )
    if not original_selectable:
        exclusions["parity_only_excluded"] = 1
    original = {
        "source": original_source,
        "solution_id": manifest.get("repair_solution_id"),
        "fingerprint": original_fingerprint,
        "response_tokens": original_response,
        "serialized_tokens": original_serialized,
        "original": True,
        "selectable": original_selectable,
        "parity_only": not original_selectable,
        "priority": None,
    }

    alternatives: list[dict[str, object]] = []
    for ordinal, (language, source) in enumerate(zip(correct_languages, correct_sources)):
        if ordinal == original_ordinal:
            continue
        if language != 3:
            exclusions["non_python_exclusion"] += 1
            continue
        if not isinstance(source, str) or not source.strip():
            exclusions["nontext_or_empty_exclusion"] += 1
            continue
        fingerprint = canonical_codecontests_source_fingerprint(source)
        if fingerprint in seen:
            exclusions["duplicate_exclusion"] += 1
            continue
        seen.add(fingerprint)
        if fingerprint in cross_label_fingerprints:
            exclusions["cross_label_exclusion"] += 1
            continue
        serialized, response = _measured_tokens(source, measure_tokens)
        if serialized > CODECONTESTS_V3_MAX_SERIALIZED_TOKENS:
            exclusions["serialization_exclusion"] += 1
            continue
        alternatives.append(
            {
                "source": source,
                "solution_id": f"{problem_id}:correct:{ordinal}",
                "fingerprint": fingerprint,
                "response_tokens": response,
                "serialized_tokens": serialized,
                "original": False,
                "selectable": True,
                "parity_only": False,
                "priority": codecontests_v3_candidate_priority(problem_id, fingerprint),
            }
        )
    alternatives.sort(key=lambda candidate: str(candidate["priority"]))
    admitted_alternatives = alternatives[: CODECONTESTS_V3_MAX_CANDIDATES - 1]
    exclusions["beyond_bound"] = len(alternatives) - len(admitted_alternatives)
    return {
        "triplet_id": manifest.get("triplet_id"),
        "candidate_problem_id": problem_id,
        "fixed_i": {
            "source": fixed_i_source,
            "solution_id": manifest.get("candidate_solution_id"),
            "fingerprint": fixed_i_fingerprint,
            "response_tokens": fixed_i_response,
            "serialized_tokens": fixed_i_serialized,
        },
        "candidates": tuple([original, *admitted_alternatives]),
        "exclusions": exclusions,
    }


def _compact_v3_attempt(
    candidate: Mapping[str, object], result: Mapping[str, object]
) -> dict[str, object]:
    return {
        "fingerprint": candidate.get("fingerprint"),
        "original": candidate.get("original") is True,
        "selectable": candidate.get("selectable") is True,
        "response_tokens": candidate.get("response_tokens"),
        "category": result.get("category"),
        "complete_contract": result.get("complete_contract") is True,
        "test_executions": result.get("test_executions"),
    }


def _compact_v3_witness(result: Mapping[str, object]) -> dict[str, object] | None:
    verdicts = result.get("test_verdicts")
    if not isinstance(verdicts, (list, tuple)):
        return None
    for verdict in verdicts:
        if not isinstance(verdict, Mapping) or verdict.get("category") == "pass":
            continue
        allowed = (
            "family",
            "ordinal",
            "category",
            "input_sha256",
            "expected_output_sha256",
            "observed_output_sha256",
        )
        return {key: verdict.get(key) for key in allowed}
    return None


def _codecontests_v3b_panel_id(
    tests: Sequence[Mapping[str, object]],
) -> str:
    """Hash the materialized judge panel after the frozen LF normalization."""

    if not isinstance(tests, (list, tuple)) or not tests:
        raise ValueError("CodeContests V3b materialized test panel is invalid")
    canonical: list[dict[str, object]] = []
    for test in tests:
        if not isinstance(test, Mapping):
            raise ValueError("CodeContests V3b materialized test panel is invalid")
        family = test.get("family")
        ordinal = test.get("ordinal")
        if (
            family not in _TEST_FAMILIES
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise ValueError("CodeContests V3b materialized test identity is invalid")
        canonical.append(
            {
                "family": family,
                "ordinal": ordinal,
                "input": _lf(test.get("input")),
                "expected_output": _lf(test.get("expected_output")),
            }
        )
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(payload)


def _codecontests_v3b_b0_identity(
    pool: Mapping[str, object], measurement_device_id: str
) -> tuple[object, str, str]:
    triplet_id = pool.get("triplet_id")
    problem_id = pool.get("candidate_problem_id")
    candidates = pool.get("candidates")
    if (
        not isinstance(triplet_id, str)
        or not triplet_id
        or not isinstance(problem_id, str)
        or not problem_id
        or not isinstance(candidates, (list, tuple))
        or not candidates
        or not isinstance(candidates[0], Mapping)
        or candidates[0].get("original") is not True
    ):
        raise ValueError("CodeContests V3b B0 pool identity is invalid")
    original = candidates[0]
    fingerprint = original.get("fingerprint")
    if (
        not isinstance(original.get("source"), str)
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or not isinstance(measurement_device_id, str)
        or len(measurement_device_id) != 64
        or any(character not in "0123456789abcdef" for character in measurement_device_id)
    ):
        raise ValueError("CodeContests V3b B0 typed identity is invalid")
    return original, problem_id, triplet_id


def adjudicate_codecontests_v3b_b0(
    pool: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
    *,
    measurement_device_id: str,
    judge,
    unsupported_original_contract: bool = False,
) -> dict[str, object]:
    """Execute and close the V3b original-R baseline exactly once per problem."""

    original, problem_id, triplet_id = _codecontests_v3b_b0_identity(
        pool, measurement_device_id
    )
    common = {
        "schema": "codecontests-v3b-b0-record-v1",
        "triplet_id": triplet_id,
        "candidate_problem_id": problem_id,
        "original_r_fingerprint": original.get("fingerprint"),
        "measurement_device_id": measurement_device_id,
    }
    if unsupported_original_contract:
        return {
            **common,
            "test_panel_status": "unsupported_original_contract",
            "test_panel_id": None,
            "category": None,
            "complete_contract": None,
            "test_execution_count": 0,
            "retry_count": 0,
            "infrastructure_attempt_count": 0,
            "unresolved_runner": False,
            "failure_witness": None,
        }

    panel_id = _codecontests_v3b_panel_id(tests)
    retry_count = 0
    infrastructure_attempt_count = 0
    result = judge(original["source"], tests, stop_on_first_failure=True)
    if not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES:
        raise ValueError("CodeContests V3b B0 judge result is invalid")
    if result.get("category") == "runner_infrastructure_error":
        infrastructure_attempt_count += 1
        retry_count = 1
        result = judge(original["source"], tests, stop_on_first_failure=True)
        if not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES:
            raise ValueError("CodeContests V3b B0 judge retry result is invalid")
        if result.get("category") == "runner_infrastructure_error":
            infrastructure_attempt_count += 1

    executions = result.get("test_executions")
    if isinstance(executions, bool) or not isinstance(executions, int) or executions < 0:
        raise ValueError("CodeContests V3b B0 test execution count is invalid")
    unresolved = result.get("category") == "runner_infrastructure_error"
    return {
        **common,
        "test_panel_status": "materialized",
        "test_panel_id": panel_id,
        "category": result.get("category"),
        "complete_contract": result.get("complete_contract") is True,
        "test_execution_count": executions,
        "retry_count": retry_count,
        "infrastructure_attempt_count": infrastructure_attempt_count,
        "unresolved_runner": unresolved,
        "failure_witness": _compact_v3_witness(result),
    }


def codecontests_v3b_historical_transition(
    v2_record: Mapping[str, object], b0_record: Mapping[str, object]
) -> dict[str, str]:
    """Describe the V2-to-V3b observation without making it a validity gate."""

    reason = v2_record.get("ineligible_reason")
    if reason == "unsupported_file_io_problem":
        v2_state = "v2_unsupported"
    elif v2_record.get("evaluator_qualified") is True:
        v2_state = "v2_r_pass"
    elif reason == "repair_did_not_pass_complete_contract":
        v2_state = "v2_r_nonpass"
    else:
        raise ValueError("CodeContests V3b historical V2 state is invalid")

    if b0_record.get("unresolved_runner") is True:
        raise ValueError("CodeContests V3b unresolved B0 has no closed transition")
    panel_status = b0_record.get("test_panel_status")
    if panel_status == "unsupported_original_contract":
        current_state = "current_unsupported"
    elif (
        panel_status == "materialized"
        and b0_record.get("category") == "pass"
        and b0_record.get("complete_contract") is True
    ):
        current_state = "current_r_pass"
    elif panel_status == "materialized" and b0_record.get("category") in _JUDGE_CATEGORIES:
        current_state = "current_r_nonpass"
    else:
        raise ValueError("CodeContests V3b current B0 state is invalid")
    return {
        "v2_state": v2_state,
        "current_state": current_state,
        "transition": f"{v2_state}->{current_state}",
    }


def compact_codecontests_v3a_runner_diagnostic(
    result: Mapping[str, object],
    *,
    retry_count: int,
    configured_c_problem: int,
    qualification_rung: int | None = None,
) -> dict[str, object]:
    """Retain only the preregistered V3a runner-diagnostic allowlist."""

    if result.get("category") != "runner_infrastructure_error":
        raise ValueError("V3a runner diagnostic requires an infrastructure result")
    if (
        isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or retry_count < 0
        or isinstance(configured_c_problem, bool)
        or not isinstance(configured_c_problem, int)
        or configured_c_problem < 1
    ):
        raise ValueError("V3a runner diagnostic context is invalid")
    diagnostic: dict[str, object] = {}
    for key in ("failure_stage", "detail", "exception_class", "sandbox_backend"):
        value = result.get(key)
        if isinstance(value, str) and value:
            diagnostic[key] = value
    for key in ("returncode", "signal"):
        value = result.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            diagnostic[key] = value
    diagnostic["retry_count"] = retry_count
    diagnostic["configured_c_problem"] = configured_c_problem
    if qualification_rung is not None:
        if (
            isinstance(qualification_rung, bool)
            or not isinstance(qualification_rung, int)
            or qualification_rung < 1
        ):
            raise ValueError("V3a qualification rung is invalid")
        diagnostic["qualification_rung"] = qualification_rung
    return diagnostic


def _v3_original_parity(
    v2_record: Mapping[str, object],
    original_result: Mapping[str, object] | None,
    i_result: Mapping[str, object] | None,
    *,
    unsupported_original_contract: bool,
) -> bool:
    reason = v2_record.get("ineligible_reason")
    if reason == "unsupported_file_io_problem":
        return unsupported_original_contract and original_result is None and i_result is None
    if original_result is None:
        return False
    original_category = original_result.get("category")
    if reason == "repair_did_not_pass_complete_contract":
        return original_category in _JUDGE_CATEGORIES - {
            "pass",
            "runner_infrastructure_error",
        }
    if v2_record.get("evaluator_qualified") is not True:
        return False
    if original_category != "pass" or original_result.get("complete_contract") is not True:
        return False
    if i_result is None or i_result.get("category") == "runner_infrastructure_error":
        return False
    observed_discrepancy = i_result.get("category") != "pass"
    expected_discrepancy = bool(
        v2_record.get("certificate_discrepant") is True
        or v2_record.get("classification") == "certificate_discrepant"
    )
    return observed_discrepancy == expected_discrepancy


def adjudicate_codecontests_v3_problem(
    pool: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
    v2_record: Mapping[str, object],
    *,
    judge,
    unsupported_original_contract: bool = False,
    _v3a_diagnostic_context: Mapping[str, object] | None = None,
    _supplied_original_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute one bounded V3 pool and retain only compact adjudication facts."""

    if unsupported_original_contract:
        parity = _v3_original_parity(
            v2_record,
            None,
            None,
            unsupported_original_contract=True,
        )
        return {
            "disposition": "unsupported_original_contract",
            "candidate_attempts": (),
            "selected_r_fingerprint": None,
            "selected_r_response_tokens": None,
            "selected_r_is_original": None,
            "i_category": None,
            "i_discrepant": None,
            "i_witness": None,
            "v2_parity_pass": parity,
            "retry_count": 0,
            "runner_invalid": False,
        }
    candidates = pool.get("candidates")
    fixed_i = pool.get("fixed_i")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("CodeContests V3 candidate pool is invalid")
    if not isinstance(fixed_i, Mapping) or not isinstance(fixed_i.get("source"), str):
        raise ValueError("CodeContests V3 fixed I is invalid")

    retry_count = 0
    repeated_runner_result: Mapping[str, object] | None = None

    def execute(source: object) -> Mapping[str, object]:
        nonlocal retry_count, repeated_runner_result
        if not isinstance(source, str):
            raise ValueError("CodeContests V3 executable source is invalid")
        result = judge(source, tests, stop_on_first_failure=True)
        if not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES:
            raise ValueError("CodeContests V3 judge result is invalid")
        if result.get("category") == "runner_infrastructure_error":
            retry_count += 1
            result = judge(source, tests, stop_on_first_failure=True)
            if not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES:
                raise ValueError("CodeContests V3 judge retry result is invalid")
            if result.get("category") == "runner_infrastructure_error":
                repeated_runner_result = result
        return result

    def runner_diagnostic() -> dict[str, object]:
        if _v3a_diagnostic_context is None or repeated_runner_result is None:
            return {}
        return {
            "runner_diagnostic": compact_codecontests_v3a_runner_diagnostic(
                repeated_runner_result,
                retry_count=retry_count,
                configured_c_problem=int(
                    _v3a_diagnostic_context["configured_c_problem"]
                ),
                qualification_rung=_v3a_diagnostic_context.get("qualification_rung"),
            )
        }

    attempts: list[dict[str, object]] = []
    original_result: Mapping[str, object] | None = None
    i_result: Mapping[str, object] | None = None
    selected: Mapping[str, object] | None = None
    selectable_count = 0
    for ordinal, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("CodeContests V3 candidate is invalid")
        if candidate.get("selectable") is True:
            selectable_count += 1
        result = (
            _supplied_original_result
            if ordinal == 0 and _supplied_original_result is not None
            else execute(candidate.get("source"))
        )
        if not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES:
            raise ValueError("CodeContests V3 supplied original result is invalid")
        attempts.append(_compact_v3_attempt(candidate, result))
        if ordinal == 0:
            if candidate.get("original") is not True:
                raise ValueError("CodeContests V3 original is not first")
            original_result = result
        if result.get("category") == "runner_infrastructure_error":
            return {
                "disposition": "unresolved_runner_error",
                "candidate_attempts": tuple(attempts),
                "selected_r_fingerprint": None,
                "selected_r_response_tokens": None,
                "selected_r_is_original": None,
                "i_category": None,
                "i_discrepant": None,
                "i_witness": None,
                "v2_parity_pass": False,
                "retry_count": retry_count,
                "runner_invalid": True,
                **runner_diagnostic(),
            }
        complete_pass = (
            result.get("category") == "pass"
            and result.get("complete_contract") is True
        )
        if (
            ordinal == 0
            and complete_pass
            and candidate.get("selectable") is not True
            and _supplied_original_result is None
        ):
            i_result = execute(fixed_i.get("source"))
            if i_result.get("category") == "runner_infrastructure_error":
                return {
                    "disposition": "unresolved_runner_error",
                    "candidate_attempts": tuple(attempts),
                    "selected_r_fingerprint": None,
                    "selected_r_response_tokens": None,
                    "selected_r_is_original": None,
                    "i_category": "runner_infrastructure_error",
                    "i_discrepant": None,
                    "i_witness": None,
                    "v2_parity_pass": False,
                    "retry_count": retry_count,
                    "runner_invalid": True,
                    **runner_diagnostic(),
                }
        if complete_pass and candidate.get("selectable") is True:
            selected = candidate
            break

    if selected is None:
        disposition = (
            "no_eligible_candidate_within_bound"
            if selectable_count == 0
            else "no_passing_r_within_bound"
        )
        return {
            "disposition": disposition,
            "candidate_attempts": tuple(attempts),
            "selected_r_fingerprint": None,
            "selected_r_response_tokens": None,
            "selected_r_is_original": None,
            "i_category": i_result.get("category") if i_result is not None else None,
            "i_discrepant": (
                i_result.get("category") != "pass" if i_result is not None else None
            ),
            "i_witness": _compact_v3_witness(i_result) if i_result is not None else None,
            "v2_parity_pass": _v3_original_parity(
                v2_record,
                original_result,
                i_result,
                unsupported_original_contract=False,
            ),
            "retry_count": retry_count,
            "runner_invalid": False,
        }

    if i_result is None:
        i_result = execute(fixed_i.get("source"))
    if i_result.get("category") == "runner_infrastructure_error":
        disposition = "unresolved_runner_error"
        runner_invalid = True
        i_discrepant = None
    else:
        runner_invalid = False
        i_discrepant = i_result.get("category") != "pass"
        if i_discrepant:
            disposition = (
                "original_r_recovered_certificate"
                if selected.get("original") is True
                else "alternative_r_recovered_certificate"
            )
        else:
            disposition = "r_pass_i_no_observed_discrepancy"
    return {
        "disposition": disposition,
        "candidate_attempts": tuple(attempts),
        "selected_r_fingerprint": selected.get("fingerprint"),
        "selected_r_response_tokens": selected.get("response_tokens"),
        "selected_r_is_original": selected.get("original") is True,
        "i_category": i_result.get("category"),
        "i_discrepant": i_discrepant,
        "i_witness": _compact_v3_witness(i_result),
        "v2_parity_pass": _v3_original_parity(
            v2_record,
            original_result,
            i_result,
            unsupported_original_contract=False,
        ),
        "retry_count": retry_count,
        "runner_invalid": runner_invalid,
        **(runner_diagnostic() if runner_invalid else {}),
    }


def adjudicate_codecontests_v3b_problem(
    pool: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
    b0_record: Mapping[str, object],
    *,
    measurement_device_id: str,
    judge,
) -> dict[str, object]:
    """Run V3b S1 while reusing, rather than re-executing, the closed B0 slot."""

    original, problem_id, triplet_id = _codecontests_v3b_b0_identity(
        pool, measurement_device_id
    )
    if b0_record.get("schema") != "codecontests-v3b-b0-record-v1":
        raise ValueError("CodeContests V3b B0 schema is invalid")
    if b0_record.get("measurement_device_id") != measurement_device_id:
        raise ValueError("CodeContests V3b B0 measurement device does not match S1")
    if (
        b0_record.get("triplet_id") != triplet_id
        or b0_record.get("candidate_problem_id") != problem_id
        or b0_record.get("original_r_fingerprint") != original.get("fingerprint")
    ):
        raise ValueError("CodeContests V3b B0 problem identity does not match S1")
    if b0_record.get("unresolved_runner") is True:
        raise ValueError("CodeContests V3b unresolved B0 cannot enter S1")

    status = b0_record.get("test_panel_status")
    if status == "unsupported_original_contract":
        if tests or b0_record.get("test_panel_id") is not None:
            raise ValueError("CodeContests V3b unsupported B0 panel identity is invalid")
        record = adjudicate_codecontests_v3_problem(
            pool,
            tests,
            {"ineligible_reason": "unsupported_file_io_problem"},
            judge=judge,
            unsupported_original_contract=True,
        )
    elif status == "materialized":
        panel_id = _codecontests_v3b_panel_id(tests)
        if b0_record.get("test_panel_id") != panel_id:
            raise ValueError("CodeContests V3b B0 test panel does not match S1")
        category = b0_record.get("category")
        test_count = b0_record.get("test_execution_count")
        if (
            category not in _JUDGE_CATEGORIES
            or category == "runner_infrastructure_error"
            or isinstance(test_count, bool)
            or not isinstance(test_count, int)
            or test_count < 0
            or not isinstance(b0_record.get("complete_contract"), bool)
        ):
            raise ValueError("CodeContests V3b closed B0 result is invalid")
        witness = b0_record.get("failure_witness")
        supplied = {
            "category": category,
            "complete_contract": b0_record.get("complete_contract"),
            "test_executions": test_count,
            "test_verdicts": (witness,) if isinstance(witness, Mapping) else (),
        }
        record = adjudicate_codecontests_v3_problem(
            pool,
            tests,
            {},
            judge=judge,
            _supplied_original_result=supplied,
        )
    else:
        raise ValueError("CodeContests V3b B0 panel status is invalid")

    record["v2_parity_pass"] = None
    record["historical_parity_status"] = "not_applicable"
    record["b0_original_reused"] = True
    record["measurement_device_id"] = measurement_device_id
    record["test_panel_status"] = status
    record["test_panel_id"] = b0_record.get("test_panel_id")
    return record


def adjudicate_codecontests_v3a_problem(
    pool: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
    v2_record: Mapping[str, object],
    *,
    judge,
    configured_c_problem: int,
    qualification_rung: int | None = None,
    unsupported_original_contract: bool = False,
) -> dict[str, object]:
    """Run unchanged V3 adjudication with V3a-only compact runner evidence."""

    return adjudicate_codecontests_v3_problem(
        pool,
        tests,
        v2_record,
        judge=judge,
        unsupported_original_contract=unsupported_original_contract,
        _v3a_diagnostic_context={
            "configured_c_problem": configured_c_problem,
            "qualification_rung": qualification_rung,
        },
    )


def run_codecontests_v3a_original_transfer(
    pool: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
    v2_record: Mapping[str, object],
    *,
    judge,
) -> dict[str, object]:
    """Replay only manifest-index 20's original R for V3a qualification."""

    if (
        pool.get("triplet_id") != "triplet:20"
        or pool.get("candidate_problem_id") != "train:10088"
    ):
        raise ValueError("V3a transfer is fixed to manifest index 20")
    candidates = pool.get("candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("V3a transfer candidate pool is invalid")
    original = candidates[0]
    if not isinstance(original, Mapping) or original.get("original") is not True:
        raise ValueError("V3a transfer original R is not first")
    source = original.get("source")
    if not isinstance(source, str):
        raise ValueError("V3a transfer original R is invalid")

    first = judge(source, tests, stop_on_first_failure=True)
    if not isinstance(first, Mapping) or first.get("category") not in _JUDGE_CATEGORIES:
        raise ValueError("V3a transfer judge result is invalid")
    infrastructure_attempt_count = int(
        first.get("category") == "runner_infrastructure_error"
    )
    retry_count = 0
    final = first
    if infrastructure_attempt_count:
        retry_count = 1
        final = judge(source, tests, stop_on_first_failure=True)
        if not isinstance(final, Mapping) or final.get("category") not in _JUDGE_CATEGORIES:
            raise ValueError("V3a transfer retry result is invalid")
        infrastructure_attempt_count += int(
            final.get("category") == "runner_infrastructure_error"
        )
    final_non_infrastructure = final.get("category") != "runner_infrastructure_error"
    parity = _v3_original_parity(
        v2_record,
        final if final_non_infrastructure else None,
        None,
        unsupported_original_contract=False,
    )
    infrastructure_free = infrastructure_attempt_count == 0 and retry_count == 0
    result: dict[str, object] = {
        "schema": "codecontests-v3a-transfer-qualification-v1",
        "triplet_id": "triplet:20",
        "candidate_problem_id": "train:10088",
        "original_only": True,
        "fixed_i_executed": False,
        "alternative_candidates_executed": 0,
        "infrastructure_attempt_count": infrastructure_attempt_count,
        "retry_count": retry_count,
        "final_non_infrastructure": final_non_infrastructure,
        "infrastructure_free": infrastructure_free,
        "coarse_v2_parity_pass": parity,
        "passed": infrastructure_free and final_non_infrastructure and parity,
    }
    if final.get("category") == "runner_infrastructure_error":
        result["runner_diagnostic"] = compact_codecontests_v3a_runner_diagnostic(
            final,
            retry_count=retry_count,
            configured_c_problem=1,
        )
    return result


def _paired_family(row: Mapping[str, object], family: str) -> tuple[tuple[int, str, str], ...]:
    value = row.get(f"{family}_tests")
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError(f"{family} test field is malformed")
    inputs = value.get("input")
    outputs = value.get("output")
    if not isinstance(inputs, (list, tuple)) or not isinstance(outputs, (list, tuple)):
        raise ValueError(f"{family} test arrays are malformed")
    if len(inputs) != len(outputs):
        raise ValueError(f"{family} test arrays are not paired")
    return tuple((index, _lf(stdin), _lf(expected)) for index, (stdin, expected) in enumerate(zip(inputs, outputs)))


def freeze_codecontests_tests(
    problem_fingerprint: str,
    row: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Freeze the outcome-blind public/all plus hash-bounded hidden contract."""

    if not isinstance(problem_fingerprint, str) or len(problem_fingerprint) != 64:
        raise ValueError("problem fingerprint is invalid")
    if row.get("input_file") not in (None, "") or row.get("output_file") not in (None, ""):
        raise ValueError("unsupported file-I/O problem")
    seen: set[tuple[str, str]] = set()
    by_family: dict[str, list[dict[str, object]]] = {family: [] for family in _TEST_FAMILIES}
    for family in _TEST_FAMILIES:
        for ordinal, stdin, expected in _paired_family(row, family):
            payload = (stdin, expected)
            if payload in seen:
                continue
            seen.add(payload)
            by_family[family].append(
                {
                    "family": family,
                    "ordinal": ordinal,
                    "input": stdin,
                    "expected_output": expected,
                }
            )
    for family in ("private", "generated"):
        by_family[family].sort(
            key=lambda test: _sha256(
                f"{problem_fingerprint}|{family}|{test['input']}|{test['expected_output']}"
            )
        )
        by_family[family] = by_family[family][:32]
    frozen = tuple(by_family["public"] + by_family["private"] + by_family["generated"])
    if not frozen:
        raise ValueError("problem has no usable paired test")
    return frozen


def codecontests_s12d_evaluation_unit_key(
    triplet_id: str,
    role: str,
    problem_id: str,
    candidate_or_mutant_id: str,
) -> tuple[str, str, str, str]:
    """Return S12d's stable identity before a panel or request exists."""

    values = (triplet_id, role, problem_id, candidate_or_mutant_id)
    if role not in {"I", "R", "M", "N", "endpoint_reference"} or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise CodeContestsS12DInputInvalid("S12d evaluation-unit key is invalid")
    return values


def classify_codecontests_s12d_problem(
    *,
    problem_id: str,
    problem_fingerprint: str,
    row: Mapping[str, object],
    panel_builder: Callable[
        [str, Mapping[str, object]], tuple[dict[str, object], ...]
    ] = freeze_codecontests_tests,
) -> dict[str, object]:
    """Classify one referenced problem through the sole S12d classifier."""

    if not isinstance(problem_id, str) or not problem_id:
        raise CodeContestsS12DInputInvalid("S12d problem locator is invalid")
    if not isinstance(problem_fingerprint, str) or len(problem_fingerprint) != 64:
        raise CodeContestsS12DInputInvalid("S12d problem fingerprint is invalid")
    if not isinstance(row, Mapping):
        raise CodeContestsS12DInputInvalid("S12d problem row is malformed")
    try:
        panel = panel_builder(problem_fingerprint, row)
    except ValueError as exc:
        reason = {
            "unsupported file-I/O problem": "unsupported_file_io_problem",
            "problem has no usable paired test": "no_usable_paired_test",
        }.get(str(exc))
        if reason is not None:
            return {
                "eligibility_disposition": "expected_exclusion",
                "exclusion_reason": reason,
                "problem_id": problem_id,
                "problem_fingerprint": problem_fingerprint,
                "panel": None,
                "panel_id": None,
            }
        raise CodeContestsS12DInputInvalid(str(exc)) from exc
    except Exception as exc:
        raise CodeContestsS12DContractInvalid(str(exc)) from exc
    try:
        panel_id, _ = _panel_identity(panel)
    except ValueError as exc:
        raise CodeContestsS12DInputInvalid(str(exc)) from exc
    return {
        "eligibility_disposition": "eligible",
        "exclusion_reason": None,
        "problem_id": problem_id,
        "problem_fingerprint": problem_fingerprint,
        "panel": panel,
        "panel_id": panel_id,
    }


_S12D_PRIMITIVE_FIELDS = (
    "manifest_ordinal",
    "triplet_id",
    "candidate_problem_id",
    "eligibility_dispositions",
    "execution_dispositions",
    "candidate_skip_counts",
    "r_attempt_order",
    "r_attempt_count",
    "r_search_complete",
    "selected_r_id",
    "selected_r_fingerprint",
    "r_provenance",
    "i_id",
    "i_category",
    "i_witness_test_id",
    "mutation_panel_anchor_test_id",
    "failure_family",
    "m_attempt_order",
    "m_attempt_count",
    "m_search_complete",
    "selected_m_id",
    "m_category",
    "n_id",
    "n_category",
    "response_tokens",
    "serialized_tokens",
    "role_objects",
    "relation_identity_sha256",
    "relation_object_count",
    "infrastructure_error_count",
    "contract_error_count",
)


def project_codecontests_s12d_primitive_row(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Project the support-blind relation used for C1/C4 parity."""

    if not isinstance(value, Mapping):
        raise CodeContestsS12DContractInvalid("S12d primitive source is malformed")
    row = {key: value.get(key) for key in _S12D_PRIMITIVE_FIELDS}
    row["schema"] = "codecontests-s12d-primitive-row-v1"
    row["eligibility_dispositions"] = list(value.get("eligibility_dispositions") or [])
    row["execution_dispositions"] = list(value.get("execution_dispositions") or [])
    row["candidate_skip_counts"] = dict(value.get("candidate_skip_counts") or {})
    row["contract_error_count"] = int(value.get("contract_error_count") or 0)
    return {"schema": row.pop("schema"), **row}


def derive_codecontests_s12d_membership(
    row: Mapping[str, object],
) -> dict[str, bool]:
    """Purely derive the three preserved memberships from one primitive row."""

    if row.get("schema") != "codecontests-s12d-primitive-row-v1":
        raise CodeContestsS12DContractInvalid("S12d primitive schema is invalid")
    clean = (
        int(row.get("infrastructure_error_count") or 0) == 0
        and int(row.get("contract_error_count") or 0) == 0
    )
    natural = bool(
        clean
        and row.get("selected_r_id")
        and row.get("i_category")
        in {"wrong_answer", "compile_failure", "runtime_failure", "timeout"}
    )
    mutation = bool(
        natural
        and row.get("selected_m_id")
        and row.get("m_category") in {"wrong_answer", "runtime_failure", "timeout"}
    )
    policy = bool(natural and row.get("n_id") and row.get("n_category") == "pass")
    return {
        "natural_member": natural,
        "mutation_member": mutation,
        "policy_member": policy,
    }


def _verdict_map(
    result: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[str, int], Mapping[str, object]] | None, str | None]:
    category = result.get("category")
    if category not in _JUDGE_CATEGORIES:
        return None, "unknown_judge_category"
    if category == "runner_infrastructure_error":
        return None, "runner_infrastructure_error"
    if category == "compile_failure":
        return {}, None
    if result.get("complete_contract") is not True or result.get("test_executions") != len(tests):
        return None, "incomplete_sampled_contract"
    rows = result.get("test_verdicts")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(tests):
        return None, "malformed_test_verdicts"
    mapped: dict[tuple[str, int], Mapping[str, object]] = {}
    expected_by_identity = {
        (str(test["family"]), int(test["ordinal"])): test for test in tests
    }
    for row in rows:
        if not isinstance(row, Mapping):
            return None, "malformed_test_verdict"
        try:
            identity = (str(row["family"]), int(row["ordinal"]))
        except (KeyError, TypeError, ValueError):
            return None, "malformed_test_identity"
        expected = expected_by_identity.get(identity)
        if expected is None or identity in mapped or row.get("category") not in _JUDGE_CATEGORIES:
            return None, "test_identity_or_category_mismatch"
        if row.get("input_sha256") != _sha256(_lf(str(expected["input"]))) or row.get(
            "expected_output_sha256"
        ) != _sha256(_lf(str(expected["expected_output"]))):
            return None, "judge_witness_hash_mismatch"
        observed = row.get("observed_output_sha256")
        if not isinstance(observed, str) or len(observed) != 64:
            return None, "observed_output_hash_invalid"
        mapped[identity] = row
    if set(mapped) != set(expected_by_identity):
        return None, "test_denominator_drift"
    return mapped, None


def _compact_witness(
    triplet: Mapping[str, object], verdict: Mapping[str, object]
) -> dict[str, object]:
    return {
        "triplet_id": triplet["triplet_id"],
        "problem_id": triplet["candidate_problem_id"],
        "problem_fingerprint": triplet["candidate_problem_fingerprint"],
        "solution_id": triplet["candidate_solution_id"],
        "solution_fingerprint": triplet["candidate_solution_fingerprint"],
        "test_family": verdict["family"],
        "test_ordinal": verdict["ordinal"],
        "failure_category": verdict["category"],
        "input_sha256": verdict["input_sha256"],
        "expected_output_sha256": verdict["expected_output_sha256"],
        "observed_output_sha256": verdict["observed_output_sha256"],
    }


def ineligible_codecontests_pair(
    triplet: Mapping[str, object], reason: str
) -> dict[str, object]:
    """Build the compact record for a protocol-defined census exclusion."""

    return {
        "triplet_id": triplet.get("triplet_id"),
        "candidate_problem_id": triplet.get("candidate_problem_id"),
        "classification": "ineligible",
        "ineligible_reason": reason,
        "evaluator_qualified": False,
        "certificate_discrepant": False,
        "high_severity": False,
        "hidden_contract_count": None,
        "hidden_failure_count": None,
        "hidden_failure_fraction": None,
        "public_witness": None,
        "hidden_witness": None,
        "compile_witness": None,
    }


def adjudicate_codecontests_pair(
    triplet: Mapping[str, object],
    tests: Sequence[Mapping[str, object]],
    repair_result: Mapping[str, object],
    incorrect_result: Mapping[str, object],
) -> dict[str, object]:
    """Adjudicate publisher-selected R/I roles solely from execution replay."""

    repair, repair_error = _verdict_map(repair_result, tests)
    if repair_error is not None:
        return ineligible_codecontests_pair(triplet, f"repair_{repair_error}")
    if repair_result.get("category") != "pass" or repair is None or any(
        row.get("category") != "pass" for row in repair.values()
    ):
        return ineligible_codecontests_pair(
            triplet, "repair_did_not_pass_complete_contract"
        )

    incorrect, incorrect_error = _verdict_map(incorrect_result, tests)
    if incorrect_error is not None:
        return ineligible_codecontests_pair(triplet, f"incorrect_{incorrect_error}")
    compile_failure = incorrect_result.get("category") == "compile_failure"
    if not compile_failure and incorrect is None:
        return ineligible_codecontests_pair(triplet, "incorrect_verdict_unavailable")

    hidden_identities = tuple(
        (str(test["family"]), int(test["ordinal"]))
        for test in tests
        if test["family"] in _HIDDEN_FAMILIES
    )
    failures = [] if compile_failure else [
        row for row in incorrect.values() if row.get("category") != "pass"
    ]
    hidden_failures = [
        row
        for row in failures
        if (str(row["family"]), int(row["ordinal"])) in hidden_identities
    ]
    discrepant = compile_failure or bool(failures)
    hidden_count = len(hidden_identities)
    hidden_failure_count = hidden_count if compile_failure else len(hidden_failures)
    hidden_fraction = (
        1.0 if compile_failure else hidden_failure_count / hidden_count
    ) if hidden_count else None
    high_severity = bool(
        discrepant
        and hidden_count >= 8
        and hidden_fraction is not None
        and hidden_fraction >= 0.25
    )
    public_failure = next(
        (row for row in failures if row.get("family") == "public"), None
    )
    hidden_failure = next(
        (row for row in failures if row.get("family") in _HIDDEN_FAMILIES), None
    )
    classification = "certificate_discrepant" if discrepant else "no_observed_discrepancy"
    return {
        "triplet_id": triplet["triplet_id"],
        "candidate_problem_id": triplet["candidate_problem_id"],
        "candidate_problem_fingerprint": triplet["candidate_problem_fingerprint"],
        "candidate_solution_id": triplet["candidate_solution_id"],
        "candidate_solution_fingerprint": triplet["candidate_solution_fingerprint"],
        "repair_solution_id": triplet["repair_solution_id"],
        "repair_solution_fingerprint": triplet["repair_solution_fingerprint"],
        "response_tokens": {
            "I": triplet["candidate_response_tokens"],
            "R": triplet["repair_response_tokens"],
            "N": triplet.get("neutral_response_tokens"),
        },
        "classification": classification,
        "ineligible_reason": None,
        "evaluator_qualified": True,
        "certificate_discrepant": discrepant,
        "high_severity": high_severity,
        "hidden_contract_count": hidden_count,
        "hidden_failure_count": hidden_failure_count,
        "hidden_failure_fraction": hidden_fraction,
        "public_witness": _compact_witness(triplet, public_failure) if public_failure else None,
        "hidden_witness": _compact_witness(triplet, hidden_failure) if hidden_failure else None,
        "compile_witness": {
            "triplet_id": triplet["triplet_id"],
            "problem_id": triplet["candidate_problem_id"],
            "problem_fingerprint": triplet["candidate_problem_fingerprint"],
            "solution_id": triplet["candidate_solution_id"],
            "solution_fingerprint": triplet["candidate_solution_fingerprint"],
            "test_family": None,
            "test_ordinal": None,
            "marker": "NO_TEST_OUTPUT_COMPILE_FAILURE",
        }
        if compile_failure
        else None,
    }


def evaluate_codecontests_support(
    pairs: Sequence[Mapping[str, object]],
    *,
    arms: Sequence[str],
    plan_closes: bool,
) -> dict[str, object]:
    """Apply the exact count/ESS/share/fit/plan support gate to aligned pairs."""

    if not arms or any(arm not in {"I", "R", "N", "CLEAN", "MUTANT"} for arm in arms):
        raise ValueError("support arms are outside the frozen registry")
    triplet_ids = [pair.get("triplet_id") for pair in pairs]
    problem_ids = [pair.get("candidate_problem_id") for pair in pairs]
    identities_valid = (
        all(isinstance(value, str) and value for value in triplet_ids + problem_ids)
        and len(set(triplet_ids)) == len(triplet_ids)
        and len(set(problem_ids)) == len(problem_ids)
    )
    serialized_fits = all(pair.get("serialized_fits") is True for pair in pairs)
    arm_metrics: dict[str, dict[str, object]] = {}
    arms_pass = True
    for arm in arms:
        weights: list[int] = []
        for pair in pairs:
            value = pair.get("response_tokens")
            weight = value.get(arm) if isinstance(value, Mapping) else None
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"support weight for arm {arm} is invalid")
            weights.append(weight)
        total = sum(weights)
        sum_squares = sum(weight * weight for weight in weights)
        kish_ess = (total * total / sum_squares) if sum_squares else 0.0
        maximum = max(weights, default=0)
        max_share = maximum / total if total else 0.0
        ess_pass = total * total >= 80 * sum_squares
        share_pass = maximum * 100 <= 2 * total
        arm_metrics[arm] = {
            "response_token_mass": total,
            "kish_ess": kish_ess,
            "max_pair_token_share": max_share,
            "ess_pass": ess_pass,
            "max_share_pass": share_pass,
        }
        arms_pass = arms_pass and ess_pass and share_pass
    count_pass = len(pairs) >= 100 and len(set(problem_ids)) >= 100
    passed = bool(
        identities_valid
        and count_pass
        and arms_pass
        and serialized_fits
        and plan_closes
    )
    return {
        "passed": passed,
        "pair_count": len(pairs),
        "unique_problem_count": len(set(problem_ids)),
        "identity_pass": identities_valid,
        "count_pass": count_pass,
        "serialized_fit_pass": serialized_fits,
        "plan_closure_pass": plan_closes,
        "arm_metrics": arm_metrics,
    }


def select_codecontests_f2(
    high_severity: Sequence[Mapping[str, object]],
    pooled_certified: Sequence[Mapping[str, object]],
    *,
    plan_closes: Mapping[str, bool],
) -> dict[str, object]:
    """Select exactly high severity, pooled fallback, or the valid stop."""

    if set(plan_closes) != {"high_severity", "pooled_certified"}:
        raise ValueError("F2 plan-closure registry is incomplete")
    high_support = evaluate_codecontests_support(
        high_severity,
        arms=("I", "R"),
        plan_closes=plan_closes["high_severity"],
    )
    pooled_support = evaluate_codecontests_support(
        pooled_certified,
        arms=("I", "R"),
        plan_closes=plan_closes["pooled_certified"],
    )
    if high_support["passed"]:
        selected_branch = "high_severity"
        selected_pairs = tuple(high_severity)
        terminal = None
    elif pooled_support["passed"]:
        selected_branch = "pooled_certified"
        selected_pairs = tuple(pooled_certified)
        terminal = None
    else:
        selected_branch = None
        selected_pairs = ()
        terminal = "COMPLETED_CODECONTESTS_V2_CENSUS_INSUFFICIENT"
    return {
        "selected_branch": selected_branch,
        "selected_pairs": selected_pairs,
        "high_severity_support": high_support,
        "pooled_certified_support": pooled_support,
        "terminal": terminal,
    }


_MUTATION_MAP = {
    "<": "<=",
    "<=": "<",
    ">": ">=",
    ">=": ">",
    "==": "!=",
    "!=": "==",
    "+": "-",
    "-": "+",
    "*": "//",
    "//": "*",
    "and": "or",
    "or": "and",
    "True": "False",
    "False": "True",
}


def generate_codecontests_mutants(
    source: str,
    *,
    max_mutable_tokens: int = 32,
) -> tuple[dict[str, object], ...]:
    """Visit the first frozen mutable source tokens without touching trivia."""

    if not isinstance(source, str) or not source:
        raise ValueError("mutation source must be nonempty text")
    if max_mutable_tokens != 32:
        raise ValueError("CodeContests V2 mutation worklist is fixed at 32 tokens")
    lines = source.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    if not lines or (lines and not lines[-1].endswith(("\n", "\r"))):
        # tokenize reports a logical final line even without a line terminator.
        if not lines:
            lines = [source]
            line_offsets = [0]
    candidates: list[dict[str, object]] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for info in tokens:
        replacement = _MUTATION_MAP.get(info.string)
        if replacement is None or info.type not in {token.OP, token.NAME}:
            continue
        start = line_offsets[info.start[0] - 1] + info.start[1]
        end = line_offsets[info.end[0] - 1] + info.end[1]
        mutated = source[:start] + replacement + source[end:]
        candidates.append(
            {
                "mutable_ordinal": len(candidates),
                "source_token_start": info.start,
                "source_token_end": info.end,
                "from_token": info.string,
                "to_token": replacement,
                "source": mutated,
                "source_sha256": _sha256(_lf(mutated)),
            }
        )
        if len(candidates) == max_mutable_tokens:
            break
    return tuple(candidates)


def select_first_killed_mutant(
    clean_source: str,
    hidden_tests: Sequence[Mapping[str, object]],
    *,
    judge,
) -> dict[str, object]:
    """Return the first compile-valid test-killed mutant in frozen order."""

    if not hidden_tests or len(hidden_tests) > 8:
        raise ValueError("mutation panel must contain between one and eight hidden tests")
    clean = judge(clean_source, hidden_tests)
    if clean.get("category") == "runner_infrastructure_error":
        raise RuntimeError("clean mutation-panel replay has a runner error")
    if clean.get("category") != "pass" or clean.get("complete_contract") is not True:
        raise ValueError("clean source does not pass the frozen mutation panel")
    attempted = 0
    compile_valid = 0
    for candidate in generate_codecontests_mutants(clean_source):
        attempted += 1
        try:
            compile(str(candidate["source"]), "<codecontests-mutant>", "exec")
        except (SyntaxError, ValueError, OverflowError):
            continue
        compile_valid += 1
        result = judge(candidate["source"], hidden_tests)
        if result.get("category") == "runner_infrastructure_error":
            raise RuntimeError("mutant mutation-panel replay has a runner error")
        if result.get("category") == "compile_failure":
            continue
        if result.get("category") in {"wrong_answer", "runtime_failure", "timeout"}:
            executions = result.get("test_executions")
            if isinstance(executions, bool) or not isinstance(executions, int) or executions <= 0:
                raise ValueError("mutant kill is not test-scoped")
            return {
                **candidate,
                "available": True,
                "attempted_mutants": attempted,
                "compile_valid_mutants": compile_valid,
                "kill_category": result.get("category"),
            }
        if result.get("category") != "pass":
            raise ValueError("mutant mutation-panel category is invalid")
    return {
        "available": False,
        "attempted_mutants": attempted,
        "compile_valid_mutants": compile_valid,
    }


def adjudicate_codecontests_s12c_triplet(
    manifest: Mapping[str, object],
    candidate_row: Mapping[str, object],
    neutral_row: Mapping[str, object],
    *,
    measure_tokens: Callable[[str], Mapping[str, object]],
    neutral_measure_tokens: Callable[[str], Mapping[str, object]] | None = None,
    payload_builder: Callable[[str], bytes],
    measurement_device_id: str,
    judge,
    candidate_panel: Sequence[Mapping[str, object]] | None = None,
    neutral_panel: Sequence[Mapping[str, object]] | None = None,
    neutral_expected_exclusion: str | None = None,
    strict_dispositions: bool = False,
) -> dict[str, object]:
    """Execute one complete bounded S12c selection relation in frozen order."""

    candidate_tests = tuple(candidate_panel) if candidate_panel is not None else freeze_codecontests_tests(
        str(manifest["candidate_problem_fingerprint"]), candidate_row
    )
    pool = build_codecontests_v3_candidate_pool(
        manifest, candidate_row, measure_tokens=measure_tokens
    )
    neutral_measure = neutral_measure_tokens or measure_tokens
    objects: list[dict[str, object]] = []
    verdicts: dict[str, Mapping[str, object]] = {}

    def execute(source, locator, role, selection_id, panel):
        value = build_codecontests_executable_object(
            judged_payload=payload_builder(str(source)), source_locator=str(locator), role=str(role),
            selection_id=str(selection_id), panel=panel,
            measurement_device_id=measurement_device_id,
        )
        result = judge(
            value["judged_payload"].decode("utf-8"), panel,
            stop_on_first_failure=True,
        )
        if strict_dispositions and (
            not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES
        ):
            raise CodeContestsS12DContractInvalid(
                "S12d judge returned an unknown execution disposition"
            )
        if not isinstance(result, Mapping) or result.get("category") not in _JUDGE_CATEGORIES:
            result = {"category": "runner_infrastructure_error"}
        returned_hash = result.get("judged_payload_sha256")
        if returned_hash is not None and returned_hash != value["judged_payload_sha256"]:
            result = {"category": "runner_infrastructure_error", "detail": "payload_identity_mismatch"}
        objects.append(value)
        verdicts[str(value["logical_object_id"])] = result
        return value, result

    selected = None
    selected_result = None
    r_attempts: list[str] = []
    for candidate in pool["candidates"]:
        locator = str(candidate["solution_id"])
        _, result = execute(
            candidate["source"], locator, "R", candidate["fingerprint"], candidate_tests
        )
        r_attempts.append(locator)
        if result.get("category") == "runner_infrastructure_error":
            break
        if (
            candidate.get("selectable") is True
            and result.get("category") == "pass"
            and result.get("complete_contract") is True
            and result.get("test_executions") == len(candidate_tests)
        ):
            selected, selected_result = candidate, result
            break

    fixed_i = pool["fixed_i"]
    i_object = None
    if selected is not None:
        i_object, _ = execute(
            fixed_i["source"], fixed_i["solution_id"], "I",
            fixed_i["fingerprint"], candidate_tests,
        )
    partial = summarize_codecontests_keyed_relation(objects, verdicts)
    compact_by_id = {
        str(row["logical_object_id"]): row for row in partial["records"]
    }
    i_compact = compact_by_id.get(str(i_object["logical_object_id"])) if i_object else None
    natural_member = bool(
        selected is not None
        and selected_result is not None
        and selected_result.get("category") == "pass"
        and isinstance(i_compact, Mapping)
        and i_compact.get("category") in {
            "wrong_answer", "compile_failure", "runtime_failure", "timeout"
        }
        and not partial["infrastructure_object_ids"]
    )

    mutation_attempts: list[str] = []
    selected_mutant = None
    selected_mutant_result = None
    if natural_member and neutral_expected_exclusion is None:
        panel_head = i_compact.get("witness_test_id") or i_compact.get(
            "mutation_panel_anchor_test_id"
        )
        tests_by_id = {
            f"{test['family']}:{test['ordinal']}": test for test in candidate_tests
        }
        if panel_head not in tests_by_id:
            raise ValueError("CodeContests S12c mutation anchor is absent")
        mutation_panel = (tests_by_id[str(panel_head)],) + tuple(
            test for test in candidate_tests
            if f"{test['family']}:{test['ordinal']}" != panel_head
        )
        mutation_panel = mutation_panel[: min(8, len(mutation_panel))]
        for mutant in generate_codecontests_mutants(str(selected["source"])):
            mutation_attempts.append(str(mutant["source_sha256"]))
            try:
                compile(str(mutant["source"]), "<codecontests-s12c-mutant>", "exec")
            except (SyntaxError, ValueError, OverflowError):
                continue
            _, result = execute(
                mutant["source"], selected["solution_id"], "M",
                f"mutation:{mutant['mutable_ordinal']}:{mutant['source_sha256']}",
                mutation_panel,
            )
            if result.get("category") == "runner_infrastructure_error":
                break
            if result.get("category") in {"wrong_answer", "runtime_failure", "timeout"}:
                selected_mutant, selected_mutant_result = mutant, result
                break

    neutral_object = None
    neutral_result = None
    if natural_member:
        ordinal = _stored_solution_ordinal(
            manifest.get("neutral_solution_id"), label="correct"
        )
        solutions = neutral_row.get("solutions")
        sources = solutions.get("solution") if isinstance(solutions, Mapping) else None
        if not isinstance(sources, (list, tuple)) or ordinal >= len(sources):
            raise ValueError("CodeContests S12c neutral source is unavailable")
        neutral_source = sources[ordinal]
        if (
            not isinstance(neutral_source, str)
            or canonical_codecontests_source_fingerprint(neutral_source)
            != manifest.get("neutral_solution_fingerprint")
        ):
            raise ValueError("CodeContests S12c neutral source identity differs")
        neutral_tests = tuple(neutral_panel) if neutral_panel is not None else freeze_codecontests_tests(
            str(manifest["neutral_problem_fingerprint"]), neutral_row
        )
        neutral_object, neutral_result = execute(
            neutral_source, manifest["neutral_solution_id"], "N",
            manifest["neutral_solution_fingerprint"], neutral_tests,
        )

    relation = summarize_codecontests_keyed_relation(objects, verdicts)
    policy_member = bool(
        natural_member
        and neutral_result is not None
        and neutral_result.get("category") == "pass"
        and neutral_result.get("complete_contract") is True
        and neutral_object is not None
        and neutral_result.get("test_executions") == len(neutral_object["panel_test_ids"])
    )
    return {
        "schema": "codecontests-s12c-csr-row-v1",
        "triplet_id": manifest.get("triplet_id"),
        "candidate_problem_id": manifest.get("candidate_problem_id"),
        "r_attempt_order": r_attempts,
        "r_attempt_count": len(r_attempts),
        "r_search_complete": not bool(relation["infrastructure_object_ids"]),
        "selected_r_id": selected.get("solution_id") if selected else None,
        "selected_r_fingerprint": selected.get("fingerprint") if selected else None,
        "r_provenance": (
            "original" if selected and selected.get("original") is True
            else "alternative" if selected else None
        ),
        "selected_r_response_tokens": selected.get("response_tokens") if selected else None,
        "selected_r_serialized_tokens": selected.get("serialized_tokens") if selected else None,
        "i_id": fixed_i.get("solution_id") if i_object else None,
        "i_response_tokens": fixed_i.get("response_tokens"),
        "i_serialized_tokens": fixed_i.get("serialized_tokens"),
        "i_category": i_compact.get("category") if isinstance(i_compact, Mapping) else None,
        "i_witness_test_id": i_compact.get("witness_test_id") if isinstance(i_compact, Mapping) else None,
        "mutation_panel_anchor_test_id": i_compact.get("mutation_panel_anchor_test_id") if isinstance(i_compact, Mapping) else None,
        "failure_family": i_compact.get("failure_family") if isinstance(i_compact, Mapping) else None,
        "m_attempt_order": mutation_attempts,
        "m_attempt_count": len(mutation_attempts),
        "m_search_complete": not bool(relation["infrastructure_object_ids"]),
        "selected_m_id": (
            f"mutation:{selected_mutant['mutable_ordinal']}:{selected_mutant['source_sha256']}"
            if selected_mutant else None
        ),
        "selected_m_response_tokens": (
            _measured_tokens(str(selected_mutant["source"]), measure_tokens)[1]
            if selected_mutant else None
        ),
        "selected_m_serialized_tokens": (
            _measured_tokens(str(selected_mutant["source"]), measure_tokens)[0]
            if selected_mutant else None
        ),
        "m_category": selected_mutant_result.get("category") if selected_mutant_result else None,
        "n_id": manifest.get("neutral_solution_id") if neutral_object else None,
        "n_category": neutral_result.get("category") if neutral_result else None,
        "n_response_tokens": manifest.get("neutral_response_tokens"),
        "n_serialized_tokens": (
            _measured_tokens(str(neutral_source), neutral_measure)[0]
            if neutral_object else None
        ),
        "natural_member": natural_member,
        "mutation_member": selected_mutant is not None,
        "policy_member": policy_member,
        "relation_summary": relation,
        "selection_execution_objects": tuple(objects),
        "selection_execution_verdicts": verdicts,
    }


_S12D_CANDIDATE_SKIP_CODES = (
    "duplicate_exclusion",
    "cross_label_exclusion",
    "non_python_exclusion",
    "nontext_or_empty_exclusion",
    "serialization_exclusion",
    "parity_only_excluded",
)


def adjudicate_codecontests_s12d_triplet(
    manifest: Mapping[str, object],
    candidate_row: Mapping[str, object],
    neutral_row: Mapping[str, object],
    *,
    measure_tokens: Callable[[str], Mapping[str, object]],
    neutral_measure_tokens: Callable[[str], Mapping[str, object]] | None = None,
    payload_builder: Callable[[str], bytes],
    measurement_device_id: str,
    judge,
) -> dict[str, object]:
    """Close one S12d triplet with local exclusions and typed dispositions."""

    triplet_id = manifest.get("triplet_id")
    candidate_problem_id = manifest.get("candidate_problem_id")
    neutral_problem_id = manifest.get("neutral_problem_id")
    if not all(isinstance(value, str) and value for value in (
        triplet_id, candidate_problem_id, neutral_problem_id
    )):
        raise CodeContestsS12DInputInvalid("S12d manifest locator is invalid")
    candidate = classify_codecontests_s12d_problem(
        problem_id=str(candidate_problem_id),
        problem_fingerprint=manifest.get("candidate_problem_fingerprint"),
        row=candidate_row,
    )
    neutral = classify_codecontests_s12d_problem(
        problem_id=str(neutral_problem_id),
        problem_fingerprint=manifest.get("neutral_problem_fingerprint"),
        row=neutral_row,
    )
    if candidate["eligibility_disposition"] == "expected_exclusion":
        key = codecontests_s12d_evaluation_unit_key(
            str(triplet_id),
            "R",
            str(candidate_problem_id),
            str(manifest.get("repair_solution_id") or "original-r"),
        )
        return {
            "schema": "codecontests-s12c-csr-row-v1",
            "triplet_id": triplet_id,
            "candidate_problem_id": candidate_problem_id,
            "r_attempt_order": [],
            "r_attempt_count": 0,
            "r_search_complete": True,
            "selected_r_id": None,
            "selected_r_fingerprint": None,
            "r_provenance": None,
            "selected_r_response_tokens": None,
            "selected_r_serialized_tokens": None,
            "i_id": None,
            "i_response_tokens": manifest.get("candidate_response_tokens"),
            "i_serialized_tokens": None,
            "i_category": None,
            "i_witness_test_id": None,
            "mutation_panel_anchor_test_id": None,
            "failure_family": None,
            "m_attempt_order": [],
            "m_attempt_count": 0,
            "m_search_complete": True,
            "selected_m_id": None,
            "selected_m_response_tokens": None,
            "selected_m_serialized_tokens": None,
            "m_category": None,
            "n_id": None,
            "n_category": None,
            "n_response_tokens": manifest.get("neutral_response_tokens"),
            "n_serialized_tokens": None,
            "natural_member": False,
            "mutation_member": False,
            "policy_member": False,
            "relation_summary": {
                "schema": "codecontests-s12c-keyed-relation-summary-v1",
                "records": [],
                "reached_object_count": 0,
                "verdict_count": 0,
                "semantic_counts": {
                    name: 0 for name in (
                        "pass", "wrong_answer", "compile_failure",
                        "runtime_failure", "timeout"
                    )
                },
                "missing_object_ids": [],
                "orphan_object_ids": [],
                "infrastructure_object_ids": [],
                "passed": True,
            },
            "selection_execution_objects": (),
            "selection_execution_verdicts": {},
            "eligibility_dispositions": [
                {
                    "evaluation_unit_key": key,
                    "eligibility_disposition": "expected_exclusion",
                    "exclusion_reason": candidate["exclusion_reason"],
                    "panel_id": None,
                    "judge_request_id": None,
                }
            ],
            "execution_dispositions": [],
            "candidate_skip_counts": {code: 0 for code in _S12D_CANDIDATE_SKIP_CODES},
            "neutral_problem_classification": {
                key: neutral[key]
                for key in ("eligibility_disposition", "exclusion_reason", "panel_id")
            },
        }

    value = adjudicate_codecontests_s12c_triplet(
        manifest,
        candidate_row,
        neutral_row,
        measure_tokens=measure_tokens,
        neutral_measure_tokens=neutral_measure_tokens,
        payload_builder=payload_builder,
        measurement_device_id=measurement_device_id,
        judge=judge,
        candidate_panel=candidate["panel"],
        neutral_panel=neutral["panel"] if neutral["eligibility_disposition"] == "eligible" else None,
        neutral_expected_exclusion=neutral.get("exclusion_reason"),
        strict_dispositions=True,
    )
    objects = value["selection_execution_objects"]
    relation = value["relation_summary"]
    record_by_id = {
        row["logical_object_id"]: row for row in relation["records"]
    }
    eligibility: list[dict[str, object]] = []
    execution: list[dict[str, object]] = []
    for obj in objects:
        role = str(obj["role"])
        problem_id = str(neutral_problem_id if role == "N" else candidate_problem_id)
        unit_key = codecontests_s12d_evaluation_unit_key(
            str(triplet_id), role, problem_id, str(obj["selection_id"])
        )
        eligibility.append(
            {
                "evaluation_unit_key": unit_key,
                "eligibility_disposition": "eligible",
                "exclusion_reason": None,
                "panel_id": obj["panel_id"],
                "judge_request_id": obj["judge_request_id"],
            }
        )
        record = record_by_id[str(obj["logical_object_id"])]
        category = record["category"]
        execution.append(
            {
                "evaluation_unit_key": unit_key,
                "judge_request_id": obj["judge_request_id"],
                "execution_disposition": (
                    "infrastructure_error"
                    if category == "runner_infrastructure_error"
                    else "semantic_verdict"
                ),
                "semantic_verdict": (
                    category if category != "runner_infrastructure_error" else None
                ),
                "witness_test_id": record.get("witness_test_id"),
            }
        )
    if value.get("natural_member") is True and neutral["eligibility_disposition"] == "expected_exclusion":
        neutral_key = codecontests_s12d_evaluation_unit_key(
            str(triplet_id),
            "N",
            str(neutral_problem_id),
            str(manifest.get("neutral_solution_id") or "aligned-n"),
        )
        eligibility.append(
            {
                "evaluation_unit_key": neutral_key,
                "eligibility_disposition": "expected_exclusion",
                "exclusion_reason": neutral["exclusion_reason"],
                "panel_id": None,
                "judge_request_id": None,
            }
        )
    pool = build_codecontests_v3_candidate_pool(
        manifest, candidate_row, measure_tokens=measure_tokens
    )
    value["eligibility_dispositions"] = eligibility
    value["execution_dispositions"] = execution
    value["candidate_skip_counts"] = {
        code: int(pool["exclusions"].get(code, 0)) for code in _S12D_CANDIDATE_SKIP_CODES
    }
    value["neutral_problem_classification"] = {
        key: neutral[key]
        for key in ("eligibility_disposition", "exclusion_reason", "panel_id")
    }
    return value


def align_codecontests_natural_support(
    selected_pairs: Sequence[Mapping[str, object]],
    neutral_qualified_pairs: Sequence[Mapping[str, object]],
    *,
    plan_closes: bool,
) -> dict[str, object]:
    """Freeze S_N for both NAT/POLICY or keep S for NAT only."""

    selected_ids = {pair.get("triplet_id") for pair in selected_pairs}
    qualified_ids = {pair.get("triplet_id") for pair in neutral_qualified_pairs}
    if not qualified_ids.issubset(selected_ids):
        raise ValueError("neutral-qualified support is not a subset of F2 selection")
    support = evaluate_codecontests_support(
        neutral_qualified_pairs,
        arms=("I", "R", "N"),
        plan_closes=plan_closes,
    )
    if support["passed"]:
        natural_support = tuple(neutral_qualified_pairs)
        policy_support = tuple(neutral_qualified_pairs)
        policy_available = True
    else:
        natural_support = tuple(selected_pairs)
        policy_support = ()
        policy_available = False
    return {
        "natural_support": natural_support,
        "policy_support": policy_support,
        "policy_available": policy_available,
        "neutral_support_gate": support,
    }


def intersect_codecontests_mutations(
    natural_support: Sequence[Mapping[str, object]],
    mutation_records: Sequence[Mapping[str, object]],
    *,
    worklist_complete: bool,
    plan_closes: bool,
) -> dict[str, object]:
    """Intersect killed mutants with final natural support before the PC gate."""

    if not worklist_complete:
        return {
            "mutation_support": "not_evaluated",
            "support": "not_evaluated",
            "terminal": "STOPPED_CODECONTESTS_V2_RESOURCE_LIMIT",
        }
    natural_order = tuple(pair.get("triplet_id") for pair in natural_support)
    natural_ids = set(natural_order)
    by_id: dict[object, Mapping[str, object]] = {}
    for mutation in mutation_records:
        triplet_id = mutation.get("triplet_id")
        if triplet_id in by_id:
            raise ValueError("duplicate killed-mutant record")
        by_id[triplet_id] = mutation
    selected = tuple(
        by_id[triplet_id] for triplet_id in natural_order if triplet_id in by_id
    )
    support_rows = tuple(
        {
            "triplet_id": row["triplet_id"],
            "candidate_problem_id": row["candidate_problem_id"],
            "response_tokens": {
                "CLEAN": row["clean_response_tokens"],
                "MUTANT": row["mutant_response_tokens"],
            },
            "serialized_fits": row.get("serialized_fits") is True,
        }
        for row in selected
    )
    support = evaluate_codecontests_support(
        support_rows,
        arms=("CLEAN", "MUTANT"),
        plan_closes=plan_closes,
    )
    terminal = (
        None
        if support["passed"]
        else "COMPLETED_CODECONTESTS_V2_POSITIVE_CONTROL_UNAVAILABLE"
    )
    return {
        "mutation_support": selected,
        "support": support,
        "terminal": terminal,
    }
