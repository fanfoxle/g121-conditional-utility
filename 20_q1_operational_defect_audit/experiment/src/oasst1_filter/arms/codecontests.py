"""Frozen CodeContests V1 I/R/N arm reconstruction.

Descriptions and stored solutions remain inert strings throughout this module.
No function compiles, imports, executes, or otherwise interprets solution text.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence
import unicodedata

from ..data.provenance import sha256_file
from ..protocol import (
    CODECONTESTS_DESIGN_ID,
    CODECONTESTS_FORMAL_JOB_ORDER,
    CODECONTESTS_PROTOCOL_ID,
    CODECONTESTS_RESPONSE_TOKEN_QUOTA,
    CODECONTESTS_ROLE_NAMES,
    CODECONTESTS_UPDATES,
    CODECONTESTS_V2_FORMAL_JOB_ORDERS,
    CODECONTESTS_V2_POLICY_BACKGROUND_QUOTA,
    CODECONTESTS_V2_POLICY_TREATMENT_QUOTA,
    CODECONTESTS_V2_PROTOCOL_ID,
    CODECONTESTS_V2_DESIGN_ID,
    CODECONTESTS_V2_STAGE_ARMS,
    derive_codecontests_role_seed,
    derive_codecontests_v2_role_seed,
)


CODECONTESTS_MANIFEST_SHA256 = (
    "0aa3eee089e6779c2edae23891e1ff91b082fedccb5fb04ec18e0b27fae3a1e9"
)
CODECONTESTS_TRIPLETS = 590
CODECONTESTS_ARM_TOKEN_MASSES = {"I": 194_998, "R": 196_901, "N": 196_196}
CODECONTESTS_PROMPT_PREFIX = (
    "Solve the following programming problem in Python 3. Return only the "
    "complete program, with no Markdown fences.\n\n"
)
CODECONTESTS_PROMPT_SUFFIX = "\n\nPython 3 solution:\n"
_PROBLEM_ID = re.compile(r"train:(0|[1-9][0-9]*)\Z")
_SOLUTION_ID = re.compile(
    r"train:(0|[1-9][0-9]*):(correct|incorrect):(0|[1-9][0-9]*)\Z"
)


class CodeContestsTokenizer(Protocol):
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


@dataclass(frozen=True)
class CodeContestsUnit:
    unit_id: str
    triplet_id: str
    arm: str
    problem_id: str
    solution_id: str
    problem_fingerprint: str
    solution_fingerprint: str
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    prompt_tokens: int
    response_tokens: int


def qualify_scope_pc_pair(prompt: str, source_r: str, source_m: str, tokenizer) -> dict[str, object]:
    """Static SCOPE eligibility; training serialization retains the original consumer boundary."""
    if not all(isinstance(text, str) and text for text in (prompt, source_r, source_m)):
        raise ValueError("SCOPE source/prompt is malformed")
    if source_r == source_m:
        raise ValueError("SCOPE selected intervention is identical")
    lines_r, lines_m = source_r.splitlines(keepends=True), source_m.splitlines(keepends=True)
    common = []
    for left, right in zip(lines_r, lines_m):
        if left != right:
            break
        common.append(left)
    prefix = "".join(common)
    suffix = source_r[len(prefix):]
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    targets = {arm: list(tokenizer.encode(source, add_special_tokens=False)) + [tokenizer.eos_token_id]
               for arm, source in (("R", source_r), ("M", source_m))}
    sequences = {arm: prompt_ids + ids for arm, ids in targets.items()}
    r, m = sequences["R"], sequences["M"]
    k = 0
    while k < min(len(r), len(m)) and r[k] == m[k]:
        k += 1
    tail = 0
    while tail < min(len(r), len(m)) - k and r[-1-tail] == m[-1-tail]:
        tail += 1
    suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    if tokenizer.decode(suffix_ids, clean_up_tokenization_spaces=False, skip_special_tokens=False) != suffix:
        raise ValueError("SCOPE reference suffix decoding is not reversible")
    prefill = list(tokenizer.encode(prompt + prefix, add_special_tokens=True))
    reasons = []
    if any(len(ids) > 2048 for ids in sequences.values()):
        reasons.append("training_sequence_over_2048")
    if len(prefill) > 2048:
        reasons.append("prefill_over_2048")
    if len(suffix_ids) > 1024:
        reasons.append("reference_suffix_over_1024")
    if k == 0 or k >= min(len(r), len(m)):
        reasons.append("no_two_next_token_targets")
    return {"eligible": not reasons, "reasons": reasons, "code_prefix": prefix,
            "reference_suffix_tokens": len(suffix_ids), "reference_full_tokens": len(targets["R"]) - 1,
            "prefill_token_ids": prefill, "full_prompt_token_ids": prompt_ids,
            "token_prefix_length": k, "common_suffix_tokens": tail,
            "token_ids": sequences, "prompt_tokens": len(prompt_ids),
            "difference_envelopes": {arm: [k, len(ids) - tail] for arm, ids in sequences.items()}}


def build_branch_pc_input(pair: Mapping[str, object], tokenizer) -> dict[str, object]:
    """Verify frozen separate encoding and expose no shared reference suffix."""
    if pair.get("role") != "D":
        raise ValueError("BRANCH inputs are D-only")
    prompt = list(tokenizer.encode(pair["prompt"], add_special_tokens=True))
    sequences = {}
    for arm in ("R", "M"):
        source = pair[f"source_{arm.lower()}"]
        response = list(tokenizer.encode(source, add_special_tokens=False))
        sequence = prompt + response + [tokenizer.eos_token_id]
        if sequence != pair["token_ids"][arm] or len(prompt) != pair["prompt_tokens"]:
            raise ValueError("BRANCH frozen tokens differ from deterministic encoding")
        if tokenizer.decode(response, clean_up_tokenization_spaces=False, skip_special_tokens=False) != source:
            raise ValueError("BRANCH whole-response round-trip failed")
        sequences[arm] = sequence
    r, m = sequences["R"], sequences["M"]
    k = 0
    while k < min(len(r), len(m)) and r[k] == m[k]:
        k += 1
    tail = 0
    while tail < min(len(r), len(m)) - k and r[-1-tail] == m[-1-tail]:
        tail += 1
    envelopes = {arm: ids[k:len(ids)-tail] for arm, ids in sequences.items()}
    special = set(tokenizer.all_special_ids)
    if k < len(prompt) or any(not e or len(e) >= 1024 or special.intersection(e) for e in envelopes.values()):
        raise ValueError("BRANCH invalid difference envelope")
    u = r[:k]
    return {"pair_id": pair["pair_id"], "group_id": pair["group_id"],
            "u": u, "prompt_tokens": len(prompt), "envelopes": envelopes,
            "aR": r[k], "aM": m[k],
            "conditions": {condition: {"input_ids": u + clamp, "max_new_tokens": 1024-len(clamp)}
                           for condition, clamp in (("U", []), ("CR", envelopes["R"]), ("CM", envelopes["M"]))}}


def _lf(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("CodeContests text must be a Unicode string")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFC", _lf(text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_codecontests_manifest(
    path: Path,
    *,
    expected_sha256: str = CODECONTESTS_MANIFEST_SHA256,
    expected_triplets: int = CODECONTESTS_TRIPLETS,
) -> tuple[dict[str, object], ...]:
    """Read the frozen metadata-only triplet manifest and fail closed."""

    if sha256_file(path) != expected_sha256:
        raise ValueError("CodeContests manifest SHA-256 mismatch")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank CodeContests manifest row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("CodeContests manifest rows must be objects")
            rows.append(value)
    if len(rows) != expected_triplets:
        raise ValueError("CodeContests manifest triplet count mismatch")
    identifiers = [row.get("triplet_id") for row in rows]
    expected_ids = [f"triplet:{index}" for index in range(expected_triplets)]
    if identifiers != expected_ids:
        raise ValueError("CodeContests manifest triplet IDs or order differ from freeze")
    return tuple(rows)


def _problem_index(value: object, *, role: str) -> int:
    if not isinstance(value, str) or (matched := _PROBLEM_ID.fullmatch(value)) is None:
        raise ValueError(f"{role} problem identity is invalid")
    return int(matched.group(1))


def _solution_identity(
    value: object, *, role: str, expected_problem: int, expected_label: str
) -> tuple[str, int]:
    if not isinstance(value, str) or (matched := _SOLUTION_ID.fullmatch(value)) is None:
        raise ValueError(f"{role} solution identity is invalid")
    problem_index, label, solution_index = matched.groups()
    if int(problem_index) != expected_problem:
        raise ValueError(f"{role} problem identity differs from solution identity")
    if label != expected_label:
        raise ValueError(f"{role} label differs from frozen arm semantics")
    return label, int(solution_index)


def _parallel_arrays(value: object, *, role: str) -> tuple[Sequence[object], Sequence[object]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} stored solution field is malformed")
    languages = value.get("language")
    solutions = value.get("solution")
    if (
        not isinstance(languages, (list, tuple))
        or not isinstance(solutions, (list, tuple))
        or len(languages) != len(solutions)
    ):
        raise ValueError(f"{role} stored solution arrays are malformed")
    return languages, solutions


def _serialized_unit(
    *,
    triplet_id: str,
    arm: str,
    role: str,
    row: Mapping[str, object],
    problem_id: str,
    solution_id: str,
    problem_fingerprint: str,
    solution_fingerprint: str,
    solution_index: int,
    label: str,
    expected_prompt_tokens: object,
    expected_response_tokens: object,
    tokenizer: CodeContestsTokenizer,
) -> CodeContestsUnit:
    description = row.get("description")
    if not isinstance(description, str):
        raise ValueError(f"{role} description is not a string")
    if _fingerprint(description) != problem_fingerprint:
        raise ValueError(f"{role} problem fingerprint mismatch")
    field = "solutions" if label == "correct" else "incorrect_solutions"
    languages, solutions = _parallel_arrays(row.get(field), role=role)
    if solution_index >= len(solutions):
        raise ValueError(f"{role} solution index is outside the stored array")
    if languages[solution_index] != 3:
        raise ValueError(f"{role} label does not select an exact PYTHON3 solution")
    code = solutions[solution_index]
    if not isinstance(code, str):
        raise ValueError(f"{role} stored solution is not a string")
    if _fingerprint(code) != solution_fingerprint:
        raise ValueError(f"{role} solution fingerprint mismatch")

    prompt = CODECONTESTS_PROMPT_PREFIX + _lf(description) + CODECONTESTS_PROMPT_SUFFIX
    response = _lf(code)
    prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
    response_content_ids = tuple(tokenizer.encode(response, add_special_tokens=False))
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, bool) or not isinstance(eos_id, int):
        raise ValueError("CodeContests tokenizer EOS ID must be an integer")
    response_ids = response_content_ids + (eos_id,)
    if len(prompt_ids) != expected_prompt_tokens or len(response_ids) != expected_response_tokens:
        raise ValueError(f"{role} token count differs from frozen manifest")
    if not response.strip() or len(prompt_ids) + len(response_ids) > 2_048:
        raise ValueError(f"{role} unit violates the frozen admission boundary")
    return CodeContestsUnit(
        unit_id=f"{triplet_id}/{arm}",
        triplet_id=triplet_id,
        arm=arm,
        problem_id=problem_id,
        solution_id=solution_id,
        problem_fingerprint=problem_fingerprint,
        solution_fingerprint=solution_fingerprint,
        token_ids=prompt_ids + response_ids,
        loss_mask=(False,) * len(prompt_ids) + (True,) * len(response_ids),
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
    )


def materialize_codecontests_rows(
    manifest_rows: Sequence[Mapping[str, object]],
    rows_by_index: Mapping[int, Mapping[str, object]],
    tokenizer: CodeContestsTokenizer,
    *,
    expected_triplets: int = CODECONTESTS_TRIPLETS,
) -> dict[str, tuple[CodeContestsUnit, ...]]:
    """Reconstruct exactly one inert string unit per triplet and arm."""

    if len(manifest_rows) != expected_triplets:
        raise ValueError("CodeContests manifest triplet count mismatch")
    arms: dict[str, list[CodeContestsUnit]] = {"I": [], "R": [], "N": []}
    seen_solutions: dict[str, set[str]] = {"I": set(), "R": set(), "N": set()}
    seen_neutral_problems: set[str] = set()
    for ordinal, manifest in enumerate(manifest_rows):
        triplet_id = manifest.get("triplet_id")
        if triplet_id != f"triplet:{ordinal}":
            raise ValueError("CodeContests manifest triplet IDs or order differ from freeze")
        candidate_index = _problem_index(
            manifest.get("candidate_problem_id"), role="candidate"
        )
        repair_index = _problem_index(manifest.get("repair_problem_id"), role="repair")
        neutral_index = _problem_index(manifest.get("neutral_problem_id"), role="neutral")
        if repair_index != candidate_index:
            raise ValueError("repair problem differs from candidate problem")
        if neutral_index == candidate_index:
            raise ValueError("neutral problem must differ from candidate problem")
        candidate_label, candidate_solution_index = _solution_identity(
            manifest.get("candidate_solution_id"),
            role="candidate",
            expected_problem=candidate_index,
            expected_label="incorrect",
        )
        repair_label, repair_solution_index = _solution_identity(
            manifest.get("repair_solution_id"),
            role="repair",
            expected_problem=repair_index,
            expected_label="correct",
        )
        neutral_label, neutral_solution_index = _solution_identity(
            manifest.get("neutral_solution_id"),
            role="neutral",
            expected_problem=neutral_index,
            expected_label="correct",
        )
        neutral_problem_id = str(manifest["neutral_problem_id"])
        neutral_solution_id = str(manifest["neutral_solution_id"])
        if neutral_problem_id in seen_neutral_problems or neutral_solution_id in seen_solutions["N"]:
            raise ValueError("neutral solution or problem is reused")
        seen_neutral_problems.add(neutral_problem_id)

        for role, arm, row_index, label, solution_index in (
            ("candidate", "I", candidate_index, candidate_label, candidate_solution_index),
            ("repair", "R", repair_index, repair_label, repair_solution_index),
            ("neutral", "N", neutral_index, neutral_label, neutral_solution_index),
        ):
            row = rows_by_index.get(row_index)
            if row is None:
                raise ValueError(f"{role} problem row is absent")
            if row.get("source") != manifest.get("source_value") or row.get(
                "difficulty"
            ) != manifest.get("difficulty_value"):
                raise ValueError(f"{role} source/difficulty differs from manifest")
            solution_id = str(manifest[f"{role}_solution_id"])
            if solution_id in seen_solutions[arm]:
                raise ValueError(f"{role} solution is reused")
            seen_solutions[arm].add(solution_id)
            unit = _serialized_unit(
                triplet_id=str(triplet_id),
                arm=arm,
                role=role,
                row=row,
                problem_id=str(manifest[f"{role}_problem_id"]),
                solution_id=solution_id,
                problem_fingerprint=str(manifest[f"{role}_problem_fingerprint"]),
                solution_fingerprint=str(manifest[f"{role}_solution_fingerprint"]),
                solution_index=solution_index,
                label=label,
                expected_prompt_tokens=manifest.get(f"{role}_prompt_tokens"),
                expected_response_tokens=manifest.get(f"{role}_response_tokens"),
                tokenizer=tokenizer,
            )
            arms[arm].append(unit)
    frozen = {arm: tuple(units) for arm, units in arms.items()}
    if expected_triplets == CODECONTESTS_TRIPLETS:
        observed = {
            arm: sum(unit.response_tokens for unit in units)
            for arm, units in frozen.items()
        }
        if observed != CODECONTESTS_ARM_TOKEN_MASSES:
            raise ValueError("CodeContests arm token masses differ from freeze")
    return frozen


def _arm_identity(units: Sequence[CodeContestsUnit]) -> dict[str, object]:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for ordinal, unit in enumerate(units):
        if unit.unit_id in seen or unit.triplet_id != f"triplet:{ordinal}":
            raise ValueError("CodeContests arm unit identities or order differ from freeze")
        seen.add(unit.unit_id)
        digest.update(
            (
                f"{unit.unit_id}|{unit.problem_id}|{unit.solution_id}|"
                f"{unit.problem_fingerprint}|{unit.solution_fingerprint}|"
                f"{unit.prompt_tokens}|{unit.response_tokens}\n"
            ).encode("ascii")
        )
    return {
        "record_count": len(units),
        "response_loss_tokens": sum(unit.response_tokens for unit in units),
        "metadata_sha256": digest.hexdigest(),
    }


def build_codecontests_run_cards(
    arms: Mapping[str, Sequence[CodeContestsUnit]],
    *,
    enforce_frozen_masses: bool = True,
) -> tuple[dict[str, object], ...]:
    """Return the closed 24-card queue from validated arm identities."""

    if tuple(arms) != ("I", "R", "N"):
        raise ValueError("CodeContests arm registry must be ordered I/R/N")
    identities = {arm: _arm_identity(arms[arm]) for arm in arms}
    if any(identity["record_count"] != CODECONTESTS_TRIPLETS for identity in identities.values()):
        raise ValueError("CodeContests run cards require exactly 590 units per arm")
    if enforce_frozen_masses and {
        arm: identity["response_loss_tokens"] for arm, identity in identities.items()
    } != CODECONTESTS_ARM_TOKEN_MASSES:
        raise ValueError("CodeContests run-card arm token masses differ from freeze")

    cards: list[dict[str, object]] = []
    for ordinal, (root, arm) in enumerate(CODECONTESTS_FORMAL_JOB_ORDER):
        cards.append(
            {
                "schema": "codecontests-v1-formal-run-card-v1",
                "protocol_id": CODECONTESTS_PROTOCOL_ID,
                "design_id": CODECONTESTS_DESIGN_ID,
                "training_ordinal": ordinal,
                "root": root,
                "arm": arm,
                "arm_identity": dict(identities[arm]),
                "response_token_quota": CODECONTESTS_RESPONSE_TOKEN_QUOTA,
                "updates": CODECONTESTS_UPDATES,
                "max_length": 2_048,
                "packing_path": "single_unit",
                "role_seeds": {
                    role: derive_codecontests_role_seed(root, role)
                    for role in CODECONTESTS_ROLE_NAMES
                },
                "outcomes_opened": False,
            }
        )
    return tuple(cards)


def build_codecontests_v2_run_cards(
    stage: str,
    plan_closures: Mapping[tuple[int, str], Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Freeze one complete V2 stage queue from pre-outcome plan closures."""

    if stage not in CODECONTESTS_V2_FORMAL_JOB_ORDERS:
        raise ValueError("unknown CodeContests V2 stage")
    order = CODECONTESTS_V2_FORMAL_JOB_ORDERS[stage]
    if set(plan_closures) != set(order) or len(plan_closures) != len(order):
        raise ValueError("CodeContests V2 plan-closure registry is incomplete")
    sha_pattern = re.compile(r"[0-9a-f]{64}\Z")
    normalized: dict[tuple[int, str], dict[str, object]] = {}
    for job in order:
        closure = dict(plan_closures[job])
        if (
            closure.get("complete") is not True
            or closure.get("response_token_quota") != CODECONTESTS_RESPONSE_TOKEN_QUOTA
            or closure.get("updates") != CODECONTESTS_UPDATES
            or not isinstance(closure.get("support_sha256"), str)
            or sha_pattern.fullmatch(str(closure["support_sha256"])) is None
            or not isinstance(closure.get("plan_sha256"), str)
            or sha_pattern.fullmatch(str(closure["plan_sha256"])) is None
        ):
            raise ValueError("CodeContests V2 plan closure is invalid")
        if stage == "policy-filter" and closure.get("component_response_tokens") != {
            "B80": CODECONTESTS_V2_POLICY_BACKGROUND_QUOTA,
            "T20": CODECONTESTS_V2_POLICY_TREATMENT_QUOTA,
        }:
            raise ValueError("CodeContests V2 policy component quota is invalid")
        normalized[job] = closure
    for root in {root for root, _ in order}:
        root_closures = [
            normalized[(root, arm)] for arm in CODECONTESTS_V2_STAGE_ARMS[stage]
        ]
        if len({item["support_sha256"] for item in root_closures}) != 1:
            raise ValueError("CodeContests V2 paired support identity differs by arm")
        if stage == "policy-filter":
            b80 = {item.get("b80_sha256") for item in root_closures}
            if len(b80) != 1 or sha_pattern.fullmatch(str(next(iter(b80)))) is None:
                raise ValueError("CodeContests V2 shared B80 identity differs by arm")

    cards: list[dict[str, object]] = []
    for ordinal, (root, arm) in enumerate(order):
        closure = normalized[(root, arm)]
        cards.append(
            {
                "schema": "codecontests-v2-formal-run-card-v1",
                "protocol_id": CODECONTESTS_V2_PROTOCOL_ID,
                "design_id": CODECONTESTS_V2_DESIGN_ID,
                "stage": stage,
                "training_ordinal": ordinal,
                "root": root,
                "arm": arm,
                "plan_closure": closure,
                "response_token_quota": CODECONTESTS_RESPONSE_TOKEN_QUOTA,
                "updates": CODECONTESTS_UPDATES,
                "max_length": 2_048,
                "packing_path": "single_unit",
                "role_seeds": {
                    role: derive_codecontests_v2_role_seed(stage, root, role)
                    for role in CODECONTESTS_ROLE_NAMES
                },
                "outcomes_opened": False,
            }
        )
    return tuple(cards)


def referenced_codecontests_row_indices(
    manifest_rows: Sequence[Mapping[str, object]],
) -> frozenset[int]:
    return frozenset(
        _problem_index(row.get(f"{role}_problem_id"), role=role)
        for row in manifest_rows
        for role in ("candidate", "repair", "neutral")
    )


def load_referenced_codecontests_rows(
    dataset_root: Path, manifest_rows: Sequence[Mapping[str, object]]
) -> dict[int, dict[str, object]]:
    """Project only allowed train columns for rows referenced by the manifest."""

    import pyarrow.parquet as pq

    paths = tuple(sorted((dataset_root / "data").glob("train-*.parquet")))
    if len(paths) != 39:
        raise ValueError("CodeContests train shard count differs from freeze")
    wanted = referenced_codecontests_row_indices(manifest_rows)
    rows: dict[int, dict[str, object]] = {}
    offset = 0
    columns = [
        "description",
        "source",
        "difficulty",
        "solutions",
        "incorrect_solutions",
    ]
    for path in paths:
        row_count = pq.ParquetFile(path).metadata.num_rows
        selected = sorted(index for index in wanted if offset <= index < offset + row_count)
        if selected:
            table = pq.read_table(path, columns=columns)
            for index in selected:
                rows[index] = table.slice(index - offset, 1).to_pylist()[0]
        offset += row_count
    if offset != 13_328 or set(rows) != set(wanted):
        raise ValueError("CodeContests referenced row reconstruction is incomplete")
    return rows
