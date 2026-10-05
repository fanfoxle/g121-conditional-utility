"""S12-APCF configuration, roots, masks, and paired arm materialization.

This module is the one S12-specific arm responsibility.  It deliberately sits
outside the byte-bound V3b measurement device and exposes a small, pure
interface for pre-outcome construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Mapping, Sequence

from ..train.stream import (
    Exposure,
    ExposureLedgerEntry,
    TokenizedUnit,
    TrainingPlan,
    single_unit_packs,
)
from ..protocol import (
    CODECONTESTS_ROLE_NAMES,
    CODECONTESTS_S12D_TERMINAL_STATES,
    derive_codecontests_role_seed,
)


_STAGES = ("PC", "NAT", "POLICY")
SCOPE_PC_PROTOCOL_ID = "codecontests-scope-pc-diagnostic-v1"
NAT_REPLACE_PROTOCOL_ID = "codecontests-nat-replace-20-v1"
NAT_REPLICA_PROTOCOL_ID = "codecontests-nat-replica-x164-v1"
NAT_SPEC_PROTOCOL_ID = "codecontests-nat-spec-20-v1"
NAT_SPEC_QUALIFICATION_NAMESPACE = (
    "codecontests-nat-spec-20-v1/qualification-sample-v1"
)
NAT_MSEP_PROTOCOL_ID = "codecontests-nat-msep-20-v1"
NAT_MSEP_QUALIFICATION_NAMESPACE = (
    "codecontests-nat-msep-20-v1/qualification-mask-v1"
)
NAT_MSEP_FORMAL_ROOT_NAMESPACE = "codecontests-nat-msep-20-v1/formal-root-v1"
NAT_MSEP_FORMAL_MASK_NAMESPACE = "codecontests-nat-msep-20-v1/formal-mask-v1"
NAT_ONOFF_PROTOCOL_ID = "q1-nat-onoff-354-v1"
NAT_ONOFF_ROOT_NAMESPACE = "q1-nat-onoff-354-v1/formal-root-v1"
_NAT_SPEC_ANON_ID = re.compile(r"r[1-9][0-9]*:u(?:0|[1-9][0-9]*):e(?:0|[1-9][0-9]*)")


@dataclass(frozen=True)
class NatSpecAnonymousExposure:
    """The complete label-blind instance visible to the NAT-SPEC sampler."""

    anon_id: str
    m_e: int


NatMsepPayloadType = tuple[tuple[int, ...], tuple[bool, ...]]


@dataclass(frozen=True)
class NatMsepAnonymousExposure:
    """The complete outcome/source-blind instance visible to NAT-MSEP."""

    anon_id: str
    m_e: int
    is_target: bool
    payload_type: NatMsepPayloadType


class NatSpecQualificationError(ValueError):
    """A scientific qualification failure with one frozen reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class NatSpecRootQualification:
    root: int
    update_rows: tuple[dict[str, object], ...]
    keep_tokens: int
    filtered_tokens: int
    target_tokens: int
    donor_tokens: int
    alpha: Fraction
    repeated_exposure_count: int
    sample_fully_removed_source_records: int
    sample_partially_removed_source_records: int
    sample_retained_source_records: int


class NatMsepQualificationError(ValueError):
    """A pre-outcome NAT-MSEP qualification failure with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code

    def __reduce__(self):
        return (type(self), (self.reason_code, str(self)))


class NatMsepDeadlineExceeded(RuntimeError):
    """The non-resetting qualification deadline was reached between updates."""


@dataclass(frozen=True)
class NatMsepRootQualification:
    root: int
    update_rows: tuple[dict[str, object], ...]
    root_row: dict[str, object]
    selected_ordinals_by_update: tuple[tuple[int, ...], ...]
    source_diagnostics: dict[str, object]
    msep_plan: TrainingPlan | None = None


class NatSpecSubsetSupport:
    """Exact finite support for one update's constrained exposure subsets."""

    def __init__(
        self,
        exposures: Sequence[NatSpecAnonymousExposure],
        *,
        k: int,
        q: int,
    ) -> None:
        values = tuple(exposures)
        if not values:
            raise ValueError("NAT-SPEC requires at least one eligible exposure")
        if any(type(item) is not NatSpecAnonymousExposure for item in values):
            raise TypeError("NAT-SPEC sampler accepts anonymous exposures only")
        if any(_NAT_SPEC_ANON_ID.fullmatch(item.anon_id) is None for item in values):
            raise ValueError("NAT-SPEC anonymous exposure identity is invalid")
        if len({item.anon_id for item in values}) != len(values):
            raise ValueError("NAT-SPEC anonymous exposure identities must be unique")
        if any(type(item.m_e) is not int or item.m_e <= 0 for item in values):
            raise ValueError("NAT-SPEC eligible exposure token counts must be positive")
        if type(k) is not int or not 0 <= k <= len(values):
            raise ValueError("NAT-SPEC subset count k is invalid")
        if type(q) is not int or q < 0:
            raise ValueError("NAT-SPEC subset token budget q is invalid")
        self.exposures = values
        self.k = k
        self.q = q
        self._counts: dict[tuple[int, int, int], int] = {}
        self._ordinals = {item.anon_id: index for index, item in enumerate(values)}
        self.omega = self._count(0, k, q)
        if self.omega <= 0:
            raise ValueError("NAT-SPEC constrained subset is infeasible")

    def _count(self, start: int, k: int, q: int) -> int:
        key = (start, k, q)
        if key in self._counts:
            return self._counts[key]
        if k == 0:
            result = int(q == 0)
        elif q < 0 or len(self.exposures) - start < k:
            result = 0
        else:
            result = self._count(start + 1, k, q)
            result += self._count(
                start + 1,
                k - 1,
                q - self.exposures[start].m_e,
            )
        self._counts[key] = result
        return result

    def unrank(self, rank: int) -> tuple[str, ...]:
        """Return the lexicographically ranked ascending anonymous-ID tuple."""

        if type(rank) is not int or not 0 <= rank < self.omega:
            raise ValueError("NAT-SPEC subset rank is outside its exact support")
        selected: list[str] = []
        start = 0
        remaining_k = self.k
        remaining_q = self.q
        while remaining_k:
            for ordinal in range(start, len(self.exposures) - remaining_k + 1):
                exposure = self.exposures[ordinal]
                completions = self._count(
                    ordinal + 1,
                    remaining_k - 1,
                    remaining_q - exposure.m_e,
                )
                if rank < completions:
                    selected.append(exposure.anon_id)
                    start = ordinal + 1
                    remaining_k -= 1
                    remaining_q -= exposure.m_e
                    break
                rank -= completions
            else:  # pragma: no cover - guarded by exact omega/rank validation
                raise AssertionError("NAT-SPEC unrank failed to close")
        if remaining_q != 0:
            raise AssertionError("NAT-SPEC unrank token budget did not close")
        return tuple(selected)

    def rank(self, subset: Sequence[str]) -> int:
        """Return the exact lexicographic rank of an ascending feasible subset."""

        identities = tuple(subset)
        try:
            ordinals = tuple(self._ordinals[value] for value in identities)
        except KeyError as exc:
            raise ValueError("NAT-SPEC subset contains an unknown anonymous identity") from exc
        if (
            len(ordinals) != self.k
            or tuple(sorted(ordinals)) != ordinals
            or len(set(ordinals)) != len(ordinals)
        ):
            raise ValueError("NAT-SPEC subset must be one ascending k-tuple")
        rank = 0
        start = 0
        remaining_k = self.k
        remaining_q = self.q
        for selected in ordinals:
            for ordinal in range(start, selected):
                rank += self._count(
                    ordinal + 1,
                    remaining_k - 1,
                    remaining_q - self.exposures[ordinal].m_e,
                )
            remaining_q -= self.exposures[selected].m_e
            remaining_k -= 1
            start = selected + 1
        if remaining_q != 0:
            raise ValueError("NAT-SPEC subset does not close q")
        if not 0 <= rank < self.omega or self.unrank(rank) != identities:
            raise ValueError("NAT-SPEC subset is not in the exact support")
        return rank

    def inclusion_count(self, anon_id: str) -> int:
        """Count exact feasible subsets containing one anonymous instance."""

        if anon_id not in self._ordinals:
            raise ValueError("NAT-SPEC inclusion identity is outside the support")
        selected = self._ordinals[anon_id]
        target_q = self.q - self.exposures[selected].m_e
        if self.k == 0 or target_q < 0:
            return 0
        counts = {(0, 0): 1}
        for ordinal, exposure in enumerate(self.exposures):
            if ordinal == selected:
                continue
            updated = dict(counts)
            for (count, tokens), ways in counts.items():
                if count < self.k - 1 and tokens + exposure.m_e <= target_q:
                    key = (count + 1, tokens + exposure.m_e)
                    updated[key] = updated.get(key, 0) + ways
            counts = updated
        return counts.get((self.k - 1, target_q), 0)


def build_nat_spec_subset_support(
    exposures: Sequence[NatSpecAnonymousExposure], *, k: int, q: int
) -> NatSpecSubsetSupport:
    """Build the exact label-blind finite support for one update."""

    return NatSpecSubsetSupport(exposures, k=k, q=q)


class NatMsepSubsetSupport:
    """Exact feasible and lexicographically joint-optimal NAT-MSEP support."""

    def __init__(
        self,
        exposures: Sequence[NatMsepAnonymousExposure],
        *,
        k: int,
        q: int,
    ) -> None:
        values = tuple(exposures)
        if not values:
            raise ValueError("NAT-MSEP requires at least one eligible exposure")
        if any(type(item) is not NatMsepAnonymousExposure for item in values):
            raise TypeError("NAT-MSEP selector accepts anonymous exposures only")
        if any(_NAT_SPEC_ANON_ID.fullmatch(item.anon_id) is None for item in values):
            raise ValueError("NAT-MSEP anonymous exposure identity is invalid")
        if len({item.anon_id for item in values}) != len(values):
            raise ValueError("NAT-MSEP anonymous exposure identities must be unique")
        if any(type(item.m_e) is not int or item.m_e <= 0 for item in values):
            raise ValueError("NAT-MSEP exposure token counts must be positive")
        if any(type(item.is_target) is not bool for item in values):
            raise TypeError("NAT-MSEP I-membership must be boolean")
        for item in values:
            payload = item.payload_type
            if (
                type(payload) is not tuple
                or len(payload) != 2
                or type(payload[0]) is not tuple
                or type(payload[1]) is not tuple
                or any(type(token) is not int for token in payload[0])
                or any(type(flag) is not bool for flag in payload[1])
                or len(payload[0]) != len(payload[1]) + 1
            ):
                raise TypeError("NAT-MSEP payload type must be exact tokens plus shifted mask")
        self.exposures = values
        self.k = k
        self.q = q
        base = tuple(
            NatSpecAnonymousExposure(item.anon_id, item.m_e) for item in values
        )
        self._feasible = build_nat_spec_subset_support(base, k=k, q=q)
        self.omega = self._feasible.omega
        by_identity = {item.anon_id: item for item in values}
        target_payload: dict[NatMsepPayloadType, int] = {}
        for item in values:
            if item.is_target:
                target_payload[item.payload_type] = (
                    target_payload.get(item.payload_type, 0) + item.m_e
                )

        optimum_id: int | None = None
        optimum_payload: int | None = None
        omega1 = 0
        joint: list[tuple[str, ...]] = []
        for feasible_rank in range(self.omega):
            subset = self._feasible.unrank(feasible_rank)
            selected = tuple(by_identity[identity] for identity in subset)
            id_overlap = sum(item.m_e for item in selected if item.is_target)
            if optimum_id is None or id_overlap < optimum_id:
                optimum_id = id_overlap
                optimum_payload = None
                omega1 = 0
                joint.clear()
            if id_overlap != optimum_id:
                continue
            omega1 += 1
            selected_payload: dict[NatMsepPayloadType, int] = {}
            for item in selected:
                selected_payload[item.payload_type] = (
                    selected_payload.get(item.payload_type, 0) + item.m_e
                )
            payload_overlap = sum(
                min(weight, target_payload.get(payload, 0))
                for payload, weight in selected_payload.items()
            )
            if optimum_payload is None or payload_overlap < optimum_payload:
                optimum_payload = payload_overlap
                joint = [subset]
            elif payload_overlap == optimum_payload:
                joint.append(subset)
        if optimum_id is None or optimum_payload is None or not joint:
            raise AssertionError("NAT-MSEP exact support failed to close")
        self.omega1 = omega1
        self.omega2 = len(joint)
        self.optimum_id_overlap = optimum_id
        self.optimum_payload_overlap = optimum_payload
        self._joint = tuple(joint)
        self._joint_ranks = {subset: rank for rank, subset in enumerate(self._joint)}

    def unrank(self, rank: int) -> tuple[str, ...]:
        """Return one canonical ascending-ordinal tuple in Omega2."""

        if type(rank) is not int or not 0 <= rank < self.omega2:
            raise ValueError("NAT-MSEP rank is outside Omega2")
        return self._joint[rank]

    def rank(self, subset: Sequence[str]) -> int:
        """Return the exact canonical rank of a joint-optimal tuple."""

        identities = tuple(subset)
        try:
            return self._joint_ranks[identities]
        except KeyError as exc:
            raise ValueError("NAT-MSEP subset is outside Omega2") from exc


def build_nat_msep_subset_support(
    exposures: Sequence[NatMsepAnonymousExposure], *, k: int, q: int
) -> NatMsepSubsetSupport:
    """Build exact Omega/Omega1/Omega2 without source or outcome fields."""

    return NatMsepSubsetSupport(exposures, k=k, q=q)


def build_nat_msep_formal_roots(
    *, seed_hex: str, prior_roots: Sequence[int]
) -> tuple[int, ...]:
    """Derive the one eight-root formal draw by exact SHA-256 rejection."""

    if (
        type(seed_hex) is not str
        or len(seed_hex) != 64
        or seed_hex != seed_hex.lower()
        or any(character not in "0123456789abcdef" for character in seed_hex)
    ):
        raise ValueError("NAT-MSEP formal root seed must be lowercase 256-bit hex")
    used: set[int] = set()
    for root in prior_roots:
        if type(root) is not int or root <= 0:
            raise ValueError("NAT-MSEP prior-root registry is invalid")
        used.add(root)
    roots: list[int] = []
    for index in range(1, 9):
        attempt = 0
        while True:
            payload = (
                f"{NAT_MSEP_FORMAL_ROOT_NAMESPACE}|{seed_hex}|{index}|{attempt}"
            ).encode("utf-8")
            candidate = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") >> 1
            if 1 <= candidate < 2_147_483_647 and candidate not in used:
                roots.append(candidate)
                used.add(candidate)
                break
            attempt += 1
    return tuple(roots)


def _bounded_sha_qualification_rank(
    *, namespace: str, root: int, update: int, omega: int, protocol_name: str
) -> int:
    if type(root) is not int or root <= 0:
        raise ValueError(f"{protocol_name} qualification root must be positive")
    if type(update) is not int or update < 0:
        raise ValueError(f"{protocol_name} qualification update must be nonnegative")
    if type(omega) is not int or omega <= 0:
        raise ValueError(f"{protocol_name} qualification omega must be positive")
    bits = omega.bit_length()
    byte_count = (bits + 7) // 8
    counter = 0
    while True:
        stream = bytearray()
        block = 0
        while len(stream) < byte_count:
            payload = (
                f"{namespace}|{root}|{update}|"
                f"{counter}|{block}"
            )
            stream.extend(hashlib.sha256(payload.encode("utf-8")).digest())
            block += 1
        excess = byte_count * 8 - bits
        candidate = int.from_bytes(stream[:byte_count], "big") >> excess
        if candidate < omega:
            return candidate
        counter += 1


def nat_spec_qualification_rank(*, root: int, update: int, omega: int) -> int:
    """Draw one deterministic uniform NAT-SPEC rank using bounded SHA256."""

    return _bounded_sha_qualification_rank(
        namespace=NAT_SPEC_QUALIFICATION_NAMESPACE,
        root=root,
        update=update,
        omega=omega,
        protocol_name="NAT-SPEC",
    )


def nat_msep_qualification_rank(*, root: int, update: int, omega: int) -> int:
    """Draw one deterministic uniform Omega2 rank using bounded SHA256."""

    return _bounded_sha_qualification_rank(
        namespace=NAT_MSEP_QUALIFICATION_NAMESPACE,
        root=root,
        update=update,
        omega=omega,
        protocol_name="NAT-MSEP",
    )


def nat_msep_formal_rank(
    *, namespace: str, root: int, update: int, omega: int
) -> int:
    """Draw one uniform Omega2 rank in the claimed formal mask namespace."""

    prefix = NAT_MSEP_FORMAL_MASK_NAMESPACE + "/"
    seed_hex = namespace[len(prefix) :] if namespace.startswith(prefix) else ""
    if (
        len(seed_hex) != 64
        or seed_hex != seed_hex.lower()
        or any(character not in "0123456789abcdef" for character in seed_hex)
    ):
        raise ValueError("NAT-MSEP formal mask namespace is invalid")
    return _bounded_sha_qualification_rank(
        namespace=namespace,
        root=root,
        update=update,
        omega=omega,
        protocol_name="NAT-MSEP formal",
    )


def _nat_spec_exposure_tokens(exposure: Exposure) -> int:
    tokens = sum(exposure.loss_mask[1:])
    if tokens <= 0 or tokens != exposure.retained_response_tokens:
        raise NatSpecQualificationError(
            "budget_or_component_contract_drift",
            "NAT-SPEC exposure loss mask and retained token count differ",
        )
    return tokens


def _nat_spec_exposure_identity(exposure: Exposure) -> tuple[object, ...]:
    return (
        exposure.unit_id,
        exposure.cycle,
        exposure.token_ids,
        exposure.loss_mask,
        exposure.retained_response_tokens,
        exposure.complete,
        exposure.component,
    )


def qualify_nat_spec_plan_pair(
    *, root: int, keep_plan: TrainingPlan, filtered_plan: TrainingPlan
) -> NatSpecRootQualification:
    """Qualify one frozen K/F plan pair without exposing labels to the sampler."""

    if type(root) is not int or root <= 0:
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC root identity is invalid"
        )
    if len(keep_plan.update_bins) != 100 or len(filtered_plan.update_bins) != 100:
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC plans must contain 100 updates"
        )
    if any(not row for row in keep_plan.update_bins + filtered_plan.update_bins):
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC plans contain an empty update"
        )
    if keep_plan.exposures != tuple(item for row in keep_plan.update_bins for item in row):
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC K exposure order differs"
        )
    if filtered_plan.exposures != tuple(
        item for row in filtered_plan.update_bins for item in row
    ):
        raise NatSpecQualificationError(
            "source_or_plan_identity_drift", "NAT-SPEC F exposure order differs"
        )

    update_rows: list[dict[str, object]] = []
    root_instances: list[tuple[str, bool]] = []
    keep_total = 0
    filtered_total = 0
    target_total = 0
    donor_total = 0
    expected_overlap = Fraction(0, 1)
    for update, (keep_row, filtered_row) in enumerate(
        zip(keep_plan.update_bins, filtered_plan.update_bins, strict=True)
    ):
        if any(item.component not in {"B80", "I20"} for item in keep_row):
            raise NatSpecQualificationError(
                "budget_or_component_contract_drift",
                "NAT-SPEC K components differ from B80+I20",
            )
        if any(item.component not in {"B80", "B20"} for item in filtered_row):
            raise NatSpecQualificationError(
                "budget_or_component_contract_drift",
                "NAT-SPEC F components differ from B80+B20",
            )
        keep_background = tuple(item for item in keep_row if item.component == "B80")
        filtered_background = tuple(
            item for item in filtered_row if item.component == "B80"
        )
        if tuple(map(_nat_spec_exposure_identity, keep_background)) != tuple(
            map(_nat_spec_exposure_identity, filtered_background)
        ):
            raise NatSpecQualificationError(
                "budget_or_component_contract_drift",
                "NAT-SPEC K/F B80 instances differ",
            )
        targets = tuple(
            (ordinal, item)
            for ordinal, item in enumerate(keep_row)
            if item.component == "I20"
        )
        donors = tuple(item for item in filtered_row if item.component == "B20")
        if not targets or not donors:
            raise NatSpecQualificationError(
                "budget_or_component_contract_drift",
                "NAT-SPEC target or donor component is empty",
            )

        weights = tuple(_nat_spec_exposure_tokens(item) for item in keep_row)
        donor_weights = tuple(_nat_spec_exposure_tokens(item) for item in donors)
        filtered_weights = tuple(
            _nat_spec_exposure_tokens(item) for item in filtered_row
        )
        k = len(targets)
        q = sum(weights[ordinal] for ordinal, _ in targets)
        anonymous = tuple(
            NatSpecAnonymousExposure(f"r{root}:u{update}:e{ordinal}", weight)
            for ordinal, weight in enumerate(weights)
        )
        try:
            support = build_nat_spec_subset_support(anonymous, k=k, q=q)
            sample_rank = nat_spec_qualification_rank(
                root=root, update=update, omega=support.omega
            )
            ranks = {0, support.omega - 1, sample_rank}
            rank_roundtrips = all(
                support.rank(support.unrank(rank)) == rank for rank in ranks
            )
            sample = support.unrank(sample_rank)
            if not rank_roundtrips:
                raise ValueError("rank/unrank mismatch")
        except (AssertionError, ValueError) as exc:
            raise NatSpecQualificationError(
                "exact_uniform_sampler_invalid",
                f"NAT-SPEC exact sampler failed at root {root} update {update}",
            ) from exc

        selected_ordinals = tuple(
            int(identity.rsplit("e", 1)[1]) for identity in sample
        )
        if (
            len(selected_ordinals) != k
            or sum(weights[value] for value in selected_ordinals) != q
        ):
            raise NatSpecQualificationError(
                "exact_uniform_sampler_invalid",
                "NAT-SPEC qualification sample does not close k/q",
            )
        remaining_ordinals = tuple(
            ordinal
            for ordinal in range(len(keep_row))
            if ordinal not in set(selected_ordinals)
        )
        if remaining_ordinals != tuple(sorted(remaining_ordinals)):
            raise NatSpecQualificationError(
                "exact_uniform_sampler_invalid",
                "NAT-SPEC remaining K order was not preserved",
            )

        target_ids = {f"r{root}:u{update}:e{ordinal}" for ordinal, _ in targets}
        update_expected = sum(
            (
                Fraction(
                    weights[ordinal] * support.inclusion_count(identity),
                    support.omega,
                )
                for ordinal, identity in (
                    (ordinal, f"r{root}:u{update}:e{ordinal}")
                    for ordinal, _ in targets
                )
            ),
            Fraction(0, 1),
        )
        actual_overlap = sum(
            weights[int(identity.rsplit("e", 1)[1])]
            for identity in sample
            if identity in target_ids
        )
        selected = set(selected_ordinals)
        source_counts: dict[str, list[int]] = {}
        for ordinal, item in enumerate(keep_row):
            source = item.unit_id.rsplit("/", 1)[0]
            counts = source_counts.setdefault(source, [0, 0])
            counts[0] += 1
            counts[1] += int(ordinal in selected)
            root_instances.append((source, ordinal in selected))
        fully_removed = sum(removed == total for total, removed in source_counts.values())
        partially_removed = sum(
            0 < removed < total for total, removed in source_counts.values()
        )
        retained = sum(removed == 0 for total, removed in source_counts.values())

        keep_tokens = sum(weights)
        filtered_tokens = sum(filtered_weights)
        donor_tokens = sum(donor_weights)
        keep_total += keep_tokens
        filtered_total += filtered_tokens
        target_total += q
        donor_total += donor_tokens
        expected_overlap += update_expected
        update_rows.append(
            {
                "root": root,
                "update": update,
                "exposures": len(keep_row),
                "k": k,
                "q": q,
                "omega": support.omega,
                "log10_omega": math.log10(support.omega),
                "omega_is_singleton": support.omega == 1,
                "qualification_rank": sample_rank,
                "rank_roundtrips": rank_roundtrips,
                "expected_i_overlap_num": update_expected.numerator,
                "expected_i_overlap_den": update_expected.denominator,
                "sample_actual_i_overlap": actual_overlap,
                "repeated_exposure_count": len(keep_row) - len(source_counts),
                "sample_fully_removed_source_records": fully_removed,
                "sample_partially_removed_source_records": partially_removed,
                "sample_retained_source_records": retained,
            }
        )

    if keep_total != 583_975 or filtered_total != 583_975:
        raise NatSpecQualificationError(
            "budget_or_component_contract_drift",
            "NAT-SPEC K/F plan total differs from 583975",
        )
    if target_total != 116_795 or donor_total != 116_795:
        raise NatSpecQualificationError(
            "budget_or_component_contract_drift",
            "NAT-SPEC target/donor total differs from 116795",
        )

    root_sources: dict[str, list[int]] = {}
    for source, was_selected in root_instances:
        counts = root_sources.setdefault(source, [0, 0])
        counts[0] += 1
        counts[1] += int(was_selected)
    return NatSpecRootQualification(
        root=root,
        update_rows=tuple(update_rows),
        keep_tokens=keep_total,
        filtered_tokens=filtered_total,
        target_tokens=target_total,
        donor_tokens=donor_total,
        alpha=expected_overlap / target_total,
        repeated_exposure_count=len(root_instances) - len(root_sources),
        sample_fully_removed_source_records=sum(
            removed == total for total, removed in root_sources.values()
        ),
        sample_partially_removed_source_records=sum(
            0 < removed < total for total, removed in root_sources.values()
        ),
        sample_retained_source_records=sum(
            removed == 0 for total, removed in root_sources.values()
        ),
    )


def _nat_msep_tokens(exposure: Exposure) -> int:
    try:
        return _nat_spec_exposure_tokens(exposure)
    except NatSpecQualificationError as exc:
        raise NatMsepQualificationError(exc.reason_code, str(exc)) from exc


def _nat_msep_source_parts(exposure: Exposure) -> tuple[str, str]:
    if "/" not in exposure.unit_id:
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift",
            "NAT-MSEP source record lacks its required /I or /N suffix",
        )
    cluster, suffix = exposure.unit_id.rsplit("/", 1)
    expected = "I" if exposure.component == "I20" else "N"
    if (
        not cluster
        or suffix != expected
        or exposure.component not in {"B80", "I20", "B20"}
    ):
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift",
            "NAT-MSEP source suffix and component mapping differ",
        )
    return cluster, suffix


def _removal_classes(
    records: Mapping[str, Sequence[int]], *, removed_index: int
) -> dict[str, int]:
    result = {"complete": 0, "partial": 0, "zero": 0}
    for total, f_removed, m_removed in records.values():
        removed = (f_removed, m_removed)[removed_index]
        if removed == total:
            result["complete"] += 1
        elif removed:
            result["partial"] += 1
        else:
            result["zero"] += 1
    return result


def _count_histogram(values: Sequence[int]) -> dict[str, int]:
    return {
        str(value): count for value, count in sorted(Counter(values).items())
    }


def _cdf_max_abs(left: Sequence[int], right: Sequence[int]) -> Fraction:
    if not left or not right:
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift",
            "NAT-MSEP retained-K input-length CDF has an empty side",
        )
    left_counts = Counter(left)
    right_counts = Counter(right)
    left_seen = 0
    right_seen = 0
    maximum = Fraction(0, 1)
    for length in sorted(set(left_counts) | set(right_counts)):
        left_seen += left_counts[length]
        right_seen += right_counts[length]
        maximum = max(
            maximum,
            abs(Fraction(left_seen, len(left)) - Fraction(right_seen, len(right))),
        )
    return maximum


def report_nat_msep_source_diagnostics(
    *,
    keep_plan: TrainingPlan,
    filtered_plan: TrainingPlan,
    selected_ordinals_by_update: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Report sources only after the anonymous NAT-MSEP masks are frozen."""

    if (
        len(keep_plan.update_bins) != len(filtered_plan.update_bins)
        or len(keep_plan.update_bins) != len(selected_ordinals_by_update)
    ):
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift", "NAT-MSEP reporter update counts differ"
        )
    source_components: dict[str, str] = {}
    source_counts: dict[str, list[int]] = {}
    cluster_counts: Counter[str] = Counter()
    i_clusters: set[str] = set()
    f_retained_b_clusters: set[str] = set()
    m_retained_b_clusters: set[str] = set()
    f_cumulative: list[int] = []
    m_cumulative: list[int] = []
    f_removed_tokens = 0
    m_removed_tokens = 0
    f_retained_k_lengths: list[int] = []
    m_retained_k_lengths: list[int] = []
    f_input_tokens = 0
    m_input_tokens = 0

    for update, (keep_row, filtered_row, selected_values) in enumerate(
        zip(
            keep_plan.update_bins,
            filtered_plan.update_bins,
            selected_ordinals_by_update,
            strict=True,
        )
    ):
        selected_tuple = tuple(selected_values)
        selected = set(selected_tuple)
        if len(selected) != len(selected_tuple) or any(
            type(ordinal) is not int or not 0 <= ordinal < len(keep_row)
            for ordinal in selected
        ):
            raise NatMsepQualificationError(
                "exact_uniform_sampler_invalid",
                "NAT-MSEP reporter received an invalid frozen ordinal mask",
            )
        for ordinal, item in enumerate(keep_row):
            cluster, _ = _nat_msep_source_parts(item)
            previous = source_components.setdefault(item.unit_id, str(item.component))
            if previous != item.component:
                raise NatMsepQualificationError(
                    "source_or_plan_identity_drift",
                    "NAT-MSEP source record maps ambiguously to components",
                )
            counts = source_counts.setdefault(item.unit_id, [0, 0, 0])
            counts[0] += 1
            cluster_counts[cluster] += 1
            f_removed = item.component == "I20"
            m_removed = ordinal in selected
            counts[1] += int(f_removed)
            counts[2] += int(m_removed)
            tokens = _nat_msep_tokens(item)
            f_removed_tokens += tokens if f_removed else 0
            m_removed_tokens += tokens if m_removed else 0
            if item.component == "I20":
                i_clusters.add(cluster)
            if not f_removed:
                f_retained_k_lengths.append(len(item.token_ids))
            if not m_removed:
                m_retained_k_lengths.append(len(item.token_ids))
            if item.component == "B80":
                if not f_removed:
                    f_retained_b_clusters.add(cluster)
                if not m_removed:
                    m_retained_b_clusters.add(cluster)
            if not m_removed:
                m_input_tokens += len(item.token_ids)
        for item in filtered_row:
            _nat_msep_source_parts(item)
            f_input_tokens += len(item.token_ids)
            if item.component == "B20":
                m_input_tokens += len(item.token_ids)
        if (update + 1) % 10 == 0:
            f_cumulative.append(f_removed_tokens)
            m_cumulative.append(m_removed_tokens)

    distance = _cdf_max_abs(f_retained_k_lengths, m_retained_k_lengths)
    ratio = Fraction(f_input_tokens, m_input_tokens)
    return {
        "f_source_removal_classes": _removal_classes(source_counts, removed_index=0),
        "m_source_removal_classes": _removal_classes(source_counts, removed_index=1),
        "source_exposure_count_histogram": _count_histogram(
            [values[0] for values in source_counts.values()]
        ),
        "cluster_exposure_count_histogram": _count_histogram(
            list(cluster_counts.values())
        ),
        "i_associated_clusters": len(i_clusters),
        "f_i_through_b_clusters": len(i_clusters & f_retained_b_clusters),
        "m_i_through_b_clusters": len(i_clusters & m_retained_b_clusters),
        "f_cumulative_removed_response_tokens": f_cumulative,
        "m_cumulative_removed_response_tokens": m_cumulative,
        "f_arm_input_tokens": f_input_tokens,
        "m_arm_input_tokens": m_input_tokens,
        "f_to_m_input_ratio_num": ratio.numerator,
        "f_to_m_input_ratio_den": ratio.denominator,
        "retained_k_input_length_cdf_max_abs_num": distance.numerator,
        "retained_k_input_length_cdf_max_abs_den": distance.denominator,
    }


def qualify_nat_msep_plan_pair(
    *,
    root: int,
    keep_plan: TrainingPlan,
    filtered_plan: TrainingPlan,
    deadline_epoch: float | None = None,
    rank_namespace: str | None = None,
    include_msep_plan: bool = False,
) -> NatMsepRootQualification:
    """Qualify one frozen K/F pair using the exact MSEP active comparator."""

    if type(root) is not int or root <= 0:
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift", "NAT-MSEP root identity is invalid"
        )
    if len(keep_plan.update_bins) != 100 or len(filtered_plan.update_bins) != 100:
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift", "NAT-MSEP plans must contain 100 updates"
        )
    if any(not row for row in keep_plan.update_bins + filtered_plan.update_bins):
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift", "NAT-MSEP plans contain an empty update"
        )
    if keep_plan.exposures != tuple(item for row in keep_plan.update_bins for item in row):
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift", "NAT-MSEP K exposure order differs"
        )
    if filtered_plan.exposures != tuple(
        item for row in filtered_plan.update_bins for item in row
    ):
        raise NatMsepQualificationError(
            "source_or_plan_identity_drift", "NAT-MSEP F exposure order differs"
        )

    update_rows: list[dict[str, object]] = []
    selected_by_update: list[tuple[int, ...]] = []
    msep_updates: list[tuple[Exposure, ...]] = []
    keep_total = 0
    filtered_total = 0
    target_total = 0
    donor_total = 0
    optimum_id_total = 0
    optimum_payload_total = 0
    f_arm_total = 0
    m_arm_total = 0

    for update, (keep_row, filtered_row) in enumerate(
        zip(keep_plan.update_bins, filtered_plan.update_bins, strict=True)
    ):
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise NatMsepDeadlineExceeded(
                f"NAT-MSEP deadline reached before root {root} update {update}"
            )
        if any(item.component not in {"B80", "I20"} for item in keep_row):
            raise NatMsepQualificationError(
                "budget_or_component_contract_drift",
                "NAT-MSEP K components differ from B80+I20",
            )
        if any(item.component not in {"B80", "B20"} for item in filtered_row):
            raise NatMsepQualificationError(
                "budget_or_component_contract_drift",
                "NAT-MSEP F components differ from B80+B20",
            )
        keep_background = tuple(item for item in keep_row if item.component == "B80")
        filtered_background = tuple(
            item for item in filtered_row if item.component == "B80"
        )
        if tuple(map(_nat_spec_exposure_identity, keep_background)) != tuple(
            map(_nat_spec_exposure_identity, filtered_background)
        ):
            raise NatMsepQualificationError(
                "budget_or_component_contract_drift",
                "NAT-MSEP K/F B80 instances differ",
            )
        targets = tuple(
            (ordinal, item)
            for ordinal, item in enumerate(keep_row)
            if item.component == "I20"
        )
        donors = tuple(item for item in filtered_row if item.component == "B20")
        if not targets or not donors:
            raise NatMsepQualificationError(
                "budget_or_component_contract_drift",
                "NAT-MSEP target or donor component is empty",
            )

        weights = tuple(_nat_msep_tokens(item) for item in keep_row)
        filtered_weights = tuple(_nat_msep_tokens(item) for item in filtered_row)
        donor_weights = tuple(_nat_msep_tokens(item) for item in donors)
        target_ordinals = {ordinal for ordinal, _ in targets}
        k = len(targets)
        q = sum(weights[ordinal] for ordinal in target_ordinals)
        anonymous = tuple(
            NatMsepAnonymousExposure(
                anon_id=f"r{root}:u{update}:e{ordinal}",
                m_e=weight,
                is_target=ordinal in target_ordinals,
                payload_type=(item.token_ids, item.loss_mask[1:]),
            )
            for ordinal, (item, weight) in enumerate(zip(keep_row, weights, strict=True))
        )
        try:
            support = build_nat_msep_subset_support(anonymous, k=k, q=q)
            sample_rank = (
                nat_msep_qualification_rank(
                    root=root, update=update, omega=support.omega2
                )
                if rank_namespace is None
                else nat_msep_formal_rank(
                    namespace=rank_namespace,
                    root=root,
                    update=update,
                    omega=support.omega2,
                )
            )
            sample = support.unrank(sample_rank)
            rank_roundtrips = all(
                support.rank(support.unrank(rank)) == rank
                for rank in {0, support.omega2 - 1, sample_rank}
            )
        except (AssertionError, TypeError, ValueError) as exc:
            raise NatMsepQualificationError(
                "exact_uniform_sampler_invalid",
                f"NAT-MSEP exact selector failed at root {root} update {update}",
            ) from exc
        selected_ordinals = tuple(
            int(identity.rsplit("e", 1)[1]) for identity in sample
        )
        selected = set(selected_ordinals)
        remaining_ordinals = tuple(
            ordinal for ordinal in range(len(keep_row)) if ordinal not in selected
        )
        m_row = tuple(keep_row[ordinal] for ordinal in remaining_ordinals) + donors
        materialization_gate = (
            rank_roundtrips
            and len(selected_ordinals) == k
            and sum(weights[ordinal] for ordinal in selected_ordinals) == q
            and remaining_ordinals == tuple(sorted(remaining_ordinals))
            and tuple(map(_nat_spec_exposure_identity, m_row[-len(donors) :]))
            == tuple(map(_nat_spec_exposure_identity, donors))
            and sum(_nat_msep_tokens(item) for item in m_row)
            == sum(filtered_weights)
        )
        if not materialization_gate:
            raise NatMsepQualificationError(
                "materialization_contract_drift",
                f"NAT-MSEP materialization failed at root {root} update {update}",
            )

        keep_tokens = sum(weights)
        filtered_tokens = sum(filtered_weights)
        donor_tokens = sum(donor_weights)
        keep_total += keep_tokens
        filtered_total += filtered_tokens
        target_total += q
        donor_total += donor_tokens
        optimum_id_total += support.optimum_id_overlap
        optimum_payload_total += support.optimum_payload_overlap
        f_arm_total += filtered_tokens
        m_arm_total += sum(_nat_msep_tokens(item) for item in m_row)
        selected_by_update.append(selected_ordinals)
        msep_updates.append(m_row)
        update_rows.append(
            {
                "root": root,
                "update": update,
                "exposures": len(keep_row),
                "k": k,
                "q": q,
                "omega": support.omega,
                "omega1": support.omega1,
                "omega2": support.omega2,
                "log10_omega": math.log10(support.omega),
                "omega_is_singleton": support.omega == 1,
                "qualification_rank": sample_rank,
                "rank_roundtrips": rank_roundtrips,
                "optimum_id_overlap": support.optimum_id_overlap,
                "optimum_payload_overlap": support.optimum_payload_overlap,
                "selected_ordinals": list(selected_ordinals),
                "selected_count": len(selected_ordinals),
                "selected_response_tokens": sum(
                    weights[ordinal] for ordinal in selected_ordinals
                ),
                "materialization_gate": materialization_gate,
            }
        )

    if keep_total != 583_975 or filtered_total != 583_975:
        raise NatMsepQualificationError(
            "budget_or_component_contract_drift",
            "NAT-MSEP K/F plan total differs from 583975",
        )
    if target_total != 116_795 or donor_total != 116_795:
        raise NatMsepQualificationError(
            "budget_or_component_contract_drift",
            "NAT-MSEP target/donor total differs from 116795",
        )
    source_diagnostics = report_nat_msep_source_diagnostics(
        keep_plan=keep_plan,
        filtered_plan=filtered_plan,
        selected_ordinals_by_update=selected_by_update,
    )
    d_id = 116_795 - optimum_id_total
    d_payload = 116_795 - optimum_payload_total
    delta_id = Fraction(d_id, 583_975)
    delta_payload = Fraction(d_payload, 583_975)
    root_row: dict[str, object] = {
        "root": root,
        "updates": len(update_rows),
        "D_id": d_id,
        "delta_id_num": delta_id.numerator,
        "delta_id_den": delta_id.denominator,
        "id_gate": 10 * d_id >= 583_975,
        "D_payload": d_payload,
        "delta_payload_num": delta_payload.numerator,
        "delta_payload_den": delta_payload.denominator,
        "payload_gate": 10 * d_payload >= 583_975,
        "singleton_updates": sum(
            bool(row["omega_is_singleton"]) for row in update_rows
        ),
        "omega_sum": sum(int(row["omega"]) for row in update_rows),
        "omega1_sum": sum(int(row["omega1"]) for row in update_rows),
        "omega2_sum": sum(int(row["omega2"]) for row in update_rows),
        "f_arm_response_tokens": f_arm_total,
        "m_arm_response_tokens": m_arm_total,
    }
    root_row.update(source_diagnostics)
    msep_plan = None
    if include_msep_plan:
        bins = tuple(msep_updates)
        exposures = tuple(item for row in bins for item in row)
        msep_plan = TrainingPlan(
            exposures=exposures,
            update_bins=bins,
            packs=single_unit_packs(bins, max_length=2_048),
            exposure_ledger=_ledger(exposures),
        )
    return NatMsepRootQualification(
        root=root,
        update_rows=tuple(update_rows),
        root_row=root_row,
        selected_ordinals_by_update=tuple(selected_by_update),
        source_diagnostics=source_diagnostics,
        msep_plan=msep_plan,
    )


def materialize_nat_msep_plan_pair(
    *,
    root: int,
    keep_plan: TrainingPlan,
    filtered_plan: TrainingPlan,
    rank_namespace: str,
    deadline_epoch: float | None = None,
) -> NatMsepRootQualification:
    """Materialize the exact formal POL_MSEP plan for one frozen K/F pair."""

    return qualify_nat_msep_plan_pair(
        root=root,
        keep_plan=keep_plan,
        filtered_plan=filtered_plan,
        deadline_epoch=deadline_epoch,
        rank_namespace=rank_namespace,
        include_msep_plan=True,
    )


def _nat_onoff_metadata_by_unit(
    rows: Sequence[Mapping[str, object]], *, label: str
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    required = {"unit_id", "record_id", "pair_id", "problem_id", "source_lf", "payload"}
    for row in rows:
        unit_id = row.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id or unit_id in result:
            raise ValueError(f"NAT-ONOFF {label} unit identity is invalid")
        if not required <= set(row) or any(row.get(key) is None for key in required):
            raise ValueError(f"NAT-ONOFF {label} metadata is incomplete")
        result[unit_id] = row
    return result


def build_nat_onoff_plans(
    *,
    root: int,
    background_plan: TrainingPlan,
    supplement_units: Sequence[TokenizedUnit],
    background_metadata: Sequence[Mapping[str, object]],
    supplement_metadata: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the frozen natural-supplement ON/OFF ledgers for one root.

    Pair IDs are deliberately absent from every exclusion predicate.  Deletion
    is common to both arms and finite: an excluded B exposure is never replaced.
    Complete B blocks are then permuted as indivisible units, while each S unit
    is appended once to the least-loaded original block.
    """

    if type(root) is not int or root <= 0:
        raise ValueError("NAT-ONOFF root must be a positive integer")
    blocks = tuple(tuple(row) for row in background_plan.update_bins)
    if not blocks or any(not row for row in blocks):
        raise ValueError("NAT-ONOFF background blocks must be nonempty")
    if tuple(item for row in blocks for item in row) != background_plan.exposures:
        raise ValueError("NAT-ONOFF background chronology differs from its blocks")
    b_meta_rows = tuple(background_metadata)
    s_meta = _nat_onoff_metadata_by_unit(supplement_metadata, label="supplement")
    supplement = tuple(supplement_units)
    if not supplement or len({unit.unit_id for unit in supplement}) != len(supplement):
        raise ValueError("NAT-ONOFF supplement units must be nonempty and unique")
    if set(s_meta) != {unit.unit_id for unit in supplement}:
        raise ValueError("NAT-ONOFF supplement metadata differs from its units")
    if len(b_meta_rows) != len(background_plan.exposures) or any(
        row.get("unit_id") != exposure.unit_id
        or not {"unit_id", "record_id", "pair_id", "problem_id", "source_lf", "payload"} <= set(row)
        or any(row.get(key) is None for key in ("record_id", "pair_id", "problem_id", "source_lf", "payload"))
        for row, exposure in zip(b_meta_rows, background_plan.exposures, strict=True)
    ):
        raise ValueError("NAT-ONOFF background metadata differs from its exposure support")
    if any(len(unit.token_ids) > 2_048 for unit in supplement):
        raise ValueError("NAT-ONOFF supplement serialization exceeds 2048")

    supplement_records = {row["record_id"] for row in s_meta.values()}
    supplement_payloads = {row["payload"] for row in s_meta.values()}
    supplement_problem_sources = {
        (row["problem_id"], row["source_lf"]) for row in s_meta.values()
    }
    exclusion_counts = {"record_id": 0, "payload": 0, "same_problem_source": 0}
    excluded_ordinals: set[int] = set()
    excluded_unit_ids: list[str] = []
    for ordinal, row in enumerate(b_meta_rows):
        reason = None
        if row["record_id"] in supplement_records:
            reason = "record_id"
        elif row["payload"] in supplement_payloads:
            reason = "payload"
        elif (row["problem_id"], row["source_lf"]) in supplement_problem_sources:
            reason = "same_problem_source"
        if reason is not None:
            excluded_ordinals.add(ordinal)
            excluded_unit_ids.append(str(row["unit_id"]))
            exclusion_counts[reason] += 1

    retained_blocks: list[tuple[Exposure, ...]] = []
    ordinal = 0
    for row in blocks:
        retained_values = []
        for item in row:
            if ordinal not in excluded_ordinals:
                retained_values.append(replace(item, component="B"))
            ordinal += 1
        retained = tuple(retained_values)
        if not retained:
            raise ValueError("NAT-ONOFF common deletion empties a background block")
        retained_blocks.append(retained)

    assigned: list[list[Exposure]] = [[] for _ in retained_blocks]
    loads = [0] * len(retained_blocks)
    ordered_s = sorted(
        supplement,
        key=lambda unit: (-sum(unit.loss_mask[1:]), str(s_meta[unit.unit_id]["record_id"])),
    )
    for unit in ordered_s:
        dose = sum(unit.loss_mask[1:])
        if dose <= 0:
            raise ValueError("NAT-ONOFF supplement has no shifted response labels")
        block = min(range(len(loads)), key=lambda index: (loads[index], index))
        assigned[block].append(
            Exposure(
                unit_id=unit.unit_id,
                cycle=0,
                token_ids=unit.token_ids,
                loss_mask=unit.loss_mask,
                retained_response_tokens=dose,
                complete=True,
                component="S",
            )
        )
        loads[block] += dose

    order = tuple(
        sorted(
            range(len(retained_blocks)),
            key=lambda index: (
                hashlib.sha256(
                    f"{NAT_ONOFF_PROTOCOL_ID}|block-order|{root}|{index}".encode("ascii")
                ).digest(),
                index,
            ),
        )
    )
    off_bins = tuple(retained_blocks[index] for index in order)
    on_bins = tuple(
        retained_blocks[index] + tuple(assigned[index]) for index in order
    )

    def make_plan(bins: tuple[tuple[Exposure, ...], ...]) -> TrainingPlan:
        exposures = tuple(item for row in bins for item in row)
        return TrainingPlan(
            exposures=exposures,
            update_bins=bins,
            packs=single_unit_packs(bins, max_length=2_048),
            exposure_ledger=_ledger(exposures),
        )

    return {
        "plans": {"ON": make_plan(on_bins), "OFF": make_plan(off_bins)},
        "block_order": order,
        "supplement_block_by_unit": {
            item.unit_id: index
            for index, values in enumerate(assigned)
            for item in values
        },
        "excluded_background_unit_ids": tuple(sorted(set(excluded_unit_ids))),
        "excluded_background_exposure_ordinals": tuple(sorted(excluded_ordinals)),
        "exclusion_counts": exclusion_counts,
        "supplement_shifted_response_tokens": sum(loads),
    }


def build_nat_e95_delete_plan(
    *,
    old_on_plan: TrainingPlan,
    supplement_category_by_unit: Mapping[str, str],
) -> dict[str, object]:
    """Delete frozen E exposures from an existing ON plan without replacement.

    Remaining B and wrong-answer exposures are the original immutable objects;
    blocks are neither reordered nor repacked across exposure boundaries.  The
    retained wrong-answer exposures keep component ``S`` so they consume the
    historical root-derived S dropout stream.
    """

    blocks = tuple(tuple(block) for block in old_on_plan.update_bins)
    if not blocks or any(not block for block in blocks):
        raise ValueError("Q1-NAT-E95-DELETE old ON blocks must be nonempty")
    if tuple(item for block in blocks for item in block) != old_on_plan.exposures:
        raise ValueError("Q1-NAT-E95-DELETE old ON chronology differs from its blocks")
    categories = dict(supplement_category_by_unit)
    allowed_e = {"compile_failure", "runtime_failure", "timeout"}
    seen_s: set[str] = set()
    retained_blocks: list[tuple[Exposure, ...]] = []
    block_rows: list[dict[str, int]] = []
    exposure_counts = {"B": 0, "W": 0, "E_deleted": 0}
    token_counts = {"B": 0, "W": 0, "E_deleted": 0}
    for update, block in enumerate(blocks):
        retained: list[Exposure] = []
        row = {"update": update, "b_tokens": 0, "w_tokens": 0, "e_tokens": 0}
        for exposure in block:
            if exposure.component == "B":
                if exposure.unit_id in categories:
                    raise ValueError("Q1-NAT-E95-DELETE B exposure has a supplement category")
                retained.append(exposure)
                exposure_counts["B"] += 1
                token_counts["B"] += exposure.retained_response_tokens
                row["b_tokens"] += exposure.retained_response_tokens
                continue
            if exposure.component != "S" or exposure.unit_id not in categories:
                raise ValueError("Q1-NAT-E95-DELETE supplement identity is unclassified")
            if exposure.unit_id in seen_s:
                raise ValueError("Q1-NAT-E95-DELETE supplement exposure is duplicated")
            seen_s.add(exposure.unit_id)
            category = categories[exposure.unit_id]
            if category == "wrong_answer":
                retained.append(exposure)
                exposure_counts["W"] += 1
                token_counts["W"] += exposure.retained_response_tokens
                row["w_tokens"] += exposure.retained_response_tokens
            elif category in allowed_e:
                exposure_counts["E_deleted"] += 1
                token_counts["E_deleted"] += exposure.retained_response_tokens
                row["e_tokens"] += exposure.retained_response_tokens
            else:
                raise ValueError("Q1-NAT-E95-DELETE supplement category is outside W/E")
        if not retained:
            raise ValueError("Q1-NAT-E95-DELETE deletion empties an update block")
        row["mid_denominator"] = row["b_tokens"] + row["w_tokens"]
        if row["mid_denominator"] <= 0:
            raise ValueError("Q1-NAT-E95-DELETE MID denominator is nonpositive")
        retained_blocks.append(tuple(retained))
        block_rows.append(row)
    if seen_s != set(categories):
        raise ValueError("Q1-NAT-E95-DELETE category ledger differs from old ON plan")
    bins = tuple(retained_blocks)
    exposures = tuple(item for block in bins for item in block)
    plan = TrainingPlan(
        exposures=exposures,
        update_bins=bins,
        packs=single_unit_packs(bins, max_length=2_048),
        exposure_ledger=_ledger(exposures),
    )
    return {
        "plan": plan,
        "component_exposures": exposure_counts,
        "component_tokens": token_counts,
        "block_rows": tuple(block_rows),
    }


def build_w259_prospective_plans(
    *,
    root: int,
    old_on_plan: TrainingPlan,
    old_block_order: Sequence[int],
    supplement_category_by_unit: Mapping[str, str],
    expected_supplement_block_by_unit: Mapping[str, int],
    expected_block_order_sha256: str,
) -> dict[str, object]:
    """Build paired B+W/B plans by reordering the original complete blocks.

    ``old_on_plan`` is stored in an earlier root-specific chronology.  Its
    frozen ``old_block_order`` is inverted first so supplement membership can
    never be silently recomputed by the old load-balancing builder.  E is then
    deleted from those canonical blocks and the new root only permutes complete
    B/W blocks.
    """

    if type(root) is not int or root <= 0:
        raise ValueError("Q1-W259-PROSPECTIVE root must be a positive integer")
    old_bins = tuple(tuple(block) for block in old_on_plan.update_bins)
    order = tuple(old_block_order)
    if (
        not old_bins
        or any(not block for block in old_bins)
        or len(order) != len(old_bins)
        or set(order) != set(range(len(old_bins)))
    ):
        raise ValueError("Q1-W259-PROSPECTIVE historical block order is not a permutation")
    if tuple(item for block in old_bins for item in block) != old_on_plan.exposures:
        raise ValueError("Q1-W259-PROSPECTIVE historical chronology differs from its blocks")

    canonical: list[tuple[Exposure, ...] | None] = [None] * len(old_bins)
    for stored_position, canonical_block in enumerate(order):
        canonical[canonical_block] = old_bins[stored_position]
    canonical_bins = tuple(block for block in canonical if block is not None)
    observed_supplement_blocks = {
        item.unit_id: canonical_block
        for canonical_block, block in enumerate(canonical_bins)
        for item in block
        if item.component == "S"
    }
    if observed_supplement_blocks != {
        str(unit_id): int(block) for unit_id, block in expected_supplement_block_by_unit.items()
    }:
        raise ValueError(
            "Q1-W259-PROSPECTIVE supplement-to-canonical-block binding differs; "
            "a re-binned W construction is invalid"
        )

    canonical_exposures = tuple(item for block in canonical_bins for item in block)
    canonical_plan = TrainingPlan(
        exposures=canonical_exposures,
        update_bins=canonical_bins,
        packs=single_unit_packs(canonical_bins, max_length=2_048),
        exposure_ledger=_ledger(canonical_exposures),
    )
    deleted = build_nat_e95_delete_plan(
        old_on_plan=canonical_plan,
        supplement_category_by_unit=supplement_category_by_unit,
    )
    use_canonical = tuple(tuple(block) for block in deleted["plan"].update_bins)
    if any(
        tuple(item.component for item in block)
        != tuple(sorted((item.component for item in block), key=lambda value: value != "B"))
        for block in use_canonical
    ):
        raise ValueError("Q1-W259-PROSPECTIVE canonical block is not B-before-W")
    omit_canonical = tuple(
        tuple(item for item in block if item.component == "B") for block in use_canonical
    )
    if any(not block for block in omit_canonical):
        raise ValueError("Q1-W259-PROSPECTIVE canonical block has no B exposure")

    new_order = tuple(
        sorted(
            range(len(use_canonical)),
            key=lambda canonical_block: (
                hashlib.sha256(
                    f"{NAT_ONOFF_PROTOCOL_ID}|block-order|{root}|{canonical_block}".encode(
                        "ascii"
                    )
                ).digest(),
                canonical_block,
            ),
        )
    )
    order_sha256 = hashlib.sha256(
        ",".join(map(str, new_order)).encode("ascii")
    ).hexdigest()
    if order_sha256 != expected_block_order_sha256:
        raise ValueError("Q1-W259-PROSPECTIVE new-root block-order digest differs")

    def make_plan(canonical_blocks: tuple[tuple[Exposure, ...], ...]) -> TrainingPlan:
        bins = tuple(canonical_blocks[index] for index in new_order)
        exposures = tuple(item for block in bins for item in block)
        return TrainingPlan(
            exposures=exposures,
            update_bins=bins,
            packs=single_unit_packs(bins, max_length=2_048),
            exposure_ledger=_ledger(exposures),
        )

    return {
        "plans": {
            "W_USE": make_plan(use_canonical),
            "W_OMIT": make_plan(omit_canonical),
        },
        "block_order": new_order,
        "block_order_sha256": order_sha256,
        "canonical_supplement_block_by_unit": observed_supplement_blocks,
        "component_exposures": deleted["component_exposures"],
        "component_tokens": deleted["component_tokens"],
        "block_rows": tuple(
            {
                "update": update,
                "canonical_block": canonical_block,
                **dict(deleted["block_rows"][canonical_block]),
            }
            for update, canonical_block in enumerate(new_order)
        ),
    }


def build_w259_factorial_plans(
    *,
    root: int,
    source_w_plan: TrainingPlan,
    source_block_order: Sequence[int],
    group_by_unit: Mapping[str, str],
    expected_block_order_sha256: str,
    arms: Sequence[str] = ("B", "BG", "BH", "BW"),
) -> dict[str, object]:
    """Build fixed B/BG/BH/BW plans from one bound B+W block schedule."""

    if tuple(arms) not in (("B", "BG", "BH", "BW"), ("B", "BG")):
        raise ValueError("fixed W259 plans require the frozen CF12 or G121 arms")
    if type(root) is not int or root <= 0:
        raise ValueError("Q1-W259-CF12 root must be a positive integer")
    source_bins = tuple(tuple(block) for block in source_w_plan.update_bins)
    source_order = tuple(int(value) for value in source_block_order)
    if (
        not source_bins
        or any(not block for block in source_bins)
        or len(source_order) != len(source_bins)
        or set(source_order) != set(range(len(source_bins)))
    ):
        raise ValueError("Q1-W259-CF12 source block order is not a permutation")
    if tuple(item for block in source_bins for item in block) != source_w_plan.exposures:
        raise ValueError("Q1-W259-CF12 source chronology differs from its blocks")

    canonical: list[tuple[Exposure, ...] | None] = [None] * len(source_bins)
    for source_position, canonical_block in enumerate(source_order):
        canonical[canonical_block] = source_bins[source_position]
    canonical_bins = tuple(block for block in canonical if block is not None)
    if any(
        not any(item.component == "B" for item in block)
        or any(item.component not in {"B", "S"} for item in block)
        or tuple(item.component for item in block)
        != tuple(sorted((item.component for item in block), key=lambda value: value != "B"))
        for block in canonical_bins
    ):
        raise ValueError("Q1-W259-CF12 canonical blocks must be B-positive and B-before-W")

    observed_w = {
        item.unit_id
        for block in canonical_bins
        for item in block
        if item.component == "S"
    }
    groups = {str(unit_id): str(group) for unit_id, group in group_by_unit.items()}
    if set(groups) != observed_w or set(groups.values()) != {"G", "H"}:
        raise ValueError("Q1-W259-CF12 locked G/H membership is not the W complement")

    canonical_by_arm = {
        "B": tuple(
            tuple(item for item in block if item.component == "B")
            for block in canonical_bins
        ),
        "BG": tuple(
            tuple(
                item
                for item in block
                if item.component == "B" or groups.get(item.unit_id) == "G"
            )
            for block in canonical_bins
        ),
        "BH": tuple(
            tuple(
                item
                for item in block
                if item.component == "B" or groups.get(item.unit_id) == "H"
            )
            for block in canonical_bins
        ),
        "BW": canonical_bins,
    }
    new_order = tuple(
        sorted(
            range(len(canonical_bins)),
            key=lambda canonical_block: (
                hashlib.sha256(
                    f"{NAT_ONOFF_PROTOCOL_ID}|block-order|{root}|{canonical_block}".encode(
                        "ascii"
                    )
                ).digest(),
                canonical_block,
            ),
        )
    )
    order_sha256 = hashlib.sha256(
        ",".join(map(str, new_order)).encode("ascii")
    ).hexdigest()
    if order_sha256 != expected_block_order_sha256:
        raise ValueError("Q1-W259-CF12 root block-order digest differs")

    def make_plan(canonical_blocks: tuple[tuple[Exposure, ...], ...]) -> TrainingPlan:
        bins = tuple(canonical_blocks[index] for index in new_order)
        exposures = tuple(item for block in bins for item in block)
        return TrainingPlan(
            exposures=exposures,
            update_bins=bins,
            packs=single_unit_packs(bins, max_length=2_048),
            exposure_ledger=_ledger(exposures),
        )

    plans = {
        arm: make_plan(canonical_blocks)
        for arm, canonical_blocks in canonical_by_arm.items()
        if arm in arms
    }
    component_exposures = {
        "B": sum(item.component == "B" for item in source_w_plan.exposures),
        "G": sum(groups.get(item.unit_id) == "G" for item in source_w_plan.exposures),
        "H": sum(groups.get(item.unit_id) == "H" for item in source_w_plan.exposures),
    }
    component_tokens = {
        "B": sum(
            item.retained_response_tokens
            for item in source_w_plan.exposures
            if item.component == "B"
        ),
        "G": sum(
            item.retained_response_tokens
            for item in source_w_plan.exposures
            if groups.get(item.unit_id) == "G"
        ),
        "H": sum(
            item.retained_response_tokens
            for item in source_w_plan.exposures
            if groups.get(item.unit_id) == "H"
        ),
    }
    return {
        "plans": plans,
        "block_order": new_order,
        "block_order_sha256": order_sha256,
        "component_exposures": component_exposures,
        "component_tokens": component_tokens,
        "block_rows": tuple(
            {
                "update": update,
                "canonical_block": canonical_block,
                **{
                    f"{arm}_tokens": sum(
                        item.retained_response_tokens
                        for item in canonical_by_arm[arm][canonical_block]
                    )
                    for arm in arms
                },
            }
            for update, canonical_block in enumerate(new_order)
        ),
    }


def build_nat_onoff_roots(
    *, seed_hex: str, prior_roots: Sequence[int]
) -> tuple[int, ...]:
    """Draw eight replayable roots once, replacing identity collisions only."""

    if (
        not isinstance(seed_hex, str)
        or len(seed_hex) != 64
        or any(character not in "0123456789abcdef" for character in seed_hex)
    ):
        raise ValueError("NAT-ONOFF root seed must be lowercase 256-bit hex")
    used = set(prior_roots)
    if any(type(root) is not int or root <= 0 for root in used):
        raise ValueError("NAT-ONOFF prior roots are malformed")
    result: list[int] = []
    for ordinal in range(1, 9):
        collision = 0
        while True:
            payload = f"{NAT_ONOFF_ROOT_NAMESPACE}|{seed_hex}|{ordinal}|{collision}"
            root = int(hashlib.sha256(payload.encode("ascii")).hexdigest()[:8], 16)
            root %= 2_147_483_647
            if root and root not in used:
                break
            collision += 1
        result.append(root)
        used.add(root)
    return tuple(result)


def build_nat_replace_roots(*, historical_roots: Sequence[int]) -> tuple[int, ...]:
    """ADR0122 roots, with pre-model mechanical identity collision handling only."""
    used = set(historical_roots)
    roots = []
    for index in range(1, 9):
        stem = f"NAT-REPLACE-20-v1/root/{index}"
        collision = 0
        while True:
            text = stem if collision == 0 else f"{stem}/collision/{collision}"
            root = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 2147483647
            if root and root not in used:
                break
            collision += 1
        roots.append(root)
        used.add(root)
    return tuple(roots)


def build_nat_replica_roots(
    *, seed_hex: str, historical_roots: Sequence[int]
) -> tuple[int, ...]:
    """Derive eight replayable roots from one outcome-blind 256-bit draw.

    The stored draw is an auditable source of randomness, not a model outcome or
    a root search.  Collision replacement consults identities only.
    """

    if (
        not isinstance(seed_hex, str)
        or len(seed_hex) != 64
        or any(character not in "0123456789abcdef" for character in seed_hex)
    ):
        raise ValueError("NAT-REPLICA root seed must be lowercase 256-bit hex")
    used = set(historical_roots)
    if any(type(root) is not int or root <= 0 for root in used):
        raise ValueError("NAT-REPLICA historical roots are malformed")
    roots: list[int] = []
    for index in range(1, 9):
        collision = 0
        while True:
            payload = (
                f"{NAT_REPLICA_PROTOCOL_ID}/root-v1|{seed_hex}|{index}|{collision}"
            )
            root = int(hashlib.sha256(payload.encode("ascii")).hexdigest()[:8], 16)
            root %= 2_147_483_647
            if root and root not in used:
                break
            collision += 1
        roots.append(root)
        used.add(root)
    return tuple(roots)


def scope_pc_identity_order(purpose: str, identity: str) -> tuple[str, str]:
    payload = f"{SCOPE_PC_PROTOCOL_ID}/split-v1|{purpose}|{identity}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), identity


def group_scope_pc_pairs(
    rows: Sequence[Mapping[str, object]], *, duplicate_links: Sequence[tuple[str, str]] = ()
) -> list[dict[str, object]]:
    """Group the entire candidate universe before selecting eligible representatives."""
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("SCOPE pair identities must be unique")
    parents = {str(row["problem_id"]): str(row["problem_id"]) for row in rows}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(left, right):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)

    statements = {}
    for row in rows:
        description = row["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError("SCOPE statement is malformed")
        identity = str(row["problem_id"])
        normalized = " ".join(unicodedata.normalize("NFC", description).split())
        if normalized in statements:
            join(identity, statements[normalized])
        statements[normalized] = identity
    for left, right in duplicate_links:
        if left not in parents or right not in parents:
            raise ValueError("SCOPE duplicate link outside candidate universe")
        join(left, right)
    members = {}
    for row in rows:
        members.setdefault(find(str(row["problem_id"])), []).append(row)
    result = []
    for group_id, group in sorted(members.items()):
        eligible = [str(row["pair_id"]) for row in group if row["eligible"] is True]
        representative = min(eligible, key=lambda key: scope_pc_identity_order("representative", key)) if eligible else None
        result.append({"group_id": group_id,
                       "member_pair_ids": sorted(str(row["pair_id"]) for row in group),
                       "member_problem_ids": sorted({str(row["problem_id"]) for row in group}),
                       "representative_pair_id": representative})
    return result
def split_scope_pc_groups(rows: Sequence[Mapping[str, object]], *, duplicate_links=()) -> dict[str, object]:
    """Fixed group-level stratified T/D/H allocation, without any model information."""
    groups = group_scope_pc_pairs(rows, duplicate_links=duplicate_links)
    eligible = [g for g in groups if g["representative_pair_id"] is not None]
    count = len(eligible)
    if count < 310:
        return {"status": "support_insufficient", "groups": groups, "G": count,
                "T": [], "D": [], "H": [], "T64": []}
    if count > 350:
        raise ValueError("SCOPE group count exceeds frozen universe")

    def allocate(total, weights, tie_order):
        mass = sum(weights.values())
        quotas = {key: total * weight // mass for key, weight in weights.items()}
        order = sorted(weights, key=lambda key: (-(total * weights[key] % mass), tie_order.index(key)))
        for key in order[:total - sum(quotas.values())]:
            quotas[key] += 1
        return quotas

    extra = allocate(count - 310, {"T": 2, "D": 1, "H": 1}, ["T", "D", "H"])
    targets = {key: base + extra[key] for key, base in (("T", 220), ("D", 30), ("H", 60))}
    by_pair = {row["pair_id"]: row for row in rows}
    strata = {}
    for group in eligible:
        row = by_pair[group["representative_pair_id"]]
        if row["provenance"] not in {"original", "alternative"} or row["failure_family"] not in {"execution_failure", "output_mismatch", "timeout"}:
            raise ValueError("SCOPE stratum is invalid")
        strata.setdefault((row["provenance"], row["failure_family"]), []).append(group["group_id"])
    for values in strata.values():
        values.sort(key=lambda key: scope_pc_identity_order("split", key))
    roles = {"T": [], "D": [], "H": [], "T64": []}
    for role in ("D", "H"):
        quotas = allocate(targets[role], {k: len(v) for k, v in strata.items()}, sorted(strata))
        for key in sorted(strata):
            roles[role].extend(strata[key][:quotas[key]])
            strata[key] = strata[key][quotas[key]:]
    probes = allocate(64, {k: len(v) for k, v in strata.items()}, sorted(strata))
    for key in sorted(strata):
        roles["T"].extend(strata[key])
        roles["T64"].extend(sorted(strata[key], key=lambda identity: scope_pc_identity_order("probe", identity))[:probes[key]])
    return {"status": "support_sufficient", "groups": groups, "G": count, **roles}


_SCIENTIFIC_TERMINALS = (
    "COMPLETED_S12_PC_UNAVAILABLE",
    "COMPLETED_S12_ASSAY_NOT_SENSITIVE",
    "COMPLETED_S12_NATURAL_NOT_SUPPORTED",
    "COMPLETED_S12_POLICY_UNAVAILABLE",
    "COMPLETED_S12_REPORT",
)
_INVALID_TERMINALS = (
    "STOPPED_S12_INPUT_INVALID",
    "STOPPED_S12_INFRASTRUCTURE_INVALID",
    "STOPPED_S12_RESOURCE_LIMIT",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class S12Config:
    protocol_id: str
    design_id: str
    formal_namespace: str
    stage_order: tuple[str, ...]
    stage_arms: Mapping[str, tuple[str, ...]]
    planning_primary_arms: Mapping[str, tuple[str, str]]
    response_token_quota: int
    updates: int
    max_length: int
    policy_background_quota: int
    policy_component_quota: int
    planning_root_count: int
    confirmatory_candidates: int
    confirmatory_small_n: int
    planning_sd_threshold: float
    precision_half_width_target: float
    practical_margin: float
    sampling_algorithm_id: str
    failure_family_map: Mapping[str, str]
    concurrency_ceilings: Mapping[str, int]
    child_wall_hours: Mapping[str, int]
    formal_elapsed_hours: int
    capacity_wait_budget_hours: int
    legal_terminals: tuple[str, ...]


def load_s12_config(path: Path) -> S12Config:
    """Load the single frozen S12 config, rejecting scientific drift."""

    payload = json.loads(path.read_text("utf-8"))
    schema = payload.get("schema")
    if schema not in {
        "codecontests-s12-apcf-config-v1",
        "codecontests-s12c-rkqp-config-v1",
        "codecontests-s12d-uqcc-config-v1",
    }:
        raise ValueError("S12 config schema differs from protocol")
    if schema == "codecontests-s12c-rkqp-config-v1" and not (
        payload.get("qualification_protocol_id") == "codecontests-s12c-rkqp-v1"
        and payload.get("qualification_design_id") == "CC_S12C_CSR_KCQS_OQEP"
        and payload.get("qualification_elapsed_hours") == 6
        and payload.get("qualification_workers") == {"canonical": 4, "serial": 1}
        and payload.get("maximum_qualification_candidates") == 3
        and payload.get("endpoint_reference_count") == 285
    ):
        raise ValueError("S12c qualification config differs from protocol")
    if schema == "codecontests-s12d-uqcc-config-v1":
        sentinels = payload.get("sentinels")
        faults = payload.get("fault_cases")
        if not (
            payload.get("qualification_protocol_id") == "codecontests-s12d-uqcc-v1"
            and payload.get("qualification_design_id") == "CC_S12D_UQCC"
            and payload.get("qualification_elapsed_hours") == 6
            and payload.get("qualification_workers") == {"serial": 1, "concurrent": 4}
            and payload.get("endpoint_reference_count") == 285
            and payload.get("expected_exclusion_reasons")
            == ["unsupported_file_io_problem", "no_usable_paired_test"]
            and payload.get("candidate_skip_reasons")
            == [
                "duplicate_exclusion",
                "cross_label_exclusion",
                "non_python_exclusion",
                "nontext_or_empty_exclusion",
                "serialization_exclusion",
                "parity_only_excluded",
            ]
            and isinstance(sentinels, list)
            and 4 <= len(sentinels) <= 6
            and len({item.get("triplet_id") for item in sentinels if isinstance(item, Mapping)})
            == len(sentinels)
            and isinstance(faults, list)
            and 5 <= len(faults) <= 8
            and len(set(faults)) == len(faults)
            and set(payload.get("qualification_legal_terminals", ()))
            == set(CODECONTESTS_S12D_TERMINAL_STATES)
        ):
            raise ValueError("S12d qualification config differs from protocol")
    expected = {
        "protocol_id": "codecontests-s12-apcf-v1",
        "design_id": "CC_S12_APCF_PC_NAT_POLICY",
        "formal_namespace": (
            "formal_20260905_1"
            if schema == "codecontests-s12d-uqcc-config-v1"
            or (schema == "codecontests-s12-apcf-config-v1" and payload.get("execution_amendment") == "ADR0118")
            else "formal_20260904_1"
            if schema == "codecontests-s12c-rkqp-config-v1"
            else "formal_20260902_1"
        ),
        "stage_order": list(_STAGES),
        "response_token_quota": 583_975,
        "updates": 100,
        "max_length": 2_048,
        "policy_background_quota": 467_180,
        "policy_component_quota": 116_795,
        "planning_root_count": 4,
        "confirmatory_candidates": 16,
        "confirmatory_small_n": 12,
        "planning_sd_threshold": 0.019673586,
        "precision_half_width_target": 0.0125,
        "practical_margin": 0.02,
        "sampling_algorithm_id": "sha256_canonical_json_pair_order_v1",
        "failure_family_map": {
            "wrong_answer": "output_mismatch",
            "compile_failure": "execution_failure",
            "runtime_failure": "execution_failure",
            "timeout": "timeout",
        },
        "concurrency_ceilings": {"train": 2, "generation": 4, "validation": 4},
        "child_wall_hours": {"training": 12, "generation": 6, "scoring": 6},
        "formal_elapsed_hours": 160,
        "capacity_wait_budget_hours": 12,
        "legal_terminals": list((*_SCIENTIFIC_TERMINALS, *_INVALID_TERMINALS)),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"S12 config field differs from protocol: {key}")
    stage_arms = {
        stage: tuple(payload.get("stage_arms", {}).get(stage, ())) for stage in _STAGES
    }
    primary = {
        stage: tuple(payload.get("planning_primary_arms", {}).get(stage, ()))
        for stage in _STAGES
    }
    expected_stage_arms = {
        "PC": ("PC_CLEAN", "PC_MUTANT"),
        "NAT": ("NAT_R", "NAT_I"),
        "POLICY": ("POL_KEEP", "POL_FILTER", "POL_REPAIR"),
    }
    expected_primary = {
        "PC": ("PC_CLEAN", "PC_MUTANT"),
        "NAT": ("NAT_R", "NAT_I"),
        "POLICY": ("POL_FILTER", "POL_KEEP"),
    }
    if stage_arms != expected_stage_arms or primary != expected_primary:
        raise ValueError("S12 stage arm identity differs from protocol")
    return S12Config(
        protocol_id=payload["protocol_id"],
        design_id=payload["design_id"],
        formal_namespace=payload["formal_namespace"],
        stage_order=tuple(payload["stage_order"]),
        stage_arms=stage_arms,
        planning_primary_arms=primary,
        response_token_quota=payload["response_token_quota"],
        updates=payload["updates"],
        max_length=payload["max_length"],
        policy_background_quota=payload["policy_background_quota"],
        policy_component_quota=payload["policy_component_quota"],
        planning_root_count=payload["planning_root_count"],
        confirmatory_candidates=payload["confirmatory_candidates"],
        confirmatory_small_n=payload["confirmatory_small_n"],
        planning_sd_threshold=payload["planning_sd_threshold"],
        precision_half_width_target=payload["precision_half_width_target"],
        practical_margin=payload["practical_margin"],
        sampling_algorithm_id=payload["sampling_algorithm_id"],
        failure_family_map=dict(payload["failure_family_map"]),
        concurrency_ceilings=dict(payload["concurrency_ceilings"]),
        child_wall_hours=dict(payload["child_wall_hours"]),
        formal_elapsed_hours=payload["formal_elapsed_hours"],
        capacity_wait_budget_hours=payload["capacity_wait_budget_hours"],
        legal_terminals=tuple(payload["legal_terminals"]),
    )


def _derive_root(protocol_id: str, stage: str, role: str, index: int, used: set[int]) -> int:
    base = f"{protocol_id}|{stage}|{role}|{index}"
    nonce = 0
    while True:
        material = base if nonce == 0 else f"{base}|collision_nonce|{nonce}"
        root = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:15], 16)
        if root not in used:
            used.add(root)
            return root
        nonce += 1


def build_s12_root_registry(
    config: S12Config, *, historical_roots: Sequence[int]
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Precompute all stage roots without consulting any model outcome."""

    used = {int(root) for root in historical_roots}
    registry: dict[str, dict[str, tuple[int, ...]]] = {}
    for stage in config.stage_order:
        registry[stage] = {}
        for role, count in (
            ("planning", config.planning_root_count),
            ("confirmatory", config.confirmatory_candidates),
        ):
            registry[stage][role] = tuple(
                _derive_root(config.protocol_id, stage, role, index, used)
                for index in range(count)
            )
    return registry


class StagePlanningMask:
    """One runtime-private scientific-arm to A/B mapping for an entire stage."""

    __slots__ = ("stage", "identity_sha256", "_mapping")

    def __init__(self, stage: str, arms: tuple[str, str], protocol_id: str) -> None:
        orientation = int(
            hashlib.sha256(f"{protocol_id}|{stage}|planning-mask-v1".encode()).hexdigest(),
            16,
        ) & 1
        aliases = ("A", "B") if orientation == 0 else ("B", "A")
        self.stage = stage
        self._mapping = dict(zip(arms, aliases, strict=True))
        self.identity_sha256 = _sha256(
            [protocol_id, stage, sorted(self._mapping.items())]
        )

    def alias_for(self, scientific_arm: str) -> str:
        try:
            return self._mapping[scientific_arm]
        except KeyError as exc:
            raise ValueError("arm is not part of the stage-global planning mask") from exc

    def card_identity(self, root: int) -> dict[str, object]:
        return {
            "stage": self.stage,
            "root": int(root),
            "mapping_identity_sha256": self.identity_sha256,
        }

    def validate_cards(self, cards: Sequence[Mapping[str, object]]) -> None:
        if len(cards) != 4 or any(
            card.get("stage") != self.stage
            or card.get("mapping_identity_sha256") != self.identity_sha256
            for card in cards
        ):
            raise ValueError("planning cards must share one stage-global planning mask")
        roots = [card.get("root") for card in cards]
        if len(set(roots)) != 4:
            raise ValueError("planning roots must be unique")


def seal_stage_planning_mask(config: S12Config, stage: str) -> StagePlanningMask:
    if stage not in config.stage_order:
        raise ValueError("unknown S12 stage")
    return StagePlanningMask(stage, config.planning_primary_arms[stage], config.protocol_id)


def _block_key(pair: Mapping[str, object]) -> tuple[str, str]:
    provenance = pair.get("provenance")
    failure_family = pair.get("failure_family")
    if provenance not in {"original", "alternative"} or failure_family not in {
        "output_mismatch",
        "execution_failure",
        "timeout",
    }:
        raise ValueError("S12 pair block identity is invalid")
    return str(provenance), str(failure_family)


def _pair_digest(
    config: S12Config, stage: str, root: int, pair: Mapping[str, object]
) -> str:
    provenance, failure_family = _block_key(pair)
    pair_id = pair.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("S12 pair identity is invalid")
    return hashlib.sha256(
        _canonical_json(
            [
                config.protocol_id,
                stage,
                int(root),
                provenance,
                failure_family,
                pair_id,
            ]
        )
    ).hexdigest()


def s12_pair_order(
    config: S12Config,
    stage: str,
    root: int,
    pairs: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return the literal frozen block/digest/pair order; arm and cycle are absent."""

    if stage not in config.stage_order:
        raise ValueError("unknown S12 stage")
    ordered = sorted(
        pairs,
        key=lambda pair: (
            _block_key(pair),
            _pair_digest(config, stage, root, pair),
            str(pair.get("pair_id")),
        ),
    )
    identities = tuple(str(pair["pair_id"]) for pair in ordered)
    if len(identities) != len(set(identities)):
        raise ValueError("S12 pair identities must be unique")
    return identities


def _unit(pair: Mapping[str, object], key: str) -> TokenizedUnit:
    units = pair.get("units")
    value = units.get(key) if isinstance(units, Mapping) else None
    if not isinstance(value, TokenizedUnit) or value.sampling_id != pair.get("pair_id"):
        raise ValueError(f"S12 pair lacks aligned tokenized role {key}")
    if len(value.token_ids) > 2_048:
        raise ValueError("S12 serialization truncation is prohibited")
    return value


def materialize_nat_replace_pairs(csr_rows: Sequence[Mapping[str, object]], *, unit_resolver) -> tuple[dict[str, object], ...]:
    """Select only certified policy support, without resolving R/M or nonmembers."""
    if any(type(row.get("policy_member")) is not bool for row in csr_rows):
        raise ValueError("NAT-REPLACE membership must be frozen booleans")
    selected = [row for row in csr_rows if row["policy_member"]]
    if len(selected) != 230 or len({row.get("triplet_id") for row in selected}) != 230:
        raise ValueError("NAT-REPLACE requires exactly 230 unique policy members")
    pairs = []
    for row in selected:
        if not all(isinstance(row.get(key), str) and row[key] for key in
                   ("triplet_id", "candidate_problem_id", "i_id", "n_id")):
            raise ValueError("NAT-REPLACE source locator is missing")
        if row.get("r_provenance") not in {"original", "alternative"} or row.get("failure_family") not in {
                "execution_failure", "output_mismatch", "timeout"}:
            raise ValueError("NAT-REPLACE block identity is invalid")
        pairs.append(dict(pair_id=row["triplet_id"], problem_id=row["candidate_problem_id"],
                          provenance=row["r_provenance"], failure_family=row["failure_family"],
                          units={role: unit_resolver(row, role) for role in ("I", "N")}))
    return tuple(pairs)


def materialize_s12c_promoted_pairs(
    csr_rows: Sequence[Mapping[str, object]],
    *,
    unit_resolver,
) -> dict[str, tuple[dict[str, object], ...]]:
    """Project frozen CSR memberships into the sole S12 plan input shape."""

    nat_pairs: list[dict[str, object]] = []
    pc_pairs: list[dict[str, object]] = []
    policy_pairs: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in csr_rows:
        pair_id = row.get("triplet_id")
        problem_id = row.get("candidate_problem_id")
        if (
            not isinstance(pair_id, str) or not pair_id or pair_id in seen
            or not isinstance(problem_id, str) or not problem_id
        ):
            raise ValueError("S12c promoted row identity is invalid")
        seen.add(pair_id)
        flags = tuple(row.get(key) for key in (
            "natural_member", "mutation_member", "policy_member"
        ))
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("S12c promoted membership is not frozen")
        natural, mutation, policy = flags
        if (mutation or policy) and not natural:
            raise ValueError("S12c promoted subsets are not nested")
        if not natural:
            continue
        provenance = row.get("r_provenance")
        failure_family = row.get("failure_family")
        if provenance not in {"original", "alternative"} or failure_family not in {
            "output_mismatch", "execution_failure", "timeout"
        }:
            raise ValueError("S12c promoted block identity is invalid")
        if not all(isinstance(row.get(key), str) and row.get(key) for key in (
            "selected_r_id", "i_id"
        )):
            raise ValueError("S12c promoted natural locator is missing")
        base = {
            "pair_id": pair_id,
            "problem_id": problem_id,
            "provenance": provenance,
            "failure_family": failure_family,
            "units": {
                "I": unit_resolver(row, "I"),
                "R": unit_resolver(row, "R"),
            },
        }
        nat_pairs.append(base)
        if mutation:
            if not isinstance(row.get("selected_m_id"), str):
                raise ValueError("S12c promoted mutant locator is missing")
            pc_pairs.append(
                {
                    **base,
                    "units": {
                        "PC_CLEAN": unit_resolver(row, "PC_CLEAN"),
                        "PC_MUTANT": unit_resolver(row, "PC_MUTANT"),
                    },
                }
            )
        if policy:
            if not isinstance(row.get("n_id"), str):
                raise ValueError("S12c promoted neutral locator is missing")
            policy_pairs.append(
                {**base, "units": {**base["units"], "N": unit_resolver(row, "N")}}
            )
    return {
        "nat_pairs": tuple(nat_pairs),
        "pc_pairs": tuple(pc_pairs),
        "policy_pairs": tuple(policy_pairs),
    }


def _midpoint_roles(stage: str) -> tuple[str, str]:
    return {
        "PC": ("PC_CLEAN", "PC_MUTANT"),
        "NAT": ("R", "I"),
        "POLICY": ("N", "I"),
    }[stage]


def _unit_key(stage: str, arm: str) -> str:
    if stage == "PC":
        return arm
    if stage == "NAT":
        return {"NAT_R": "R", "NAT_I": "I"}[arm]
    return {"POL_KEEP": "I", "POL_FILTER": "N", "POL_REPAIR": "R"}[arm]


def _block_quotas(
    pairs: Sequence[Mapping[str, object]], *, stage: str, quota: int
) -> dict[tuple[str, str], int]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for pair in pairs:
        grouped.setdefault(_block_key(pair), []).append(pair)
    left, right = _midpoint_roles(stage)
    weights = {
        block: sum(
            _unit(pair, left).response_tokens + _unit(pair, right).response_tokens
            for pair in members
        )
        for block, members in grouped.items()
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("S12 block mass is empty")
    allocations = {block: quota * weight // total for block, weight in weights.items()}
    remainder = quota - sum(allocations.values())
    ranked = sorted(
        weights,
        key=lambda block: (-(quota * weights[block] % total), block),
    )
    for block in ranked[:remainder]:
        allocations[block] += 1
    if any(value <= 0 for value in allocations.values()) or sum(allocations.values()) != quota:
        raise ValueError("S12 largest-remainder block allocation failed")
    return allocations


def _ordered_block_pairs(
    config: S12Config,
    stage: str,
    root: int,
    pairs: Sequence[Mapping[str, object]],
    block: tuple[str, str],
) -> tuple[Mapping[str, object], ...]:
    members = tuple(pair for pair in pairs if _block_key(pair) == block)
    return tuple(
        sorted(
            members,
            key=lambda pair: (
                _pair_digest(config, stage, root, pair),
                str(pair["pair_id"]),
            ),
        )
    )


def _take_block_quota(
    ordered: Sequence[Mapping[str, object]],
    *,
    unit_key: str,
    quota: int,
    component: str | None,
    cursor: tuple[int, int] = (0, 0),
) -> tuple[tuple[Exposure, ...], tuple[int, int]]:
    if not ordered or quota <= 0:
        raise ValueError("S12 block exposure requires support and positive quota")
    cycle, position = cursor
    exposures: list[Exposure] = []
    retained_total = 0
    while retained_total < quota:
        if position >= len(ordered):
            cycle += position // len(ordered)
            position %= len(ordered)
        pair = ordered[position]
        value = _unit(pair, unit_key)
        retained = min(value.response_tokens, quota - retained_total)
        prompt_tokens = value.loss_mask.index(True)
        retained_length = prompt_tokens + retained
        exposures.append(
            Exposure(
                unit_id=value.unit_id,
                cycle=cycle,
                token_ids=value.token_ids[:retained_length],
                loss_mask=value.loss_mask[:retained_length],
                retained_response_tokens=retained,
                complete=retained == value.response_tokens,
                component=component,
            )
        )
        retained_total += retained
        position += 1
    return tuple(exposures), (cycle, position)


def _partition_s12_updates(
    exposures: tuple[Exposure, ...], *, quota: int, updates: int
) -> tuple[tuple[Exposure, ...], ...]:
    if len(exposures) < updates or sum(item.retained_response_tokens for item in exposures) != quota:
        raise ValueError("S12 exposure stream cannot close Q/U")
    cumulative: list[int] = []
    total = 0
    for exposure in exposures:
        total += exposure.retained_response_tokens
        cumulative.append(total)
    boundaries: list[int] = []
    previous = 0
    for index in range(1, updates):
        maximum = len(exposures) - (updates - index)
        candidates = range(previous + 1, maximum + 1)
        target = (index * quota + updates // 2) // updates
        selected = min(candidates, key=lambda position: (abs(cumulative[position - 1] - target), position))
        boundaries.append(selected)
        previous = selected
    bins: list[tuple[Exposure, ...]] = []
    start = 0
    for end in (*boundaries, len(exposures)):
        bins.append(exposures[start:end])
        start = end
    if any(not update for update in bins):
        raise ValueError("S12 exposure plan contains an empty update")
    return tuple(bins)


def _ledger(exposures: Sequence[Exposure]) -> tuple[ExposureLedgerEntry, ...]:
    totals: dict[str, list[int]] = {}
    for exposure in exposures:
        value = totals.setdefault(exposure.unit_id, [0, 0, 0])
        value[0 if exposure.complete else 1] += 1
        value[2] += exposure.retained_response_tokens
    return tuple(
        ExposureLedgerEntry(unit_id, values[0], values[1], values[2])
        for unit_id, values in totals.items()
    )


def _plan(exposures: tuple[Exposure, ...], config: S12Config) -> TrainingPlan:
    bins = _partition_s12_updates(
        exposures, quota=config.response_token_quota, updates=config.updates
    )
    packs = single_unit_packs(bins, max_length=config.max_length)
    return TrainingPlan(exposures, bins, packs, _ledger(exposures))


def build_s12_stage_plans(
    config: S12Config,
    *,
    stage: str,
    root: int,
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, TrainingPlan]:
    """Materialize one root's stage plans behind the paired-block interface."""

    if stage not in config.stage_order or not pairs:
        raise ValueError("S12 stage plan identity is invalid")
    by_block = {
        block: _ordered_block_pairs(config, stage, root, pairs, block)
        for block in sorted({_block_key(pair) for pair in pairs})
    }
    if stage != "POLICY":
        quotas = _block_quotas(pairs, stage=stage, quota=config.response_token_quota)
        result: dict[str, TrainingPlan] = {}
        for arm in config.stage_arms[stage]:
            arm_exposures: list[Exposure] = []
            for block, ordered in by_block.items():
                values, _ = _take_block_quota(
                    ordered,
                    unit_key=_unit_key(stage, arm),
                    quota=quotas[block],
                    component=stage,
                )
                arm_exposures.extend(values)
            result[arm] = _plan(tuple(arm_exposures), config)
        return result

    background_quotas = _block_quotas(
        pairs, stage=stage, quota=config.policy_background_quota
    )
    component_quotas = _block_quotas(
        pairs, stage=stage, quota=config.policy_component_quota
    )
    common: dict[tuple[str, str], tuple[Exposure, ...]] = {}
    cursors: dict[tuple[str, str], tuple[int, int]] = {}
    for block, ordered in by_block.items():
        common[block], cursors[block] = _take_block_quota(
            ordered,
            unit_key="N",
            quota=background_quotas[block],
            component="B80",
        )
    background = tuple(item for block in by_block for item in common[block])
    background_bins = _partition_s12_updates(
        background,
        quota=config.policy_background_quota,
        updates=config.updates,
    )
    result = {}
    for arm in config.stage_arms[stage]:
        component_exposures: list[Exposure] = []
        for block, ordered in by_block.items():
            cursor = cursors[block] if arm == "POL_FILTER" else (0, 0)
            values, _ = _take_block_quota(
                ordered,
                unit_key=_unit_key(stage, arm),
                quota=component_quotas[block],
                component={
                    "POL_KEEP": "I20",
                    "POL_FILTER": "B20",
                    "POL_REPAIR": "R20",
                }[arm],
                cursor=cursor,
            )
            component_exposures.extend(values)
        component_stream = tuple(component_exposures)
        component_bins = _partition_s12_updates(
            component_stream,
            quota=config.policy_component_quota,
            updates=config.updates,
        )
        update_bins = tuple(
            background_bin + component_bin
            for background_bin, component_bin in zip(
                background_bins, component_bins, strict=True
            )
        )
        exposures = tuple(item for update in update_bins for item in update)
        packs = single_unit_packs(update_bins, max_length=config.max_length)
        result[arm] = TrainingPlan(
            exposures,
            update_bins,
            packs,
            _ledger(exposures),
        )
    return result


def build_scope_pc_roots(*, historical_roots: Sequence[int]) -> tuple[int, ...]:
    """Four mechanically bound roots; collision handling is not seed search."""
    used = set(historical_roots)
    roots = []
    for index in range(1, 5):
        nonce = 0
        while True:
            text = f"{SCOPE_PC_PROTOCOL_ID}/root-v1|{index}|{nonce}"
            root = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 2147483646 + 1
            if root not in used:
                break
            nonce += 1
        roots.append(root)
        used.add(root)
    return tuple(roots)


def build_nat_replace_plans(
    pairs: Sequence[Mapping[str, object]],
    *,
    root: int,
    protocol_id: str = NAT_REPLACE_PROTOCOL_ID,
) -> dict[str, TrainingPlan]:
    """Only the fixed I/N replacement arms; reuse the historical quota algorithm."""
    if len(pairs) != 230 or len({p["pair_id"] for p in pairs}) != 230:
        raise ValueError("NAT-REPLACE requires exactly 230 unique qualified pairs")
    if any(set(p["units"]) != {"I", "N"} for p in pairs):
        raise ValueError("NAT-REPLACE input must contain only I/N units")
    if not isinstance(root, int) or not 0 < root < 2147483647:
        raise ValueError("NAT-REPLACE root must be a positive modulus residue")
    config = load_s12_config(Path(__file__).parents[3] / "configs/codecontests_s12_apcf_v1.json")
    if protocol_id not in {
        NAT_REPLACE_PROTOCOL_ID,
        NAT_REPLICA_PROTOCOL_ID,
        NAT_MSEP_PROTOCOL_ID,
    }:
        raise ValueError("NAT replacement plan protocol is unknown")
    config = replace(config, protocol_id=protocol_id,
                     stage_order=("POLICY",), stage_arms={"POLICY": ("POL_KEEP", "POL_FILTER")})
    return build_s12_stage_plans(config, stage="POLICY", root=root, pairs=pairs)


def build_scope_pc_plans(pairs: Sequence[Mapping[str, object]], *, root: int) -> dict[str, TrainingPlan]:
    """Reuse the original paired midpoint/quota algorithm on T alone."""
    config = load_s12_config(Path(__file__).parents[3] / "configs/codecontests_s12_apcf_v1.json")
    config = replace(config, protocol_id=SCOPE_PC_PROTOCOL_ID)
    inputs = []
    for pair in pairs:
        units = {}
        for role, key in (("R", "PC_CLEAN"), ("M", "PC_MUTANT")):
            tokens = tuple(pair["token_ids"][role])
            prompt_count = int(pair["prompt_tokens"])
            units[key] = TokenizedUnit(unit_id=f"{pair['pair_id']}/{'CLEAN' if role == 'R' else 'MUTANT'}",
                                      token_ids=tokens,
                                      loss_mask=(False,) * prompt_count + (True,) * (len(tokens) - prompt_count),
                                      sampling_id=pair["pair_id"])
        inputs.append({**pair, "units": units})
    plans = build_s12_stage_plans(config, stage="PC", root=root, pairs=inputs)
    return {"SCOPE_R": plans["PC_CLEAN"], "SCOPE_M": plans["PC_MUTANT"]}


def s12_training_plan_sha256(plan: TrainingPlan) -> str:
    return _sha256(
        [
            [
                exposure.unit_id,
                exposure.cycle,
                hashlib.sha256(bytes(str(exposure.token_ids), "utf-8")).hexdigest(),
                exposure.retained_response_tokens,
                exposure.complete,
                exposure.component,
            ]
            for exposure in plan.exposures
        ]
    )


def dump_s12_training_plan(path: Path, plan: TrainingPlan) -> str:
    """Persist one exact child plan without raw source text or outcome data."""

    identity = s12_training_plan_sha256(plan)
    payload = {
        "schema": "codecontests-s12-training-plan-v1",
        "plan_sha256": identity,
        "updates": [
            [
                {
                    "unit_id": item.unit_id,
                    "cycle": item.cycle,
                    "token_ids": list(item.token_ids),
                    "loss_mask": list(item.loss_mask),
                    "retained_response_tokens": item.retained_response_tokens,
                    "complete": item.complete,
                    "component": item.component,
                }
                for item in update
            ]
            for update in plan.update_bins
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), "utf-8")
    temporary.replace(path)
    return identity


def load_s12_training_plan(path: Path) -> TrainingPlan:
    """Load and identity-check a plan written by :func:`dump_s12_training_plan`."""

    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema") != "codecontests-s12-training-plan-v1":
        raise ValueError("S12 serialized training plan schema is invalid")
    raw_updates = payload.get("updates")
    if not isinstance(raw_updates, list) or len(raw_updates) != 100:
        raise ValueError("S12 serialized training plan must contain 100 updates")
    updates = tuple(
        tuple(
            Exposure(
                unit_id=str(item["unit_id"]),
                cycle=int(item["cycle"]),
                token_ids=tuple(int(value) for value in item["token_ids"]),
                loss_mask=tuple(bool(value) for value in item["loss_mask"]),
                retained_response_tokens=int(item["retained_response_tokens"]),
                complete=bool(item["complete"]),
                component=item.get("component"),
            )
            for item in update
        )
        for update in raw_updates
    )
    if any(not update for update in updates):
        raise ValueError("S12 serialized training plan contains an empty update")
    exposures = tuple(item for update in updates for item in update)
    plan = TrainingPlan(
        exposures,
        updates,
        single_unit_packs(updates, max_length=2_048),
        _ledger(exposures),
    )
    if payload.get("plan_sha256") != s12_training_plan_sha256(plan):
        raise ValueError("S12 serialized training plan identity differs")
    return plan


def build_s12_run_cards(
    config: S12Config,
    *,
    stage: str,
    role: str,
    roots: Sequence[int],
    plans_by_root: Mapping[int, Mapping[str, TrainingPlan]],
    planning_mask: StagePlanningMask | None = None,
) -> tuple[dict[str, object], ...]:
    """Build opaque planning cards or scientific confirmatory cards."""

    if role not in {"planning", "confirmatory"} or stage not in config.stage_order:
        raise ValueError("S12 run-card stage/role is invalid")
    arms = config.planning_primary_arms[stage] if role == "planning" else config.stage_arms[stage]
    if role == "planning" and planning_mask is None:
        raise ValueError("S12 planning cards require one sealed stage-global mask")
    if role == "confirmatory" and planning_mask is not None:
        raise ValueError("S12 confirmatory cards do not carry a planning mask")
    cards: list[dict[str, object]] = []
    for root in roots:
        root_plans = plans_by_root.get(int(root))
        if not isinstance(root_plans, Mapping):
            raise ValueError("S12 run-card plan registry is incomplete")
        for ordinal, arm in enumerate(arms):
            if arm not in root_plans:
                raise ValueError("S12 run-card lacks an arm plan")
            card: dict[str, object] = {
                "schema": "codecontests-s12-run-card-v1",
                "protocol_id": config.protocol_id,
                "design_id": config.design_id,
                "stage": stage,
                "role": role,
                "root": int(root),
                "arm_ordinal": ordinal,
                "response_token_quota": config.response_token_quota,
                "updates": config.updates,
                "max_length": config.max_length,
                "packing_path": "single_unit",
                "role_seeds": {
                    seed_role: derive_codecontests_role_seed(int(root), seed_role)
                    for seed_role in CODECONTESTS_ROLE_NAMES
                },
                "plan_sha256": s12_training_plan_sha256(root_plans[arm]),
                "outcomes_opened": False,
            }
            if role == "planning":
                assert planning_mask is not None
                card["arm_alias"] = planning_mask.alias_for(arm)
                card["mapping_identity_sha256"] = planning_mask.identity_sha256
            else:
                card["scientific_arm"] = arm
            card["card_sha256"] = _sha256(card)
            cards.append(card)
    if role == "planning":
        assert planning_mask is not None
        planning_mask.validate_cards(
            [planning_mask.card_identity(root) for root in roots]
        )
    return tuple(cards)
