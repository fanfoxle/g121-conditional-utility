"""Quality-blind baseline selection and frozen P1 tri-state derivation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..protocol import baseline_priority, canonical_uuid


@dataclass(frozen=True)
class LabelAggregate:
    value: float
    count: int


@dataclass(frozen=True)
class SyntheticResponse:
    response_id: str
    rank: int | None
    quality: LabelAggregate | None
    helpfulness: LabelAggregate | None


@dataclass(frozen=True)
class SyntheticRoot:
    root_id: str
    responses: tuple[SyntheticResponse, ...]


class CandidateState(str, Enum):
    CANDIDATE = "candidate"
    ELIGIBLE_NONCANDIDATE = "eligible_noncandidate"
    ABSTENTION = "abstention"


@dataclass(frozen=True)
class CandidateWitness:
    root_id: str
    selected_response_id: str
    sibling_ranks: tuple[tuple[str, int], ...]
    quality: LabelAggregate | None
    helpfulness: LabelAggregate | None
    trigger_dimensions: tuple[str, ...]


@dataclass(frozen=True)
class CandidateDecision:
    root_id: str
    selected_response_id: str
    state: CandidateState
    reason: str
    witness: CandidateWitness


def _valid_rank(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _supported_value(label: LabelAggregate | None) -> bool:
    return (
        label is not None
        and isinstance(label.count, int)
        and not isinstance(label.count, bool)
        and label.count >= 3
        and isinstance(label.value, (int, float))
        and not isinstance(label.value, bool)
        and math.isfinite(float(label.value))
        and 0.0 <= float(label.value) <= 1.0
    )


def _select_response(root: SyntheticRoot) -> SyntheticResponse:
    root_id = canonical_uuid(root.root_id)
    if not root.responses:
        raise ValueError("root must contain at least one eligible response")
    response_ids = [canonical_uuid(item.response_id) for item in root.responses]
    if len(set(response_ids)) != len(response_ids):
        raise ValueError("response identifiers must be unique within a root")
    # Only stable identifiers cross this quality-blind selection seam.
    selected_id = select_baseline_response_id(root_id, tuple(response_ids))
    return next(item for item in root.responses if item.response_id == selected_id)


def select_baseline_response_id(root_id: str, response_ids: tuple[str, ...]) -> str:
    """Select from identifiers alone so rank/rating data cannot enter the hash seam."""

    root_id = canonical_uuid(root_id)
    canonical = tuple(canonical_uuid(value) for value in response_ids)
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("baseline response identifiers must be nonempty and unique")
    return min(canonical, key=lambda value: (baseline_priority(root_id, value), value))


def _derive_one(root: SyntheticRoot) -> CandidateDecision:
    root_id = canonical_uuid(root.root_id)
    selected = _select_response(root)
    ranks = tuple((item.response_id, item.rank) for item in root.responses)
    valid_rank_pairs = tuple((item_id, rank) for item_id, rank in ranks if _valid_rank(rank))
    witness = CandidateWitness(
        root_id=root_id,
        selected_response_id=selected.response_id,
        sibling_ranks=valid_rank_pairs,
        quality=selected.quality,
        helpfulness=selected.helpfulness,
        trigger_dimensions=(),
    )
    if len(root.responses) < 2:
        return CandidateDecision(
            root_id, selected.response_id, CandidateState.ABSTENTION,
            "eligible_sibling_count_below_2", witness,
        )
    if len(valid_rank_pairs) != len(root.responses):
        return CandidateDecision(
            root_id, selected.response_id, CandidateState.ABSTENTION,
            "rank_missing_or_malformed", witness,
        )

    maximum = max(rank for _, rank in valid_rank_pairs)
    bottom_ids = [item_id for item_id, rank in valid_rank_pairs if rank == maximum]
    if len(bottom_ids) != 1:
        return CandidateDecision(
            root_id, selected.response_id, CandidateState.ABSTENTION,
            "rank_not_unique_bottom", witness,
        )
    if selected.response_id != bottom_ids[0]:
        return CandidateDecision(
            root_id, selected.response_id, CandidateState.ELIGIBLE_NONCANDIDATE,
            "selected_response_not_bottom", witness,
        )

    triggers = tuple(
        name
        for name, label in (
            ("quality", selected.quality),
            ("helpfulness", selected.helpfulness),
        )
        if _supported_value(label) and float(label.value) < 0.5
    )
    if triggers:
        witness = CandidateWitness(
            root_id=witness.root_id,
            selected_response_id=witness.selected_response_id,
            sibling_ranks=witness.sibling_ranks,
            quality=witness.quality,
            helpfulness=witness.helpfulness,
            trigger_dimensions=triggers,
        )
        return CandidateDecision(
            root_id, selected.response_id, CandidateState.CANDIDATE,
            "unique_bottom_with_supported_low_rating", witness,
        )

    both_supported_nonlow = all(
        _supported_value(label) and float(label.value) >= 0.5
        for label in (selected.quality, selected.helpfulness)
    )
    if both_supported_nonlow:
        return CandidateDecision(
            root_id, selected.response_id, CandidateState.ELIGIBLE_NONCANDIDATE,
            "unique_bottom_with_both_supported_nonlow_ratings", witness,
        )
    return CandidateDecision(
        root_id, selected.response_id, CandidateState.ABSTENTION,
        "unique_bottom_rating_evidence_inconclusive", witness,
    )


def derive_candidates(roots: tuple[SyntheticRoot, ...]) -> tuple[CandidateDecision, ...]:
    """Derive one quality-blind baseline and frozen tri-state per root."""

    root_ids = [canonical_uuid(root.root_id) for root in roots]
    if len(set(root_ids)) != len(root_ids):
        raise ValueError("root identifiers must be unique")
    return tuple(_derive_one(root) for root in roots)
