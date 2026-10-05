"""SAM1 ordering, BQ1 quota/update bins, PKO1 packing, and PAD1 fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import Callable

from ..protocol import (
    CODECONTESTS_PROTOCOL_ID,
    CODECONTESTS_RESPONSE_TOKEN_QUOTA,
    CODECONTESTS_UPDATES,
    CODECONTESTS_V2_POLICY_BACKGROUND_QUOTA,
    CODECONTESTS_V2_POLICY_TREATMENT_QUOTA,
    CODECONTESTS_V2_PROTOCOL_ID,
    CODECONTESTS_V2_STAGE_ORDER,
    NAMESPACE,
    SAMPLER_KEY,
    derive_codecontests_role_seed,
    derive_codecontests_v2_role_seed,
    unit_id as canonical_unit_id,
)


_CODECONTESTS_SAMPLING_ID = re.compile(r"triplet:(0|[1-9][0-9]*)\Z")
_CODECONTESTS_UNIT_ID = re.compile(
    r"triplet:(0|[1-9][0-9]*)/(?:I|R|N|CLEAN|MUTANT)\Z"
)


@dataclass(frozen=True)
class TokenizedUnit:
    unit_id: str
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    sampling_id: str | None = None

    def __post_init__(self) -> None:
        if self.sampling_id is None:
            _validate_unit_id(self.unit_id)
        else:
            _validate_codecontests_unit_id(self.unit_id, self.sampling_id)
        if not self.token_ids or len(self.token_ids) != len(self.loss_mask):
            raise ValueError("token IDs and loss mask must be nonempty and aligned")
        response_tokens = sum(self.loss_mask)
        if response_tokens <= 0:
            raise ValueError("each unit requires at least one loss-bearing token")
        first = self.loss_mask.index(True)
        if any(self.loss_mask[:first]) or not all(self.loss_mask[first:]):
            raise ValueError("SER1 loss mask must be one contiguous response suffix")

    @property
    def response_tokens(self) -> int:
        return sum(self.loss_mask)


@dataclass(frozen=True)
class Exposure:
    unit_id: str
    cycle: int
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    retained_response_tokens: int
    complete: bool
    component: str | None = None


@dataclass(frozen=True)
class ExposureLedgerEntry:
    unit_id: str
    complete_exposures: int
    partial_exposures: int
    response_tokens: int


@dataclass(frozen=True)
class Pack:
    update_index: int
    pack_index: int
    exposures: tuple[Exposure, ...]
    total_tokens: int


@dataclass(frozen=True)
class PackedTensors:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    position_ids: tuple[int, ...]
    causal_mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True)
class PaddedBatch:
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrainingPlan:
    exposures: tuple[Exposure, ...]
    update_bins: tuple[tuple[Exposure, ...], ...]
    packs: tuple[tuple[Pack, ...], ...]
    exposure_ledger: tuple[ExposureLedgerEntry, ...]


def _validate_unit_id(value: str) -> None:
    pieces = value.split("/")
    if len(pieces) != 2 or canonical_unit_id(pieces[0], pieces[1]) != value:
        raise ValueError("unit ID must use two canonical UUIDs")


def _validate_codecontests_unit_id(value: str, sampling_id: str) -> None:
    if (
        _CODECONTESTS_SAMPLING_ID.fullmatch(sampling_id) is None
        or _CODECONTESTS_UNIT_ID.fullmatch(value) is None
        or value.rsplit("/", 1)[0] != sampling_id
    ):
        raise ValueError("CodeContests unit ID must be a frozen triplet/arm identity")


def sam1_priority(stable_unit_id: str, *, root: int, cycle: int) -> bytes:
    _validate_unit_id(stable_unit_id)
    if (
        isinstance(root, bool)
        or not isinstance(root, int)
        or root < 0
        or isinstance(cycle, bool)
        or not isinstance(cycle, int)
        or cycle < 0
    ):
        raise ValueError("SAM1 root and cycle must be nonnegative integers")
    message = f"{NAMESPACE}|sam1|{root}|{cycle}|{stable_unit_id}".encode("utf-8")
    return hmac.new(SAMPLER_KEY, message, hashlib.sha256).digest()


def sam1_cycle_order(
    units: tuple[TokenizedUnit, ...], *, root: int, cycle: int
) -> tuple[TokenizedUnit, ...]:
    identifiers = [item.unit_id for item in units]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("SAM1 units must have unique stable IDs")
    return tuple(
        sorted(
            units,
            key=lambda item: (sam1_priority(item.unit_id, root=root, cycle=cycle), item.unit_id),
        )
    )


def codecontests_priority(stable_triplet_id: str, *, root: int, cycle: int) -> bytes:
    """Return an arm-invariant root/cycle priority for one frozen triplet."""

    if _CODECONTESTS_SAMPLING_ID.fullmatch(stable_triplet_id) is None:
        raise ValueError("CodeContests sampling ID must be a frozen triplet identity")
    if (
        isinstance(root, bool)
        or not isinstance(root, int)
        or root < 0
        or isinstance(cycle, bool)
        or not isinstance(cycle, int)
        or cycle < 0
    ):
        raise ValueError("CodeContests root and cycle must be nonnegative integers")
    seed = derive_codecontests_role_seed(root, "sampler")
    message = (
        f"{CODECONTESTS_PROTOCOL_ID}|sampler|{seed}|cycle|{cycle}|"
        f"{stable_triplet_id}"
    )
    return hashlib.sha256(message.encode("utf-8")).digest()


def codecontests_cycle_order(
    units: tuple[TokenizedUnit, ...], *, root: int, cycle: int
) -> tuple[TokenizedUnit, ...]:
    sampling_ids = [item.sampling_id for item in units]
    if any(value is None for value in sampling_ids) or len(set(sampling_ids)) != len(
        sampling_ids
    ):
        raise ValueError("CodeContests units require unique arm-invariant sampling IDs")
    return tuple(
        sorted(
            units,
            key=lambda item: (
                codecontests_priority(str(item.sampling_id), root=root, cycle=cycle),
                str(item.sampling_id),
            ),
        )
    )


def codecontests_v2_priority(
    stable_triplet_id: str, *, stage: str, root: int, cycle: int
) -> bytes:
    """Return the V2 stage/root priority shared by every arm in that stage."""

    if _CODECONTESTS_SAMPLING_ID.fullmatch(stable_triplet_id) is None:
        raise ValueError("CodeContests sampling ID must be a frozen triplet identity")
    if stage not in CODECONTESTS_V2_STAGE_ORDER:
        raise ValueError("stage is outside the frozen CodeContests V2 sequence")
    if (
        isinstance(root, bool)
        or not isinstance(root, int)
        or root < 0
        or isinstance(cycle, bool)
        or not isinstance(cycle, int)
        or cycle < 0
    ):
        raise ValueError("CodeContests root and cycle must be nonnegative integers")
    seed = derive_codecontests_v2_role_seed(stage, root, "sampler")
    message = (
        f"{CODECONTESTS_V2_PROTOCOL_ID}|{stage}|sampler|{seed}|cycle|{cycle}|"
        f"{stable_triplet_id}"
    )
    return hashlib.sha256(message.encode("utf-8")).digest()


def codecontests_v2_cycle_order(
    units: tuple[TokenizedUnit, ...], *, stage: str, root: int, cycle: int
) -> tuple[TokenizedUnit, ...]:
    sampling_ids = [item.sampling_id for item in units]
    if any(value is None for value in sampling_ids) or len(set(sampling_ids)) != len(
        sampling_ids
    ):
        raise ValueError("CodeContests units require unique arm-invariant sampling IDs")
    return tuple(
        sorted(
            units,
            key=lambda item: (
                codecontests_v2_priority(
                    str(item.sampling_id), stage=stage, root=root, cycle=cycle
                ),
                str(item.sampling_id),
            ),
        )
    )


def _quota_exposures_from_cursor(
    units: tuple[TokenizedUnit, ...],
    *,
    quota: int,
    cycle_order: Callable[[int], tuple[TokenizedUnit, ...]],
    start_cycle: int = 0,
    start_position: int = 0,
    component: str | None = None,
) -> tuple[tuple[Exposure, ...], tuple[int, int]]:
    """Take a quota and return the first unused whole-occurrence cursor."""

    if not units or quota <= 0 or isinstance(quota, bool):
        raise ValueError("BQ1 requires retained units and a positive quota")
    if start_cycle < 0 or start_position < 0:
        raise ValueError("stream cursor must be nonnegative")
    exposures: list[Exposure] = []
    cumulative = 0
    cycle = start_cycle
    position = start_position
    while cumulative < quota:
        ordered = cycle_order(cycle)
        if len(ordered) != len(units):
            raise ValueError("cycle order changed the frozen unit count")
        if position >= len(ordered):
            cycle += position // len(ordered)
            position %= len(ordered)
            ordered = cycle_order(cycle)
        while position < len(ordered) and cumulative < quota:
            item = ordered[position]
            remaining = quota - cumulative
            retained = min(item.response_tokens, remaining)
            prompt_tokens = item.loss_mask.index(True)
            retained_length = prompt_tokens + retained
            exposures.append(
                Exposure(
                    unit_id=item.unit_id,
                    cycle=cycle,
                    token_ids=item.token_ids[:retained_length],
                    loss_mask=item.loss_mask[:retained_length],
                    retained_response_tokens=retained,
                    complete=retained == item.response_tokens,
                    component=component,
                )
            )
            cumulative += retained
            # A terminal partial occurrence is consumed; the replacement stream
            # starts at the next occurrence rather than reusing its suffix.
            position += 1
        if position == len(ordered):
            cycle += 1
            position = 0
    return tuple(exposures), (cycle, position)


def _quota_exposures(
    units: tuple[TokenizedUnit, ...],
    *,
    root: int,
    quota: int,
    cycle_order: Callable[..., tuple[TokenizedUnit, ...]] = sam1_cycle_order,
) -> tuple[Exposure, ...]:
    if not units or quota <= 0 or isinstance(quota, bool):
        raise ValueError("BQ1 requires retained units and a positive quota")
    exposures: list[Exposure] = []
    cumulative = 0
    cycle = 0
    while cumulative < quota:
        for item in cycle_order(units, root=root, cycle=cycle):
            remaining = quota - cumulative
            if item.response_tokens <= remaining:
                exposures.append(
                    Exposure(
                        unit_id=item.unit_id,
                        cycle=cycle,
                        token_ids=item.token_ids,
                        loss_mask=item.loss_mask,
                        retained_response_tokens=item.response_tokens,
                        complete=True,
                    )
                )
                cumulative += item.response_tokens
            else:
                prompt_tokens = item.loss_mask.index(True)
                retained_length = prompt_tokens + remaining
                exposures.append(
                    Exposure(
                        unit_id=item.unit_id,
                        cycle=cycle,
                        token_ids=item.token_ids[:retained_length],
                        loss_mask=item.loss_mask[:retained_length],
                        retained_response_tokens=remaining,
                        complete=False,
                    )
                )
                cumulative += remaining
            if cumulative == quota:
                return tuple(exposures)
        cycle += 1
    raise AssertionError("unreachable BQ1 quota state")


def partition_update_bins(
    exposures: tuple[Exposure, ...],
    *,
    quota: int,
    updates: int,
    round_half_up_targets: bool = False,
) -> tuple[tuple[Exposure, ...], ...]:
    if updates <= 0 or quota <= 0 or len(exposures) < updates:
        raise ValueError("BQ1 requires positive Q/U and at least one exposure per update")
    if sum(item.retained_response_tokens for item in exposures) != quota:
        raise ValueError("exposure stream does not close the exact quota")
    if any(not item.complete for item in exposures[:-1]):
        raise ValueError("only the terminal exposure may be partial")
    cumulative: list[int] = []
    total = 0
    for item in exposures:
        if item.retained_response_tokens <= 0:
            raise ValueError("every exposure must carry loss tokens")
        total += item.retained_response_tokens
        cumulative.append(total)

    boundaries: list[int] = []
    previous = 0
    count = len(exposures)
    for boundary_index in range(1, updates):
        maximum = count - (updates - boundary_index)
        eligible = [
            position
            for position in range(previous + 1, maximum + 1)
            if exposures[position - 1].complete
        ]
        if not eligible:
            raise ValueError("no eligible complete-exposure BQ1 boundary")
        if round_half_up_targets:
            target = (boundary_index * quota + updates // 2) // updates
            selected = min(
                eligible,
                key=lambda position: (
                    abs(cumulative[position - 1] - target),
                    position,
                ),
            )
        else:
            selected = min(
                eligible,
                key=lambda position: (
                    abs(updates * cumulative[position - 1] - boundary_index * quota),
                    position,
                ),
            )
        boundaries.append(selected)
        previous = selected
    result: list[tuple[Exposure, ...]] = []
    start = 0
    for end in (*boundaries, count):
        result.append(exposures[start:end])
        start = end
    if any(not update for update in result):
        raise ValueError("BQ1 produced an empty optimizer update")
    if any(not item.complete for update in result[:-1] for item in update):
        raise ValueError("terminal partial exposure must remain in the final update")
    return tuple(result)


def next_fit_packs(
    update_bins: tuple[tuple[Exposure, ...], ...], *, max_length: int
) -> tuple[tuple[Pack, ...], ...]:
    if max_length <= 0:
        raise ValueError("maximum length must be positive")
    all_updates: list[tuple[Pack, ...]] = []
    for update_index, update in enumerate(update_bins):
        packs: list[Pack] = []
        open_exposures: list[Exposure] = []
        open_length = 0
        for item in update:
            length = len(item.token_ids)
            if length > max_length:
                raise ValueError("PKO1 cannot pack an overlength exposure")
            if open_exposures and open_length + length > max_length:
                packs.append(
                    Pack(update_index, len(packs), tuple(open_exposures), open_length)
                )
                open_exposures = []
                open_length = 0
            open_exposures.append(item)
            open_length += length
        if open_exposures:
            packs.append(Pack(update_index, len(packs), tuple(open_exposures), open_length))
        if not packs:
            raise ValueError("every update must contain at least one pack")
        all_updates.append(tuple(packs))
    return tuple(all_updates)


def single_unit_packs(
    update_bins: tuple[tuple[Exposure, ...], ...], *, max_length: int
) -> tuple[tuple[Pack, ...], ...]:
    """Keep every CodeContests exposure in its own physical forward pass."""

    if max_length <= 0:
        raise ValueError("maximum length must be positive")
    result: list[tuple[Pack, ...]] = []
    for update_index, update in enumerate(update_bins):
        if not update:
            raise ValueError("every update must contain at least one exposure")
        packs = []
        for pack_index, exposure in enumerate(update):
            if len(exposure.token_ids) > max_length:
                raise ValueError("CodeContests exposure exceeds maximum length")
            packs.append(
                Pack(
                    update_index=update_index,
                    pack_index=pack_index,
                    exposures=(exposure,),
                    total_tokens=len(exposure.token_ids),
                )
            )
        result.append(tuple(packs))
    return tuple(result)


def response_token_loss_weights(exposures: tuple[Exposure, ...]) -> tuple[float, ...]:
    """Weights whose application to mean-token NLL yields update token mean."""

    total = sum(item.retained_response_tokens for item in exposures)
    if total <= 0:
        raise ValueError("an optimizer update requires response-loss tokens")
    return tuple(item.retained_response_tokens / total for item in exposures)


def build_packed_tensors(pack: Pack) -> PackedTensors:
    input_ids = tuple(token for item in pack.exposures for token in item.token_ids)
    labels = tuple(
        token if active else -100
        for item in pack.exposures
        for token, active in zip(item.token_ids, item.loss_mask)
    )
    position_ids = tuple(
        position for item in pack.exposures for position in range(len(item.token_ids))
    )
    unit_indices = tuple(
        unit_index
        for unit_index, item in enumerate(pack.exposures)
        for _ in item.token_ids
    )
    local_positions = position_ids
    causal_mask = tuple(
        tuple(
            unit_indices[row] == unit_indices[column]
            and local_positions[column] <= local_positions[row]
            for column in range(len(input_ids))
        )
        for row in range(len(input_ids))
    )
    return PackedTensors(input_ids, labels, position_ids, causal_mask)


def right_pad(
    sequences: tuple[PackedTensors, ...], *, pad_token_id: int
) -> PaddedBatch:
    if not sequences:
        raise ValueError("PAD1 requires at least one physical sequence")
    width = max(len(item.input_ids) for item in sequences)
    input_ids: list[tuple[int, ...]] = []
    attention: list[tuple[int, ...]] = []
    labels: list[tuple[int, ...]] = []
    positions: list[tuple[int, ...]] = []
    for item in sequences:
        padding = width - len(item.input_ids)
        input_ids.append(item.input_ids + (pad_token_id,) * padding)
        attention.append((1,) * len(item.input_ids) + (0,) * padding)
        labels.append(item.labels + (-100,) * padding)
        positions.append(item.position_ids + (0,) * padding)
    return PaddedBatch(tuple(input_ids), tuple(attention), tuple(labels), tuple(positions))


def _ledger(exposures: tuple[Exposure, ...]) -> tuple[ExposureLedgerEntry, ...]:
    totals: dict[str, list[int]] = {}
    for item in exposures:
        aggregate = totals.setdefault(item.unit_id, [0, 0, 0])
        aggregate[0 if item.complete else 1] += 1
        aggregate[2] += item.retained_response_tokens
    return tuple(
        ExposureLedgerEntry(unit_id_value, values[0], values[1], values[2])
        for unit_id_value, values in totals.items()
    )


def build_training_plan(
    units: tuple[TokenizedUnit, ...],
    *,
    root: int,
    quota: int,
    updates: int,
    max_length: int,
) -> TrainingPlan:
    """Materialize the frozen deterministic stream without invoking a trainer."""

    if any(len(item.token_ids) > max_length for item in units):
        raise ValueError("OV1 requires all admitted ordinary units to fit L")
    exposures = _quota_exposures(units, root=root, quota=quota)
    update_bins = partition_update_bins(exposures, quota=quota, updates=updates)
    packs = next_fit_packs(update_bins, max_length=max_length)
    return TrainingPlan(exposures, update_bins, packs, _ledger(exposures))


def build_codecontests_training_plan(
    units: tuple[TokenizedUnit, ...],
    *,
    root: int,
    quota: int = CODECONTESTS_RESPONSE_TOKEN_QUOTA,
    updates: int = CODECONTESTS_UPDATES,
    max_length: int = 2_048,
) -> TrainingPlan:
    """Build the frozen CodeContests quota/bins with single-unit forwards."""

    if any(item.sampling_id is None for item in units):
        raise ValueError("CodeContests training requires triplet sampling identities")
    if any(len(item.token_ids) > max_length for item in units):
        raise ValueError("CodeContests materialized unit exceeds maximum length")
    exposures = _quota_exposures(
        units,
        root=root,
        quota=quota,
        cycle_order=codecontests_cycle_order,
    )
    update_bins = partition_update_bins(
        exposures,
        quota=quota,
        updates=updates,
        round_half_up_targets=True,
    )
    packs = single_unit_packs(update_bins, max_length=max_length)
    return TrainingPlan(exposures, update_bins, packs, _ledger(exposures))


def _validate_codecontests_v2_units(
    units: tuple[TokenizedUnit, ...], *, max_length: int
) -> None:
    if any(item.sampling_id is None for item in units):
        raise ValueError("CodeContests V2 training requires sampling identities")
    if any(len(item.token_ids) > max_length for item in units):
        raise ValueError("CodeContests V2 materialized unit exceeds maximum length")


def build_codecontests_v2_training_plan(
    units: tuple[TokenizedUnit, ...],
    *,
    stage: str,
    root: int,
    quota: int = CODECONTESTS_RESPONSE_TOKEN_QUOTA,
    updates: int = CODECONTESTS_UPDATES,
    max_length: int = 2_048,
) -> TrainingPlan:
    """Build a two-arm V2 stage plan with stage-bound arm-invariant ordering."""

    if stage not in ("positive-control", "natural-certificate"):
        raise ValueError("ordinary V2 plans are limited to two-arm stages")
    _validate_codecontests_v2_units(units, max_length=max_length)
    exposures, _ = _quota_exposures_from_cursor(
        units,
        quota=quota,
        cycle_order=lambda cycle: codecontests_v2_cycle_order(
            units, stage=stage, root=root, cycle=cycle
        ),
    )
    update_bins = partition_update_bins(
        exposures,
        quota=quota,
        updates=updates,
        round_half_up_targets=True,
    )
    packs = single_unit_packs(update_bins, max_length=max_length)
    return TrainingPlan(exposures, update_bins, packs, _ledger(exposures))


def build_codecontests_v2_policy_plans(
    background: tuple[TokenizedUnit, ...],
    incorrect: tuple[TokenizedUnit, ...],
    repair: tuple[TokenizedUnit, ...],
    *,
    root: int,
    max_length: int = 2_048,
    background_quota: int = CODECONTESTS_V2_POLICY_BACKGROUND_QUOTA,
    treatment_quota: int = CODECONTESTS_V2_POLICY_TREATMENT_QUOTA,
    updates: int = CODECONTESTS_UPDATES,
) -> dict[str, TrainingPlan]:
    """Build the three policy arms around one exact root-shared B80 core."""

    for units in (background, incorrect, repair):
        _validate_codecontests_v2_units(units, max_length=max_length)
    identities = tuple(str(item.sampling_id) for item in background)
    if set(identities) != {str(item.sampling_id) for item in incorrect} or set(
        identities
    ) != {str(item.sampling_id) for item in repair}:
        raise ValueError("policy components must use the same frozen support identities")

    background_order = lambda cycle: codecontests_v2_cycle_order(
        background, stage="policy-filter", root=root, cycle=cycle
    )
    b80, b20_cursor = _quota_exposures_from_cursor(
        background,
        quota=background_quota,
        cycle_order=background_order,
        component="B80",
    )
    b20, _ = _quota_exposures_from_cursor(
        background,
        quota=treatment_quota,
        cycle_order=background_order,
        start_cycle=b20_cursor[0],
        start_position=b20_cursor[1],
        component="T20",
    )
    i20, _ = _quota_exposures_from_cursor(
        incorrect,
        quota=treatment_quota,
        cycle_order=lambda cycle: codecontests_v2_cycle_order(
            incorrect, stage="policy-filter", root=root, cycle=cycle
        ),
        component="T20",
    )
    r20, _ = _quota_exposures_from_cursor(
        repair,
        quota=treatment_quota,
        cycle_order=lambda cycle: codecontests_v2_cycle_order(
            repair, stage="policy-filter", root=root, cycle=cycle
        ),
        component="T20",
    )
    b80_bins = partition_update_bins(
        b80,
        quota=background_quota,
        updates=updates,
        round_half_up_targets=True,
    )

    plans: dict[str, TrainingPlan] = {}
    for arm, treatment in (
        ("POL_KEEP", i20),
        ("POL_FILTER", b20),
        ("POL_REPAIR", r20),
    ):
        treatment_bins = partition_update_bins(
            treatment,
            quota=treatment_quota,
            updates=updates,
            round_half_up_targets=True,
        )
        update_bins = tuple(
            background_bin + treatment_bin
            for background_bin, treatment_bin in zip(b80_bins, treatment_bins)
        )
        exposures = tuple(item for update in update_bins for item in update)
        packs = single_unit_packs(update_bins, max_length=max_length)
        plans[arm] = TrainingPlan(
            exposures, update_bins, packs, _ledger(exposures)
        )
    return plans


def summarize_codecontests_v2_plan(
    plan: TrainingPlan, *, support_sha256: str
) -> dict[str, object]:
    """Return the compact content identity consumed by V2 formal run cards."""

    if re.fullmatch(r"[0-9a-f]{64}", support_sha256) is None:
        raise ValueError("support identity must be a lowercase SHA-256")
    digest = hashlib.sha256()
    b80 = hashlib.sha256()
    component_tokens: dict[str, int] = {}
    exposure_count = 0
    for update_index, update in enumerate(plan.update_bins):
        if not update:
            raise ValueError("V2 plan contains an empty update")
        for exposure in update:
            exposure_count += 1
            component = exposure.component or "FULL"
            component_tokens[component] = (
                component_tokens.get(component, 0) + exposure.retained_response_tokens
            )
            record = (
                f"{update_index}|{exposure.unit_id}|{exposure.cycle}|{component}|"
                f"{exposure.retained_response_tokens}|{int(exposure.complete)}|"
                f"{hashlib.sha256(bytes(str(exposure.token_ids), 'ascii')).hexdigest()}\n"
            ).encode("ascii")
            digest.update(record)
            if component == "B80":
                b80.update(record)
    quota = sum(item.retained_response_tokens for item in plan.exposures)
    closure: dict[str, object] = {
        "complete": quota == CODECONTESTS_RESPONSE_TOKEN_QUOTA
        and len(plan.update_bins) == CODECONTESTS_UPDATES,
        "response_token_quota": quota,
        "updates": len(plan.update_bins),
        "exposure_count": exposure_count,
        "support_sha256": support_sha256,
        "plan_sha256": digest.hexdigest(),
        "component_response_tokens": component_tokens,
    }
    if "B80" in component_tokens:
        closure["b80_sha256"] = b80.hexdigest()
    return closure
