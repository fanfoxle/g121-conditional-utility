"""Byte-exact public commitments from ADRs 0072 and 0083."""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID


V2_PROTOCOL_ID = "oasst1-natural-filter-v2"
V3_PROTOCOL_ID = "oasst1-natural-filter-v3"
V4_PROTOCOL_ID = "oasst1-natural-filter-v4"
V5_EXECUTION_PROTOCOL_ID = "oasst1-natural-filter-v5"
V5_SOURCE_DESIGN_PROTOCOL_ID = V4_PROTOCOL_ID
V6_PROTOCOL_ID = "oasst1-natural-filter-v6"
V6_SOURCE_DESIGN_PROTOCOL_ID = V4_PROTOCOL_ID
V7_EXECUTION_PROTOCOL_ID = "oasst1-natural-filter-v7"
V7_SCIENTIFIC_PROTOCOL_ID = V6_PROTOCOL_ID
V7_SOURCE_DESIGN_PROTOCOL_ID = V4_PROTOCOL_ID
V7_TERMINAL_STATES = frozenset(
    {
        "STOPPED_V7_CARRY_FORWARD",
        "STOPPED_V7_PREFLIGHT",
        "STOPPED_V7_RESOURCE",
        "STOPPED_V7_O10_GATE",
        "STOPPED_V7_FORMAL_INVALID",
        "COMPLETED_V7_P8_RETURN",
    }
)
V7A_EXECUTION_PROTOCOL_ID = "oasst1-natural-filter-v7a"
V7A_SCIENTIFIC_PROTOCOL_ID = V6_PROTOCOL_ID
V7A_SOURCE_DESIGN_PROTOCOL_ID = V4_PROTOCOL_ID
V7A_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
V7A_TERMINAL_STATES = frozenset(
    {
        "STOPPED_V7A_CARRY_FORWARD",
        "STOPPED_V7A_PREFLIGHT",
        "STOPPED_V7A_RESOURCE",
        "STOPPED_V7A_O10_GATE",
        "STOPPED_V7A_FORMAL_INVALID",
        "COMPLETED_V7A_P8_RETURN",
    }
)
V8_PROTOCOL_ID = "oasst1-natural-filter-v8"
V8_DESIGN_ID = "D2_CONCURRENT_OFA_N10_J1_95"
V8_TERMINAL_STATES = frozenset(
    {
        "STOPPED_V8_INPUT_CLOSURE",
        "STOPPED_V8_PREFLIGHT",
        "STOPPED_V8_RESOURCE",
        "STOPPED_V8_FORMAL_INVALID",
        "COMPLETED_V8_P8_RETURN",
    }
)
# Historical modules continue to resolve this alias to V2; V3/V4 entry points
# use explicit constants so terminal earlier artifacts remain reproducible.
PROTOCOL_ID = V2_PROTOCOL_ID
RANDOMIZATION_NAMESPACE = "oasst1-natural-filter-v1"
# Backward-compatible name used by every frozen v1 HMAC/seed derivation.
NAMESPACE = RANDOMIZATION_NAMESPACE

PARTITION_KEY = bytes.fromhex(
    "c6b98f56edfcc0ff3327383a2f888ea592b34582798e35bf175115b9b1532956"
)
BASELINE_KEY = bytes.fromhex(
    "0afa7406332cd44b57e380dbcb4ae83434f3e43c68ec5ce781785e6a70c0c61a"
)
SAMPLER_KEY = bytes.fromhex(
    "3c20ed6ce45472f582fe1a8c9e0762eec225a9499b044a8610d402e9773a34be"
)
TRAINING_ORDER_KEY = bytes.fromhex(
    "1532bf42d0b8bea2f2a6595bc438f2256e9ebc11cede3e81e6777bcaa715124c"
)
EVALUATION_ORDER_KEY = bytes.fromhex(
    "02fc52336b13fb597cf4bacc4290bc149f4c6225467170acc9944cc696ef049c"
)
MATCHING_TIE_KEY = bytes.fromhex(
    "1f0a12bdc13916c36ee67653fc17bfaed704007ccc14721e77cfdd559018116d"
)
WITHIN_PAIR_KEY = bytes.fromhex(
    "8c21bb3d56ad82ddd89df8d57adffc472472a48930c43749be1650a2072045d3"
)

FORMAL_ROOTS = (
    203705620492079560,
    1046600215625497828,
    954796266612112323,
    267271149525524599,
    168235003955983788,
)
RECIPE_ROOT = 589628436199292070
O_ORDER = (
    954796266612112323,
    1046600215625497828,
    168235003955983788,
    267271149525524599,
    203705620492079560,
)
PAIRED_TRAINING_ORDER = (
    168235003955983788,
    267271149525524599,
    954796266612112323,
    203705620492079560,
    1046600215625497828,
)
PAIRED_ENDPOINT_ORDER = (
    954796266612112323,
    203705620492079560,
    168235003955983788,
    267271149525524599,
    1046600215625497828,
)
V4_CORE_ARMS = ("O", "F", "Mset-A", "Mset-B")
V4_O_ORDER = tuple((root, "O") for root in O_ORDER)
V4_TREATMENT_TRAINING_ORDER = (
    (168235003955983788, ("F", "Mset-A", "Mset-B")),
    (267271149525524599, ("Mset-A", "Mset-B", "F")),
    (954796266612112323, ("Mset-B", "F", "Mset-A")),
    (203705620492079560, ("F", "Mset-B", "Mset-A")),
    (1046600215625497828, ("Mset-A", "F", "Mset-B")),
)
V4_TREATMENT_GENERATION_ORDER = tuple(
    (root, tuple(reversed(arms)))
    for root, arms in V4_TREATMENT_TRAINING_ORDER
)
V4_FORMAL_TRAINING_ORDER = V4_O_ORDER + tuple(
    (root, arm)
    for root, arms in V4_TREATMENT_TRAINING_ORDER
    for arm in arms
)

# V5 changes only execution concurrency. These fixed waves preserve the v4
# scientific order and remain the same when the admitted generation degree is
# one; jobs are then executed serially within each listed wave.
V5_C_TRAIN = 1
V5_O_GENERATION_WAVES = (
    V4_O_ORDER[0:2],
    V4_O_ORDER[2:4],
    V4_O_ORDER[4:5],
)
V5_TREATMENT_GENERATION_WAVES = tuple(
    (
        ((root, generation_order[0]), (root, generation_order[1])),
        ((root, generation_order[2]),),
    )
    for root, generation_order in V4_TREATMENT_GENERATION_ORDER
)

# V6 carries the first five scientific roots and adds exactly labels 06--10.
# The v1 randomization namespace remains load-bearing for every derived seed.
V6_EXTENSION_ROOTS = tuple(
    int(
        hashlib.sha256(
            f"{RANDOMIZATION_NAMESPACE}|formal-root|{label}".encode("utf-8")
        ).hexdigest()[:15],
        16,
    )
    for label in ("06", "07", "08", "09", "10")
)
V6_ANALYSIS_ORDER = PAIRED_TRAINING_ORDER + V6_EXTENSION_ROOTS
V6_NEW_O_ORDER = tuple((root, "O") for root in V6_EXTENSION_ROOTS)
V6_TREATMENT_TRAINING_ORDER = V4_TREATMENT_TRAINING_ORDER + (
    (V6_EXTENSION_ROOTS[0], ("Mset-B", "Mset-A", "F")),
    (V6_EXTENSION_ROOTS[1], ("F", "Mset-A", "Mset-B")),
    (V6_EXTENSION_ROOTS[2], ("Mset-A", "Mset-B", "F")),
    (V6_EXTENSION_ROOTS[3], ("Mset-B", "F", "Mset-A")),
    (V6_EXTENSION_ROOTS[4], ("F", "Mset-B", "Mset-A")),
)
V6_TREATMENT_GENERATION_ORDER = tuple(
    (root, tuple(reversed(arms)))
    for root, arms in V6_TREATMENT_TRAINING_ORDER
)
V6_EXECUTION_TRAINING_ORDER = V6_NEW_O_ORDER + tuple(
    (root, arm)
    for root, arms in V6_TREATMENT_TRAINING_ORDER
    for arm in arms
)
V6_EXTENSION_RUN_CARD_ORDER = (
    V6_EXECUTION_TRAINING_ORDER[:5] + V6_EXECUTION_TRAINING_ORDER[20:]
)
V6_O_GENERATION_WAVES = (
    V6_NEW_O_ORDER[:2],
    V6_NEW_O_ORDER[2:4],
    V6_NEW_O_ORDER[4:],
)
V6_TREATMENT_GENERATION_WAVES = tuple(
    (
        ((root, generation_order[0]), (root, generation_order[1])),
        ((root, generation_order[2]),),
    )
    for root, generation_order in V6_TREATMENT_GENERATION_ORDER
)

# V8 is a fresh scientific experiment. Its roots use the preserved v1
# derivation, and the fixed O/F/A order balances every pairwise precedence 5:5.
V8_ROOT_LABELS = tuple(f"formal-root|{value:02d}" for value in range(11, 21))
V8_ROOTS = tuple(
    int(
        hashlib.sha256(
            f"{RANDOMIZATION_NAMESPACE}|formal-root|{value:02d}".encode("utf-8")
        ).hexdigest()[:15],
        16,
    )
    for value in range(11, 21)
)
V8_ROOT_ARM_ORDERS = tuple(
    zip(
        V8_ROOTS,
        (
            ("O", "F", "A"),
            ("A", "F", "O"),
            ("O", "A", "F"),
            ("F", "A", "O"),
            ("F", "O", "A"),
            ("A", "O", "F"),
            ("O", "F", "A"),
            ("A", "F", "O"),
            ("F", "O", "A"),
            ("A", "O", "F"),
        ),
    )
)
V8_FORMAL_JOB_ORDER = tuple(
    (root, arm) for root, arms in V8_ROOT_ARM_ORDERS for arm in arms
)
V8_CANONICAL_MERGE_ORDER = tuple(
    (root, arm) for root in V8_ROOTS for arm in ("O", "F", "A")
)
V8_C_TRAIN = 1

# CodeContests V1 is a separate scientific protocol that reuses this stable
# package without inheriting the OASST1 randomization namespace or recipe.
CODECONTESTS_PROTOCOL_ID = "codecontests-executable-substitution-v1"
CODECONTESTS_DESIGN_ID = "CC_EXEC_IRN_N8_J1_95"
CODECONTESTS_FORMAL_ROOTS = (
    503077502716645340,
    1123586167472521567,
    453702182598054146,
    886316059423133647,
    250512255387879391,
    712608184351010942,
    1130180233282329491,
    883819863077066269,
)
CODECONTESTS_ROOT_ARM_ORDERS = tuple(
    zip(
        CODECONTESTS_FORMAL_ROOTS,
        (
            ("I", "R", "N"),
            ("R", "N", "I"),
            ("N", "I", "R"),
            ("I", "R", "N"),
            ("R", "N", "I"),
            ("N", "I", "R"),
            ("I", "R", "N"),
            ("R", "N", "I"),
        ),
    )
)
CODECONTESTS_FORMAL_JOB_ORDER = tuple(
    (root, arm) for root, arms in CODECONTESTS_ROOT_ARM_ORDERS for arm in arms
)
CODECONTESTS_RESPONSE_TOKEN_QUOTA = 583_975
CODECONTESTS_UPDATES = 100
CODECONTESTS_PEAK_LEARNING_RATE = 2e-4
CODECONTESTS_LAUNCH_HEADROOM_FRACTION = 0.03
CODECONTESTS_ROLE_NAMES = (
    "lora-initialization",
    "sampler",
    "update-binning",
    "dropout",
    "dataloader",
    "framework",
)
CODECONTESTS_TERMINAL_STATES = frozenset(
    {
        "STOPPED_CODECONTESTS_V1_INPUT_INVALID",
        "STOPPED_CODECONTESTS_V1_FORMAL_INVALID",
        "STOPPED_CODECONTESTS_V1_ENDPOINT_INVALID",
        "STOPPED_CODECONTESTS_V1_RESOURCE_LIMIT",
        "COMPLETED_CODECONTESTS_V1_REPORT",
    }
)
CODECONTESTS_PAUSE_STATES = frozenset(
    {"AWAITING_USER_TRAINING_AUTHORIZATION", "WAITING_FOR_GPU_CAPACITY"}
)

# CodeContests V2 is an explicit successor protocol in the same stable tree.
# V1 constants above remain immutable historical inputs and are intentionally
# not aliases for the V2 staged design.
CODECONTESTS_V2_PROTOCOL_ID = "codecontests-certificate-calibrated-filtering-v2"
CODECONTESTS_V2_DESIGN_ID = "CCV2_F1_F2_F5_N12"
CODECONTESTS_V2_STAGE_ORDER = (
    "positive-control",
    "natural-certificate",
    "policy-filter",
)


def _derive_codecontests_v2_root(stage: str, index: int) -> int:
    message = f"{CODECONTESTS_V2_PROTOCOL_ID}|{stage}|formal-root|{index}"
    return int(hashlib.sha256(message.encode("utf-8")).hexdigest()[:15], 16)


CODECONTESTS_V2_FORMAL_ROOTS = {
    stage: tuple(_derive_codecontests_v2_root(stage, index) for index in range(1, 13))
    for stage in CODECONTESTS_V2_STAGE_ORDER
}
CODECONTESTS_V2_STAGE_ARMS = {
    "positive-control": ("PC_CLEAN", "PC_MUTANT"),
    "natural-certificate": ("NAT_R", "NAT_I"),
    "policy-filter": ("POL_KEEP", "POL_FILTER", "POL_REPAIR"),
}


def _v2_arm_order(stage: str, index: int) -> tuple[str, ...]:
    arms = CODECONTESTS_V2_STAGE_ARMS[stage]
    if len(arms) == 2:
        return arms if index % 2 == 0 else tuple(reversed(arms))
    shift = index % 3
    return arms[shift:] + arms[:shift]


CODECONTESTS_V2_ROOT_ARM_ORDERS = {
    stage: tuple(
        (root, _v2_arm_order(stage, index))
        for index, root in enumerate(CODECONTESTS_V2_FORMAL_ROOTS[stage])
    )
    for stage in CODECONTESTS_V2_STAGE_ORDER
}
CODECONTESTS_V2_FORMAL_JOB_ORDERS = {
    stage: tuple(
        (root, arm)
        for root, arms in CODECONTESTS_V2_ROOT_ARM_ORDERS[stage]
        for arm in arms
    )
    for stage in CODECONTESTS_V2_STAGE_ORDER
}
CODECONTESTS_V2_POLICY_BACKGROUND_QUOTA = 467_180
CODECONTESTS_V2_POLICY_TREATMENT_QUOTA = 116_795
CODECONTESTS_V2_TERMINAL_STATES = frozenset(
    {
        "STOPPED_CODECONTESTS_V2_INPUT_INVALID",
        "STOPPED_CODECONTESTS_V2_FORMAL_INVALID",
        "STOPPED_CODECONTESTS_V2_ENDPOINT_INVALID",
        "STOPPED_CODECONTESTS_V2_RESOURCE_LIMIT",
        "COMPLETED_CODECONTESTS_V2_CENSUS_INSUFFICIENT",
        "COMPLETED_CODECONTESTS_V2_POSITIVE_CONTROL_UNAVAILABLE",
        "COMPLETED_CODECONTESTS_V2_ASSAY_NOT_SENSITIVE",
        "COMPLETED_CODECONTESTS_V2_NATURAL_NOT_SUPPORTED",
        "COMPLETED_CODECONTESTS_V2_POLICY_SUPPORT_INSUFFICIENT",
        "COMPLETED_CODECONTESTS_V2_REPORT",
    }
)
CODECONTESTS_V2_PAUSE_STATES = frozenset(
    {"AWAITING_USER_CODECONTESTS_V2_TRAINING_AUTHORIZATION", "WAITING_FOR_GPU_CAPACITY"}
)

# CodeContests V3 is a CPU-only feasibility successor in the same stable tree.
CODECONTESTS_V3_PROTOCOL_ID = "codecontests-v3-cpu-recovery-crosswalk"
CODECONTESTS_V3_DESIGN_ID = "CCV3_S1_S4_CPU_ONLY"
CODECONTESTS_V3_MAX_CANDIDATES = 16
CODECONTESTS_V3_MAX_SERIALIZED_TOKENS = 2_048
CODECONTESTS_V3_MIN_SUPPORT_PAIRS = 100
CODECONTESTS_V3_MIN_KISH_ESS = 80
CODECONTESTS_V3_CCPLUS_REVISION = "96c850540fade31d384a25766461e0da6b08f5fc"
CODECONTESTS_V3_CCPLUS_CONFIG = "1x"
CODECONTESTS_V3_CCPLUS_SPLIT = "train"
CODECONTESTS_V3_CCPLUS_SHARD_COUNT = 20
CODECONTESTS_V3_CCPLUS_TOTAL_BYTES = 46_015_020_393
CODECONTESTS_V3_CCPLUS_TRANSPORT_REPO = (
    "bytedance-community/Code-Contests-Plus"
)
CODECONTESTS_V3_CCPLUS_TRANSPORT_REVISION = (
    "0197deb5e35b6edb1cb0b9dc2c6f55f1cd021f52"
)
CODECONTESTS_V3_CCPLUS_SHARD_SIZES = (
    2_524_010_025,
    2_840_376_912,
    2_334_958_829,
    2_404_368_756,
    2_066_890_643,
    1_978_007_413,
    2_395_672_980,
    2_126_571_920,
    1_992_312_611,
    2_413_217_466,
    2_641_409_775,
    2_458_582_842,
    2_645_651_608,
    2_414_480_469,
    2_379_209_285,
    1_791_806_055,
    2_049_020_829,
    2_118_542_400,
    2_542_271_501,
    1_897_658_074,
)
CODECONTESTS_V3_CCPLUS_SHARD_SHA256 = (
    "4501a6ab694177803a90d03089f8082f8664dc62905a7961842335a514d5db79",
    "f99147aa9c010885556abb649716abf578ebf3cd3f64f2c05312985bc63e4789",
    "1ad3ee4e6cf0d32d542e1afb743509e7dfe0e413dbd09af14ac10e39b1331ebe",
    "a4e969a7640942450f154a80af1dcbf38ace5b8ffcfb6d431b6a3e139504fb63",
    "7242a767b23d2c3be40e44fab522c9e14edce0509ed6de2b49e0838ff74e02e3",
    "72837ac2990b3131c878c1bbd073d68ad8a713b701f0aef19dff2f1799b4aed6",
    "e7b0769db9a3c8d6fe7b47189bc04f414615f6713fd7ff1208256c90863ea354",
    "ac6f0eb692f1d8370e4d42011ad1d7886743597c99f82b7517441cee13e2b953",
    "2b9cdcc3d924c5ab61603b1a27424f5c90e96dafd6723468d1265365427a5daa",
    "4460a54cecc7af2a06454a15fa227bf0cf6eb5bfdcbef650dfa9036c3bd4a544",
    "e43bff588fda59396beb6bade0df98a3c9a5afd024c5c4880c2add872877e17b",
    "b443aade344a286668033fc43c617dcc738ff9040d7ddd308caf1e918acb5c44",
    "cc9a64d113993a1ba8d85ba81a0114a61540d2d3ae856c7c99a15bb3483fddae",
    "72cca6892f41d505bd84c7911d3df9a6cee2aad2fbb9bef19754494c15e68b29",
    "8a580d6bda8c01bcaade2df83de0abe17b13f8b49f930324324c0b0e05bb5e71",
    "d3ae334036323020b502b6d77bcf820359d2ab9fefadc17bfde24fc77c37f7ce",
    "bc7f3188e5e41763412a9e03c577eb771531cda939dcd77a0f7662b467ebf84e",
    "811bb7c1b894112158eb3fffd29f620d3b2d78362e1daa3189c0c142d85b56fb",
    "0ddc660b13e656e8d872dac246cf6ae8abe7461b6d9e8c6df8251cb42961167b",
    "e07b3a9505beac36b5acd4b45258c753cd8c65a8355e5223d3797880f896f389",
)
CODECONTESTS_V3_CCPLUS_PROJECTED_CAP_BYTES = 12 * 2**30
CODECONTESTS_V3_CCPLUS_ACCESS_LIMIT_SECONDS = 6 * 60 * 60
CODECONTESTS_V3_FORMAL_WALL_LIMIT_SECONDS = 48 * 60 * 60
CODECONTESTS_V3_UPSTREAM_ID_GRAMMAR = None
CODECONTESTS_V3_TERMINAL_STATES = frozenset(
    {
        "COMPLETED_CODECONTESTS_V3_S1_SUPPORT_SUFFICIENT",
        "COMPLETED_CODECONTESTS_V3_S4_SUPPORT_SUFFICIENT",
        "COMPLETED_CODECONTESTS_V3_S4_CROSSWALK_INSUFFICIENT",
        "COMPLETED_CODECONTESTS_V3_SUPPORT_INSUFFICIENT",
        "STOPPED_CODECONTESTS_V3_INPUT_INVALID",
        "STOPPED_CODECONTESTS_V3_ORIGINAL_REPLAY_DRIFT",
        "STOPPED_CODECONTESTS_V3_RUNNER_INVALID",
        "STOPPED_CODECONTESTS_V3_CCPLUS_ACCESS_INFEASIBLE",
        "STOPPED_CODECONTESTS_V3_RESOURCE_LIMIT",
    }
)

# V3a repairs only the V3 execution device.  Its scientific candidate ordering
# deliberately remains in the V3 namespace.
CODECONTESTS_V3A_PROTOCOL_ID = "codecontests-v3a-runner-safe-recovery-crosswalk"
CODECONTESTS_V3A_DESIGN_ID = "CCV3A_S1_S4_CPU_ONLY"
CODECONTESTS_V3A_CANDIDATE_PRIORITY_NAMESPACE = CODECONTESTS_V3_PROTOCOL_ID
CODECONTESTS_V3A_SHARED_UID_NPROC_LIMIT = 32
CODECONTESTS_V3A_PROCESS_RESERVE = 8
CODECONTESTS_V3A_CANDIDATE_RUNGS = (8, 4)
CODECONTESTS_V3A_TERMINAL_STATES = frozenset(
    {
        "COMPLETED_CODECONTESTS_V3A_S1_SUPPORT_SUFFICIENT",
        "COMPLETED_CODECONTESTS_V3A_S4_CROSSWALK_INSUFFICIENT",
        "COMPLETED_CODECONTESTS_V3A_S4_SUPPORT_SUFFICIENT",
        "COMPLETED_CODECONTESTS_V3A_SUPPORT_INSUFFICIENT",
        "STOPPED_CODECONTESTS_V3A_INPUT_INVALID",
        "STOPPED_CODECONTESTS_V3A_RUNNER_QUALIFICATION_FAILED",
        "STOPPED_CODECONTESTS_V3A_ORIGINAL_REPLAY_DRIFT",
        "STOPPED_CODECONTESTS_V3A_RUNNER_INVALID",
        "STOPPED_CODECONTESTS_V3A_CCPLUS_ACCESS_INFEASIBLE",
        "STOPPED_CODECONTESTS_V3A_RESOURCE_LIMIT",
    }
)

# V3b rebaselines original R and bounded recovery under one current device.
# Candidate ordering remains byte-identical to V3/V3a.
CODECONTESTS_V3B_PROTOCOL_ID = "codecontests-v3b-same-device-rebaseline"
CODECONTESTS_V3B_DESIGN_ID = "CCV3B_B0_S1_S4_CPU_ONLY"
CODECONTESTS_V3B_CANDIDATE_PRIORITY_NAMESPACE = CODECONTESTS_V3_PROTOCOL_ID
CODECONTESTS_V3B_TERMINAL_STATES = frozenset(
    {
        "COMPLETED_CODECONTESTS_V3B_S1_SUPPORT_SUFFICIENT",
        "COMPLETED_CODECONTESTS_V3B_S4_CROSSWALK_INSUFFICIENT",
        "COMPLETED_CODECONTESTS_V3B_S4_SUPPORT_SUFFICIENT",
        "COMPLETED_CODECONTESTS_V3B_SUPPORT_INSUFFICIENT",
        "STOPPED_CODECONTESTS_V3B_INPUT_INVALID",
        "STOPPED_CODECONTESTS_V3B_RUNNER_QUALIFICATION_FAILED",
        "STOPPED_CODECONTESTS_V3B_RUNNER_INVALID",
        "STOPPED_CODECONTESTS_V3B_CCPLUS_ACCESS_INFEASIBLE",
        "STOPPED_CODECONTESTS_V3B_RESOURCE_LIMIT",
    }
)

# S12d closes the finite CPU qualification contract without changing the
# preserved S12/APCF science.  The tuple order is itself part of the protocol:
# the complete serial reference always precedes the complete C4 pass.
CODECONTESTS_S12D_PROTOCOL_ID = "codecontests-s12d-uqcc-v1"
CODECONTESTS_S12D_DESIGN_ID = "CC_S12D_UQCC"
CODECONTESTS_S12D_EXECUTION_MATRIX = (1, 4)
CODECONTESTS_S12D_TERMINAL_STATES = frozenset(
    {
        "COMPLETED_S12D_PREFORMAL_QUALIFIED",
        "COMPLETED_S12D_NATURAL_SUPPORT_INSUFFICIENT",
        "COMPLETED_S12D_PC_UNAVAILABLE",
        "STOPPED_S12D_INPUT_INVALID",
        "STOPPED_S12D_CONTRACT_INVALID",
        "STOPPED_S12D_INFRASTRUCTURE_INVALID",
        "STOPPED_S12D_ENDPOINT_INVALID",
        "STOPPED_S12D_RESOURCE_LIMIT",
    }
)


def v8_generation_waves(c_gen: int) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Group adjacent formal jobs without changing V8's logical order."""

    if c_gen not in (1, 2):
        raise ValueError("V8 generation concurrency must be one or two")
    return tuple(
        V8_FORMAL_JOB_ORDER[index : index + c_gen]
        for index in range(0, len(V8_FORMAL_JOB_ORDER), c_gen)
    )


def v8_scoring_batches(c_val: int) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Group consecutive V8 jobs for scoring; canonical merge stays separate."""

    if c_val not in (1, 4):
        raise ValueError("V8 scoring concurrency must be one or four")
    return tuple(
        V8_FORMAL_JOB_ORDER[index : index + c_val]
        for index in range(0, len(V8_FORMAL_JOB_ORDER), c_val)
    )
ROLE_NAMES = (
    "lora-initialization",
    "sampler",
    "packing",
    "dropout",
    "dataloader",
    "framework",
)
DIAGNOSTIC_THRESHOLD = 922337203685477580
RECIPE_PEAK_LEARNING_RATES = (2e-4, 1e-4, 5e-5)
FORMAL_RESPONSE_TOKEN_QUOTA = 4_194_304
FORMAL_UPDATES = 1_024
RECIPE_RESPONSE_TOKEN_QUOTA = 524_288
RECIPE_UPDATES = 128


@dataclass(frozen=True)
class GateResult:
    """Shared machine-readable gate envelope without cross-domain imports."""

    gate: str
    passed: bool
    reason: str
    observed: Mapping[str, int | float | str | bool]


@dataclass(frozen=True)
class CodeContestsProtocol:
    """Closed public protocol seam for the CodeContests V1 implementation."""

    protocol_id: str
    design_id: str
    formal_roots: tuple[int, ...]
    root_arm_orders: tuple[tuple[int, tuple[str, ...]], ...]
    formal_job_order: tuple[tuple[int, str], ...]
    response_token_quota: int
    updates: int
    launch_headroom_fraction: float
    terminal_states: frozenset[str]


@dataclass(frozen=True)
class CodeContestsV2Protocol:
    """Closed public protocol seam for staged CodeContests V2 work."""

    protocol_id: str
    design_id: str
    stage_order: tuple[str, ...]
    formal_roots: Mapping[str, tuple[int, ...]]
    root_arm_orders: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]]
    formal_job_orders: Mapping[str, tuple[tuple[int, str], ...]]
    response_token_quota: int
    updates: int
    launch_headroom_fraction: float
    terminal_states: frozenset[str]


@dataclass(frozen=True)
class CodeContestsV3Protocol:
    """Closed public protocol seam for the CPU-only V3 successor."""

    protocol_id: str
    design_id: str
    max_candidates: int
    ccplus_revision: str
    terminal_states: frozenset[str]


@dataclass(frozen=True)
class CodeContestsS12DProtocol:
    """Closed public contract for the S12d CPU qualification."""

    protocol_id: str
    design_id: str
    execution_matrix: tuple[int, int]
    terminal_states: frozenset[str]


def canonical_uuid(value: str) -> str:
    """Validate and return ADR 0072's lowercase hyphenated UUID form."""

    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("identifier is not a UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("identifier is not lowercase canonical UUID text")
    return canonical


def unit_id(root_id: str, response_id: str) -> str:
    return f"{canonical_uuid(root_id)}/{canonical_uuid(response_id)}"


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def partition_for_root(root_id: str) -> str:
    root = canonical_uuid(root_id)
    digest = _hmac(PARTITION_KEY, f"{NAMESPACE}|partition|{root}")
    value = int.from_bytes(digest[:8], "big", signed=False)
    return "diagnostic" if value < DIAGNOSTIC_THRESHOLD else "train"


def baseline_priority(root_id: str, response_id: str) -> bytes:
    stable_unit = unit_id(root_id, response_id)
    return _hmac(BASELINE_KEY, f"{NAMESPACE}|baseline|{stable_unit}")


def derive_role_seed(root: int, role_name: str) -> int:
    if root < 0 or isinstance(root, bool):
        raise ValueError("root must be a nonnegative integer")
    if role_name not in ROLE_NAMES:
        raise ValueError("role name is outside the frozen registry")
    message = f"{NAMESPACE}|root|{root}|role|{role_name}"
    return int(hashlib.sha256(message.encode("utf-8")).hexdigest()[:15], 16)


def derive_codecontests_role_seed(root: int, role_name: str) -> int:
    """Derive one arm-invariant CodeContests role seed from the frozen namespace."""

    if root < 0 or isinstance(root, bool):
        raise ValueError("root must be a nonnegative integer")
    if role_name not in CODECONTESTS_ROLE_NAMES:
        raise ValueError("role name is outside the frozen CodeContests registry")
    message = f"{CODECONTESTS_PROTOCOL_ID}|root|{root}|role|{role_name}"
    return int(hashlib.sha256(message.encode("utf-8")).hexdigest()[:15], 16)


def derive_codecontests_v2_role_seed(stage: str, root: int, role_name: str) -> int:
    """Derive an arm-invariant V2 role seed within one frozen stage/root."""

    if stage not in CODECONTESTS_V2_STAGE_ORDER:
        raise ValueError("stage is outside the frozen CodeContests V2 sequence")
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("root must be a nonnegative integer")
    if role_name not in CODECONTESTS_ROLE_NAMES:
        raise ValueError("role name is outside the frozen CodeContests registry")
    message = f"{CODECONTESTS_V2_PROTOCOL_ID}|{stage}|{root}|role|{role_name}"
    return int(hashlib.sha256(message.encode("utf-8")).hexdigest()[:15], 16)


def codecontests_learning_rate_at(update_index: int) -> float:
    """Return the frozen 100-update CodeContests warmup/cosine schedule."""

    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or not 0 <= update_index < CODECONTESTS_UPDATES
    ):
        raise ValueError("update index must be in [0, 99]")
    peak = CODECONTESTS_PEAK_LEARNING_RATE
    if update_index < 4:
        return peak * (update_index + 1) / 4
    progress = (update_index - 4) / (99 - 4)
    return peak * (0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0)


def resolve_codecontests_protocol(protocol_id: str) -> CodeContestsProtocol:
    """Resolve the only frozen CodeContests protocol; reject implicit variants."""

    if protocol_id != CODECONTESTS_PROTOCOL_ID:
        raise ValueError(f"unknown CodeContests protocol: {protocol_id}")
    return CodeContestsProtocol(
        protocol_id=CODECONTESTS_PROTOCOL_ID,
        design_id=CODECONTESTS_DESIGN_ID,
        formal_roots=CODECONTESTS_FORMAL_ROOTS,
        root_arm_orders=CODECONTESTS_ROOT_ARM_ORDERS,
        formal_job_order=CODECONTESTS_FORMAL_JOB_ORDER,
        response_token_quota=CODECONTESTS_RESPONSE_TOKEN_QUOTA,
        updates=CODECONTESTS_UPDATES,
        launch_headroom_fraction=CODECONTESTS_LAUNCH_HEADROOM_FRACTION,
        terminal_states=CODECONTESTS_TERMINAL_STATES,
    )


def resolve_codecontests_v2_protocol(protocol_id: str) -> CodeContestsV2Protocol:
    """Resolve the frozen V2 protocol without changing V1 compatibility."""

    if protocol_id != CODECONTESTS_V2_PROTOCOL_ID:
        raise ValueError(f"unknown CodeContests V2 protocol: {protocol_id}")
    return CodeContestsV2Protocol(
        protocol_id=CODECONTESTS_V2_PROTOCOL_ID,
        design_id=CODECONTESTS_V2_DESIGN_ID,
        stage_order=CODECONTESTS_V2_STAGE_ORDER,
        formal_roots=CODECONTESTS_V2_FORMAL_ROOTS,
        root_arm_orders=CODECONTESTS_V2_ROOT_ARM_ORDERS,
        formal_job_orders=CODECONTESTS_V2_FORMAL_JOB_ORDERS,
        response_token_quota=CODECONTESTS_RESPONSE_TOKEN_QUOTA,
        updates=CODECONTESTS_UPDATES,
        launch_headroom_fraction=CODECONTESTS_LAUNCH_HEADROOM_FRACTION,
        terminal_states=CODECONTESTS_V2_TERMINAL_STATES,
    )


def derive_codecontests_v3_worker_counts(
    *, logical_cpu_count: int, available_memory_bytes: int
) -> tuple[int, int]:
    """Derive the frozen CPU problem and projected-I/O worker counts."""

    if (
        isinstance(logical_cpu_count, bool)
        or not isinstance(logical_cpu_count, int)
        or isinstance(available_memory_bytes, bool)
        or not isinstance(available_memory_bytes, int)
    ):
        raise ValueError("CodeContests V3 resource inputs must be integer byte facts")
    memory_workers = (available_memory_bytes - 8 * 2**30) // (5 * 2**28)
    problem_workers = min(64, logical_cpu_count - 2, memory_workers)
    if problem_workers < 4:
        raise ValueError("CodeContests V3 requires at least four problem workers")
    return problem_workers, min(8, problem_workers)


def codecontests_v3_not_evaluated() -> dict[str, object]:
    """Return the sole representation for an unreached V3 value."""

    return {"status": "not_evaluated", "value": None}


def codecontests_v3a_shared_uid_process_bound(c_problem: int) -> int:
    """Return V3a's conservative shared-UID process bound ``2*C + 8``."""

    if isinstance(c_problem, bool) or not isinstance(c_problem, int) or c_problem < 1:
        raise ValueError("V3a problem concurrency must be a positive integer")
    return 2 * c_problem + CODECONTESTS_V3A_PROCESS_RESERVE


def codecontests_v3a_candidate_rungs(
    *, shared_uid_nproc_limit: int = CODECONTESTS_V3A_SHARED_UID_NPROC_LIMIT
) -> tuple[int, ...]:
    """Return only the prespecified safe V3a rungs, in frozen order."""

    if (
        isinstance(shared_uid_nproc_limit, bool)
        or not isinstance(shared_uid_nproc_limit, int)
        or shared_uid_nproc_limit < 1
    ):
        raise ValueError("shared-UID NPROC limit must be a positive integer")
    return tuple(
        rung
        for rung in CODECONTESTS_V3A_CANDIDATE_RUNGS
        if codecontests_v3a_shared_uid_process_bound(rung) <= shared_uid_nproc_limit
    )


def validate_codecontests_v3a_worker_selection(
    *,
    c_problem: int,
    qualified_rungs: tuple[int, ...],
    shared_uid_nproc_limit: int = CODECONTESTS_V3A_SHARED_UID_NPROC_LIMIT,
) -> tuple[int, int]:
    """Validate a pre-freeze qualified V3a worker selection."""

    process_bound = codecontests_v3a_shared_uid_process_bound(c_problem)
    if process_bound > shared_uid_nproc_limit:
        raise ValueError(
            "V3a shared-UID process bound exceeds the frozen NPROC limit"
        )
    safe_rungs = codecontests_v3a_candidate_rungs(
        shared_uid_nproc_limit=shared_uid_nproc_limit
    )
    if c_problem not in safe_rungs or c_problem not in qualified_rungs:
        raise ValueError("V3a worker selection must be a qualified rung")
    return c_problem, min(8, c_problem)


def resolve_codecontests_v3_protocol(protocol_id: str) -> CodeContestsV3Protocol:
    """Resolve the frozen CodeContests V3 CPU-only protocol."""

    if protocol_id != CODECONTESTS_V3_PROTOCOL_ID:
        raise ValueError(f"unknown CodeContests V3 protocol: {protocol_id}")
    return CodeContestsV3Protocol(
        protocol_id=CODECONTESTS_V3_PROTOCOL_ID,
        design_id=CODECONTESTS_V3_DESIGN_ID,
        max_candidates=CODECONTESTS_V3_MAX_CANDIDATES,
        ccplus_revision=CODECONTESTS_V3_CCPLUS_REVISION,
        terminal_states=CODECONTESTS_V3_TERMINAL_STATES,
    )


def resolve_codecontests_v3a_protocol(protocol_id: str) -> CodeContestsV3Protocol:
    """Resolve the V3a execution repair without changing V3 science."""

    if protocol_id != CODECONTESTS_V3A_PROTOCOL_ID:
        raise ValueError(f"unknown CodeContests V3a protocol: {protocol_id}")
    return CodeContestsV3Protocol(
        protocol_id=CODECONTESTS_V3A_PROTOCOL_ID,
        design_id=CODECONTESTS_V3A_DESIGN_ID,
        max_candidates=CODECONTESTS_V3_MAX_CANDIDATES,
        ccplus_revision=CODECONTESTS_V3_CCPLUS_REVISION,
        terminal_states=CODECONTESTS_V3A_TERMINAL_STATES,
    )


def resolve_codecontests_v3b_protocol(protocol_id: str) -> CodeContestsV3Protocol:
    """Resolve the V3b same-device rebaseline without changing V3 science."""

    if protocol_id != CODECONTESTS_V3B_PROTOCOL_ID:
        raise ValueError(f"unknown CodeContests V3b protocol: {protocol_id}")
    return CodeContestsV3Protocol(
        protocol_id=CODECONTESTS_V3B_PROTOCOL_ID,
        design_id=CODECONTESTS_V3B_DESIGN_ID,
        max_candidates=CODECONTESTS_V3_MAX_CANDIDATES,
        ccplus_revision=CODECONTESTS_V3_CCPLUS_REVISION,
        terminal_states=CODECONTESTS_V3B_TERMINAL_STATES,
    )


def resolve_codecontests_s12d_protocol(protocol_id: str) -> CodeContestsS12DProtocol:
    """Resolve only the adopted S12d UQCC identity."""

    if protocol_id != CODECONTESTS_S12D_PROTOCOL_ID:
        raise ValueError(f"unknown CodeContests S12d protocol: {protocol_id}")
    return CodeContestsS12DProtocol(
        protocol_id=CODECONTESTS_S12D_PROTOCOL_ID,
        design_id=CODECONTESTS_S12D_DESIGN_ID,
        execution_matrix=CODECONTESTS_S12D_EXECUTION_MATRIX,
        terminal_states=CODECONTESTS_S12D_TERMINAL_STATES,
    )


def learning_rate_at(update_index: int, peak: float) -> float:
    """ADR 0073's 1,024-update warmup/cosine schedule."""

    if peak not in RECIPE_PEAK_LEARNING_RATES:
        raise ValueError("peak learning rate is outside the frozen recipe ladder")
    if isinstance(update_index, bool) or not isinstance(update_index, int) or not 0 <= update_index < 1024:
        raise ValueError("update index must be in [0, 1023]")
    if update_index < 32:
        return peak * (update_index + 1) / 32
    progress = (update_index - 32) / (1023 - 32)
    multiplier = 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0
    return peak * multiplier
