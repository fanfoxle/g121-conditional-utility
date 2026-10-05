"""One-job CodeContests V1 training and recovery primitives.

The module deliberately owns no queue.  It validates and advances one closed
run card; the finite supervisor remains the sole owner of job ordering.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np

from ..data.provenance import sha256_file, verify_file
from ..protocol import (
    CODECONTESTS_DESIGN_ID,
    CODECONTESTS_FORMAL_JOB_ORDER,
    CODECONTESTS_PEAK_LEARNING_RATE,
    CODECONTESTS_PROTOCOL_ID,
    CODECONTESTS_RESPONSE_TOKEN_QUOTA,
    CODECONTESTS_ROLE_NAMES,
    CODECONTESTS_UPDATES,
    codecontests_learning_rate_at,
    derive_codecontests_role_seed,
    derive_codecontests_v2_role_seed,
)


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


W259_PAIR_SPECS = {
    "P01": (94_157_648, 432_176_365),
    "P02": (1_409_923_109, 992_380_055),
    "P03": (1_709_473_788, 1_478_250_915),
    "P04": (122_961_837, 1_875_991_562),
    "P05": (36_289_367, 1_022_018_198),
    "P06": (1_761_626_600, 1_728_384_408),
    "P07": (1_653_610_645, 1_907_824_140),
    "P08": (2_021_254_069, 1_422_873_367),
}


def validate_w259_initial_state_set(
    receipts: Sequence[Mapping[str, object]],
    *,
    expected_pairs: Mapping[str, int],
) -> dict[str, str]:
    """Close the one-fresh-initial-state-per-pair identity contract."""

    expected = {str(pair_id): int(seed) for pair_id, seed in expected_pairs.items()}
    by_pair: dict[str, str] = {}
    for receipt in receipts:
        pair_id = str(receipt.get("pair_id"))
        theta = receipt.get("theta_i_sha256")
        files = receipt.get("adapter_files")
        valid_files = (
            isinstance(files, list)
            and bool(files)
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("path"), str)
                and isinstance(row.get("sha256"), str)
                and len(str(row["sha256"])) == 64
                and all(character in "0123456789abcdef" for character in str(row["sha256"]))
                for row in files
            )
        )
        if not (
            receipt.get("schema") == "q1-w259-prospective-initial-state-v1"
            and receipt.get("protocol_id") == "q1-w259-prospective-v1"
            and pair_id in expected
            and receipt.get("init_seed") == expected[pair_id]
            and isinstance(theta, str)
            and len(theta) == 64
            and all(character in "0123456789abcdef" for character in theta)
            and receipt.get("lora_b_zero") is True
            and valid_files
            and pair_id not in by_pair
        ):
            raise ValueError("Q1-W259-PROSPECTIVE initial-state receipt differs")
        by_pair[pair_id] = theta
    if set(by_pair) != set(expected):
        raise ValueError("Q1-W259-PROSPECTIVE initial-state pair set differs")
    if len(set(by_pair.values())) != len(by_pair):
        raise ValueError("Q1-W259-PROSPECTIVE different pairs share theta_i")
    return by_pair


def validate_w259_context_initial_state_set(
    receipts: Sequence[Mapping[str, object]],
    *,
    expected_contexts: Mapping[str, int],
    protocol_id: str = "q1-w259-cf12-v1",
) -> dict[str, str]:
    """Close the one-fresh-initial-state-per-CF12-context contract."""

    schema = {"q1-w259-cf12-v1": "q1-w259-cf12-initial-state-v1",
              "Q1-G121-FIXED-REPLICATION-16-v1": "q1-g121-initial-state-v1"}.get(protocol_id)
    if schema is None:
        raise ValueError("unknown context initialization protocol")
    expected = {
        str(context_id): int(seed) for context_id, seed in expected_contexts.items()
    }
    by_context: dict[str, str] = {}
    for receipt in receipts:
        context_id = str(receipt.get("context_id"))
        theta = receipt.get("theta_i_sha256")
        files = receipt.get("adapter_files")
        valid_files = (
            isinstance(files, list)
            and bool(files)
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("path"), str)
                and isinstance(row.get("sha256"), str)
                and len(str(row["sha256"])) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in str(row["sha256"])
                )
                for row in files
            )
        )
        if not (
            receipt.get("schema") == schema
            and receipt.get("protocol_id") == protocol_id
            and context_id in expected
            and receipt.get("init_seed") == expected[context_id]
            and isinstance(theta, str)
            and len(theta) == 64
            and all(character in "0123456789abcdef" for character in theta)
            and receipt.get("lora_b_zero") is True
            and valid_files
            and context_id not in by_context
        ):
            raise ValueError("Q1-W259-CF12 initial-state receipt differs")
        by_context[context_id] = theta
    if set(by_context) != set(expected):
        raise ValueError("Q1-W259-CF12 initial-state context set differs")
    if len(set(by_context.values())) != len(by_context):
        raise ValueError("Q1-W259-CF12 different contexts share theta_i")
    return by_context


def validate_w259_pre_update_facts(
    *,
    theta_i_sha256: str,
    expected_theta_i_sha256: str,
    lora_b_nonzero_count: int,
    optimizer_state_count: int,
) -> None:
    """Fail closed immediately before the first W259 optimizer update."""

    if theta_i_sha256 != expected_theta_i_sha256:
        raise ValueError("Q1-W259-PROSPECTIVE pre-update theta_i differs")
    if lora_b_nonzero_count != 0:
        raise ValueError("Q1-W259-PROSPECTIVE pre-update LoRA B is nonzero")
    if optimizer_state_count != 0:
        raise ValueError("Q1-W259-PROSPECTIVE pre-update optimizer is not fresh")


def w259_training_run_identity(
    card: Mapping[str, object], card_sha256: str | None = None
) -> str:
    """Bind recovery to the complete pair/arm/root/plan training identity."""

    identity = card_sha256 or validate_nat_onoff_run_card(card)
    return _canonical_json_sha256(
        {
            "protocol_id": card.get("protocol_id"),
            "pair_id": card.get("pair_id"),
            "context_id": card.get("context_id"),
            "root": card.get("root"),
            "arm": card.get("arm"),
            "init_seed": card.get("init_seed"),
            "expected_theta_i_sha256": card.get("expected_theta_i_sha256"),
            "card_sha256": identity,
            "plan_sha256": card.get("plan_sha256"),
        }
    )


class NatOnOffRngStreams:
    """Two complete framework RNG streams used by B and S exposures.

    The context spans forward and backward so non-reentrant checkpoint replay
    consumes the same selected stream.  Ambient state is restored after every
    exposure; inserting an S exposure therefore cannot advance B dropout.
    """

    def __init__(self, states: Mapping[str, Mapping[str, object]]) -> None:
        if set(states) != {"B", "S"}:
            raise ValueError("NAT-ONOFF requires exactly B and S RNG streams")
        self._states = {name: self._copy_state(value) for name, value in states.items()}

    @staticmethod
    def _capture() -> dict[str, object]:
        import torch

        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        }

    @staticmethod
    def _copy_state(value: Mapping[str, object]) -> dict[str, object]:
        import torch

        if set(value) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("NAT-ONOFF RNG stream state is incomplete")
        if not isinstance(value["torch"], torch.Tensor) or not isinstance(value["cuda"], list):
            raise ValueError("NAT-ONOFF RNG stream tensor state is malformed")
        numpy_state = value["numpy"]
        if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
            raise ValueError("NAT-ONOFF NumPy RNG state is malformed")
        return {
            "python": value["python"],
            "numpy": (
                numpy_state[0], numpy_state[1].copy(), numpy_state[2],
                numpy_state[3], numpy_state[4],
            ),
            "torch": value["torch"].clone(),
            "cuda": [state.clone() for state in value["cuda"]],
        }

    @staticmethod
    def _restore(value: Mapping[str, object]) -> None:
        import torch

        random.setstate(value["python"])
        np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch"])
        if torch.cuda.is_available():
            cuda = value["cuda"]
            if len(cuda) != torch.cuda.device_count():
                raise ValueError("NAT-ONOFF CUDA RNG device registry differs")
            torch.cuda.set_rng_state_all(cuda)

    @classmethod
    def from_seeds(cls, *, background_seed: int, supplement_seed: int):
        import torch

        if any(type(seed) is not int or seed < 0 for seed in (background_seed, supplement_seed)):
            raise ValueError("NAT-ONOFF RNG seeds must be nonnegative integers")
        ambient = cls._capture()
        states = {}
        try:
            for name, seed in (("B", background_seed), ("S", supplement_seed)):
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                states[name] = cls._capture()
        finally:
            cls._restore(ambient)
        return cls(states)

    @contextmanager
    def activate(self, component: str):
        if component not in self._states:
            raise ValueError("NAT-ONOFF exposure component must be B or S")
        ambient = self._capture()
        self._restore(self._states[component])
        try:
            yield
        finally:
            self._states[component] = self._capture()
            self._restore(ambient)

    def state_dict(self) -> dict[str, dict[str, object]]:
        return {name: self._copy_state(value) for name, value in self._states.items()}

    def load_state_dict(self, states: Mapping[str, Mapping[str, object]]) -> None:
        if set(states) != {"B", "S"}:
            raise ValueError("NAT-ONOFF recovery requires both RNG streams")
        self._states = {name: self._copy_state(value) for name, value in states.items()}


def validate_codecontests_run_card(card: Mapping[str, object]) -> str:
    """Validate the frozen recipe-bearing card and return its stable identity."""

    try:
        ordinal = int(card["training_ordinal"])
        root = int(card["root"])
        arm = str(card["arm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("CodeContests run-card identity is incomplete") from exc
    if not 0 <= ordinal < len(CODECONTESTS_FORMAL_JOB_ORDER) or (
        root,
        arm,
    ) != CODECONTESTS_FORMAL_JOB_ORDER[ordinal]:
        raise ValueError("CodeContests run-card queue identity differs from freeze")
    expected_seeds = {
        role: derive_codecontests_role_seed(root, role)
        for role in CODECONTESTS_ROLE_NAMES
    }
    recipe = (
        card.get("schema") == "codecontests-v1-formal-run-card-v1"
        and card.get("protocol_id") == CODECONTESTS_PROTOCOL_ID
        and card.get("design_id") == CODECONTESTS_DESIGN_ID
        and card.get("response_token_quota") == CODECONTESTS_RESPONSE_TOKEN_QUOTA
        and card.get("updates") == CODECONTESTS_UPDATES
        and card.get("max_length") == 2_048
        and card.get("packing_path") == "single_unit"
        and card.get("role_seeds") == expected_seeds
        and card.get("outcomes_opened") is False
    )
    if not recipe:
        raise ValueError("CodeContests run-card recipe differs from freeze")
    arm_identity = card.get("arm_identity")
    if (
        not isinstance(arm_identity, Mapping)
        or arm_identity.get("record_count") != 590
        or not isinstance(arm_identity.get("response_loss_tokens"), int)
        or not isinstance(arm_identity.get("metadata_sha256"), str)
    ):
        raise ValueError("CodeContests run-card arm identity is incomplete")
    return _canonical_json_sha256(card)


def validate_codecontests_s12_run_card(card: Mapping[str, object]) -> str:
    """Validate the shared trainer's explicit S12 or NAT-family recipe identity."""

    try:
        root_value = card["root"]
        if isinstance(root_value, bool) or not isinstance(root_value, int):
            raise ValueError
        root = root_value
        stage = str(card["stage"])
        role = str(card["role"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("S12 run-card identity is incomplete") from exc
    stage_arms = {
        "PC": ("PC_CLEAN", "PC_MUTANT"),
        "NAT": ("NAT_R", "NAT_I"),
        "POLICY": ("POL_KEEP", "POL_FILTER", "POL_REPAIR"),
    }
    nat_profiles = {
        "codecontests-nat-replace-20-v1": (
            "codecontests-nat-replace-20-run-card-v1",
            "CC_NAT_REPLACE_20",
            "exploratory",
            ("POL_KEEP", "POL_FILTER"),
        ),
        "codecontests-nat-replica-x164-v1": (
            "codecontests-nat-replica-x164-run-card-v1",
            "CC_NAT_REPLICA_X164",
            "exploratory_external_replication",
            ("POL_KEEP", "POL_FILTER"),
        ),
        "codecontests-nat-msep-20-v1": (
            "codecontests-nat-msep-20-run-card-v1",
            "CC_NAT_MSEP_20",
            "exploratory_active_contrast",
            ("POL_FILTER", "POL_MSEP"),
        ),
    }
    nat_profile = nat_profiles.get(card.get("protocol_id"))
    if nat_profile:
        stage_arms = {"POLICY": nat_profile[3]}
        design_sha = card.get("design_sha256")
        if not (0 < root < 2147483647 and isinstance(design_sha, str)
                and len(design_sha) == 64 and all(c in "0123456789abcdef" for c in design_sha)):
            raise ValueError("NAT-REPLACE design/root binding is invalid")
    expected_seeds = {
        seed_role: derive_codecontests_role_seed(root, seed_role)
        for seed_role in CODECONTESTS_ROLE_NAMES
    }
    plan_sha = card.get("plan_sha256")
    card_sha = card.get("card_sha256")
    arm_ordinal = card.get("arm_ordinal")
    common = (
        card.get("schema") == (nat_profile[0] if nat_profile else "codecontests-s12-run-card-v1")
        and card.get("protocol_id") == (card.get("protocol_id") if nat_profile else "codecontests-s12-apcf-v1")
        and card.get("design_id") == (nat_profile[1] if nat_profile else "CC_S12_APCF_PC_NAT_POLICY")
        and stage in stage_arms
        and role in ({nat_profile[2]} if nat_profile else {"planning", "confirmatory"})
        and root >= 0
        and not isinstance(arm_ordinal, bool)
        and isinstance(arm_ordinal, int)
        and 0 <= arm_ordinal < len(
            stage_arms[stage] if role == "confirmatory" else stage_arms[stage][:2]
        )
        and card.get("response_token_quota") == CODECONTESTS_RESPONSE_TOKEN_QUOTA
        and card.get("updates") == CODECONTESTS_UPDATES
        and card.get("max_length") == 2_048
        and card.get("packing_path") == "single_unit"
        and card.get("role_seeds") == expected_seeds
        and card.get("outcomes_opened") is False
        and isinstance(plan_sha, str)
        and len(plan_sha) == 64
        and all(character in "0123456789abcdef" for character in plan_sha)
        and isinstance(card_sha, str)
        and len(card_sha) == 64
        and all(character in "0123456789abcdef" for character in card_sha)
    )
    if not common:
        raise ValueError("S12 run-card recipe differs from freeze")
    if role == "planning":
        planning_valid = (
            card.get("arm_alias") in {"A", "B"}
            and "scientific_arm" not in card
            and isinstance(card.get("mapping_identity_sha256"), str)
            and len(str(card["mapping_identity_sha256"])) == 64
        )
        if not planning_valid:
            raise ValueError("S12 planning card does not preserve blinding")
    else:
        if (
            card.get("scientific_arm") != stage_arms[stage][arm_ordinal]
            or "arm_alias" in card
            or "mapping_identity_sha256" in card
        ):
            raise ValueError("S12 confirmatory card arm identity is invalid")
    without_identity = dict(card)
    without_identity.pop("card_sha256", None)
    if _canonical_json_sha256(without_identity) != card_sha:
        raise ValueError("S12 run-card content identity differs")
    return str(card_sha)


def validate_nat_onoff_run_card(card: Mapping[str, object]) -> str:
    """Validate a fixed-init Q1 natural-deletion training contract."""

    root = card.get("root")
    response_tokens = card.get("response_tokens")
    identity = card.get("card_sha256")
    plan_sha = card.get("plan_sha256")
    protocol = card.get("protocol_id")
    profile = {
        "Q1-G121-FIXED-REPLICATION-16-v1": ("q1-g121-run-card-v1", {"B", "BG"}),
        "q1-nat-onoff-354-v1": ("q1-nat-onoff-354-run-card-v1", {"ON", "OFF"}),
        "q1-nat-e95-delete-v1": ("q1-nat-e95-delete-run-card-v1", {"MID"}),
        "q1-w259-prospective-v1": (
            "q1-w259-prospective-run-card-v1",
            {"W_USE", "W_OMIT"},
        ),
        "q1-w259-cf12-v1": (
            "q1-w259-cf12-run-card-v1",
            {"B", "BG", "BH", "BW"},
        ),
    }.get(protocol)
    theta_target = card.get("expected_theta0_sha256")
    theta_binding_valid = (
        theta_target is None
        if protocol == "q1-nat-onoff-354-v1"
        else (
            isinstance(card.get("expected_theta_i_sha256"), str)
            and len(str(card["expected_theta_i_sha256"])) == 64
            and all(
                character in "0123456789abcdef"
                for character in str(card["expected_theta_i_sha256"])
            )
        )
        if protocol in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        else theta_target
        == "407355b826579dc7463da4dd250282846084309c0a2be802371fe73be48d3f97"
    )
    w259_spec = W259_PAIR_SPECS.get(str(card.get("pair_id")))
    w259_binding_valid = (
        True
        if protocol not in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        else (
            w259_spec == (card.get("init_seed"), root)
            and isinstance(card.get("initial_adapter_path"), str)
            and bool(card.get("initial_adapter_path"))
            and response_tokens
            == {"W_USE": 663_404, "W_OMIT": 583_975}.get(card.get("arm"))
        )
        if protocol == "q1-w259-prospective-v1"
        else (
            isinstance(card.get("context_id"), str)
            and len(str(card["context_id"])) == 3
            and str(card["context_id"]).startswith("R" if protocol == "Q1-G121-FIXED-REPLICATION-16-v1" else "C")
            and str(card["context_id"])[1:].isdigit()
            and 1 <= int(str(card["context_id"])[1:]) <= (16 if protocol == "Q1-G121-FIXED-REPLICATION-16-v1" else 12)
            and (protocol != "Q1-G121-FIXED-REPLICATION-16-v1" or (
                card.get("pair_id") == card["context_id"]
                and isinstance(card.get("run_id"), str) and bool(card["run_id"])))
            and type(card.get("init_seed")) is int
            and 0 < int(card["init_seed"]) < 2_147_483_647
            and isinstance(card.get("initial_adapter_path"), str)
            and bool(card.get("initial_adapter_path"))
            and response_tokens
            == {"B": 583_975, "BG": 623_689, "BH": 623_690, "BW": 663_404}.get(
                card.get("arm")
            )
        )
    )
    valid = (
        profile is not None
        and card.get("schema") == profile[0]
        and type(root) is int and 0 < root < 2_147_483_647
        and card.get("arm") in profile[1]
        and (
            card.get("init_seed") == 2_026_091_201
            if protocol not in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
            else True
        )
        and card.get("updates") == 100
        and card.get("max_length") == 2_048
        and type(response_tokens) is int and response_tokens > 0
        and theta_binding_valid
        and w259_binding_valid
        and card.get("outcomes_opened") is False
        and isinstance(plan_sha, str) and len(plan_sha) == 64
        and all(character in "0123456789abcdef" for character in plan_sha)
        and isinstance(identity, str) and len(identity) == 64
        and all(character in "0123456789abcdef" for character in identity)
    )
    without_identity = dict(card)
    without_identity.pop("card_sha256", None)
    if not valid or _canonical_json_sha256(without_identity) != identity:
        raise ValueError("Q1-NAT-ONOFF run-card recipe differs from freeze")
    return identity


def derive_nat_onoff_stream_seed(root: int, component: str) -> int:
    if type(root) is not int or root <= 0 or component not in {"B", "S"}:
        raise ValueError("NAT-ONOFF stream identity is invalid")
    payload = f"q1-nat-onoff-354-v1/dropout-stream-v1|{root}|{component}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def validate_nat_onoff_training_terminal(
    path: Path, *, expected_card_sha256: str, expected_plan_sha256: str,
    expected_protocol_id: str = "q1-nat-onoff-354-v1",
    expected_theta0_sha256: str | None = None,
) -> dict[str, object]:
    value = json.loads(Path(path).read_text("utf-8"))
    schema = {
        "Q1-G121-FIXED-REPLICATION-16-v1": "q1-g121-terminal-adapter-v1",
        "q1-nat-onoff-354-v1": "q1-nat-onoff-354-terminal-adapter-v1",
        "q1-nat-e95-delete-v1": "q1-nat-e95-delete-terminal-adapter-v1",
        "q1-w259-prospective-v1": "q1-w259-prospective-terminal-adapter-v1",
        "q1-w259-cf12-v1": "q1-w259-cf12-terminal-adapter-v1",
    }.get(expected_protocol_id)
    theta_key = (
        "theta_i_sha256"
        if expected_protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        else "theta0_sha256"
    )
    if not (
        schema is not None
        and value.get("schema") == schema
        and value.get("protocol_id") == expected_protocol_id
        and value.get("card_sha256") == expected_card_sha256
        and value.get("plan_sha256") == expected_plan_sha256
        and value.get("completed_updates") == 100
        and value.get("finite") is True
        and value.get("outcomes_opened") is False
        and isinstance(value.get(theta_key), str)
        and len(str(value[theta_key])) == 64
    ):
        raise ValueError("Q1-NAT-ONOFF terminal training identity differs")
    if (
        expected_theta0_sha256 is not None
        and value.get(theta_key) != expected_theta0_sha256
    ):
        raise ValueError("Q1 natural-deletion terminal theta0 differs")
    adapter = Path(path).parent / "adapter100"
    if value.get("adapter_files") != _adapter_file_identities(adapter):
        raise ValueError("Q1-NAT-ONOFF terminal adapter files differ")
    return value


def execute_nat_onoff_training(
    *, plan, card: Mapping[str, object], model_path: Path, output_dir: Path,
    resume: bool = False, stop_after: int | None = None,
) -> dict[str, object]:
    """Train one fixed ON/OFF state from the shared theta0 with recoverable streams."""

    import torch

    from ..arms.paired_filtering import s12_training_plan_sha256
    from .modeling import build_frozen_optimizer, configure_determinism, load_frozen_model

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Q1-NAT-ONOFF training requires frozen CUBLAS workspace config")
    card_identity = validate_nat_onoff_run_card(card)
    plan_identity = s12_training_plan_sha256(plan)
    if plan_identity != card["plan_sha256"]:
        raise ValueError("Q1-NAT-ONOFF plan differs from its run card")
    actual_tokens = sum(item.retained_response_tokens for item in plan.exposures)
    if len(plan.update_bins) != 100 or actual_tokens != card["response_tokens"]:
        raise ValueError("Q1-NAT-ONOFF plan dose differs from its run card")
    if any(item.component not in {"B", "S"} for item in plan.exposures):
        raise ValueError("Q1-NAT-ONOFF plan contains a foreign component")
    if card["arm"] in {"OFF", "W_OMIT", "B"} and any(
        item.component == "S" for item in plan.exposures
    ):
        raise ValueError("Q1 natural-deletion omit plan contains supplement exposure")
    output_dir = Path(output_dir)
    terminal_path = output_dir / "TERMINAL_TRAINING_RESULT.json"
    if terminal_path.exists():
        if not resume:
            raise FileExistsError("Q1-NAT-ONOFF terminal training already exists")
        return validate_nat_onoff_training_terminal(
            terminal_path,
            expected_card_sha256=card_identity,
            expected_plan_sha256=plan_identity,
            expected_protocol_id=str(card["protocol_id"]),
            expected_theta0_sha256=(
                card.get("expected_theta_i_sha256")
                if card.get("protocol_id")
                in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
                else card.get("expected_theta0_sha256")
            ),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir = output_dir / "recovery"
    if not resume and (recovery_dir / "CURRENT.json").exists():
        raise FileExistsError("Q1-NAT-ONOFF recovery needs explicit resume")

    protocol_id = str(card["protocol_id"])
    configure_determinism(int(card["init_seed"]))
    initial_adapter = (
        Path(str(card["initial_adapter_path"]))
        if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        else None
    )
    model, _tokenizer, trainable = load_frozen_model(
        model_path,
        int(card["init_seed"]),
        initial_adapter_path=initial_adapter,
    )
    theta0 = canonical_trainable_tensor_digest(model)
    expected_theta0 = (
        card.get("expected_theta_i_sha256")
        if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        else card.get("expected_theta0_sha256")
    )
    if expected_theta0 is not None and theta0 != expected_theta0:
        raise ValueError("Q1 natural-deletion reconstructed initial digest differs")
    optimizer = build_frozen_optimizer(trainable, CODECONTESTS_PEAK_LEARNING_RATE)
    if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"} and not resume:
        validate_w259_pre_update_facts(
            theta_i_sha256=theta0,
            expected_theta_i_sha256=str(expected_theta0),
            lora_b_nonzero_count=sum(
                int(torch.count_nonzero(parameter).item())
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and "lora_B" in name
            ),
            optimizer_state_count=len(optimizer.state),
        )
    root = int(card["root"])
    streams = NatOnOffRngStreams.from_seeds(
        background_seed=derive_nat_onoff_stream_seed(root, "B"),
        supplement_seed=derive_nat_onoff_stream_seed(root, "S"),
    )
    run_identity = (
        w259_training_run_identity(card, card_identity)
        if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        else _canonical_json_sha256(
            {"card_sha256": card_identity, "plan_sha256": plan_identity}
        )
    )
    development_boundary = (
        protocol_id
        in {"q1-nat-e95-delete-v1", "q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
        and stop_after is not None
        and 0 < int(stop_after) <= 3
    )
    completed, losses, timings = 0, [], []
    if resume:
        completed, losses, timings = restore_codecontests_recovery(
            recovery_dir=recovery_dir,
            model=model,
            optimizer=optimizer,
            run_identity=run_identity,
            plan_sha256=plan_identity,
            development_boundary=development_boundary,
            extra_state_loader=lambda value: streams.load_state_dict(
                value["nat_onoff_rng_streams"]
            ),
        )
    until = 100 if stop_after is None else min(100, int(stop_after))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for index in range(completed, until):
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses.append(
            complete_codecontests_update(
                model=model,
                optimizer=optimizer,
                exposures=plan.update_bins[index],
                update_index=index,
                exposure_rng=streams.activate,
            )
        )
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        completed = index + 1
        if completed < 100 and completed % 10 == 0:
            save_codecontests_recovery(
                recovery_dir=recovery_dir,
                completed_updates=completed,
                model=model,
                optimizer=optimizer,
                run_identity=run_identity,
                plan_sha256=plan_identity,
                losses=losses,
                update_seconds=timings,
                development_boundary=development_boundary,
                extra_state={"nat_onoff_rng_streams": streams.state_dict()},
            )
        _atomic_write_json(
            output_dir / "training_progress.json",
            {"completed_updates": completed, "updates": 100},
        )
    if completed < 100:
        # Development acceptance deliberately interrupts at non-periodic
        # boundaries (one and three updates).  Persist that exact boundary so
        # the same formal recovery owner, optimizer state, and B/S cursors are
        # exercised rather than a test-only checkpoint path.
        if completed > 0:
            save_codecontests_recovery(
                recovery_dir=recovery_dir,
                completed_updates=completed,
                model=model,
                optimizer=optimizer,
                run_identity=run_identity,
                plan_sha256=plan_identity,
                losses=losses,
                update_seconds=timings,
                development_boundary=development_boundary,
                extra_state={"nat_onoff_rng_streams": streams.state_dict()},
            )
        return {
            "status": "checkpointed",
            "completed_updates": completed,
            (
                "theta_i_sha256"
                if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
                else "theta0_sha256"
            ): theta0,
            "trainable_tensor_sha256": canonical_trainable_tensor_digest(model),
            "finite": all(math.isfinite(value) for value in (*losses, *timings)),
            "losses": list(losses),
            "update_seconds": list(timings),
            "process_seconds_this_invocation": time.perf_counter() - began,
            "recovery_available": (recovery_dir / "CURRENT.json").is_file(),
        }
    if not all(math.isfinite(value) for value in (*losses, *timings)):
        raise RuntimeError("Q1-NAT-ONOFF training did not close finitely")
    adapter_tmp = output_dir / "adapter100.tmp"
    adapter = output_dir / "adapter100"
    if adapter.exists() or adapter_tmp.exists():
        raise FileExistsError("Q1-NAT-ONOFF adapter destination already exists")
    ambient = NatOnOffRngStreams._capture()
    try:
        model.save_pretrained(adapter_tmp, safe_serialization=True)
        os.replace(adapter_tmp, adapter)
    finally:
        NatOnOffRngStreams._restore(ambient)
    terminal = {
        "schema": {
            "Q1-G121-FIXED-REPLICATION-16-v1": "q1-g121-terminal-adapter-v1",
            "q1-nat-e95-delete-v1": "q1-nat-e95-delete-terminal-adapter-v1",
            "q1-w259-prospective-v1": "q1-w259-prospective-terminal-adapter-v1",
            "q1-w259-cf12-v1": "q1-w259-cf12-terminal-adapter-v1",
        }.get(protocol_id, "q1-nat-onoff-354-terminal-adapter-v1"),
        "protocol_id": card["protocol_id"],
        **({"run_id": card["run_id"], "pair_id": card["pair_id"], "context_id": card["context_id"],
            "init_seed": card["init_seed"]} if protocol_id == "Q1-G121-FIXED-REPLICATION-16-v1" else {}),
        **(
            {"pair_id": card["pair_id"], "init_seed": card["init_seed"]}
            if protocol_id == "q1-w259-prospective-v1"
            else {"context_id": card["context_id"], "init_seed": card["init_seed"]}
            if protocol_id == "q1-w259-cf12-v1"
            else {}
        ),
        "root": root,
        "arm": card["arm"],
        "card_sha256": card_identity,
        "plan_sha256": plan_identity,
        "run_identity": run_identity,
        **(
            {"theta_i_sha256": theta0}
            if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
            else {"theta0_sha256": theta0}
        ),
        "completed_updates": completed,
        "response_tokens": actual_tokens,
        "component_tokens": (
            {
                "B": sum(item.retained_response_tokens for item in plan.exposures if item.component == "B"),
                "W": sum(item.retained_response_tokens for item in plan.exposures if item.component == "S"),
            }
            if protocol_id
            in {"q1-nat-e95-delete-v1", "q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
            else {
                component: sum(
                    item.retained_response_tokens
                    for item in plan.exposures
                    if item.component == component
                )
                for component in ("B", "S")
            }
        ),
        "adapter_files": _adapter_file_identities(adapter),
        "trainable_tensor_sha256": canonical_trainable_tensor_digest(model),
        "telemetry": {
            "process_seconds_this_invocation": time.perf_counter() - began,
            "update_seconds": timings,
            "losses": losses,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "finite": True,
        "outcomes_opened": False,
    }
    _atomic_write_json(terminal_path, terminal)
    return validate_nat_onoff_training_terminal(
        terminal_path,
        expected_card_sha256=card_identity,
        expected_plan_sha256=plan_identity,
        expected_protocol_id=str(card["protocol_id"]),
        expected_theta0_sha256=(
            card.get("expected_theta_i_sha256")
            if protocol_id in {"q1-w259-prospective-v1", "q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}
            else card.get("expected_theta0_sha256")
        ),
    )


def validate_codecontests_s12_terminal(
    path: Path,
    *,
    expected_card_sha256: str,
    expected_plan_sha256: str,
    expected_root: int | None = None,
    expected_protocol_id: str = "codecontests-s12-apcf-v1",
    expected_arm: str | None = None,
) -> dict[str, object]:
    """Verify one S12 terminal adapter and every file it binds."""

    value = json.loads(path.read_text("utf-8"))
    root = value.get("root")
    nat_schemas = {
        "codecontests-nat-replace-20-v1": (
            "codecontests-nat-replace-20-terminal-adapter-v1",
            "exploratory",
            {"POL_KEEP", "POL_FILTER"},
        ),
        "codecontests-nat-replica-x164-v1": (
            "codecontests-nat-replica-x164-terminal-adapter-v1",
            "exploratory_external_replication",
            {"POL_KEEP", "POL_FILTER"},
        ),
        "codecontests-nat-msep-20-v1": (
            "codecontests-nat-msep-20-terminal-adapter-v1",
            "exploratory_active_contrast",
            {"POL_FILTER", "POL_MSEP"},
        ),
    }
    nat_profile = nat_schemas.get(expected_protocol_id)
    nat_replace = nat_profile is not None
    if expected_protocol_id not in {"codecontests-s12-apcf-v1", *nat_schemas}:
        raise ValueError("Unknown training terminal protocol")
    if not (
        value.get("schema") == (nat_profile[0] if nat_profile else "codecontests-s12-terminal-adapter-v1")
        and value.get("protocol_id") == expected_protocol_id
        and value.get("stage") in ({"POLICY"} if nat_replace else {"PC", "NAT", "POLICY"})
        and value.get("role") in ({nat_profile[1]} if nat_profile else {"planning", "confirmatory"})
        and isinstance(root, int)
        and not isinstance(root, bool)
        and (expected_root is None or root == expected_root)
        and (expected_arm is None or value.get("arm") == expected_arm)
        and (not nat_replace or 0 < root < 2147483647)
        and value.get("arm") in (nat_profile[2] if nat_profile else {
            "A", "B", "PC_CLEAN", "PC_MUTANT", "NAT_R", "NAT_I",
            "POL_KEEP", "POL_FILTER", "POL_REPAIR",
        })
        and value.get("card_sha256") == expected_card_sha256
        and value.get("plan_sha256") == expected_plan_sha256
        and value.get("completed_updates") == CODECONTESTS_UPDATES
        and value.get("response_tokens") == CODECONTESTS_RESPONSE_TOKEN_QUOTA
        and value.get("finite") is True
        and value.get("outcomes_opened") is False
        and isinstance(value.get("adapter_files"), list)
        and (not nat_replace or bool(value.get("adapter_files")))
    ):
        raise ValueError("S12 training terminal identity is invalid")
    for identity in value["adapter_files"]:
        artifact = path.parent / str(identity.get("path"))
        if not (
            artifact.is_file()
            and artifact.stat().st_size == identity.get("bytes")
            and sha256_file(artifact) == identity.get("sha256")
        ):
            raise ValueError("S12 terminal adapter artifact identity mismatch")
    return value


def codecontests_formal_run_identity(*, card_sha256: str, plan_sha256: str) -> str:
    """Bind one formal child to both the frozen card and exposure plan."""

    for value in (card_sha256, plan_sha256):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("CodeContests formal identity requires lowercase SHA-256 values")
    return _canonical_json_sha256(
        {
            "schema": "codecontests-v1-formal-run-identity-v1",
            "card_sha256": card_sha256,
            "plan_sha256": plan_sha256,
        }
    )


def canonical_trainable_tensor_digest(model) -> str:
    """Digest canonical trainable names, dtypes, shapes, and tensor bytes."""

    import torch

    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not values:
        raise ValueError("model has no trainable tensors")
    digest = hashlib.sha256()
    for name, parameter in sorted(values):
        tensor = parameter.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("nonfinite trainable tensor")
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode("utf-8"))
        # PyTorch forbids changing element size with ``view(dtype)`` on a
        # zero-dimensional tensor.  Flattening preserves the same contiguous
        # byte order and lets small scalar-model behavior tests use this seam.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def codecontests_plan_sha256(plan) -> str:
    digest = hashlib.sha256()
    for update_index, update in enumerate(plan.update_bins):
        for exposure in update:
            digest.update(
                (
                    f"{update_index}|{exposure.unit_id}|{exposure.cycle}|"
                    f"{exposure.retained_response_tokens}|{int(exposure.complete)}\n"
                ).encode("ascii")
            )
    return digest.hexdigest()


def validate_codecontests_recovery_payload(
    payload: Mapping[str, object],
    *,
    expected_run_identity: str,
    expected_plan_sha256: str,
    expected_trainable_names: set[str],
    scope_pc: bool = False,
    development_boundary: bool = False,
) -> int:
    """Fail closed before any recovery state is copied into live objects."""

    if (
        payload.get("schema") != "codecontests-v1-recovery-state-v1"
        or payload.get("run_identity") != expected_run_identity
        or payload.get("plan_sha256") != expected_plan_sha256
    ):
        raise ValueError("CodeContests recovery identity differs from frozen run")
    completed = payload.get("completed_updates")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed <= 0
        or completed >= CODECONTESTS_UPDATES
        or (
            completed % 10
            and not (scope_pc and completed == 25)
            and not (development_boundary and completed <= 3)
        )
    ):
        raise ValueError("CodeContests recovery update boundary is invalid")
    required_rng = {"python_rng", "numpy_rng", "torch_rng", "cuda_rng"}
    if not required_rng <= set(payload):
        raise ValueError("CodeContests recovery RNG state is incomplete")
    trainable = payload.get("trainable")
    if not isinstance(trainable, Mapping) or set(trainable) != expected_trainable_names:
        raise ValueError("CodeContests recovery trainable registry differs from live model")
    import torch

    if any(not isinstance(value, torch.Tensor) for value in trainable.values()):
        raise ValueError("CodeContests recovery trainable state is malformed")
    if any(not torch.isfinite(value).all() for value in trainable.values()):
        raise FloatingPointError("nonfinite CodeContests recovery tensor")
    losses = payload.get("losses")
    timings = payload.get("update_seconds")
    if (
        not isinstance(losses, Sequence)
        or isinstance(losses, (str, bytes))
        or not isinstance(timings, Sequence)
        or isinstance(timings, (str, bytes))
        or len(losses) != completed
        or len(timings) != completed
        or any(not math.isfinite(float(value)) for value in (*losses, *timings))
    ):
        raise ValueError("CodeContests recovery telemetry does not close at its boundary")
    if not isinstance(payload.get("optimizer"), Mapping):
        raise ValueError("CodeContests recovery optimizer state is malformed")
    return completed


def _atomic_torch_save(path: Path, value: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_codecontests_recovery(
    *,
    recovery_dir: Path,
    completed_updates: int,
    model,
    optimizer,
    run_identity: str,
    plan_sha256: str,
    losses: Sequence[float],
    update_seconds: Sequence[float],
    scope_pc: bool = False,
    development_boundary: bool = False,
    extra_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically replace the ten-update recovery state, retaining one state."""

    import torch

    if (
        completed_updates <= 0
        or completed_updates >= CODECONTESTS_UPDATES
        or (
            completed_updates % 10
            and not (scope_pc and completed_updates == 25)
            and not (development_boundary and completed_updates <= 3)
        )
    ):
        raise ValueError("CodeContests recovery update boundary is invalid")
    trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "schema": "codecontests-v1-recovery-state-v1",
        "completed_updates": completed_updates,
        "run_identity": run_identity,
        "plan_sha256": plan_sha256,
        "trainable": trainable,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "losses": list(losses),
        "update_seconds": list(update_seconds),
    }
    if extra_state is not None:
        if not isinstance(extra_state, Mapping) or not extra_state:
            raise ValueError("CodeContests extra recovery state is malformed")
        payload["extra_state"] = dict(extra_state)
    validate_codecontests_recovery_payload(
        payload,
        expected_run_identity=run_identity,
        expected_plan_sha256=plan_sha256,
        expected_trainable_names=set(trainable),
        scope_pc=scope_pc,
        development_boundary=development_boundary,
    )
    state_path = recovery_dir / "state.pt"
    _atomic_torch_save(state_path, payload)
    pointer = {
        "schema": "codecontests-v1-recovery-pointer-v1",
        "completed_updates": completed_updates,
        "run_identity": run_identity,
        "plan_sha256": plan_sha256,
        "state": {
            "path": state_path.name,
            "bytes": state_path.stat().st_size,
            "sha256": sha256_file(state_path),
        },
    }
    _atomic_write_json(recovery_dir / "CURRENT.json", pointer)
    return pointer


def restore_codecontests_recovery(
    *,
    recovery_dir: Path,
    model,
    optimizer,
    run_identity: str,
    plan_sha256: str,
    scope_pc: bool = False,
    development_boundary: bool = False,
    extra_state_loader=None,
) -> tuple[int, list[float], list[float]]:
    import torch

    pointer = json.loads((recovery_dir / "CURRENT.json").read_text("utf-8"))
    if (
        pointer.get("schema") != "codecontests-v1-recovery-pointer-v1"
        or pointer.get("run_identity") != run_identity
        or pointer.get("plan_sha256") != plan_sha256
    ):
        raise ValueError("CodeContests recovery pointer identity differs from frozen run")
    identity = pointer.get("state")
    if not isinstance(identity, Mapping):
        raise ValueError("CodeContests recovery state identity is absent")
    state_path = recovery_dir / str(identity.get("path"))
    verify_file(
        state_path,
        expected_size=int(identity.get("bytes", -1)),
        expected_sha256=str(identity.get("sha256")),
    )
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    by_name = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    completed = validate_codecontests_recovery_payload(
        payload,
        expected_run_identity=run_identity,
        expected_plan_sha256=plan_sha256,
        expected_trainable_names=set(by_name),
        scope_pc=scope_pc,
        development_boundary=development_boundary,
    )
    for name, value in payload["trainable"].items():
        by_name[name].data.copy_(value.to(by_name[name].device))
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if extra_state_loader is not None:
        extra = payload.get("extra_state")
        if not isinstance(extra, Mapping):
            raise ValueError("CodeContests extra recovery state is absent")
        extra_state_loader(extra)
    return completed, list(payload["losses"]), list(payload["update_seconds"])


def complete_codecontests_update(
    *, model, optimizer, exposures, update_index: int, device: str = "cuda:0",
    loss_observer=None, exposure_rng=None
) -> float:
    """Execute one response-token-normalized optimizer update."""

    import torch

    active_tokens = sum(item.retained_response_tokens for item in exposures)
    if active_tokens <= 0:
        raise ValueError("CodeContests update has no response-loss tokens")
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for exposure in exposures:
        input_ids = torch.tensor(exposure.token_ids, dtype=torch.long, device=device).unsqueeze(0)
        labels = input_ids.clone()
        mask = torch.tensor(exposure.loss_mask, dtype=torch.bool, device=device).unsqueeze(0)
        labels.masked_fill_(~mask, -100)
        rng_context = exposure_rng(exposure.component) if exposure_rng is not None else nullcontext()
        with rng_context:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=torch.ones_like(input_ids),
                    position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
                    use_cache=False,
                ).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite CodeContests training loss")
            if loss_observer is not None:
                actual_positions = (labels[0, 1:] != -100).nonzero().flatten().cpu().tolist()
                actual_positions = [position + 1 for position in actual_positions]
                if len(actual_positions) != exposure.retained_response_tokens:
                    raise ValueError("SCOPE actual shifted labels differ from planned response dose")
                loss_observer(exposure, actual_positions)
            weight = exposure.retained_response_tokens / active_tokens
            (loss.float() * weight).backward()
        weighted_loss += float(loss.detach().float().cpu()) * weight
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite CodeContests gradient norm")
    learning_rate = codecontests_learning_rate_at(update_index)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.step()
    canonical_trainable_tensor_digest(model)
    return weighted_loss


def validate_codecontests_terminal(
    terminal: Mapping[str, object],
    *,
    expected_run_identity: str,
    expected_plan_sha256: str,
) -> None:
    adapter_files = terminal.get("adapter_files")
    adapter_valid = (
        isinstance(adapter_files, Sequence)
        and not isinstance(adapter_files, (str, bytes))
        and len(adapter_files) > 0
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and str(item.get("path")).startswith("adapter/")
            and isinstance(item.get("bytes"), int)
            and int(item.get("bytes")) > 0
            and isinstance(item.get("sha256"), str)
            and len(str(item.get("sha256"))) == 64
            for item in adapter_files
        )
    )
    if not (
        terminal.get("schema") == "codecontests-v1-terminal-adapter-v1"
        and terminal.get("run_identity") == expected_run_identity
        and terminal.get("plan_sha256") == expected_plan_sha256
        and terminal.get("completed_updates") == CODECONTESTS_UPDATES
        and terminal.get("response_tokens") == CODECONTESTS_RESPONSE_TOKEN_QUOTA
        and isinstance(terminal.get("trainable_tensor_sha256"), str)
        and len(str(terminal.get("trainable_tensor_sha256"))) == 64
        and terminal.get("finite") is True
        and adapter_valid
    ):
        raise ValueError("CodeContests terminal adapter identity is invalid")


def _adapter_file_identities(adapter_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(adapter_dir.parent).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    ]


def prepare_w259_pair_initial_state(
    *,
    model_path: Path,
    pair_dir: Path,
    pair_id: str,
    init_seed: int,
) -> dict[str, object]:
    """Create exactly one reusable fresh LoRA state for a frozen W259 pair."""

    import torch

    from .modeling import load_frozen_model

    if W259_PAIR_SPECS.get(pair_id, (None, None))[0] != init_seed:
        raise ValueError("Q1-W259-PROSPECTIVE pair/init seed differs")
    pair_dir = Path(pair_dir)
    receipt_path = pair_dir / "INITIAL_STATE.json"
    adapter = pair_dir / "adapter"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text("utf-8"))
        validate_w259_initial_state_set(
            (receipt,), expected_pairs={pair_id: init_seed}
        )
        if receipt.get("adapter_files") != _adapter_file_identities(adapter):
            raise ValueError("Q1-W259-PROSPECTIVE initial adapter files differ")
        return receipt
    if pair_dir.exists() and any(pair_dir.iterdir()):
        raise FileExistsError("Q1-W259-PROSPECTIVE partial initial state exists")
    pair_dir.mkdir(parents=True, exist_ok=True)
    temporary = pair_dir / "adapter.tmp"
    model, _tokenizer, _trainable = load_frozen_model(Path(model_path), init_seed)
    theta_i = canonical_trainable_tensor_digest(model)
    b_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    validate_w259_pre_update_facts(
        theta_i_sha256=theta_i,
        expected_theta_i_sha256=theta_i,
        lora_b_nonzero_count=b_nonzero,
        optimizer_state_count=0,
    )
    model.save_pretrained(temporary, safe_serialization=True)
    os.replace(temporary, adapter)
    receipt = {
        "schema": "q1-w259-prospective-initial-state-v1",
        "protocol_id": "q1-w259-prospective-v1",
        "pair_id": pair_id,
        "init_seed": init_seed,
        "theta_i_sha256": theta_i,
        "lora_b_zero": True,
        "adapter_files": _adapter_file_identities(adapter),
    }
    validate_w259_initial_state_set(
        (receipt,), expected_pairs={pair_id: init_seed}
    )
    _atomic_write_json(receipt_path, receipt)
    del model
    torch.cuda.empty_cache()
    return receipt


def prepare_w259_context_initial_state(
    *,
    model_path: Path,
    context_dir: Path,
    context_id: str,
    init_seed: int,
    protocol_id: str = "q1-w259-cf12-v1",
) -> dict[str, object]:
    """Create one reusable fresh LoRA state for a frozen CF12 context."""

    import torch

    from .modeling import load_frozen_model

    replication = protocol_id == "Q1-G121-FIXED-REPLICATION-16-v1"
    if protocol_id not in {"q1-w259-cf12-v1", "Q1-G121-FIXED-REPLICATION-16-v1"}:
        raise ValueError("unknown context initialization protocol")
    if context_id not in {f'{"R" if replication else "C"}{index:02d}' for index in range(1, 17 if replication else 13)}:
        raise ValueError("Q1-W259-CF12 context id differs")
    if not isinstance(init_seed, int) or isinstance(init_seed, bool) or init_seed <= 0:
        raise ValueError("Q1-W259-CF12 context seed is invalid")
    context_dir = Path(context_dir)
    receipt_path = context_dir / "INITIAL_STATE.json"
    adapter = context_dir / "adapter"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text("utf-8"))
        validate_w259_context_initial_state_set(
            (receipt,), expected_contexts={context_id: init_seed}, protocol_id=protocol_id
        )
        if receipt.get("adapter_files") != _adapter_file_identities(adapter):
            raise ValueError("Q1-W259-CF12 initial adapter files differ")
        return receipt
    if context_dir.exists() and any(context_dir.iterdir()):
        raise FileExistsError("Q1-W259-CF12 partial initial state exists")
    context_dir.mkdir(parents=True, exist_ok=True)
    temporary = context_dir / "adapter.tmp"
    model, _tokenizer, _trainable = load_frozen_model(Path(model_path), init_seed)
    theta_i = canonical_trainable_tensor_digest(model)
    b_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    validate_w259_pre_update_facts(
        theta_i_sha256=theta_i,
        expected_theta_i_sha256=theta_i,
        lora_b_nonzero_count=b_nonzero,
        optimizer_state_count=0,
    )
    model.save_pretrained(temporary, safe_serialization=True)
    os.replace(temporary, adapter)
    receipt = {
        "schema": "q1-g121-initial-state-v1" if replication else "q1-w259-cf12-initial-state-v1",
        "protocol_id": protocol_id,
        "context_id": context_id,
        "init_seed": init_seed,
        "theta_i_sha256": theta_i,
        "lora_b_zero": True,
        "adapter_files": _adapter_file_identities(adapter),
    }
    validate_w259_context_initial_state_set(
        (receipt,), expected_contexts={context_id: init_seed}, protocol_id=protocol_id
    )
    _atomic_write_json(receipt_path, receipt)
    del model
    torch.cuda.empty_cache()
    return receipt


def execute_codecontests_formal(
    *,
    units,
    card: Mapping[str, object],
    model_path: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, object]:
    """Execute one frozen 100-update formal child and persist one adapter."""

    import torch

    from .modeling import build_frozen_optimizer, configure_determinism, load_frozen_model
    from .stream import TokenizedUnit, build_codecontests_training_plan

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CodeContests formal training requires frozen CUBLAS workspace config")
    card_identity = validate_codecontests_run_card(card)
    arm = str(card["arm"])
    root = int(card["root"])
    if len(units) != 590 or any(unit.arm != arm for unit in units):
        raise ValueError("CodeContests formal units differ from the run card")
    tokenized = tuple(
        TokenizedUnit(
            unit_id=unit.unit_id,
            token_ids=unit.token_ids,
            loss_mask=unit.loss_mask,
            sampling_id=unit.triplet_id,
        )
        for unit in units
    )
    plan = build_codecontests_training_plan(tokenized, root=root)
    plan_identity = codecontests_plan_sha256(plan)
    run_identity = codecontests_formal_run_identity(
        card_sha256=card_identity, plan_sha256=plan_identity
    )
    terminal_path = output_dir / "TERMINAL_TRAINING_RESULT.json"
    recovery_dir = output_dir / "recovery"
    adapter_dir = output_dir / "adapter"
    adapter_temporary = output_dir / "adapter.tmp"
    if terminal_path.exists():
        raise FileExistsError("CodeContests formal job already has a terminal result")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (recovery_dir / "CURRENT.json").exists():
        raise FileExistsError("CodeContests recovery exists; explicit resume is required")
    if adapter_dir.exists() or adapter_temporary.exists():
        raise FileExistsError("CodeContests terminal adapter output already exists")

    role_seeds = card["role_seeds"]
    configure_determinism(int(role_seeds["framework"]))
    model, _tokenizer, trainable = load_frozen_model(
        model_path, int(role_seeds["lora-initialization"])
    )
    configure_determinism(int(role_seeds["dropout"]))
    optimizer = build_frozen_optimizer(trainable, CODECONTESTS_PEAK_LEARNING_RATE)
    completed = 0
    losses: list[float] = []
    update_seconds: list[float] = []
    if resume:
        completed, losses, update_seconds = restore_codecontests_recovery(
            recovery_dir=recovery_dir,
            model=model,
            optimizer=optimizer,
            run_identity=run_identity,
            plan_sha256=plan_identity,
        )

    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update_index in range(completed, CODECONTESTS_UPDATES):
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        losses.append(
            complete_codecontests_update(
                model=model,
                optimizer=optimizer,
                exposures=plan.update_bins[update_index],
                update_index=update_index,
            )
        )
        torch.cuda.synchronize()
        update_seconds.append(time.perf_counter() - update_started)
        completed = update_index + 1
        if completed % 10 == 0 and completed < CODECONTESTS_UPDATES:
            save_codecontests_recovery(
                recovery_dir=recovery_dir,
                completed_updates=completed,
                model=model,
                optimizer=optimizer,
                run_identity=run_identity,
                plan_sha256=plan_identity,
                losses=losses,
                update_seconds=update_seconds,
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if completed != CODECONTESTS_UPDATES or not all(
        math.isfinite(value) for value in (*losses, *update_seconds)
    ):
        raise RuntimeError("CodeContests formal training did not close finitely")

    model.save_pretrained(adapter_temporary, safe_serialization=True)
    os.replace(adapter_temporary, adapter_dir)
    terminal = {
        "schema": "codecontests-v1-terminal-adapter-v1",
        "protocol_id": CODECONTESTS_PROTOCOL_ID,
        "design_id": CODECONTESTS_DESIGN_ID,
        "training_ordinal": int(card["training_ordinal"]),
        "root": root,
        "arm": arm,
        "run_card_sha256": card_identity,
        "run_identity": run_identity,
        "plan_sha256": plan_identity,
        "completed_updates": completed,
        "response_tokens": sum(
            exposure.retained_response_tokens
            for update in plan.update_bins
            for exposure in update
        ),
        "peak_learning_rate": CODECONTESTS_PEAK_LEARNING_RATE,
        "trainable_tensor_sha256": canonical_trainable_tensor_digest(model),
        "adapter_files": _adapter_file_identities(adapter_dir),
        "telemetry": {
            "process_seconds_this_invocation": elapsed,
            "update_seconds": update_seconds,
            "losses": losses,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "total_device_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "finite": True,
        "outcomes_opened": False,
        "raw_solution_persisted": False,
    }
    validate_codecontests_terminal(
        terminal,
        expected_run_identity=run_identity,
        expected_plan_sha256=plan_identity,
    )
    _atomic_write_json(terminal_path, terminal)
    return terminal


def execute_codecontests_s12_formal(
    *,
    plan,
    card: Mapping[str, object],
    model_path: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, object]:
    """Execute one frozen S12 plan while keeping planning arm meaning opaque."""

    import torch

    from ..arms.paired_filtering import s12_training_plan_sha256
    from .modeling import build_frozen_optimizer, configure_determinism, load_frozen_model

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("S12 formal training requires frozen CUBLAS workspace config")
    card_identity = validate_codecontests_s12_run_card(card)
    plan_identity = s12_training_plan_sha256(plan)
    if plan_identity != card.get("plan_sha256"):
        raise ValueError("S12 supplied training plan differs from its run card")
    if (
        len(plan.update_bins) != CODECONTESTS_UPDATES
        or sum(item.retained_response_tokens for item in plan.exposures)
        != CODECONTESTS_RESPONSE_TOKEN_QUOTA
    ):
        raise ValueError("S12 supplied training plan differs from the frozen recipe")
    run_identity = codecontests_formal_run_identity(
        card_sha256=card_identity, plan_sha256=plan_identity
    )
    terminal_path = output_dir / "TERMINAL_TRAINING_RESULT.json"
    recovery_dir = output_dir / "recovery"
    nat_replace = card["protocol_id"] in {
        "codecontests-nat-replace-20-v1",
        "codecontests-nat-replica-x164-v1",
        "codecontests-nat-msep-20-v1",
    }
    adapter_name = "adapter100" if nat_replace else "adapter"
    adapter_dir = output_dir / adapter_name
    adapter_temporary = output_dir / (adapter_name + ".tmp")
    if terminal_path.exists():
        if nat_replace and resume:
            return validate_codecontests_s12_terminal(
                terminal_path, expected_card_sha256=card_identity,
                expected_plan_sha256=plan_identity, expected_root=int(card["root"]),
                expected_protocol_id=str(card["protocol_id"]),
                expected_arm=str(card["scientific_arm"]))
        raise FileExistsError("S12 formal job already has a terminal result")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (recovery_dir / "CURRENT.json").exists():
        raise FileExistsError("S12 recovery exists; explicit resume is required")
    if adapter_dir.exists() or adapter_temporary.exists():
        raise FileExistsError("S12 terminal adapter output already exists")

    role_seeds = card["role_seeds"]
    configure_determinism(int(role_seeds["framework"]))
    model, _tokenizer, trainable = load_frozen_model(
        model_path, int(role_seeds["lora-initialization"])
    )
    configure_determinism(int(role_seeds["dropout"]))
    optimizer = build_frozen_optimizer(trainable, CODECONTESTS_PEAK_LEARNING_RATE)
    completed = 0
    losses: list[float] = []
    update_seconds: list[float] = []
    if resume:
        completed, losses, update_seconds = restore_codecontests_recovery(
            recovery_dir=recovery_dir,
            model=model,
            optimizer=optimizer,
            run_identity=run_identity,
            plan_sha256=plan_identity,
        )
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update_index in range(completed, CODECONTESTS_UPDATES):
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        losses.append(
            complete_codecontests_update(
                model=model,
                optimizer=optimizer,
                exposures=plan.update_bins[update_index],
                update_index=update_index,
            )
        )
        torch.cuda.synchronize()
        update_seconds.append(time.perf_counter() - update_started)
        completed = update_index + 1
        if completed % 10 == 0 and completed < CODECONTESTS_UPDATES:
            save_codecontests_recovery(
                recovery_dir=recovery_dir,
                completed_updates=completed,
                model=model,
                optimizer=optimizer,
                run_identity=run_identity,
                plan_sha256=plan_identity,
                losses=losses,
                update_seconds=update_seconds,
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if completed != CODECONTESTS_UPDATES or not all(
        math.isfinite(value) for value in (*losses, *update_seconds)
    ):
        raise RuntimeError("S12 formal training did not close finitely")
    model.save_pretrained(adapter_temporary, safe_serialization=True)
    os.replace(adapter_temporary, adapter_dir)
    arm = card.get("arm_alias") if card["role"] == "planning" else card.get("scientific_arm")
    terminal = {
        "schema": (
            "codecontests-nat-msep-20-terminal-adapter-v1"
            if card["protocol_id"] == "codecontests-nat-msep-20-v1"
            else "codecontests-nat-replica-x164-terminal-adapter-v1"
            if card["protocol_id"] == "codecontests-nat-replica-x164-v1"
            else "codecontests-nat-replace-20-terminal-adapter-v1"
            if nat_replace
            else "codecontests-s12-terminal-adapter-v1"
        ),
        "protocol_id": card["protocol_id"],
        "stage": card["stage"],
        "role": card["role"],
        "root": card["root"],
        "arm": arm,
        "card_sha256": card_identity,
        "plan_sha256": plan_identity,
        "run_identity": run_identity,
        "completed_updates": completed,
        "response_tokens": CODECONTESTS_RESPONSE_TOKEN_QUOTA,
        "trainable_tensor_sha256": canonical_trainable_tensor_digest(model),
        "adapter_files": _adapter_file_identities(adapter_dir),
        "telemetry": {
            "process_seconds_this_invocation": elapsed,
            "update_seconds": update_seconds,
            "losses": losses,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "total_device_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "finite": True,
        "outcomes_opened": False,
        "raw_solution_persisted": False,
    }
    _atomic_write_json(terminal_path, terminal)
    validate_codecontests_s12_terminal(
        terminal_path,
        expected_card_sha256=card_identity,
        expected_plan_sha256=plan_identity,
        expected_root=int(card["root"]),
        expected_protocol_id=str(card["protocol_id"]),
        expected_arm=str(arm),
    )
    return terminal


def reconstruct_scope_pc_missing_snapshot(output_dir: Path, *, run_identity: str, plan_identity: str):
    """Preserve one interrupted job and reconstruct its original dose if step25 is lost."""
    pointer_path = output_dir / "recovery/CURRENT.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text("utf-8"))
    if pointer.get("run_identity") != run_identity or pointer.get("plan_sha256") != plan_identity:
        raise ValueError("SCOPE recovery identity differs; cannot reconstruct a foreign job")
    if pointer["completed_updates"] < 25 or (output_dir / "adapter25/SNAPSHOT.json").is_file():
        return None
    source = output_dir.resolve()
    if source.parent == source or source.name in {"runtime", "training", "artifacts"}:
        raise ValueError("SCOPE reconstruction target must be a single job")
    evidence_parent = source.parent / "repair_evidence"
    evidence_parent.mkdir(exist_ok=True)
    index = 1
    evidence = evidence_parent / f"{source.name}_missing25_{index}"
    while evidence.exists():
        index += 1
        evidence = evidence_parent / f"{source.name}_missing25_{index}"
    if not evidence.resolve().is_relative_to(source.parent):
        raise ValueError("SCOPE recovery evidence escaped task job parent")
    source.rename(evidence)
    source.mkdir()
    record = {"reason": "missing_true_step25_snapshot", "evidence_path": str(evidence),
              "run_identity": run_identity, "plan_sha256": plan_identity, "same_original_job": True}
    _atomic_write_json(source / "RECONSTRUCTION.json", record)
    return {**record, "evidence_path": evidence}


def validate_scope_pc_snapshot(path: Path, *, step: int, run_identity: str) -> dict[str, object]:
    """A metadata file alone is never evidence that a historical adapter exists."""
    value = json.loads((path / "SNAPSHOT.json").read_text("utf-8"))
    if value.get("step") != step or value.get("run_identity") != run_identity or not value.get("files"):
        raise ValueError("SCOPE snapshot identity or payload inventory missing")
    for name, digest in value["files"].items():
        member = path / name
        if not member.is_file() or sha256_file(member) != digest:
            raise ValueError("SCOPE snapshot payload differs or is missing")
    return value


def execute_scope_pc_updates(
    *, model, optimizer, plan, output_dir: Path, run_identity: str,
    plan_identity: str, device: str = "cuda:0", resume: bool = False,
    stop_after: int | None = None,
    geometry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Shared real update/snapshot control; CPU fixtures and formal jobs use this loop."""
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery = output_dir / "recovery"
    completed, losses, timings = 0, [], []
    audit_path = output_dir / "supervision.json"
    audits = []
    if resume:
        completed, losses, timings = restore_codecontests_recovery(
            recovery_dir=recovery, model=model, optimizer=optimizer,
            run_identity=run_identity, plan_sha256=plan_identity, scope_pc=True)
        if completed >= 25 and not (output_dir / "adapter25/SNAPSHOT.json").exists():
            raise ValueError("SCOPE missing step25 snapshot; original job must be reconstructed")
        if completed >= 25:
            validate_scope_pc_snapshot(output_dir / "adapter25", step=25, run_identity=run_identity)
        if geometry is not None:
            saved = json.loads(audit_path.read_text("utf-8"))
            if saved["run_identity"] != run_identity or len(saved["updates"]) < completed:
                raise ValueError("SCOPE supervision recovery is incomplete")
            audits = saved["updates"][:completed]
    model.train()
    total = len(plan.update_bins)
    until = total if stop_after is None else min(stop_after, total)
    for index in range(completed, until):
        start = time.perf_counter()
        observed = {}

        def observe(exposure, positions):
            if geometry is None:
                return
            bounds = geometry[exposure.unit_id]
            row = observed.setdefault(exposure.unit_id, {key: 0 for key in (
                "tokens", "direct", "context", "first_divergence", "eos", "exposures", "partial_exposures")})
            row["tokens"] += len(positions)
            row["direct"] += sum(bounds["envelope"][0] <= p < bounds["envelope"][1] for p in positions)
            row["context"] += sum(p > bounds["k"] for p in positions)
            row["first_divergence"] += bounds["k"] in positions
            row["eos"] += bounds["eos_index"] in positions
            row["exposures"] += 1
            row["partial_exposures"] += not exposure.complete

        losses.append(complete_codecontests_update(model=model, optimizer=optimizer,
            exposures=plan.update_bins[index], update_index=index, device=device, loss_observer=observe))
        timings.append(time.perf_counter() - start)
        completed = index + 1
        audits.append(observed)
        if geometry is not None:
            _atomic_write_json(audit_path, {"run_identity": run_identity, "updates": audits})
        if completed in (25, 100):
            destination = output_dir / f"adapter{completed}"
            if destination.exists():
                snapshot = validate_scope_pc_snapshot(destination, step=completed, run_identity=run_identity)
                if snapshot["trainable_tensor_sha256"] != canonical_trainable_tensor_digest(model):
                    raise ValueError("SCOPE existing snapshot differs from original update trajectory")
            if not destination.exists():
                # Export is observational even if a backend's saver uses RNG internally.
                rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                       torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
                temporary = output_dir / f"adapter{completed}.tmp"
                try:
                    model.save_pretrained(temporary, safe_serialization=True)
                    _atomic_write_json(temporary / "SNAPSHOT.json", {
                        "step": completed, "run_identity": run_identity,
                        "files": {p.relative_to(temporary).as_posix(): sha256_file(p)
                                  for p in sorted(temporary.rglob("*")) if p.is_file() and p.name != "SNAPSHOT.json"},
                        "trainable_tensor_sha256": canonical_trainable_tensor_digest(model)})
                    os.replace(temporary, destination)
                finally:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rng[3])
        if completed < 100 and (completed % 10 == 0 or completed == 25):
            save_codecontests_recovery(recovery_dir=recovery, completed_updates=completed,
                model=model, optimizer=optimizer, run_identity=run_identity, plan_sha256=plan_identity,
                losses=losses, update_seconds=timings, scope_pc=True)
        _atomic_write_json(output_dir / "training_progress.json", {"completed_updates": completed, "updates": total})
    totals = {}
    for update in audits:
        for unit_id, metrics in update.items():
            row = totals.setdefault(unit_id, {key: 0 for key in metrics})
            for key, value in metrics.items():
                row[key] += value
    return {"completed_updates": completed, "losses": losses, "update_seconds": timings,
            "supervision": totals}


def execute_scope_pc_training(*, plan, card: Mapping[str, object], model_path: Path,
                              output_dir: Path, geometry: Mapping[str, object], resume: bool = False) -> dict[str, object]:
    """One fixed SCOPE job from BASE, sharing the qualified optimizer/update owner."""
    from ..arms.paired_filtering import SCOPE_PC_PROTOCOL_ID, s12_training_plan_sha256
    if len(plan.update_bins) != 100 or sum(e.retained_response_tokens for e in plan.exposures) != 583975:
        raise ValueError("SCOPE formal Q/U differs from frozen dose")
    if card.get("protocol_id") != SCOPE_PC_PROTOCOL_ID or card.get("arm") not in {"R", "M"}:
        raise ValueError("SCOPE formal card identity invalid")
    plan_identity = s12_training_plan_sha256(plan)
    if card.get("plan_sha256") != plan_identity or not card.get("design_bound"):
        raise ValueError("SCOPE plan or design binding missing")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("SCOPE deterministic CUBLAS environment missing")
    run_identity = _canonical_json_sha256(card)
    terminal_path = output_dir / "TRAINING_RESULT.json"
    if terminal_path.exists():
        terminal = json.loads(terminal_path.read_text("utf-8"))
        if (terminal.get("run_identity") != run_identity or terminal.get("plan_sha256") != plan_identity
                or terminal.get("completed_updates") != 100 or terminal.get("response_tokens") != 583975):
            raise ValueError("SCOPE existing terminal identity differs")
        for step in (25, 100):
            validate_scope_pc_snapshot(output_dir / f"adapter{step}", step=step, run_identity=run_identity)
            if terminal.get("snapshots", {}).get(str(step)) != _adapter_file_identities(output_dir / f"adapter{step}"):
                raise ValueError("SCOPE existing terminal snapshot inventory differs")
        return terminal
    if resume and reconstruct_scope_pc_missing_snapshot(output_dir, run_identity=run_identity, plan_identity=plan_identity):
        resume = False
    import torch
    from .modeling import build_frozen_optimizer, configure_determinism, load_frozen_model
    seeds = {role: derive_codecontests_role_seed(int(card["root"]), role) for role in CODECONTESTS_ROLE_NAMES}
    if card.get("role_seeds") != seeds:
        raise ValueError("SCOPE role seeds differ from registered root")
    configure_determinism(seeds["framework"])
    model, _tokenizer, trainable = load_frozen_model(model_path, seeds["lora-initialization"])
    configure_determinism(seeds["dropout"])
    optimizer = build_frozen_optimizer(trainable, CODECONTESTS_PEAK_LEARNING_RATE)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = execute_scope_pc_updates(model=model, optimizer=optimizer, plan=plan, output_dir=output_dir,
        run_identity=run_identity, plan_identity=plan_identity, resume=resume, geometry=geometry)
    observed = sum(v["tokens"] for v in result["supervision"].values())
    if observed != 583975 or result["completed_updates"] != 100:
        raise RuntimeError("SCOPE actual supervision does not close Q/U")
    result.update(protocol_id=SCOPE_PC_PROTOCOL_ID, root=card["root"], arm=card["arm"],
        run_identity=run_identity, plan_sha256=plan_identity, response_tokens=observed,
        wall_seconds=time.perf_counter() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        snapshots={str(step): _adapter_file_identities(output_dir / f"adapter{step}") for step in (25, 100)})
    _atomic_write_json(terminal_path, result)
    return result


def execute_codecontests_diagnostic(
    *,
    units,
    arm: str,
    root: int,
    model_path: Path,
    output_dir: Path,
    quota: int,
    updates: int,
    stop_after: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Run one bounded non-formal CUDA diagnostic through the exact recipe."""

    import torch

    from .modeling import build_frozen_optimizer, configure_determinism, load_frozen_model
    from .stream import TokenizedUnit, build_codecontests_training_plan

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CodeContests diagnostic requires frozen CUBLAS workspace config")
    if arm not in ("I", "R", "N") or updates <= 0 or updates >= CODECONTESTS_UPDATES:
        raise ValueError("CodeContests diagnostic identity is invalid")
    if stop_after is not None and not 0 < stop_after <= updates:
        raise ValueError("CodeContests diagnostic stop boundary is invalid")
    tokenized = tuple(
        TokenizedUnit(
            unit_id=unit.unit_id,
            token_ids=unit.token_ids,
            loss_mask=unit.loss_mask,
            sampling_id=unit.triplet_id,
        )
        for unit in units
    )
    plan = build_codecontests_training_plan(
        tokenized,
        root=root,
        quota=quota,
        updates=updates,
    )
    plan_identity = codecontests_plan_sha256(plan)
    run_identity = _canonical_json_sha256(
        {
            "schema": "codecontests-v1-diagnostic-run-v1",
            "arm": arm,
            "root": root,
            "quota": quota,
            "updates": updates,
            "plan_sha256": plan_identity,
            "unit_metadata": [
                (unit.unit_id, unit.problem_fingerprint, unit.solution_fingerprint)
                for unit in units
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_determinism(derive_codecontests_role_seed(root, "framework"))
    model, _tokenizer, trainable = load_frozen_model(
        model_path, derive_codecontests_role_seed(root, "lora-initialization")
    )
    configure_determinism(derive_codecontests_role_seed(root, "dropout"))
    optimizer = build_frozen_optimizer(trainable, CODECONTESTS_PEAK_LEARNING_RATE)
    completed = 0
    losses: list[float] = []
    update_seconds: list[float] = []
    recovery_dir = output_dir / "recovery"
    if resume:
        completed, losses, update_seconds = restore_codecontests_recovery(
            recovery_dir=recovery_dir,
            model=model,
            optimizer=optimizer,
            run_identity=run_identity,
            plan_sha256=plan_identity,
        )
    elif (output_dir / "DIAGNOSTIC_RESULT.json").exists() or recovery_dir.exists():
        raise FileExistsError("CodeContests diagnostic output already contains state")
    limit = updates if stop_after is None else stop_after
    if completed >= limit:
        raise ValueError("CodeContests diagnostic resume boundary does not advance")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update_index in range(completed, limit):
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        losses.append(
            complete_codecontests_update(
                model=model,
                optimizer=optimizer,
                exposures=plan.update_bins[update_index],
                update_index=update_index,
            )
        )
        torch.cuda.synchronize()
        update_seconds.append(time.perf_counter() - update_started)
        completed = update_index + 1
        if completed % 10 == 0 and completed < updates:
            save_codecontests_recovery(
                recovery_dir=recovery_dir,
                completed_updates=completed,
                model=model,
                optimizer=optimizer,
                run_identity=run_identity,
                plan_sha256=plan_identity,
                losses=losses,
                update_seconds=update_seconds,
            )
    process_seconds = time.perf_counter() - started
    peak_reserved = int(torch.cuda.max_memory_reserved())
    result = {
        "schema": "codecontests-v1-diagnostic-result-v1",
        "formal": False,
        "root": root,
        "arm": arm,
        "run_identity": run_identity,
        "plan_sha256": plan_identity,
        "completed_updates": completed,
        "target_updates": updates,
        "response_tokens": sum(
            exposure.retained_response_tokens
            for update in plan.update_bins[:completed]
            for exposure in update
        ),
        "losses": losses,
        "update_seconds": update_seconds,
        "process_seconds_this_invocation": process_seconds,
        "peak_reserved_bytes": peak_reserved,
        "trainable_tensor_sha256": canonical_trainable_tensor_digest(model),
        "finite": all(math.isfinite(value) for value in (*losses, *update_seconds)),
        "status": "complete" if completed == updates else "checkpointed",
        "raw_solution_persisted": False,
    }
    _atomic_write_json(output_dir / "DIAGNOSTIC_RESULT.json", result)
    return result


def execute_codecontests_v2_diagnostic(
    *,
    plan,
    stage: str,
    arm: str,
    root: int,
    model_path: Path,
    output_dir: Path,
    save_adapter: bool = False,
) -> dict[str, object]:
    """Run at most three non-formal V2 updates through the frozen CUDA recipe."""

    import torch

    from .modeling import build_frozen_optimizer, configure_determinism, load_frozen_model

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CodeContests V2 diagnostic requires frozen CUBLAS workspace config")
    updates = len(plan.update_bins)
    if updates <= 0 or updates > 3 or any(not update for update in plan.update_bins):
        raise ValueError("CodeContests V2 diagnostic is capped at three nonempty updates")
    plan_identity = codecontests_plan_sha256(plan)
    run_identity = _canonical_json_sha256(
        {
            "schema": "codecontests-v2-diagnostic-run-v1",
            "stage": stage,
            "arm": arm,
            "root": root,
            "plan_sha256": plan_identity,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_determinism(derive_codecontests_v2_role_seed(stage, root, "framework"))
    model, _tokenizer, trainable = load_frozen_model(
        model_path,
        derive_codecontests_v2_role_seed(stage, root, "lora-initialization"),
    )
    configure_determinism(derive_codecontests_v2_role_seed(stage, root, "dropout"))
    optimizer = build_frozen_optimizer(trainable, CODECONTESTS_PEAK_LEARNING_RATE)
    losses: list[float] = []
    timings: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for update_index, exposures in enumerate(plan.update_bins):
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        losses.append(
            complete_codecontests_update(
                model=model,
                optimizer=optimizer,
                exposures=exposures,
                update_index=update_index,
            )
        )
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - update_started)
    result = {
        "schema": "codecontests-v2-diagnostic-result-v1",
        "formal": False,
        "stage": stage,
        "arm": arm,
        "root": root,
        "run_identity": run_identity,
        "plan_sha256": plan_identity,
        "completed_updates": updates,
        "response_tokens": sum(
            exposure.retained_response_tokens
            for update in plan.update_bins
            for exposure in update
        ),
        "losses": losses,
        "update_seconds": timings,
        "process_seconds": time.perf_counter() - started,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "trainable_tensor_sha256": canonical_trainable_tensor_digest(model),
        "finite": all(math.isfinite(value) for value in (*losses, *timings)),
        "status": "complete",
        "raw_solution_persisted": False,
    }
    if save_adapter:
        if not result["finite"]:
            raise RuntimeError("development adapter cannot be published from nonfinite training")
        temporary = output_dir / "adapter.tmp"
        model.save_pretrained(temporary, safe_serialization=True)
        os.replace(temporary, output_dir / "adapter")
        result["adapter_files"] = _adapter_file_identities(output_dir / "adapter")
    _atomic_write_json(output_dir / "DIAGNOSTIC_RESULT.json", result)
    return result


def _diagnostic_cli() -> int:
    import argparse

    from transformers import AutoTokenizer

    from ..arms.codecontests import (
        build_codecontests_run_cards,
        load_referenced_codecontests_rows,
        materialize_codecontests_rows,
        read_codecontests_manifest,
    )

    parser = argparse.ArgumentParser(description="Run one CodeContests training child")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("I", "R", "N"))
    parser.add_argument("--root", type=int)
    parser.add_argument("--quota", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--s12-plan", type=Path)
    arguments = parser.parse_args()
    arms = None
    if arguments.s12_plan is None:
        manifest = read_codecontests_manifest(arguments.manifest)
        rows = load_referenced_codecontests_rows(arguments.dataset_root, manifest)
        tokenizer = AutoTokenizer.from_pretrained(
            str(arguments.model), local_files_only=True, trust_remote_code=False
        )
        arms = materialize_codecontests_rows(manifest, rows, tokenizer)
    if arguments.formal:
        if arguments.run_card is None or any(
            value is not None
            for value in (arguments.arm, arguments.root, arguments.quota, arguments.updates, arguments.stop_after)
        ):
            parser.error("formal mode requires only --run-card, not diagnostic boundaries")
        card = json.loads(arguments.run_card.read_text("utf-8"))
        if arguments.s12_plan is not None:
            from ..arms.paired_filtering import load_s12_training_plan

            result = execute_codecontests_s12_formal(
                plan=load_s12_training_plan(arguments.s12_plan),
                card=card,
                model_path=arguments.model,
                output_dir=arguments.output,
                resume=arguments.resume,
            )
        else:
            assert arms is not None
            ordinal = int(card.get("training_ordinal", -1))
            frozen_cards = build_codecontests_run_cards(arms)
            if not 0 <= ordinal < len(frozen_cards) or card != frozen_cards[ordinal]:
                raise ValueError("CodeContests supplied formal run card differs from materialized freeze")
            result = execute_codecontests_formal(
                units=arms[str(card["arm"])],
                card=card,
                model_path=arguments.model,
                output_dir=arguments.output,
                resume=arguments.resume,
            )
    else:
        if any(value is None for value in (arguments.arm, arguments.root, arguments.quota, arguments.updates)):
            parser.error("diagnostic mode requires --arm, --root, --quota, and --updates")
        if arguments.run_card is not None:
            parser.error("--run-card is formal-only")
        if arguments.s12_plan is not None:
            parser.error("--s12-plan is formal-only")
        result = execute_codecontests_diagnostic(
            units=arms[arguments.arm],
            arm=arguments.arm,
            root=arguments.root,
            model_path=arguments.model,
            output_dir=arguments.output,
            quota=arguments.quota,
            updates=arguments.updates,
            stop_after=arguments.stop_after,
            resume=arguments.resume,
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "arm",
                    "root",
                    "completed_updates",
                    "response_tokens",
                    "trainable_tensor_sha256",
                    "finite",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_diagnostic_cli())
