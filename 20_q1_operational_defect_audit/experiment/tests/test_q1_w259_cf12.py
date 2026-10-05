"""Frozen public behavior seams for Q1-W259-CF12."""

from __future__ import annotations


def _exposure(name: str, response_tokens: int, component: str):
    from oasst1_filter.train.stream import Exposure

    return Exposure(
        unit_id=name,
        cycle=0,
        token_ids=tuple(range(response_tokens + 2)),
        loss_mask=(False, False) + (True,) * response_tokens,
        retained_response_tokens=response_tokens,
        complete=True,
        component=component,
    )


def test_cf12_plan_filters_locked_complements_without_rebinning() -> None:
    from oasst1_filter.arms.paired_filtering import build_w259_factorial_plans
    from oasst1_filter.train.stream import TrainingPlan, single_unit_packs

    canonical = (
        (_exposure("b0", 2, "B"), _exposure("g0", 1, "S"), _exposure("h0", 3, "S")),
        (_exposure("b1", 4, "B"),),
        (_exposure("b2", 5, "B"), _exposure("g2", 2, "S")),
    )
    source_order = (1, 0, 2)
    source_bins = tuple(canonical[index] for index in source_order)
    source = TrainingPlan(
        tuple(item for block in source_bins for item in block),
        source_bins,
        single_unit_packs(source_bins, max_length=2048),
        (),
    )

    built = build_w259_factorial_plans(
        root=123,
        source_w_plan=source,
        source_block_order=source_order,
        group_by_unit={"g0": "G", "g2": "G", "h0": "H"},
        expected_block_order_sha256=(
            "cea67d8b58d72bb9387778326578aa3e58971aeb9f43ff6eaa599fbdeb5a3103"
        ),
    )

    assert built["block_order"] == (2, 0, 1)
    assert built["component_exposures"] == {"B": 3, "G": 2, "H": 1}
    assert built["component_tokens"] == {"B": 11, "G": 3, "H": 3}
    assert {
        arm: [[item.unit_id for item in block] for block in plan.update_bins]
        for arm, plan in built["plans"].items()
    } == {
        "B": [["b2"], ["b0"], ["b1"]],
        "BG": [["b2", "g2"], ["b0", "g0"], ["b1"]],
        "BH": [["b2"], ["b0", "h0"], ["b1"]],
        "BW": [["b2", "g2"], ["b0", "g0", "h0"], ["b1"]],
    }
    assert [
        sum(item.retained_response_tokens for item in plan.exposures)
        for plan in built["plans"].values()
    ] == [11, 14, 14, 17]
    assert all(
        {id(item) for item in plan.exposures}
        <= {id(item) for item in source.exposures}
        for plan in built["plans"].values()
    )


def test_cf12_context_theta_and_recovery_identity_bind_all_four_arms() -> None:
    import hashlib
    import json

    import pytest
    from oasst1_filter.train.run_codecontests import (
        validate_nat_onoff_run_card,
        validate_w259_context_initial_state_set,
        validate_w259_pre_update_facts,
        w259_training_run_identity,
    )

    receipts = (
        {
            "schema": "q1-w259-cf12-initial-state-v1",
            "protocol_id": "q1-w259-cf12-v1",
            "context_id": "C01",
            "init_seed": 101,
            "theta_i_sha256": "1" * 64,
            "lora_b_zero": True,
            "adapter_files": [{"path": "adapter_config.json", "sha256": "a" * 64}],
        },
        {
            "schema": "q1-w259-cf12-initial-state-v1",
            "protocol_id": "q1-w259-cf12-v1",
            "context_id": "C02",
            "init_seed": 202,
            "theta_i_sha256": "2" * 64,
            "lora_b_zero": True,
            "adapter_files": [{"path": "adapter_config.json", "sha256": "b" * 64}],
        },
    )
    assert validate_w259_context_initial_state_set(
        receipts, expected_contexts={"C01": 101, "C02": 202}
    ) == {"C01": "1" * 64, "C02": "2" * 64}

    def card(context: str, arm: str, tokens: int, suffix: str) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "q1-w259-cf12-run-card-v1",
            "protocol_id": "q1-w259-cf12-v1",
            "context_id": context,
            "root": 303 if context == "C01" else 404,
            "arm": arm,
            "init_seed": 101 if context == "C01" else 202,
            "updates": 100,
            "max_length": 2048,
            "plan_sha256": suffix * 64,
            "response_tokens": tokens,
            "expected_theta_i_sha256": "1" * 64 if context == "C01" else "2" * 64,
            "initial_adapter_path": f"/task/initial/{context}/adapter",
            "outcomes_opened": False,
        }
        value["card_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value

    cards = (
        card("C01", "B", 583975, "a"),
        card("C01", "BG", 623689, "b"),
        card("C01", "BH", 623690, "c"),
        card("C01", "BW", 663404, "d"),
    )
    identities = [
        w259_training_run_identity(value, validate_nat_onoff_run_card(value))
        for value in cards
    ]
    assert len(set(identities)) == 4
    other_context = card("C02", "B", 583975, "e")
    assert w259_training_run_identity(
        other_context, validate_nat_onoff_run_card(other_context)
    ) not in identities

    wrong_arm = dict(cards[0], arm="BG")
    wrong_arm["card_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in wrong_arm.items() if key != "card_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="recipe"):
        validate_nat_onoff_run_card(wrong_arm)

    validate_w259_pre_update_facts(
        theta_i_sha256="1" * 64,
        expected_theta_i_sha256="1" * 64,
        lora_b_nonzero_count=0,
        optimizer_state_count=0,
    )


def test_cf12_registry_six_wave_queue_and_dual_budget_are_frozen() -> None:
    import pytest
    from oasst1_filter.runtime.nat_spec import (
        derive_w259_cf12_contexts,
        next_w259_cf12_jobs,
        w259_cf12_budget_admission,
        w259_cf12_jobs,
        w259_cf12_state_ids,
    )

    contexts = derive_w259_cf12_contexts(excluded_values=set())
    assert contexts[0] == {
        "context_id": "C01",
        "init_seed": 1231984934,
        "init_k": 0,
        "root": 1949682730,
        "root_k": 0,
    }
    assert contexts[-1] == {
        "context_id": "C12",
        "init_seed": 699503101,
        "init_k": 0,
        "root": 802365995,
        "root_k": 0,
    }
    states = w259_cf12_state_ids(contexts)
    assert states[:8] == (
        "C01_BW", "C01_BH", "C01_BG", "C01_B",
        "C02_BH", "C02_BG", "C02_B", "C02_BW",
    )
    assert len(states) == 48
    assert all(
        sum(state.endswith(f"_{arm}") for state in states[index::4]) == 3
        for index in range(4)
        for arm in ("B", "BG", "BH", "BW")
    )
    jobs = w259_cf12_jobs(contexts)
    assert len(jobs) == 144

    concurrency = {"train": 2, "generation": 4, "score": 4}
    first = next_w259_cf12_jobs(
        contexts, completed=set(), active=set(), concurrency=concurrency
    )
    assert first == (
        {"kind": "train", "job_id": "C01_BW"},
        {"kind": "train", "job_id": "C01_BH"},
    )
    first_wave = states[:8]
    seven_trained = {("train", state) for state in first_wave[:7]}
    assert next_w259_cf12_jobs(
        contexts, completed=seven_trained, active=set(), concurrency=concurrency
    ) == ({"kind": "train", "job_id": first_wave[7]},)
    trained = {("train", state) for state in first_wave}
    generation = next_w259_cf12_jobs(
        contexts, completed=trained, active=set(), concurrency=concurrency
    )
    assert generation == tuple(
        {"kind": "generation", "job_id": state} for state in first_wave[:4]
    )
    active_generation = {(job["kind"], job["job_id"]) for job in generation}
    assert not any(
        job["kind"] == "train"
        for job in next_w259_cf12_jobs(
            contexts,
            completed=trained,
            active=active_generation,
            concurrency=concurrency,
        )
    )
    closed = trained | {
        (kind, state) for state in first_wave for kind in ("generation", "score")
    }
    assert next_w259_cf12_jobs(
        contexts, completed=closed, active=set(), concurrency=concurrency
    )[0] == {"kind": "train", "job_id": states[8]}

    admission = w259_cf12_budget_admission(
        wall_elapsed_seconds=0.0,
        gpu_active_seconds=0.0,
        remaining_train_states=48,
        remaining_e2_states=48,
        completed_train_hours_per_state=(),
        completed_e2_hours_per_state=(),
        concurrency=concurrency,
    )
    assert admission["admit"] is True
    assert admission["wall_required_hours"] == pytest.approx(47.8555353)
    assert admission["gpu_required_hours"] == pytest.approx(42.8555353)
    assert w259_cf12_budget_admission(
        wall_elapsed_seconds=65.9 * 3600,
        gpu_active_seconds=1.0,
        remaining_train_states=1,
        remaining_e2_states=1,
        completed_train_hours_per_state=(),
        completed_e2_hours_per_state=(),
        concurrency=concurrency,
    )["reason"] == "WALL_WIND_DOWN"
    with pytest.raises(ValueError, match="unknown"):
        w259_cf12_budget_admission(
            wall_elapsed_seconds=1.0,
            gpu_active_seconds=None,
            remaining_train_states=1,
            remaining_e2_states=1,
            completed_train_hours_per_state=(),
            completed_e2_hours_per_state=(),
            concurrency=concurrency,
        )


def test_cf12_analysis_uses_twelve_contexts_six_literal_contrasts_and_separate_labels() -> None:
    import pytest
    from oasst1_filter.analysis.codecontests import (
        analyze_w259_cf12_positions,
        resolve_w259_cf12_main,
    )
    from oasst1_filter.runtime.nat_spec import derive_w259_cf12_contexts

    contexts = derive_w259_cf12_contexts(excluded_values=set())
    tasks = tuple(f"HumanEval/{index}" for index in range(164))
    pass_counts: dict[str, dict[str, int]] = {}
    rows: list[dict[str, object]] = []
    for index, context in enumerate(contexts):
        counts = {
            "B": 100,
            "BG": 90 + index,
            "BH": 95 + index % 3,
            "BW": 85 + index % 4,
        }
        context_id = str(context["context_id"])
        pass_counts[context_id] = counts
        for arm, passed in counts.items():
            for task_index, task_id in enumerate(tasks):
                rows.append(
                    {
                        "context_id": context_id,
                        "root": context["root"],
                        "arm": arm,
                        "task_id": task_id,
                        "Y": int(task_index < passed),
                    }
                )

    analysis = analyze_w259_cf12_positions(
        rows, contexts=contexts, task_ids=tasks
    )
    assert analysis["complete"] is True
    assert analysis["context_count"] == 12
    assert analysis["position_count"] == 7872
    assert analysis["t_critical_95"] == pytest.approx(2.200985160082949)
    assert analysis["t_critical_bonferroni_97_5"] == pytest.approx(2.593092682539353)
    first = analysis["root_rows"][0]
    assert {
        name: first[name]
        for name in ("hG_pp", "hH_pp", "cG_pp", "cH_pp", "hW_pp", "J_pp")
    } == pytest.approx(
        {
            "hG_pp": 1000 / 164,
            "hH_pp": 500 / 164,
            "cG_pp": 1000 / 164,
            "cH_pp": 500 / 164,
            "hW_pp": 1500 / 164,
            "J_pp": 0.0,
        }
    )
    assert first["J_pp"] == pytest.approx(
        first["hW_pp"] - first["hG_pp"] - first["hH_pp"]
    )
    assert first["J_pp"] == pytest.approx(first["cG_pp"] - first["hG_pp"])
    assert first["J_pp"] == pytest.approx(first["cH_pp"] - first["hH_pp"])
    assert set(analysis["contrasts"]) == {"hG", "hH", "cG", "cH", "hW", "J"}
    assert analysis["contrasts"]["hH"]["n"] == 12
    assert analysis["main_labels"]["hH"] == {
        "scientific": "POSITIVE_EXPLORATORY",
        "investment": "ELIGIBLE_FOR_REVIEW",
    }
    assert analysis["terminal"] == "CF12_SUBGROUP_SUPPORT_EXPLORATORY"
    assert resolve_w259_cf12_main(
        complete=True,
        all_zero_or_full=False,
        mean=1.0,
        sd=0.0,
        lower=None,
        upper=None,
    ) == {
        "scientific": "UNRESOLVED_DEGENERATE_VARIANCE",
        "investment": "NO",
    }

    rows[0]["Y"] = None
    incomplete = analyze_w259_cf12_positions(rows, contexts=contexts, task_ids=tasks)
    assert incomplete["complete"] is False
    assert incomplete["main_labels"] is None
    assert incomplete["terminal"] is None
    assert incomplete["contrasts"]["hG"]["missingness_bound_pp"] is not None


def test_cf12_publisher_emits_exact_five_file_archive(tmp_path) -> None:
    import json
    import zipfile
    from oasst1_filter.runtime.nat_spec import (
        derive_w259_cf12_contexts,
        publish_w259_cf12,
    )

    contexts = derive_w259_cf12_contexts(excluded_values=set())
    tasks = [f"HumanEval/{index}" for index in range(164)]
    rows = []
    for context in contexts:
        for arm in ("B", "BG", "BH", "BW"):
            for index, task_id in enumerate(tasks):
                rows.append(
                    {
                        "state_id": f"{context['context_id']}_{arm}",
                        "context_id": context["context_id"],
                        "root": context["root"],
                        "arm": arm,
                        "task_id": task_id,
                        "Y": int(index < 100),
                        "verdict": "pass" if index < 100 else "fail",
                        "completion_sha256": f"{index:064x}"[-64:],
                        "missing_reason": None,
                    }
                )
    freeze = {
        "protocol_id": "q1-w259-cf12-v1",
        "scientific_identity_sha256": "f" * 64,
        "contexts": list(contexts),
        "task_ids": tasks,
    }
    (tmp_path / "EXECUTION_FREEZE.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )
    result = publish_w259_cf12(
        output_root=tmp_path,
        freeze=freeze,
        position_rows=rows,
        phase_seconds={"E1": 1.0, "E2": 2.0},
        effective_concurrency={"train": 2, "generation": 4, "score": 4},
        repairs=(),
        gpu_active_seconds=3.0,
        cloud_wall_seconds=4.0,
    )
    assert result["expected_positions"] == 7872
    with zipfile.ZipFile(tmp_path / "DELIVERY.zip") as archive:
        assert set(archive.namelist()) == {
            "EXECUTION_FREEZE.json", "POSITION_TABLE.csv", "ROOT_TABLE.csv",
            "RESULT.json", "REPORT.md",
        }
