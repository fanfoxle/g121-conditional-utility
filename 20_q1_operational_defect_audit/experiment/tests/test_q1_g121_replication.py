import hashlib
import json

import pytest


def test_formal_authorization_and_budget_are_separate_from_preparation():
    with pytest.raises(ValueError, match="authorization"):
        nat_spec.validate_g121_authorization({}, run_id="formal_test")
    approved = {"protocol_id":nat_spec.G121_PROTOCOL_ID,"run_id":"formal_test",
                "explicit_user_approval":True,"wall_cap_seconds":180000,
                "gpu_cap_seconds":165600,"states":32,"pairs":16,
                "current_window_available":True,"user_message":"approved exact scope",
                "approved_utc":"2026-09-18T12:00:00+00:00"}
    assert nat_spec.validate_g121_authorization(approved,run_id="formal_test") == approved
    with pytest.raises(ValueError):
        nat_spec.validate_g121_authorization(approved,run_id="different")
    assert nat_spec.g121_budget_admission(wall_elapsed_seconds=0,gpu_active_seconds=0,
        remaining_states=32,observed_wave_seconds=[])['admit']
    assert not nat_spec.g121_budget_admission(wall_elapsed_seconds=179000,gpu_active_seconds=0,
        remaining_states=1,observed_wave_seconds=[])['admit']


def test_repairs_preserve_trustworthy_upstream_and_pair_training_symmetry():
    pairs=nat_spec.derive_g121_pairs(excluded_values=set())
    assert nat_spec.g121_repair_scope(pairs,kind="score",state_id="R01_B") == (("score","R01_B"),)
    assert set(nat_spec.g121_repair_scope(pairs,kind="generation",state_id="R01_B")) == {("generation","R01_B"),("score","R01_B")}
    scope=nat_spec.g121_repair_scope(pairs,kind="training_computation",state_id="R01_BG")
    assert set(scope)=={(k,"R01_"+a) for k in ("train","generation","score") for a in ("B","BG")}


def test_scores_overlap_next_wave_without_gpu_mixing_or_global_oversubscription():
    pairs=nat_spec.derive_g121_pairs(excluded_values=set())
    states=nat_spec.g121_state_ids(pairs)
    completed={(k,s) for s in states[:4] for k in ("train","generation")}
    jobs=nat_spec.next_g121_jobs(pairs,completed=completed,active=set())
    assert sum(j['kind']=='score' for j in jobs)==4
    assert sum(j['kind']=='train' for j in jobs)==2
    active={(j['kind'],j['job_id']) for j in jobs}
    assert nat_spec.next_g121_jobs(pairs,completed=completed,active=active)==()


def test_missing_training_identity_prevents_complete_publication(tmp_path):
    import zipfile
    pairs=nat_spec.derive_g121_pairs(excluded_values=set())
    tasks=[f"HumanEval/{i}" for i in range(164)]
    rows=[dict(pair_id=p['pair_id'],state_id=p['pair_id']+'_'+a,init_seed=p['init_seed'],root=p['root'],
               arm=a,task_id=t,Y=0,verdict='fail',completion_sha256=None,missing_reason=None)
          for p in pairs for a in ('B','BG') for t in tasks]
    freeze={'contexts':list(pairs),'task_ids':tasks,'run_id':'test','scientific_identity_sha256':'a'*64,
            'acceptance':{'gpu_active_seconds':320,'wall_seconds':337}}
    (tmp_path/'EXECUTION_FREEZE.json').write_text(json.dumps(freeze))
    result=nat_spec.publish_g121(output_root=tmp_path,freeze=freeze,position_rows=rows,
        completed_training=set(nat_spec.g121_state_ids(pairs)) - {'R01_B'},repairs=[],
        gpu_active_seconds=10,cloud_wall_seconds=20)
    assert result['terminal']=='MEASUREMENT_INVALID' and not result['complete']
    assert result['valid_positions']==5084 and result['analysis']['mean_pp'] is None
    with zipfile.ZipFile(tmp_path/'DELIVERY.zip') as archive:
        assert archive.testzip() is None and len(archive.namelist())==5


def test_remote_guardian_enforces_persisted_deadline_without_training(tmp_path,monkeypatch):
    pytest.importorskip('fcntl')
    freeze={'runtime':{'python_judge':'/unused/python'},'started_utc':'2000-01-01T00:00:00+00:00',
            'deadline_utc':'2000-01-03T02:00:00+00:00'}
    monkeypatch.setattr(nat_spec,'prepare_g121_execution',lambda **kw:freeze)
    monkeypatch.setattr(nat_spec,'_g121_owned_processes',lambda _:[])
    monkeypatch.setattr(nat_spec,'_completed_w259_cf12_jobs',lambda *args:set())
    monkeypatch.setattr(nat_spec,'_w259_cf12_position_rows',lambda *args,**kw:[])
    monkeypatch.setattr(nat_spec,'_stop_nat_msep_process',lambda p:None)
    monkeypatch.setattr(nat_spec.subprocess,'Popen',lambda *args,**kw:object())
    monkeypatch.setattr(nat_spec.time,'sleep',lambda _:None)
    monkeypatch.setattr(nat_spec,'publish_g121',lambda **kw:kw)
    result=nat_spec.run_g121_detached(task_root=tmp_path/'task',output_root=tmp_path/'out',authorization_path=tmp_path/'none')
    assert result['forced_terminal']=='INCOMPLETE_RESOURCE_STOP'
    assert result['completed_training']==set()
    assert result['cloud_wall_seconds']>180000


def test_score_repair_keeps_training_generation_and_nonresetting_clock(tmp_path,monkeypatch):
    pairs=nat_spec.derive_g121_pairs(excluded_values=set())
    monkeypatch.setattr(nat_spec,'_load_w259_cf12_freeze',lambda _: {'protocol_id':nat_spec.G121_PROTOCOL_ID,'contexts':pairs})
    monkeypatch.setattr(nat_spec,'_g121_owned_processes',lambda _:[])
    for stage in ('training','generation','scores'):
        path=tmp_path/stage/'R01_B'
        path.mkdir(parents=True)
        (path/'evidence.json').write_text('{}')
    (tmp_path/'FORMAL_CLOCK.json').write_text('{"started":123}')
    result=nat_spec.repair_g121_stage(task_root=tmp_path,output_root=tmp_path/'out',
        kind='score',state_id='R01_B',reason='restore frozen scoring implementation')
    assert (tmp_path/'training/R01_B/evidence.json').is_file()
    assert (tmp_path/'generation/R01_B/evidence.json').is_file()
    assert not (tmp_path/'scores/R01_B').exists()
    assert (tmp_path/'FORMAL_CLOCK.json').read_text()=='{"started":123}'
    assert len(result['preserved'])==1 and not result['clock_reset']

from oasst1_filter.arms.paired_filtering import build_w259_factorial_plans
from oasst1_filter.runtime import nat_spec
from oasst1_filter.train.stream import Exposure, TrainingPlan, single_unit_packs


def test_fixed_g_plan_only_builds_requested_two_arms_and_keeps_original_blocks():
    def item(name, component):
        return Exposure(name, 0, (1, 2, 3), (False, False, True), 1, True, component)
    b0, b1, g, h = item("b0", "B"), item("b1", "B"), item("g", "S"), item("h", "S")
    bins = ((b1,), (b0, g, h))
    source = TrainingPlan(tuple(x for b in bins for x in b), bins, single_unit_packs(bins, max_length=2048), ())
    order = sorted(range(2), key=lambda i: (hashlib.sha256(f"q1-nat-onoff-354-v1|block-order|123|{i}".encode()).digest(), i))
    built = build_w259_factorial_plans(root=123, source_w_plan=source, source_block_order=(1, 0),
        group_by_unit={"g":"G","h":"H"},
        expected_block_order_sha256=hashlib.sha256(",".join(map(str,order)).encode()).hexdigest(), arms=("B","BG"))
    assert set(built["plans"]) == {"B", "BG"}
    assert [x.unit_id for x in built["plans"]["B"].exposures] == ["b0", "b1"]
    assert [x.unit_id for x in built["plans"]["BG"].exposures] == ["b0", "g", "b1"]
    assert all(x is b0 or x is b1 or x is g for x in built["plans"]["BG"].exposures)


def test_replication_seeds_collision_only_and_two_pair_wave_rounds():
    pairs = nat_spec.derive_g121_pairs(excluded_values={2026091201})
    assert [r["pair_id"] for r in pairs] == [f"R{i:02d}" for i in range(1,17)]
    assert len({r[k] for r in pairs for k in ("init_seed","root")}) == 32
    assert sum(r["first_arm"] == "B" for r in pairs) == 8
    first = 1 + int.from_bytes(hashlib.sha256(b"Q1-G121-FIXED-REPLICATION-16-v1|01|init|0").digest()[:8], "big") % (2**31-2)
    assert pairs[0]["init_seed"] == first
    collision = nat_spec.derive_g121_pairs(excluded_values={2026091201, first})
    assert collision[0]["init_counter"] == 1
    jobs = nat_spec.g121_jobs(pairs)
    assert len(jobs) == 96
    chosen = nat_spec.next_g121_jobs(pairs, completed=set(), active=set())
    assert [j["job_id"] for j in chosen] == [f'{p["pair_id"]}_{p["first_arm"]}' for p in pairs[:2]]
    active = {(j["kind"],j["job_id"]) for j in chosen}
    assert nat_spec.next_g121_jobs(pairs, completed=set(), active=active) == ()
    assert not any(j["kind"] == "generation" for j in nat_spec.next_g121_jobs(pairs, completed={next(iter(active))}, active=active-{next(iter(active))}))


def test_replication_run_card_binds_run_pair_arm_and_initial_theta():
    from oasst1_filter.train.run_codecontests import validate_nat_onoff_run_card, w259_training_run_identity
    card = {"schema":"q1-g121-run-card-v1","protocol_id":nat_spec.G121_PROTOCOL_ID,
            "run_id":"formal_test", "pair_id":"R01", "context_id":"R01", "init_seed":101,"root":202,
            "arm":"BG","response_tokens":623689,"updates":100,"max_length":2048,
            "plan_sha256":"a"*64,"expected_theta_i_sha256":"b"*64,
            "initial_adapter_path":"/initial/R01/adapter","outcomes_opened":False}
    def seal(c):
        c = {k:v for k,v in c.items() if k != "card_sha256"}
        c["card_sha256"] = hashlib.sha256(json.dumps(c,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        return c
    card = seal(card)
    assert validate_nat_onoff_run_card(card) == card["card_sha256"]
    identity = w259_training_run_identity(card)
    for change in ({"run_id":"other"}, {"pair_id":"R02","context_id":"R02"}, {"arm":"B","response_tokens":583975}):
        assert w259_training_run_identity(seal({**card,**change})) != identity
    with pytest.raises(ValueError):
        validate_nat_onoff_run_card(seal({**card,"pair_id":"R02"}))
    with pytest.raises(ValueError):
        validate_nat_onoff_run_card(seal({**card,"arm":"BH","response_tokens":623690}))


def test_replication_initial_states_require_fresh_distinct_pair_receipts():
    from oasst1_filter.train.run_codecontests import validate_w259_context_initial_state_set
    receipts = [{"schema":"q1-g121-initial-state-v1","protocol_id":nat_spec.G121_PROTOCOL_ID,
                 "context_id":f"R{i:02d}","init_seed":i,"theta_i_sha256":str(i)*64,"lora_b_zero":True,
                 "adapter_files":[{"path":"adapter.safetensors","sha256":"a"*64}]} for i in (1,2)]
    assert validate_w259_context_initial_state_set(receipts,expected_contexts={"R01":1,"R02":2},protocol_id=nat_spec.G121_PROTOCOL_ID) == {"R01":"1"*64,"R02":"2"*64}
    receipts[1]["theta_i_sha256"] = "1"*64
    with pytest.raises(ValueError):
        validate_w259_context_initial_state_set(receipts,expected_contexts={"R01":1,"R02":2},protocol_id=nat_spec.G121_PROTOCOL_ID)


def test_replication_analysis_fixed_16_and_missing_not_zero():
    from oasst1_filter.analysis import codecontests
    pairs = nat_spec.derive_g121_pairs(excluded_values={2026091201})
    tasks = [f"HumanEval/{i}" for i in range(164)]
    rows = [{"pair_id":p["pair_id"], "root":p["root"], "init_seed":p["init_seed"],"arm":a,"task_id":t,
             "Y": int(a=="B" and j < (3 if i<11 else 4))}
            for i,p in enumerate(pairs) for a in ("B","BG") for j,t in enumerate(tasks)]
    result = codecontests.analyze_g121_positions(rows,pairs=pairs,task_ids=tasks)
    assert result["net_pass"] == 53
    assert result["terminal"] == "POSITIVE_CONDITIONAL_REPLICATION"
    assert result["mean_pp"] == pytest.approx(100*53/2624)
    assert result["sample_sd_pp"] == pytest.approx((55/240)**.5 *100/164)
    assert result["gain"] == 53 and result["loss"] == 0
    assert result["ci95_pp"][0] > 0
    rows[0]["Y"] = None
    missing = codecontests.analyze_g121_positions(rows,pairs=pairs,task_ids=tasks)
    assert missing["complete"] is False
    assert missing["terminal"] == "MEASUREMENT_INVALID" and missing["mean_pp"] is None
    with pytest.raises(ValueError):
        codecontests.analyze_g121_positions(rows+rows[:1],pairs=pairs,task_ids=tasks)


@pytest.mark.parametrize("mean,sd,lo,hi,expected", [
    (2,0,2,2,"UNCERTAINTY_DEGENERATE"),(-2,1,-3,-1,"DIRECTION_REVERSED"),
    (2,1,.1,3.9,"POSITIVE_CONDITIONAL_REPLICATION"),(1,1,.01,1.99,"POSITIVE_BELOW_CONTINUATION_SCALE"),
    (1,1,0,1.99,"NO_SUPPORT_AT_CONTINUATION_SCALE"),(1,1,0,2,"WITHIN_BUDGET_UNRESOLVED")])
def test_replication_terminal_strict_priority(mean,sd,lo,hi,expected):
    from oasst1_filter.analysis.codecontests import resolve_g121_terminal
    assert resolve_g121_terminal(complete=True,mean=mean,sd=sd,lower=lo,upper=hi) == expected
