"""Acceptance checks for selection configs, consumed laws and actual checkpoints."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from tabular_foundation.train import run, validate_config, macroepisode_task_counts
from tabular_foundation.producer import TaskProducer, prepare_task
from tabular_foundation.runtime import profile_schedule

ROOT=Path(__file__).resolve().parents[1]

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

materializer=load_script('materialize_prior_selection')
submitter=load_script('submit_training')

@pytest.fixture
def spec():
    return json.loads((ROOT/'research/specs/prior_selection_v1.json').read_text())

@pytest.fixture
def cfg():
    c=json.loads((ROOT/'configs/smoke_selection.json').read_text())
    c.update(producer_workers=0, steps=2, stage_end_steps=[1,2,2], global_batch=1)
    # Deliberately avoid upstream dependency: two new mechanism samples retain
    # the actual common response and observation implementation.
    c['selection_options']['mechanism_weights']={'R':0,'forest':.5,'hierarchy':0,'smooth_local':.5,'sparse_interaction':0}
    return c


def test_all_declared_configs_validate_and_volume_targets_match():
    for path in (ROOT/'configs').glob('*.json'):
        if path.name == 'data_schema.example.json':continue
        validate_config(json.loads(path.read_text()))
    c=json.loads((ROOT/'configs/selection_candidate_standard.json').read_text())
    f=json.loads((ROOT/'configs/selection_candidate_fraud.json').read_text())
    assert c['steps']*c['global_batch']==64_000_000
    assert f['steps']*f['global_batch']==1_024_000
    assert c['gpu_hour_cap']+f['gpu_hour_cap']==12_000
    assert f['initial_checkpoint_required'] is True


def test_materializer_P_is_runnable_and_later_phases_require_decisions(spec,tmp_path):
    result=materializer.materialize(spec,'P',steps=100,output=tmp_path/'P')
    assert len(result['runs'])==16
    assert result['total_gpu_hour_ceiling']==4800
    configs=[json.loads(Path(r['config']).read_text()) for r in result['runs']]
    assert {c['seed'] for c in configs}=={0,1}
    assert all(c['standard_prior']=='selection' and c['stage_end_steps']==[90,99,100] for c in configs)
    assert all(c['reference_envelope']==configs[0]['reference_envelope'] for c in configs)
    for phase in ('M','A','V','F','FINAL_STANDARD','FINAL_FRAUD'):
        with pytest.raises(ValueError,match='requires reviewed'):
            materializer.materialize(spec,phase,steps=400 if phase in ('F','FINAL_FRAUD') else 100,output=tmp_path/phase)
        assert not (tmp_path/phase).exists()
    with pytest.raises(FileExistsError):
        materializer.materialize(spec,'P',steps=100,output=tmp_path/'P')


def decision(spec):
    return {'reviewed':True,'evidence':'pilot-results/report.json',
            'mechanism_weights':spec['mechanism_weights'],
            'observation_weights':spec['observation_weights'],'model':'persistent_small'}


def test_materializer_reviewed_M_A_V_preserves_pairing_and_fresh_confirmation(spec,tmp_path):
    choices={name:decision(spec) for name in ('after_P','after_M','after_A')}
    for phase in ('M','A','V'):
        manifest=materializer.materialize(spec,phase,steps=200,output=tmp_path/phase,decisions=choices)
        configs=[json.loads(Path(r['config']).read_text()) for r in manifest['runs']]
        if phase=='M':
            assert {round(c['selection_options']['mechanism_weights']['R'],2) for c in configs}=={.3,.5,.7}
            assert {c['seed'] for c in configs}=={2,3}
        if phase=='A':assert {c['model'] for c in configs}=={'base','persistent_small'}
        if phase=='V':assert {c['seed'] for c in configs}=={4,5}
        assert all(c['stage_end_steps']==[180,198,200] for c in configs)
    with pytest.raises(ValueError,match='existing parent_checkpoint'):
        materializer.materialize(spec,'F',steps=400,output=tmp_path/'F',decisions={'after_V':decision(spec)})


def test_F_uses_one_actual_parent_and_explicit_selector_control(spec,tmp_path):
    chosen=decision(spec)
    chosen['model']='base'
    # Materialization checks checkpoint identity/config; weight loading is tested
    # below with a real tiny trained checkpoint.
    parent_cfg=json.loads((ROOT/'configs/selection_candidate_standard.json').read_text())
    parent=tmp_path/'parent.pt'
    torch.save({'config':parent_cfg,'model':{},'step':10},parent)
    chosen['parent_checkpoint']=str(parent)
    manifest=materializer.materialize(spec,'F',steps=400,output=tmp_path/'F',decisions={'after_V':chosen})
    configs={r['id']:json.loads(Path(r['config']).read_text()) for r in manifest['runs']}
    assert len(configs)==6
    assert {c['initial_checkpoint'] for c in configs.values()}=={str(parent)}
    for key,c in configs.items():
        assert c['adam_lr']==pytest.approx(3e-5)
        if 'standard_replay' in key:assert c['finance_share']==0
        else:
            assert c['finance_share']==1 and c['finance_overrides']['prevalence']==.0001
            assert c['finance_overrides']['feature_view']=='hide_h'
            assert c['finance_overrides'].get('local_cap')==(0 if 'reservoir' in key else None)


def test_scheduler_cap_is_derived_from_remaining_allocation(tmp_path):
    c=json.loads((ROOT/'configs/selection_candidate_standard.json').read_text())
    c.update(gpu_hour_cap=64,project_ledger=str(tmp_path/'ledger.json'))
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    args,env=submitter.command(path,tmp_path/'run')
    assert args[args.index('--time')+1]=='0-01:00:00'
    assert args[args.index('--nodes')+1]=='8'
    assert env['TFM_GPUS_PER_NODE']=='8'
    saved={'config':c,'run_id':str(tmp_path/'run'),'cumulative_gpu_hours':16}
    checkpoint=tmp_path/'saved.pt';torch.save(saved,checkpoint)
    (tmp_path/'ledger.json').write_text(json.dumps({'runs':{str(tmp_path/'run'):32},'total_gpu_hours':32}))
    args,env=submitter.command(path,tmp_path/'run',resume=checkpoint)
    assert args[args.index('--time')+1]=='0-00:30:00'
    assert env['TFM_RESUME']==str(checkpoint)
    with pytest.raises(ValueError,match='cannot override'):
        submitter.command(path,tmp_path/'run',sbatch_args=['--time=999'])


def test_selection_training_checkpoint_resume_and_actual_exposure(cfg,tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED','0')
    import tabular_foundation.train as trainer
    first=tmp_path/'first.pt'
    original=trainer._save_checkpoint
    def capture(path,model,opts,config,step,attempt,hours):
        original(path,model,opts,config,step,attempt,hours)
        if step==1 and not first.exists():shutil.copyfile(path,first)
    monkeypatch.setattr(trainer,'_save_checkpoint',capture)
    trained=run(cfg,tmp_path/'whole')
    restored=run(cfg,tmp_path/'resume',resume=first)
    for key,value in trained.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
    records=[json.loads(line) for line in (tmp_path/'whole'/'metrics.jsonl').read_text().splitlines()]
    assert sum(r['macroepisodes'] for r in records)==2
    for row in records:
        counts=row['consumed_task_counts']
        assert sum(counts[f'observation_requested_{k}'] for k in ('identity','mcar','mar','mnar','coarsen'))==1
        assert counts['forest']+counts['smooth_local']==1
    checkpoint=torch.load(tmp_path/'whole'/'last.pt',weights_only=True)
    assert checkpoint['config']['selection_options']==cfg['selection_options']
    child=copy.deepcopy(cfg);child.update(initial_checkpoint=str(tmp_path/'whole'/'last.pt'),adam_lr=0.,muon_lr=0.,steps=1,stage_end_steps=[1,1,1])
    initialized=run(child,tmp_path/'child')
    for key,value in trained.state_dict().items():
        torch.testing.assert_close(value,initialized.state_dict()[key],rtol=0,atol=0)


def test_multiple_producer_processes_preserve_episode_order_and_values(cfg,monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED','0')
    entries=profile_schedule(cfg['seed'],0,4,0)
    expected=[prepare_task(cfg,e,1) for e in entries]
    producer=TaskProducer(dict(cfg,producer_workers=2,prefetch_tasks=3))
    try:
        assert producer.pool._max_workers==2
        actual=list(producer.ordered(entries,1))
    finally:producer.close()
    assert [e for e,_,_ in actual]==entries
    for (_,got,_),want in zip(actual,expected):
        for a,b in zip(got[0],want[0]):
            for field in ('x_support','y_support','x_query','y_query'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
            assert a.metadata['selection']==b.metadata['selection']


def test_invalid_config_and_missing_continuation_parent_rejected(cfg,tmp_path):
    for bad in (-1,True,1.5):
        with pytest.raises(ValueError,match='producer_workers'):validate_config(dict(cfg,producer_workers=bad))
    bad=copy.deepcopy(cfg);bad['selection_options']['mechanism_weights']['R']=.1
    with pytest.raises(ValueError,match='simplex'):validate_config(bad)
    with pytest.raises(ValueError,match='requires --initialize-from'):
        run(dict(cfg,initial_checkpoint_required=True),tmp_path/'bad')


def test_research_materialization_rejects_rounded_curricula(spec,tmp_path):
    with pytest.raises(ValueError,match='multiple of 100'):
        materializer.materialize(spec,'P',steps=123,output=tmp_path/'P')
    with pytest.raises(ValueError,match='multiple of 400'):
        materializer.materialize(spec,'F',steps=100,output=tmp_path/'F')


def test_scheduler_also_caps_to_unspent_project_allocation(tmp_path):
    c=json.loads((ROOT/'configs/selection_candidate_standard.json').read_text())
    c.update(gpu_hour_cap=64,project_ledger=str(tmp_path/'ledger.json'))
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    (tmp_path/'ledger.json').write_text(json.dumps({'runs':{'other':49968},'total_gpu_hours':49968,'cap_gpu_hours':50000}))
    args,env=submitter.command(path,tmp_path/'run')
    assert args[args.index('--time')+1]=='0-00:30:00'
    assert args[args.index('--cpus-per-task')+1]=='24'
    assert env['TFM_PYTHON'].endswith('python') or 'python' in Path(env['TFM_PYTHON']).name


def test_finance_selector_controls_keep_population_and_query_draws_paired():
    from tabular_foundation.finance_prior import generate_finance_world
    common=dict(history_size=800,future_size=320,prevalence=.1,width=16,
                complete_adjudication=True,reservoir_cap=24,positive_cap=12,
                candidate_cap=80,query_count=32,calibration_rows=64,
                mechanism_family='risk_partition',feature_view='hide_h',loss_normalization='unit')
    worlds=[generate_finance_world(32,task='binary',overrides=dict(common,**control)) for control in
            ({'local_cap':16,'max_centers':4,'max_positive_centers':2},
             {'local_cap':0,'max_centers':1,'max_positive_centers':0},
             {'local_cap':0,'max_centers':1,'max_positive_centers':0,'scan_chunk_size':7,'index_dimension':16})]
    assert len({w.population_id for w in worlds})==1
    assert len({w.world_id for w in worlds})==3
    macros=[w.sample_macroepisode(711) for w in worlds]
    for macro in macros[1:]:
        np.testing.assert_array_equal(macro.audit['finite_query_ids'],macros[0].audit['finite_query_ids'])
        np.testing.assert_array_equal(macro.audit['query_weights'],macros[0].audit['query_weights'])
    assert all(len(route.x_support)<=36 for route in macros[1].routes)


def test_measured_per_trial_horizons_are_exhaustive_and_keep_stage_shares(spec,tmp_path):
    trials=[t for t in materializer.build_plan(spec)['trials'] if t['phase']=='P']
    horizons={t['id']:100*(i+1) for i,t in enumerate(trials)}
    result=materializer.materialize(spec,'P',horizons=horizons,output=tmp_path/'good')
    assert result['horizon_policy']=='measured_per_trial'
    for run in result['runs']:
        c=json.loads(Path(run['config']).read_text());n=horizons[run['id']]
        assert c['steps']==run['steps']==n
        assert c['stage_end_steps']==[n*90//100,n*99//100,n]
        assert run['accepted_macroepisode_target']==n*256
    for i,bad in enumerate(({**horizons,'unknown':100},dict(list(horizons.items())[:-1]),[],{**horizons,trials[0]['id']:125})):
        with pytest.raises(ValueError):
            materializer.materialize(spec,'P',horizons=bad,output=tmp_path/str(i))
        assert not (tmp_path/str(i)).exists()
    with pytest.raises(ValueError,match='not both'):
        materializer.materialize(spec,'P',steps=100,horizons=horizons,output=tmp_path/'ambiguous')


def test_evaluation_reserve_is_separate_from_consumed_time_and_conserves_budget(spec,tmp_path):
    result=materializer.materialize(spec,'P',steps=100,output=tmp_path/'reserved',evaluation_reserve_gpu_hours=25)
    assert result['total_gpu_hour_ceiling']==4800
    assert result['training_gpu_hour_ceiling']==4400
    assert result['evaluation_reserve_gpu_hours']==400
    assert result['automatic_evaluation_charging'] is False
    assert result['evaluation_reserve_status']=='explicitly_reserved'
    for run in result['runs']:
        c=json.loads(Path(run['config']).read_text())
        assert c['gpu_hour_cap']==run['training_gpu_hour_ceiling']==275
        assert run['package_gpu_hour_ceiling']==run['gpu_hour_ceiling']==300
        assert run['evaluation_reserve_gpu_hours']==25
        assert Path(run['evaluation_allocation_id']).name==run['id']
    for i,bad in enumerate((True,-1,float('nan'),300,301)):
        with pytest.raises(ValueError):
            materializer.materialize(spec,'P',steps=100,output=tmp_path/f'bad{i}',evaluation_reserve_gpu_hours=bad)


def test_recorded_allocation_is_cumulative_idempotent_and_keeps_evaluation_separate(tmp_path):
    recorder=load_script('record_allocation')
    ledger=tmp_path/'ledger.json'
    a=recorder.record(ledger,'train-run',30,cap_gpu_hours=100)
    assert a['total_gpu_hours']==30
    assert recorder.record(ledger,'train-run',30)['total_gpu_hours']==30
    assert recorder.record(ledger,'eval-run',5)['total_gpu_hours']==35
    assert recorder.record(ledger,'train-run',34)['total_gpu_hours']==39
    with pytest.raises(ValueError,match='backwards'):
        recorder.record(ledger,'train-run',33)
    with pytest.raises(ValueError,match='existing project cap'):
        recorder.record(ledger,'train-run',34,cap_gpu_hours=200)
    with pytest.raises(ValueError,match='finite and nonnegative'):
        recorder.record(ledger,'train-run',float('nan'))
    over=recorder.record(ledger,'other-run',70)
    assert over['total_gpu_hours']==109 and not over['within_project_cap']
    assert json.loads(ledger.read_text())['runs']['eval-run']==5
