import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tabular_foundation.producer import TaskProducer, prepare_task
from tabular_foundation.runtime import profile_schedule
from tabular_foundation.train import run, validate_config
from tabular_foundation.inference import load_checkpoint, Predictor


def config():
    return dict(model='tiny', device='cpu', cpu_threads=1, seed=1729,
                steps=2, global_batch=2, finance_share=0, finance_adapter=False,
                optimizer='adamw', adam_lr=.0003, bf16=False, checkpoint_every=1,
                static_overrides=dict(n_support=16,n_query=8,n_features=8),
                model_options=dict(categorical_encoding='hash_bits',regression_head='quantile',
                                   regression_bins=1025,regression_quantiles=999,
                                   length_scaling='logarithmic'))


def test_ordered_process_generation_equals_serial():
    cfg=config()
    entries=profile_schedule(cfg['seed'],0,3,0)
    expected=[prepare_task(cfg,e,1) for e in entries]
    producer=TaskProducer(dict(cfg,producer_workers=1,prefetch_tasks=2))
    try:
        actual=list(producer.ordered(entries,1))
    finally:
        producer.close()
    assert [e for e,_,_ in actual]==entries
    for (_,got,_), want in zip(actual,expected):
        assert got[1:3]==want[1:3]
        for a,b in zip(got[0],want[0]):
            assert a.task==b.task
            for field in ('x_support','y_support','x_query','y_query'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))


def test_prefetch_training_preserves_weights_and_checkpoint_options(tmp_path):
    cfg=config()
    serial=run(cfg,tmp_path/'serial')
    prefetched=run(dict(cfg,producer_workers=1,prefetch_tasks=2),tmp_path/'prefetched')
    for key,value in serial.state_dict().items():
        torch.testing.assert_close(value,prefetched.state_dict()[key],rtol=0,atol=0)
    restored,saved=load_checkpoint(tmp_path/'prefetched'/'last.pt')
    assert restored.config.regression_head=='quantile'
    assert restored.config.categorical_encoding=='hash_bits'
    episode=prepare_task(cfg,profile_schedule(1729,5,1,0)[0],1)[0][0]
    predictor=Predictor(restored,bf16=False).fit_context(
        episode.x_support,episode.y_support,episode.categorical,episode.task,episode.n_classes)
    result=predictor.predict(episode.x_query)
    assert np.isfinite(result).all()
    metrics=[json.loads(x) for x in (tmp_path/'prefetched'/'metrics.jsonl').read_text().splitlines()]
    assert all(row['rank0_input_wait_seconds']>=0 for row in metrics)


def test_reference_requires_explicit_hash_seed(monkeypatch):
    monkeypatch.delenv('PYTHONHASHSEED',raising=False)
    with pytest.raises(ValueError,match='PYTHONHASHSEED'):
        TaskProducer(dict(config(),standard_prior='R'))


def test_reference_overrides_and_unknown_fields_rejected():
    with pytest.raises(ValueError,match='reference_shape'):
        validate_config(dict(config(),standard_prior='R'))
    with pytest.raises(ValueError,match='Unknown'):
        validate_config(dict(config(),pretend_setting=True))


def test_new_run_initialization_has_fresh_optimizer_and_parent_identity(tmp_path):
    cfg=config()
    parent=run(cfg,tmp_path/'parent')
    child_config=dict(cfg,steps=1,adam_lr=0.)
    child=run(child_config,tmp_path/'child',initialize_from=tmp_path/'parent'/'last.pt')
    for key,value in parent.state_dict().items():
        torch.testing.assert_close(value,child.state_dict()[key],rtol=0,atol=0)
    saved=torch.load(tmp_path/'child'/'last.pt',weights_only=True)
    assert saved['step']==1 and saved['attempt']==1
    assert saved['initialization']['parent_step']==2
    assert saved['initialization']['optimizer_reset']
    assert saved['run_id'] != saved['initialization']['parent_run_id']
    assert len(saved['initialization']['sha256'])==64
