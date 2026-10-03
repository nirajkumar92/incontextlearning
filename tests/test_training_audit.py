"""Adversarial integration checks, including actual distributed reduction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tabular_foundation.train import episode_loss, synchronize_gradients
from tabular_foundation.runtime import profile_schedule


@pytest.mark.parametrize('workers,device_request,local_rank,expected', [
    (1, 'cuda:3', '7', 3),
    (1, 'cuda', '7', 0),
    (2, 'cuda:3', '1', 1),
])
def test_explicit_cuda_device_and_distributed_local_rank(monkeypatch, workers, device_request, local_rank, expected):
    import tabular_foundation.train as training
    monkeypatch.setenv('WORLD_SIZE', str(workers))
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', local_rank)
    selected, initialized = [], []
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    monkeypatch.setattr(training.dist, 'init_process_group', initialized.append)
    device, rank, size = training._distributed_device(device_request)
    assert device == torch.device('cuda', expected)
    assert selected == [expected]
    assert (rank, size) == (0, workers)
    assert initialized == (['nccl'] if workers > 1 else [])


def test_loss_only_classification_law_rejects_negative_mass():
    logits = torch.tensor([[.1, -.3]], requires_grad=True)
    episode = SimpleNamespace(y_query=np.array([1]), query_weights=None,
                              metadata={'latent_conditional_law': np.array([[-.1, 1.1]])})
    with pytest.raises(ValueError, match='conditional classification law'):
        episode_loss({'logits': logits}, episode, 'cpu', conditional_labels=True)
    episode.metadata['latent_conditional_law'] = np.array([[.25, .75]])
    loss, count = episode_loss({'logits': logits}, episode, 'cpu', conditional_labels=True)
    loss.backward()
    torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - torch.tensor([[.25, .75]]))
    assert count == 1


def test_regression_episode_loss_keeps_target_precision_until_standardization():
    from tabular_foundation.distributions import ContinuousDistribution
    params = torch.zeros(2, 132, requires_grad=True)
    distribution = ContinuousDistribution(params, center=100000000., scale=.5)
    episode = SimpleNamespace(y_query=np.array([100000000., 100000001.], dtype=np.float64),
                              query_weights=None)
    loss, count = episode_loss({'distribution': distribution}, episode, 'cpu')
    expected = -distribution.log_prob(torch.tensor(episode.y_query, dtype=torch.float64)).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert count == 2
    assert not torch.equal(params.grad[0], params.grad[1])


def test_importance_weights_use_original_macro_denominator_not_route_means():
    weights = np.array([.000002, 1.999998, .000002, .000002])
    target = np.array([1, 0, 1, 1])
    logits = torch.tensor([[.1, -.3], [.3, .1], [-.2, .1], [.5, -.4]], requires_grad=True)
    full, count = episode_loss({"logits": logits}, SimpleNamespace(y_query=target, query_weights=weights), "cpu")
    reference = full / count
    ref_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    routed = 0
    for indices in (np.array([0]), np.array([1, 2, 3])):
        summed, _ = episode_loss({"logits": logits[indices]}, SimpleNamespace(y_query=target[indices], query_weights=weights[indices]), "cpu")
        routed = routed + summed / count
    route_grad = torch.autograd.grad(routed, logits)[0]
    torch.testing.assert_close(route_grad, ref_grad)
    # This test has deliberately unequal route sizes and importance masses.
    assert not torch.isclose(reference.detach(), full.detach() / weights.sum())


class _DistributedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([.2, -.3]))
        self.left = torch.nn.Parameter(torch.tensor(.1))
        self.right = torch.nn.Parameter(torch.tensor(-.2))
        self.unused = torch.nn.Parameter(torch.tensor(9.))


def _macro_loss(model, rank):
    if rank == 0:
        x = torch.tensor([1., 2., -1.])
        y = np.array([1, 0, 1])
        w = np.array([.001, 1.999, .001])
        offset = model.left
        routes = ([0], [1, 2])
    else:
        x = torch.tensor([.5])
        y, w, offset, routes = np.array([0]), np.array([1.]), model.right, ([0],)
    logits = x[:, None] * model.shared[None, :] + torch.stack((offset, -offset))[None, :]
    result = 0
    for route in routes:
        summed, _ = episode_loss({"logits": logits[route]}, SimpleNamespace(y_query=y[route], query_weights=w[route]), "cpu")
        result = result + summed / len(y)
    return result


def _distributed_worker(rank, rendezvous, output):
    import datetime
    import os
    import sys
    import torch.distributed as dist
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    model = _DistributedModel()
    # W/B=1 because there are two ranks and two logical macroepisodes.
    _macro_loss(model, rank).backward()
    synchronize_gradients(model)
    if rank == 0:
        Path(output).write_text(json.dumps({name: None if p.grad is None else p.grad.tolist()
                                           for name, p in model.named_parameters()}))
    dist.destroy_process_group()


def test_two_rank_ragged_reduction_matches_serial_macro_objective(tmp_path):
    if os.environ.get("TFM_RUN_DISTRIBUTED_TESTS") != "1":
        pytest.skip("Set TFM_RUN_DISTRIBUTED_TESTS=1 where local Gloo sockets are permitted")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable in this PyTorch build")
    import torch.multiprocessing as mp
    output = tmp_path / "gradients.json"
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"), str(output)), nprocs=2, join=True)
    actual = json.loads(output.read_text())
    expected = _DistributedModel()
    ((_macro_loss(expected, 0) + _macro_loss(expected, 1)) / 2).backward()
    for name, parameter in expected.named_parameters():
        if parameter.grad is None:
            assert actual[name] is None
        else:
            torch.testing.assert_close(torch.tensor(actual[name]), parameter.grad)


def test_static_checkpoint_resume_replays_the_same_second_update(tmp_path, monkeypatch):
    import tabular_foundation.train as training
    config = {"model": "tiny", "finance_adapter": True, "device": "cpu", "cpu_threads": 1,
              "seed": 125, "steps": 2, "stage_end_steps": [2, 2, 2], "global_batch": 2,
              "finance_share": 0, "optimizer": "adamw", "checkpoint_every": 1,
              "static_overrides": {"family": "P0", "subfamily": "linear_gam", "task": "binary",
                                   "n_support": 4, "n_query": 3, "n_features": 4}}
    first = tmp_path / "first-update.pt"
    real_save = training._save_checkpoint

    def save_and_preserve(path, model, optimizers, cfg, step, attempt, hours):
        real_save(path, model, optimizers, cfg, step, attempt, hours)
        if step == 1 and not first.exists():
            shutil.copyfile(path, first)
    monkeypatch.setattr(training, "_save_checkpoint", save_and_preserve)
    uninterrupted = training.run(config, tmp_path / "uninterrupted")
    resumed = training.run(config, tmp_path / "resumed", resume=first)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    saved = torch.load(first, map_location="cpu", weights_only=True)
    assert saved["step"] == saved["attempt"] == 1
    assert profile_schedule(125, saved["attempt"], 2, 0) == profile_schedule(125, 1, 2, 0)


def test_fractional_evaluation_labels_are_rejected():
    from tabular_foundation.metrics import binary_metrics, multiclass_metrics
    with pytest.raises(ValueError):
        binary_metrics([0., .9], [.1, .9])
    with pytest.raises(ValueError):
        multiclass_metrics([0., 1.9], [[.9, .1], [.1, .9]])


def test_resume_keeps_work_charged_after_last_checkpoint(tmp_path, monkeypatch):
    import tabular_foundation.train as training
    from tabular_foundation.runtime import ProjectLedger
    directory = tmp_path / "run"
    ledger_path = tmp_path / "project-ledger.json"
    config = {"model": "tiny", "finance_adapter": True, "device": "cpu", "cpu_threads": 1,
              "seed": 91, "steps": 2, "stage_end_steps": [2, 2, 2], "global_batch": 1,
              "finance_share": 0, "optimizer": "adamw", "checkpoint_every": 1,
              "project_ledger": str(ledger_path),
              "static_overrides": {"family": "P0", "subfamily": "linear_gam", "task": "binary",
                                   "n_support": 4, "n_query": 3, "n_features": 4}}
    first = tmp_path / "first-update.pt"
    real_save = training._save_checkpoint

    def capture(path, model, optimizers, cfg, step, attempt, hours):
        real_save(path, model, optimizers, cfg, step, attempt, hours)
        if step == 1 and not first.exists():
            shutil.copyfile(path, first)
    monkeypatch.setattr(training, "_save_checkpoint", capture)
    expected = training.run(config, directory)
    checkpoint = torch.load(first, map_location="cpu", weights_only=True)
    run_id = str(directory.resolve())
    assert checkpoint["run_id"] == run_id
    assert checkpoint["cumulative_gpu_hours"] == 0.
    ledger = ProjectLedger(ledger_path)
    # Emulate scheduler-reconciled work consumed after the last durable update.
    ledger.update(run_id, 7.5)
    resumed = training.run(config, directory, resume=first)
    assert ledger.recorded_hours(run_id) == 7.5
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    assert json.loads((directory / "completion.json").read_text())["cumulative_gpu_hours"] == 7.5
    with pytest.raises(ValueError, match="original output"):
        training.run(config, tmp_path / "different-run", resume=first)
    # Checkpoints written before run_id was added remain resumable in-place.
    checkpoint.pop("run_id")
    legacy = directory / "legacy.pt"
    torch.save(checkpoint, legacy)
    training.run(config, directory, resume=legacy)
    assert ledger.recorded_hours(run_id) == 7.5


def test_nan_timestamps_cannot_pass_the_finance_availability_contract():
    from tabular_foundation.data import assert_time_contract
    support = {"X": np.zeros((1, 4)), "event_time": np.array([np.nan]), "label_available_time": np.array([0.])}
    query = {"X": np.zeros((1, 4)), "event_time": np.array([2.])}
    with pytest.raises(ValueError):
        assert_time_contract(support, query, cutoff=1)
    support["event_time"] = np.array([0.])
    query["event_time"] = np.array([np.nan])
    with pytest.raises(ValueError):
        assert_time_contract(support, query, cutoff=1)
