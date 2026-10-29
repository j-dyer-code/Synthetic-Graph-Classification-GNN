from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import os
from pathlib import Path
import random
import sys
import unittest
import numpy as np
import torch
from normalization import BN_NAMES, ChannelMoments, calibrated_copy, state_hash

DEVICE = os.environ.get("NORMALIZATION_TEST_DEVICE", "cpu")


class NoLabels:
    def __init__(self, x, edge_index, batch, graph_features):
        self.x, self.edge_index = (x, edge_index)
        self.batch, self.graph_features = (batch, graph_features)

    @property
    def y(self):
        raise AssertionError("A training label was accessed")


def make_batch(graphs):
    xs, memberships, offsets, edges, descriptors = ([], [], 0, [], [])
    for index, (x, graph_features) in enumerate(graphs):
        xs.append(x)
        memberships.append(torch.full((len(x),), index, dtype=torch.long))
        nodes = torch.arange(len(x), dtype=torch.long) + offsets
        edges.append(torch.stack([nodes, nodes.roll(1)]))
        descriptors.append(graph_features)
        offsets += len(x)
    return NoLabels(
        torch.cat(xs),
        torch.cat(edges, dim=1),
        torch.cat(memberships),
        torch.stack(descriptors),
    )


def fixtures(dtype=torch.float64):
    graphs = []
    for graph in range(7):
        n = 1 + graph % 4
        x = torch.arange(n * 2, dtype=dtype).reshape(n, 2) / 3 + graph / 7
        graphs.append((x, torch.tensor([graph / 4, 1 + graph / 9], dtype=dtype)))
    return graphs


def batched(graphs, sizes=(2, 3, 2)):
    offset = 0
    result = []
    for size in sizes:
        result.append(make_batch(graphs[offset : offset + size]))
        offset += size
    assert offset == len(graphs)
    return result


def device_batch(batch):
    return NoLabels(
        *(
            getattr(batch, name).to(DEVICE)
            for name in ("x", "edge_index", "batch", "graph_features")
        )
    )


def pooled(x, membership):
    count = int(membership.max()) + 1
    out = torch.zeros((count, x.shape[1]), dtype=x.dtype, device=x.device)
    out.index_add_(0, membership, x)
    return out / torch.bincount(membership).to(x.dtype).unsqueeze(-1)


class KnownNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(2, eps=0.02).double()
        self.bn2 = torch.nn.BatchNorm1d(2, eps=0.03).double()
        self.bn3 = torch.nn.BatchNorm1d(2, eps=0.04).double()
        self.dropout = torch.nn.Dropout(0.75)
        self.output_calls = 0
        with torch.no_grad():
            for index, name in enumerate(BN_NAMES):
                layer = getattr(self, name)
                layer.running_mean.fill_(100 + index)
                layer.running_var.fill_(15 + index)
                layer.weight.copy_(torch.tensor([1.3, 0.7]))
                layer.bias.copy_(torch.tensor([-0.2, 0.4]))

    def forward(self, data):
        x = self.dropout(self.bn1(data.x))
        x = self.dropout(self.bn2(torch.relu(x) * 2 + 0.5))
        x = pooled(x, data.batch) + data.graph_features
        x = self.dropout(self.bn3(x))
        self.output_calls += 1
        return (x, x)


def expected_moments(model, graphs):
    combined = make_batch(graphs)
    x = combined.x
    result = []
    for name in BN_NAMES:
        layer = getattr(model, name)
        mean, variance = (x.mean(0), x.var(0, correction=0))
        result.append((mean, variance, len(x)))
        x = (x - mean) / torch.sqrt(variance + layer.eps)
        x = x * layer.weight.detach().cpu() + layer.bias.detach().cpu()
        if name == "bn1":
            x = torch.relu(x) * 2 + 0.5
        elif name == "bn2":
            x = pooled(x, combined.batch) + combined.graph_features
    return result


class NormalizationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_welford_unequal_batches_large_offset_and_constant_channels(self):
        values = torch.tensor(
            [[1000000000.0 + n * 0.125, 7.0] for n in range(13)], dtype=torch.float64
        )
        accumulator = ChannelMoments("known")
        for part in (values[:1], values[1:4], values[4:12], values[12:]):
            accumulator.update(part)
        self.assertEqual(accumulator.count, 13)
        torch.testing.assert_close(accumulator.mean, values.mean(0), rtol=0, atol=1e-07)
        torch.testing.assert_close(
            accumulator.variance, values.var(0, correction=0), rtol=0, atol=1e-10
        )
        self.assertEqual(accumulator.variance[1].item(), 0)
        with self.assertRaises(FloatingPointError):
            accumulator.update(torch.tensor([[float("nan"), 7.0]]))
        with self.assertRaises(ValueError):
            ChannelMoments().variance

    def test_sequential_training_moments_labels_and_invariants(self):
        graphs = fixtures()
        original = KnownNetwork().to(DEVICE).train()
        original.dropout.eval()
        for parameter in original.parameters():
            parameter.grad = torch.full_like(parameter, 0.314)
        before = state_hash(original)
        original_flags = [m.training for m in original.modules()]
        original_grads = [p.grad.clone() for p in original.parameters()]
        python_state = random.getstate()
        numpy_state = copy.deepcopy(np.random.get_state())
        torch_state = torch.get_rng_state().clone()
        cuda_states = (
            torch.cuda.get_rng_state_all() if DEVICE.startswith("cuda") else None
        )
        calls = []

        def loader():
            calls.append(1)
            random.random()
            np.random.random(3)
            torch.rand(5)
            if DEVICE.startswith("cuda"):
                torch.rand(5, device=DEVICE)
            yield from batched(graphs)

        expected = expected_moments(original, graphs)
        candidate, audit = calibrated_copy(original, loader, DEVICE)
        self.assertEqual(len(calls), 3)
        self.assertEqual(candidate.output_calls, 0)
        for name, (mean, variance, count), layer_audit in zip(
            BN_NAMES, expected, audit["layers"]
        ):
            layer = getattr(candidate, name)
            torch.testing.assert_close(
                layer.running_mean.cpu(), mean, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                layer.running_var.cpu(), variance, rtol=1e-12, atol=1e-12
            )
            self.assertEqual(layer_audit["moments"]["observations"], count)
            self.assertEqual(layer_audit["training_graphs"], 7)
            self.assertEqual(int(layer.num_batches_tracked), 3)
        self.assertEqual(before, state_hash(original))
        self.assertEqual(original_flags, [m.training for m in original.modules()])
        for actual, expected_grad in zip(original.parameters(), original_grads):
            self.assertTrue(torch.equal(actual.grad, expected_grad))
        self.assertEqual(python_state, random.getstate())
        current_numpy_state = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy_state[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy_state[1])
        self.assertEqual(numpy_state[2:], current_numpy_state[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        if cuda_states is not None:
            self.assertTrue(
                all(
                    (
                        torch.equal(a, b)
                        for a, b in zip(cuda_states, torch.cuda.get_rng_state_all())
                    )
                )
            )
        self.assertTrue(all((not m.training for m in candidate.modules())))
        self.assertTrue(all((not p.requires_grad for p in candidate.parameters())))
        json.dumps(audit, allow_nan=False)
        self.assertEqual(
            audit["learned_parameters_sha256_before"],
            audit["learned_parameters_sha256_after"],
        )

    def test_calibrated_inference_is_independent_of_validation_batching(self):
        graphs = fixtures()
        model, _ = calibrated_copy(
            KnownNetwork().to(DEVICE), lambda: batched(graphs), DEVICE
        )
        with torch.no_grad():
            one = torch.cat(
                [model(device_batch(batch))[0] for batch in batched(graphs, (1,) * 7)]
            )
            many = torch.cat(
                [model(device_batch(batch))[0] for batch in batched(graphs, (3, 4))]
            )
        torch.testing.assert_close(one, many, atol=1e-12, rtol=1e-12)

    def test_rng_restored_when_loader_fails(self):
        model = KnownNetwork().to(DEVICE)
        before = torch.get_rng_state().clone()
        state = state_hash(model)

        def bad_loader():
            torch.rand(2)
            raise RuntimeError("Deliberate loader failure")
            yield

        with self.assertRaisesRegex(RuntimeError, "Deliberate loader failure"):
            calibrated_copy(model, bad_loader, DEVICE)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(state, state_hash(model))

    def test_all_six_architectures_and_label_forbidden_forward(self):
        from lib import models

        graphs = fixtures(torch.float32)
        for architecture in ("GCN", "GAT", "GATV2", "SAGE", "GIN", "GTN"):
            with self.subTest(architecture=architecture):
                model = getattr(models, "GraphFamilyClassifier" + architecture)(
                    2, 4, 5, 2, 0.5
                ).to(DEVICE)
                candidate, audit = calibrated_copy(
                    model, lambda: batched(graphs), DEVICE
                )
                with torch.no_grad():
                    output, embedding = candidate(device_batch(make_batch(graphs[:2])))
                self.assertEqual(output.shape, (2, 5))
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertEqual(
                    audit["original_state_sha256_before"],
                    audit["original_state_sha256_after"],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
