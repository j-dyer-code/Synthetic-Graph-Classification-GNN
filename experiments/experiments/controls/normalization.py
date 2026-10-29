import copy
import hashlib
import random
import time
from contextlib import contextmanager
import numpy as np
import torch

BN_NAMES = ("bn1", "bn2", "bn3")


def tensor_mapping_hash(items):
    result = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        metadata = f"{name}\x00{tensor.dtype}\x00{tuple(tensor.shape)}\x00".encode()
        result.update(len(metadata).to_bytes(8, "big"))
        result.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(len(raw).to_bytes(8, "big"))
        result.update(raw)
    return result.hexdigest()


def state_hash(model):
    return tensor_mapping_hash(model.state_dict().items())


def parameter_hash(model):
    return tensor_mapping_hash(model.named_parameters())


@contextmanager
def preserve_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class ChannelMoments:
    def __init__(self, location="activation"):
        self.location = location
        self.count = 0
        self.mean = self.m2 = self.minimum = self.maximum = None

    def update(self, tensor):
        if tensor.ndim != 2:
            raise ValueError(f"{self.location}: expected [observations, channels]")
        values = tensor.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(
                f"{self.location}: nonfinite calibration activation"
            )
        size = int(values.shape[0])
        if size == 0:
            return
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        m2 = variance * size
        low, high = (values.amin(dim=0), values.amax(dim=0))
        if self.count == 0:
            self.mean, self.m2, self.minimum, self.maximum = (mean, m2, low, high)
        else:
            if self.mean.shape != mean.shape or self.mean.device != mean.device:
                raise ValueError(f"{self.location}: channel shape or device changed")
            total = self.count + size
            delta = mean - self.mean
            self.m2 += m2 + delta.square() * (self.count * size / total)
            self.mean += delta * (size / total)
            self.minimum = torch.minimum(self.minimum, low)
            self.maximum = torch.maximum(self.maximum, high)
        self.count += size

    @property
    def variance(self):
        if self.count == 0:
            raise ValueError(f"{self.location}: empty training calibration iterator")
        value = (self.m2 / self.count).clamp_min(0)
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(value).all()):
            raise FloatingPointError(f"{self.location}: nonfinite accumulated moments")
        return value

    def report(self):
        return {
            "observations": self.count,
            "mean": self.mean.detach().cpu().tolist(),
            "population_variance": self.variance.detach().cpu().tolist(),
            "minimum": self.minimum.detach().cpu().tolist(),
            "maximum": self.maximum.detach().cpu().tolist(),
            "nonfinite_count": 0,
        }


class _Inputs:
    __slots__ = ("x", "edge_index", "batch", "graph_features")

    def __init__(self, batch, device):
        for name in self.__slots__:
            value = getattr(batch, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Calibration input {name} must be a tensor")
            setattr(self, name, value.to(device))

    @property
    def y(self):
        raise RuntimeError("Labels are forbidden during population calibration")


class _LayerReached(Exception):
    pass


def _snapshot_bn(model):
    result = {}
    for name in BN_NAMES:
        layer = getattr(model, name)
        if not isinstance(layer, torch.nn.BatchNorm1d) or not layer.track_running_stats:
            raise TypeError(f"{name} must be BatchNorm1d with tracked statistics")
        if layer.running_mean is None or layer.running_var is None:
            raise ValueError(f"{name} has no running buffers")
        variance = layer.running_var.detach().double().cpu()
        weight = (
            layer.weight.detach().double().cpu()
            if layer.affine
            else torch.ones_like(variance)
        )
        bias = (
            layer.bias.detach().double().cpu()
            if layer.affine
            else torch.zeros_like(variance)
        )
        gain = weight / torch.sqrt(variance + layer.eps)
        result[name] = {
            "running_mean": layer.running_mean.detach().double().cpu().tolist(),
            "running_variance": variance.tolist(),
            "num_batches_tracked": int(layer.num_batches_tracked),
            "eps": layer.eps,
            "affine_weight": weight.tolist(),
            "affine_bias": bias.tolist(),
            "effective_gain": gain.tolist(),
            "max_absolute_effective_gain": float(gain.abs().max()),
        }
    return result


def _calibrate_layer(model, name, loader_factory, device):
    layer = getattr(model, name)
    moments = ChannelMoments(name)
    batches = 0
    graph_count = 0
    started = time.perf_counter()

    def capture_and_stop(module, inputs):
        moments.update(inputs[0])
        raise _LayerReached()

    handle = layer.register_forward_pre_hook(capture_and_stop)
    try:
        with torch.no_grad():
            for batch in loader_factory():
                inputs = _Inputs(batch, device)
                if inputs.batch.numel() == 0:
                    raise ValueError("Empty graph batch supplied for calibration")
                graph_count += int(inputs.batch.max().item()) + 1
                try:
                    model(inputs)
                except _LayerReached:
                    pass
                else:
                    raise RuntimeError(f"Model forward did not reach {name}")
                batches += 1
    finally:
        handle.remove()
    variance = moments.variance
    with torch.no_grad():
        layer.running_mean.copy_(moments.mean.to(layer.running_mean))
        layer.running_var.copy_(variance.to(layer.running_var))
        layer.num_batches_tracked.fill_(batches)
    if not bool(
        torch.isfinite(layer.running_mean).all()
        and torch.isfinite(layer.running_var).all()
    ):
        raise FloatingPointError(f"{name}: buffer conversion produced nonfinite values")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "layer": name,
        "fitting_partition": "train",
        "weighting": "node" if name in ("bn1", "bn2") else "graph",
        "dropout": "disabled",
        "variance_convention": "population variance M2/N",
        "batches": batches,
        "training_graphs": graph_count,
        "moments": moments.report(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def calibrated_copy(model, training_loader_factory, device):
    if not callable(training_loader_factory):
        raise TypeError("training_loader_factory must return a fresh iterator")
    device = torch.device(device)
    started = time.perf_counter()
    original_state_before = state_hash(model)
    parameters_before = parameter_hash(model)
    flags_before = {name: module.training for name, module in model.named_modules()}
    requires_grad_before = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    before = _snapshot_bn(model)
    with preserve_rng():
        candidate = copy.deepcopy(model).to(device)
        candidate.eval()
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        layers = [
            _calibrate_layer(candidate, name, training_loader_factory, device)
            for name in BN_NAMES
        ]
        counts = {layer["training_graphs"] for layer in layers}
        if len(counts) != 1:
            raise RuntimeError("Training graph counts differ across calibration passes")
        if layers[0]["moments"]["observations"] != layers[1]["moments"]["observations"]:
            raise RuntimeError("Training node counts differ across calibration passes")
        if layers[2]["moments"]["observations"] != layers[2]["training_graphs"]:
            raise RuntimeError(
                "bn3 observation count is not the number of training graphs"
            )
        parameters_after = parameter_hash(candidate)
        if parameters_before != parameters_after:
            raise RuntimeError("Population calibration changed learned parameters")
        allowed = {
            f"{name}.{buffer}"
            for name in BN_NAMES
            for buffer in ("running_mean", "running_var", "num_batches_tracked")
        }
        untouched_before = tensor_mapping_hash(
            (
                (name, value)
                for name, value in model.state_dict().items()
                if name not in allowed
            )
        )
        untouched_after = tensor_mapping_hash(
            (
                (name, value)
                for name, value in candidate.state_dict().items()
                if name not in allowed
            )
        )
        if untouched_before != untouched_after:
            raise RuntimeError("Population calibration changed a non-BatchNorm buffer")
        after = _snapshot_bn(candidate)
        calibrated_state_hash = state_hash(candidate)
    original_state_after = state_hash(model)
    if original_state_before != original_state_after:
        raise RuntimeError("Training model state changed during calibration")
    if flags_before != {
        name: module.training for name, module in model.named_modules()
    }:
        raise RuntimeError("Training model flags changed during calibration")
    if requires_grad_before != {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }:
        raise RuntimeError(
            "Training parameter requires_grad flags changed during calibration"
        )
    return (
        candidate,
        {
            "schema": "training-population-batchnorm-v1",
            "fitting_partition": "train",
            "labels_accessed": False,
            "layer_order": list(BN_NAMES),
            "layers": layers,
            "batchnorm_before": before,
            "batchnorm_after": after,
            "original_state_sha256_before": original_state_before,
            "original_state_sha256_after": original_state_after,
            "calibrated_state_sha256": calibrated_state_hash,
            "learned_parameters_sha256_before": parameters_before,
            "learned_parameters_sha256_after": parameters_after,
            "noncalibration_state_sha256_before": untouched_before,
            "noncalibration_state_sha256_after": untouched_after,
            "training_state_unchanged": True,
            "training_flags_unchanged": True,
            "rng_restored": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
