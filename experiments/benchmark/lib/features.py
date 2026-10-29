import hashlib
import numpy as np

CONDITIONS = (
    "topology",
    "constant",
    "independent",
    "permuted",
    "mixture_025",
    "mixture_050",
    "mixture_075",
    "community",
)


def stable_seed(seed, graph_id, stream):
    payload = f"r1.3-v1\x00{int(seed)}\x00{graph_id}\x00{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_features(z_topology, community, condition, seed, graph_id):
    z = np.asarray(z_topology, dtype=np.float32)
    if z.ndim != 2 or not all(z.shape) or (not np.isfinite(z).all()):
        raise ValueError("z_topology must be a finite, nonempty N x D matrix")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose {CONDITIONS}")
    if condition == "topology":
        return z.copy()
    if condition == "constant":
        return np.ones_like(z)
    if condition == "permuted":
        rng = np.random.default_rng(stable_seed(seed, graph_id, "row_permutation"))
        return z[rng.permutation(len(z))].copy()
    noise = (
        np.random.default_rng(stable_seed(seed, graph_id, "node_noise"))
        .normal(size=z.shape)
        .astype(np.float32)
    )
    if condition == "independent":
        return noise
    if condition.startswith("mixture_"):
        alpha = int(condition.rsplit("_", 1)[1]) / 100.0
        return (np.sqrt(alpha) * z + np.sqrt(1.0 - alpha) * noise).astype(np.float32)
    membership = np.asarray(community)
    if membership.shape != (len(z),) or not np.issubdtype(membership.dtype, np.integer):
        raise ValueError("community must contain one integer membership per node")
    _, inverse = np.unique(membership, return_inverse=True)
    rng = np.random.default_rng(stable_seed(seed, graph_id, "community_centres"))
    centres = rng.normal(size=(int(inverse.max()) + 1, z.shape[1])).astype(np.float32)
    return ((centres[inverse] + noise) / np.sqrt(2.0)).astype(np.float32)
