import hashlib
import json
import math
import os
import time
from pathlib import Path
import uuid
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.doc2vec_inner import train_document_dbow
import numpy as np

_INT_KEYS = (
    "dimension",
    "wl_iterations",
    "train_epochs",
    "inference_epochs",
    "negative",
    "min_count",
    "chunk_size",
)
_FLOAT_KEYS = ("sample", "alpha", "min_alpha")
_PROTOCOL = "degree-wl-dbow-chunked-v1"


def _stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _seed(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return _stable_hash(payload)


def _settings(config):
    out = {}
    for key in _INT_KEYS:
        value = config[key]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key} must be an integer")
        out[key] = int(value)
    for key in _FLOAT_KEYS:
        out[key] = float(config[key])
        if not math.isfinite(out[key]):
            raise ValueError(f"{key} must be finite")
    if any((out[k] < 1 for k in _INT_KEYS if k != "wl_iterations")):
        raise ValueError(
            "Dimensions, epochs, negative samples, minimum count and chunk size must be positive"
        )
    if out["wl_iterations"] < 0 or out["chunk_size"] > 8192:
        raise ValueError(
            "WL iterations must be nonnegative and chunk_size must be at most 8192"
        )
    if not 0 <= out["sample"] < 1 or not 0 < out["min_alpha"] <= out["alpha"]:
        raise ValueError("Invalid subsampling or learning-rate settings")
    return out


def wl_tokens(edge_index, n_nodes, iterations=2):
    if isinstance(n_nodes, bool) or int(n_nodes) != n_nodes or n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    n_nodes, iterations = (int(n_nodes), int(iterations))
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edge_index must be an integer array with shape (2, E)")
    neighbors = [set() for _ in range(n_nodes)]
    seen = set()
    for u, v in edges.T:
        u, v = (int(u), int(v))
        if not 0 <= u < n_nodes or not 0 <= v < n_nodes:
            raise ValueError("An edge endpoint is outside the vertex range")
        if u == v:
            raise ValueError(
                "Degree-WL input must be a simple graph without self-loops"
            )
        if (u, v) in seen:
            raise ValueError("Repeated directed edge entry")
        seen.add((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)
    labels = [f"h0:{len(adj)}" for adj in neighbors]
    tokens = list(labels)
    for level in range(1, iterations + 1):
        new_labels = []
        for vertex, adj in enumerate(neighbors):
            signature = (
                labels[vertex] + "|" + ",".join(sorted((labels[v] for v in adj)))
            )
            new_labels.append(
                f"h{level}:" + hashlib.sha256(signature.encode("ascii")).hexdigest()
            )
        labels = new_labels
        tokens.extend(labels)
    return sorted(tokens)


class _ChunkDocuments:
    def __init__(self, documents, chunk_size, seed, epoch):
        self.documents = documents
        self.chunk_size = chunk_size
        self.seed = seed
        self.epoch = epoch

    def __iter__(self):
        seen_tags = set()
        for graph_id, tokens in self.documents:
            tag = str(graph_id)
            if tag in seen_tags:
                raise ValueError(f"Duplicate graph tag in training corpus: {tag!r}")
            seen_tags.add(tag)
            words = sorted(tokens)
            if not words or any((not isinstance(word, str) for word in words)):
                raise ValueError(
                    "Every graph must provide a nonempty sequence of string tokens"
                )
            if self.epoch >= 0:
                np.random.RandomState(_seed(self.seed, self.epoch, tag)).shuffle(words)
            for start in range(0, len(words), self.chunk_size):
                yield TaggedDocument(words[start : start + self.chunk_size], [tag])


def _checkpoint_metadata(seed, settings, identity):
    spec = {
        "protocol": _PROTOCOL,
        "gensim": gensim.__version__,
        "seed": int(seed),
        "settings": settings,
        "checkpoint_identity": identity,
    }
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**spec, "signature": digest}


def _save_checkpoint(model, checkpoint_dir, rotate_current=True):
    current = checkpoint_dir / "graph2vec-current.model"
    previous = checkpoint_dir / "graph2vec-previous.model"
    temporary = checkpoint_dir / f"graph2vec-{uuid.uuid4().hex}.tmp"
    try:
        model.save(str(temporary), separately=[])
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if rotate_current and current.exists():
            os.replace(current, previous)
        os.replace(temporary, current)
        directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def train_model(documents, seed, config, checkpoint_dir):
    if gensim.__version__ != "4.4.0":
        raise RuntimeError("This protocol requires gensim==4.4.0")
    if iter(documents) is documents:
        raise TypeError("documents must be restartable, not a one-shot generator")
    settings = _settings(config)
    identity = config.get("checkpoint_identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(
            "config['checkpoint_identity'] must identify the caller's corpus/split/protocol"
        )
    metadata = _checkpoint_metadata(seed, settings, identity)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = None
    rotate_current = True
    present, failures = ([], [])
    for name in ("graph2vec-current.model", "graph2vec-previous.model"):
        path = checkpoint_dir / name
        if not path.exists():
            continue
        present.append(path)
        try:
            candidate = Doc2Vec.load(str(path))
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            continue
        saved = getattr(candidate, "structural_svm_checkpoint", {})
        if saved.get("signature") != metadata["signature"]:
            raise ValueError(f"Checkpoint identity/configuration mismatch: {path}")
        completed = saved.get("completed_epochs")
        if (
            not isinstance(completed, int)
            or not 0 <= completed <= settings["train_epochs"]
        ):
            raise ValueError(f"Invalid completed-epoch metadata: {path}")
        model = candidate
        rotate_current = name == "graph2vec-current.model"
        print(
            f"GRAPH2VEC RESUME epoch={completed}/{settings['train_epochs']}", flush=True
        )
        break
    if model is None and present:
        raise RuntimeError(
            "No readable checkpoint; refusing to overwrite evidence: "
            + "; ".join(failures)
        )
    if model is None:
        model = Doc2Vec(
            vector_size=settings["dimension"],
            dm=0,
            dbow_words=0,
            window=0,
            hs=0,
            negative=settings["negative"],
            min_count=settings["min_count"],
            sample=settings["sample"],
            alpha=settings["alpha"],
            min_alpha=settings["min_alpha"],
            workers=1,
            seed=int(seed),
            epochs=settings["train_epochs"],
            sorted_vocab=1,
            hashfxn=_stable_hash,
            batch_words=8192,
        )
        corpus = _ChunkDocuments(documents, settings["chunk_size"], int(seed), -1)
        vocabulary_start = time.perf_counter()
        model.build_vocab(corpus)
        vocabulary_seconds = time.perf_counter() - vocabulary_start
        if model.corpus_count < 1 or len(model.wv) < 1:
            raise ValueError("Training corpus has no usable graph tokens")
        model.structural_svm_checkpoint = {
            **metadata,
            "completed_epochs": 0,
            "vocabulary_seconds": vocabulary_seconds,
            "epoch_training_seconds": [],
            "cumulative_fit_seconds": vocabulary_seconds,
            "timing_scope": "Vocabulary construction plus completed epoch training; excludes model initialization, checkpoint I/O, interrupted epochs and inference",
        }
        _save_checkpoint(model, checkpoint_dir)
        print(
            f"GRAPH2VEC VOCAB graphs={len(model.dv)} chunks={model.corpus_count} tokens={model.corpus_total_words} vocabulary={len(model.wv)}",
            flush=True,
        )
    completed = model.structural_svm_checkpoint["completed_epochs"]
    for epoch in range(completed, settings["train_epochs"]):
        span = settings["alpha"] - settings["min_alpha"]
        start_alpha = settings["alpha"] - span * epoch / settings["train_epochs"]
        end_alpha = settings["alpha"] - span * (epoch + 1) / settings["train_epochs"]
        corpus = _ChunkDocuments(documents, settings["chunk_size"], int(seed), epoch)
        epoch_start = time.perf_counter()
        model.train(
            corpus_iterable=corpus,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
            epochs=1,
            start_alpha=start_alpha,
            end_alpha=end_alpha,
        )
        elapsed = time.perf_counter() - epoch_start
        saved_times = model.structural_svm_checkpoint
        epoch_times = saved_times["epoch_training_seconds"] + [elapsed]
        vocabulary_seconds = saved_times["vocabulary_seconds"]
        model.structural_svm_checkpoint = {
            **metadata,
            "completed_epochs": epoch + 1,
            "vocabulary_seconds": vocabulary_seconds,
            "epoch_training_seconds": epoch_times,
            "cumulative_fit_seconds": vocabulary_seconds + sum(epoch_times),
            "timing_scope": saved_times["timing_scope"],
        }
        _save_checkpoint(model, checkpoint_dir, rotate_current=rotate_current)
        rotate_current = True
        print(
            f"GRAPH2VEC EPOCH {epoch + 1}/{settings['train_epochs']} checkpoint_complete",
            flush=True,
        )
    if not rotate_current:
        _save_checkpoint(model, checkpoint_dir, rotate_current=False)
    return model


def infer_graph(model, tokens, seed, config):
    settings = _settings(config)
    if model.dm or model.vector_size != settings["dimension"]:
        raise ValueError("Model must be a DBOW model with the configured dimension")
    words = sorted(tokens)
    if not words:
        raise ValueError("Cannot infer an empty graph document")
    content_hash = hashlib.sha256()
    for word in words:
        encoded = word.encode("utf-8")
        content_hash.update(len(encoded).to_bytes(8, "little"))
        content_hash.update(encoded)
    graph_seed = _seed(int(seed), content_hash.hexdigest(), "inference")
    initializer = np.random.RandomState(_seed(graph_seed, "initialization"))
    vectors = (
        (initializer.random_sample((1, model.vector_size)) - 0.5) / model.vector_size
    ).astype(np.float32)
    lockf = np.ones(1, dtype=np.float32)
    work = np.zeros(model.layer1_size, dtype=np.float32)
    ordering = np.random.RandomState(_seed(graph_seed, "order"))
    saved_random = model.random
    model.random = np.random.RandomState(_seed(graph_seed, "negative_sampling"))
    try:
        for epoch in range(settings["inference_epochs"]):
            alpha = settings["alpha"] - (
                settings["alpha"] - settings["min_alpha"]
            ) * epoch / max(settings["inference_epochs"] - 1, 1)
            ordering.shuffle(words)
            for start in range(0, len(words), settings["chunk_size"]):
                train_document_dbow(
                    model,
                    words[start : start + settings["chunk_size"]],
                    [0],
                    alpha,
                    work=work,
                    train_words=False,
                    learn_words=False,
                    learn_hidden=False,
                    learn_doctags=True,
                    doctag_vectors=vectors,
                    doctags_lockf=lockf,
                )
    finally:
        model.random = saved_random
    if not np.isfinite(vectors).all():
        raise FloatingPointError("Graph embedding contains nonfinite values")
    return vectors[0].copy()


def vocabulary_coverage(model, tokens):
    levels = {}
    for word in tokens:
        level = word.partition(":")[0]
        entry = levels.setdefault(
            level,
            {
                "tokens": 0,
                "in_vocabulary": 0,
                "unique": set(),
                "retained_unique": set(),
            },
        )
        entry["tokens"] += 1
        entry["unique"].add(word)
        if word in model.wv.key_to_index:
            entry["in_vocabulary"] += 1
            entry["retained_unique"].add(word)
    result = {}
    for level, entry in sorted(levels.items()):
        total, retained = (entry["tokens"], entry["in_vocabulary"])
        result[level] = {
            "tokens": total,
            "in_vocabulary": retained,
            "oov": total - retained,
            "coverage": retained / total if total else None,
            "unique_tokens": len(entry["unique"]),
            "unique_in_vocabulary": len(entry["retained_unique"]),
        }
    return result


def frozen_weights_hash(model):
    digest = hashlib.sha256()
    arrays = {"word_vectors": model.wv.vectors, "graph_vectors": model.dv.vectors}
    for name in ("syn1neg", "syn1"):
        array = getattr(model, name, None)
        if array is not None:
            arrays[name] = array
    for name, array in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(contiguous.shape).encode())
        view = memoryview(contiguous).cast("B")
        for start in range(0, len(view), 1024 * 1024):
            digest.update(view[start : start + 1024 * 1024])
    for word in model.wv.index_to_key:
        digest.update(word.encode("utf-8") + b"\x00")
        digest.update(str(model.wv.get_vecattr(word, "count")).encode() + b"\x00")
    return digest.hexdigest()
