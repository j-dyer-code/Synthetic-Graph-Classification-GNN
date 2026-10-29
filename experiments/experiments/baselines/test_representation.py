from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import representations as rep

CONFIG = {
    "dimension": 8,
    "wl_iterations": 2,
    "train_epochs": 4,
    "inference_epochs": 3,
    "negative": 2,
    "min_count": 1,
    "sample": 0.0,
    "alpha": 0.025,
    "min_alpha": 0.0001,
    "chunk_size": 8192,
    "checkpoint_identity": "artificial-fixture-v1",
}


def fixture_documents():
    return [
        ("a", ["h0:a"] * 8000 + ["h1:a"] * 4501),
        ("b", ["h0:b"] * 7800 + ["h1:b"] * 4303),
    ]


class RepresentationTests(unittest.TestCase):
    def test_wl_permutation_invariance_and_multiplicity(self):
        edges = np.array([[0, 1, 2, 2], [1, 2, 0, 3]], dtype=np.int64)
        relabel = np.array([3, 0, 4, 1, 2])
        reordered = relabel[edges][:, [2, 0, 3, 1]]
        tokens = rep.wl_tokens(edges, 5)
        self.assertEqual(tokens, rep.wl_tokens(reordered, 5))
        self.assertEqual(
            tokens, rep.wl_tokens(np.concatenate([edges, edges[::-1]], axis=1), 5)
        )
        self.assertEqual(len(tokens), 15)
        self.assertEqual(tokens.count("h0:2"), 2)
        self.assertIn("h0:0", tokens)
        self.assertEqual(sum((t.startswith("h2:") for t in tokens)), 5)
        with self.assertRaises(ValueError):
            rep.wl_tokens(np.array([[0, 0], [1, 1]]), 5)
        with self.assertRaises(ValueError):
            rep.wl_tokens(np.array([[0], [0]]), 5)

    def test_training_chunks_share_graph_tags_and_preserve_every_token(self):
        chunks = list(rep._ChunkDocuments(fixture_documents(), 8192, 42, 0))
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all((len(c.words) <= 8192 for c in chunks)))
        self.assertEqual([c.tags for c in chunks], [["a"], ["a"], ["b"], ["b"]])
        for tag, expected in fixture_documents():
            actual = [w for c in chunks if c.tags == [tag] for w in c.words]
            self.assertEqual(sorted(expected), sorted(actual))

    def test_full_document_frozen_inference_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = rep.train_model(fixture_documents(), 42, CONFIG, Path(tmp))
            before = rep.frozen_weights_hash(model)
            random_state_before = model.random.get_state()
            original = rep.train_document_dbow
            calls = []

            def recording(*args, **kwargs):
                calls.append((len(args[1]), id(kwargs["doctag_vectors"])))
                self.assertFalse(kwargs["learn_hidden"])
                self.assertFalse(kwargs["learn_words"])
                return original(*args, **kwargs)

            tokens = fixture_documents()[0][1]
            with patch.object(rep, "train_document_dbow", side_effect=recording):
                vector = rep.infer_graph(model, tokens, 42, CONFIG)
            self.assertEqual(
                sum((c[0] for c in calls)), len(tokens) * CONFIG["inference_epochs"]
            )
            self.assertTrue(all((c[0] <= 8192 for c in calls)))
            self.assertEqual(len({c[1] for c in calls}), 1)
            self.assertEqual(before, rep.frozen_weights_hash(model))
            np.testing.assert_array_equal(
                random_state_before[1], model.random.get_state()[1]
            )
            np.testing.assert_array_equal(
                vector, rep.infer_graph(model, tokens[::-1], 42, CONFIG)
            )
            np.testing.assert_array_equal(
                vector, rep.infer_graph(model, tokens, 42, CONFIG)
            )
            self.assertGreater(float(np.linalg.norm(vector)), 0)
            cov = rep.vocabulary_coverage(model, tokens + ["h2:unseen"])
            self.assertEqual(cov["h0"]["coverage"], 1.0)
            self.assertEqual(cov["h2"]["oov"], 1)
            self.assertEqual(cov["h2"]["coverage"], 0.0)

    def test_interrupted_epoch_checkpoint_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = rep.train_model(fixture_documents(), 43, CONFIG, root / "full")
            save = rep._save_checkpoint

            def interrupted(model, directory, **kwargs):
                save(model, directory, **kwargs)
                if model.structural_svm_checkpoint["completed_epochs"] == 2:
                    raise RuntimeError(
                        "simulated process interruption after durable checkpoint"
                    )

            with patch.object(rep, "_save_checkpoint", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    rep.train_model(fixture_documents(), 43, CONFIG, root / "resume")
            observed = rep.train_model(fixture_documents(), 43, CONFIG, root / "resume")
            self.assertEqual(
                rep.frozen_weights_hash(expected), rep.frozen_weights_hash(observed)
            )
            np.testing.assert_array_equal(
                expected.random.get_state()[1], observed.random.get_state()[1]
            )
            self.assertEqual(len(list((root / "resume").glob("*.model"))), 2)
            self.assertEqual(observed.structural_svm_checkpoint["completed_epochs"], 4)
            self.assertEqual(
                len(observed.structural_svm_checkpoint["epoch_training_seconds"]), 4
            )
            self.assertGreater(
                observed.structural_svm_checkpoint["cumulative_fit_seconds"], 0
            )
            changed = {**CONFIG, "checkpoint_identity": "different-training-split"}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                rep.train_model(fixture_documents(), 43, changed, root / "resume")

    def test_reject_one_shot_training_iterator(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TypeError, "restartable"):
                rep.train_model(iter(fixture_documents()), 42, CONFIG, Path(tmp))


if __name__ == "__main__":
    unittest.main()
