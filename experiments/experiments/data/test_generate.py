import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import generate


class GenerationFailures(unittest.TestCase):
    def test_generator_errors_are_propagated(self):
        cause = ValueError("Invalid generator arguments")
        generator = Mock(side_effect=cause)
        backend = SimpleNamespace(Graph=SimpleNamespace(Erdos_Renyi=generator))
        with patch.object(generate, "ig", backend):
            with self.assertRaisesRegex(
                RuntimeError, "erdos_renyi with 20 nodes"
            ) as caught:
                generate.generate_single_graph("erdos_renyi", 20)
        self.assertIs(caught.exception.__cause__, cause)
        generator.assert_called_once()

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unknown with 20 nodes") as caught:
            generate.generate_single_graph("unknown", 20)
        self.assertIsInstance(caught.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
