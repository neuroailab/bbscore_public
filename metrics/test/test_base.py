import unittest
import numpy as np
from metrics.base import BaseMetric


class TestBaseMetric(unittest.TestCase):

    def test_apply_ceiling(self):
        # Create dummy metric inheriting from BaseMetric
        class DummyMetric(BaseMetric):
            def compute_raw(self, source, target, test_source=None, test_target=None):
                # return dummy scores
                return {"score": np.array([1.0, 2.0, 3.0])}

        # Test without ceiling
        metric_no_ceiling = DummyMetric()
        scores = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(
            metric_no_ceiling.apply_ceiling(scores), scores))

        # Test with ceiling
        metric_with_ceiling = DummyMetric(ceiling=2.0)
        expected_ceiled_scores = np.array([0.5, 1.0, 1.5])
        self.assertTrue(np.allclose(
            metric_with_ceiling.apply_ceiling(scores), expected_ceiled_scores))

    def test_compute(self):
        # Create dummy metric for compute test
        class DummyMetric(BaseMetric):
            def compute_raw(self, source, target, test_source=None, test_target=None):
                # return dummy scores
                return {"score": np.array([[1.0, 2.0], [3.0, 4.0]])}

        metric = DummyMetric()
        result = metric.compute(None, None)  # Source and target not used
        self.assertIsInstance(result, dict)
        self.assertIn("raw_score", result)
        self.assertIn("ceiled_score", result)
        self.assertIn("median_score", result)
        self.assertIn("final_score", result)
        self.assertTrue(np.allclose(
            result["raw_score"], np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(np.allclose(
            result["ceiled_score"], np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(np.allclose(
            result["median_score"], np.array([1.5, 3.5])))
        self.assertAlmostEqual(result["final_score"], 2.5)

        # Test RSA compute (should early return the raw score)

        class RSADummy(BaseMetric):
            def compute_raw(self, source, target, test_source=None, test_target=None):
                return 0.8  # returns a single value

        rsa = RSADummy()
        result_rsa = rsa.compute(None, None)
        self.assertEqual(result_rsa, 0.8)


if __name__ == '__main__':
    unittest.main()
