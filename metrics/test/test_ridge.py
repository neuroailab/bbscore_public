import unittest
import numpy as np
from metrics.ridge import RidgeMetric, TorchRidgeMetric


class TestRidgeMetric(unittest.TestCase):
    def setUp(self):
        self.source = np.random.rand(100, 50)
        self.target = np.random.rand(100, 60)

    def test_ridge_metric_sklearn(self):
        metric = RidgeMetric(mode="sklearn")
        result = metric.compute(self.source, self.target)
        self.assertIn("final_pearson", result)
        self.assertIn("final_r2", result)

    def test_ridge_metric_torch(self):
        metric = TorchRidgeMetric()
        result = metric.compute(self.source, self.target)
        self.assertIn("final_pearson", result)
        self.assertIn("final_r2", result)

    def test_ridge_sklearn_with_test_set(self):
        test_source = np.random.rand(50, 50)
        test_target = np.random.rand(50, 60)
        metric = RidgeMetric(mode="sklearn")
        result = metric.compute(self.source, self.target,
                                test_source, test_target)
        self.assertIn("final_pearson", result)
        self.assertIn("final_r2", result)
        # Further checks to ensure test set was used (e.g., check shape/dimensions)
        self.assertEqual(result["raw_pearson"].shape,
                         (10, 60))  # Should be one fold
        self.assertEqual(result["raw_r2"].shape, (10, 60))

    def test_ridge_torch_with_test_set(self):
        test_source = np.random.rand(50, 50)
        test_target = np.random.rand(50, 60)
        metric = TorchRidgeMetric()
        result = metric.compute(self.source, self.target,
                                test_source, test_target)
        self.assertIn("final_pearson", result)
        self.assertIn("final_r2", result)
        self.assertEqual(result["raw_pearson"].shape, (10, 60))
        self.assertEqual(result["raw_r2"].shape, (10, 60))

    if __name__ == '__main__':
        unittest.main()
