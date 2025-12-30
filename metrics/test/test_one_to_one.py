import unittest
import numpy as np
from metrics.one_to_one import OneToOneMappingMetric


class TestOneToOneMappingMetric(unittest.TestCase):
    def setUp(self):
        # Generate deterministic data with 20 samples
        np.random.seed(42)
        self.source = np.random.rand(20, 5)  # 20 samples, 5 features
        self.target = np.random.rand(20, 5)
        self.test_source = np.random.rand(5, 5)  # Test set with 5 samples
        self.test_target = np.random.rand(5, 5)

    def test_one_to_one_metric(self):
        metric = OneToOneMappingMetric()
        result = metric.compute(self.source, self.target)
        self.assertIn("final_pearson", result)
        # Should only contain scores
        self.assertEqual(result['raw_pearson'].ndim, 1)
        self.assertEqual(result['raw_pearson'].shape[0], 10)  # 10 folds

    def test_one_to_one_metric_with_test_set(self):
        metric = OneToOneMappingMetric()
        result = metric.compute(self.source, self.target,
                                self.test_source, self.test_target)
        self.assertIn('final_pearson', result)
        self.assertEqual(result['raw_pearson'].ndim, 1)
        self.assertEqual(result['raw_pearson'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
