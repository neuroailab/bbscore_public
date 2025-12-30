import unittest
import numpy as np
from metrics.versa import VeRSAMetric


class TestVeRSAMetric(unittest.TestCase):
    def setUp(self):
        self.source = np.random.rand(100, 50)
        self.target = np.random.rand(100, 60)
        self.test_source = np.random.rand(50, 50)
        self.test_target = np.random.rand(50, 60)

    def test_versa_metric(self):
        metric = VeRSAMetric()
        result = metric.compute(self.source, self.target)
        self.assertIn("final_pearson", result)
        self.assertEqual(result['raw_pearson'].ndim, 1)
        self.assertEqual(result['raw_pearson'].shape[0], 10)

    def test_versa_metric_with_test_set(self):
        metric = VeRSAMetric()
        result = metric.compute(self.source, self.target,
                                self.test_source, self.test_target)
        self.assertIn("final_pearson", result)
        self.assertEqual(result['raw_pearson'].ndim, 1)
        self.assertEqual(result['raw_pearson'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
