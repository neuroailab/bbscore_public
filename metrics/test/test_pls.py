import unittest
import numpy as np
from metrics.pls import PLSMetric


class TestPLSMetric(unittest.TestCase):
    def setUp(self):
        self.source = np.random.rand(100, 50)
        self.target = np.random.rand(100, 60)

    def test_pls_metric(self):
        metric = PLSMetric()
        result = metric.compute(self.source, self.target)
        self.assertIn("final_pearson", result)
        self.assertIn("final_r2", result)

    def test_pls_metric_with_test_set(self):
        test_source = np.random.rand(20, 50)
        test_target = np.random.rand(20, 60)
        metric = PLSMetric()
        result = metric.compute(self.source, self.target,
                                test_source, test_target)
        self.assertIn("final_pearson", result)
        self.assertIn("final_r2", result)
        self.assertEqual(result['raw_pearson'].shape, (10, 60))
        self.assertEqual(result['raw_r2'].shape, (10, 60))


if __name__ == '__main__':
    unittest.main()
