import unittest
import numpy as np
from metrics.bidirectional import BidirectionalMappingMetric


class TestBidirectionalMappingMetric(unittest.TestCase):
    def setUp(self):
        self.source = np.random.rand(100, 50)
        # Keep dimensions the same for simplicity
        self.target = np.random.rand(100, 50)
        self.test_source = np.random.rand(50, 50)
        self.test_target = np.random.rand(50, 50)

    def test_bidirectional_metric_r2(self):
        metric = BidirectionalMappingMetric(score_type="r2")
        result = metric.compute(self.source, self.target)
        self.assertIn("final_r2", result)
        self.assertEqual(result['raw_r2'].shape, (10, 2))

    def test_bidirectional_metric_pearson(self):
        metric = BidirectionalMappingMetric(score_type="pearson")
        result = metric.compute(self.source, self.target)
        self.assertIn("final_pearson", result)
        self.assertEqual(result['raw_pearson'].shape, (10, 2))

    def test_bidirectional_with_test_set_r2(self):
        metric = BidirectionalMappingMetric(score_type="r2")
        result = metric.compute(self.source, self.target,
                                self.test_source, self.test_target)
        self.assertIn("final_r2", result)
        self.assertEqual(result['raw_r2'].shape, (2, 1))

    def test_bidirectional_with_test_set_pearson(self):
        metric = BidirectionalMappingMetric(score_type="pearson")
        result = metric.compute(self.source, self.target,
                                self.test_source, self.test_target)
        self.assertIn("final_pearson", result)
        self.assertEqual(result['raw_pearson'].shape, (2, 1))


if __name__ == '__main__':
    unittest.main()
