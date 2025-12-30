import unittest
import numpy as np
from metrics.rsa import RSAMetric


class TestRSAMetric(unittest.TestCase):
    def setUp(self):
        self.source = np.random.rand(100, 50)  # Example data
        self.target = np.random.rand(100, 60)

    def test_rsa_metric(self):
        metric = RSAMetric()
        result = metric.compute(self.source, self.target)
        # RSA should return a single float
        self.assertIsInstance(result, float)
        self.assertTrue(-1.0 <= result <= 1.0)  # Pearson correlation range


if __name__ == '__main__':
    unittest.main()
