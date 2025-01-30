import unittest
from utils import expectedValue

class ExpectedValueTests(unittest.TestCase):
    def test_empty_array(self):
        y_true = []
        y_preds = []
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, 0)

    def test_true_positive(self):
        y_true = [1]
        y_preds = [1]
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, 1)

    def test_true_negative(self):
        y_true = [0]
        y_preds = [0]
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, 10)

    def test_false_negative(self):
        y_true = [1]
        y_preds = [0]
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, -10)

    def test_false_positive(self):
        y_true = [0]
        y_preds = [1]
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, 0)

    def test_combination(self):
        y_true = [0, 1, 1, 0]
        y_preds = [1, 0, 1, 0]
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, 1)

    def test_many_true_negatives(self):
        y_true = [0, 0, 0, 0]
        y_preds = [0, 0, 0, 0]
        result = expectedValue.expectedValue(y_true, y_preds)
        self.assertEqual(result, 40)

if __name__ == '__main__':
    unittest.main()
