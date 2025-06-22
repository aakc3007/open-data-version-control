import unittest
from src.utils import evaluation


class TestEvaluation(unittest.TestCase):

    def test_accuracy_score(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        acc = evaluation.accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 1.0)

        y_pred = [0, 0, 2]
        acc = evaluation.accuracy_score(y_true, y_pred)
        self.assertLess(acc, 1.0)
