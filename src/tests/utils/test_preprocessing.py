import unittest
import pandas as pd
from src.utils import preprocessing


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })

    def test_split_features_target(self):
        X, y = preprocessing.split_features_target(self.df, target_col='target')
        self.assertIn('feature1', X.columns)
        self.assertNotIn('target', X.columns)
        self.assertEqual(len(y), 3)

    def test_scale_features_fit(self):
        X = self.df[['feature1', 'feature2']]
        X_scaled, scaler = preprocessing.scale_features(X, fit=True)
        self.assertEqual(X_scaled.shape, (3, 2))

    def test_scale_features_transform(self):
        X = self.df[['feature1', 'feature2']]
        _, scaler = preprocessing.scale_features(X, fit=True)
        X_new = pd.DataFrame({'feature1': [10, 11], 'feature2': [12, 13]})
        X_scaled, _ = preprocessing.scale_features(X_new, fit=False, scaler=scaler)
        self.assertEqual(X_scaled.shape, (2, 2))
