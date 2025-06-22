import unittest
from unittest.mock import patch, MagicMock
from src.model_code import train


class TestTrain(unittest.TestCase):

    @patch('src.model_code.train.pd.read_csv')
    @patch('src.model_code.train.joblib.dump')
    def test_main(self, mock_dump, mock_read_csv):
        # Realistic dummy DataFrame for X and y
        import pandas as pd
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        mock_read_csv.return_value = df

        with patch('src.model_code.train.split_features_target') as mock_split, \
             patch('src.model_code.train.scale_features') as mock_scale, \
             patch('src.model_code.train.LogisticRegression') as mock_lr:

            # Return consistent X and y
            X = df[['feature1', 'feature2']]
            y = df['target']
            mock_split.return_value = (X, y)

            # Just return X as "scaled"
            mock_scale.return_value = (X, MagicMock())

            model_instance = MagicMock()
            model_instance.predict.return_value = [0, 1, 0]  # same shape as y
            mock_lr.return_value = model_instance

            # Call the function
            train.main("dummy.csv", "dummy_model.pkl", C=1.0, max_iter=100)

            model_instance.fit.assert_called_once()
            model_instance.predict.assert_called_once_with(X)
            mock_dump.assert_called_once()
