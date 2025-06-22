import unittest
from unittest.mock import patch, MagicMock
from src.model_code import inference


class TestInference(unittest.TestCase):

    @patch('src.model_code.inference.pd.DataFrame.to_csv')
    @patch('src.model_code.inference.joblib.load')
    @patch('src.model_code.inference.pd.read_csv')
    @patch('src.model_code.inference.os.path.exists')
    def test_main(self, mock_exists, mock_read_csv, mock_load, mock_to_csv):
        # Mock path existence check
        mock_exists.return_value = True

        # Use real-looking DataFrames
        import pandas as pd
        X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        y = pd.DataFrame({'target': [0, 1]})
        mock_read_csv.side_effect = [X, y]

        # Mock model and scaler
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1]
        mock_scaler = MagicMock()
        mock_load.return_value = {'model': mock_model, 'scaler': mock_scaler}

        with patch('src.model_code.inference.scale_features') as mock_scale, \
             patch('src.model_code.inference.accuracy_score') as mock_acc:

            mock_scale.return_value = (X, mock_scaler)
            mock_acc.return_value = 1.0

            # Run inference
            inference.main(
                data_path_features='dummy_features.csv',
                data_path_target='dummy_target.csv',
                model_path='dummy_model.pkl',
                output_path='dummy_output.csv'
            )

            mock_model.predict.assert_called_once_with(X)
            mock_to_csv.assert_called_once()
