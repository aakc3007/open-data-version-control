import pandas as pd
import joblib
import os
from src.utils.preprocessing import scale_features
from src.utils.evaluation import accuracy_score


def main(data_path_features: str,
         data_path_target: str,
         model_path: str,
         output_path: str):

    # Load inference data
    X_infer = pd.read_csv(data_path_features)
    y_true = pd.read_csv(data_path_target).values.ravel()

    # Ensure numeric type
    X_infer = X_infer.astype(float)

    # Load model and scaler
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_bundle = joblib.load(model_path)
    model = model_bundle['model']
    scaler = model_bundle['scaler']

    # Scale inference data
    X_scaled, _ = scale_features(X_infer, fit=False, scaler=scaler)

    # Predict
    y_pred = model.predict(X_scaled)

    # Evaluate
    acc = accuracy_score(y_true, y_pred)
    print(f"=== Inference Accuracy: {acc:.4f} ===")

    # Save predictions
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({'predicted': y_pred}).to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
