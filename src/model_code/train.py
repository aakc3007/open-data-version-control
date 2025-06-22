import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.preprocessing import split_features_target, scale_features
from src.utils.evaluation import accuracy_score

def main(data_path, model_path, C=1.0, max_iter=1000):
    df = pd.read_csv(data_path)
    X, y = split_features_target(df, target_col='target')
    X_scaled, scaler = scale_features(X, fit=True)

    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_scaled, y)

    joblib.dump({'model': model, 'scaler': scaler}, model_path)

    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)
    print(f"Training complete. Accuracy on training set: {acc:.4f}")

if __name__ == '__main__':
    import fire
    fire.Fire(main)

## can also be called via python train.py --data_path ../src/data/iris_train.csv --model_path ../src/models/iris_model.pkl --C 0.5
