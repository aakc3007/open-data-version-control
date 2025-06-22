from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_features_target(df, target_col='target'):
    return df.drop(columns=[target_col]), df[target_col]

def scale_features(X, fit=False, scaler=None):
    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    return scaler.transform(X), scaler
