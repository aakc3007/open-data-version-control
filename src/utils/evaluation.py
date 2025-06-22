from sklearn.metrics import accuracy_score as sk_accuracy_score

def accuracy_score(y_true, y_pred):
    return sk_accuracy_score(y_true, y_pred)