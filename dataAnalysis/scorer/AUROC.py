from sklearn.metrics import roc_auc_score


def area_under_curve(est, X, y, sample_weight=None):
    auroc = roc_auc_score(y, est.predict_proba(X)[:, 1])
    return auroc

