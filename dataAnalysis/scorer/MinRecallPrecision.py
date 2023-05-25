from sklearn.metrics import recall_score, precision_score

# Important: set make_scorer-function around scoring-function
def min_recall_precision(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)
