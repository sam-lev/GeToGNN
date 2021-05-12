from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, average_precision_score
from sklearn.metrics import f1_score


# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
# #negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
def recall(predictions=None, labels=None):
    preds = predictions
    labels = labels
    recall_micro = recall_score(y_true=labels, y_pred=preds, average="micro")
    recall_macro = recall_score(y_true=labels, y_pred=preds, average="macro")
    recall_weighted = recall_score(y_true=labels, y_pred=preds, average="weighted")
    print("Recall(micro) Score ( tp / (tp +fn) ): ", recall_micro)
    print("Recall(macro) Score ( tp / (tp +fn) ): ", recall_macro)
    print("Recall(weighted) Score ( tp / (tp +fn) ): ", recall_weighted)
    score_dict = {"weighted": recall_weighted, "binary": None, "micro": recall_micro, "macro": recall_macro}
    return score_dict





# tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
def precision(predictions=None, labels=None):
    preds = predictions
    labels = labels
    precision_micro = precision_score(y_true=labels, y_pred=preds, average="micro")
    precision_macro = precision_score(y_true=labels, y_pred=preds, average="macro")
    precision_weighted = precision_score(y_true=labels, y_pred=preds, average="weighted")
    precision_avg = average_precision_score(y_true=labels, y_score=preds)
    print("Precision Score (micro): ", precision_micro)
    print("Precision Score (macro): ", precision_macro)
    print("Precision Score (weighted): ", precision_weighted)
    print("Precision Score (average): ", precision_avg)
    score_dict = {"weighted": precision_weighted, "average": precision_avg,
                  "micro": precision_micro, "macro": precision_macro}
    return score_dict


# F1 = 2 * (precision * recall) / (precision + recall)
def f1(predictions=None, labels=None):
    preds = predictions
    labels = labels
    f1_w = f1_score(labels, preds, average="weighted")
    print("F1 Score (weighted): ", f1_w)
    f1_b = f1_score(labels, preds, average="binary")
    print("F1 Score (binary):   ", f1_b)
    f1_mi = f1_score(labels, preds, average="micro")
    print("F1 Score (micro): ", f1_mi)
    f1_ma = f1_score(labels, preds, average="macro")
    print("F1 Score (macro): ", f1_ma)
    fs = [str(f1_w),str(f1_b), str(f1_mi),str(f1_ma)]
    score_dict = {"weighted":f1_w, "binary": f1_b, "micro": f1_mi, "macro":f1_ma}
    return score_dict

def compute_quality_metrics(predictions=None, labels=None):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    p = precision(predictions=predictions, labels=labels)
    r = recall(predictions=predictions, labels=labels)
    fs = f1(predictions=predictions, labels=labels)
    return p,r,fs