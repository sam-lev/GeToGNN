import numpy as np

from getognn import GeToGNN
from ml.utils import make_binary_prediction_label_pairs
from .prediction_score import recall
from .prediction_score import precision
from .prediction_score import f1
from .utils import write_model_scores
#
# Take gid to X dict and create prediction and label array
# where idx prediction is idx of label for same node
def _build_prediction_label_array_pairs(gid_to_prediction, gid_to_label, sigmoid=False):
    predictions = []
    labels = []
    for gid in gid_to_prediction.keys():
        if isinstance(gid_to_prediction[gid], list):
            #continue
            print("Empty pred, gid:", gid, 'pred',gid_to_prediction[gid])
            predictions.append(-1)#gid_to_prediction[gid][1])
        else:

            predictions.append(gid_to_prediction[gid])
        labels.append(gid_to_label[gid][1])
    predictions = np.array(predictions)
    labels = np.array(labels)
    if not sigmoid:
        predictions[predictions >= 0.5] = 1.
        predictions[predictions < 0.5] = 0.
    return predictions, labels



def compute_getognn_metrics(getognn):
    predictions, labels = _build_prediction_label_array_pairs(getognn.node_gid_to_prediction,
                                                              getognn.node_gid_to_label)


    #get dictionary of scores
    # keys are
    f1_dict = f1(predictions=predictions, labels=labels)
    recall_dict = recall(predictions=predictions, labels=labels)
    precision_dict = precision(predictions=predictions, labels=labels)

    f1_recall_precision = {'f1': f1_dict , 'recall': recall_dict , 'precision': precision_dict}
    for scoring_type in f1_recall_precision.keys():
        write_model_scores(model='getognn',
                           scoring=scoring_type, scoring_dict=f1_recall_precision[scoring_type],
                           out_folder=getognn.pred_session_run_path)


def compute_prediction_metrics(model, predictions, labels, out_folder):
    predictions, labels = make_binary_prediction_label_pairs(predictions=predictions,
                                                             labels=labels)
    #get dictionary of scores
    # keys are
    f1_dict = f1(predictions=predictions, labels=labels)
    recall_dict = recall(predictions=predictions, labels=labels)
    precision_dict = precision(predictions=predictions, labels=labels)

    f1_recall_precision = {'f1': f1_dict , 'recall': recall_dict , 'precision': precision_dict}
    for scoring_type in f1_recall_precision.keys():
        write_model_scores(model=model,
                           scoring=scoring_type, scoring_dict=f1_recall_precision[scoring_type],
                           out_folder=out_folder)