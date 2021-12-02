import numpy as np
from sklearn.metrics import f1_score

from ml.utils import make_binary_prediction_label_pairs
from .prediction_score import recall
from .prediction_score import precision
from .prediction_score import f1
from .utils import write_model_scores
#
# Take gid to X dict and create prediction and label array
# where idx prediction is idx of label for same node
def _build_prediction_label_array_pairs(getognn,
                                        gid_to_prediction,
                                        gid_to_label,
                                        node_gid_to_partition,
                                        threshold=0.5,
                                        sigmoid=False):
    predictions = []
    labels = []
    for gid in gid_to_prediction.keys():


        if isinstance(gid_to_prediction[gid], list):
            #continue
            print("Empty pred, gid:", gid, 'pred',gid_to_prediction[gid])
            if node_gid_to_partition[gid] != 'train':

                predictions.append(gid_to_prediction[gid][1])
        else:
            if node_gid_to_partition[gid] != 'train':

                predictions.append(gid_to_prediction[gid])
                labels.append(gid_to_label[gid][1])
    predictions = np.array(predictions)
    labels = np.array(labels)
    cutoffs = np.arange(0.01, 0.98, 0.01)
    F1_log = {}
    max_f1 = 0
    opt_thresh = 0
    for thresh in cutoffs:

        threshed_arc_segmentation_logits = [logit > thresh for logit in labels]
        threshed_arc_predictions_proba = [logit > thresh for logit in predictions]

        F1_score_topo = f1_score(y_true=threshed_arc_segmentation_logits,
                                 y_pred=threshed_arc_predictions_proba, average=None)[-1]

        # self.F1_log[F1_score_topo] = thresh
        if F1_score_topo >= max_f1:
            max_f1 = F1_score_topo

            F1_log[max_f1] = thresh
            opt_thresh = thresh
    labels = [logit > opt_thresh for logit in labels]
    predictions = [logit > opt_thresh for logit in predictions]


    return predictions, labels, opt_thresh



def compute_getognn_metrics(getognn, threshold=0.5):
    predictions, labels, opt_thresh = _build_prediction_label_array_pairs(getognn,
                                                              getognn.node_gid_to_prediction,
                                                              getognn.node_gid_to_label,
                                                              getognn.node_gid_to_partition,
                                                              threshold=threshold)


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
    return predictions, labels, opt_thresh



def compute_prediction_metrics(model, predictions, labels, out_folder, threshold=0.5):
    predictions, labels = make_binary_prediction_label_pairs(predictions=predictions,
                                                             labels=labels,threshold=threshold)
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

def compute_opt_f1(model, predictions, labels, out_folder):
    for thresh in [0.3, 0.4, 0.5,0.6]:
        predictions, labels = make_binary_prediction_label_pairs(predictions=predictions,
                                                             labels=labels,threshold=thresh)
        #get dictionary of scores
        # keys are
        f1_dict = f1(predictions=predictions, labels=labels)
        recall_dict = recall(predictions=predictions, labels=labels)
        precision_dict = precision(predictions=predictions, labels=labels)

        f1_recall_precision = {'f1': f1_dict , 'recall': recall_dict , 'precision': precision_dict}
        for scoring_type in f1_recall_precision.keys():
            write_model_scores(model=model,threshold=str(thresh),
                               scoring=scoring_type, scoring_dict=f1_recall_precision[scoring_type],
                               out_folder=out_folder)