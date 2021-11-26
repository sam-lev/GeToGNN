from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, average_precision_score
from sklearn.metrics import f1_score
import numpy as np

from ml.features import get_points_from_vertices

# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
# #negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
def recall(predictions=None, labels=None, print_score=False):
    preds = predictions
    labels = labels
    recall_micro = recall_score(y_true=labels, y_pred=preds, average="micro")
    recall_macro = recall_score(y_true=labels, y_pred=preds, average="macro")
    recall_weighted = recall_score(y_true=labels, y_pred=preds, average="weighted")
    if print_score:
        print("Recall(micro) Score ( tp / (tp +fn) ): ", recall_micro)
        print("Recall(macro) Score ( tp / (tp +fn) ): ", recall_macro)
        print("Recall(weighted) Score ( tp / (tp +fn) ): ", recall_weighted)
    score_dict = {"weighted": recall_weighted, "binary": None, "micro": recall_micro, "macro": recall_macro}
    return score_dict


def get_topology_prediction_score(predicted, segmentation,
                                  gid_gnode_dict, node_gid_to_prediction, node_gid_to_label, pred_thresh=0.4,
                                  X = None, Y =None, ranges=None,use_torch=False):
    # toggle model to eval mode
    #model.eval()

    # turn off gradients since they will not be used here
    # this is to make the inference faster

    def _map_topo(predicted, segmentation,gid_gnode_dict, node_gid_to_prediction,
                  node_gid_to_label, pred_thresh=0.4,
                  X = None, Y =None, ranges=None):
        arc_predictions = []
        arc_segmentation_logits = []
        logits_predicted = np.zeros([X, Y])
        segmentations = np.zeros([X, Y])
        # run through several batches, does inference for each and store inference results
        # and store both target labels and inferenced scores
        # for image, segmentation in data_loader:
        # image = image.cuda()
        # image = image.unsqueeze(1)
        # logit_predicted = model(image)
        logits_predicted = predicted
        #np.concatenate((logits_predicted, predicted),
        #                                  axis=0)
        segmentation = segmentation
        # print('shape ', segmentation.shape)
        # print('shape mask ', mask.shape)

        segmentation[segmentation > 0] = 1
        #if len(segmentation.shape) != 4:
        #    segmentation = segmentation[None]  # .unsqueeze(1)#[:, :, :, :]

        pixel_predictions = []
        gt_pixel_segmentation = []
        def __get_prediction_correctness(segmentation, prediction, center_point):
            x = center_point[0]
            y = center_point[1]
            x = X - 2 if x + 1 >= X else x
            y = Y - 2 if y + 1 >= Y else y
            yield (prediction[ x, y], segmentation[ x, y])
            yield (prediction[ x - 1, y], segmentation[ x - 1, y])
            yield (prediction[ x, y - 1], segmentation[ x, y - 1])
            yield (prediction[ x - 1, y - 1], segmentation[ x - 1, y - 1])
            yield (prediction[ x + 1, y], segmentation[ x + 1, y])
            yield (prediction[ x, y + 1], segmentation[ x, y + 1])
            yield (prediction[ x + 1, y + 1], segmentation[ x + 1, y + 1])
            yield (prediction[ x + 1, y - 1], segmentation[ x + 1, y - 1])
            yield (prediction[ x - 1, y + 1], segmentation[ x - 1, y + 1])

        if ranges is not None:
            ranges = ranges.cpu().detach().numpy()
            print(ranges)
            x_range = ranges[0][0]  # list(map(int, ranges[0][1][0]))
            y_range = ranges[0][1]  # list(map(int, ranges[0][0][0]))

            print(y_range, "y_range")
            print(x_range, "x_range")
        else:
            x_range = [0, 0]
            y_range = [0, 0]

        for gid in gid_gnode_dict.keys():
            gnode = gid_gnode_dict[gid]
            points = gnode.points
            points = get_points_from_vertices([gnode])
            if ranges is not None:
                points[:, 0] = points[:, 0] #+ x_range[0]
                points[:, 1] = points[:, 1] #+ y_range[0]

            arc_pix_pred = []
            arc_pix_gt = []
            for p in points:
                x = p[1]  # + x_range[0]
                y = p[0]  # + y_range[0]
                arc_pix_pred += [pred[0] > pred_thresh for pred in __get_prediction_correctness(segmentation,
                                                                                                   logits_predicted,
                                                                                                   (x, y))]
                arc_pix_gt += [pred[1] > pred_thresh for pred in __get_prediction_correctness(segmentation,
                                                                                                          logits_predicted,
                                                                                                           (x, y))]
            pixel_predictions += arc_pix_pred             # pixel preds
            gt_pixel_segmentation += arc_pix_gt



            spread_pred = np.bincount(arc_pix_pred, minlength=2)

            pred_unet_val = spread_pred[1] / np.sum(spread_pred)
            pred_unet = spread_pred[1] / np.sum(spread_pred) > pred_thresh
            arc_predictions.append(pred_unet)                           # average pix pred --> arc

            spread_seg = np.bincount(arc_pix_gt, minlength=2)
            gt_val = spread_seg[1] / np.sum(spread_seg)
            gt = spread_seg[1] / np.sum(spread_seg) > pred_thresh
            arc_segmentation_logits.append(gt)                                    # gt avg pix --> arc

            node_gid_to_prediction[gid] = [1.0 - pred_unet_val, pred_unet_val]
            node_gid_to_label[gid] = [1.0 - gt_val, gt_val]

            # returns a list of scores, one for each of the labels
        print("    * len pixpred", len(pixel_predictions))
        print("    * len pixpred", len(gt_pixel_segmentation))

        segmentations = np.array(arc_segmentation_logits)
        logits_predicted = np.array(arc_predictions)
        binary_logits_predicted = logits_predicted  # .astype(int)

        print(segmentations[0:10])
        print(" logits,", logits_predicted[0:10])

        return gt_pixel_segmentation , pixel_predictions, \
               arc_segmentation_logits, arc_predictions, node_gid_to_prediction, node_gid_to_label#
        #binary_logits_predicted, node_gid_to_prediction, node_gid_to_label


    if use_torch:
        import torch
        with torch.no_grad():
            gt_pixel_segmentation, pixel_predictions, \
            arc_segmentation_logits, arc_predictions, node_gid_to_prediction, node_gid_to_label = _map_topo(predicted, segmentation,
                                                                                       gid_gnode_dict,
                                                                                       node_gid_to_prediction=node_gid_to_prediction,
                                                                                       node_gid_to_label=node_gid_to_label,
                                                                                       pred_thresh=pred_thresh,
                                                                                       X=Y, Y=Y, ranges=None)
    else:
        gt_pixel_segmentation, pixel_predictions, \
        arc_segmentation_logits, arc_predictions, node_gid_to_prediction, node_gid_to_label = _map_topo(predicted, segmentation,
                                                                                       gid_gnode_dict,
                                                                                       node_gid_to_prediction=node_gid_to_prediction,
                                                                                       node_gid_to_label=node_gid_to_label,
                                                                                       pred_thresh=pred_thresh,
                                                                                       X=X, Y=Y, ranges=None)
    return f1_score(arc_segmentation_logits, arc_predictions), gt_pixel_segmentation , pixel_predictions, \
               gt_pixel_segmentation , pixel_predictions, node_gid_to_prediction, node_gid_to_label


# tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
def precision(predictions=None, labels=None, print_score=False):
    preds = predictions
    labels = labels
    precision_micro = precision_score(y_true=labels, y_pred=preds, average="micro")
    precision_macro = precision_score(y_true=labels, y_pred=preds, average="macro")
    precision_weighted = precision_score(y_true=labels, y_pred=preds, average="weighted")
    precision_avg = average_precision_score(y_true=labels, y_score=preds)
    if print_score:
        print("Precision Score (micro): ", precision_micro)
        print("Precision Score (macro): ", precision_macro)
        print("Precision Score (weighted): ", precision_weighted)
        print("Precision Score (average): ", precision_avg)
    score_dict = {"weighted": precision_weighted, "average": precision_avg,
                  "micro": precision_micro, "macro": precision_macro}
    return score_dict


# F1 = 2 * (precision * recall) / (precision + recall)
def f1(predictions=None, labels=None, print_score=False):
    preds = predictions
    labels = labels
    f1_w = f1_score(labels, preds, average="weighted")
    f1_b = f1_score(labels, preds, average="binary")
    f1_mi = f1_score(labels, preds, average="micro")
    f1_ma = f1_score(labels, preds, average="macro")
    f1_class = f1_score(labels, preds, average=None)
    print("    * F1 class", f1_class)
    f1_class = f1_class[-1]
    fs = [str(f1_w),str(f1_b), str(f1_mi),str(f1_ma)]
    if print_score:
        print("F1 Score (weighted): ", f1_w)
        print("F1 Score (binary):   ", f1_b)
        print("F1 Score (micro): ", f1_mi)
        print("F1 Score (macro): ", f1_ma)
    score_dict = {"weighted": f1_w, "binary": f1_b, "micro": f1_mi, "macro": f1_ma, "class":f1_class}
    return score_dict

def compute_quality_metrics(predictions=None, labels=None):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    p = precision(predictions=predictions, labels=labels)
    r = recall(predictions=predictions, labels=labels)
    fs = f1(predictions=predictions, labels=labels)
    return p,r,fs