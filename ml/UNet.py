import os
import sys
import random
import warnings

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import timeit
from sklearn.metrics import f1_score
import matplotlib as mplt
from copy import deepcopy

from attributes import Attributes
from mlgraph import MLGraph
from ml.features import get_points_from_vertices
from metrics.model_metrics import compute_prediction_metrics
from data_ops.collect_data import collect_training_data, compute_geomsc, collect_datasets
from ml.utils import get_partition_feature_label_pairs
from data_ops.utils import dbgprint as dprint
from compute_multirun_metrics import multi_run_metrics
from data_ops.utils import tile_region

from localsetup import LocalSetup
LocalSetup = LocalSetup()




def get_score_model(model, data_loader, X = None, Y =None):
    # toggle model to eval mode
    model.eval()

    # turn off gradients since they will not be used here
    # this is to make the inference faster

    logits_predicted = np.zeros([0, 1, X, Y])
    segmentations = np.zeros([0, 1, X, Y])
    # run through several batches, does inference for each and store inference results
    # and store both target labels and inferenced scores
    for image, segmentation, _ in data_loader:
        image = image.cuda()
        image = image.unsqueeze(1)
        logit_predicted = model(image)
        if len(logit_predicted.shape) != 4:
            logit_predicted = logit_predicted[None]
        logits_predicted = np.concatenate((logits_predicted, logit_predicted.cpu().detach().numpy() ),
                                          axis=0)
        # print('shape ', segmentation.shape)
        # print('shape mask ', mask.shape)

        segmentation[segmentation > 0] = 1
        segmentation = segmentation.unsqueeze(1)#[:, :, :, :]
        # mask = mask.permute(1,2,0)
        # print(segmentation)
        segmentations = np.concatenate((segmentations, segmentation.cpu().detach().numpy() ), axis=0)
        # returns a list of scores, one for each of the labels
        segmentations = segmentations.reshape([-1])
        logits_predicted = logits_predicted.reshape([-1]) > 0
        binary_logits_predicted = logits_predicted#.astype(int)
    return f1_score(segmentations, logits_predicted) , segmentations, binary_logits_predicted

def get_image_prediction_score(predicted, segmentation, X = None, Y =None):
    # toggle model to eval mode
    #model.eval()

    # turn off gradients since they will not be used here
    # this is to make the inference faster

    logits_predicted = np.zeros([0, 1, X, Y])
    segmentations = np.zeros([0, 1, X, Y])
    # run through several batches, does inference for each and store inference results
    # and store both target labels and inferenced scores
    #for image, segmentation in data_loader:
    #image = image.cuda()()
    #image = image.unsqueeze(1)
    #logit_predicted = model(image)
    logits_predicted = np.concatenate((logits_predicted, predicted ),
                                      axis=0)
    # print('shape ', segmentation.shape)
    # print('shape mask ', mask.shape)

    segmentation[segmentation > 0] = 1
    if len(segmentation.shape) != 4:
        segmentation = segmentation[None]#.unsqueeze(1)#[:, :, :, :]


    # print(segmentation)
    segmentations = np.concatenate((segmentations, segmentation ),
                                   axis=0)
    # returns a list of scores, one for each of the labels
    segmentations = segmentations.reshape([-1]) > 0
    logits_predicted = logits_predicted.reshape([-1]) > 0
    binary_logits_predicted = logits_predicted#.astype(int)
    return f1_score(segmentations, logits_predicted) , segmentations, binary_logits_predicted

def get_topology_prediction_score(predicted, segmentation,
                                  gid_gnode_dict, node_gid_to_prediction, node_gid_to_label,
                                  msc_logit_map = None, pred_thresh=0.4,
                                  X = None, Y =None, ranges=None):
    # toggle model to eval mode
    #model.eval()

    # turn off gradients since they will not be used here
    # this is to make the inference faster

    arc_pixel_predictions_proba = []
    arc_segmentation_proba = []
    arc_pixel_predictions = []
    segmentation_logits = []

    # run through several batches, does inference for each and store inference results
    # and store both target labels and inferenced scores

    # plt.figure()
    # plt.title("Predicted Segmentation")
    # plt.imshow(predicted, cmap=plt.cm.Greys_r)
    # plt.figure()
    # plt.title("Segmentation")
    # plt.imshow(segmentation, cmap=plt.cm.Greys_r)
    # plt.show()

    logits_predicted = predicted#np.concatenate((logits_predicted, predicted),
    #                                  axis=0)
    segmentation = segmentation
    # print('shape ', segmentation.shape)
    # print('shape mask ', mask.shape)

    # segmentation[segmentation > 0] = 1
    #if len(segmentation.shape) != 4:
    #    segmentation = segmentation[None]#.unsqueeze(1)#[:, :, :, :]




    def __get_prediction_correctness(segmentation, prediction, center_point):
        x = center_point[0]
        y = center_point[1]
        x = X - 2 if x + 1 >= X else x
        y = Y - 2 if y + 1 >= Y else y
        yield (prediction[x, y], segmentation[x, y])
        yield (prediction[x - 1, y], segmentation[x - 1, y])
        yield (prediction[x, y - 1], segmentation[x, y - 1])
        yield (prediction[x - 1, y - 1], segmentation[x - 1, y - 1])
        yield (prediction[x + 1, y], segmentation[x + 1, y])
        yield (prediction[x, y + 1], segmentation[x, y + 1])
        yield (prediction[x + 1, y + 1], segmentation[x + 1, y + 1])
        yield (prediction[x + 1, y - 1], segmentation[x + 1, y - 1])
        yield (prediction[x - 1, y + 1], segmentation[x - 1, y + 1])

    if ranges is not None:
        #ranges = ranges.cpu().detach().numpy()
        x_range = [ranges[0], ranges[1]]# list(map(int, ranges[0][1][0]))
        y_range = [ranges[2], ranges[3]]#list(map(int, ranges[0][0][0]))

    else:
        x_range = [0,0]
        y_range=[0,0]

    pixel_predictions =[]
    gt_pixel_labels = []

    for gid in gid_gnode_dict.keys():
        gnode = gid_gnode_dict[gid]
        label = node_gid_to_label[gid]
        label = label if type(label) != list else label[1]
        points = gnode.points
        points = get_points_from_vertices([gnode])
        if ranges is not None:
            points[:,0] = points[:,0] #- x_range[0]
            points[:,1] = points[:,1] #- y_range[0]

        arc_predictions = []
        arc_segmentation_logits = []
        arc_predictions_proba = []
        arc_segmentation_logits_proba = []
        for p in points:
            x = p[1] #+ x_range[0]
            y = p[0] #+ y_range[0]
            arc_predictions_proba += [pred[0] for pred in __get_prediction_correctness(segmentation,
                                                                           logits_predicted,(x,y))]
            arc_segmentation_logits_proba += [label for i in range(9)]#
            #[pred[1] for pred in __get_prediction_correctness(segmentation,
            #                                                               logits_predicted,(x,y))]

            arc_segmentation_logits += [ logit > pred_thresh for logit in arc_segmentation_logits_proba]
            arc_predictions +=  [logit > pred_thresh for logit in arc_predictions_proba]

            if msc_logit_map is not None:
                x = X - 2 if x + 1 >= X else x
                y = Y - 2 if y + 1 >= Y else y
                msc_logit_map[x,y] = arc_predictions_proba[-1]


        arc_pixel_predictions_proba += np.average(arc_predictions_proba)
        arc_segmentation_proba += np.average(arc_segmentation_logits_proba)

        pixel_predictions += arc_predictions
        gt_pixel_labels += arc_segmentation_logits

        spread_pred = np.bincount(arc_predictions,minlength=2)

        pred_unet_val = spread_pred[1]/np.sum(spread_pred)
        pred_unet = spread_pred[1]/np.sum(spread_pred) > pred_thresh
        arc_pixel_predictions.append(pred_unet)



        spread_seg = np.bincount(arc_segmentation_logits,minlength=2)
        gt_val = spread_seg[1] / np.sum(spread_seg)
        gt = spread_seg[1] / np.sum(spread_seg) > pred_thresh
        segmentation_logits.append(gt)

        node_gid_to_prediction[gid] = [1.0-pred_unet_val , pred_unet_val]
        #node_gid_to_label[gid] = [1.0-gt_val, gt_val]
        gnode.prediction = [1.0-pred_unet_val , pred_unet_val]
        #gnode.label = label #gt_val


        # returns a list of scores, one for each of the labels
    segmentations = np.array(segmentation_logits)
    logits_predicted = np.array(arc_pixel_predictions)
    binary_logits_predicted = logits_predicted#.astype(int)



    if msc_logit_map is None:
        return f1_score(gt_pixel_labels, pixel_predictions) , \
               arc_segmentation_proba, arc_pixel_predictions_proba,\
               segmentation_logits, arc_pixel_predictions, node_gid_to_prediction, node_gid_to_label
    else:
        return f1_score(gt_pixel_labels, pixel_predictions), \
               arc_segmentation_proba, arc_pixel_predictions_proba,\
               segmentation_logits, arc_pixel_predictions,msc_logit_map,\
               node_gid_to_prediction, node_gid_to_label

# Dataset class for the retina dataset
# each item of the dataset is a tuple with three items:
# - the first element is the input image to be segmented
# - the second element is the segmentation ground truth image
# - the third element is a mask to know what parts of the input image should be used (for training and for scoring)
class dataset():
    def transpose_first_index(self, x, with_hand_seg=False, with_range=True):
        if with_range:
            x2 =(x[0], x[1], x[2])#(np.transpose(x[0], [1, 0]), x[1])
            #, np.transpose(x[2], [2, 0, 1]))
        else:
            x2 =(x[0], x[1])#(np.transpose(x[0], [1, 0]), x[1])#(np.transpose(x[0], [2, 0, 1]), np.transpose(x[1], [2, 0, 1]), np.transpose(x[2], [2, 0, 1]),
            #      np.transpose(x[3], [2, 0, 1]))
        return x2

    def __init__(self, data_array, split='train', do_transform=False,
                 with_hand_seg=False, with_range=True):

        self.with_hand_seg = with_hand_seg
        self.with_range = with_range

        indexes_this_split = np.arange(len(data_array))#get_split(np.arange(len(retina_array), dtype=np.int), split)
        self.data_array = [self.transpose_first_index(data_array[i],
                                                      self.with_hand_seg,
                                                      with_range=self.with_range) for i in
                             indexes_this_split]



        self.split = split
        self.do_transform = do_transform

    def __getitem__(self, index):
        if self.with_range:
            sample = [self.data_array[index][0],
                      self.data_array[index][1],
                      self.data_array[index][2]]
        else:
            sample = [self.data_array[index][0],
                      self.data_array[index][1]]


        return sample

    def __len__(self):
        return len(self.data_array)

    def get_images(self):
        return np.array([np.array(samp[0],dtype=np.float32) for samp in self.data_array ], dtype=np.float32)

    def get_segmentations(self):
        return np.array([np.array(samp[1],dtype=np.uint8) for samp in self.data_array], dtype=np.uint8)



###
#
#
#


#
#   U-Net
#
#


class UNetwork( MLGraph):






    def get_data_crops(self, image=None, x_range=None,y_range=None, with_range=True,train_set=True,
                       dataset=[], resize=False, full_img=False,growth_radius = 1):



        segmentations = []


        seg = np.zeros(self.image.shape[:2]).astype(np.uint8)

        x_full = [0, self.image.shape[0]]
        y_full = [0, self.image.shape[1]]
        full_bounds = (x_full, y_full)
        seg = self.generate_pixel_labeling(full_bounds,
                                           seg_image=seg, growth_radius=growth_radius,
                                           train_set=train_set)

        range_group = zip(x_range, y_range)

        for x_rng, y_rng in range_group:
            bounds = (x_rng,y_rng)
            #train_im_crop = deepcopy(image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])
            #seg = np.zeros(train_im_crop.shape).astype(np.uint8)
            #if full_img:

            train_im_crop = deepcopy(self.image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])
            seg_crop = deepcopy(seg[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])
            if full_img:
                return seg
            if with_range:
                dataset.append((train_im_crop, seg_crop, (x_rng, y_rng)))
            else:
                dataset.append((train_im_crop, seg_crop))

        return dataset

    def generate_pixel_labeling(self, train_region, seg_image=None ,growth_radius = 1, train_set=True):
        x_box = train_region[0]
        y_box = train_region[1]
        dim_train_region = (   int(x_box[1] - x_box[0]),int(y_box[1]-y_box[0]) )
        train_im = seg_image#np.zeros(dim_train_region)
        X = train_im.shape[0]
        Y = train_im.shape[1]

        def __box_grow_label(train_im, center_point, label, train_set):
            x = center_point[0]
            y = center_point[1]
            x[x + 1 >= X] = X - 2
            y[y+1>= Y] = Y - 2
            train_im[x , y] = label
            train_im[x-1,y] = label
            train_im[x, y-1] = label
            train_im[x - 1, y-1] = label
            train_im[x+1,y] = label
            train_im[x, y+1] = label
            train_im[x + 1, y+1] = label
            train_im[x - 1, y + 1] = label
            train_im[x + 1, y - 1] = label

        for gid in self.gid_gnode_dict.keys():
            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            if train_set:
                self.node_gid_to_partition[gid] = 'test'
            label = label
            points = gnode.points
            p1 = points[0]
            p2 = points[-1]
            in_box = False
            not_all = False
            end_points = (p1, p2)
            points = get_points_from_vertices([gnode])
            interior_points = []
            for p in points:
                if x_box[0] < p[1] < x_box[1] and y_box[0] < p[0] < y_box[1]:
                    in_box = True
                    interior_points.append(p)

                else:
                    not_all = True
            if not_all and in_box:
                if train_set:
                    self.node_gid_to_partition[gid] = 'train'
                points = np.array(interior_points)
            elif not_all:
                continue

            mapped_y = points[:,0] - y_box[0]
            mapped_x = points[:,1] - x_box[0]
            if int(label[1]) == 1:
                train_im[mapped_x, mapped_y] = int(1)
                #__box_grow_label(train_im, (mapped_x,mapped_y), int(1))
            #else:
            #    train_im[mapped_x, mapped_y] = int(0)
            #    __box_grow_label(train_im, (mapped_x, mapped_y), int(0))
        train_im = ndimage.maximum_filter(train_im, size=growth_radius)
        n_samples = dim_train_region[0] * dim_train_region[1]
        n_classes = 2
        class_bins = np.bincount(train_im.astype(np.int64).flatten())
        self.class_weights = 1.0#n_samples / (n_classes * class_bins)
        return train_im.astype(np.int8)

    def save_image(self, image=None, dirpath=None, gt_seg=None, pred_seg = None,
                   image_seg_set = None, as_grey=False, INTERACTIVE=False):
        if image_seg_set is not None:
            #shape_im = image.shape
            #shape_gt_seg = gt_seg.shape

            if len(image_seg_set[0].shape) == 4:
                image = image_seg_set[0][0, 0, :, :]
                gt_seg = image_seg_set[1][0, 0, :, :]
                if len(image_seg_set) > 2:
                    # shape_pred_seg = pred_seg.shape
                    pred_seg = image_seg_set[2][0, 0, :, :]
            else:
                image = image_seg_set[0]
                gt_seg = image_seg_set[1]
                if len(image_seg_set) > 2:
                    # shape_pred_seg = pred_seg.shape
                    pred_seg = image_seg_set[2]

        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
        ax[0].imshow(image)
        #ax[0].contour(training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[0].set_title('Image')
        ax[1].imshow(gt_seg)
        #ax[1].contour(training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[1].set_title('ground truth')
        ax[2].imshow(pred_seg)
        #ax[2].contour(training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[2].set_title('prediction')
        #ax[3].imshow(pred_labels)
        #ax[3].contour(training_reg_bg, [0.0, 0.15], linewidths=0.5)
        #ax[3].set_title('unet2lines-prediction')
        fig.tight_layout()
        if INTERACTIVE:
            plt.show()
        else:
            plt.savefig( "unet_imgs.pdf")

    def see_image(self, image=None, gt_seg=None, pred_seg = None, image_seg_set = None,
                  dirpath=None, as_grey=False, save=True, names=None):

        if image_seg_set is not None:
            #shape_im = image.shape
            #shape_gt_seg = gt_seg.shape
            if len(image_seg_set[0].shape)==4:
                image = image_seg_set[0][0,0,:,:]
                gt_seg = image_seg_set[1][0,0,:,:]
                if len(image_seg_set) > 2:
                    #shape_pred_seg = pred_seg.shape
                    pred_seg = image_seg_set[2][0,0,:,:]
            else:
                image = image_seg_set[0]
                gt_seg = image_seg_set[1]
                if len(image_seg_set) > 2:
                    # shape_pred_seg = pred_seg.shape
                    pred_seg = image_seg_set[2]
        og_im_name = 'subset_image'
        gt_name = 'gt_seg'
        pred_name = 'pred_seg'
        if names is not None:
            og_im_name = names[0]
            gt_name = names[1]
            pred_name = names[2]
        if image is not None:
            plt.figure()
            plt.title("Input Image")
            if as_grey:
                if not save:
                    plt.imshow(image,cmap=mplt.cm.Greys_r)
                else:
                    plt.imsave(os.path.join(dirpath,og_im_name + '.png'),image, cmap=mplt.cm.Greys_r)
            else:
                if not save:
                    plt.imshow(image)
                else:
                    plt.imsave(os.path.join(dirpath,og_im_name + '.png'),image)
            if not save:
                plt.show()
            plt.close()
        if gt_seg is not None:
            plt.figure()
            plt.title("Input Segmentation")
            if as_grey:
                if not save:
                    plt.imshow(gt_seg,  cmap=mplt.cm.Greys_r)
                else:
                    plt.imsave(os.path.join(dirpath,gt_name + '.png'),gt_seg, cmap=mplt.cm.Greys_r)
            else:
                if not save:
                    plt.imshow(gt_seg)
                else:
                    plt.imsave(os.path.join(dirpath,gt_name + '.png'),gt_seg)
            if not save:
                plt.show()
        if pred_seg is not None:
            plt.figure()
            plt.title("Predicted Segmentation")
            if as_grey:
                if not save:
                    plt.imshow(pred_seg,cmap=plt.cm.Greys_r)

                else:
                    plt.imsave(os.path.join(dirpath,pred_name + '.png'),pred_seg,  cmap=mplt.cm.Greys_r)

            else:
                if not save:
                    plt.imshow(pred_seg)
                else:
                    plt.imsave(os.path.join(dirpath,pred_name + '.png'),pred_seg)
            if not save:
                plt.show()



    def mean_iou(self, y_true, y_pred):
        I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
        U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
        return tf.reduce_mean(I / U)



    def mean_iou_old(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.cast(y_pred > t, tf.int32)

            m = tf.keras.metrics.MeanIoU(num_classes=2)
            m.update_state(y_true, y_pred_)
            score, up_opt = m.result()
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)


    def __init__(self,run_num=0, parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None,
                 model_name=None, load_feature_graph_name=False,image=None, **kwargs):

        self.type = 'unet'

        k = [parameter_file_number, run_num, geomsc_fname_base, label_file, image,
             model_name, load_feature_graph_name]
        st_k = ['parameter_file_number', 'run_num', 'geomsc_fname_base', 'label_file', 'image',
                'model_name', 'load_feature_graph_name']
        for name, attr in zip(st_k, k):
            kwargs[name] = attr

        MLGraph.__init__(self, **kwargs)

        #, **kwargs)

    def set_attributes(self,in_channel, out_channel, skip_connect=True, kernnel_size=2,
                 ground_truth_label_file=None,run_num=0, parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None, growth_radius=2,
                 model_name=None, load_feature_graph_name=False,image=None,
                 X_BOX =None,Y_BOX=None,boxes=None,all_boxes=None,
                 X_BOX_all =None,Y_BOX_all=None,
                 training_size=None, region_list=None,
                       BEGIN_LOADING_FEATURES=False,
                       **kwargs):









        self.trained_model = None


        max_val = np.max(self.image)
        min_val = np.min(self.image)
        self.image = (self.image - min_val) / (max_val - min_val)

        self.training_size = training_size

        self.growth_radius = growth_radius


        self.running_best_model = None

        self.kernnel_size = kernnel_size
        self.expansion_out_padding = 0
        self.expansion_padding = 0# 1#'same'#0

        #
        # Training / val /test sets
        #
        self.subgraph_sample_set = {}
        self.subgraph_sample_set_ids = {}
        self.positive_arc_ids = set()
        self.selected_positive_arc_ids = set()
        self.negative_arc_ids = set()
        self.selected_negative_arc_ids = set()
        self.positive_arcs = set()
        self.selected_positive_arcs = set()
        self.negative_arcs = set()
        self.selected_negative_arcs = set()

        if BEGIN_LOADING_FEATURES:
            self.params['load_features'] = True
            self.params['write_features'] = False
            self.params['load_features'] = True
            self.params['write_feature_names'] = False
            self.params['save_filtered_images'] = False
            self.params['collect_features'] = False
            self.params['load_preprocessed'] = True
            self.params['load_geto_attr'] = True
            self.params['load_feature_names'] = True

        if self.params['load_geto_attr']:
            if self.params['geto_as_feat']:
                self.load_geto_features()
                self.load_geto_feature_names()
        else:
            if self.params['geto_as_feat']:
                self.build_geto_adj_list(influence_type=self.params['geto_influence_type'])
                self.write_geto_features(self.session_name)
                self.write_geto_feature_names()

        # features
        if not self.params['load_features']:
            self.compile_features(include_geto=self.params['geto_as_feat'])
            self.write_gnode_features(self.session_name)
            self.write_feature_names()
        else:
            print("    * * * * LOADING IN FEATURES")
            self.load_gnode_features()
            self.load_feature_names()
            #if 'geto' in self.params['aggregator']:
            #    self.load_geto_features()
            #    self.load_geto_feature_names()

        if self.params['write_features']:
            self.write_gnode_features(self.session_name)

        if self.params['write_feature_names']:
            self.write_feature_names()


        if self.params['write_feature_names']:
            if self.params['geto_as_feat']:
                self.write_geto_feature_names()
        if self.params['write_features']:
            if self.params['geto_as_feat']:
                self.write_geto_features(self.session_name)
        # training info, selection, partition train/val/test
        self.label_file = ground_truth_label_file
        self.read_labels_from_file(file=ground_truth_label_file)

        self.data_array = self.train_dataloader

        # self.region_list = region_list


        self.box_regions = boxes
        self.all_boxes = all_boxes
        self.X_BOX = X_BOX
        self.Y_BOX = Y_BOX
        self.X_BOX_all = X_BOX_all
        self.Y_BOX_all = Y_BOX_all









    def model(self, input=None):
        self.UNET_FORCE_RECOMPUTE = False
        UNET_IMSIZE = self.train_dataset[0][0].shape[0]



        self.IMG_WIDTH = UNET_IMSIZE  # for faster computing on kaggle
        self.IMG_HEIGHT = UNET_IMSIZE  # for faster computing on kaggle
        self.IMG_CHANNELS = 1


        self.inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = Lambda(lambda x: x)(self.inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        self.outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)


    def __group_pairs(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i: i + 2])

    def collect_boxes(self, resize=False, run_num=-1, training_set=False):# region_list,
        # X_BOX = []
        # Y_BOX = []
        # print(' boxesboxe', self.box_regions)
        # box_sets = []
        # for box in self.box_regions:
        #     box_set = tile_region(step_X=64, step_Y=64, step=0.5,
        #                           Y_START=box[0], Y_END=box[1],
        #                           X_START=box[2], X_END=box[3])
        #     box_sets.append(box_set)
        # print("    *: BOX TILE SETS", box_sets)
        # for box_pair in box_sets:
        #     for box in box_pair:
        #         X_BOX.append(box[0])
        #         Y_BOX.append(box[1])
        # self.X_BOX = X_BOX
        # self.Y_BOX = Y_BOX





        self.all_region_boxes =self.all_boxes
        self.training_boxes = list(zip(self.X_BOX, self.Y_BOX))


        if training_set:
            train_dataset = self.get_data_crops(self.image, x_range=self.X_BOX,
                                               with_range = not training_set,
                                                    y_range=self.Y_BOX,
                                               dataset=[],
                                               resize=resize,
                                               growth_radius = self.growth_radius,
                                               train_set=training_set)
            #self.see_image(gt_seg=test_dataset[0][1], save=False)





            self.pred_run_path = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                              self.params['write_folder'],
                                              'runs')

            if not os.path.exists(self.pred_run_path):
               os.makedirs(os.path.join(self.pred_run_path))

            self.run_num = self.training_size
            self.pred_session_run_path = os.path.join(self.pred_run_path,
                                                      str(self.training_size))
            if not os.path.exists(self.pred_session_run_path):
                os.makedirs(os.path.join(self.pred_session_run_path))







            inf_or_train_dataset = dataset(train_dataset, do_transform= False,
                                            with_hand_seg=False, with_range= not training_set)

            val_dataset = train_dataset

        else:#if not training_set:
            self.training_labels = self.get_data_crops(self.image, x_range=[[0, self.image.shape[0]]],
                                               y_range=[[0, self.image.shape[1]]],
                                                with_range= not training_set,
                                               full_img=True,
                                                dataset=[],
                                                resize=resize,
                                               train_set=training_set,
                                               growth_radius = self.growth_radius)

            inf_crops = self.get_data_crops(self.image, x_range=self.X_BOX_all,
                                               with_range=not training_set,
                                               y_range=self.Y_BOX_all,
                                               dataset=[],
                                               resize=resize,
                                               growth_radius=self.growth_radius,
                                               train_set=training_set)
            inf_or_train_dataset = dataset(inf_crops, do_transform=False,
                                           with_hand_seg=False, with_range=not training_set)
            val_dataset = inf_crops


        return inf_or_train_dataset, val_dataset

    def train(self, test=False):

        self.update_run_info(batch_multi_run=str(self.training_size))
        out_folder = os.path.join(self.pred_session_run_path)  # ,
        #                          str(self.training_size))

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        self.write_selection_bounds(dir=out_folder,
                                    x_box=self.X_BOX, y_box=self.Y_BOX,
                                    mode='w')



        self.train_dataset, self.val_dataset = self.collect_boxes(#region_list=self.region_list,
                                                                  # number_samples=self.training_size,
                                                                  training_set=True)

        self.val_dataset = self.train_dataset

        self.model()

        BATCH_SIZE = max(len(self.X_BOX)//4, 1)#self.params['batch_size']




        # Creating the training Image and Mask generator
        image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2,
                                                 width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
        mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2,
                                                width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

        # Keep the same seed for image and mask generators so they fit together
        #print("    * train dataset shape", self.train_dataset.shape)
        #print("      * t set ", self.train_dataset.shape)

        self.X_train = self.train_dataset.get_images()#[:,0]

        self.Y_train = self.train_dataset.get_segmentations()#[:,1]
        self.X_train = self.X_train.reshape((*self.X_train.shape, 1))
        self.Y_train = self.Y_train.reshape((*self.Y_train.shape, 1))

        # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))
        # ax[0].imshow(np.squeeze(self.X_train[0], axis=2))
        # #ax[0].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        # ax[0].set_title('Image')
        # ax[1].imshow(np.squeeze(self.Y_train[0], axis=2))
        # #ax[1].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        # ax[1].set_title('ground truth')
        # plt.show()





        seed = 101
        np.random.seed(97)

        train_partition = 0.9 #if self.training_size >= 10 else 0.9
        val_partition = 1.0 - train_partition
        if self.training_size == 1:
            train_partition = 1.0
            val_partition = 0.

        image_datagen.fit(self.X_train[:int(self.X_train.shape[0] * train_partition)], augment=True, seed=seed)
        mask_datagen.fit(self.Y_train[:int(self.Y_train.shape[0] * train_partition)], augment=True, seed=seed)

        x = image_datagen.flow(self.X_train[:int(self.X_train.shape[0] * train_partition)], batch_size=BATCH_SIZE, shuffle=True, seed=seed)
        y = mask_datagen.flow(self.Y_train[:int(self.Y_train.shape[0] * train_partition)], batch_size=BATCH_SIZE, shuffle=True, seed=seed)

        # Creating the validation Image and Mask generator
        image_datagen_val = image.ImageDataGenerator()
        mask_datagen_val = image.ImageDataGenerator()

        image_datagen_val.fit(self.X_train[int(self.X_train.shape[0] * val_partition):], augment=True, seed=seed)
        mask_datagen_val.fit(self.Y_train[int(self.Y_train.shape[0] * val_partition):], augment=True, seed=seed)

        x_val = image_datagen_val.flow(self.X_train[int(self.X_train.shape[0] * val_partition):],
                                       batch_size=BATCH_SIZE, shuffle=True,
                                       seed=seed)
        y_val = mask_datagen_val.flow(self.Y_train[int(self.Y_train.shape[0] * val_partition):],
                                      batch_size=BATCH_SIZE, shuffle=True,
                                      seed=seed)

        train_generator = zip(x, y)
        val_generator = zip(x_val, y_val)

        #self.run_num = self.training_size
        #self.update_run_info(batch_multi_run=str(self.training_size))
        out_folder = os.path.join(self.pred_session_run_path)  # ,
        #                          str(self.training_size))
        if not os.path.exists(out_folder):
             os.makedirs(out_folder)


        self.model_file = os.path.join(out_folder, self.label_file + ".h5")


        DID_TRAINING = False
        #if self.UNET_FORCE_RECOMPUTE or not os.path.isfile(self.model_file):
        DID_TRAINING = True
        model = Model(inputs=[self.inputs], outputs=[self.outputs])
        optmzr = 'adam' #Adam(learning_rate=0.001)
        model.compile(optimizer=optmzr, loss='binary_crossentropy', metrics=[self.mean_iou])#MeanIoU(num_classes=2)])
        model.summary()
        print("about to fit generator")
        start_train = timeit.default_timer()
        earlystopper = EarlyStopping(patience=15, verbose=1)

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        #                               min_delta=0.0001, cooldown=1,
        #                               patience=6, min_lr=0, verbose=1)


        save_best_only = True#self.training_size > 1

        steps_per_epoch = max(int(self.X_train.shape[0])*5,250)

        checkpointer = ModelCheckpoint(self.model_file, verbose=1, save_best_only=save_best_only)
        self.results = model.fit_generator(train_generator,
                                           validation_data=val_generator,
                                           validation_steps=5,
                                           validation_freq=1,
                                           steps_per_epoch= 10,#steps_per_epoch,        #   !!!!!
                                           epochs=self.params['epochs'],
                                           callbacks=[earlystopper,
                                                      checkpointer])
                                                      # reduce_lr])
        end_train = timeit.default_timer()

        self.record_time(round(end_train -start_train , 4), dir=out_folder, type='train')

        self.trained_model = model

        print("    * Finished training")

    def infer(self, model=None, dataset=None, training_window_file=None, INTERACTIVE=False,
              load_pretrained=False, view_results=False, test=True, pred_thresh=0.5):

        out_folder = os.path.join(self.pred_session_run_path)  # ,
        #                          str(self.training_size))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # print( "    * :" "loading inference windows from: ")
        # print( "    ", training_window_file)
        # training_window_file = training_window_file.split('.')[0] + '_infer.txt'
        # print("    * ", training_window_file)
        # f = open(training_window_file, 'r')
        # box_dict = {}
        # param_lines = f.readlines()
        # f.close()
        #
        # if test and len(param_lines) > 16:
        #     print("    * Truncating number of training regions for testing...........")
        #     param_lines = param_lines[0:8]

        test_dataset, val_dataset = self.collect_boxes(# region_list = param_lines ,
                                                        training_set=False)

        self.X_test = test_dataset.get_images()

        self.X_test = self.X_test.reshape((*self.X_test.shape, 1))
        self.Y_test = test_dataset.get_segmentations()
        self.Y_test = self.Y_test.reshape((*self.Y_test.shape, 1))


        #self.training_labels = self.training_labels[0][1]


        self.X_train = self.train_dataset.get_images()

        self.Y_train = self.train_dataset.get_segmentations()
        self.X_train = self.X_train.reshape((*self.X_train.shape, 1))
        self.Y_train = self.Y_train.reshape((*self.Y_train.shape, 1))


        if load_pretrained:
            print("about to load and predict")
            model = load_model(self.model_file, custom_objects={'mean_iou':self.mean_iou})# MeanIoU(num_classes=2)})
        else:
            print("using pre-trained model to predict")
            model = self.trained_model

        partition = 1.0
        preds_train = model.predict(self.X_train[:int(self.X_train.shape[0] * partition)],
                                    verbose=1 if not test else 0)
        # preds_val = model.predict(self.X_train[int(self.X_train.shape[0] * 1.0-partition):], verbose=0)




        start_test = timeit.default_timer()

        preds_test = model.predict(self.X_test, verbose=1 if not test else 0)

        end_test = timeit.default_timer()
        self.record_time(round( end_test - start_test, 4), dir=out_folder, type='pred')


        self.training_reg_bg = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for idx, box_set in enumerate(self.training_boxes):

            x_box = box_set[0]
            y_box = box_set[1]
            self.training_reg_bg[x_box[0]:x_box[1], y_box[0]:y_box[1]] = 1





        return preds_test


    def compute_metrics(self, pred_images,# scores, pred_labels, pred_thresh,
                        INTERACTIVE=False):#predictions_topo, labels_topo,

        use_average = True

        self.pred_val_im = np.zeros(self.image.shape[:2], dtype=np.float32)
        tile_samples = []
        sample_box = []
        for idx, box_set in enumerate(self.all_region_boxes):

            x_box = box_set[0]
            y_box = box_set[1]

            pad = 8
            pred_tile = np.squeeze(pred_images[idx], axis=2)
            self.pred_val_im[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad] = \
                pred_tile[pad:pred_tile.shape[0] - pad, pad:pred_tile.shape[1] - pad]

            if idx ==len(self.all_region_boxes)//4:
                sample_box = box_set
                im_tile = self.image[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
                seg_tile = self.training_labels[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
                p_tile = pred_tile[pad:pred_tile.shape[0] - pad, pad:pred_tile.shape[1] - pad]
                tile_samples = [im_tile,seg_tile,p_tile]


        self.cutoffs = np.arange(0.01, 0.98, 0.01)
        scores = np.zeros((len(self.cutoffs), 4), dtype="int32")
        self.pred_prob_im = np.ones(self.image.shape[:2], dtype=np.float32)*0.25
        predictions = []
        true_labels = []
        for gid in self.node_gid_to_label.keys():
            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            label = label if type(label) != list else label[1]
            line = get_points_from_vertices([gnode])
            # else is fg
            vals = []

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                vals.append(self.pred_val_im[lx, ly])

            inferred = np.array(vals, dtype="float32")
            infval = np.average(inferred)
            pred_mode = infval
            if not use_average:
                vals, counts = np.unique(inferred, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))
                pred_mode = inferred[mode_value].flatten().tolist()[0]

            infval = infval if use_average else pred_mode



            self.node_gid_to_prediction[gid] = [1. - infval, infval]

            for idx, cutoff in enumerate(self.cutoffs):
                if infval >= cutoff:
                    if label == 1:
                        scores[idx, 0] += len(line)  # true positive
                    else:
                        scores[idx, 2] += len(line)  # false positive
                else:
                    if label == 1:
                        scores[idx, 1] += len(line)  # false negative
                    else:
                        scores[idx, 3] += len(line)  # true negative

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                self.pred_prob_im[lx, ly] = infval
                if self.training_reg_bg[lx, ly] != 1:
                    predictions.append(infval )
                    true_labels.append(label )


        # print("       labelslslslsls", labels_topo)
        # print("After sampling back to lines, using 0.5 cutoff:")
        restab = pd.DataFrame(scores.T)
        restab.columns = [str(x) for x in self.cutoffs]

        self.F1_log = {}
        self.max_f1 = 0
        self.opt_thresh = 0
        for thresh in self.cutoffs:

            threshed_arc_segmentation_logits = [logit > thresh for logit in true_labels]
            threshed_arc_predictions_proba = [logit > thresh for logit in predictions]

            F1_score_topo = f1_score(y_true=threshed_arc_segmentation_logits,
                                     y_pred=threshed_arc_predictions_proba,average=None)[-1]

            #self.F1_log[F1_score_topo] = thresh
            if F1_score_topo >= self.max_f1:
                self.max_f1 = F1_score_topo

                self.F1_log[self.max_f1] = thresh
                self.opt_thresh = thresh


        gt_polyline_labels = []
        pred_labels_conf_matrix = np.zeros(self.image.shape[:2], dtype=np.float32)#dtype=np.uint8)
        pred_labels_msc = np.ones(self.image.shape[:2], dtype=np.float32) * 0.25
        predictions_topo_bool = []
        labels_topo_bool = []
        check=30
        for gid in self.node_gid_to_label.keys():  # zip(mygraph.labels, mygraph.polylines):

            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            label = label if type(label) != list else label[1]
            line = get_points_from_vertices([gnode])
            # else is fg
            cutoff = self.F1_log[self.max_f1]

            vals = []

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                pred = self.pred_val_im[lx, ly]
                vals.append(pred)

            if not use_average:
                vals = list(map(lambda x : round(x,2),vals))
            inferred = np.array(vals, dtype="float32")
            infval = np.average(inferred)
            pred_mode = infval
            if not use_average:
                vals, counts = np.unique(inferred, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))
                pred_mode = inferred[mode_value].flatten().tolist()[0]

            infval = infval if use_average else pred_mode

            self.node_gid_to_prediction[gid] = [1.-infval, infval]
            if check >= 0:

                check -= 1

            t = 0
            if infval >= self.opt_thresh:
                if label == 1:
                    t = 0.25 # 1
                else:
                    t = .75 #3
            else:
                if label == 1:
                    t = .5 # 2
                else:
                    t = 1 # 4

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                pred_labels_conf_matrix[lx, ly] = t
                pred_labels_msc[lx,ly] = 1 if infval >= self.F1_log[self.max_f1] else 0
                if self.training_reg_bg[lx, ly] != 1:
                    self.node_gid_to_partition[gid] = 'test'
                    predictions_topo_bool.append(infval >= cutoff)
                    gt_label = self.training_labels[lx, ly]
                    labels_topo_bool.append(label >= cutoff)
                else:
                    self.node_gid_to_partition[gid] = 'train'


        out_folder = os.path.join(self.pred_session_run_path)#,
        #                          str(self.training_size))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)



        exp_folder = os.path.join(self.params['experiment_folder'])#, 'runs')

        batch_metric_folder = os.path.join(exp_folder, 'batch_metrics')
        if not os.path.exists(batch_metric_folder):
            os.makedirs(batch_metric_folder)

        self.draw_segmentation(dirpath=out_folder)
        compute_prediction_metrics('unet', predictions_topo_bool,
                                   labels_topo_bool,
                                   out_folder)
        self.write_arc_predictions(dir=out_folder)
        self.write_training_percentages(dir=out_folder,msc_segmentation=self.training_labels)
        self.write_training_percentages(dir=out_folder,train_regions=self.training_reg_bg)
        self.draw_segmentation(dirpath=out_folder)
        self.write_gnode_partitions(dir=out_folder)

        multi_run_metrics(model='unet', exp_folder=exp_folder,
                          batch_multi_run=True,
                          bins=7, runs='runs',#str(self.training_size),
                          plt_title=exp_folder.split('/')[-1])

        print("UNET_MAX_F1:", self.max_f1)
        print("pthresh:", self.F1_log[self.max_f1], 'opt', self.opt_thresh)#cutoffs[F1_MAX_ID])
        print("Num Pixels:", self.image.shape[0] * self.image.shape[1], "Num Pixels training:", np.sum(self.training_labels), "Percent:",
              100.0 * np.sum(self.training_labels) / (self.image.shape[0] * self.image.shape[1]))




        num_fig = 6
        fig, ax = plt.subplots(1, num_fig, sharex=True, sharey=True, figsize=(num_fig*4, num_fig-1))
        ax[0].imshow(self.image)
        ax[0].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[0].set_title('Image')
        ax[1].imshow(self.training_labels)
        ax[1].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[1].set_title('Ground Truth Segmentation')
        ax[2].imshow(self.pred_val_im)
        ax[2].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[2].set_title('Predicted Segmentation')
        ax[3].imshow(pred_labels_conf_matrix)
        ax[3].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[3].set_title('MSC TF TP TN FN')
        ax[4].imshow(self.pred_prob_im)
        ax[4].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[4].set_title('MSC Pixel Inference Confidence')
        ax[5].imshow(pred_labels_msc)
        ax[5].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[5].set_title('MSC Binary Prediction')
        #fig.tight_layout()
        if INTERACTIVE:
            plt.show()

        #batch_folder = os.path.join(self.params['experiment_folder'],'batch_metrics', 'prediction')
        #if not os.path.exists(batch_folder):
        #    os.makedirs(batch_folder)
        plt.savefig(os.path.join(out_folder,"unet_imgs.pdf"))



        # tile samples
        x_box = sample_box[0]
        y_box = sample_box[1]
        pad=128

        im_tile = self.image[300:464, 300:464]#[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
        seg_tile = self.training_labels[300:464, 300:464]#[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
        p_tile = self.pred_val_im[300:464, 300:464]#[pad:pred_tile.shape[0] - pad, pad:pred_tile.shape[1] - pad]

        sample_conf_mat = pred_labels_conf_matrix[300:464, 300:464]#[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
        sample_proba = self.pred_prob_im[300:464, 300:464]#[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
        sample_pred_label = pred_labels_msc[300:464, 300:464]#[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
        num_fig = 6
        fig, ax = plt.subplots(1, num_fig, sharex=True, sharey=True, figsize=(num_fig * 4, num_fig - 1))
        ax[0].imshow(im_tile)
        # ax[0].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[0].set_title('Image')

        ax[2].imshow(seg_tile)
        # ax[2].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[2].set_title('Ground Truth Segmentation')
        ax[1].imshow(p_tile)
        # ax[1].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[1].set_title('Predicted Segmentation')
        ax[3].imshow(sample_conf_mat)
        # ax[0].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[3].set_title('MSC TF TP TN FN')
        ax[4].imshow(sample_proba)
        # ax[1].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[4].set_title('Polyline Inference Confidence')
        ax[5].imshow(sample_pred_label)
        # ax[2].contour(self.training_reg_bg, [0.0, 0.15], linewidths=0.5)
        ax[5].set_title('MSC Binary Prediction')
        if INTERACTIVE:
            plt.show()
        plt.savefig(os.path.join(out_folder, "unet_imgs_sample.pdf"))



        np.savez_compressed(os.path.join(out_folder,'pred_matrix.npz'),self.pred_val_im)
        np.savez_compressed(os.path.join(out_folder,'training_matrix.npz'),self.training_reg_bg)
        # dict_data = np.load('data.npz')
        # # extract the first array
        # data = dict_data['arr_0']
        # # print the array
        # print(data)




