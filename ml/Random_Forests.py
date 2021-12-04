from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skimage import  segmentation, feature, future
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from copy import deepcopy
import time

from mlgraph import MLGraph
from getofeaturegraph import GeToFeatureGraph
from getograph import  Attributes
from proc_manager import experiment_manager
from data_ops import set_parameters
from data_ops.utils import dbgprint
from ml.features import get_points_from_vertices
from metrics.prediction_score import get_topology_prediction_score
from sklearn.metrics import f1_score
from ml.features import multiscale_basic_features
from scipy import ndimage
from data_ops import dataflow
from data_ops.utils import plot

from data_ops.collect_data import collect_training_data, compute_geomsc, collect_datasets



class RandomForest(MLGraph):

    def __init__(self, training_selection_type='box',run_num=1, classifier_type='msc',parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None, X_BOX=None,Y_BOX=None,boxes=None,
                 model_name=None, load_feature_graph_name=False,image=None,  **kwargs):

        self.model_name = model_name

        self.details = "Random forest classifier with feature importance"

        self.type = "random_forest"+'_'+classifier_type



        #if self.params is None:
        self.parameter_file_number = parameter_file_number

        '''self.params = {}
        if parameter_file_number is None:
            self.params = kwargs
        else:
            for param in kwargs:
                self.params[param] = kwargs[param]'''

        #self.write_folder = self.params['write_folder']

        self.G = None
        self.G_dict = {}

        super(RandomForest, self).__init__(parameter_file_number=parameter_file_number, run_num=run_num,
                                      name=kwargs['name'], geomsc_fname_base=geomsc_fname_base,
                                      label_file=label_file, image=image, write_folder=kwargs['write_folder'],
                                      model_name=model_name, load_feature_graph_name=load_feature_graph_name)


        self.attributes = self.get_attributes()
        '''for param in kwargs:
            self.params[param] = kwargs[param]
        # for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]
        if 'params' in kwargs.keys():
            param_add_ons = kwargs['params']
            for l, v in param_add_ons.items():
                self.params[l] = v'''



        self.run_num = run_num
        self.logger = experiment_manager.experiment_logger(experiment_folder=self.experiment_folder,
                                                           input_folder=self.input_folder)
        self.param_file = os.path.join(self.LocalSetup.project_base_path,
                                       'parameter_list_' + str(parameter_file_number) + '.txt')
        self.topo_image_name = label_file.split('.labels')[0]
        self.logger.record_filename(label_file=label_file,
                                    parameter_list_file=self.param_file,
                                    image_name=image,
                                    topo_image_name=self.topo_image_name)

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

        # self.X_BOX = X_BOX
        # self.Y_BOX = Y_BOX
        # self.boxes = boxes
        self.name = kwargs['name']
        self.image_path = image

        self.X_BOX = X_BOX
        self.Y_BOX = Y_BOX



    def build_random_forest(self,
                            BEGIN_LOADING_FEATURES=False,
                 ground_truth_label_file=None, write_path=None, type = 'pixel',
                            feature_file=None,X_BOX=None, Y_BOX=None, boxes=None,
                 window_file=None, model_name="GeToGNN"):

        self.attributes = self.get_attributes()


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
        else:
            self.params['load_features'] = False
            self.params['write_features'] = True
            self.params['load_features'] = False
            self.params['write_feature_names'] = True
            self.params['save_filtered_images'] = True
            self.params['collect_features'] = True
            self.params['load_preprocessed'] = False
            self.params['load_geto_attr'] = False
            self.params['load_feature_names'] = False

        if self.params['load_geto_attr']:
            if self.params['geto_as_feat']:
                self.load_geto_features()
                self.load_geto_feature_names()

        # features
        if not self.params['load_features']:
            self.compile_features(include_geto=self.params['geto_as_feat'])
            self.write_gnode_features(self.session_name)
            self.write_feature_names()
        else:
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
        self.read_labels_from_file(file=ground_truth_label_file)


        # if self.type == 'pixel':
        #     _ , _ , box_set = self.box_select_geomsc_training(x_range=X_BOX,
        #                                                                                  y_range=Y_BOX,
        #                                                                                  boxes=boxes)#
        #     self.X_BOX, self.Y_BOX = box_set
        # else:
        X_BOX = []
        Y_BOX = []
        box_sets = []
        for box in boxes:
            X_BOX.append((box[0],box[1]))
            Y_BOX.append((box[2], box[3]))

        for box_pair in box_sets:
            for box in box_pair:
                X_BOX.append(box[0])
                Y_BOX.append(box[1])
        _, _, box_set = self.box_select_geomsc_training(x_range=X_BOX,
                                                        y_range=Y_BOX,
                                                        boxes=None)  #
        self.X_BOX, self.Y_BOX = box_set


        self.get_train_test_val_sugraph_split(collect_validation=False, validation_hops = 1,
                                                 validation_samples = 1)

        self.box_regions = boxes
        # X_BOX = []
        # Y_BOX = []
        # box_sets = []
        # for box in boxes:
        #     box_set = tile_region(step_X=64, step_Y=64, step=0.5,
        #                           Y_START=box[0], Y_END=box[1],
        #                           X_START=box[2], X_END=box[3])
        #     box_sets.append(box_set)
        #
        # for box_pair in box_sets:
        #     for box in box_pair:
        #         X_BOX.append(box[0])
        #         Y_BOX.append(box[1])
        #self.X_BOX = X_BOX
        #self.Y_BOX = Y_BOX

        # X_BOX_all = []
        # Y_BOX_all = []
        # box_sets_test = []
        # for box in [[0, self.image.shape[0], 0, self.image.shape[1]]]:
        #     box_set_test = tile_region(step_X=64, step_Y=64, step=0.5,
        #                           Y_START=0, Y_END=self.image.shape[1],
        #                           X_START=0, X_END=self.image.shape[0])
        #     box_sets_test.append(box_set_test)
        #
        #
        # for box_pair in box_sets_test:
        #     for box in box_pair:
        #         X_BOX_all.append(box[0])
        #         Y_BOX_all.append(box[1])
        # self.X_BOX_all = X_BOX_all
        # self.Y_BOX_all = Y_BOX_all

        num_percent = 0
        for xbox,ybox in zip(self.X_BOX, self.Y_BOX):
            num_percent += float((xbox[1] - xbox[0]) * (ybox[1] - ybox[0]))
        percent = num_percent / float(self.image.shape[0] * self.image.shape[1])
        percent_f = percent * 100
        print("    * ", percent_f)
        percent = int(round(percent_f))
        self.training_size = percent
        self.run_num = percent

        self.update_run_info(batch_multi_run=str(self.training_size))
        out_folder = os.path.join(self.pred_session_run_path)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        self.training_reg_bg = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for region in self.box_regions:
            self.training_reg_bg[region[0]:region[1], region[2]:region[3]] = 1

        self.data_array, self.data_set = collect_datasets(name=self.name, image=self.image_path,
                                                          dim_invert=self.params['dim_invert'],
                                                          format=self.params['format'])


        self.train_dataloader = collect_training_data(
            dataset=self.data_set,
            data_array=self.data_array,
            params=self.params,
            name=self.name,
            format=format,
            msc_file=None,
            dim_invert=self.params['dim_invert'])

        self.image, self.msc_collection, self.mask, self.segmentation = self.train_dataloader[
            int(self.params['train_data_idx'])]
        self.image = self.image.astype(np.float32)
        max_val = np.max(self.image)
        min_val = np.min(self.image)
        self.image = (self.image - min_val) / (max_val - min_val)
        # self.image = self.image if len(self.image.shape) == 2 else np.transpose(np.mean(self.image, axis=1), (1, 0))

        self.X = self.image.shape[0]
        self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]





    def classifier(self, node_gid_to_prediction, train_features=None, train_labels=None,
                   test_features=None, test_labels=None, feature_map=False,class_1_weight=1.0,
                   class_2_weight=1.0,threshold=0.5,
                   n_trees=10, depth=4, weighted_distribution=False):
        print("_____________________________________________________")
        print("                Random_Forest MSC   ")
        print("number trees: ", n_trees)
        print("depth: ", depth)




        self.update_run_info(batch_multi_run=self.run_num)

        self.write_gnode_partitions(self.pred_session_run_path)
        self.write_selection_bounds(self.pred_session_run_path)

        # Import the model we are using
        # Instantiate model with 1000 decision trees
        train_gid_feat_dict = train_features
        train_gid_label_dict = train_labels
        test_gid_feat_dict = test_features
        test_gid_label_dict = test_labels

        train_features = np.array(list(train_features.values()))
        train_labels = list(train_labels.values())
        train_labels_binary = [l[1] for l in train_labels]
        train_labels = np.array(train_labels_binary)

        if test_features is not None:
            test_features = np.array(list(test_features.values()))
            test_labels = list(test_labels.values())
            test_labels_binary = [l[1] for l in test_labels]
            test_labels = np.array(test_labels_binary)

        wn = 1
        wp = 1
        if weighted_distribution:
            wn = float(len(train_labels)) / (2.*(len(train_labels) - np.sum(train_labels)))
            wp = float(len(train_labels)) / (2.* np.sum(train_labels))

        print("    * RF train shape", train_features.shape)

        print("    * RF test shape", test_features.shape)

        print("    * RF msc feat shape", test_features.shape)
        #####################################
        s = time.time()
        rf = RandomForestClassifier(max_depth=10,
                                    n_estimators=50, class_weight={0:wn,1:wp}, random_state=666)

        rf.fit(train_features, train_labels)
        f = time.time()
        self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='train')

        # Use the forest's predict method on the test data
        if test_features is not None:
            s = time.time()
            pred_proba_test = rf.predict_proba(test_features)
            f = time.time()
            self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='pred')

            # pred_proba_train = rf.predict_proba(train_features)
            # train_arcs = [arc for arc in self.msc.arcs if arc.partition == 'train']
            # test_arcs = [arc for arc in self.msc.arcs if arc.partition == 'test']
            # for arc, pred in zip(train_arcs, list(pred_proba_train)):
            #    arc.prediction = pred[1]#[1-pred[0], pred[0]]

            preds = []
            true_labels = []
            for gid, pred in zip(test_gid_feat_dict.keys(),pred_proba_test):

                self.node_gid_to_prediction[gid] = pred[1]
                if self.node_gid_to_partition[gid] != 'train':
                    preds.append(pred[1])
                    label = self.node_gid_to_label[gid]
                    true_labels.append(label[1])

            self.F1_log = {}
            self.max_f1 = 0
            self.opt_thresh = 0
            self.cutoffs = np.arange(0.01, 0.98, 0.01)
            for thresh in self.cutoffs:

                threshed_arc_segmentation_logits = [logit > thresh for logit in true_labels]
                threshed_arc_predictions_proba = [logit > thresh for logit in preds]

                F1_score_topo = f1_score(y_true=threshed_arc_segmentation_logits,
                                         y_pred=threshed_arc_predictions_proba, average=None)[-1]

                # self.F1_log[F1_score_topo] = thresh
                if F1_score_topo >= self.max_f1:
                    self.max_f1 = F1_score_topo

                    self.F1_log[self.max_f1] = thresh
                    self.opt_thresh = thresh

            labels = [logit > self.opt_thresh for logit in true_labels]
            predictions = [logit > self.opt_thresh for logit in preds]
            #preds = [l[len(l) - 1] for l in preds]

            # errors = abs(self.preds - self.labels)  # Print out the mean absolute error (mae)
            # round(np.mean(errors), 2), 'degrees.')

            mse = rf.score(test_features, test_labels_binary)  # np.array(list(test_features) + list(train_features)),
            #                                       np.array(list(test_labels) + list(train_labels)))

            #p, r, fs = compute_quality_metrics(preds, test_labels_binary)

            return predictions, labels, node_gid_to_prediction #,p, r, fs, mse,

    def get_train_labels(self,image=None, x_range=None, y_range=None,
                         growth_radius=2,dataset=[], resize=False):



        def get_data_crops( image=None, x_range=None, y_range=None,
                            dataset=[],growth_radius=2, resize=False):

            segmentations = []




            #print('x range', x_range, 'y', y_range)
            #if len(x_range) == 1:
            #    x_range  = x_range[0]
            #    y_range= y_range[0]
            range_group = zip(self.X_BOX,self.Y_BOX)#zip(x_range, y_range)
            #print(range_group, 'range group')

            seg_whole = np.zeros(self.image.shape).astype(np.uint8)
            seg_whole = generate_pixel_labeling(([0, self.image.shape[0]],
                                               [0, self.image.shape[1]]), growth_radius=growth_radius,
                                    seg_image=seg_whole)

            seg = np.zeros(self.image.shape).astype(np.uint8)
            region_contour = np.zeros(self.image.shape).astype(np.uint8)
            for x_rng, y_rng in range_group:

                #print('y_range', y_rng)
                #print('x_range', x_rng)

                #train_im_crop = deepcopy(image[y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]])
                #seg = np.zeros(image.shape).astype(np.uint8)

                seg[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1]] = int(1)

                #region_contour = np.zeros(image.shape).astype(np.uint8)
                region_contour[x_rng[0]:x_rng[1], y_rng[0]:y_rng[1]] = int(1)

                #print(' seg shape', seg.shape)
                segmentation = generate_pixel_labeling((x_rng, y_rng), growth_radius=growth_radius,
                                                            seg_image=seg)
                # segmentation = segmentation[y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]]

                #print("segmentation shape", segmentation.shape)
                dataset.append((image, segmentation, region_contour, seg_whole))
            # segmentation = generate_pixel_labeling((self.X_BOX, self.Y_BOX), growth_radius=growth_radius,
            #                                        seg_image=seg)
            dataset = (self.image,segmentation,self.training_reg_bg,seg_whole )
            return dataset#[len(dataset)-1]

        def generate_pixel_labeling( train_region, growth_radius=2, seg_image=None):
            x_box = train_region[0]
            y_box = train_region[1]
            #dim_train_region = (int(y_box[1] - y_box[0]), int(x_box[1] - x_box[0]))
            train_im = seg_image  # np.zeros(dim_train_region)
            X = train_im.shape[0]
            Y = train_im.shape[1]

            def __box_grow_label(train_im, center_point, label):
                x = center_point[0]
                y = center_point[1]
                x[x + 1 >= X] = X - 2
                y[y + 1 >= Y] = Y - 2
                train_im[x, y] = label
                train_im[x - 1, y] = label
                train_im[x, y - 1] = label
                train_im[x - 1, y - 1] = label
                train_im[x + 1, y] = label
                train_im[x, y + 1] = label
                train_im[x + 1, y + 1] = label
                train_im[x - 1, y + 1] = label
                train_im[x + 1, y - 1] = label

            for gid in self.gid_gnode_dict.keys():
                gnode = self.gid_gnode_dict[gid]
                label = self.node_gid_to_label[gid]
                points = gnode.points
                p1 = points[0]
                p2 = points[-1]
                in_box = False
                not_all = False
                end_points = (p1, p2)
                points = get_points_from_vertices([gnode])
                points =  np.reshape(points, (-1, 2))
                interior_points = []
                for p in points:
                    if x_box[0] <= p[1] <= x_box[1] and y_box[0] <= p[0] <= y_box[1]:
                        in_box = True
                        interior_points.append(p)
                        mapped_y = points[:, 0]  # - x_box[0]
                        mapped_x = points[:, 1]  # - y_box[0]
                        if int(label[1]) == 1:
                            train_im[mapped_x, mapped_y] = 2

                    else:
                        not_all = True
                if in_box:
                    points = np.array(interior_points)

                #mapped_y = points[:, 0] #- x_box[0]
                #mapped_x = points[:, 1] #- y_box[0]
                #    # __box_grow_label(train_im, (mapped_x, mapped_y), 2)

            train_im = ndimage.maximum_filter(train_im, size=growth_radius)
            # n_samples = dim_train_region[0] * dim_train_region[1]
            # n_classes = 2
            # class_bins = np.bincount(train_im.astype(np.int64).flatten())
            self.class_weights = 1.#n_samples / (n_classes * class_bins)


            return train_im

        dataset = get_data_crops(self.image, x_range=x_range,growth_radius=growth_radius,
                                           y_range=y_range, dataset=dataset, resize=resize)

        return dataset

    def label_msc(self):
        dummy_validation_pos = list(self.gid_gnode_dict.values())[0]
        dummy_validation_neg = list(self.gid_gnode_dict.values())[1]
        dummy_validation_pos.partition = 'val'
        dummy_validation_neg.partition = 'val'
        self.node_gid_to_partition[dummy_validation_pos.gid] = 'val'
        self.node_gid_to_partition[dummy_validation_neg.gid] = 'val'
        dp = set()
        dn = set()
        dp_g = set()
        dn_g = set()
        dp_g.add(dummy_validation_pos)
        dn_g.add(dummy_validation_neg)
        dp.add(dummy_validation_pos.gid)
        dn.add(dummy_validation_neg.gid)
        self.validation_set_ids = {"positive": dp, "negative": dn}
        self.validation_set = dp_g.union(dn_g)
        for gid in self.validation_set_ids["positive"].union(self.validation_set_ids["negative"]):
            self.node_gid_to_partition[gid] = 'val'
        all_validation = dp.union(dn)

        all_selected = self.selected_positive_arc_ids.union(self.selected_negative_arc_ids)



        for gnode in self.gid_gnode_dict.values():

            partition = self.node_gid_to_partition[gnode.gid]
            features = self.node_gid_to_feature[gnode.gid].tolist()

            if gnode.label is not None and gnode.gid not in all_selected:
                label = self.node_gid_to_label[gnode.gid]  # [0 , 1] if gnode.label > 0 else [1, 0]#[
            else:
                label = [
                    int(gnode.gid in self.negative_arc_ids),
                    int(gnode.gid in self.positive_arc_ids)
                ]
                self.node_gid_to_label[gnode.gid] = label

            nx_gid = self.node_gid_to_graph_idx[gnode.gid]
            node = self.G.node[nx_gid]
            node["features"] = features  # gnode.features.tolist()
            node["gid"] = gnode.gid
            # getoelm = self.gid_geto_elm_dict[gnode.gid]
            # polyline = getoelm.points
            # node["geto_elm"] = polyline
            node["key"] = gnode.key
            node["box"] = gnode.box
            node["partition"] = partition
            # assign partition to node
            node["train"] = partition == 'train'
            node["test"] = partition == 'test'
            node["val"] = partition == 'val'
            node["label"] = label
            if self.selection_type == 'map':
                node["label_accuracy"] = gnode.label_accuracy
            node["prediction"] = []
            self.node_gid_to_prediction[gnode.gid] = []

    def pixel_classifier(self, node_gid_to_prediction, train_features=None, train_labels=None,
                   test_features=None, test_labels=None, feature_map=False,class_1_weight=1.0,
                   class_2_weight=1.0,threshold=0.5, INTERACTIVE=False,growth_radius=2,
                   n_trees=10, depth=4, weighted_distribution=False):
        print("_____________________________________________________")
        print("                Random_Forest Pixel    ")
        print("number trees: ", n_trees)
        print("depth: ", depth)

        self.update_run_info(batch_multi_run=self.run_num)

        self.write_gnode_partitions(self.pred_session_run_path)
        self.write_selection_bounds(self.pred_session_run_path)

        use_average = True

        X = self.image.shape[0]
        Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]

        max_val = np.max(self.image)
        min_val = np.min(self.image)
        self.image = (self.image - min_val) / (max_val - min_val)



        train_im_crop, segmentation, region_contour,\
        seg_whole =self.get_train_labels(y_range=self.Y_BOX,#[self.params['x_box']],
                                                    growth_radius=growth_radius,
                                         x_range=self.X_BOX)


        # (x_range=[(0, X)],
        # y_range = [(0, Y)])
        training_labels = segmentation

        ################## BUILD FEATURES  ################

        sigma_min = 1
        sigma_max = 64
        features_func = partial(multiscale_basic_features,
                                intensity=True, edges=False, texture=True,
                                sigma_min=sigma_min, sigma_max=sigma_max,
                                multichannel=False)



        # features_func.
        features = features_func(train_im_crop)

        aug_ims = []
        filtered_im_folder = os.path.join(self.experiment_folder,'filtered_images')
        df = dataflow()
        filtered_imgs = df.read_images(filetype='.png',screen='feat-func_', dest_folder=filtered_im_folder)
        for im in filtered_imgs:
            np.max(im)
            min_val = np.min(im)
            im = (im- min_val) / (max_val - min_val)
            print("loaded im shape:",im.shape)
            im = np.mean(im,axis=2)
            im = np.expand_dims(im, axis=-1)
            print(im.shape)
            features=np.concatenate((im, features), axis=2)

        print("Features shape:", features.shape)
        print("i,j, shape:", features[0, 0].shape)

        s = time.time()
        clf = RandomForestRegressor(n_estimators=50, bootstrap=True, n_jobs=-1,
                                    max_depth=10, max_samples=0.05)
        clf = fit_segmenter(training_labels, features, clf)
        f = time.time()

        self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='train')

        s = time.time()
        result = predict_segmenter(features, clf) -1

        f = time.time()

        self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='pred')




        result =np.array(result).astype(np.float32)
        # opt_f1 = 0
        # opt_thresh = 0
        # for thresh in [0.2,0.3,0.4,0.5,0.6,0.7,0.9]:
        #     F1_score_topo, gt_pixel_segmentation, pixel_predictions, \
        #     arc_segmentation_logits, arc_predictions, node_gid_to_prediction, node_gid_to_label = get_topology_prediction_score(
        #         predicted=result,
        #         segmentation=segmentation,
        #         gid_gnode_dict=self.gid_gnode_dict,
        #         node_gid_to_prediction=self.node_gid_to_prediction,
        #         node_gid_to_label=self.node_gid_to_label,
        #         X=X, Y=Y,
        #         pred_thresh=thresh)
        #     print("    F1 ", F1_score_topo)
        #     if F1_score_topo > opt_f1:
        #         opt_f1 = F1_score_topo
        #         opt_thresh = thresh
        #
        # F1_score_topo, gt_pixel_segmentation, pixel_predictions, \
        # arc_segmentation_logits, arc_predictions,\
        # self.node_gid_to_prediction, node_gid_to_label = get_topology_prediction_score(
        #                                                     predicted=result,
        #                                                     segmentation=segmentation,
        #                                                     gid_gnode_dict=self.gid_gnode_dict,
        #                                                     node_gid_to_prediction=self.node_gid_to_prediction,
        #                                                     node_gid_to_label=self.node_gid_to_label,
        #                                                     X=X, Y=Y,
        #                                                     pred_thresh=opt_thresh)
        self.cutoffs = np.arange(0.01, 0.98, 0.01)
        scores = np.zeros((len(self.cutoffs), 4), dtype="int32")
        self.pred_prob_im = np.zeros(self.image.shape[:2], dtype=np.float32)
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
                vals.append(result[lx, ly])

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
                gt_val = seg_whole[lx,ly]
                if region_contour[lx,ly] != 1:
                    predictions.append(infval)
                    true_labels.append(gt_val)

        self.F1_log = {}
        self.max_f1 = 0
        self.opt_thresh = 0
        for thresh in self.cutoffs:

            threshed_arc_segmentation_logits = [logit > thresh for logit in true_labels]
            threshed_arc_predictions_proba = [logit > thresh for logit in predictions]

            F1_score_topo = f1_score(y_true=threshed_arc_segmentation_logits,
                                     y_pred=threshed_arc_predictions_proba, average=None)[-1]

            # self.F1_log[F1_score_topo] = thresh
            if F1_score_topo >= self.max_f1:
                self.max_f1 = F1_score_topo

                self.F1_log[self.max_f1] = thresh
                self.opt_thresh = thresh

        gt_polyline_labels = []
        pred_labels_conf_matrix = np.zeros(self.image.shape[:2], dtype=np.float32) #* min(0.25self.opt_thresh/2.) # dtype=np.uint8)
        overlay_image = np.zeros(self.image.shape[:2], dtype=np.float32)
        pred_labels_msc = np.zeros(self.image.shape[:2], dtype=np.float32) #* min(0.25, self.opt_thresh/2.)
        predictions_topo_bool = []
        labels_topo_bool = []
        check = 30
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
                pred = result[lx, ly]
                vals.append(pred)

            if not use_average:
                vals = list(map(lambda x: round(x, 2), vals))
            inferred = np.array(vals, dtype="float32")
            infval = np.average(inferred)
            pred_mode = infval
            if not use_average:
                vals, counts = np.unique(inferred, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))
                pred_mode = inferred[mode_value].flatten().tolist()[0]

            infval = infval if use_average else pred_mode

            self.node_gid_to_prediction[gid] = [1. - infval, infval]
            if check >= 0:

                check -= 1

            t = 0
            if infval >= self.opt_thresh:
                if label == 1:
                    t = 4#0.25  # 1
                else:
                    t = 2#.75  # 3  yellow
            else:
                if label == 1:
                    t = 3#.5  # 2 dark blue
                else:
                    t = 1  # 4 light blue

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                pred_labels_conf_matrix[lx, ly] = t
                pred_labels_msc[lx, ly] = 1 if infval >= self.F1_log[self.max_f1] else 0
                if region_contour[lx,ly] != 1:
                    self.node_gid_to_partition[gid] = 'test'
                    predictions_topo_bool.append(infval >= cutoff)
                    gt_label = seg_whole[lx, ly]
                    labels_topo_bool.append(gt_label >= cutoff)
                else:
                    self.node_gid_to_partition[gid] = 'train'

        out_folder = os.path.join(self.pred_session_run_path)  # ,
        #                          str(self.training_size))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # msc_pred_segmentation = self.predicted_msc(arc_predictions,
        #                                            arc_segmentation_logits, result, self.opt_thresh)



        # self.plot(self.image, region_contour.astype(np.uint8),
        #           result,
        #           msc_pred_segmentation,
        #           opt_thresh)
        # num_fig = 6
        # fig, ax = plt.subplots(1, num_fig, sharex=True, sharey=True, figsize=(num_fig * 4, num_fig - 1))
        # ax[0].imshow(self.image)
        # ax[0].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[0].set_title('Image')
        # ax[1].imshow(seg_whole)
        # ax[1].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[1].set_title('Ground Truth Segmentation')
        # ax[2].imshow(result)
        # ax[2].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[2].set_title('Prediction')
        # #ax[3].imshow(pred_labels_conf_matrix)
        # #ax[3].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # #ax[3].set_title('MSC TF TP TN FN')
        # ax[4].imshow(self.pred_prob_im)
        # ax[4].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[4].set_title('Pixel to lines Foreground Probability')
        # ax[5].imshow(pred_labels_msc)
        # ax[5].contour(region_contour, [0.0, 0.15], linewidths=0.5) #grey
        # ax[5].set_title('TP FP FN TN')
        # # fig.tight_layout()
        # if INTERACTIVE:
        #     plt.show()
        #
        # # batch_folder = os.path.join(self.params['experiment_folder'],'batch_metrics', 'prediction')
        # # if not os.path.exists(batch_folder):
        # #    os.makedirs(batch_folder)
        # plt.savefig(os.path.join(out_folder, "rf_pix_imgs.pdf"))
        #
        # num_fig = 6
        # fig, ax = plt.subplots(1, num_fig, sharex=True, sharey=True, figsize=(num_fig * 4, num_fig - 1))
        # ax[0].imshow(self.image)
        # #ax[0].contour(region_contour[400:464,400:464], [0.0, 0.15], linewidths=0.5)
        # ax[0].set_title('Image')
        # ax[1].imshow(seg_whole[300:464,300:464])
        # #ax[1].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[1].set_title('Ground Truth Segmentation')
        # ax[2].imshow(result[300:464,300:464])
        # #ax[2].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[2].set_title('Predicted Segmentation')
        # ax[3].imshow(pred_labels_conf_matrix[300:464,300:464])
        # #ax[3].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[3].set_title('MSC TF TP TN FN')
        # ax[4].imshow(self.pred_prob_im[300:464,300:464])
        # #ax[4].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[4].set_title('MSC Pixel Inference Confidence')
        # ax[5].imshow(pred_labels_msc[300:464,300:464])
        # #ax[5].contour(region_contour, [0.0, 0.15], linewidths=0.5)
        # ax[5].set_title('MSC Binary Prediction')
        # # fig.tight_layout()
        # if INTERACTIVE:
        #     plt.show()
        #
        # # batch_folder = os.path.join(self.params['experiment_folder'],'batch_metrics', 'prediction')
        # # if not os.path.exists(batch_folder):
        # #    os.makedirs(batch_folder)
        # plt.savefig(os.path.join(out_folder, "rf_pix_zoom_imgs.pdf"))


        images =[self.image, seg_whole, result,
                 self.pred_prob_im]
        names = ["Image", "Ground Truth Segmentation", "Predicted Foreground Segmentation",
                 "Pixel to Lines Foreground Probability"]
        for image, name in zip(images, names):
            plot(image_set=[image,region_contour], name=name, type='contour', write_path=out_folder)


        image_set  = [pred_labels_msc, region_contour, pred_labels_conf_matrix]
        plot(image_set, name="TP FP TF TN Line Prediction",
             type='confidence',write_path=out_folder)

        plot(image_set, name="TP FP TF TN Line Prediction",
             type='zoom', write_path=out_folder)


        for image, name in zip(images, names):
            plot(image_set=[image,region_contour], name=name, type='zoom', write_path=out_folder)

        np.savez_compressed(os.path.join(out_folder, 'pred_matrix.npz'), result)
        np.savez_compressed(os.path.join(out_folder, 'training_matrix.npz'), region_contour)

        return labels_topo_bool, predictions_topo_bool, self.node_gid_to_prediction #,p, r, fs, mse,






try:
    from sklearn.exceptions import NotFittedError
    from sklearn.ensemble import RandomForestClassifier
    has_sklearn = True
except ImportError:
    has_sklearn = False

    class NotFittedError(Exception):
        pass


class TrainableSegmenter(object):
    """Estimator for classifying pixels.
    Parameters
    ----------
    clf : classifier object, optional
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    features_func : function, optional
        function computing features on all pixels of the image, to be passed
        to the classifier. The output should be of shape
        ``(m_features, *labels.shape)``. If None,
        :func:`skimage.segmentation.multiscale_basic_features` is used.
    Methods
    -------
    compute_features
    fit
    predict
    """

    def __init__(self, clf=None, features_func=None):
        if clf is None:
            if has_sklearn:
                self.clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            else:
                raise ImportError(
                    "Please install scikit-learn or pass a classifier instance"
                    "to TrainableSegmenter."
                )
        else:
            self.clf = clf
        self.features_func = features_func

    def compute_features(self, image):
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        self.features = self.features_func(image)

    def fit(self, image, labels):
        """Train classifier using partially labeled (annotated) image.
        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.
        labels : ndarray of ints
            Labeled array of shape compatible with ``image`` (same shape for a
            single-channel image). Labels >= 1 correspond to the training set and
            label 0 to unlabeled pixels to be segmented.
        """
        self.compute_features(image)
        clf = fit_segmenter(labels, self.features, self.clf)

    def predict(self, image):
        """Segment new image using trained internal classifier.
        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.
        Raises
        ------
        NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
        """
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        features = self.features_func(image)
        return predict_segmenter(features, self.clf)


def fit_segmenter(labels, features, clf):
    """Segmentation using labeled parts of the image and a classifier.
    Parameters
    ----------
    labels : ndarray of ints
        Image of labels. Labels >= 1 correspond to the training set and
        label 0 to unlabeled pixels to be segmented.
    features : ndarray
        Array of features, with the first dimension corresponding to the number
        of features, and the other dimensions correspond to ``labels.shape``.
    clf : classifier object
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    Returns
    -------
    clf : classifier object
        classifier trained on ``labels``
    Raises
    ------
    NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
    """
    mask = labels > 0
    training_data = features[mask]
    training_labels = labels[mask].ravel()
    clf.fit(training_data, training_labels)
    return clf


def predict_segmenter(features, clf):
    """Segmentation of images using a pretrained classifier.
    Parameters
    ----------
    features : ndarray
        Array of features, with the last dimension corresponding to the number
        of features, and the other dimensions are compatible with the shape of
        the image to segment, or a flattened image.
    clf : classifier object
        trained classifier object, exposing a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
        classifier must be already trained, for example with
        :func:`skimage.segmentation.fit_segmenter`.
    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier.
    """
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))

    try:
        predicted_labels = clf.predict(features)
    except NotFittedError:
        raise NotFittedError(
            "You must train the classifier `clf` first"
            "for example with the `fit_segmenter` function."
        )
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(
                err.args[0] + '\n' +
                "Maybe you did not use the same type of features for training the classifier."
                )
    output = predicted_labels.reshape(sh[:-1])
    return output