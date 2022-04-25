import os
import numpy as np
from skimage.measure import block_reduce
from skimage import morphology
from PIL import Image
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ml.utils import pout

#resize the images of the dataset to be half the height and half the width of the original images, so
# that models states can fit on the GPU memory
def resize_img(img):
    if len(img.shape)==3:
        img = np.array(Image.fromarray(img).resize(((img.shape[1]+1)//2,(img.shape[0]+1)//2), Image.BILINEAR))
    else:
        img = block_reduce(img, block_size=(2, 2), func=np.max)
    return img



#delete small regions (<size) of binary images
def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def linear_idx_from_coord(x, y, X, Y):
    return y*X + x

def coord_from_linear_idx(idx, dim_x, dim_y, x, y, z):
    x = idx % (dim_x)
    idx /= (dim_x)
    y = idx % (dim_y)
    idx /= (dim_y)
    z = idx

def grow_box(img, boxes, training_labels=None):
    IMG_WIDTH = img.shape[0]
    IMG_HEIGHT = img.shape[1]
    ###### make the test data
    all_test_regions = []
    all_label_regions = []
    all_region_boxes = []
    for xstart in range(0, img.shape[0] - (IMG_WIDTH - int(IMG_WIDTH / 4)), int(IMG_WIDTH / 4)):
        if xstart + IMG_WIDTH >= img.shape[0]:
            xstart = img.shape[0] - IMG_WIDTH
        for ystart in range(0, img.shape[1] - (IMG_HEIGHT - int(IMG_HEIGHT / 4)), int(IMG_HEIGHT / 4)):
            if ystart + IMG_HEIGHT >= img.shape[1]:
                ystart = img.shape[1] - IMG_HEIGHT
            # get subimage
            all_region_boxes.append([xstart, xstart + IMG_WIDTH, ystart, ystart + IMG_HEIGHT])
            test_im = img[xstart:xstart + IMG_WIDTH, ystart:ystart + IMG_HEIGHT]
            all_test_regions.append(test_im)
            if training_labels is not None:
                lab_im = training_labels[xstart:xstart + IMG_WIDTH, ystart:ystart + IMG_HEIGHT]

                all_label_regions.append(lab_im)

    all_training_images = []
    all_training_labels = []
    train_img_boxes = []
    # all_training_dist_images = []
    for box in boxes:
        # quad of xmin xmax ymin ymax
        for xstart in range(box[0], box[1] - IMG_WIDTH, int(IMG_WIDTH / 2)):
            for ystart in range(box[2], box[3] - IMG_HEIGHT, int(IMG_HEIGHT / 2)):
                # get subimage

                # dist_subimage = core_lines_dist[xstart:xstart+IMG_WIDTH, ystart:ystart+IMG_HEIGHT]
                train_im = img[xstart:xstart + IMG_WIDTH, ystart:ystart + IMG_HEIGHT]
                train_img_boxes.append([xstart, xstart + IMG_WIDTH, ystart, ystart + IMG_HEIGHT])
                if training_labels is not None:
                    train_lab = training_labels[xstart:xstart + IMG_WIDTH, ystart:ystart + IMG_HEIGHT]

                    all_training_labels.append(train_lab)
                all_training_images.append(train_im)
    return all_training_images, all_training_labels, train_img_boxes

def tile_region(step_X, step_Y, step, X_START, X_END, Y_START, Y_END,
                X_MIN=None, Y_MIN=None, X_MAX=None, Y_MAX=None , INVERT=False):

    if X_MIN  == None:
        X_MIN = Y_START
        Y_MIN = X_START
        Y_MAX = X_END
        X_MAX = Y_END
    if INVERT:
        def swap(a , b):
            a_temp = a
            a = b
            b = a_temp
            return a,b
        step_X, step_Y = swap(step_X, step_Y)
        X_START, Y_START = swap(X_START, Y_START)
        X_END , Y_END = swap(X_END, Y_END)

    increment_x = (1./step)*((X_END - X_START) / float(step_X ))+1
    increment_y = (1./step)*((Y_END - Y_START) / float(step_Y ))+1

    entire_boxes_x = (X_MAX - X_MIN) / float(step_X )
    entire_boxes_y = (Y_MAX - Y_MIN) / float(step_Y )

    x_boxes = []
    y_boxes = []

    box_set = []

    box_set_y = []
    box_set_x = []

    #print("step", step)
    #print("x range", increment_x, 'y range', increment_y)
    ends = 0
    begun_y = True
    begun_x = True

    Y_STOP = Y_END
    X_STOP = X_END

    Y_INIT = Y_START
    X_INIT = X_START
    Y_START = Y_INIT
    X_START = X_INIT
    zero = True
    # for j in range(int(increment_y * increment_x )):# removed * incj
    #
    #     Y_START =Y_START + step_Y*step
    #     Y_STOP = Y_START + step_Y
    #
    #
    #     if  Y_STOP > Y_END :
    #         y_box = (int(Y_END - step_Y), int(Y_END))
    #         Y_START = Y_INIT
    #     else:
    #         y_box = (int(Y_START), int(Y_STOP))
    #
    #     for i in range( int(increment_x ) ):  # * (1. / step))):
    #
    #         X_START = X_START + step_X * step
    #         X_STOP = X_START + step_X
    #
    #
    #         if X_STOP > X_END :
    #             x_box = ( int(X_END - step_X), int(X_END))
    #             X_START = X_INIT
    #         else:
    #             x_box = (int(X_START), int(X_STOP))
    #         if (x_box, y_box) not in box_set:
    #             box_set.append((x_box, y_box))
    #         begun_y = False
    for j in range(int(increment_y * increment_x )):# removed * incj

        Y_START = j*step_Y*step +Y_INIT
        Y_STOP = Y_START + step_Y


        if  Y_STOP > Y_END :
            y_box = (int(Y_END - step_Y), int(Y_END))
        else:
            y_box = (int(Y_START), int(Y_STOP))

        for i in range( int(increment_x ) ):  # * (1. / step))):

            X_START = i * step_X * step + X_INIT
            X_STOP = X_START + step_X


            if X_STOP > X_END :
                x_box = ( int(X_END - step_X), int(X_END))
            else:
                x_box = (int(X_START), int(X_STOP))
            if (x_box, y_box) not in box_set:
                box_set.append((x_box, y_box))
            begun_y = False




    outer = x_boxes if len(x_boxes) < len(y_boxes) else y_boxes
    inner = x_boxes if len(x_boxes) > len(y_boxes) else y_boxes

    if INVERT:
        box_set_inverted = []
        for x_box,y_box in box_set:
            box_set_inverted.append((y_box,x_box))
        return box_set_inverted
    return box_set

def plot( image_set, name='RF-Lines Prediction', type='contour',write_path=None, INTERACTIVE=False):
    image = image_set[0]

    if len(image_set) > 2:
        contour = image_set[1]
        image2 = image_set[2]
        num_fig = 1
        mycmap = colors.ListedColormap(["lightgray", "blue", "yellow", "cyan", "red",'mediumspringgreen'])
    else:
        contour = image_set[1]
        num_fig = 1

    fig, ax = plt.subplots(1, num_fig, sharex=True, sharey=True,figsize=(9,4))
    if 'foam' in str(write_path):
        start_x, start_y = 550, 20
        zoom_x = 250
    if 'diadem' in str(write_path):
        start_x, start_y = 100, 150
        zoom_x = 250
    if 'berghia' in str(write_path):
        start_x, start_y = 200, 200
        zoom_x = 250
    if 'neuron2' in str(write_path):
        start_x, start_y = 890, 895
        zoom_x = 250
    if 'retinal' in str(write_path):
        start_x, start_y = 250,50
        zoom_x = 250
    zoom_y = round(image.shape[1]/image.shape[0] * zoom_x)
    if type =='zoom':
        ax.imshow(image[start_x:start_x+zoom_x,start_y:start_y+zoom_y], cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)

    if len(image_set) > 2: # show the image to make the plot have the right shape
        if type == "zoom":
            ax.pcolormesh(image2[start_x:start_x+zoom_x,start_y:start_y+zoom_y], cmap=mycmap,
                          rasterized=True)  # covers the image, good looking plot
        else:
            ax.pcolormesh(image2, cmap=mycmap, rasterized=True)
            if name == "Ground Truth MSC":
                ax.contour(image_set[1],[0.0, 0.15], linewidths=0.5)

    if type == 'contour':
        ax.contour(contour, [0.0, 0.15], linewidths=0.5)  # add the training region

    ax.set_title(name)
    # ax[3].imshow(msc_pred_probs)
    # ax[3].contour(training_labels, [0.0, 0.15], linewidths=0.5)
    # ax[3].set_title('MSC Probability Result')
    fig.tight_layout()
    if INTERACTIVE:
        plt.show()
    else:
        if type != 'zoom':
            plt.savefig(os.path.join(write_path ,
                                     name+".imgs.png"),
                        dpi=300)
        else:
            plt.savefig(os.path.join(write_path,
                                     name + ".zoom-imgs.png"),
                        dpi=300)
    plt.close(fig)
    if name == 'Image':
        zoom_region = np.zeros(image.shape[:2], dtype=np.uint8)
        zoom_region[start_x:start_x + zoom_x, start_y:start_y + zoom_y] = 1
        fig, ax = plt.subplots(1, num_fig, sharex=True, sharey=True, figsize=(9, 4))
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.contour(zoom_region, [0.0, 0.15], linewidths=0.5, cmap='Greens')
        plt.savefig(os.path.join(write_path,
                                     "original_img_zoom_box.png"),
                        dpi=300)

def compute_features(model=None):
    #features must go before geto, see load and write func

    # features
    if model.params['collect_features'] and not model.params['load_features']:
        model.compile_features(include_geto=model.params['geto_as_feat'])
        model.write_gnode_features(model.session_name)
        model.write_feature_names()
    elif model.params['load_features']:
        model.load_gnode_features()
        model.load_feature_names()

    # geto feat
    if model.params['load_geto_attr']:
        #if 'geto' in self.getognn.params['aggregator']:
        model.load_geto_features()
        model.load_geto_feature_names()
    elif model.params['geto_as_feat'] and not model.params['load_geto_attr']:
        #if 'geto' in self.getognn.params['aggregator']:
        model.build_geto_adj_list(influence_type=model.params['geto_influence_type'])
        model.write_geto_features(model.session_name)
        model.write_geto_feature_names()
        include_generic_feat = model.params['collect_features'] or model.params['load_features']
        model.compile_features(include_geto=model.params['geto_as_feat'],
                               include_generic_feat=include_generic_feat)

    #
    pout(["Feature collection done, now setting up feature handling"])
    #
    if (model.params['collect_features'] or model.params['load_geto_attr']) \
            and (model.params['geto_as_feat'] or model.params['load_features']) and \
            not model.params['feats_independent']:
        pout(["Concated STD and GEOM features"])
        features = []
        feat_idx = 0
        for gid, gnode in model.gid_gnode_dict.items():
            feats = model.node_gid_to_standard_feature[gid]
            geomfeats = model.node_gid_to_geom_feature[gid]
            combined_feats = feats + geomfeats
            model.node_gid_to_feature[gid] = np.array(combined_feats)
            gnode.features = np.array(combined_feats)
            features.append(combined_feats)
            model.node_gid_to_feat_idx[gid] = feat_idx
            feat_idx += 1
        model.features = np.array(features)
    elif (model.params['collect_features'] or model.params['load_geto_attr']) \
            and (model.params['geto_as_feat'] or model.params['load_features']) and \
            model.params['feats_independent']:
        pout(["Independent GEOM and STD features"])
        features = []
        getoelms = []
        feat_idx = 0
        for gid, gnode in model.gid_gnode_dict.items():
            feats = model.node_gid_to_standard_feature[gid]
            geomfeats = model.node_gid_to_geom_feature[gid]
            #                                                 # MLP random forest use combined!
            combined_feats = feats + geomfeats              # check where used!
            model.node_gid_to_feature[gid] = None#np.array(combined_feats)#
            gnode.features = None#np.array(combined_feats)
            features.append(feats)#combined_feats)
            getoelms.append(geomfeats)
            model.node_gid_to_feat_idx[gid] = feat_idx
            model.gid_to_getoelm_idx[gid]   = feat_idx
            feat_idx += 1
        model.features = np.array(features)
        model.getoelms = np.array(getoelms)
    elif (model.params['collect_features'] or model.params['load_features']) and not (model.params['geto_as_feat'] or model.params['load_geto_attr']):
        pout(["STD features only"])
        features = []
        feat_idx = 0
        for gid, gnode in model.gid_gnode_dict.items():
            feats = model.node_gid_to_standard_feature[gid]
            #geomfeats = model.node_gid_to_geom_feature[gid]
            #combined_feats = feats + geomfeats
            model.node_gid_to_feature[gid] = None# np.array(feats)
            gnode.features = None#np.array(feats)
            features.append(feats)
            model.node_gid_to_feat_idx[gid] = feat_idx
            feat_idx += 1
        model.features = np.array(features)
        model.getoelms = None
    else:
        pout(["GEOM features only"])
        features = []
        feat_idx = 0
        for gid, gnode in model.gid_gnode_dict.items():
            #feats = model.node_gid_to_standard_feature[gid]
            geomfeats = model.node_gid_to_geom_feature[gid]
            #combined_feats = feats + geomfeats
            model.node_gid_to_feature[gid] = None#np.array(geomfeats)
            gnode.features = None#np.array(geomfeats)
            features.append(geomfeats)
            model.node_gid_to_feat_idx[gid] = feat_idx
            feat_idx += 1
        model.features = np.array(features)

def pout(show=None):
    if isinstance(show, list):
        print("    *")
        for elm in show:
            if isinstance(elm, str):
                print("    * ",elm)
            else:
                print("    * ", str(elm))
        print("    *")
    else:
        print("    *")
        if isinstance(show, str):
            print("    * ", show)
        else:
            print("    * ", str(show))
        print("    *")

def dbgprint(x,name=""):
    print("    *")
    print("    *:", name, x)
    print("    *")