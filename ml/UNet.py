#importing the libraries used in the rest of hte code
import os
import sys
import gzip
import shutil
import tarfile
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from copy import deepcopy
#%matplotlib inline
import zipfile
import collections
from skimage import morphology
from skimage.measure import block_reduce
import scipy
from torch.utils.data import Dataset
import PIL
from PIL import Image
from sklearn.metrics import f1_score
import copy
import random
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from attributes import Attributes
from mlgraph import MLGraph
from ml.features import get_points_from_vertices
from metrics.model_metrics import compute_prediction_metrics
from data_ops.collect_data import collect_training_data, compute_geomsc, collect_datasets
from ml.utils import get_partition_feature_label_pairs
from data_ops.utils import dbgprint as dprint
from compute_multirun_metrics import multi_run_metrics

from localsetup import LocalSetup
LocalSetup = LocalSetup()

# with this function you set the value of the environment variable CUDA_VISIBLE_DEVICES
# to set which GPU to use
# it also reserves this amount of memory for your exclusive use. This might be important for
# not having other people using the resources you need in shared systems
# the homework was tested in a GPU with 4GB of memory, and running this function will require at least
# as much
# if you want to test in a GPU with less memory, you can call this function
# with the argument minimum_memory_mb specifying how much memory from the GPU you want to reserve
def define_gpu_to_use(minimum_memory_mb = 3800, gpu_to_use=None):
    try:
        os.environ['CUDA_VISIBLE_DEVICES']
        print('GPU already assigned before: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
        return
    except:
        pass
    torch.cuda.empty_cache()
    for i in range(16):
        '''free_memory = !nvidia-smi --query-gpu=memory.free -i $i --format=csv,nounits,noheader
        if free_memory[0] == 'No devices were found':
            break
        free_memory = int(free_memory[0])
        if free_memory>minimum_memory_mb-500:
            gpu_to_use = i
            break'''
    if False:#gpu_to_use is None:
        print('Could not find any GPU available with the required free memory of ' +str(minimum_memory_mb) + 'MB. Please use a different system for this assignment.')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)
        print('Chosen GPU: ' + str(gpu_to_use))
        #x = torch.rand((256,1024,minimum_memory_mb-500)).cuda()
        #x = torch.rand((1,1)).cuda()
        #del x

#delete small regions (<size) of binary images
def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

#resize the images of the dataset to match that used in training
# also, resize to scale image back to normal when infering on larger image
def resize_img(img, X=None, Y=None, sampling="lanczos", downsample=False):
    if not downsample:
        if sampling == "lanczos":
            print("    Image shape: ", img.shape)
            img = Image.fromarray(img) # .convert('RGB')
            img = img.resize((X,Y), PIL.Image.LANCZOS) # PIL.Image.BOX)
            #  PIL.Image.LANCZOS, PIL.Image.BICUBIC, PIL.Image.HAMMING
            img = np.array(img).astype('float32')  # .convert('L'))
        elif sampling == "hamming":
            img = Image.fromarray(img) # bilinear #hamming
            img = img.resize((X,Y), PIL.Image.HAMMING)#PIL.Image.BOX)
            #  PIL.Image.LANCZOS, PIL.Image.BICUBIC, PIL.Image.HAMMING
            img = np.array(img).astype('float32')#.astype('float32')#.convert('L'))
        elif sampling == "bicubic":
            img = Image.fromarray(img) # bilinear #hamming
            img = img.resize((X,Y), PIL.Image.BICUBIC)#PIL.Image.BOX)
            #  PIL.Image.LANCZOS, PIL.Image.BICUBIC, PIL.Image.HAMMING
            img = np.array(img).astype('float32')#.astype('float32')#.convert('L'))
    #if downsample and len(img.shape)!=3:
    #    img = block_reduce(img, block_size=(2, 2), func=np.max)
    return img


#for segmentations tasks, the exact transformations that are applied to
# the input image should be applied, down to the random number used, should
# also be applied to the ground truth and to the masks. We redefine a few of
# PyTorch classes

#apply transoforms to all tensors in list x
def _iterate_transforms(transform, x):
    for i, xi in enumerate(x):
        x[i] = transform(x[i])
    return x

#redefining composed transform so that it uses the _iterate_transforms function
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = _iterate_transforms(transform, x)
        return x

class Normalize(object):
   def __call__(self, img):
      return transforms.Normalize((0.5,), (0.5,))
#class to rerandomize the vertical flip transformation
class RandomVerticalFlipGenerator(object):
    def __call__(self, img):
        self.random_n = random.uniform(0, 1)
        return img

#class to perform vertical flip using randomization provided by gen
class RandomVerticalFlip(object):
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, img):
        if self.gen.random_n < 0.5:
            return torch.flip(img, [1])
        return img

#class to rerandomize the horizontal flip transformation
class RandomHorizontalFlipGenerator(object):
    def __call__(self, img):
        self.random_n = random.uniform(0, 1)
        return img

#class to perform horizontal flip using randomization provided by gen
class RandomHorizontalFlip(object):
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, img):
        if self.gen.random_n < 0.5:
            return torch.flip(img, [1])
        return img

    # use this function to score your models


def get_score_model(model, data_loader, X = None, Y =None):
    # toggle model to eval mode
    model.eval()

    # turn off gradients since they will not be used here
    # this is to make the inference faster
    with torch.no_grad():
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
    with torch.no_grad():
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
        print("    * :", "pred", predicted.shape)
        print("    * :", "seg", segmentation.shape)

        # print(segmentation)
        segmentations = np.concatenate((segmentations, segmentation ),
                                       axis=0)
        # returns a list of scores, one for each of the labels
        segmentations = segmentations.reshape([-1]) > 0
        logits_predicted = logits_predicted.reshape([-1]) > 0
        binary_logits_predicted = logits_predicted#.astype(int)
    return f1_score(segmentations, logits_predicted) , segmentations, binary_logits_predicted

def get_topology_prediction_score(predicted, segmentation,
                                  gid_gnode_dict, node_gid_to_prediction, node_gid_to_label, pred_thresh=0.4,
                                  X = None, Y =None, ranges=None):
    # toggle model to eval mode
    #model.eval()

    # turn off gradients since they will not be used here
    # this is to make the inference faster
    arc_pixel_predictions = []
    segmentation_logits = []
    with torch.no_grad():
        logits_predicted = np.zeros([0, 1, X, Y])
        segmentations = np.zeros([0, 1, X, Y])
        # run through several batches, does inference for each and store inference results
        # and store both target labels and inferenced scores
        #for image, segmentation in data_loader:
        #image = image.cuda()
        #image = image.unsqueeze(1)
        #logit_predicted = model(image)
        logits_predicted = np.concatenate((logits_predicted, predicted),
                                          axis=0)
        segmentation = segmentation
        # print('shape ', segmentation.shape)
        # print('shape mask ', mask.shape)

        segmentation[segmentation > 0] = 1
        if len(segmentation.shape) != 4:
            segmentation = segmentation[None]#.unsqueeze(1)#[:, :, :, :]
        print("    * :", "pred_topo", predicted.shape)
        print("    * :", "seg_topo", segmentation.shape)



        def __get_prediction_correctness(segmentation, prediction, center_point):
            x = center_point[0]
            y = center_point[1]
            x = X-2 if x + 1 >= X else x
            y = Y-2 if y + 1 >= Y else y
            yield (prediction[0,0,x,y] , segmentation[0,0,x,y])
            yield (prediction[0,0,x-1,y] , segmentation[0,0,x-1,y])
            yield (prediction[0,0,x,y-1] , segmentation[0,0,x, y-1])
            yield (prediction[0,0,x-1,y-1] , segmentation[0,0,x - 1, y-1])
            yield (prediction[0,0,x+1,y] , segmentation[0,0,x+1,y])
            yield (prediction[0,0,x,y+1] , segmentation[0,0,x, y+1])
            yield (prediction[0,0,x+1,y+1] , segmentation[0,0,x + 1, y+1])
            yield (prediction[0, 0, x + 1, y - 1], segmentation[0, 0, x + 1, y -1])
            yield (prediction[0, 0, x - 1, y + 1], segmentation[0, 0, x - 1, y + 1])

        if ranges is not None:
            ranges = ranges.cpu().detach().numpy()
            dprint(ranges)
            x_range = ranges[0][0]# list(map(int, ranges[0][1][0]))
            y_range = ranges[0][1]#list(map(int, ranges[0][0][0]))

            dprint(y_range, "y_range")
            dprint(x_range,"x_range")
        else:
            x_range = [0,0]
            y_range=[0,0]


        for gid in gid_gnode_dict.keys():
            gnode = gid_gnode_dict[gid]
            points = gnode.points
            points = get_points_from_vertices([gnode])
            if ranges is not None:
                points[:,0] = points[:,0] + x_range[0]
                points[:,1] = points[:,1] + y_range[0]

            arc_predictions = []
            arc_segmentation_logits = []
            for p in points:
                x = p[0] #+ x_range[0]
                y = p[1] #+ y_range[0]
                arc_predictions += [pred[0] > pred_thresh for pred in __get_prediction_correctness(segmentation,
                                                                               logits_predicted,(x,y))]
                arc_segmentation_logits += [pred[1] > pred_thresh for pred in __get_prediction_correctness(segmentation,
                                                                               logits_predicted,(x,y))]
            spread_pred = np.bincount(arc_predictions,minlength=2)

            pred_unet_val = spread_pred[1]/np.sum(spread_pred)
            pred_unet = spread_pred[1]/np.sum(spread_pred) > pred_thresh
            arc_pixel_predictions.append(pred_unet)



            spread_seg = np.bincount(arc_segmentation_logits,minlength=2)
            gt_val = spread_seg[1] / np.sum(spread_seg)
            gt = spread_seg[1] / np.sum(spread_seg) > pred_thresh
            segmentation_logits.append(gt)

            node_gid_to_prediction[gid] = [1.0-pred_unet_val , pred_unet_val]
            node_gid_to_label[gid] = [1.0-gt_val, gt_val]

            # returns a list of scores, one for each of the labels
        segmentations = np.array(segmentation_logits)
        logits_predicted = np.array(arc_pixel_predictions)
        binary_logits_predicted = logits_predicted#.astype(int)

        print(segmentations[0:10])
        print(" logits,", logits_predicted[0:10])

    return f1_score(segmentations, logits_predicted) , segmentations, binary_logits_predicted, node_gid_to_prediction, node_gid_to_label


#
#
#
#
###
# im_positive = []
# im_negative = []
# for batch_idx, (img, segmentation, mask) in enumerate(RetinaDataset(retina_array)):
#     segmentation[segmentation > 0] = 1
#     num_pos = np.sum(segmentation.cpu().numpy() == 1)
#     num_neg = np.sum(segmentation.cpu().numpy() == 0)
#     bg_space = np.sum(mask.cpu().numpy() == 0)
#     num_neg = num_neg - bg_space
#     im_positive.append(num_pos)
#     im_negative.append(num_neg)

# total_pos = np.sum(np.array(im_positive))
# total_neg = np.sum(np.array(im_negative))
# print("total positive samples: ", total_pos)
# print("total negative samples: ", total_neg)
# print("average positive samples: ", np.average(np.array(im_positive)))
# print("average negative samples: ", np.average(np.array(im_negative)))
# print("...")
# pos_weights = total_neg / total_pos
# print("PROPORTION NEGATIVE / POSITIVE: ", pos_weights)
# print("...")
# print("Positive weight Factor divided by number persistence MSC ground Truth")
# print("used for training: ", pos_weights/float(number_persistence_vals))
#pos_weights = pos_weights / 2  # float(number_persistence_vals/2)

norm_transform = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5),(0.5)) ])

# Dataset class for the retina dataset
# each item of the dataset is a tuple with three items:
# - the first element is the input image to be segmented
# - the second element is the segmentation ground truth image
# - the third element is a mask to know what parts of the input image should be used (for training and for scoring)
class dataset(Dataset):
    def transpose_first_index(self, x, with_hand_seg=False):
        if not with_hand_seg:
            x2 =(x[0], x[1], x[2])#(np.transpose(x[0], [1, 0]), x[1])
            #, np.transpose(x[2], [2, 0, 1]))
        else:
            x2 =(x[0], x[1], x[2])#(np.transpose(x[0], [1, 0]), x[1])#(np.transpose(x[0], [2, 0, 1]), np.transpose(x[1], [2, 0, 1]), np.transpose(x[2], [2, 0, 1]),
            #      np.transpose(x[3], [2, 0, 1]))
        return x2

    def __init__(self, data_array, split='train', do_transform=False, with_hand_seg=False):
        self.with_hand_seg = with_hand_seg
        indexes_this_split = np.arange(len(data_array))#get_split(np.arange(len(retina_array), dtype=np.int), split)
        self.data_array = [self.transpose_first_index(data_array[i], self.with_hand_seg) for i in
                             indexes_this_split]
        self.data_array = data_array
        self.split = split
        self.do_transform = do_transform

    def __getitem__(self, index):
        sample = [torch.as_tensor(x,dtype=torch.float) for x in self.data_array[index]]
        if self.do_transform:
            v_gen = RandomVerticalFlipGenerator()
            h_gen = RandomHorizontalFlipGenerator()

            t = Compose([
                #transforms.Normalize((0.5,), (0.5,)),
                v_gen,
                RandomVerticalFlip(gen=v_gen),
                h_gen,
                RandomHorizontalFlip(gen=h_gen),
            ])
            sample = t(sample)
        return sample

    def __len__(self):
        return len(self.data_array)



###
#
#
#


#
#   U-Net
#
#
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.uniform_(m.weight, -1, 1)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1. / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        # if m.bias is not None:
        #    m.bias.data.uniform_(-stdv, stdv)
    if isinstance(m, torch.nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.uniform_(m.weight, -1, 1)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1. / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

class nnModule(nn.Module):
    def __init__(self, **kwargs):
        super(nnModule,self).__init__()

class UNetwork( MLGraph, nnModule, object):

    '''def __getattr__(self, name):
        module = object.__getattribute__(self, "_modules")["module"]
        if name == "module":
            return module
        return getattr(module, name)'''

    def get_data_crops(self, image=None, x_range=None,y_range=None, dataset=[], resize=False):



        segmentations = []
        if False:  # np.array(x_range).shape == np.array([6,9]).shape:
            x_1 = x_range[0]
            y_1 = y_range[0]
            train_im_crop = deepcopy(image[y_1[0]:y_1[1],x_1[0]:x_1[1]])
            train_im_crop = train_im_crop
            # if resize:
            #     train_im_crop = resize_img(train_im_crop, X=resize[0], Y=resize[1])
            #     #segmentation = resize_img(segmentation, X=resize[0], Y=resize[1])
            segmentation = self.generate_pixel_labeling((x_1,y_1),
                                                        seg_image=np.zeros(train_im_crop.shape).astype(np.int8))

            og_segmentation = deepcopy(segmentation)
            if resize:
               train_im_crop = resize_img(train_im_crop, X=resize[0], Y=resize[1], sampling="lanczos")
               segmentation = resize_img(segmentation, X=resize[0], Y=resize[1], sampling="hamming")
            #segmentation = segmentation[y_range[0]:y_range[1],x_range[0]:x_range[1]]

            dataset.append((train_im_crop,segmentation, (x_range,y_range)))
        else:
            X=self.X
            Y=self.Y
            img_stack = np.zeros([0, 1, y_range[0][1], x_range[0][1]])
            segmentations = np.zeros([0, 1, y_range[0][1], x_range[0][1]])
            range_group = zip(x_range, y_range)
            for x_rng, y_rng in range_group:
                train_im_crop = deepcopy(image[y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]])
                segmentation = self.generate_pixel_labeling( (x_rng, y_rng),seg_image=np.zeros(train_im_crop.shape).astype(np.int8))
                #segmentation = segmentation[y_rng[0]:y_rng[1], x_rng[0]:x_rng[1]]

                dataset.append((train_im_crop, segmentation, (x_rng, y_rng)))

        return dataset

    def generate_pixel_labeling(self, train_region, seg_image=None ):
        x_box = train_region[0]
        y_box = train_region[1]
        dim_train_region = (   int(y_box[1] - y_box[0]),int(x_box[1]-x_box[0]) )
        train_im = seg_image#np.zeros(dim_train_region)
        X = train_im.shape[0]
        Y = train_im.shape[1]

        def __box_grow_label(train_im, center_point, label):
            x = center_point[0]
            y = center_point[1]
            x[x + 1 >= X] = Y - 2
            y[y+1>= Y] = X - 2
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
            points = gnode.points
            p1 = points[0]
            p2 = points[-1]
            in_box = False
            not_all = False
            end_points = (p1, p2)
            points = get_points_from_vertices([gnode])
            interior_points = []
            for p in points:
                if x_box[0] < p[0] < x_box[1] and y_box[0] < p[1] < y_box[1]:
                    in_box = True
                    interior_points.append(p)

                else:
                    not_all = True
            if not_all and in_box:
                points = np.array(interior_points)
            elif not_all:
                continue

            mapped_y = points[:,0] - x_box[0]
            mapped_x = points[:,1] - y_box[0]
            if int(label[1]) == 1:
                train_im[mapped_x, mapped_y] = int(1)
                __box_grow_label(train_im, (mapped_x,mapped_y), int(1))
            #else:
            #    train_im[mapped_x, mapped_y] = int(0)
            #    __box_grow_label(train_im, (mapped_x, mapped_y), int(0))
        n_samples = dim_train_region[0] * dim_train_region[1]
        n_classes = 2
        class_bins = np.bincount(train_im.astype(np.int64).flatten())
        self.class_weights = n_samples / (n_classes * class_bins)
        return train_im.astype(np.int8)

    def save_image(self, image=None, dirpath=None, gt_seg=None, pred_seg = None, image_seg_set = None, as_grey=False):
        if image_seg_set is not None:
            #shape_im = image.shape
            #shape_gt_seg = gt_seg.shape

            image = image_seg_set[0][0,0,:,:]
            gt_seg = image_seg_set[1][0,0,:,:]
            if len(image_seg_set) > 2:
                #shape_pred_seg = pred_seg.shape
                pred_seg = image_seg_set[2][0,0,:,:]

        if image is not None:
            Img = Image.fromarray(
                image.astype(np.int8))#mapped_img)
            Img.save(os.path.join(dirpath, 'image.png'))
        if gt_seg is not None:
            Img = Image.fromarray(
                gt_seg.astype(np.int8))#mapped_img)
            Img.save(os.path.join(dirpath, 'ground_truth.png'))
        if pred_seg is not None:
            Img = Image.fromarray(
                pred_seg.astype(np.int8))#mapped_img)
            Img.save(os.path.join(dirpath, 'prediction.png'))

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
                    plt.imsave(image)
            if not save:
                plt.show()
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
                    plt.imsave(gt_seg)
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
                    plt.imsave(pred_seg)
            if not save:
                plt.show()

    # Contraction Block
    # Structure Block: three tensors total w/ two convolutions followed by max-pooling
    #            in tensor -> ReLu(conv 3x3) -> Relu(3x3) -> MaxPool 2x2
    # Contraction dim: 64 -> 128 -> 256 -> 512 ->1024
    # Total 3 blocks
    def contraction_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        #if padding:
        #    out_channels = in_channels
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=out_channels, padding=int(padding)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),

            # torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels,
            #                 out_channels=out_channels, padding=int(padding)),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(out_channels),

            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels,
                            out_channels=out_channels, padding=int(padding)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    # Expansion of endoded img
    # Structure Block: three tensors w/ two convolutions followed by up-pooling
    #                 in tensor -> ReLu(conv 3x3) -> ReLu(conv 3x3) -> up-pool 2x2
    # Expansion dim: (1024 -> 512 -> 512) -> (512 -> 256 -> 256) -> (256-> 128 -> 128) -> final_block
    # Total blocks 3 plus final_block
    def expansion_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):#True):
        #if padding:
        #    mid_channel = in_channels
        #    out_channel = mid_channel
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=mid_channel, padding=int(padding)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),

            # torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
            #                 out_channels=mid_channel, padding=int(padding)),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(mid_channel),

            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
                            out_channels=mid_channel, padding=int(padding)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            #
            # Unpool with transpose convolution!
            #
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels,
                                     kernel_size=kernel_size-1, stride=2,
                                     padding=self.expansion_padding , output_padding=self.expansion_out_padding )
        )
        return block

    # Final block giving UNet output
    # Structure Block: four tensors with two 3x3 conv, one 1x1 conv
    #                 in tensor -> ReLu(conv 3x3) -> ReLu(conv 3x3) -> conv 1x1
    # Final block dims: 128 -> 64 -> 64 -> 2
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):#True):#
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=mid_channel, padding=int(padding)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
                            out_channels=mid_channel, padding=int(padding)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size-2, in_channels=mid_channel,
                            out_channels=out_channels, padding=padding-1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block



    def __init__(self,in_channel, out_channel, skip_connect=True, kernnel_size=2,
                 ground_truth_label_file=None,run_num=0, parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None,
                 model_name=None, load_feature_graph_name=False,image=None,
                 train_dataset=None, val_dataset=None, compute_features=False,
                 training_size=None, region_list=None, **kwargs):

        self.type = 'unet'

        k = [parameter_file_number, run_num, geomsc_fname_base, label_file, image,
             model_name, load_feature_graph_name]
        st_k = ['parameter_file_number', 'run_num', 'geomsc_fname_base', 'label_file', 'image',
                'model_name', 'load_feature_graph_name']
        for name, attr in zip(st_k, k):
            kwargs[name] = attr



        nnModule.__init__(self)
        MLGraph.__init__(self, **kwargs)

        self.training_size = training_size
        if region_list is not None:
            # self.pred_run_path = os.path.join(self.pred_run_path, str(training_size))
            #
            # if not os.path.exists(self.pred_run_path):
            #     os.makedirs(os.path.join(self.pred_run_path))

            self.pred_run_path = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                              self.params['write_folder'],
                                              'runs')

            if not os.path.exists(self.pred_run_path):
               os.makedirs(os.path.join(self.pred_run_path))

            self.pred_session_run_path = os.path.join(self.pred_run_path,
                                                      str(training_size))
            if not os.path.exists(self.pred_session_run_path):
                os.makedirs(os.path.join(self.pred_session_run_path))



        self.running_best_model = None

        self.kernnel_size = kernnel_size
        self.expansion_out_padding = 0
        self.expansion_padding = 0

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

        if compute_features:
            #
            # Perform remainder of runs and don't need to read feats again
            #
            # if not UNet.params['load_features']:
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
        self.read_labels_from_file(file=ground_truth_label_file)

        self.data_array = self.train_dataloader
        if train_dataset is None and training_size is None:
            training_set , test_and_val_set = self.box_select_geomsc_training(x_range=self.params['x_box'], y_range=self.params['y_box'])
            self.get_train_test_val_sugraph_split(collect_validation=False, validation_hops=1,
                                                      validation_samples=1, test_samples=None)
            self.train_dataset = self.get_data_crops(self.image,x_range=self.params['x_box'], y_range=self.params['y_box'])
            self.train_dataset = dataset(self.train_dataset, do_transform=False, with_hand_seg=False)
            self.val_dataset = self.train_dataset
        else:
            self.train_dataset , self.val_dataset = self.collect_boxes(region_list=region_list,
                                                                       number_samples=self.training_size,
                                                                       training_set=True)
            dprint(len(self.val_dataset), " val ")
            dprint(len(self.train_dataset), "train")
            self.val_dataset = self.train_dataset
        #
        # Get data computed in getofeaturegraph
        #
        self.image, self.msc_collection, self.mask, self.segmentation = self.train_dataloader[
            int(self.params['train_data_idx'])]
        self.image = self.image
        # self.image = self.image if len(self.image.shape) == 2 else np.transpose(np.mean(self.image, axis=1), (1, 0))

        self.X = self.image.shape[0]
        self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]


        #self.attributes = deepcopy(self.get_attributes())


        # contracted down pooling
        # dimensions and multiplier. Took a lot of tpying around with bc
        # my gpu does not have enough memory
        self.multiplier = 2
        self.bilinear_factor = self.multiplier
        # to turn of skip connections from contraction to expansion layers
        self.skip_connect = skip_connect
        if not self.skip_connect:
            self.skip_scale = 2
        else:
            self.skip_scale = 1

        self.init_expansion = 28

        self.contract1 = self.contraction_block(in_channels=in_channel,
                                                out_channels=self.init_expansion)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.contract2 = self.contraction_block(self.init_expansion,
                                                self.init_expansion * self.multiplier)  # 64  # 24,192
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.contract3 = self.contraction_block(self.init_expansion * self.multiplier,
                                                self.init_expansion * (self.multiplier ** 2))  # 128 # 192, 1536
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.contract4 = self.contraction_block(self.init_expansion * (self.multiplier ** 2),
                                                self.init_expansion * (self.multiplier ** 3))  # 256  # 1536, 1536
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        # 'copy and crop', base encoding of 'U'
        # switch directions

        self.base_channel_dim = self.init_expansion * (self.multiplier**3)

        #print("    * : base",self.base_channel_dim)
        self.reverse_direction = torch.nn.Sequential(
            # base encoding lowest block
            torch.nn.Conv2d(kernel_size=3, in_channels=self.init_expansion * (self.multiplier**2),
                            out_channels=self.base_channel_dim, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.base_channel_dim),

            # torch.nn.Conv2d(kernel_size=1, in_channels=self.base_channel_dim,
            #                 out_channels=self.base_channel_dim),#, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(self.base_channel_dim),
            #
            # torch.nn.Conv2d(kernel_size=1, in_channels=self.base_channel_dim,
            #                 out_channels=self.base_channel_dim),  # , padding=1),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(self.base_channel_dim),

            torch.nn.Conv2d(kernel_size=3, in_channels=self.base_channel_dim,
                            out_channels=self.base_channel_dim, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.base_channel_dim),


            # begin going up, up-pooling

            # begin unpooling, upward part of U
            # upward expansion along with skip connections from dowpooling

            torch.nn.ConvTranspose2d(in_channels=self.base_channel_dim,
                                     out_channels=self.init_expansion * (self.multiplier**2) ,#(self.base_channel_dim//(self.multiplier)) * self.skip_scale,
                                     kernel_size=2, stride=2,
                                     padding=self.expansion_padding , output_padding=self.expansion_out_padding )
        )  # 1024
        #self.multiplier *= 2
        self.expansion4 = self.expansion_block(self.base_channel_dim ,#self.init_expansion * (self.multiplier**3) *2,#(self.base_channel_dim//(self.multiplier )) * self.skip_scale,
                                               self.init_expansion * (self.multiplier**2),# * self.skip_scale,#(self.base_channel_dim//(self.multiplier )) * self.skip_scale,  #* ,
                                                self.init_expansion * (self.multiplier ** 1) )# * self.skip_scale)#* 2)

        self.expansion3 = self.expansion_block( (self.init_expansion * (self.multiplier**2) ),#(self.init_expansion * (self.multiplier ** 2)) * 2 * self.skip_scale,  #* 4,
                                               self.init_expansion * (self.multiplier ** 1),# * self.skip_scale,#(self.init_expansion * (self.multiplier ** 2)) * self.skip_scale,  #* 2,
                                               self.init_expansion * (self.multiplier ** 0) )#(self.init_expansion * (self.multiplier)) * self.skip_scale)  # 512   1536, 1536, 192

        self.expansion2 = self.expansion_block( (self.init_expansion * (self.multiplier**2) ),#(self.init_expansion * (self.multiplier ** 2)) * 2 * self.skip_scale,  #* 4,
                                               self.init_expansion * (self.multiplier ** 1),# * self.skip_scale,#(self.init_expansion * (self.multiplier ** 2)) * self.skip_scale,  #* 2,
                                               self.init_expansion * (self.multiplier ** 0) )
        self.final_layer = self.final_block(self.init_expansion * self.multiplier,#(self.init_expansion * self.multiplier) * self.skip_scale,  #* 2,
                                            self.init_expansion,
                                            out_channel) #128   192, 24, 1



    def crop_and_concat(self, upsampled, bypass, crop=True):#True):
        # print(upsampled.size())
        # print(bypass.size())
        if crop:
            diffx = -(bypass.size()[2] - upsampled.size()[2])
            diffy = -(bypass.size()[3] - upsampled.size()[3])
            #print("    * : ,",diffx, " y", diffy)
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            d = (bypass.size()[3] - upsampled.size()[3]) // 2
            #print("    * : c ", c)
            bypass = F.pad(bypass,[-c, -c, -c, -c])# [diffy//2, diffy//2, diffx, diffy//2] )#[-c, -c, -c, -c])
        # print(upsampled.shape)
        # print(bypass.shape)
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        down_block1 = self.contract1(x)
        down_pool1 = self.maxpool1(down_block1)
        down_block2 = self.contract2(down_pool1)
        down_pool2 = self.maxpool2(down_block2)
        down_block3 = self.contract3(down_pool2)
        down_pool3 = self.maxpool3(down_block3)
        down_block4 = self.contract4(down_pool3)
        down_pool4 = self.maxpool4(down_block4)
        # Base of 'U'
        reverse_direction = self.reverse_direction(down_pool3)
        # Move upward with concatonation and without crop due to pooling

        up_block4 = reverse_direction if not self.skip_connect else self.crop_and_concat(reverse_direction,
                                                                                         down_block3)
        cat_layer3 = self.expansion4(up_block4)

        up_block3 = cat_layer3 if not self.skip_connect else self.crop_and_concat(cat_layer3,
                                                                                  down_block2)
        cat_layer2 = self.expansion3(up_block3)
        up_block2 = cat_layer2 if not self.skip_connect else self.crop_and_concat(cat_layer2, down_block1)
        # cat_layer1 = self.expansion2(up_block2)
        # up_block1 = cat_layer1 if not self.skip_connect else self.crop_and_concat(cat_layer1, down_block1)
        final_block = self.final_layer(up_block2)
        return final_block



    def __group_pairs(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i: i + 2])

    def collect_boxes(self, region_list, number_samples=None, resize=False, run_num=-1, training_set=False):
        boxes = [
            i for i in self.__group_pairs(region_list)
        ]
        test_dataset = []
        number_samples = number_samples+1 if number_samples is not None else number_samples
        if training_set:
            number_boxes = range(len(boxes))[0:number_samples]
        else:
            number_boxes = range(len(boxes))
        if training_set:
            self.x_set = []
            self.y_set = []
        current_box_dict = {}
        for current_box_idx in number_boxes:

            run_num += 1

            counter_file = os.path.join(LocalSetup.project_base_path, 'run_count.txt')
            f = open(counter_file, 'r')
            c = f.readlines()
            run_num = int(c[0]) + 1
            f.close()
            f = open(counter_file, 'w')
            f.write(str(run_num))
            f.close()
            print("&&&& run num ", run_num)

            active_file = os.path.join(LocalSetup.project_base_path, 'continue_active.txt')
            f = open(active_file, 'w')
            f.write('0')
            f.close()

            #self.update_run_info()


            current_box = boxes[current_box_idx]

            for bounds in current_box:
                name_value = bounds.split(' ')
                print(name_value)

                # window selection(s)
                # for single training box window
                if name_value[0] not in current_box_dict.keys():
                    current_box_dict[name_value[0]] = list(map(int, name_value[1].split(',')))
                # for multiple boxes
                else:
                    current_box_dict[name_value[0]].extend(list(map(int, name_value[1].split(','))))

            X_BOX = [
                i for i in self.__group_pairs([i for i in current_box_dict['x_box']])
            ]
            Y_BOX = [
                i for i in self.__group_pairs([i for i in current_box_dict['y_box']])
            ]

            #     out_folder = os.path.join(self.pred_session_run_path)
            #     self.write_selection_bounds(dir=out_folder,x_box=name_value[0], y_box=name_value[1], mode='a')
            training_set, test_and_val_set = self.box_select_geomsc_training(x_range=X_BOX,
                                                                                  y_range=Y_BOX)

            #
            # ensure selected training is reasonable
            #
            flag_class_empty = False
            cardinality_training_sets = 0
            for i, t_class in enumerate(training_set):
                flag_class_empty = len(t_class) == 0 if not flag_class_empty else flag_class_empty
                cardinality_training_sets += len(t_class)
                print("LENGTH .. Training Set", i, 'length:', len(t_class))
            print(".. length test: ", len(test_and_val_set))
            # skip box if no training arcs present in region
            if cardinality_training_sets < 1 or flag_class_empty:
                removed_file = os.path.join(self.LocalSetup.project_base_path,
                                            'datasets', self.params['write_folder'],
                                            'removed_windows.txt')
                if not os.path.exists(removed_file):
                    open(removed_file, 'w').close()
                removed_box_file = open(os.path.join(self.LocalSetup.project_base_path,
                                                     'datasets', self.params['write_folder'],
                                                     'removed_windows.txt'), 'a+')
                removed_box_file.write(
                    str(self.run_num) + ' x_box ' + str(X_BOX[0][0]) + ',' + str(X_BOX[0][1]) + '\n')
                removed_box_file.write(
                    str(self.run_num) + ' y_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
                continue

            self.get_train_test_val_sugraph_split(collect_validation=False, validation_hops=1,
                                                       validation_samples=1, test_samples=None)

            all_validation = self.validation_set_ids["positive"].union(self.validation_set_ids["negative"])
            all_selected = [self.selected_positive_arc_ids, self.selected_negative_arc_ids]
            # self.model.selected_positive_arc_ids.union(self.model.selected_negative_arc_ids)
            # if not self.check_valid_partitions(all_selected, all_validation):
            #     removed_box_file = open(os.path.join(self.LocalSetup.project_base_path,
            #                                          'datasets', self.params['write_folder'],
            #                                          'removed_windows.txt'), 'a+')
            #     removed_box_file.write(
            #         str(self.run_num) + ' x_box ' + str(X_BOX[0][0]) + ',' + str(X_BOX[0][1]) + '\n')
            #     removed_box_file.write(
            #         str(self.run_num) + ' y_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
            #     continue

            if training_set:
                self.x_set += X_BOX
                self.y_set += Y_BOX

            X = self.image.shape[0]
            Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]

            if resize:
                resize = (self.X_train, self.Y_train) if (self.X, self.Y) != (self.X_train, self.Y_train) else False
            self.resize = resize

            #self.data_array = self.train_dataloader
            test_dataset = self.get_data_crops(self.image, x_range=X_BOX,
                                                    y_range=Y_BOX, dataset=test_dataset, resize=resize)
        if training_set:
            inf_or_train_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
            val_dataset = test_dataset
        else:
            inf_or_train_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
            val_dataset = test_dataset
        return inf_or_train_dataset, val_dataset

    def infer(self, running_best_model, dataset=None, training_window_file=None,
              load=False, view_results=False,
              infer_subsets=False, test=True, pred_thresh=0.5):

        #self.UNet = UNet

        image, msc_collection, mask, segmentation = self.train_dataloader[
            int(self.params['train_data_idx'])]



        f = open(training_window_file, 'r')
        box_dict = {}
        param_lines = f.readlines()
        f.close()




        if test:
            param_lines = param_lines[0:6]



        test_dataset = dataset
        if infer_subsets:
            test_dataset, val_dataset = self.collect_boxes( region_list = param_lines ,
                                                            training_set=False,
                                                            number_samples=None)
        else:
            X = image.shape[0]
            Y = image.shape[1] if len(image.shape) == 2 else image.shape[2]
            X_train = self.X_train
            Y_train = self.Y_train

            data_array = self.train_dataloader

            new_width = round(self.X_train * Y / X)
            new_height = round(self.Y_train * X / Y)

            if self.resize:
                self.resize = (self.X_train, self.Y_train) if (X, Y) != (self.X_train, self.Y_train) else False

            test_dataset = self.get_data_crops(image, x_range=[(0, X)],
                                                    y_range=[(0, Y)], dataset=test_dataset,
                                                    resize=self.resize)

            test_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
            test_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
        if load:
            running_best_model = torch.load('UNet_F1_opt.pth')
        else:
            running_best_model = running_best_model  # self.UNet.running_best_model

        print(running_best_model)

        given = False
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        # hand_seg = dataset(self.data_array, with_hand_seg=True)

        if 'sci' in LocalSetup.project_base_path:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        running_best_model.eval()

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor(self.class_weights, dtype=torch.float)).to(device)

        test_loss, running_val_loss = 0, 0
        val_loss, running_val_loss, loss_val_norm = 0, 0, 0
        test_losses, val_imgs, val_segs, val_img_preds = [], [], [], []
        F1_scores = []
        F1_score_img, labels_img, predictions_img = None, None, None

        total_inf_img = np.zeros((self.X , self.Y))
        total_gt_seg = np.zeros((self.X , self.Y)).astype(np.int8)
        total_img = np.zeros((self.X , self.Y))
        pad = 8
        im, se, _ = test_dataset[0]

        pred_list = []
        range_list = []
        seg_tile = np.zeros([0, 1, se.shape[0], se.shape[1]])

        num_val = 0
        self.run_num = 0
        out_folder = ''
        with torch.no_grad():
            for image, segmentation, ranges in test_loader:
                seg_tile = np.zeros([0, 1, se.shape[0], se.shape[1]])


                # segmentation = hand_segmentation

                num_val += 1
                image, segmentation = image.to(device), segmentation.to(device)
                image = image.unsqueeze(1)

                X = image.shape[-2]
                Y = image.shape[-1]

                segmentation = segmentation.unsqueeze(1)  # .permute(1,2,0)

                predicted = running_best_model(deepcopy(image))

                predicted_seg = predicted#.cpu().detach().numpy()
                pred_tile =  predicted_seg.cpu().detach().numpy()#np.concatenate((pred_tile,
                #                             predicted_seg),
                #                             axis=0)
                pred_list.append(pred_tile)
                ranges = ranges

                ground_truth_seg = segmentation.cpu().detach().numpy()
                seg_tile = deepcopy(ground_truth_seg)
                image_subset = image.cpu().detach().numpy()

                print("    * ", type(image_subset))
                print("    * ", type(predicted_seg))

                # only consider loss for pixels
                # within masked region

                # print('mask ', mask.shape, ' pred ', predicted.shape, ' image ', image.shape, ' seg ', segmentation.shape)

                # Compute Loss from forward pass
                # val_loss = criterion(predicted, segmentation)#[:,:,:1], segmentation.permute(1,2,0)[:,:,:1])

                running_val_loss += val_loss  # .item()
                # collect sample to observe performance
                val_imgs.append(image)
                val_segs.append(segmentation)  # .cpu().numpy())
                val_img_preds.append(predicted_seg)  # .cpu().numpy())

                test_losses.append(val_loss)  # .item())

                self.run_num = num_val - 1
                self.update_run_info(batch_multi_run=str(self.training_size))

                # F1_score, labels, predictions = get_score_model(running_best_model, test_loader,
                #                                                          X=self.X, Y=self.Y)
                # segmentation = og_segmentation[None].cpu().detach().numpy()
                # self.ground_truth_seg = segmentatiodimen
                if self.resize:
                    dprint("Resizing image during inference...")
                    if (X, Y) != (self.X_train, self.Y_train):
                        image = resize_img(image_subset[0, 0, :, :],
                                           Y=X, X=Y, sampling='lanczos')
                        segmentation = resize_img(ground_truth_seg[0, 0, :, :],
                                                  Y=X, X=Y, sampling='hamming')
                        predicted = resize_img(predicted[0, 0, :, :],
                                               Y=segmentation.shape[-1], X=segmentation.shape[-1],
                                               sampling='lanczos')
                        image = image[None][None]
                        predicted = predicted[None][None]
                        segmentation = segmentation[None][None]

                        ground_truth_seg, predicted_seg, image_subset = segmentation,\
                                                                                                predicted_seg,\
                                                                                                image
                X = X if self.resize else self.X_train
                Y = Y if self.resize else self.Y_train



                if view_results:
                    with torch.no_grad():
                        self.see_image(
                            image_seg_set=(image_subset, ground_truth_seg, predicted),
                            as_grey=True, save=False)  # False)


                dprint(ranges)
                ranges = ranges.cpu().detach().numpy()
                x_range = list(map(int , ranges[0][1][0]))
                y_range = list(map(int,ranges[0][0][0]))
                range_list.append([x_range,y_range])


                print("    * xrange yrange", x_range, y_range)
                with torch.no_grad():
                    #total_inf_img[x_range[0]:x_range[1],y_range[0]:y_range[1]] = pred_tile[0,0,:,:]
                    total_img[x_range[0]:x_range[1], y_range[0]:y_range[1]] =  deepcopy(image_subset)
                    total_gt_seg[x_range[0]:x_range[1], y_range[0]:y_range[1]] = seg_tile

                    # self.save_image(image_seg_set=(image_subset, ground_truth_seg, predicted_segmentation),#(total_img, total_gt_seg, total_inf_img),
                    #                 as_grey=True,
                    #                 dirpath=out_folder)
                    #                #save=True)
                    out_folder = os.path.join(self.pred_session_run_path,
                                              str(self.training_size))
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)

                    self.write_selection_bounds(dir=out_folder, x_box=self.x_set, y_box=self.y_set,
                                                mode='w')


                    self.see_image(image_seg_set=(image_subset, ground_truth_seg, predicted_seg),
                                   names=("og_tile"+str(self.run_num), "groundseg_tile"+str(self.run_num),
                                          "pred_tile"+str(self.run_num)),

                                   as_grey=True,
                                    dirpath=out_folder,
                                    save=True)
                    # self.see_image(image_seg_set=(image_subset, ground_truth_seg, predicted_segmentation),
                    #                # (total_img, total_gt_seg, total_inf_img),
                    #                as_grey=True,
                    #                dirpath=out_folder,
                    #                save=False)

        for ranges, pred_im in zip(range_list, pred_list):
            x_range = ranges[0]
            y_range = ranges[1]
            total_inf_img[x_range[0]:x_range[1], y_range[0]:y_range[1]] = pred_im
            total_inf_img = total_inf_img.cpu().detach().numpy()
        dprint(total_inf_img.shape,"inf shape")
        dprint(total_gt_seg.shape,"gt seg shape")
        F1_score_img, labels_img, predictions_img = get_image_prediction_score(predicted=total_inf_img[None][None],
                                                                               segmentation=total_gt_seg[None][None],
                                                                               X=self.X, Y=self.Y)
        F1_score_topo, labels_topo, predictions_topo, \
        self.node_gid_to_prediction, self.node_gid_to_label = get_topology_prediction_score(predicted=total_inf_img[None][None],
                                                                                            segmentation=total_gt_seg[None][None],
                                                                                            gid_gnode_dict=self.gid_gnode_dict,
                                                                                            node_gid_to_prediction=self.node_gid_to_prediction,
                                                                                            node_gid_to_label=self.node_gid_to_label,
                                                                                            X=self.X, Y=self.Y,
                                                                                            pred_thresh=pred_thresh)
        #out_folder = os.path.join(self.pred_session_run_path, str(self.run_num))
        # self.pred_session_run_path = out_folder
        #
        out_folder = os.path.join(self.pred_session_run_path,
                                  str(self.training_size))

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        compute_prediction_metrics('unet', predictions_topo, labels_topo, out_folder)
        self.write_arc_predictions(dir=out_folder)
        self.draw_segmentation(dirpath=out_folder)
        self.write_gnode_partitions(dir=out_folder)
        # self.write_selection_bounds(dir=out_folder, x_box=self.x_set, y_box=self.y_set,
        #                             mode='w+')  # name_value[0], y_box=name_value[1], mode='a')






        self.see_image(image_seg_set=(total_img, total_gt_seg, total_inf_img),
                                    names=("total_og_tiled", "total_groundseg", "total_prediction"),
                                    as_grey=True,
                                    dirpath=out_folder,
                                    save=True)

        exp_folder = os.path.join(self.params['experiment_folder'], 'runs')

        # batch_metric_folder = os.path.join(exp_folder,
        #                                    str(self.training_size),'f1')
        # if not os.path.exists(batch_metric_folder):
        #     os.makedirs(batch_metric_folder)

        # compute_prediction_metrics('unet', predictions, labels, out_folder)
        #
        # UNet.write_arc_predictions(UNet.session_name)
        # UNet.draw_segmentation(dirpath=UNet.pred_session_run_path)
        print("    * ","after single batch")
        multi_run_metrics(model='unet', exp_folder=exp_folder,
                          batch_multi_run=True,
                          bins=7, runs=str(self.training_size),
                          plt_title=exp_folder.split('/')[-1])
        return test_losses, val_imgs, val_segs, val_img_preds, running_val_loss, F1_score_img, labels_img, predictions_img


    #
    #
    #                   Train U-Net
    #
    #
class UNet_Trainer:
    def __init__(self, UNet : UNetwork, train_dataset=None, val_dataset=None,
                 class_weights=None):
        sys.setrecursionlimit(3000)#10000)
        #print("     * : recursion limit ", )
        torch.cuda.empty_cache()
        #define_gpu_to_use(gpu_to_use=0)
        # instantiate your model here:

        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        if 'sci' in LocalSetup.project_base_path:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.UNet = UNet.to(self.device)
        self.params = self.UNet.params
        #self.attributes = UNet.get_attributes()
        # Instantiate U-Net network

        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.class_weights = UNet.class_weights

        if train_dataset is None or val_dataset is None:
            self.val_dataset = self.UNet.val_dataset
            self.train_dataset = self.UNet.train_dataset
        if class_weights is None:
            self.class_weights = UNet.class_weights



    def launch_training(self, view_results=False,pred_thresh=0.5):


        # Initialize weights
        self.UNet.apply(weights_init)

        optimizer = torch.optim.SGD(self.UNet.parameters(), lr=self.UNet.params['learning_rate'], momentum=0.9, nesterov=True)
        n_epochs = self.params['epochs']

        import torch.utils.data as dutil
        batch_size = 1
        # train_dataloaders = dutil.DataLoader(train_dataset, batch_size=batch_size,
        #                                   shuffle=True, num_workers=4)
        val_dataloaders = torch.utils.data.DataLoader(self.val_dataset, batch_size=int(batch_size),
                                           shuffle=False, num_workers=0)

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=int(batch_size),
                                                   shuffle=False, num_workers=0)
        dataloaders = {'val': val_dataloaders}  # 'train': train_dataloaders,

        # Learning rate is reduced after plateauing to stabilize the end of training.
        # use the learning rate scheduler as defined here. Example on how to integrate it to training in
        # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250], gamma=0.1)

        # train your model here:
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor(self.class_weights[1], dtype=torch.float)).to(self.device)

        steps = 0
        print_every = 3  # batch_size/2.
        val_sample_rate = 15000
        train_losses, test_losses, F1_scores = [], [], []
        val_imgs, val_segs, val_img_preds, sample_losses = [], [], [], []

        test_loss = 0

        running_best_model = copy.deepcopy(self.UNet)
        self.running_best_model = running_best_model
        best_loss = 1
        best_f1, max_f1 = 0, 0
        last_mark = 0
        # Loop over epochs
        self.predicted_segmentation = None
        self.UNet.train()
        for epoch in range(n_epochs):
            # Training
            scheduler.step()  # notify lr scheduler
            running_loss = 0

            iter_count = 0

            # for batch_idx, (image, segmentation, mask) in enumerate( RetinaDataset(train_dataset)):
            for image, segmentation, ranges in train_loader:
                steps += 1
                iter_count += 1
                #self.image = image


                image, segmentation = image.to(self.device), segmentation.to(self.device)
                print("image," , image.shape)
                print("seg,", segmentation.shape)
                self.X = image.shape[-2]
                self.Y = image.shape[-1]
                self.X_train = self.X
                self.Y_train = self.Y
                self.UNet.X_train = self.X
                self.UNet.Y_train = self.Y
                # Variable(val_batch.cuda(),volatile=True)
                # segmentation[segmentation > 0] = 1
                optimizer.zero_grad()  # zero gradients for forward/bakward pass

                image = image.unsqueeze(1)
                segmentation = segmentation.unsqueeze(1)

                # Forward pass with network model
                predicted = self.UNet(image)  # [None])
                self.predicted_segmentation = predicted.cpu().detach().numpy()
                self.ground_truth_seg = segmentation.cpu().detach().numpy()
                self.image_crop = image.cpu().detach().numpy()

                # only consider loss for pixels
                # within masked region
                # predicted = predicted * mask  # .permute(1,2,0)

                # segmentation = segmentation.permute(1,2,0)[0,:,:]
                # mask = mask.permute(1,2,0)

                # segmentation = segmentation * mask

                # Compute Loss from forward pass
                print("    *: pred size", predicted.size())
                print("    *: seg size", segmentation.size())
                train_loss = criterion(predicted, segmentation)  # [None])
                # train update with backprop
                train_loss.backward()
                optimizer.step()
                running_loss += train_loss.item()

                torch.cuda.empty_cache()

                if steps % print_every == 0:

                    self.UNet.eval()

                    test_loss, accuracy = 0, 0
                    val_loss, running_val_loss, loss_val_norm = 0, 0, 0
                    num_val = 0
                    with torch.no_grad():
                        for image, segmentation, _ in val_dataloaders:
                            num_val += 1
                            image, segmentation = image.to(self.device), segmentation.to(self.device)
                            image = image.unsqueeze(1)
                            segmentation = segmentation.unsqueeze(1)
                            # segmentation[segmentation > 0] = 1

                            predicted = self.UNet(image)

                            # only consider loss for pixels
                            # within masked region
                            # predicted = predicted * mask.permute(1, 2, 0)

                            #segmentation = segmentation.permute(1, 2, 0)[0, :, :]
                            # mask = mask.permute(1, 2, 0)

                            # segmentation = segmentation * mask

                            # Compute Loss from forward pass
                            val_loss = criterion(predicted, segmentation)
                            running_val_loss += val_loss.item()
                            # collect sample to observe performance
                            if steps % val_sample_rate:
                                sample_losses.append(running_val_loss / num_val)
                                val_imgs.append(image.cpu().numpy())
                                val_segs.append(segmentation.cpu().numpy())
                                val_img_preds.append(predicted.cpu().numpy())

                    # get F1 score

                    # val_score, segmentation_gt, binary_preds = get_score_model(self.UNet,
                    #                                                            val_dataloaders,
                    #                                                            X=self.X, Y=self.Y)
                    predicted = predicted.cpu().detach().numpy()  # [0,:,:,:]
                    segmentation = segmentation.cpu().detach().numpy()
                    F1_score_img, labels_img, predictions_img = get_image_prediction_score(predicted=predicted,
                                                                                           segmentation=segmentation,
                                                                                           X=self.X, Y=self.Y)
                    F1_score_topo, labels_topo, predictions_topo, _, _= get_topology_prediction_score(
                        predicted=predicted,
                        segmentation=segmentation,
                        gid_gnode_dict=self.UNet.gid_gnode_dict,
                        node_gid_to_prediction=self.UNet.node_gid_to_prediction,
                        node_gid_to_label=self.UNet.node_gid_to_label,
                        pred_thresh=pred_thresh,
                        X=self.X, Y=self.Y,
                    ranges=ranges)

                    current_training_loss = running_loss / print_every
                    current_validation_loss = running_val_loss / num_val

                    train_losses.append(current_training_loss)
                    test_losses.append(current_validation_loss)
                    F1_scores.append(F1_score_topo)#val_score)

                    print("Epoch {epoch}/{epochs}.. ".format(epoch=epoch + 1, epochs=n_epochs))
                    print("Train loss: {rl}.. ".format(rl=current_training_loss))  # loss over 20 iterations

                    print("Validation loss: {test_loss}.. ".format(test_loss=current_validation_loss))
                    print("Validation F1 image: {acc}".format(acc=F1_score_img))
                    print("Validation F1 topo: {acc}".format(acc=F1_score_topo))

                    # update lr if plateau in val_loss
                    # plat_lr_scheduler.step(loss_val_mean/loss_val_norm) #update learning rate if validation loss plateaus

                    if F1_score_topo > best_f1:
                        max_f1 = F1_score_topo
                    if F1_score_topo > best_f1:  # and val_score > 0.35:
                        best_f1 = F1_score_topo
                        running_best_model = copy.deepcopy(self.UNet)
                        self.running_best_model = running_best_model
                        #torch.save(running_best_model, 'UNet_F1_opt.pth')

                    print("current opt F1: ", best_f1)
                    print(".....")
                    # My GPU is small I need to free memory
                    del current_training_loss
                    del current_validation_loss
                    del predicted
                    del segmentation
                    del image

                    running_loss = 0

                    torch.cuda.empty_cache()

                    self.UNet.train()
        self.UNet = self.UNet.cpu()#.detach()

        if max_f1 > best_f1:
            best_f1 = max_f1
        print('iterations per epoch: ', iter_count)
        #torch.save(running_best_model, 'UNet_trained.pth')

        if view_results:
            with torch.no_grad():
                    self.UNet.view_image(image_seg_set=(self.image_crop, self.ground_truth_seg, self.predicted_segmentation),
                                         as_grey=True)#False)

        print(train_losses, test_losses, F1_scores, best_f1, val_imgs, val_segs, sample_losses, val_img_preds)
        torch.cuda.empty_cache()
        return train_losses, test_losses, F1_scores, best_f1, val_imgs, val_segs, sample_losses, val_img_preds, running_best_model


class  UNet_Classifier:
    def __init__(self, UNet : UNetwork, training_window_file):

        self.UNet  = UNet
        self.train_dataloader = UNet.train_dataloader
        self.image, self.msc_collection, self.mask, self.segmentation = self.train_dataloader[int(UNet.params['train_data_idx'])]

        self.params = UNet.params
        #self.attributes = UNet.get_attributes()
        self.node_gid_to_feature = UNet.node_gid_to_feature
        print("       * 8*****",self.node_gid_to_feature)



        f = open(training_window_file, 'r')
        self.box_dict = {}
        self.param_lines = f.readlines()
        f.close()

        self.param_lines = self.param_lines[0:4]
        self.test_dataset = None

    def __group_pairs(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def collect_boxes(self, resize=False):
        boxes = [
            i for i in self.__group_pairs(self.param_lines)
        ]
        test_dataset = []
        for current_box_idx in range(len(boxes)):

            self.UNet.run_num += 1

            self.counter_file = os.path.join(LocalSetup.project_base_path, 'run_count.txt')
            f = open(self.counter_file, 'r')
            c = f.readlines()
            self.run_num = int(c[0]) + 1
            f.close()
            f = open(self.counter_file, 'w')
            f.write(str(self.run_num))
            f.close()
            print("&&&& run num ", self.run_num)

            self.active_file = os.path.join(LocalSetup.project_base_path, 'continue_active.txt')
            f = open(self.active_file, 'w')
            f.write('0')
            f.close()

            self.UNet.update_run_info()

            current_box_dict = {}
            current_box = boxes[current_box_idx]

            for bounds in current_box:
                name_value = bounds.split(' ')
                print(name_value)

                # window selection(s)
                # for single training box window
                if name_value[0] not in current_box_dict.keys():
                    current_box_dict[name_value[0]] = list(map(int, name_value[1].split(',')))
                # for multiple boxes
                else:
                    current_box_dict[name_value[0]].extend(list(map(int, name_value[1].split(','))))

            X_BOX = [
                i for i in self.__group_pairs([i for i in current_box_dict['x_box']])
            ]
            Y_BOX = [
                i for i in self.__group_pairs([i for i in current_box_dict['y_box']])
            ]

            training_set, test_and_val_set = self.UNet.box_select_geomsc_training(x_range=X_BOX,
                                                                             y_range=Y_BOX)

            partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                self.UNet.node_gid_to_partition,
                self.UNet.node_gid_to_feature,
                self.UNet.node_gid_to_label,
                test_all=True)

            gid_features_dict = partition_feat_dict['all']
            gid_label_dict = partition_label_dict['all']
            train_gid_label_dict = partition_label_dict['train']
            train_gid_feat_dict = partition_feat_dict['train']

            self.UNet.get_train_test_val_sugraph_split(collect_validation=False, validation_hops=1,
                                                  validation_samples=1, test_samples=None)


            self.X = self.image.shape[0]
            self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]
            self.X_train = self.UNet.X_train
            self.Y_train = self.UNet.Y_train

            if resize:
                resize = (self.X_train , self.Y_train) if (self.X, self.Y) != (self.X_train , self.Y_train) else False
            self.resize = resize

            self.data_array = self.train_dataloader
            test_dataset = self.UNet.get_data_crops(self.image, x_range=X_BOX,
                                                     y_range=Y_BOX, dataset=test_dataset, resize=resize)


        self.test_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
        self.val_dataset = self.test_dataset
        self.class_weights = self.UNet.class_weights



    def infer(self,running_best_model, load=False, view_results=False, infer_subsets=False):
        test_dataset=[]
        if infer_subsets:
            self.collect_boxes()
        else:
            self.X = self.image.shape[0]
            self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]
            self.X_train = self.UNet.X_train
            self.Y_train = self.UNet.Y_train


            self.data_array = self.train_dataloader

            new_width = round(self.X_train * self.Y/self.X)
            new_height = round(self.Y_train * self.X/self.Y)

            if self.resize:
                resize = (self.X_train, self.Y_train)  if (self.X, self.Y) != (self.X_train, self.Y_train) else False

            test_dataset = self.UNet.get_data_crops(self.image, x_range=[(0,self.X)],
                                                    y_range=[(0,self.Y)], dataset=test_dataset,
                                                    resize=self.resize)

            self.test_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
            self.val_dataset = self.test_dataset
            self.class_weights = self.UNet.class_weights
        self.test_dataset = dataset(test_dataset, do_transform=False, with_hand_seg=False)
        if load:
            running_best_model = torch.load('UNet_F1_opt.pth')
        else:
            running_best_model = running_best_model# self.UNet.running_best_model

        print(running_best_model)



        given = False
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
        #hand_seg = dataset(self.data_array, with_hand_seg=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        running_best_model.eval()

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor(self.class_weights,dtype=torch.float)).to(device)

        test_loss, running_val_loss = 0, 0
        val_loss, running_val_loss, loss_val_norm = 0, 0, 0
        test_losses, val_imgs, val_segs, val_img_preds = [], [], [], []
        F1_scores = []
        num_val = 0
        with torch.no_grad():
            for image, segmentation, og_segmentation in test_loader:
                #segmentation = hand_segmentation

                num_val += 1
                image, segmentation = image.to(device), segmentation.to(device)
                image = image.unsqueeze(1)

                X = image.shape[-2]
                Y = image.shape[-1]
                print("    * : X ", X, "Y ",Y)

                print("    *: im size ",image.shape)
                print("    *: seg size",segmentation.shape)

                segmentation = segmentation.unsqueeze(1)  # .permute(1,2,0)

                predicted = running_best_model(image)




                self.predicted_segmentation = predicted.cpu().detach().numpy()
                self.ground_truth_seg = segmentation.cpu().detach().numpy()
                self.image_subset = image.cpu().detach().numpy()

                print("    * ",type(self.image_subset))
                print("    * ",type(self.predicted_segmentation))


                # only consider loss for pixels
                # within masked region
                predicted = predicted.cpu().detach().numpy()#[0,:,:,:]
                segmentation = segmentation.cpu().detach().numpy()#.permute(1,2,0)#[0, 0, :, :] #* mask
                print("    *: pred size", predicted.shape)
                print("    *: seg size", segmentation.shape)
                # print('mask ', mask.shape, ' pred ', predicted.shape, ' image ', image.shape, ' seg ', segmentation.shape)

                # Compute Loss from forward pass
                #val_loss = criterion(predicted, segmentation)#[:,:,:1], segmentation.permute(1,2,0)[:,:,:1])

                running_val_loss += val_loss#.item()
                # collect sample to observe performance
                val_imgs.append(image)
                val_segs.append(segmentation)#.cpu().numpy())
                val_img_preds.append(predicted)#.cpu().numpy())

                test_losses.append(val_loss)#.item())

                self.UNet.run_num = num_val-1
                self.UNet.update_run_info()

                #F1_score, labels, predictions = get_score_model(running_best_model, test_loader,
                #                                                          X=self.X, Y=self.Y)
                #segmentation = og_segmentation[None].cpu().detach().numpy()
                #self.ground_truth_seg = segmentatiodimen
                if self.resize:
                    if (self.X, self.Y) != (self.X_train, self.Y_train):
                        image = resize_img(self.image_subset[0, 0, :, :],
                                           Y=self.X, X=self.Y, sampling='lanczos')
                        segmentation = resize_img(self.ground_truth_seg[0, 0, :, :],
                                                  Y=self.X, X=self.Y, sampling='hamming')
                        predicted = resize_img(self.predicted_segmentation[0, 0, :, :],
                                               Y=segmentation.shape[-1], X=segmentation.shape[-1], sampling='lanczos')
                        image = image[None][None]
                        predicted = predicted[None][None]
                        segmentation = segmentation[None][None]

                        self.ground_truth_seg, self.predicted_segmentation, self.image_subset = segmentation, predicted, image



                F1_score_img, labels_img, predictions_img = get_image_prediction_score(predicted=predicted,
                                                                           segmentation=segmentation,
                                                                           X=self.X, Y=self.Y)
                F1_score_topo, labels_topo, predictions_topo = get_topology_prediction_score(predicted=predicted,
                                                                           segmentation=segmentation,
                                                                                       gid_gnode_dict=self.UNet.gid_gnode_dict,
                                                                                       X=self.X, Y=self.Y)




                out_folder = self.UNet.pred_session_run_path
                compute_prediction_metrics('unet', predictions_img, labels_img, out_folder)
                #current_training_loss = running_loss / print_every
                current_validation_loss = running_val_loss / num_val

                #train_losses.append(current_training_loss)
                test_losses.append(current_validation_loss)
                F1_scores.append(F1_score_img)

                #print("Epoch {epoch}/{epochs}.. ".format(epoch=epoch + 1, epochs=n_epochs))
                #print("Train loss: {rl}.. ".format(rl=current_training_loss))  # loss over 20 iterations

                #print("Inference loss: {test_loss}.. ".format(test_loss=current_validation_loss))
                print("Inference F1 over image: {acc}".format(acc=F1_score_img))
                print("Inference F1 over topology: {acc}".format(acc=F1_score_topo))

                if view_results:
                    with torch.no_grad():
                        self.UNet.view_image(image_seg_set=(self.image_subset, self.ground_truth_seg, self.predicted_segmentation),
                                             as_grey=True)#False)


        return test_losses, val_imgs, val_segs, val_img_preds, running_val_loss, F1_score_img, labels_img, predictions_img









# Visualing a few cases in the training set
def view_UNet_preds_during_training(val_imgs, val_segs, sample_losses, val_img_preds, num_im=20,
                                    train_or_test='Training', net_2=False):
    half_max = num_im
    print("...")
    if train_or_test == 'Training':
        print("....Initial and Final Predictions During Training....")
    else:
        print("....Predictions During Testing....")
        half_max = 0
    print("...")

    if train_or_test == 'Training':
        for idx, image in enumerate(val_imgs):
            half_max -= 1
            if half_max > 0:
                # print(image.shape)
                plt.figure()
                plt.title("Input Image")
                plt.imshow(image.transpose((1, 2, 0)))
                plt.figure()
                plt.title("Segmentation ground truth")
                # print(val_img_preds[idx].shape)
                if train_or_test == 'Training' and not net_2:
                    # plt.imshow(val_segs[idx].transpose((1,2,0))[0,:,:])#.cpu().numpy())
                    plt.imshow(val_segs[idx][0, :, :])
                else:
                    plt.imshow(val_segs[idx].transpose((1, 2, 0))[:, :, 0])
                plt.figure()
                plt.title("Predicted Segmentation W/ LOSS: " + str(sample_losses[idx]))
                plt.imshow(val_img_preds[idx][0, 0, :, :])  # .cpu().numpy())
    half_max = 0
    for idx, image in enumerate(val_imgs[::-1]):
        half_max += 1
        if half_max < num_im:
            # print(image.shape)
            plt.figure()
            plt.title("Input Image")
            plt.imshow(image.transpose((1, 2, 0)))
            plt.figure()
            plt.title("Segmentation ground truth")
            # print(val_img_preds[idx].shape)
            if train_or_test == 'Training' and not net_2:
                plt.imshow(val_segs[idx][0, :, :])  # .cpu().numpy())
                plt.figure()
                plt.title("Predicted Segmentation W/ LOSS: " + str(sample_losses[idx]))
                plt.imshow(val_img_preds[idx][0, 0, :, :])  # .cpu().numpy())
            else:
                plt.imshow(val_segs[idx].transpose((1, 2, 0))[:, :, 0])
                plt.figure()
                plt.title("Predicted Segmentation W/ LOSS: " + str(sample_losses[idx]))
                plt.imshow(val_img_preds[idx][0, :, :])  # .cpu().numpy())