import os
import numpy as np
from skimage.measure import block_reduce
from skimage import morphology
from PIL import Image




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
    for j in range(int(increment_y * increment_x )):# removed * incj

        Y_START =Y_START + step_Y*step
        Y_STOP = Y_START + step_Y


        if  Y_STOP > Y_END :
            y_box = (int(Y_END - step_Y), int(Y_END))
            Y_START = Y_INIT
        else:
            y_box = (int(Y_START), int(Y_STOP))

        for i in range( int(increment_x ) ):  # * (1. / step))):

            X_START = X_START + step_X * step
            X_STOP = X_START + step_X


            if X_STOP > X_END :
                x_box = ( int(X_END - step_X), int(X_END))
                X_START = X_INIT
            else:
                x_box = (int(X_START), int(X_STOP))
            if (x_box, y_box) not in box_set:
                box_set.append((x_box, y_box))
            begun_y = False
    for j in range(int(increment_y * increment_x )):# removed * incj

        Y_START =Y_START + step_Y*step
        Y_STOP = Y_START + step_Y


        if  Y_STOP > Y_END :
            y_box = (int(Y_END - step_Y), int(Y_END))
            Y_START = Y_INIT
        else:
            y_box = (int(Y_START), int(Y_STOP))

        for i in range( int(increment_x ) ):  # * (1. / step))):

            X_START = X_START + step_X * step
            X_STOP = X_START + step_X


            if X_STOP > X_END :
                x_box = ( int(X_END - step_X), int(X_END))
                X_START = X_INIT
            else:
                x_box = (int(X_START), int(X_STOP))
            if (x_box, y_box) not in box_set:
                box_set.append((x_box, y_box))
            begun_y = False




    outer = x_boxes if len(x_boxes) < len(y_boxes) else y_boxes
    inner = x_boxes if len(x_boxes) > len(y_boxes) else y_boxes
    # for x_box,y_box in zip(x_boxes, y_boxes):
    #    box_set.add((x_box, y_box))
    # for x_box in x_boxes:
    #     for y_box in y_boxes:
    #         box_set.add((x_box, y_box))
    return box_set


def dbgprint(x,name=""):
    print("    *")
    print("    *:", name, x)
    print("    *")