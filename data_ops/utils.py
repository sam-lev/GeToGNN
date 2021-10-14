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



def dbgprint(x,name=""):
    print("    *")
    print("    *:", name, x)
    print("    *")