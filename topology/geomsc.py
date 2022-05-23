
from skimage import io
from PIL import Image
import numpy as np
import copy

import imageio
from skimage.io import imsave
from skimage.util import invert
from skimage import exposure
import subprocess

#class with local file paths
from ml.features import *
from localsetup import *
from localsetup import LocalSetup as LS
from ml.utils import pout
import getograph

def scale_intensity( original_image, fname_base, scale_range=(0, 255)):
    scaled_image = exposure.rescale_intensity(original_image)  # , in_range=(0, 255))
    fname_raw = fname_base + "_scaled.raw"
    scaled_image.tofile(fname_raw)
    return scaled_image, fname_raw

def blur_and_save( original_image, fname_base, blur_sigma=2, grey_scale=True):
    blurred_image = gaussian_blur_filter(original_image, sigma=blur_sigma, as_grey=grey_scale).astype(
        "float32"
    )
    fname_raw = fname_base# + '.raw'#"_smoothed.raw"
    blurred_image.tofile(fname_raw)
    return blurred_image, fname_raw


def augment_image_channels(original_image, fname_base, channels=[0, 1]):
    import copy
    import cv2
    augmented_image = copy.deepcopy(original_image)

    # [ 0 = blue, 1 = green, 2 = red ]
    for c in channels:
        augmented_image[:, :, c] = 0  # cv2.equalizeHist(augmented_image[:,:,c])
    fname_components = fname_base.split('.')
    fname_aug = fname_components[0] + "_aug." + fname_components[1]
    imsave(fname_aug, augmented_image)
    return augmented_image, fname_aug

def call_geomscsegmentation( image_filename=None
                   , image=None
                   , geomsc_exec_path='.'
                   , X=None, Y=None
                   , persistence=1
                   , blur=True, blur_sigma=2
                   , write_path='', delete_msc_files=False
                   , fname_raw=None
                   , invert_image=False
                   , grey_scale=True
                   , scale_intensities=False
                   , augment_channels=[]):

    if write_path is None and image_filename is not None:
        write_path = os.path.dirname(image_filename)
        os.mkdir(os.path.join(write_path, 'raw_images'))
    img_name = ""
    # if image_filename is not None:
    #     img_name = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]

    use_image = True if image is not None else False

    image = image_filename if image is None else image  # np.mean(image,axis=0) if grey_scale else image
    if image_filename is not None and image_filename.split('.')[-1] == 'raw':
        if '/' in image_filename:
            image_no_path_name = image_filename.split(os.path.dirname(image_filename))[-1].split('/')[-1]
        else:
            image_no_path_name = image_filename
        fname_raw = os.path.join(write_path, image_no_path_name)
    if fname_raw:
        fname = image_filename
        image = np.fromfile(fname, dtype="float32")[:(X * Y)].reshape((X, Y))
        # image = io.imread(fname, as_gray=grey_scale, flatten=True)
        if invert_image:
            image = invert(image)
        #image.astype('float32').tofile(fname_raw)
    else:
        if '/' in image_filename:
            image_no_path_name = image_filename.split(os.path.dirname(image_filename))[-1].split('/')[-1]
        else:
            image_no_path_name = image_filename
        fname_raw = os.path.join(write_path, image_no_path_name)

    if scale_intensities:
        raw_image = image if image is not None else np.fromfile(image, dtype="float32")[:(X * Y)].reshape((X, Y))
        if invert_image:
            raw_image = invert(raw_image)
        if write_path:
            raw_path = fname_raw#os.path.join(write_path.rsplit(".", 1)[0], 'raw_images', img_name)
        else:
            raw_path = image.rsplit(".", 1)[0]
        image, fname_raw = scale_intensity(raw_image,
                                           raw_path, scale_range=(0, 255))

    if len(augment_channels) > 0:
        im_name = img_name

        im = image if image is not None else np.fromfile(image, dtype="float32")[:(X * Y)].reshape((X, Y))  # cv2.imread(image)
        im = np.array(im)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        write_dir = write_path#os.path.join(write_path.rsplit(".", 1)[0], 'augmented_images')
        if write_path:
            raw_path = fname_raw#os.path.join(write_dir, img_name)
        else:
            raw_path = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]  # image.rsplit(".", 1)[0]
        type = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[1]
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        im, image = augment_image_channels(im, raw_path + '.' + type, augment_channels)  # im_name.split('.')[1]

    if blur:
        raw_image = image if image is not None else np.fromfile(image, dtype="float32")[:(X * Y)].reshape((X, Y))
        if invert_image:
            raw_image = invert(raw_image)
        if write_path:
            raw_path = fname_raw#os.path.join(write_path.rsplit(".", 1)[0],  img_name)
        else:
            raw_path = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
        image, fname_raw = blur_and_save(raw_image, raw_path , blur_sigma=blur_sigma,
                                         grey_scale=grey_scale) #+ 'PERS' + str(persistence)
    else:
        raw_image = image if image is not None else np.fromfile(image, dtype="float32")[:(X * Y)].reshape((X, Y))
        if invert_image:
            raw_image = invert(raw_image)
        if write_path:
            raw_path = os.path.join(write_path.rsplit(".", 1)[0], img_name)
        else:
            raw_path = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
        fname_raw = raw_path + ".raw" #'PERS' + str(persistence)
        image = raw_image.astype("float32")
        image.tofile(fname_raw)

    sanity_check = False
    if sanity_check:
        print("SHAPE ##### : ", image.shape)
        print("fname raw ###### : ", fname_raw)
        io.imshow(np.transpose(image, (1, 2, 0)))
        from matplotlib import pyplot as plt
        plt.show()

    if fname_raw is None:
        fname_raw = image_filename
    # hard coded path, different for other systems
    # will remain like this while still editing
    # geometric morse smale complex. After which
    # we will use pybind to eliminate the use of the shell
    # utilizing Attila's script directly.
    # /home/sam/Documents/PhD/Research/GradIntegrator/build/extract2dridgegraph
    # print("%%%%%%%%%%%% image file path %%%%%%: ", fname_raw)

    starting_dir = geomsc_exec_path  # os.path.join( os.path.dirname(os.path.abspath(__file__)), "..")
    msc_build = "/home/sam/Documents/PhD/Research/GradIntegrator/build/extract2dridgegraph"


    # print("list ex dir  %%%%%%%%%%", os.listdir(starting_dir))

    if X is None and Y is None:
        X = image.shape[1]
        Y = image.shape[2]

    proc = subprocess.Popen([
        os.path.join(msc_build, "extract2dridgegraph"),  # note: '.' needed to run executable.
        fname_raw,
        str(Y),
        str(X),
        str(persistence)],
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = proc.communicate()
    # for line in proc.stderr:
    #    sys.stdout.write(line)
    # for line in proc.stdout:
    #    sys.stdout.write(line)
    geomsc = getograph.GeToGraph(geomsc_fname_base=fname_raw)
    #geomsc.read_from_file(fname_raw)

    if delete_msc_files:
        raw_folder = os.path.join(write_path.rsplit(".", 1)[0], 'raw_images')
        for the_file in os.listdir(raw_folder):
            file_path = os.path.join(raw_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    return geomsc




def msc_arc_accuracy(self, vertex=None,geomsc=None, labeled_segmentation=None,
                     labeled_mask=None, invert=True):
    geo_element_accuracy = 0
    percent_interior = 0
    for point in vertex.points:
        x = 0
        y = 1
        if invert:
            x = 1
            y = 0
        if labeled_segmentation[int(point[x]),int(point[y])] > 0:
            geo_element_accuracy += 1.
        if labeled_mask[int(point[x]),int(point[y])] > 0:
            percent_interior += 1.
    label_accuracy = geo_element_accuracy/float(len(vertex.points))
    if label_accuracy == 0.:
        label_accuracy = 1e-4
    return label_accuracy

def map_labeling(image=None, geomsc=None, labeled_segmentation=None, labeled_mask = None, invert=False):

    for vertex in geomsc.gid_vertex_dict.values():
        vertex.label_accuracy = msc_arc_accuracy(vertex=vertex
                                                    , labeled_segmentation=labeled_segmentation
                                                    ,labeled_mask=labeled_mask
                                                    ,invert=invert)



    return geomsc

def label_msc(geomsc=None, labeled_segmentation=None, labeled_mask=None, invert=False):
    labeled_msc = map_labeling(geomsc=geomsc, labeled_segmentation=labeled_segmentation,
                                      labeled_mask = labeled_mask, invert=invert)
    return labeled_msc

def compute_geomsc( persistence_values, blur_sigmas,X=None,Y=None
                          , data_buffer = None, data_path = None, segmentation_path=None,
                          write_path = None, labeled_segmentation=None, label = False
                          , save=False, save_binary_seg=False, number_images=None, persistence_cardinality = None
                          , valley=True, ridge=True, env='multivax'):
    #LocalSetup = LS(env=env)

    # check needed folders present else make
    # if not os.path.exists(os.path.join(write_path, 'raw_images')):
    #     os.mkdir(os.path.join(write_path, 'raw_images'))
    # iterate through images and compute msc for each image
    # at various persistence values
    images = None
    if data_path and segmentation_path is not None and os.path.isdir(data_path):
        images = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,
                                                                                         f)) and any(image_type in f.rsplit('.', 1)[1] for image_type in ['tif','gif','jpg','png','ppm','vk.ppm','raw'])])
        seg_image_paths = sorted([f for f in os.listdir(segmentation_path)])
        seg_images = []
        for path in seg_image_paths:
            #print(segmentation_path," path: ", path)
            seg_images.append(imageio.imread(os.path.join(segmentation_path, path)))
        im_count = range(len(images))
        if data_buffer is None:
            print("no data buffer given composing new data buffer")
            data_buffer = zip(images, seg_images, im_count)
    else:
        data_buffer = zip([None],[None],[None],[None])
        images = [data_path]
        number_images = 1 if number_images is None else number_images
    if persistence_cardinality is None:
        persistence_cardinality = {}
        for i in range(number_images):
            persistence_cardinality[i] = number_images
    persistence_cardinality = copy.deepcopy(persistence_cardinality)

    msc_segmentations = []
    images_ = []
    masks_ = []
    segmentations_ = []
    count = 0
    image_cap = -1
    for image, msc_collection , mask, segmentation in data_buffer:
        image_cap+=1
        if number_images is not None and image_cap == number_images:
            break

        im_path = images[count]
        labeled_segmentation = None
        labeled_mask = None
        if segmentation is not None:
            segmentation[segmentation > 0] = 1
            labeled_segmentation = np.mean(segmentation, axis=0)
        if mask is not None:
            mask[mask > 0] = 1
            labeled_mask = np.mean(mask, axis=0)
            rgblabeled_mask = np.zeros_like(image)
            rgblabeled_mask[0, :, :] = mask
            rgblabeled_mask[1, :, :] = mask
            rgblabeled_mask[2, :, :] = mask
            print("labeled mask shape ", labeled_mask.shape)
        # collect to return data buffer with msc
        images_.append(image)
        masks_.append(mask)
        segmentations_.append(segmentation)
        count+=1
        msc_collection= {}
        for blur_sigma in sorted(blur_sigmas):
            for pers_count, pers in enumerate(sorted(persistence_values)):
                pers_cap = persistence_cardinality[pers_count]
                if pers_cap <= 0:
                    continue


                #image = copy.deepcopy(image)
                if mask is not None:
                    image[rgblabeled_mask==0] = 1.0

                image_name_and_path = im_path#os.path.join(data_path,im_path)
                print(">>>>")
                print(image_name_and_path)
                print(">>>>")
                # if len(image.shape) == 2:
                #     X=image.shape[0]
                #     Y=image.shape[1]
                # else:
                #     X = image.shape[2]
                #     Y = image.shape[1]
                if X is None or Y is None:
                    X = int(im_path.split('_')[-2])
                    Y = int(im_path.split('_')[-1].split('.')[0])
                geomsc = call_geomscsegmentation(image_filename =  image_name_and_path
                                           ,image=image
                                           ,X=X, Y=Y
                                           ,geomsc_exec_path=None#os.path.join(LocalSetup.project_base_path,'..')
                                           , persistence = pers
                                           , blur=True
                                           , blur_sigma=blur_sigma
                                           , grey_scale=True
                                           , write_path = write_path
                                           , scale_intensities=False
                                           , augment_channels = [])

                if label:
                    geomsc = label_msc(geomsc=geomsc
                                                 ,labeled_segmentation=labeled_segmentation
                                                 ,labeled_mask=labeled_mask,invert=True)
                # compute geomsc over image
                if save:
                    image_filename =  os.path.join(data_path,im_path)
                    img_name = image_filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
                    msc_seg_path = os.path.join(write_path, 'msc_seg')

                    if not os.path.exists(os.path.join(msc_seg_path, 'blur_'+str(blur_sigma)+'persistence_' + str(pers))):
                        os.mkdir(os.path.join(msc_seg_path,'blur_'+str(blur_sigma)+ 'persistence_' + str(pers)))
                    msc_seg_path = os.path.join(msc_seg_path, 'blur_'+str(blur_sigma)+ 'persistence_' + str(pers))



                    seg_img = os.path.join(write_path, 'ground_truth_seg', img_name + '_seg.gif')
                    msc_path_and_name = os.path.join(msc_seg_path, str(count-1) + 'Blur'+str(blur_sigma)+'pers' + str(pers) + '-MSC.tif')
                    geomsc.draw_segmentation(filename=msc_path_and_name
                                          , X=X, Y=Y
                                          , reshape_out=False, dpi=164
                                          , valley=True, ridge=True,original_image=image)

                    geomsc.write_getograph(filename=msc_path_and_name, msc=geomsc, label=label)

                persistence_cardinality[pers_count] = pers_cap - 1
                print("computed msc for persistence:")
                print(pers)
                print("over image:")
                print(image_name_and_path)

                msc_collection[(pers,blur_sigma)] = geomsc

        msc_segmentations.append(geomsc)#msc_collection)

    # if data_buffer is not None:
    #     data_buffer_with_msc = list(zip(images_
    #                                     ,msc_segmentations
    #                                     ,masks_
    #                                     ,segmentations_))
    #     return data_buffer_with_msc
    # else:
    return msc_segmentations[0]


class MSCNode:
    def __init__(self):
        self.arcs = []

    def read_from_line(self, line):
        tmplist = line.split(',')
        self.id = int(tmplist[0])
        #self.index = int(tmplist[1])
        #self.value = float(tmplist[2])
        #self.boundary = int(tmplist[3])
        self.points = [(float(tmplist[1]), float(tmplist[2]))]

    def add_arc(self, arc):
        self.arcs.append(arc)


class MSCArc:
    def __init(self):
        self.node_ids = []
        self.label_accuracy = None

    def __group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def read_from_line(self, line):
        tmplist = line.split(',')
        self.id = int(tmplist[0])
        self.node_ids = [int(tmplist[1]), int(tmplist[2])]
        self.points = [
            i for i in self.__group_xy([float(i) for i in tmplist[3:]])
        ]


class MSC:
    def __init__(self):
        self.nodes = {}
        self.arcs = []
        self.image = None

    def read_from_file(self, fname_base):
        nodesname = fname_base + ".nodes.txt"
        arcsname = fname_base + ".arcs.txt"
        node_file = open(nodesname, "r")
        nodes_lines = node_file.readlines()
        node_file.close()
        for l in nodes_lines:
            n = MSCNode()
            n.read_from_line(l)
            self.nodes[n.id] = n
        arcs_file = open(arcsname, "r")
        arcs_lines = arcs_file.readlines()
        arcs_file.close()
        for l in arcs_lines:
            a = MSCArc()
            a.read_from_line(l)
            n1 = self.nodes[a.node_ids[0]]
            n2 = self.nodes[a.node_ids[1]]
            n1.add_arc(a)
            n2.add_arc(a)
            a.nodes = [n1, n2]
            self.arcs.append(a)

