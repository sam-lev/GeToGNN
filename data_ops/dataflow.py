import os
import gzip
import shutil
import imageio
import numpy as np
from localsetup import LocalSetup

LocalSetup = LocalSetup()
from data_ops.utils import *

class dataflow:
    def __init__(self, persistence_values=[], blur_sigmas=[], training_data_path=None, validation_data_path=None,
                 test_data_path=None
                 , training_write_path=None, validation_write_path=None, test_write_path=None
                 , training_set=None, validation_set=None, test_set=None):
        self.validation_set = validation_set
        self.training_set = training_set
        self.test_set = test_set
        self.data_array = []

    def __getitem__(self, index):
        sample = [x for x in self.data_array[index]]
        return sample

    def __len__(self):
        return len(self.data_array)

    # split can be 'train', 'val', and 'test'
    # this is the function that splits a dataset into training, validation and testing set
    # We are using a split of 70%-10%-20%, for train-val-test, respectively
    # this function is used internally to the defined dataset classes
    def get_split(self, array_to_split, split, shuffle=True):
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(array_to_split)
            np.random.seed()
        if split == 'train':
            array_to_split = array_to_split[:int(len(array_to_split) * 0.7)]
        elif split == 'val':
            array_to_split = array_to_split[int(len(array_to_split) * 0.7):int(len(array_to_split) * 0.8)]
        elif split == 'test':
            array_to_split = array_to_split[int(len(array_to_split) * 0.8):]
        elif split is None:
            return array_to_split
        return array_to_split

    def read_images(self, filetype, dest_folder, color_invert=False, image=None, dim_invert=False):
        all_images = []
        if filetype != 'raw':
            sorted_items = sorted(os.listdir(dest_folder))
            if image is not None:
                sorted_items = [image]
            for item in sorted_items:
                if dest_folder[-1] != '/':
                    dest_folder = dest_folder + "/"
                if item.endswith(filetype):
                    img = imageio.imread(dest_folder + item)
                    if len(img.shape) == 3:
                        img = np.pad(img, ((12, 12), (69, 70), (0, 0)), mode='constant')
                    else:
                        img = np.pad(img, ((12, 12), (69, 70)), mode='constant')
                    img = img.astype(np.float32)
                    if len(img.shape) == 2:
                        img = img.astype(np.float32)
                        img = np.expand_dims(img, axis=2)
                    all_images.append(img)
        else:
            sorted_items = sorted(os.listdir(dest_folder))
            if image is not None:
                sorted_items = [image]
            for item in sorted_items:
                if not item.endswith('.raw'):
                    continue
                if dest_folder[-1] != '/':
                    dest_folder = dest_folder + "/"
                print(dest_folder + item)

                path = dest_folder + item

                X = int(path.split("_")[-2])
                Y = int(path.split("_")[-1].split('.')[0])
                image = np.fromfile(dest_folder+item, dtype="float32")[:(X * Y)].reshape((Y, X))
                print("shape im: ", image.shape)


                if dim_invert:
                    xtemp = X
                    X = Y
                    Y = xtemp

                image = np.reshape(image, (X, Y))
                max_p = np.max(image)

                print("..max pixel ", max_p)
                if max_p < 1:
                    image = 255 * image
                all_images.append(image)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.title("Input Image")
        # plt.imshow(all_images[0])
        # plt.show()
        return all_images

    def transpose_first_index(self, x, with_hand_seg=False):
        if not with_hand_seg:
            x2 = (np.transpose(x[0], [2, 0, 1]), np.transpose(x[1], [2, 0, 1]), np.transpose(x[2], [2, 0, 1]))
        else:
            x2 = (np.transpose(x[0], [2, 0, 1]), x[1], np.transpose(x[2], [2, 0, 1]),
                  np.transpose(x[3], [2, 0, 1]))
        return x2

class retinadataset(dataflow):
    def __init__(self, retina_array=None, split='train', do_transform=False
                 , with_hand_seg=False, shuffle=True):
        super(retinadataset, self).__init__()
        self.with_hand_seg = with_hand_seg
        if retina_array is not None:
            indexes_this_split = self.get_split(np.arange(len(retina_array), dtype=np.int), split, shuffle=shuffle)
            self.retina_array = [self.transpose_first_index(retina_array[i], self.with_hand_seg) for i in
                                 indexes_this_split]
        self.split = split
        self.do_transform = do_transform

    def __getitem__(self, index):
        sample = [x for x in self.retina_array[index]]
        return sample

    def stare_read_images(self, tar_filename, dest_folder, do_mask=False):
        all_images = []
        all_masks = []
        for item in sorted(os.listdir(dest_folder)):
            if dest_folder[-1] != '/':
                dest_folder = dest_folder + "/"
            if item.endswith('gz'):
                with gzip.open(dest_folder + item, 'rb') as f_in:
                    with open(dest_folder + item[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(dest_folder + item)
            img = imageio.imread(os.path.join(dest_folder, item))
            if len(img.shape) == 3:
                img = np.pad(img, ((1, 2), (2, 2), (0, 0)), mode='constant')
            else:
                img = np.pad(img, ((1, 2), (2, 2)), mode='constant')
            img = img / 255.
            img = img.astype(np.float32)
            if len(img.shape) == 2:
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis=2)
            all_images.append(img)
            if do_mask:
                mask = (1 - remove_small_regions(np.prod((img < 50 / 255.) * 1.0, axis=2) > 0.5, 1000)) * 1.0
                mask = np.expand_dims(mask, axis=2)
                all_masks.append(mask.astype(np.float32))
            if do_mask:
                return all_images, all_masks
            else:
                return all_images

    def drive_read_images(self, filetype, dest_folder):
        all_images = []
        sorted_items = sorted(os.listdir(dest_folder))
        for item in sorted_items:
            if dest_folder[-1] != '/':
                dest_folder = dest_folder + "/"
            if item.endswith(filetype):
                img = imageio.imread(dest_folder + item)
                if len(img.shape) == 3:
                    img = np.pad(img, ((12, 12), (69, 70), (0, 0)), mode='constant')
                else:
                    img = np.pad(img, ((12, 12), (69, 70)), mode='constant')
                img = img / 255.
                img = img.astype(np.float32)
                if len(img.shape) == 2:
                    img = img.astype(np.float32)
                    img = np.expand_dims(img, axis=2)
                all_images.append(img)
        return all_images

    def get_retina_array(self, partial=False, use_local_setup=True, msc=True
                         , stare_only=False, drive_training_only=False, drive_test_only=False
                         ,persistence_values=[], env='multivax', number_images=None):

        if use_local_setup:

            drive_training_path = LocalSetup.drive_training_path
            drive_segmentation_path = LocalSetup.drive_training_segmentation_path
            drive_test_path = LocalSetup.drive_test_path
            drive_training_mask_path = LocalSetup.drive_training_mask_path
            drive_test_segmentation_path = LocalSetup.drive_test_segmentation_path
            drive_test_mask_path = LocalSetup.drive_test_mask_path
            stare_image_path = LocalSetup.stare_training_data_path
            stare_segmentation_path = LocalSetup.stare_segmentations
            drive_training_msc_segmentation_path = LocalSetup.drive_training_msc_segmentation_path
            stare_msc_segmentation_path = LocalSetup.stare_msc_segmentation_path
        else:
            stare_image_path = 'datasets/stare/images/'
            stare_segmentation_path = 'datasets/stare/segmentations/'
            drive_segmentation_path = 'datasets/drive/DRIVE/training/1st_manual/'
            drive_test_path = 'datasets/drive/DRIVE/test/images/'
            drive_training_path = 'datasets/drive/DRIVE/training/images/'
            drive_training_mask_path = 'datasets/drive/DRIVE/training/mask/'
            drive_test_segmentation_path = 'datasets/drive/DRIVE/test/1st_manual/'
            drive_test_mask_path = 'datasets/drive/DRIVE/test/mask/'
            drive_training_msc_segmentation_path = 'datasets/drive/DRIVE/training/msc_seg'
            stare_msc_segmentation_path = 'datasets/stare/training/msc_seg'

        self.number_persistence_vals = len(persistence_values)
        stare_images, stare_mask = self.stare_read_images("ppm", stare_image_path, do_mask=True)
        stare_segmentation = self.stare_read_images("ppm", stare_segmentation_path)
        drive_training_images = self.drive_read_images('tif', drive_training_path)
        drive_training_mask = self.drive_read_images('gif', drive_training_mask_path)
        drive_test_images = self.drive_read_images('tif', drive_test_path)

        drive_training_segmentations = self.drive_read_images('gif', drive_segmentation_path)

        # hand draw ground truth
        drive_test_segmentation = self.drive_read_images('gif', drive_test_segmentation_path)
        drive_test_mask = self.drive_read_images('gif', drive_test_mask_path)

        if msc:
            # collect pre-computed msc from directories
            drive_training_msc_segmentations = [os.path.join(drive_training_msc_segmentation_path, o)
                                                for o in os.listdir(drive_training_msc_segmentation_path)
                                                if os.path.isdir(os.path.join(drive_training_msc_segmentation_path, o))]

            drive_training_msc = []
            for msc_seg in sorted(drive_training_msc_segmentations):
                msc_group = self.drive_read_images('tif', msc_seg)
                drive_training_msc += msc_group
            # stare
            stare_msc_segmentations = [os.path.join(stare_msc_segmentation_path, o) for o in
                                       os.listdir(stare_msc_segmentation_path) if
                                       os.path.isdir(os.path.join(stare_msc_segmentation_path, o))]
            stare_msc = []
            for msc_seg in sorted(stare_msc_segmentations):
                msc_group = self.drive_read_images('tif', msc_seg)
                stare_msc += msc_group
            if drive_training_only:
                total_msc_segmentation = drive_training_msc
            elif stare_only:
                total_msc_segmentation = stare_msc
            else:
                total_msc_segmentation = stare_msc + drive_training_msc
        else:
            #dummy space filler
            if drive_training_only:
                total_msc_segmentation = drive_training_segmentations
            elif stare_only:
                total_msc_segmentation = stare_segmentation
            elif drive_test_only:
                total_msc_segmentation = []
            else:
                total_msc_segmentation = stare_segmentation + drive_training_segmentations
        if drive_training_only:
            if number_images is None:
                self.retina_array = list(zip(drive_training_images,
                                             total_msc_segmentation,
                                             drive_training_mask,
                                             drive_training_segmentations))
            else:
                self.retina_array = list(zip(drive_training_images[:number_images],
                                             total_msc_segmentation[:number_images],
                                             drive_training_mask[:number_images],
                                             drive_training_segmentations[:number_images]))
            return self.retina_array
        if drive_test_only:
            if number_images is None:
                self.retina_array = list(zip(drive_test_images,
                                             drive_test_segmentation,
                                             drive_test_mask,
                                             drive_test_segmentation))
            else:
                self.retina_array = list(zip(drive_test_images[:number_images],
                                             drive_test_segmentation[:number_images],
                                             drive_test_mask[:number_images],
                                             drive_test_segmentation[:number_images]))
            return self.retina_array
        if stare_only:
            if number_images is None:
                self.retina_array = list(zip(stare_images,
                                             total_msc_segmentation,
                                             stare_mask,
                                             stare_segmentation))
            else:
                self.retina_array = list(zip(stare_images[:number_images],
                                             total_msc_segmentation[:number_images],
                                             stare_mask[:number_images],
                                             stare_segmentation[:number_images]))
            return self.retina_array
        if partial:
            if number_images is None:
                self.retina_array = list(zip(stare_images + drive_training_images + drive_test_images,
                                total_msc_segmentation + drive_test_segmentation,
                                stare_mask + drive_training_mask + drive_test_mask))
            else:
                self.retina_array = list(zip(stare_images[:number_images] + drive_training_images[:number_images] + drive_test_images[:number_images],
                                             total_msc_segmentation[:number_images] + drive_test_segmentation[:number_images],
                                             stare_mask[:number_images] + drive_training_mask[:number_images] + drive_test_mask[:number_images]))
        else:
            if number_images is None:
                self.retina_array = list(zip(stare_images + drive_training_images + drive_test_images,
                                total_msc_segmentation + drive_test_segmentation,
                                stare_mask + drive_training_mask + drive_test_mask,
                                stare_segmentation + drive_training_segmentations + drive_test_segmentation))
            else:
                self.retina_array = list(zip(stare_images[:number_images] + drive_training_images[:number_images] + drive_test_images[:number_images],
                                             total_msc_segmentation[:number_images] + drive_test_segmentation[:number_images],
                                             stare_mask[:number_images] + drive_training_mask[:number_images] + drive_test_mask[:number_images],
                                             stare_segmentation[:number_images] + drive_training_segmentations[:number_images] + drive_test_segmentation[:number_images]))
        return self.retina_array

class raw2ddataset(dataflow):
    def __init__(self, data_array=None, dataset='neuron2', env='multivax', image= None, split='train', do_transform=False
                 , with_hand_seg=False, shuffle=True, dim_invert = False):
        super().__init__()
        self.project_base_path = LocalSetup.project_base_path
        self.dataset = dataset
        self.dim_invert = dim_invert
        if data_array is not None:
            self.data_array = data_array
        else:
            self.data_array = self._get_raw_data(image=image)

    def get_raw_data(self):
        return self.data_array

    def _get_raw_data(self, number_images=None, image=None):

        data_folder =os.path.join(self.project_base_path,"datasets", str(self.dataset),'input')

        images = super().read_images(filetype='raw', dest_folder=data_folder, image =image,
                                     color_invert=False, dim_invert=self.dim_invert)
        msc_list = images
        mask= [None]
        training_msc_list = [None]

        if number_images is None:
            self.data_array = list(zip(images,
                                       msc_list,
                                       mask,
                                       training_msc_list))
        else:
            self.data_array = list(zip(images[:number_images],
                                       msc_list,
                                       mask,
                                       training_msc_list))
        return self.data_array