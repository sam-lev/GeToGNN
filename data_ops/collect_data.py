from topology.geomscsegmentation import geomscsegmentation
from .dataflow import retinadataset
from .dataflow import raw2ddataset
from localsetup import LocalSetup

def collect_datasets(dataset='neuron', name='neuron2', dim_invert=False, format='raw', image=None):
    print(" %%% collecting data buffers")
    if set == 'retina':
        data_set = retinadataset(with_hand_seg=True)
        DataSet = data_set.get_retina_array(partial=False, msc=False
                                                              , drive_training_only=True
                                                              , env='multivax')
        data_array = retinadataset(DataSet, split=None
                                           , do_transform=False, with_hand_seg=True, shuffle=False)
    if dataset == 'neuron' or format == 'raw':
        data_set = raw2ddataset(dataset=name, dim_invert=dim_invert,image=image)
        data_array = data_set.get_raw_data()
    print("%collection complete")
    return data_array, data_set


def compute_geomsc(params, data_array, data_path, segmentation_path, msc_write_path, map_labels):

    data_array = geomscsegmentation(persistence_values=params['persistence_values']
                                                                 , blur_sigmas=params['blur_sigmas']
                                                                 , data_buffer=data_array
                                                                 , data_path=data_path
                                                                 , segmentation_path=segmentation_path
                                                                 , write_path=msc_write_path
                                                                 , label=map_labels  # not self.select_label
                                                                 , save=False
                                                                 , valley=True, ridge=True
                                                                 , env='multivax'
                                                                 , number_images=params['number_images']
                                                                 , persistence_cardinality=params['persistence_cardinality'])

    return data_array

def collect_training_data(params, dataset, data_array,
                          dataset_group='neuron', name='neuron2', image=None,
                          format='raw', msc_file=None, dim_invert=False):
    ## Get total retina array with newly computed msc
    ## and partition into train, test and val

    print(" %%% performing data buffer train, validation, test split ")
    #images, msc_collections, masks, msc_segmentations = list(zip(*data_array))

    if dataset_group == 'retina':
        retina_dataset = data_array
        dataset.retina_array = retina_dataset
        dataset.get_retina_array(partial=False, msc=False
                                               , number_images=params['number_images'])
        train_dataloader = retinadataset(retina_dataset, split=None, image=image
                                                 , shuffle=False, do_transform=False, with_hand_seg=True)
        # val_dataloader = MSCRetinaDataset(retina_dataset, split = "val", do_transform = False, with_hand_seg=True)
        # test_dataloader = MSCRetinaDataset(retina_dataset, split = "test", do_transform = False, with_hand_seg=True)
        print(" %%% data buffer split complete")
    if dataset_group == 'neuron' or format == 'raw':
        train_dataloader = raw2ddataset(data_array, image=image, dataset=name)
    return train_dataloader