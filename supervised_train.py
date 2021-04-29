import numpy as np
import os

from getognn import GeToGNN
from localsetup import LocalSetup
from proc_manager import Run_Manager

LocalSetup = LocalSetup()

#
# selection of dataset done here
#
sample_idx = 0

#
#for naming, reassigned in getognn
#
experiment_num = 1
experiment_name = "fourthDimStepHalf"
window_file_base = 'windows_'+experiment_name+'.txt'
run_num = 0
model_name = 'GeToGNN'#'experiment'#'input_select_from_1st_inference'


#
# input
#
name = ['retinal','neuron1', 'neuron2','mat2_lines', 'berghia'][sample_idx]
image = ['im0236_o_700_605.raw',
         'MAX_neuron_640_640.raw',
         'MAX__0030_Image0001_01_o_1737_1785.raw',
         'sub_CMC_example_o_969_843.raw',
         'berghia_o_891_897.raw'][sample_idx]   # neuron1
label_file = ['im0236_la2_700_605.raw.labels_2.txt',
              'MAX_neuron_640_640.raw.labels_3.txt',
              'MAX__0030_Image0001_01_s2_C001Z031_1737_1785.raw.labels_4.txt',
              'sub_CMC_example_l1_969_843.raw.labels_0.txt',
              'berghia_prwpr_e4_891_896.raw.labels_3.txt'][sample_idx]   #neuron1
msc_file = os.path.join(LocalSetup.project_base_path, 'datasets', name,
                        'input',label_file.split('raw')[0]+'raw')
#'/retinal/input', 'im0236_o_700_605.raw')
ground_truth_label_file = os.path.join(LocalSetup.project_base_path, 'datasets',
                                       name,'input', label_file )
#/retinal/input/im0236_la2_700_605.raw.labels_2.txt'
write_path = 'experiment_'+str(experiment_num)+'_'+experiment_name
feature_file = os.path.join(name,write_path,'features',model_name)
window_file = os.path.join(LocalSetup.project_base_path,"datasets",
                           name, write_path,
                           window_file_base)
parameter_file_number = 1
format = 'raw'



getognn = GeToGNN(training_selection_type='box',
                  run_num=run_num,
                  parameter_file_number = parameter_file_number,
                  name=name,
                  image=image,
                  feature_file=feature_file,
                  geomsc_fname_base = msc_file,
                  label_file=ground_truth_label_file,
                  write_folder=write_path,
                 model_name=model_name,
                  load_feature_graph_name=None,
                  write_json_graph = False)

# features
if not getognn.params['load_features']:
    getognn.compile_features()
else:
    getognn.load_gnode_features(filename=model_name)
if getognn.params['write_features']:
    getognn.write_gnode_features(getognn.session_name)
if getognn.params['write_feature_names']:
    getognn.write_feature_names(getognn.session_name)

# training info, selection, partition train/val/test
getognn.read_labels_from_file(file=ground_truth_label_file)

training_set , test_and_val_set = getognn.box_select_geomsc_training(x_range=getognn.params['x_box'], y_range=getognn.params['y_box'])

getognn.get_train_test_val_sugraph_split(collect_validation=True, validation_hops = 1,
                                         validation_samples = 1)
if getognn.params['write_json_graph']:
    getognn.write_json_graph_data(folder_path=getognn.pred_session_run_path, name=model_name + '_' + getognn.params['name'])


getognn.write_gnode_partitions(getognn.session_name)
getognn.write_selection_bounds(getognn.session_name)

# random walks
if not getognn.params['load_preprocessed_walks']:
    walk_embedding_file = os.path.join(getognn.LocalSetup.project_base_path, 'datasets',
                                       getognn.params['write_folder'],'walk_embeddings',
                                       'run-'+str(getognn.run_num)+'_walks')
    getognn.params['load_walks'] = walk_embedding_file
    getognn.run_random_walks(walk_embedding_file=walk_embedding_file)


#training
getognn.supervised_train()
G = getognn.get_graph()
getognn.equate_graph(G)

getognn.write_arc_predictions(getognn.session_name)
getognn.draw_segmentation(dirpath=getognn.pred_session_run_path)

run_manager = Run_Manager(getognn=getognn,
                          training_window_file=window_file,
                          features_file=model_name,
                          sample_idx=sample_idx,
                          model_name=model_name,
                          format=format)
if not getognn.params['load_features']:
    getognn.params['write_features'] = False
    getognn.params['load_features'] = True
    getognn.params['write_feature_names'] = False
    getognn.params['save_filtered_images'] = False
    getognn.params['collect_features'] = False
run_manager.perform_runs()

