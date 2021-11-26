model =  'pixel'# 'unet'#
import os
from localsetup import LocalSetup
LocalSetup = LocalSetup()
model_type = os.path.join(LocalSetup.project_base_path, 'model_type.txt')
log_model = open(model_type, 'w')
log_model.write(model)
log_model.close()


from proc_manager import experiment_manager
# send experiment runs 
#for dim in [str(16), str(64), str(256)]:
#
# input
#
name = ['retinal',                      # 0
             'neuron1',                 # 1
             'neuron2',                 # 2
             'mat2_lines',              # 3
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              # 7
        'foam_cell',                    # 8
        'diadem_sub1',                  # 9
        'berghia_membrane']             # 10

box_list = [[[ 500,680,300,450], [ 0,250,50,200]],                              # 0
             'neuron1',                 # 1
             [[ 700,1200,600,800],[ 700,900,1400,1600], [0,200,1500,1700]],    # 2
             'mat2_lines',              # 3
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              # 7
         [[ 200, 500,200,500]],                                                 # 8
        [[530,702,810,982]] ,                                                   # 9
        [ [ 500,550,100,150], [ 350, 400,650, 700]]]                            # 10
# brg og[ [ 500,850,100,350], [ 350, 550,650, 850]]

dim_image = [[605,700],                      # 0
             'neuron1',                 #
             [1785,1737],                 # 2
             'mat2_lines',              #
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              #
        [846,828],                    # 8
        [1438,1170],                  # 9
        [896,891]]                   # 10

batch = 1
plot_only = True
data_index = [10]
def unit_run():
    exp_runner = None
    for idx in data_index:
        for dataset_idx, window_file in zip([idx], [name[idx]]):
            window_file = window_file+'_growing_windows_infer.txt'#_growifoamng_window.txt'
            for exp,model in zip(["Random_Forest_Pixel"],['random_forest']):#["GNN"],['getognn']):#
                #["Random_Forest_Pixel"],['random_forest']):#
                # ["UNet"],['unet']):#
                # for exp in ['fourth_windows_hidden-geto_maxpool']:
                exp_runner = experiment_manager.runner(experiment_name=exp,
                                                       sample_idx=dataset_idx,
                                                       window_file_base=window_file,
                                                       parameter_file_number=1,
                                                       multi_run=True,
                                                       clear_runs= not plot_only)
                #                            !        !  ^
                #                           ! CLEAR? ! _/
                #                          !        !
                if not plot_only:
                    learn_type = 'supervised'
                    if model == 'random_forest':
                        learn_type = 'pixel' if 'Pixel' in exp else 'msc'

                    boxes = box_list[idx]
                    dims = dim_image[idx]

                    exp_runner.start(model, boxes=boxes, dims=dims, learning=learn_type)

            plot_models = 1
            if plot_models:
                    exp_runner.multi_model_metrics(['Random Forest Pixel'],#'GNN','UNet',
                                                   ["Random_Forest_Pixel"], None)




def batch_runs():
    # launch random Forest
    for dataset_idx, window_file in zip([10,9,8,2,0],[name[10],name[9],name[8],name[2],name[0]]):
        window_file = window_file+'_growing_windows.txt'
        exp_runner = None
        for exp, model in zip(["UNet","Random_Forest_Pixel",'GNN',"Random_Forest_MSC"],
                              [  'random_forest','getognn', 'random_forest']):
            # if dataset_idx==10:
            #     continue
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True,
                                                   clear_runs=True)
            learn_type = 'supervised'
            if model == 'random_forest':
                learn_type = 'pixel' if 'Pixel' in exp else 'msc'

            init_box = box_list[dataset_idx]
            dim_im = dim_image[dataset_idx]

            exp_runner.start(model, boxes = init_box , dims=dim_im, learning=learn_type)

        exp_runner.multi_model_metrics(['UNet', 'GNN','Random Forest MSC', 'Random Forest Pixel'],
                                       ['UNet', 'GNN', "Random_Forest_MSC", "Random_Forest_Pixel"], None)




if batch:
    batch_runs()
else:
    unit_run()