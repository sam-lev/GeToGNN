from proc_manager import experiment_manager
# send experiment runs 
#for dim in [str(16), str(64), str(256)]:
#
# input
#
name = ['retinal',                     # 0
             'neuron1',                # 1
             'neuron2',                # 2
             'mat2_lines',             # 3
             'berghia',                # 4
             'faults_exmouth',         # 5
             'transform_tests',        # 6
             'map_border']             # 7

batch = 0

def unit_run():
    for dataset_idx, window_file in zip([0], [name[0]]):
        window_file = window_file+'_half_step_sliding_windows.txt'
        for exp in ["UNet_test"]:#"]:#,"fourth_windows_hidden-geto-edge"]:
            # for exp in ['fourth_windows_hidden-geto_maxpool']:
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True)
            exp_runner.start('unet', learning='supervised')

def batch_runs():
    # launch random Forest
    for dataset_idx, window_file in zip([2,0], [name[2]+'_unet_sliding_windows.txt',
                                                   name[0]+'_unet_sliding_windows.txt',
                                                    # name[5]+'_unet_sliding_windows.txt',
                                                    #'transform_fourth_halfStep_windows.txt',
                                                    # ,#zip([0,5,6],['quarterWindow_fullStep.txt',#'windows_fourthDimStepHalf.txt',
                                                    #'neuron2_fourth_window_fullStep.txt'
                                                    ]):
        # for exp, model in zip(["fourth_windows_graphsage-meanpool","Random_Forest"],
        #                ['getognn' , 'random_forest']):
        for exp, model in zip(["Random_Forest"],
                              ['random_forest']):
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True)
            exp_runner.start(model, learning='supervised')

    '''for dataset_idx, window_file in zip([2],[# 'retinal_fourth_window_fullStep.txt',
                                                 #'transform_fourth_halfStep_windows.txt',
                                             #    'fault_fourth_window_fullStep.txt',
                                                'neuron2_fourth_window_fullStep.txt'
                                                 ]):
        for exp in ["fourth_windows_hidden-geto-maxpool",
                    "fourth_windows_hidden-geto-meanpool",
                    "fourth_windows_graphsage-meanpool",
                    "fourth_windows_graphsage-maxpool",
            "fourth_windows_hidden-geto-meanpool_geto-sampling",
            "fourth_windows_hidden-geto-edge",
                    #"fourth_windows_geto-meanpool",
                    #"fourth_windows_geto-maxpool",
                    #"fourth_windows_geto-meanpool_geto-sampling",
                    #"fourth_windows_geto-sampling_maxpool"
                    ]:

            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True)#True)
            exp_runner.start('getognn', learning='supervised')'''


if batch:
    batch_runs()
else:
    unit_run()