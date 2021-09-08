from proc_manager import experiment_manager
# send experiment runs 
#for dim in [str(16), str(64), str(256)]:

def unit_run():
    for dataset_idx, window_file in zip([0], ['retinal_fourth_window_fullStep.txt']):
        for exp in ["fourth_windows_hidden-geto-meanpool"]:#,"fourth_windows_hidden-geto-edge"]:
            # for exp in ['fourth_windows_hidden-geto_maxpool']:
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=window_file,
                                                   parameter_file_number=1,
                                                   multi_run=False)#True)
            exp_runner.start('getognn', learning='supervised')

def batch_runs():
    # launch random Forest
    '''for dataset_idx, window_file in zip([  2], [#'retinal_fourth_window_fullStep.txt',
                                                    #'fault_fourth_window_fullStep.txt',
                                                    #'transform_fourth_halfStep_windows.txt',
                                                    # ,#zip([0,5,6],['quarterWindow_fullStep.txt',#'windows_fourthDimStepHalf.txt',
                                                    'neuron2_fourth_window_fullStep.txt'
                                                    ]):
        for exp in [
                    "Random_Forest"]:
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True)
            exp_runner.start('random_forest')'''

    for dataset_idx, window_file in zip([2],[# 'retinal_fourth_window_fullStep.txt',
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
            exp_runner.start('getognn', learning='supervised')



batch_runs()
#unit_run()