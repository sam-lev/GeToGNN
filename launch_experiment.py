from proc_manager import experiment_manager
# send experiment runs 
#for dim in [str(16), str(64), str(256)]:
for exp in ["fourth_windows_graphsage-meanpool",
            "fourth_windows_graphsage-maxpool","fourth_windows_geto-sampling_maxpool",
            "fourth_windows_geto-meanpool","fourth_windows_geto-meanpool_geto-sampling",
            "fourth_windows_geto-maxpool","fourth_windows_geto-maxpool_geto-sampling"]:

    exp_runner = experiment_manager.runner(experiment_name=exp,
                                           sample_idx=6,
                                           window_file_base='quarterWindow_fullStep.txt',
                                           parameter_file_number=1,
                                           multi_run=True)
    exp_runner.start('getognn', learning='supervised')

# launch random Forest
#exp_runner = experiment_manager.runner(experiment_name="exp1_randomForest"
#                                       , sample_idx=7,
#                                       window_file_base='quarterWindow_y.txt')
#exp_runner.start('random_forest')