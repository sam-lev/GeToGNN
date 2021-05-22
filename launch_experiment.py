from proc_manager import experiment_manager
# send experiment runs
#for dim in [str(16), str(64), str(256)]:
for exp in ["getognn_2nbrsample"]:#, "getognn_4nbrsample", "getognn_meanpoolAgg"]:
    exp_runner = experiment_manager.runner(experiment_name=exp
                                           , sample_idx=6,
                                           window_file_base='quarterWindow_fullStep.txt',
                                           multi_run=False)
    exp_runner.start('getognn')

# launch random Forest
#exp_runner = experiment_manager.runner(experiment_name="randForest_fourth_halfstep"
#                                       , sample_idx=6,
#                                       window_file_base='transform_fourth_halfStep_windows.txt')
#exp_runner.start('random_forest')