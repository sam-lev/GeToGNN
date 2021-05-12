from proc_manager import experiment_manager

exp_runner = experiment_manager.runner(experiment_name="exp_fourth_halfstep_getognn"
                                       , sample_idx=0,
                                       window_file_base='retinal_eighth_halfStep.txt')
exp_runner.start('getognn')
#exp_runner.update_run_info(experiment_folder_name="exp_1_randForest")
exp_runner = experiment_manager.runner(experiment_name="exp_fourth_halfstep_randForest"
                                       , sample_idx=0,
                                       window_file_base='retinal_eighth_halfStep.txt')
exp_runner.start('random_forest')