model =  'msc'#
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
#                                                         590    375         125     125
box_list = [[[375,439,590,654 ], [125,189,125,189]]   ,# [[500,680,300,450 ], [ 0,250, 50,200,]]                                  # 0
             'neuron1',                 # 1
            [[700, 764,950, 1014], [1500, 1564,700, 828], [1600, 1728,64, 192]]  ,
            #                                      # [[700,1200,600,800],[700,900,1400,1600], [0,200,1500,1700]] # 2
             'mat2_lines',              # 3               950     700        800    1500        100     1600
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              # 7                 350      350
        [[ 350,414,350, 414]],#,                      [[ 200, 500,200,500]],                         # 8
            #                                             616     896
        [[896,960,675,739]],      #                  [[530,702,810,982]]                             # 9
            #                                             675      225       450      750
        [[225,289,675,739], [750,814,450,514]] ]    #[ [500,850,100,350], [350,550,650,850]]         # 10


dim_image = [[700,605],                      # 0
             'neuron1',                 #
             [1737,1785],                 # 2
             'mat2_lines',              #
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              #
        [828,846],                    # 8
        [1170,1438],                  # 9
        [891,896]]                   # 10


batch         = 0
plot_only     = 1
region_thresh = 0
clear_runs = True if not plot_only and region_thresh == 0 else False

def unit_run():
    exp_runner = None
    for dataset_idx in [10]:
        # window_file = window_file+'_growing_windows_infer.txt'#_growifoamng_window.txt'
        pm = False
        for exp,model in zip(['Random_Forest_MSC','Random_Forest_Pixel'],
                              ['random_forest','random_forest']):
            #["Random_Forest_Pixel"],['random_forest']):#
            # ["UNet"],['unet']):#
            # for exp in ['fourth_windows_hidden-geto_maxpool']:
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=None,#window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True,
                                                   percent_train_thresh=region_thresh,
                                                   clear_runs= not plot_only)
            #                            !        !  ^
            #                           ! CLEAR? ! _/
            #                          !        !
            if not plot_only:
                learn_type = 'supervised'
                if model == 'random_forest':
                    learn_type = 'pixel' if 'Pixel' in exp else 'msc'

                boxes = box_list[dataset_idx]
                dims = dim_image[dataset_idx]

                exp_runner.start(model, boxes=boxes, dims=dims, learning=learn_type)

            pm = 'Pixel' in exp

        plot_models = pm
        if plot_models:
            exp_runner.multi_model_metrics( [ "UNet", 'GNN', "Random_Forest_MSC", "Random_Forest_Pixel"],
                                           [ "UNet", 'GNN', "Random_Forest_MSC", "Random_Forest_Pixel"], None,metric='time')
            exp_runner.multi_model_metrics(["UNet", 'GNN', "Random_Forest_MSC", "Random_Forest_Pixel"],
                                           ["UNet", 'GNN', "Random_Forest_MSC", "Random_Forest_Pixel"], None)




def batch_runs():
    # launch random Forest
    for dataset_idx in [0]:#10   #10,8,9,2,
        exp_runner = None
        model_exp = []
        for exp, model in zip(['UNet', 'Random_Forest_Pixel','GNN','Random_Forest_MSC'],#,'GNN', 'Random_Forest_Pixel', 'Random_Forest_MSC'],
                              ['Unet','random_forest','getognn','random_forest']):#,'getognn', "random_forest", "random_forest"]):

            #window_file = window_file+'_growing_windows_infer.txt'

        

            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=None,#window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True,
                                                   percent_train_thresh=region_thresh,
                                                   clear_runs=not plot_only)
            if not plot_only:
                learn_type = 'supervised'
                if model == 'random_forest':
                    learn_type = 'pixel' if 'Pixel' in exp else 'msc'

                init_box = box_list[dataset_idx]
                dim_im = dim_image[dataset_idx]

                exp_runner.start(model, boxes = init_box , dims=dim_im, learning=learn_type)

                model_exp.append(exp)

        exp_runner.multi_model_metrics(['Random Forest Pixel'],# 'UNet','Random Forest Pixel','GNN', 'Random Forest MSC',],#'GNN','UNet',
                                               ['UNet', 'Random_Forest_Pixel','GNN','Random_Forest_MSC'], None, metric='time')
        #model_exp, model_exp, None)   #  'UNet',"Random_Forest_Pixel", 'GNN', "Random_Forest_MSC",




if batch:
    batch_runs()
else:
    unit_run()
