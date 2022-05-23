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
name = ['retinal',                        # 0
             'neuron1',                 # 1
             'neuron2',                   # 2
             'mat2_lines',              # 3
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              # 7
        'foam_cell',                      # 8
        'diadem_sub1',                    # 9
        'berghia_membrane']               # 10
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
             [1737,1785],                    # 2
             'mat2_lines',              #
             'berghia',                 # 4
             'faults_exmouth',          # 5
             'transform_tests',         # 6
             'map_border',              #
        [828,846],                           # 8
        [1170,1438],                         # 9
        [891,896]]                           # 10


batch         =         0
plot_only     =         0
overide_plots =         0
region_thresh =         5
break_training_thresh = 25 #60 45,57
#feat control
load_features              = 1
compute_features           = 0
load_geto_features         = 0
compute_geto_features      = 0              # !!!!
feats_independent          = 1# geom / std separate # mlp unet and random need to node_gid_to_standard_feature
compute_complex = True
clear_runs = True if not plot_only else False

experiments = [ #      "UNet",
                #      "Random_Forest_Pixel",
                #      "Random_Forest_MSC",
                #      'Random_Forest_MSC_Geom',
                #      'GNN',
                #      'GNN_Geom',
                      'GNN_SUB',
                #      'MLP_MSC',
                #      'MLP_Pixel'
                ]
models      = [ #    'unet',
                #    'random_forest',
                #    'random_forest',        #     retinal   neuron   foam   diadem   berghia
                #    'getognn',              #        0         2       8       9       10
                    'getognn',
                #    'mlp',
                #    'mlp'
                ]
datasets    = [ 0 ]

plot_experiments = [  "UNet",
                      "Random_Forest_Pixel",
                      "Random_Forest_MSC",
                      #'Random_Forest_MSC_Geom',
                      'GNN',
                      'GNN_Geom',
                      'GNN_SUB',
                      'MLP_MSC',
                      'MLP_Pixel'
                ]

def unit_run():
    exp_runner = None
    for exp, model in zip(experiments,
                          models):
        current_exp = False
        #load_features = True if 'Geom' not in exp else False
        for dataset_idx in datasets:
            exp_runner = experiment_manager.runner(experiment_name=exp,
                                                   sample_idx=dataset_idx,
                                                   window_file_base=None,#window_file,
                                                   parameter_file_number=1,
                                                   multi_run=True,
                                                   load_features=load_features,
                                                   compute_features=compute_features,
                                                   load_geto_features=load_geto_features,
                                                   compute_geto_features=compute_geto_features,
                                                   feats_independent = feats_independent,
                                                   percent_train_thresh=region_thresh,
                                                   break_training_size=break_training_thresh,
                                                   clear_runs= not plot_only)
            #                            !        !  ^
            #                           ! CLEAR? ! _/
            #                          !        !
            if not plot_only:
                learn_type = 'subcomplex' if 'SUB' in exp else 'supervised'
                if model == 'random_forest' or model == 'mlp':
                    learn_type = 'pixel' if 'Pixel' in exp else 'msc'

                boxes = box_list[dataset_idx]
                dims = dim_image[dataset_idx]



                exp_runner.start(model, boxes=boxes, dims=dims, learning=learn_type,
                                 compute_complex=compute_complex)

            current_exp = exp

        plot_models = current_exp == experiments[-1] if not overide_plots else 0
        if plot_models:
            exp_runner.multi_model_metrics(plot_experiments,
                                           plot_experiments,
                                            None,
                                            metric='time')
            exp_runner.multi_model_metrics(plot_experiments,
                                           plot_experiments,
                                           None)




def batch_runs():
    # launch random Forest
    for dataset_idx in [0]:#10   #10,8,9,2,
        exp_runner = None
        model_exp = []
        for exp, model in zip(['UNet', 'Random_Forest_Pixel','GNN','Random_Forest_MSC'],#,'GNN', 'Random_Forest_Pixel', 'Random_Forest_MSC'],
                              ['unet','random_forest','getognn','random_forest']):#,'getognn', "random_forest", "random_forest"]):

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
