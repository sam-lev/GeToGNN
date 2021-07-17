from localsetup import LocalSetup
import os

LS = LocalSetup()

def parse_params(param_dict):
    selection_type = int(param_dict['selection_type'])

    def __group_xy(lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])
    box_x_range =  [
                i for i in __group_xy([i for i in param_dict['x_box']])
            ]
    box_y_range = [
        i for i in __group_xy([i for i in param_dict['y_box']])
    ]

    if isinstance(param_dict['gpu'],list):
        gpus = param_dict['gpu']
        gpus = [str(i) for i in gpus]
        gpus = (',').join(gpus)
    else:
        gpus = str(param_dict['gpu'])[:-1]

    def tobool(b):
        return "True" in b

    pers_cards = param_dict['persistence_cardinality']
    params_dict = {
        'selection_type' : int(param_dict['selection_type']),
        'load_preprocessed' : tobool(param_dict['load_preprocessed']),
        'load_preprocessed_walks' : tobool(param_dict['load_preprocessed_walks']),
        'write_msc' : tobool(param_dict['write_msc']),
        'write_growing_boxes' : tobool(param_dict['write_growing_boxes']),
        'write_features' : tobool(param_dict['write_features']),
        'write_partitions': tobool(param_dict['write_partitions']),
        'save_filtered_images' :  tobool(param_dict['save_filtered_images']),
        'load_features': tobool(param_dict['load_features']),
        'write_feature_names' : tobool(param_dict['write_feature_names']),
        'collect_features' : tobool(param_dict['collect_features']),
        'test_param' : tobool(param_dict['test_param']),
        'unsupervised' : tobool(param_dict['unsupervised']),
        'active_learning' : tobool(param_dict['active_learning']),
        'select_label' : tobool(param_dict['select_label']),
        'invert_draw' : tobool(param_dict['invert_draw']),
        'dim_invert' : tobool(param_dict['dim_invert']),
        'union_space' : tobool(param_dict['union_space']),
        'use_ground_truth' : tobool(param_dict['use_ground_truth']),
        'reset_run' : tobool(param_dict['reset_run']),
        'x_box' : box_x_range,
        'y_box' : box_y_range,
        'number_images' : int(param_dict['number_images']),
        'min_number_features' : int(param_dict['min_number_features']),
        'number_features' : int(param_dict['number_features']),
        'persistence_values' : float(param_dict['persistence_values']),
        'persistence_cardinality' : {i:pers_cards[k+1] for k,i in enumerate(pers_cards) if k!=len(pers_cards)-1},
        'pers_train_idx' : int(param_dict['pers_train_idx']),
        'pers_inf_idx' : int(param_dict['pers_inf_idx']),
        'blur' : float(param_dict['blur']),
        'learning_rate' : float(param_dict['learning_rate']),
        'weight_decay' : float(param_dict['weight_decay']),
        'polarity' : int(param_dict['polarity']),
        'epochs' : int(param_dict['epochs']),
        'depth' : int(param_dict['depth']),
        'getognn_class_weights' : tobool(param_dict['getognn_class_weights']),
        'walk_length' : int(param_dict['walk_length']),
        'number_walks' : int(param_dict['number_walks']),
        'random_context' : tobool(param_dict['random_context']),
        'validation_samples' : int(param_dict['validation_samples']),
        'validation_hops' : int(param_dict['validation_hops']),
        'batch_size' : int(param_dict['batch_size']),
        'max_node_degree' : int(param_dict['max_node_degree']),
        'degree_l1' : int(param_dict['degree_l1']),
        'degree_l2' : int(param_dict['degree_l2']),
        'degree_l3' : int(param_dict['degree_l3']),
        'aggregator' : str(param_dict['aggregator'])[:-1],
        'out_dim_1' : int(param_dict['out_dim_1']),
        'out_dim_2' : int(param_dict['out_dim_2']),
        'hidden_dim_1' : int(param_dict['hidden_dim_1']),
        'hidden_dim_2' : int(param_dict['hidden_dim_2']),
        'load_walks' : tobool(param_dict['load_walks']),
        'concat' : tobool(param_dict['concat']),
        'jumping_knowledge' : tobool(param_dict['jumping_knowledge']),
        'jump_type' : str(param_dict['jump_type'])[:-1],
        'forest_depth' : int(param_dict['forest_depth']),
        'number_forests' : int(param_dict['number_forests']),
        'forest_class_weights' : tobool(param_dict['forest_class_weights']),
        'class_1_weight' : float(param_dict['class_1_weight']),
        'class_2_weight' : float(param_dict['class_2_weight']),
        'mlp_lr' : float(param_dict['mlp_lr']),
        'mlp_epochs' : int(param_dict['mlp_epochs']),
        'gpu' : int(gpus),
        'mlp_batch_size' : int(param_dict['mlp_batch_size']),
        'mlp_out_dim_1' : int(param_dict['mlp_out_dim_1']),
        'mlp_out_dim_2' : int(param_dict['mlp_out_dim_2']),
        'mlp_out_dim_3' : int(param_dict['mlp_out_dim_3']),
        'env' : str(param_dict['env'])[:-1],
        'getofeaturegraph_file' : tobool(param_dict['getofeaturegraph_file']),
        'train_data_idx' : str(param_dict['train_data_idx'])[:-1],
        'inference_data_idx' : str(param_dict['inference_data_idx'])[:-1],
        'blur_sigmas' : [int(param_dict['blur_sigmas'])],
        'model_size' : str(param_dict['hidden_dim_1'])[:-1],
        'val_model' : str(param_dict['val_model'])[:-1]
    }
    return params_dict


def set_parameters(x_1 = None, x_2 = None, y_1 = None, y_2 = None,
                   read_params_from = None, growing_windows_from = None, iteration = None,
                   experiment_folder=None):
    if x_1 is not None:#not args.read_param:
        x_1 = x_1#args.x_min
        x_2 = x_2#args.x_max
        y_1 = y_1#args.y_min
        y_2 = y_2#args.y_max
    if read_params_from is not None:
        param_file = os.path.join(LS.project_base_path,'datasets',experiment_folder, 'parameter_list_'+str(read_params_from)+'.txt')
        f = open(param_file, 'r')
        param_dict = {}
        params = f.readlines()
        for param in params:
            name_value = param.split(' ')
            print(name_value)
            if ',' in name_value[1]:
                if name_value[0] not in param_dict.keys():
                    param_dict[name_value[0]] = list( map( int , name_value[1].split(',') ))
                else:
                    param_dict[name_value[0]].extend(list( map(int,name_value[1].split(',')) ))
            else:
                param_dict[name_value[0]] = name_value[1]
    if growing_windows_from is not None:
        box_line_num = int(iteration)
        param_file = os.path.join(LS.project_base_path, 'growing_windows_' + str(growing_windows_from) + '.txt')
        f= open(param_file,'r')
        param_dict = {}
        params = f.readlines()
        for x_y in [box_line_num, box_line_num+1]:
            param = params[x_y]
            name_value = param.split(' ')
            print(name_value)
            if ',' in name_value[1]:
                if name_value[0] not in param_dict.keys():
                    param_dict[name_value[0]] = list(map(int, name_value[1].split(',')))
                else:
                    param_dict[name_value[0]].extend(list(map(int, name_value[1].split(','))))

            else:
                param_dict[name_value[0]] = int(name_value[1])
    return parse_params(param_dict)