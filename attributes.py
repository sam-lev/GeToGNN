import os
import numpy as np
from data_ops.set_params import set_parameters
from localsetup import LocalSetup

class Attributes(object):
    def __init__(self, gid_gnode_dict=None, gid_edge_dict=None,
                 write_folder=None, parameter_file_number= None, **kwargs):



        self.gid_gnode_dict = {}
        self.gid_edge_dict = {}
        self.node_gid_to_label = {}
        self.node_gid_to_partition = {}
        self.node_gid_to_feature = {}
        self.node_gid_to_feat_idx = {}
        self.node_gid_to_graph_idx = {}
        self.node_gid_to_prediction = {}


        self.getoelms = None
        self.lin_adj_idx_to_getoelm_idx = None
        self.gid_to_getoelm_idx = {}
        self.gid_geto_elm_dict = {}
        self.graph_idx_to_gid = {}

        self.G = None
        #self.G_dict
        #self.select_points / points
        #self.select_key_map / key_map
        self.fname_to_featidx = {}
        self.feature_names = []
        self.idx_to_feat = {}

        #
        # Write Paths
        #
        #self.run_num = 0
        self.LocalSetup = LocalSetup()
        print("KWARG")
        print(kwargs)
        self.params = {}


        self.model_name = None
        self.data_name = None
        self.pred_run_path = None
        self.experiment_folder = None
        self.input_folder = None
        self.segmentation_path = None
        self.msc_write_path = None
        self.session_name = None
        self.pred_session_run_path = None
        msc_info = None
        self.image = None
        self.X = None
        self.Y = None
        self.run_name=''

        if parameter_file_number is not None:
            print("setting params")
            self.params = set_parameters(read_params_from=parameter_file_number,
                                         experiment_folder=write_folder)

        #else:
        #    self.params = kwargs['params']
        '''for param in kwargs:
            self.params[param] = kwargs[param]
        #for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]
        if 'params' in kwargs.keys():
            param_add_ons = kwargs['params']
            for k, v in param_add_ons.items():
                self.params[k] = v'''

    def update_run_info(self, write_folder=None, batch_multi_run=None):

        #self.params['write_folder'] = new # experiment
        if write_folder is not None:
            self.params['write_folder'] = os.path.join(self.data_name, write_folder)
        self.experiment_folder = os.path.join(self.LocalSetup.project_base_path, 'datasets'
                                              ,self.params['write_folder'])
        self.input_folder = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                         self.data_name, 'input')
        self.params['experiment_folder'] = self.experiment_folder
        self.params['input_folder'] = self.input_folder

        self.pred_run_path = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                      self.params['write_folder'],
                                      'runs')#+self.run_name)
        if not os.path.exists(self.pred_run_path):
            os.makedirs(os.path.join(self.pred_run_path))

        self.experiment_folder = os.path.join(self.LocalSetup.project_base_path,
                                              'datasets',
                                              self.params['write_folder'])
        if batch_multi_run is None:
            self.session_name = str(self.run_num)
            self.pred_session_run_path = os.path.join(self.pred_run_path,
                                                      self.session_name)
            if not os.path.exists(self.pred_session_run_path):
                os.makedirs(self.pred_session_run_path)

        if batch_multi_run is not None:
            self.pred_run_path = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                      self.params['write_folder'],
                                      'runs')

            #if not os.path.exists(self.pred_run_path):
            #    os.makedirs(os.path.join(self.pred_run_path))

            self.session_name = str(self.run_num)
            self.pred_session_run_path = os.path.join(self.pred_run_path, str(batch_multi_run))
            if not os.path.exists(self.pred_session_run_path):
                os.makedirs(self.pred_session_run_path)




    def get_attributes(self):
        return self