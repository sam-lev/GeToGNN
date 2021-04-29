import os
import shutil

class experiment_logger():
    def __init__(self, experiment_folder, input_folder):
        self.experiment_folder = experiment_folder
        self.input_folder = input_folder
        self.dataset_base_path = os.path.basename(experiment_folder)
        self.parameter_list_file = None
        self.window_list_file = None
        self.image_name = None
        self.topo_image_name = None
        self.label_file = None

        self.input_list = [ self.topo_image_name,
                            self.image_name,
                            self.label_file]

    def record_filename(self, **kwargs):
        if 'parameter_list_file' in kwargs.keys():
            self.parameter_list_file = kwargs['parameter_list_file']
        if 'window_list_file' in kwargs.keys():
            self.window_list_file = kwargs['window_list_file']
        if 'label_file' in kwargs.keys():
            self.label_file = kwargs['label_file']
            self.label_file = os.path.split(self.label_file)[1]
        if 'image_name' in kwargs.keys():
            self.image_name = kwargs['image_name']
            self.image_name = os.path.split(self.image_name)[1]
        if 'topo_image_name' in kwargs.keys():
            self.topo_image_name = kwargs['topo_image_name']
            self.topo_image_name = os.path.split(self.topo_image_name)[1]

        self.input_list = [self.topo_image_name,
                           self.image_name,
                           self.label_file]

    def write_experiment_info(self):

        def write_input_info():
            description_file = os.path.join(self.input_folder, 'description.txt')
            print("... Writing bounds file to:", description_file)
            description_file = open(description_file, "w+")
            for fname in self.input_list[:-1]:
                description_file.write(fname + '\n')
            description_file.write(str(self.input_list[-1]))
            description_file.close()

        def write_experiment_info():
            shutil.copy(self.parameter_list_file, self.experiment_folder)
            #shutil.copy(self.window_list_file, self.experiment_folder)

        write_input_info()
        write_experiment_info()
