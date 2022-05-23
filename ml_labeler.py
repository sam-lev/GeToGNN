import subprocess
import os
from os.path import basename
import re
import shutil
import subprocess
import sys
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from shutil import copyfile
import matplotlib.pyplot as plt
import glob
NUM_PROCESSES = 8
import statistics
import tkinter as tk
from tkinter import filedialog
import json

from localsetup import LocalSetup
LocalSetup = LocalSetup()
# the tools that are used
ridgegraph_tool_exe = os.path.join(LocalSetup.complex_compute_path,'build','extract2dridgegraph','extract2dridgegraph') #'.exe?
#regiongraph_tool_exe = "C:/Users/jediati/Desktop/JEDIATI/builds/test_GradIntegrator/extract2dregiongraph/Release/extract2dregiongraph.exe"
labeler_tool_exe = os.path.join(LocalSetup.topo_label_path,'build','polyline_labeler','PolylineLabeler')
#reglabeler_tool_exe = "C:/Users/jediati/Desktop/JEDIATI/builds/TopoViewerFLTK/region_labeler/Release/RegionLabeler.exe"

polyline_explorer_tool_exe = os.path.join(LocalSetup.topo_label_path,'build','polyline_explorer','PolylineExplorer.exe')
polyline_multirun_explorer_tool_exe = os.path.join(LocalSetup.topo_label_path,'build','polyline_multirun_explorer','PolylineMultirunExplorer.exe')

dir_path = os.path.dirname(os.path.realpath(__file__))
color_file = os.path.join(dir_path, "colors.txt")
HACK_output_file = os.path.join(os.path.realpath(__file__), "GNNvis/GNNvis/data/labels.txt")

#filename_raw = "C://Users//jediati//Desktop//JEDIATI//data//foam//dfdata//DFdata_1427_810_812.raw"
#filename_raw = "C://Users/jediati//Desktop//JEDIATI/data//foam//dfdata//test_DFDATA_240_162_301.raw"
#filename_raw = "C://Users/jediati//Desktop//JEDIATI/data//foam//synth//stitched_351_396_295.raw"
#filename_raw = "C://Users/jediati//Desktop//JEDIATI/data//foam//synth//stitched_176_198_148.raw"
#filename_raw = "C://Users/jediati//Desktop//JEDIATI/data//foam//synth//Quinn_Foam_244_233_202.raw"
#filename_raw = "C://Users/jediati//Desktop//JEDIATI/data//foam//dfdata//Kris_Foam_424_424_504.raw"
#filename_raw = 'C:/Users/jediati/Desktop/JEDIATI/data/lung_brian/Stroma/Stroma_s8_t1699_1125_512_300.raw'
filename_raw = "C:/Users/jediati/Desktop/JEDIATI/data/2d_ml_inputs/retinal/old/optic1.tif_invrt_smoothed_565_584.raw"
filename_raw = "C:/Users/jediati/Desktop/JEDIATI/data/2d_ml_inputs/mat2/sub_CMC_example_l1_969_843.raw"

filename_render = "C:/Users/jediati/Desktop/JEDIATI/data/2d_ml_inputs/mat2/sub_CMC_example_o_969_843.raw"

run_num = 0

def deduce_dimensions(filename) :
    dirname, filename = os.path.split(filename)
    print("deducing dimensions from:", filename)
    filenamebase = os.path.splitext(filename)[0]
    # check if last token has "x" in it
    last_token = filenamebase.split("_")[-1]
    if "x" in last_token :
        # the dimensions are formatted as XxY
        dim_string_list = last_token.split("x")
    else :
        dim_string_list = filenamebase.split("_")[-2:]
    return dim_string_list



def do_run(force_recompute_graph) :
    global run_num
    print("=================> run num: ", run_num, " <=============================")
    run_num += 1

    pers_threshold = pers_string.get()

    cuts_values = cuts_string.get()
    cuts_list = []
    if cuts_values:
        cuts_list = re.split('; |, ', cuts_values)
    print("cuts at:", cuts_list)

    do_line_network = False
    do_region_graph = False
    if compute_type.get() == "1":
        use_valleys = False
        do_line_network = True
    elif compute_type.get() == "2":
        use_valleys = True
        do_line_network = True
    elif compute_type.get() == "3":
        use_valleys = False
        do_line_network = False
    elif compute_type.get() == "4":
        use_valleys = True
        do_line_network = False
    else:
        print("TYpe is not supported yet", compute_type.get())
        exit()

    filename_raw = topo_func_name.get()
    filename_render = render_func_name.get()
    X, Y = deduce_dimensions(filename_raw)
    print(X, Y)

    dirname, filename = os.path.split(filename_raw)
    print(dirname, filename)
    filenamebase = os.path.splitext(filename)[0]
    intermediate_dir = dirname + "/" + filenamebase + "_dir"
    print("placing results in:", intermediate_dir)
    if not os.path.isdir(intermediate_dir):
        os.mkdir(intermediate_dir)

    filename_nodes = filename_raw + ".mlg_nodes.txt"
    filename_arcs = filename_raw + ".mlg_edges.txt"
    filename_geoms = filename_raw + ".mlg_geom.txt"

    def MyBlockingRun(arg_list):
        print("command line:", arg_list)
        proc = subprocess.Popen(arg_list)
        proc.wait()

    if force_recompute_graph or not os.path.isfile(filename_nodes) or not os.path.isfile(filename_arcs) or not os.path.isfile(filename_geoms):
        # do recompute
        if do_line_network :
            args_list = [ridgegraph_tool_exe, filename_raw, X, Y, pers_threshold, str(int(use_valleys))] + cuts_list
        #else:
        #    args_list = [regiongraph_tool_exe, filename_raw, X, Y, pers_threshold, str(int(use_valleys))] + cuts_list

        MyBlockingRun(args_list)

    if do_line_network:
        label_args = [labeler_tool_exe, X, Y, filename_raw, filename_render]
    #else:
    #    label_args = [reglabeler_tool_exe, X, Y, filename_raw, filename_render]

    print("running tool:", label_args)
    proc = subprocess.Popen(label_args)


def run_labeler() :
    do_run(False)

def run_all():
    do_run(True)

def run_explorer() :

    # make a config string list to hold the result
    config_string_list = []

    # run directory is the experiment name inside the folder structure
    experiment_dir = os.path.normpath(tk.filedialog.askdirectory())
    if experiment_dir is None:
        print("no directory selected")
        return
    print("Using experiment: ", experiment_dir)

    # get dataset directory
    dataset_dir = os.path.dirname(experiment_dir)
    print("Using datadir:", dataset_dir)

    # get input directory
    input_dir = os.path.join(dataset_dir, "input")
    if not os.path.isdir(input_dir):
        print("No input dir found, exiting:", input_dir)
        return
    print("Using input: ", input_dir)

    #find description.txt to figure out what files to use
    desc_file_name = os.path.join(input_dir, "description.txt")
    if not os.path.isfile(desc_file_name) :
        print("Description file not found, exiting:", desc_file_name)
        return
    print("Using description file:", desc_file_name)

    # read description file
    with open(desc_file_name) as desc_file:
        all_file_names = desc_file.read().splitlines()

    #set topo and original file anmes
    topo_file_name = os.path.join(input_dir, all_file_names[0])
    original_file_name = os.path.join(input_dir, all_file_names[1])
    label_file_name = os.path.join(input_dir, all_file_names[2])
    if not os.path.isfile(topo_file_name) :
        print("Topofile not found, exiting:", topo_file_name)
        return
    print("Using Topofile:", topo_file_name)
    filename_nodes = topo_file_name + ".mlg_nodes.txt"
    filename_arcs = topo_file_name + ".mlg_edges.txt"
    filename_geoms = topo_file_name + ".mlg_geom.txt"
    for filename in [filename_nodes, filename_arcs, filename_geoms]:
        if not os.path.isfile(filename):
            print("Did not find graph file, exiting:", filename)
            return
        print("Graph files:", filename)
    if not os.path.isfile(original_file_name) :
        print("Originalfile not found, exiting:", original_file_name)
    print("Using original file:", original_file_name)
    if not os.path.isfile(label_file_name) :
        print("labelfile not found, exiting:", label_file_name)
    print("Using ground truth labeling file:", label_file_name)

    # ok now we have all the inputs gathered
    # deduce the size of the image from the name
    X, Y = deduce_dimensions(topo_file_name)
    print("Using x,y dimensions:", X, Y)


    # ADD TO CONFIG FILE LIST
    config_string_list.append(X + " " + Y)
    config_string_list.append(topo_file_name)
    config_string_list.append(original_file_name)
    config_string_list.append(label_file_name)
    config_string_list.append(filename_nodes)
    config_string_list.append(filename_arcs)
    config_string_list.append(filename_geoms)

    # gather additional filenames
    other_file_names_list = []
    for filename in  all_file_names[3:]:
        filename_join = os.path.join(input_dir, filename)
        if not os.path.isfile(filename_join):
            print("Did not find additional file, ignoring:", filename_join)
        else:
            other_file_names_list.append(filename_join)

    # search for run directories
    run_base_dir = os.path.join(experiment_dir, "runs")
    if not os.path.isdir(run_base_dir):
        print("No runs dir found, exiting:", run_base_dir)
        return

    # find all the runs
    run_dir_list = os.listdir(run_base_dir)
    print("found runs:")

    # how many runs there will be:
    config_string_list.append(str(len(run_dir_list)))

    for d in run_dir_list:
        dirname = os.path.join(run_base_dir, d)
        print(" --", dirname)
        try:
            partitions_file = os.path.normpath(glob.glob(dirname + "//*partitions.txt")[0])
            preds_file = os.path.normpath(glob.glob(dirname + "//*preds.txt")[0])
            window_file = os.path.normpath(glob.glob(dirname + "//*window.txt")[0])
        except IndexError:
            print("error in:", dirname, "skipping")
            continue
        with open(window_file) as w:
            contents = w.readlines()
            stripped = [s.replace("[","").replace("]","").replace("(","").replace(")","").replace(",","").split()[-2:] for s in contents]
            flat_list = sum(stripped, [])
            box_string = " ".join(flat_list)
        print("    -- box:", box_string)
        print("    -- predfile:", preds_file)
        print("    -- partitionfile:", partitions_file)
        config_string_list.append(box_string)
        config_string_list.append(preds_file)
        config_string_list.append(partitions_file)

    config_file_name = os.path.normpath(os.path.join(experiment_dir, "__config.txt"))
    with open(config_file_name, 'w') as outfile:
        outfile.writelines("%s\n" % line for line in config_string_list)

    # now launch the tool!
    label_args = [polyline_explorer_tool_exe, config_file_name]
    print("running tool:", label_args)
    proc = subprocess.Popen(label_args)
    print("got here")


def run_multirun_explorer() :

    config_files = []

    # run directory is the experiment name inside the folder structure
    multi_dirs = []
    while True:
        dir = tk.filedialog.askdirectory()
        if not dir:
            break
        print("got", dir)
        multi_dirs.append(dir)

    experiment_dirs = [os.path.normpath(x) for x in multi_dirs]
    for experiment_dir in experiment_dirs :
        if experiment_dir is None:
            print("no directory selected")
            return
        print("Using experiment: ", experiment_dir)

        # make a config string list to hold the result
        config_string_list = []

        # get dataset directory
        dataset_dir = os.path.dirname(experiment_dir)
        print("Using datadir:", dataset_dir)

        # get input directory
        input_dir = os.path.join(dataset_dir, "input")
        if not os.path.isdir(input_dir):
            print("No input dir found, exiting:", input_dir)
            return
        print("Using input: ", input_dir)

        #find description.txt to figure out what files to use
        desc_file_name = os.path.join(input_dir, "description.txt")
        if not os.path.isfile(desc_file_name) :
            print("Description file not found, exiting:", desc_file_name)
            return
        print("Using description file:", desc_file_name)

        # read description file
        with open(desc_file_name) as desc_file:
            all_file_names = desc_file.read().splitlines()

        #set topo and original file anmes
        topo_file_name = os.path.join(input_dir, all_file_names[0])
        original_file_name = os.path.join(input_dir, all_file_names[1])
        label_file_name = os.path.join(input_dir, all_file_names[2])
        if not os.path.isfile(topo_file_name) :
            print("Topofile not found, exiting:", topo_file_name)
            return
        print("Using Topofile:", topo_file_name)
        filename_nodes = topo_file_name + ".mlg_nodes.txt"
        filename_arcs = topo_file_name + ".mlg_edges.txt"
        filename_geoms = topo_file_name + ".mlg_geom.txt"
        for filename in [filename_nodes, filename_arcs, filename_geoms]:
            if not os.path.isfile(filename):
                print("Did not find graph file, exiting:", filename)
                return
            print("Graph files:", filename)
        if not os.path.isfile(original_file_name) :
            print("Originalfile not found, exiting:", original_file_name)
        print("Using original file:", original_file_name)
        if not os.path.isfile(label_file_name) :
            print("labelfile not found, exiting:", label_file_name)
        print("Using ground truth labeling file:", label_file_name)

        # ok now we have all the inputs gathered
        # deduce the size of the image from the name
        X, Y = deduce_dimensions(topo_file_name)
        print("Using x,y dimensions:", X, Y)


        # ADD TO CONFIG FILE LIST
        config_string_list.append(X + " " + Y)
        config_string_list.append(topo_file_name)
        config_string_list.append(original_file_name)
        config_string_list.append(label_file_name)
        config_string_list.append(filename_nodes)
        config_string_list.append(filename_arcs)
        config_string_list.append(filename_geoms)

        # gather additional filenames
        other_file_names_list = []
        for filename in  all_file_names[3:]:
            filename_join = os.path.join(input_dir, filename)
            if not os.path.isfile(filename_join):
                print("Did not find additional file, ignoring:", filename_join)
            else:
                other_file_names_list.append(filename_join)

        # search for run directories
        run_base_dir = os.path.join(experiment_dir, "runs")
        if not os.path.isdir(run_base_dir):
            print("No runs dir found, exiting:", run_base_dir)
            return

        # find all the runs
        run_dir_list = os.listdir(run_base_dir)
        print("found runs:")

        # how many runs there will be:
        config_string_list.append(str(len(run_dir_list)))

        for d in run_dir_list:
            dirname = os.path.join(run_base_dir, d)
            print(" --", dirname)
            try:
                partitions_file = os.path.normpath(glob.glob(dirname + "//*partitions.txt")[0])
                preds_file = os.path.normpath(glob.glob(dirname + "//*preds.txt")[0])
                window_file = os.path.normpath(glob.glob(dirname + "//*window.txt")[0])
            except IndexError:
                print("error in:", dirname, "skipping")
                continue
            with open(window_file) as w:
                contents = w.readlines()
                stripped = [s.replace("[","").replace("]","").replace("(","").replace(")","").replace(",","").split()[-2:] for s in contents]
                flat_list = sum(stripped, [])
                box_string = " ".join(flat_list)
            print("    -- box:", box_string)
            print("    -- predfile:", preds_file)
            print("    -- partitionfile:", partitions_file)
            config_string_list.append(box_string)
            config_string_list.append(preds_file)
            config_string_list.append(partitions_file)

        config_file_name = os.path.normpath(os.path.join(experiment_dir, "__config.txt"))
        with open(config_file_name, 'w') as outfile:
            outfile.writelines("%s\n" % line for line in config_string_list)
        config_files.append(config_file_name)

    # now launch the tool!
    label_args = [polyline_multirun_explorer_tool_exe, *config_files]
    print("running tool:", label_args)
    proc = subprocess.Popen(label_args)
    print("got here")


def do_topo_func_name():
    filename_raw = tk.filedialog.askopenfilename()
    topo_func_name.set(filename_raw)
    render_func_name.set(filename_raw)

def do_render_func_name():
    filename_render = tk.filedialog.askopenfilename()
    render_func_name.set(filename_render)

root = tk.Tk()
topo_func_name = tk.StringVar(root)
topo_func_name.set('None')
render_func_name = tk.StringVar(root)
render_func_name.set('None')

button_pick_topo_func = tk.Button(root, text="pick topo function", command = do_topo_func_name)
button_pick_topo_func.pack(side=tk.TOP, fill=tk.BOTH)
label_topo_function = tk.Label(textvariable = topo_func_name)
label_topo_function.pack(side=tk.TOP, fill=tk.BOTH)

button_pick_render_func = tk.Button(root, text="pick render function", command = do_render_func_name)
button_pick_render_func.pack(side=tk.TOP, fill=tk.BOTH)
label_topo_function = tk.Label(textvariable = render_func_name)
label_topo_function.pack(side=tk.TOP, fill=tk.BOTH)
# Tkinter string variable
# able to store any string value
compute_type = tk.StringVar(root, "1")

# Dictionary to create multiple buttons
values = {"Ridge Lines": "1",
          "Valley Lines": "2",
          "Mountains": "3",
          "Basins": "4"}

# Loop is used to create multiple Radiobuttons
# rather than creating each button separately
for (text, value) in values.items():
    tk.Radiobutton(root, text=text, variable=compute_type,
                   value=value).pack(side=tk.TOP, fill=tk.BOTH, ipady=5)
label_persistence = tk.Label(root, text= "Persistence Threshold")
label_persistence.pack(side=tk.TOP, fill=tk.BOTH)
pers_string = tk.StringVar(root, "0.01")
entry_persistence = tk.Entry(root, textvariable=pers_string)
entry_persistence.pack(side=tk.TOP, fill= tk.BOTH)

label_cuts = tk.Label(root, text= "Cut Value(s) for lines?")
label_cuts.pack(side=tk.TOP, fill=tk.BOTH)
cuts_string = tk.StringVar(root, "")
entry_cuts = tk.Entry(root, textvariable=cuts_string)
entry_cuts.pack(side=tk.TOP, fill= tk.BOTH)

button_run = tk.Button(root, text="Run Graph Compute And Labeler", command=run_all)
button_run.pack(side = tk.TOP, fill= tk.BOTH)

button_run2 = tk.Button(root, text="Run Labeler Only", command=run_labeler)
button_run2.pack(side = tk.TOP, fill= tk.BOTH)

button_run3 = tk.Button(root, text="Run Results Explorer", command=run_explorer)
button_run3.pack(side = tk.TOP, fill= tk.BOTH)


button_run4 = tk.Button(root, text="Run Multirun Results Explorer", command=run_multirun_explorer)
button_run4.pack(side = tk.TOP, fill= tk.BOTH)


root.mainloop()




