import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from localsetup import LocalSetup

LS = LocalSetup()

#
# params:
#    model "_f1", "_precision", "_recall"
#
#
#
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help="Name to prepend on model during training")
parser.add_argument('--data_folder', type=str, default='retinal', help='specific dataset in datasets folder')
parser.add_argument('--exp_folder', type=str, default='exp1', help='experiment folder name with runs')
parser.add_argument('--metric', type=str,default='f1', help='recall, precision, F1')
parser.add_argument('--bins', type=int, help='fraction of bins from total number runs for histograms')
parser.add_argument('--runs', type=str, default='runs', help='optional, specific run folder if dif than default runs')
parser.add_argument('--plt_title', type=str, default=None, help='optional, specific run folder if dif than default runs')

args = parser.parse_args()

exp_path = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
                                      str(args.exp_folder))


def collect_run_metrics(metric = None):
    runs_folder = os.listdir(os.path.join(LS.project_base_path, 'datasets', args.data_folder,
                              args.exp_folder, args.runs))
    print(runs_folder)
    results_dict = {}
    results_dict['weighted'] = []
    if metric != 'precision' and metric != 'recall':
        results_dict['binary'] = []
    elif metric == 'precision':
        results_dict['average'] = []
    results_dict['micro'] = []
    results_dict['macro'] = []
    attributes_dict = {}
    attributes_dict['model'] = args.model
    attributes_dict['metric'] = args.metric
    attributes_dict['count'] = len(runs_folder)
    attributes_dict['bins'] = args.bins

    for run in runs_folder:
        #for metric in metrics:
        metric_file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
                                  str(args.exp_folder), args.runs, str(run),
                                  str(args.model)+'_'+str(metric)+'.txt')
        f = open(metric_file, 'r')
        result_lines = f.readlines()
        f.close()
        #for metric_subtype in results_dict.keys():
        for result in result_lines:
            name_value = result.split(' ')
            avg_type_name = name_value[0]
            if metric == 'recall' and avg_type_name == 'binary':
                continue
            else:
                val = float(name_value[1])
            results_dict[str(avg_type_name)].append(val)

    return results_dict, attributes_dict


def plot_histogram(x, y, n_bins, metric_name='', plt_title = '', write_folder=None):
    if args.plt_title is not None:
        plt_title = args.plt_title +' '+ plt_title

    fig, axs = plt.subplots(1, 2, tight_layout=False)



    num_per_bin, bins_ranges = np.histogram(y, bins=n_bins)
    x = [bins_ranges[len(bins_ranges[bins_ranges < val]) - 1] for val in y]


    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs[0].hist(y, bins=n_bins)
    # add binned points
    axs[0].scatter(x, y, zorder=100)
    # calculate bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # draw errobars, use the sqrt error. You can use what you want there
    # poissonian 1 sigma intervals would make more sense
    axs[0].errorbar(bin_centers, N, yerr=np.sqrt(N), fmt='r.')



    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

        #add perecentage
        #x_p = thispatch.get_x() + thispatch.get_width() / 2
        #y_p = thispatch.get_height() + .05
        #plt.annotate('{:.1f}%'.format(y_p), (x_p, y_p), ha='center')

    # We can also normalize our inputs by the total number of counts
    axs[1].hist(y, bins=n_bins)#, density=True)
    #axs[1].scatter(x,y,zorder=100)#hist(y, bins=n_bins, density=True)
    axs[1].errorbar(bin_centers, N, yerr=np.sqrt(N), fmt='r.')

    # Now we format the y-axis to display percentage
    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=len(y)))



    plt.xlabel(metric_name)
    plt.ylabel('count')
    plt.title(plt_title)

    if write_folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(write_folder,plt_title+'.png'))

def plot_2d_hist(x, y):
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(x, y)

def compute_metric_histogram(results_dict=None,
                             attributes_dict=None,
                             metric=None,
                             write_folder=None):
    bins     = attributes_dict['bins']
    count = attributes_dict['count']

    names = results_dict.keys()
    metric_results = map( lambda res_list : np.array(res_list) , results_dict.values())
    for name, result in zip(names, metric_results):
        plot_histogram(x=count, y=result, n_bins=bins, metric_name=metric,
                       plt_title=args.model + ' ' + name + ' ' + metric,
                       write_folder = write_folder)




def main():
    metric_plot_folder = os.path.join(str(exp_path),'metric_plots')
    if not os.path.exists(metric_plot_folder):
        os.makedirs(metric_plot_folder)
    metrics = ['precision', 'recall', 'f1']
    for metric in metrics:
        metric_subfolder = os.path.join(metric_plot_folder,metric)
        if not os.path.exists(metric_subfolder):
            os.makedirs(metric_subfolder)
        results_dict, attributes_dict = collect_run_metrics(metric=metric)
        compute_metric_histogram(results_dict=results_dict,
                                 attributes_dict=attributes_dict,
                                 metric=metric,
                                 write_folder = metric_subfolder)
def compute_metric_plots(model, data_folder, exp_folder, metric, bins, runs, plt_title):

    args.model = model
    args.data_fodler = data_folder
    args.exp_folder = exp_folder
    args.metric = metric
    args.bins = bins
    args.runs = runs
    args.plt_title = plt_title

    metric_plot_folder = os.path.join(str(exp_path), 'metric_plots')
    if not os.path.exists(metric_plot_folder):
        os.makedirs(metric_plot_folder)
    metrics = ['precision', 'recall', 'f1']
    for metric in metrics:
        metric_subfolder = os.path.join(metric_plot_folder, metric)
        if not os.path.exists(metric_subfolder):
            os.makedirs(metric_subfolder)
        results_dict, attributes_dict = collect_run_metrics(metric=metric)
        compute_metric_histogram(results_dict=results_dict,
                                 attributes_dict=attributes_dict,
                                 metric=metric,
                                 write_folder=metric_subfolder)

if __name__ == "__main__":
    main()
