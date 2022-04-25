import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from decimal import getcontext, Decimal

from localsetup import LocalSetup

LS = LocalSetup()

#
# params:
#    model "_f1", "_precision", "_recall"
#
# Example Run:
#
#     python compute_multirun_metrics.py --metric f1 --bins 7
#     --title '' --batch True --data_dir diadem_sub1 --exp_dir diadem_sub1/GNN
#     --model getognn --runs runs
#
# Example 2:
# python compute_multirun_metrics.py  --model getognn --data_folder retinal
#   --exp_folder fourth_windows_graphsage-meanpool --bins 15 --runs runs
#   --plt_title graphsage_meanpool
#
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help="Name to prepend on model during training")
parser.add_argument('--data_dir', type=str, default='retinal', help='specific dataset in datasets folder')
parser.add_argument('--exp_dir', type=str, default='exp1', help='experiment folder name with runs')
parser.add_argument('--metric', type=str, default='f1', help='recall, precision, F1')
parser.add_argument('--bins', type=int, help='fraction of bins from total number runs for histograms')
parser.add_argument('--runs', type=str, default='runs', help='optional, specific run folder if dif than default runs')
parser.add_argument('--title', type=str, default=None, help='optional, specific run folder if dif than default runs')
parser.add_argument('--batch', type=bool, default=True, help='optional, specific run folder if dif than default runs')
parser.add_argument('--exp_folder', type=str, default='exp1', help='experiment folder name with runs')
parser.add_argument('--data_folder', type=str, default='exp1', help='experiment folder name with runs')
args = parser.parse_args()

exp_path = None


def group_pairs(lst):
    for i in range(0, len(lst), 2):
        yield tuple(lst[i: i + 2])


def collect_run_metrics(run_path=None, metric=None, runs_folder=None,
                        batch_of_batch=False, avg_multi=False,
                        window_file='window.txt', batch_multi_run=False, **kwargs):
    results_dict = {}
    results_dict['weighted'] = []
    if metric != 'precision' and metric != 'recall':
        results_dict['binary'] = []
    elif metric == 'precision':
        results_dict['average'] = []
    results_dict['micro'] = []
    results_dict['macro'] = []
    attributes_dict = {}
    attributes_dict['model'] = kwargs['model']
    attributes_dict['metric'] = metric
    attributes_dict['count'] = len(runs_folder)
    attributes_dict['bins'] = kwargs['bins']

    model = kwargs['model']

    run_region_dict = {}
    run_region_percent = {}
    run_result_dict = {}
    idx_run_dict = {}
    run_region_dict['X'] = None  # first line window.txt
    run_region_dict['Y'] = None  # second
    # run_region_dict['X_BOX'] = [] #...
    # run_region_dict['Y_BOX'] = []

    runs_folder = list(map(int, runs_folder))
    runs_folder.sort()
    for idx, run in enumerate(sorted(runs_folder)):
        # for metric in metrics:
        if metric != 'time':
            metric_file = os.path.join(run_path, str(run), str(metric) + '.txt')
        else:
            metric_file = os.path.join(run_path, str(run), 'train_'+str(metric) + '.txt')
        # if batch_of_batch:
        #    metric_file = os.path.join(run_path, str(run),str(metric), 'metric_averages' + '.txt')
        f = open(metric_file, 'r')
        result_lines = f.readlines()
        f.close()
        # for metric_subtype in results_dict.keys():
        if metric != 'time':
            micro = 0
            macro = 0
            for result in result_lines:
                name_value = result.split(' ')
                avg_type_name = name_value[0]
                if '_' in avg_type_name:
                    avg_type_name = avg_type_name.split('_')[0]
                if metric == 'recall' and 'binary' in avg_type_name:
                    continue
                else:
                    val = float(name_value[1])
                if str(avg_type_name) in results_dict.keys():
                    results_dict[str(avg_type_name)].append(val)
                else:
                    results_dict[str(avg_type_name)] = [val]

                if 'f1' in metric:
                    if 'micro' in avg_type_name:
                        if run not in run_result_dict.keys():
                            run_result_dict[run] = {'micro': val}
                        else:
                            run_result_dict[run]['micro'] = val
                    if 'macro' in avg_type_name:
                        if run not in run_result_dict.keys():
                            run_result_dict[run] = {'macro': val}
                        else:
                            run_result_dict[run]['macro'] = val
                    if 'class' in avg_type_name:
                        if run not in run_result_dict.keys():
                            run_result_dict[run] = {'class': val}
                        else:
                            run_result_dict[run]['class'] = val
                    # if  'weighted' in avg_type_name:
                    #     if run not in run_result_dict.keys():
                    #         run_result_dict[run] = {'weighted': val}
                    #     else:
                    #         run_result_dict[run]['weighted'] = val
        if metric == 'f1':
            # if batch_multi_run:
            window_f = _file = os.path.join(run_path, str(run), "region_percents.txt")
            # else:
            #    window_f = _file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
            #                                    str(args.exp_folder),  str(run),str(run), "region_percents.txt")
            f = open(window_f, 'r')
            lines = f.readlines()
            f.close()
            region_percentage = float(lines[-1].split(' ')[1])
            run_region_percent[run] = region_percentage
        if metric == 'f1':
            if batch_multi_run:
                window_f = _file = os.path.join(run_path, str(run), window_file)
            else:
                window_f = _file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
                                                str(args.exp_folder), str(run), str(run), window_file)
            f = open(window_f, 'r')
            window_lines = f.readlines()
            f.close()

            run_region_dict['X'] = float(window_lines[0])  # first line window.txt
            run_region_dict['Y'] = float(window_lines[1])

            boxes = [
                i for i in group_pairs(window_lines[2:])
            ]
            boxes = [y for x in boxes for y in x]
            current_box_dict = {}

            x_box_set = []  # list([None for run in range(len(runs_folder))])
            y_box_set = []  # list([list([]) for run in range(len(runs_folder))])

            for bounds in boxes:
                name_value = bounds.split(' ')

                # window selection(s)
                # for single training box window

                if name_value[0] == 'x_box':  # not in current_box_dict.keys():
                    # if len(x_box_set) <:
                    #    x_box_set =list(map(float,
                    #                          name_value[1].split(',')))
                    # else:
                    x_box_set.append(tuple(map(float,
                                               name_value[1].split(','))))
                if name_value[0] == 'y_box':  # not in current_box_dict.keys():
                    # if len(y_box_set[run-1]) == 0:
                    #     y_box_set = list(map(float,
                    #                          name_value[1].split(',')))
                    # else:
                    y_box_set.append(tuple(map(float,
                                               name_value[1].split(','))))
                    # box_set = set()
                    # current_box_dict[name_value[0]] = box_set.add(tuple(map(float,
                    #                                                        name_value[1].split(','))))
                # # for multiple boxes
                # else:
                #     current_box_dict[name_value[0]].add(tuple(map(float, name_value[1].split(','))))

            # X_BOX = [
            #     i for i in group_pairs([i for i in current_box_dict['x_box']])
            # ]
            # Y_BOX = [
            #     i for i in group_pairs([i for i in current_box_dict['y_box']])
            # ]
            def elim_ordered_duplicates(seq):
                seen = set()
                seen_add = seen.add
                return [x for x in seq if not (x in seen or seen_add(x))]

            x_box_set = elim_ordered_duplicates(x_box_set)
            y_box_set = elim_ordered_duplicates(y_box_set)
            run_region_dict[run] = [x_box_set, y_box_set]  # (X_BOX, Y_BOX)
            # idx += 1
        if metric == 'time':
            # if batch_multi_run:
            f1_f = _file = os.path.join(run_path, str(run), "f1.txt")
            # else:
            #    window_f = _file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
            #                                    str(args.exp_folder),  str(run),str(run), "region_percents.txt")
            f = open(f1_f, 'r')
            lines = f.readlines()
            f.close()
            f1 = float(lines[-1].split(' ')[1])

            train_f = os.path.join(run_path, str(run), "train_time.txt")
            # else:
            #    window_f = _file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
            #                                    str(args.exp_folder),  str(run),str(run), "region_percents.txt")
            t_train = open(train_f, 'r')
            lines = t_train.readlines()
            t_train.close()
            train_t = float(lines[0])  # [-1].split(' ')[1])

            pred_f = _file = os.path.join(run_path, str(run), "pred_time.txt")
            # else:
            #    window_f = _file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
            #                                    str(args.exp_folder),  str(run),str(run), "region_percents.txt")
            t_pred = open(pred_f, 'r')
            lines = t_pred.readlines()
            t_pred.close()
            pred_t = float(lines[0])  # [-1].split(' ')[1])

            run_region_percent[run] = (train_t, pred_t, f1)

    return results_dict, attributes_dict, run_region_dict, run_result_dict, run_region_percent


def plot_histogram(x, y, n_bins, metric_name='', plt_title='',
                   write_folder=None, exp_name=''):
    plt_title = plt_title + "_" + exp_name.split('/')[-1]

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
    fracs = N  # / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

        # add perecentage
        # x_p = thispatch.get_x() + thispatch.get_width() / 2
        # y_p = thispatch.get_height() + .05
        # plt.annotate('{:.1f}%'.format(y_p), (x_p, y_p), ha='center')

    # We can also normalize our inputs by the total number of counts
    axs[1].hist(y, bins=n_bins)  # , density=True)
    # axs[1].scatter(x,y,zorder=100)#hist(y, bins=n_bins, density=True)
    axs[1].errorbar(bin_centers, N, yerr=np.sqrt(N), fmt='r.')

    # Now we format the y-axis to display percentage
    axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=len(y)))

    plt.xlabel(metric_name)
    plt.ylabel('count')
    plt.title(plt_title)

    print("    * writing to: ", write_folder)
    if write_folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(write_folder, plt_title + '.png'))
    plt.close(fig)


def plot_2d_hist(x, y):
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(x, y)


def compute_metric_histogram(results_dict=None,
                             attributes_dict=None,
                             metric=None,
                             write_folder=None,
                             exp_name=''):
    bins = attributes_dict['bins']
    count = attributes_dict['count']

    names = results_dict.keys()
    metric_results = map(lambda res_list: np.array(res_list), results_dict.values())
    for name, result in zip(names, metric_results):
        plot_histogram(x=count, y=result, n_bins=bins, metric_name=metric,
                       plt_title=' ' + name + ' ' + metric,
                       write_folder=write_folder, exp_name=exp_name)


def plot_region_results():
    return 0


def plot_time_to_f1(run_region_dict=None, run_percentage_dict=None, results_dict=None, run_result_dict=None,
                    metric=None,
                    plt_title='',
                    write_folder=None,
                    exp_name='', batch_of_batch=False,
                    plot_each=True
                    ):

    x = np.zeros(len(run_percentage_dict.keys()))# - 2)
    x_pred = np.zeros(len(run_percentage_dict.keys()))# - 2)
    y = np.zeros(len(run_percentage_dict.keys()))# - 2)

    percent_regions_metrics = {}

    del run_region_dict['X']
    del run_region_dict['Y']
    sorted_runs = sorted(list(map(int, list(run_percentage_dict.keys()))))
    for idx, run_num in enumerate(sorted_runs):  # range(len(run_region_dict)-2):
        # if batch_of_batch:
        run_result_num = run_num
        # else:
        #    run_result_num = idx

        if run_num == 'X_BOX' or run_num == 'Y_BOX' or \
                run_num == 'X' or run_num == 'Y':
            continue

        # cleaned_run_num = cleaned_run_num - 4




        # percent_region = 100.0 * (subsection_area / (float(X)*float(Y)))
        t_train, t_pred, f1 = run_percentage_dict[run_num]



        run_num = int(run_num)
        x[idx] = t_train
        x_pred[idx] = t_pred
        # percent_regions_metrics[idx] = (percent_region, macro_f1, micro_f1, class_f1)
        y[idx] = f1

    # p_regions =  sorted(percent_regions_metrics.items(), key=lambda x: x[1][0])
    # p_regions = dict(p_regions)
    # x = [ i[1][0] for i in p_regions]
    # x = x[1:]    #x       = x[1:]#-1]
    # y_macro =  [ i[1][1] for i in p_regions][1:]#y_macro[1:]#-1]
    # y_micro =  [ i[1][2] for i in p_regions][1:]#y_micro[1:]#-1]
    # y_class =  [ i[1][3] for i in p_regions][1:]#y_class[1:]#-1]

    if plot_each:
        fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(10, 10))
        colors = plt.cm.get_cmap('Dark2')

        # ax.scatter(x,y_macro, color=colors(0), label='Macro F1',
        #            marker="d", s=10, edgecolors=colors(1), linewidths=0.75)
        # ax.scatter(x, y_micro, color=colors(2), label='Micro F1',
        #            marker="s", s=10, edgecolors=colors(3), linewidths=0.75)

        # ax.scatter(x, y, color=colors(2),
        #            marker="s", s=10, edgecolors=colors(3), linewidths=0.75)


        # ax.plot(x, y_macro, color=colors(1), label='Macro F1')
        # ax.plot(x, y_micro, color=colors(3), label='Micro F1')

        ax.loglog(x, y,basex=2,basey=10, color=colors(1))  # , label='Class F1')

        # ax.legend(loc='lower right')
        ax.set(
               xlabel='Time (s) Taken to Train', ylabel='F1 Score vs Training Time')
        # plt.yscale('log')

        fig.tight_layout()
        plt.savefig(os.path.join(write_folder, plt_title + '.png'))
        plt.close(fig)

    # names = results_dict.keys()
    # metric_results = map(lambda res_list: np.array(res_list), results_dict.values())
    # for name, result in zip(names, metric_results):
    #     plot_region_results()
    return x, y


def plot_region_to_f1(run_region_dict=None, run_percentage_dict=None, results_dict=None, run_result_dict=None,
                      metric=None,
                      plt_title='',
                      write_folder=None,
                      exp_name='', batch_of_batch=False,
                      plot_each=True
                      ):
    X = run_region_dict['X']
    Y = run_region_dict['Y']
    # window_file = region_dict['window_file']
    macro_f1s = results_dict['macro']
    micro_f1s = results_dict['micro']
    x = np.zeros(len(run_region_dict.keys()) - 2)
    y_macro = np.zeros(len(run_region_dict.keys()) - 2)
    y_micro = np.zeros(len(run_region_dict.keys()) - 2)
    y_class = np.zeros(len(run_region_dict.keys()) - 2)

    percent_regions_metrics = {}

    del run_region_dict['X']
    del run_region_dict['Y']
    sorted_runs = sorted(list(map(int, list(run_percentage_dict.keys()))))
    for idx, run_num in enumerate(sorted_runs):  # range(len(run_region_dict)-2):
        # if batch_of_batch:
        run_result_num = run_num
        # else:
        #    run_result_num = idx

        if run_num == 'X_BOX' or run_num == 'Y_BOX' or \
                run_num == 'X' or run_num == 'Y':
            continue

        # cleaned_run_num = cleaned_run_num - 4

        boxes = run_region_dict[run_num]
        X_BOX = boxes[0]
        Y_BOX = boxes[1]
        total_x = 0.
        total_y = 0.
        subsection_area = 0.
        for bounds_x, bounds_y in zip(X_BOX, Y_BOX):
            total_x = bounds_x[1] - bounds_x[0]
            total_y = bounds_y[1] - bounds_y[0]
            subsection_area += float(total_x) * float(total_y)

        # percent_region = 100.0 * (subsection_area / (float(X)*float(Y)))
        percent_region = run_percentage_dict[run_num]

        macro_f1 = run_result_dict[run_result_num]['macro']
        micro_f1 = run_result_dict[run_result_num]['micro']
        class_f1 = run_result_dict[run_result_num]['class']

        run_num = int(run_num)
        x[idx] = percent_region
        percent_regions_metrics[idx] = (percent_region, macro_f1, micro_f1, class_f1)
        y_macro[idx] = macro_f1
        y_micro[idx] = micro_f1
        y_class[idx] = class_f1

    p_regions = sorted(percent_regions_metrics.items(), key=lambda x: x[1][0])
    # p_regions = dict(p_regions)
    x = [i[1][0] for i in p_regions]
    x = x[0:]  # x       = x[1:]#-1]
    y_macro = [i[1][1] for i in p_regions][0:]  # y_macro[1:]#-1]
    y_micro = [i[1][2] for i in p_regions][0:]  # y_micro[1:]#-1]
    y_class = [i[1][3] for i in p_regions][0:]  # y_class[1:]#-1]

    if plot_each:
        fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(10, 10))
        colors = plt.cm.get_cmap('Dark2')

        # ax.scatter(x,y_macro, color=colors(0), label='Macro F1',
        #            marker="d", s=10, edgecolors=colors(1), linewidths=0.75)
        # ax.scatter(x, y_micro, color=colors(2), label='Micro F1',
        #            marker="s", s=10, edgecolors=colors(3), linewidths=0.75)
        ax.scatter(x, y_class, color=colors(2),
                   marker="s", s=10, edgecolors=colors(3), linewidths=0.75)
        # ax.plot(x, y_macro, color=colors(1), label='Macro F1')
        # ax.plot(x, y_micro, color=colors(3), label='Micro F1')
        ax.plot(x, y_class, color=colors(1))  # , label='Class F1')
        # ax.legend(loc='lower right')
        ax.set(
               xlabel='Percent Sub-Region Used for Training', ylabel='F1 Score')
        # plt.yscale('log')

        fig.tight_layout()
        plt.savefig(os.path.join(write_folder, plt_title + '.png'))
        plt.close(fig)

    # names = results_dict.keys()
    # metric_results = map(lambda res_list: np.array(res_list), results_dict.values())
    # for name, result in zip(names, metric_results):
    #     plot_region_results()
    return x, y_class


def average_metric_score(results_dict=None,
                         metric=None,
                         write_folder=None,
                         exp_name=''):
    getcontext().prec = 5
    names = results_dict.keys()
    metric_results = map(lambda res_list: np.array(res_list), results_dict.values())
    metric_average_file = os.path.join(write_folder, "metric_averages.txt")
    metric_average_file = open(metric_average_file, "w+")

    for name, result in zip(names, metric_results):
        (mu, sigma) = norm.fit(result)
        metric_average_file.write(name + "_" + metric + " " +
                                  str(Decimal(str(mu)) * Decimal("1")) + ' +/- ' + str(
            Decimal(str(sigma)) * Decimal("1")) + '\n')
    metric_average_file.close()


def metric_standard_deviation(results_dict=None,
                              metric=None,
                              write_folder=None,
                              exp_name=''):
    names = results_dict.keys()
    metric_results = map(lambda res_list: np.array(res_list), results_dict.values())
    metric_std_file = os.path.join(write_folder, "metric_std.txt")
    metric_std_file = open(metric_std_file, "w+")

    for name, result in zip(names, metric_results):
        metric_std_file.write(name + "_" + metric + " " + str(np.std(result)) + '\n')
    metric_std_file.close()


# def main():
#     exp_path = os.path.join(LS.project_base_path, 'datasets', args.data_folder,
#                                           args.exp_folder)
#     runs_folder = os.listdir(os.path.join(LS.project_base_path, 'datasets', args.data_folder,
#                                           args.exp_folder, args.runs))
#     removed_windows_file = os.path.join(str(LS.project_base_path), 'datasets', str(args.data_folder),
#                                         str(args.exp_folder), 'removed_windows.txt')
#     rm_runs = []
#     if os.path.exists(removed_windows_file):
#         removed_runs = open(removed_windows_file, 'r+')
#         lines = removed_runs.readlines()
#         rm_runs = []
#         for l in lines:
#             run_window = l.split(" ")
#             r = run_window[0]
#
#
#             rm_runs.append(str(r))
#         removed_runs.close()
#
#     cleaned_runs_folder = []
#     for rf in runs_folder:
#         size_r = len(os.listdir(os.path.join(exp_path, args.runs, str(rf))))
#         if rf not in rm_runs and size_r > 0:
#             cleaned_runs_folder.append(rf)
#     runs_folder = cleaned_runs_folder
#
#
#     metric_plot_folder = os.path.join(str(exp_path),'batch_metrics')
#     if not os.path.exists(metric_plot_folder):
#         os.makedirs(metric_plot_folder)
#     metrics = ['precision', 'recall', 'f1']
#     for metric in metrics:
#         metric_subfolder = os.path.join(metric_plot_folder,metric)
#         if not os.path.exists(metric_subfolder):
#             os.makedirs(metric_subfolder)
#         results_dict, attributes_dict, run_region_dict, idx_run_dict = collect_run_metrics(metric=metric,
#                                                             runs_folder=runs_folder)
#
#
#
#         compute_metric_histogram(results_dict=results_dict,
#                                  attributes_dict=attributes_dict,
#                                  metric=metric,
#                                  write_folder = metric_subfolder)
#         average_metric_score(results_dict=results_dict,
#                                  metric=metric,
#                                  write_folder = metric_subfolder)
#         #metric_standard_deviation(results_dict=results_dict,
#         #                         metric=metric,
#         #                         write_folder = metric_subfolder)
#     results_dict, attributes_dict, run_region_dict, idx_run_dict = collect_run_metrics(metric='f1',
#                                                                                        runs_folder=runs_folder)
#     plot_region_to_f1(run_region_dict=run_region_dict,
#                       results_dict=results_dict,
#                       run_result_dict=idx_run_dict)
def multi_run_metrics(model, exp_folder, bins=None, runs='runs', plt_title='', avg_multi=False,
                      batch_of_batch=False, data_folder=None,
                      box_dims=None, window_file='window.txt', batch_multi_run=False, metric='f1'):
    # if args.exp_folder is not None:
    # args.exp_folder = exp_folder
    # if args.runs is None:
    args.model = model
    args.data_folder = str(data_folder)
    args.exp_dir = exp_folder
    args.exp_folder = exp_folder
    # args.metric = None#metric
    args.bins = bins
    args.runs = runs
    args.plt_title = plt_title
    # if args.exp_folder is None:
    #     args.__setattr__('data_folder', data_folder)
    #     args.__setattr__('exp_folder', exp_folder)
    # data_folder=str(data_folder)
    # print("    *",
    #     args.exp_folder
    # )
    # removed_windows_file = os.path.join(exp_folder, 'removed_windows.txt')
    #
    # print("     removed windows *", removed_windows_file)
    # r_folder = os.path.join(exp_folder, runs)
    # rm_runs = []
    # if os.path.exists(removed_windows_file):
    #     removed_runs = open(removed_windows_file,'r+')
    #     print("    * exp:", exp_folder)
    #     #print("    * removed file: ", removed_runs)
    #     lines = removed_runs.readlines()
    #     print("    * removed windows", lines)
    #     rm_runs = []
    #     for l in lines:
    #         run_window = l.split(" ")
    #         r =  run_window[0]
    #         rm_runs.append(str(r))
    #     removed_runs.close()

    # clear_removed_windows = open(removed_windows_file,'w+')
    # clear_removed_windows.close()
    run_path = os.path.join(str(exp_folder), runs)

    print(" multi model metric exp run folder")

    print("exp", exp_folder)
    print('runs', runs)
    exp_name = exp_folder.replace('_', ' ').replace('2', '').title()
    runs_folder = os.listdir(os.path.join(str(exp_folder), runs)) if not batch_multi_run else \
        os.listdir(os.path.join(str(exp_folder), runs))

    # cleaned_runs_folder= []
    # for rf in runs_folder:
    #     size_r = len(os.listdir(os.path.join(r_folder, str(rf))))
    #     if rf not in rm_runs and size_r > 0:
    #         cleaned_runs_folder.append(rf)
    # runs_folder = cleaned_runs_folder

    metric_plot_folder = os.path.join(str(exp_folder), 'batch_metrics') if not batch_multi_run else \
        os.path.join(str(exp_folder), 'batch_metrics', runs)
    if not os.path.exists(metric_plot_folder):
        os.makedirs(metric_plot_folder)
    metrics = ['precision', 'recall', 'f1']

    if not avg_multi:
        for metric in metrics:
            metric_subfolder = os.path.join(metric_plot_folder, metric)
            if not os.path.exists(metric_subfolder):
                os.makedirs(metric_subfolder)
            metric_file = None

            results_dict, attributes_dict, run_region_dict,\
            idx_run_dict, run_percent_dict = collect_run_metrics(
                metric=metric,
                run_path=run_path,
                runs_folder=runs_folder,
                batch_multi_run=batch_multi_run,
                batch_of_batch=batch_of_batch,
                model=model, bins=bins)

            # compute_metric_histogram(results_dict=results_dict,
            #                          attributes_dict=attributes_dict,
            #                          metric=metric,
            #                          write_folder=metric_subfolder,
            #                          exp_name= str(exp_folder))
            # average_metric_score(results_dict=results_dict,
            #                      metric=metric,
            #                      write_folder=metric_subfolder,
            #                      exp_name= str(exp_folder))
            # metric_standard_deviation(results_dict=results_dict,
            #                           metric=metric,
            #                           write_folder=metric_subfolder,
            #                           exp_name= str(exp_folder))
    # metric_subfolder = os.path.join(metric_plot_folder, 'f1')
    # if not os.path.exists(metric_subfolder):
    #    os.makedirs(metric_subfolder)
    else:
        for metric in metrics:
            metric_subfolder = os.path.join(metric_plot_folder, metric)
            if not os.path.exists(metric_subfolder):
                os.makedirs(metric_subfolder)
            results_dict, attributes_dict, \
            run_region_dict, idx_run_dict, run_percent_dict = collect_run_metrics(metric=metric,
                                                                                  batch_of_batch=batch_of_batch,
                                                                                  run_path=run_path,
                                                                                  batch_multi_run=batch_multi_run,
                                                                                  runs_folder=runs_folder,
                                                                                  avg_multi=avg_multi, model=model,
                                                                                  bins=bins)

            # compute_metric_histogram(results_dict=results_dict,
            #                          attributes_dict=attributes_dict,
            #                          metric=metric,
            #                          write_folder=metric_subfolder,
            #                          exp_name=str(exp_folder))
            # average_metric_score(results_dict=results_dict,
            #                      metric=metric,
            #                      write_folder=metric_subfolder,
            #                      exp_name=str(exp_folder))
    model_type = "Graphsage MeanPool" if str(model) == 'getognn' else str(model)
    model_type = "Random Forest" if str(model) == 'random_forest' else str(model)

    dataset = str(exp_folder).split('/')[-2]
    dataset = dataset.replace('_', ' ')

    plot_region_to_f1(run_region_dict=run_region_dict,
                      run_percentage_dict=run_percent_dict,
                      batch_of_batch=batch_of_batch,
                      results_dict=results_dict,
                      run_result_dict=idx_run_dict,
                      write_folder=metric_subfolder,
                      plt_title=model_type + ' ' + dataset)


def multi_model_metrics(models, exp_dirs, write_dir, bins=None, runs='runs', data_name='',
                        plt_title='', avg_multi=False,
                        batch_of_batch=False, batch_multi_run=False, metric='f1'):
    model_statistics = {}

    for model, exp_folder in zip(models, exp_dirs):

        print("model: ", model)
        print("exp folder: ", exp_folder)

        model_statistics[model] = []


        run_path = os.path.join(str(exp_folder), runs)

        runs_folder = os.listdir(os.path.join(str(exp_folder), runs)) if not batch_multi_run else \
            os.listdir(os.path.join(str(exp_folder), runs))
        # runs_folder.sort()
        print("    * runs folder", runs_folder)



        metric_plot_folder = os.path.join(str(exp_folder), 'batch_metrics') if not batch_multi_run else \
            os.path.join(str(exp_folder), 'batch_metrics', runs)
        if not os.path.exists(metric_plot_folder):
            os.makedirs(metric_plot_folder)

        if metric != 'time':
            metrics = ['precision', 'recall', 'f1']
        else:
            metrics = ['time']

        if not avg_multi:
            for metric in metrics:
                metric_subfolder = os.path.join(metric_plot_folder, metric)
                if not os.path.exists(metric_subfolder):
                    os.makedirs(metric_subfolder)
                metric_file = None

                results_dict, attributes_dict, run_region_dict, idx_run_dict, \
                run_percent_dict = collect_run_metrics(
                    metric=metric,
                    run_path=run_path,
                    runs_folder=runs_folder,
                    batch_multi_run=True,
                    batch_of_batch=batch_of_batch,
                    model=model, bins=bins)

                # compute_metric_histogram(results_dict=results_dict,
                #                          attributes_dict=attributes_dict,
                #                          metric=metric,
                #                          write_folder=metric_subfolder,
                #                          exp_name= str(exp_folder))
                # average_metric_score(results_dict=results_dict,
                #                      metric=metric,
                #                      write_folder=metric_subfolder,
                #                      exp_name= str(exp_folder))
                # metric_standard_deviation(results_dict=results_dict,
                #                           metric=metric,
                #                           write_folder=metric_subfolder,
                #                           exp_name= str(exp_folder))
        # metric_subfolder = os.path.join(metric_plot_folder, 'f1')
        # if not os.path.exists(metric_subfolder):
        #    os.makedirs(metric_subfolder)
        else:
            for metric in metrics:
                metric_subfolder = os.path.join(metric_plot_folder, metric)
                if not os.path.exists(metric_subfolder):
                    os.makedirs(metric_subfolder)

                results_dict, attributes_dict, \
                run_region_dict, idx_run_dict, run_percent_dict = collect_run_metrics(metric=metric,
                                                                                      run_path=run_path,
                                                                                      batch_multi_run=True,
                                                                                      runs_folder=runs_folder,
                                                                                      avg_multi=avg_multi, model=model,
                                                                                      bins=bins)

                # compute_metric_histogram(results_dict=results_dict,
                #                          attributes_dict=attributes_dict,
                #                          metric=metric,
                #                          write_folder=metric_subfolder,
                #                          exp_name=str(exp_folder))
                # average_metric_score(results_dict=results_dict,
                #                      metric=metric,
                #                      write_folder=metric_subfolder,
                #                      exp_name=str(exp_folder))
        model_type = "Graphsage MeanPool" if str(model) == 'getognn' else str(model)
        model_type = "Random Forest" if str(model) == 'random_forest' else str(model)

        dataset = str(exp_folder).split('/')[-2]
        dataset = dataset.replace('_', ' ')

        if metric != 'time':
            x, y = plot_region_to_f1(run_region_dict=run_region_dict,
                                     run_percentage_dict=run_percent_dict,
                                     batch_of_batch=avg_multi,
                                     results_dict=results_dict,
                                     run_result_dict=idx_run_dict,
                                     write_folder=metric_subfolder,
                                     plt_title=model_type + ' ' + dataset)
        else:
            x, y = plot_time_to_f1(run_region_dict=run_region_dict,
                                   run_percentage_dict=run_percent_dict,
                                   batch_of_batch=avg_multi,
                                   results_dict=results_dict,
                                   run_result_dict=idx_run_dict,
                                   write_folder=metric_subfolder,
                                   plt_title=model_type + ' ' + dataset)

        model_statistics[model] = [x, y]

    exp_name = data_name.replace('_', ' ').replace('2', '').title()

    if metric == 'time':
        title = exp_name + ' Training Time log(s) to F1'
        xlab = 'Time log(s) Taken to Train'
    else:
        title = exp_name + ' F1 to Percent Training Size'
        xlab = 'Percent Sub-Region Used for Training'

    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(10, 10))
    colors = plt.cm.get_cmap('Dark2')

    plot_experiments = [
                        "Random_Forest_Pixel",
                        'MLP_Pixel',
                        "UNet",
                        "Random_Forest_MSC",
                        # 'Random_Forest_MSC_Geom',
                        'MLP_MSC',
                        'GNN',
                        'GNN_Geom',
                        'GNN_SUB'
                        ]
    legend_order = ['RF-Pixel', 'MLP-Pixel', 'U-Net' , 'RF-Priors', 'MLP-Priors', 'GNN','GNN-Geom']
    #for model_name, stats in model_statistics.items():
    print("    * ","plotting models")
    print("    * ", model_statistics.keys())
    for legend_row, model_exp in enumerate(model_statistics.keys()):
        model_name = plot_experiments[legend_row].replace('_', ' ')
        stats = model_statistics[model_name]
        x = stats[0]
        y = stats[1]
        #y = [float(round(i, 2)) for i in list(y)]

        #ax.scatter(x, y, color=colors(c),
        #           marker="s", s=10, edgecolors=colors(c + 1), linewidths=0.75)

        # ax.plot(x, y_macro, color=colors(1), label='Macro F1')
        # ax.plot(x, y_micro, color=colors(3), label='Micro F1')
        #ax.yaxis.get_major_formatter().set_scientific(False)
        #ax.yaxis.get_major_formatter().set_useOffset(False)
        import matplotlib.ticker as ticker
        from matplotlib.ticker import EngFormatter

        if metric == 'time':
            ax.set_xscale('log')
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            formatter0 = EngFormatter(unit='s')
            ax.xaxis.set_major_formatter(formatter0)
        else:
            formatter0 = PercentFormatter()
            ax.xaxis.set_major_formatter(formatter0)
        if model_name=='UNet':
            model_name = 'U-Net'
            c = 'blue'
            lstyle = (0, (1, 1)) # densely dotted 'solid' #(0, (5, 1))#'densely dashed'
            lw = 2.0
        if 'Pixel' in model_name and 'Forest' in model_name and 'Geom' not in model_name:
            model_name = 'Random Forest Pixel'
            c = 'seagreen'# 'blueviolet'#'goldenrod'
            lstyle =  (0, (1, 1)) #(0, (3, 1, 1, 1, 1, 1)) # densely daddotted
            ## (0, (1, 1)) # densely dotted(0, (5, 1)) # dashed (0, (1, 1))# 'densely dotted'
            lw = 2.0
        if 'Pixel' in model_name and 'MLP' in model_name:
            model_name = 'MLP Pixel'
            c = 'sienna'#'peru'# 'darkgoldenrod'
            lstyle =  (0, (1, 1)) #(0, (3, 1, 1, 1, 1, 1)) # densely daddotted
            ## (0, (1, 1)) # densely dotted(0, (5, 1)) # dashed (0, (1, 1))# 'densely dotted'
            lw = 2.0
        if model_name == 'GNN':
            c = 'red'
            lstyle = (0, (5, 1)) #dense dash'solid'# (0, (3, 5, 1, 5, 1, 5))#'dashdotdotted'
            lw = 2.0
        if 'GNN' in model_name and 'Geom' in model_name:
            c = 'gainsboro'#
            lstyle = (0, (5, 1)) #dense dash'solid'# (0, (3, 5, 1, 5, 1, 5))#'dashdotdotted'
            lw = 2.0
        if 'MSC' in model_name and 'Forest' in model_name and 'Geom' not in model_name:
            model_name = 'Random Forest Priors'
            c = 'springgreen'#'darkorange'
            lstyle = (0, (5, 1))#(0, (1, 1)) #(0, (5, 1))#'densely dashed'   #(0, (3, 5, 1, 5))#'dashdotted'
            lw = 2.0
        if 'MSC' in model_name and 'Forest' in model_name and 'Geom' in model_name:
            model_name = 'Random Forest Geom'
            c = 'aqua'
            lstyle = (0, (5, 1))#(0, (1, 1)) #(0, (5, 1))#'densely dashed'   #(0, (3, 5, 1, 5))#'dashdotted'
            lw = 2.0
        if 'Pixel' in model_name and 'Forest' in model_name and 'Geom' in model_name:
            model_name = 'Random Forest Pixel Geom'
            c = 'deeppink'
            lstyle = (0, (1, 1))#(0, (1, 1)) #(0, (5, 1))#'densely dashed'   #(0, (3, 5, 1, 5))#'dashdotted'
            lw = 2.0
        if 'MSC' in model_name and 'MLP' in model_name:
            model_name = 'MLP Priors'
            c =  'darkorange'# for pixel chartreuse
            lstyle = (0, (5, 1))#(0, (1, 1))  #(0, (3, 5, 1, 5))#'dashdotted'
            lw = 2.0
        if 'GNN' in model_name and 'SUB' in model_name:
            model_name = 'GNN Complex Informed'
            c = 'orchid'#
            lstyle = (0, (5, 1)) #dense dash'solid'# (0, (3, 5, 1, 5, 1, 5))#'dashdotdotted'
            lw = 2.0
        ax.plot(x, y,color=c, label=model_name, linestyle=lstyle, linewidth=lw)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if 'retinal' in data_name:
        ax.legend(loc='lower right',fontsize=11)
        if metric != 'time':
            plt.xlabel("Percent Sub-Set Used for Training",fontsize=17)
        else:
            plt.xlabel("Time log(s) Taken to Train",fontsize=17)
    else:
        plt.xlabel(" ")
        plt.ylabel(" ")

    # ax.set(
    #        xlabel=xlab, ylabel='F1 Score')#, fontsize=12)


    fig.tight_layout()
    write_folder = os.path.join(str(write_dir), 'batch_metrics')
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    #ax.set_xticklabels(x, fontsize=5)
    #ax.set_yticklabels(y, fontsize=5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    if metric != 'time':
        ax.xaxis.grid(color='gray', linestyle='dashed')
    print("    * : placing plots in ", write_folder)
    plt.savefig(os.path.join(write_folder, data_name + '_multi_model_performance_' + metric + '.png'))
    plt.close(fig)


def main():
    model = args.model
    bins = args.bins
    plt_title = args.title
    batch_multi_run = args.batch
    exp_folder = os.path.join('/home/sam/Documents/PhD/Research/GeToGNN/datasets', args.exp_dir)
    args.__setattr__('exp_folder', exp_folder)
    data_dir = args.data_dir
    args.__setattr__('data_folder', data_dir)
    metric = args.metric
    runs = args.runs
    args.__setattr__('plt_title', plt_title)

    multi_run_metrics(exp_folder, model=model, bins=bins, metric=metric, runs=runs, plt_title=plt_title,
                      avg_multi=False,
                      batch_of_batch=False, data_folder=data_dir,
                      box_dims=None, window_file='window.txt',
                      batch_multi_run=batch_multi_run)


if __name__ == "__main__":
    main()
