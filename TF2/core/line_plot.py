import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns


def plot_data(data, title, labels, y_axis):
    fig = plt.figure(figsize=(10, 5), dpi=100)
    for idx, line in enumerate(data):
        y = [x for _, x in sorted(zip(line[0], line[1]))]
        x = sorted(line[0])
        plt.plot(x, y, 'o-', label=labels[idx], linewidth=2)

    plt.title(title)
    plt.xlabel('US rate')
    plt.ylabel(y_axis)
    plt.ylim(0, None)
    plt.legend()
    plt.show()
    plt.close()
    return fig


def preprocessing_data(files):
    data = []
    for file in files:
        x_axis = []
        y_axis = []
        f = open(file, 'r')
        for line in f.readlines():
            x_raw = [j for j in [i for i in line.split('_') if 'US' in i][0].split(':') if 'US' in j]
            x = int(''.join(filter(str.isdigit, *x_raw)))
            y = float([i for i in line.split(':')][-1])
            x_axis.append(x)
            y_axis.append(y)
        data.append([x_axis, y_axis])
        f.close()

    return data


def filter_data(files, mode='all'):
    for file in files:
        EAE = []
        EPE = []
        f = open(file, 'r')
        for line in f.readlines():
            if mode != 'all':
                US_rate = [int(i.replace('US', '')) for i in line.split('_') if 'US' in i]
                if (US_rate[0] % 2) == 0 and mode == 'odd':
                    continue
                if (US_rate[0] % 2) == 1 and mode == 'even':
                    continue
            if 'EPE' in line:
                epe = '_'.join([i for i in re.split('_', line) if 'EPE' not in i])
                EPE.append(epe)
            elif 'EAE' in line:
                eae = '_'.join([i for i in re.split('_', line) if 'EAE' not in i])
                EAE.append(eae)
        f.close()
        save_file_EAE = file.split('.')[0] + '_EAE.txt'
        save_file_EPE = file.split('.')[0] + '_EPE.txt'
        with open(save_file_EAE, "w") as outfile:
            outfile.write("".join(EAE))
        with open(save_file_EPE, "w") as outfile:
            outfile.write("".join(EPE))


def save_img(result, save_path, format='png'):
    result.savefig(save_path + '.' + format)


def plot_box_plot(save_dir):
    sns.set_style("dark")
    fig, axes = plt.subplots(nrows=2, ncols=7)
    labels = ['0', '0.1', '0.2', '0.3', '0.4']
    name = ['1x', '5x', '10x', '15x', '20x', '25x', '30x']
    for i, acc in enumerate([1, 5, 10, 15, 20, 25, 30]):
        for j, error in enumerate(['EPE', 'EAE']):
            files = [f'/home/user/lapnet/results/EAE_integration/10_0/US{acc}_{error}_loss.txt',
                     f'/home/user/lapnet/results/EAE_integration/9_1/US{acc}_{error}_loss.txt',
                     f'/home/user/lapnet/results/EAE_integration/8_2/US{acc}_{error}_loss.txt',
                     f'/home/user/lapnet/results/EAE_integration/7_3/US{acc}_{error}_loss.txt',
                     f'/home/user/lapnet/results/EAE_integration/6_4/US{acc}_{error}_loss.txt']
            clrs = sns.color_palette("husl", len(files))
            res = np.zeros((20, len(files)), dtype=np.float32)
            for idx in range(len(files)):
                f = open(files[idx], 'r')
                error_list = []
                for line in f.readlines():
                    error_list.append(float(line))
                f.close()
                res[:, idx] = np.asarray(error_list)
                bp = axes[j, i].boxplot(res, showfliers=False, patch_artist=True)
                axes[j, i].set_xticklabels([])
                axes[0, i].set_title(name[i], fontsize=10)
                legend_list = []
                for num, patch in enumerate(bp['boxes']):
                    patch.set(facecolor=clrs[num])
                    legend_list.append(patch)
                if i > 0:
                    axes[j, i].set_yticklabels([])

                axes[0, i].set_ylim(0.1, 0.65)
                axes[1, i].set_ylim(5, 21)

    fig.legend(legend_list, labels, loc='upper center', ncol=5, fancybox=True, shadow=True)
    axes[0, 0].set(ylabel='EPE(pixel)')
    axes[1, 0].set(ylabel='EAE(degree)')
    # ax2.set(xlabel='acceleration', ylabel='EAE(degree)')
    plt.show()
    fig.savefig(save_dir)
    plt.close()


def plot_fully(save_dir):
    sns.set_style("dark")
    fig, axes = plt.subplots(nrows=2, ncols=1)
    labels = ['1', '0.1', '0.2', '0.3', '0.4']
    # name = ['1x', '5x', '10x', '15x', '20x', '25x', '30x']
    for i, acc in enumerate([1]):
        for j, error in enumerate(['EPE', 'EAE']):
            files = [f'/home/user/lapnet/results/elastix/US1_{error}_loss.txt',
                     f'/home/user/lapnet/results/LAPNet/US1_{error}_loss.txt']
            clrs = sns.color_palette("husl", len(files))

            for idx in range(len(files)):
                res = np.zeros((20, len(files)), dtype=np.float32)
                f = open(files[idx], 'r')
                error_list = []
                for line in f.readlines():
                    error_list.append(float(line))
                f.close()
                res[:, idx] = np.asarray(error_list)
                bp = axes[j, i].boxplot(res, showfliers=False, patch_artist=True)
                axes[j, i].set_xticklabels([])
                # axes[0, i].set_title(name[i], fontsize=10)
                legend_list = []
                for num, patch in enumerate(bp['boxes']):
                    patch.set(facecolor=clrs[num])
                    legend_list.append(patch)
                if i > 0:
                    axes[j, i].set_yticklabels([])

                axes[0, i].set_ylim(0.1, 0.65)
                axes[1, i].set_ylim(5, 21)

    # fig.legend(legend_list, labels, loc='right', ncol=1, fancybox=True, shadow=True)
    axes[0, 0].set(ylabel='EPE(pixel)')
    axes[1, 0].set(ylabel='EAE(degree)')
    # ax2.set(xlabel='acceleration', ylabel='EAE(degree)')
    plt.show()
    fig.savefig(save_dir)
    plt.close()


def plot_compare(files, names, save_dir):
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    x = np.linspace(1, 30, 30)
    clrs = sns.color_palette("husl", len(files))
    for i in range(len(files)):
        f = open(files[i], 'r')
        EAE = []
        EPE = []
        standard_dev_EPE = []
        standard_dev_EAE = []
        for line in f.readlines():
            if 'EPE' in line:
                epe = re.split(',', line)[1]
                EPE.append(float(epe))
                std = re.split(',', line)[2]
                standard_dev_EPE.append(float(std))
            elif 'EAE' in line:
                eae = re.split(',', line)[1]
                EAE.append(float(eae))
                std = re.split(',', line)[2]
                standard_dev_EAE.append(float(std))

        f.close()
        ax1.plot(x, EPE, label=names[i], c=clrs[i])
        negative_error = [a_i - b_i for a_i, b_i in zip(EPE, standard_dev_EPE)]
        positive_error = [a_i + b_i for a_i, b_i in zip(EPE, standard_dev_EPE)]
        ax1.fill_between(x, negative_error, positive_error, alpha=0.3, facecolor=clrs[i])

        ax2.plot(x, EAE, c=clrs[i])
        negative_error = [a_i - b_i for a_i, b_i in zip(EAE, standard_dev_EAE)]
        positive_error = [a_i + b_i for a_i, b_i in zip(EAE, standard_dev_EAE)]
        ax2.fill_between(x, negative_error, positive_error, alpha=0.3, facecolor=clrs[i])

    fig.legend(loc='upper center', ncol=4, fancybox=True, shadow=True)
    ax1.set(ylabel='EPE(pixel)')
    ax2.set(xlabel='acceleration', ylabel='EAE(degree)')
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)
    # ax1.set_xticks(np.arange(, 31, 5))
    # ax2.set_xticks(np.arange(1, 31, 5))
    ax1.set_xlim(1, 30)
    ax2.set_xlim(1, 30)
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')
    plt.show()
    fig.savefig(save_dir)
    plt.close()


if __name__ == '__main__':
    # filter_files = ['/Users/Peter/Desktop/MA_Exp/1104/srx424_drUS_drUS_test.txt',
    #                 '/Users/Peter/Desktop/MA_Exp/1104/flown_srx424_noUS_drUS_test.txt']
    # filter_data(filter_files, mode='odd')

    plot_files = ['/home/jpa19/PycharmProjects/MA/UnFlow/output/flown_srx424_noUS_2003/loss/mean_loss_EPE.txt',
                  '/home/jpa19/PycharmProjects/MA/UnFlow/output/flown_srx424_crUS_2105/loss/mean_loss_EPE.txt',
                  '/home/jpa19/PycharmProjects/MA/UnFlow/output/srx424_drUS_1603/loss/mean_loss_EPE.txt',
                  '/home/jpa19/PycharmProjects/MA/UnFlow/output/srx424_crUS13_test_1104/loss/mean_loss_EPE.txt', ]
    labels = ['FS-Random', 'FS-Center', 'LAP-Random', 'LAP-Center']
    titel = 'Test EPE of Center-US Training'
    save_path = '/home/jpa19/PycharmProjects/MA/UnFlow/line_plot'

    data = preprocessing_data(plot_files)
    fig = plot_data(data, title=titel, labels=labels, y_axis='EPE')
    save_img(fig, save_path=save_path, format='png')
