import matplotlib.pyplot as plt
import numpy as np
import re
import pickle


def plot_data(data, title, labels, y_axis):
    fig = plt.figure(figsize=(10,5), dpi=100)
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
        save_file_EAE = file.split('.')[0]+'_EAE.txt'
        save_file_EPE = file.split('.')[0] + '_EPE.txt'
        with open(save_file_EAE, "w") as outfile:
            outfile.write("".join(EAE))
        with open(save_file_EPE, "w") as outfile:
            outfile.write("".join(EPE))


def save_img(result, save_path, format='png'):
    result.savefig(save_path+'.'+format)


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
