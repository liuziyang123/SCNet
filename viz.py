import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
from matplotlib import mathtext
import math

plt.rcParams.update({'mathtext.fontset': 'stix'})

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15}


def plot_Xsamples(y):
    color1 = [1, 0.2196, 0]
    color2 = [1, 0.55294, 0]
    color3 = [0, 0.66275, 1]
    color = [color3, color2, color1]
    label = ['18th channel', '32nd channel', '56th channel']
    # plt.figure(figsize=(9.12, 5.12))
    x = np.linspace(0, 999, 1000)
    for i in range(y.shape[0]):
        plt.plot(x, y[i], label=label[i], color=color[i], linewidth=0.75)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.yticks(fontproperties='Times New Roman')  # size=15
    plt.xticks(fontproperties='Times New Roman')
    plt.xlabel(r'1000 samples from the selected validation dataset', font)
    plt.ylabel(r'$d_w$', font)
    plt.legend(prop=font, edgecolor='black')
    plt.savefig('1.pdf')
    plt.show()


def plot_Xchannels(y):
    x = np.linspace(0, 127, 128).astype(np.uint8)
    plt.figure(figsize=(18.24, 5.12))
    map_vir = cm.get_cmap(name='jet')
    norm = plt.Normalize(y.min(), y.max())
    norm_y = norm(y)
    colors = map_vir(norm_y)
    plt.bar(x, y, width=1, color=colors)  # edgecolor='blue',
    sm = cm.ScalarMappable(cmap=map_vir, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm)
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
        # l.set_size(15)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    plt.yticks(fontproperties='Times New Roman')  # size=15
    plt.xticks(fontproperties='Times New Roman')
    plt.xlabel(r'128 feature channels from the encoder', font)
    plt.ylabel('$d_w$', font)
    # plt.title('plot')
    plt.savefig('1.pdf')
    plt.show()


def plot_Xhistogram(y):

    plt.figure(figsize=(9.12, 5.12))
    bins = np.arange(6, 38, 2)
    plt.hist(y, edgecolor='black', bins=bins, color=[0, 0.5, 1])
    # plt.bar(x, y[:28], width=1, edgecolor='black')
    plt.legend()
    a, b = np.histogram(y, bins)
    for a_, b_ in zip(a, b):
        if a_ != 0:
            plt.text(b_+1, a_, '%d' % a_, ha='center', va='bottom', family='Times New Roman')
    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid(axis='y', linestyle=':')

    # u, sig = 14.969576188369581, 11.149000139428104
    # x = np.linspace(8, 36, 280)
    # uniform = 857 * np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig)
    # plt.plot(x, uniform)

    plt.yticks(fontproperties='Times New Roman')  # size=15
    plt.xticks(fontproperties='Times New Roman')
    plt.xlabel(r'Weighted depth value', font)
    plt.ylabel(r'Number of feature channels', font)
    # plt.title('plot')
    plt.savefig('1.pdf')
    plt.show()


def plot_Xhistogram_multi(y, y_):

    color1 = [1, 0.2196, 0]
    color2 = [1, 0.55294, 0]
    color3 = [0, 0.66275, 1]

    # plt.figure(figsize=(9.12, 5.12))
    bins = np.arange(6, 38, 2)
    # plt.hist(y, edgecolor='black', bins=bins, color=[0, 0.5, 1])
    # plt.bar(x, y[:28], width=1, edgecolor='black')

    width = 0.9
    a, b = np.histogram(y, bins)
    a1, b1 = np.histogram(y_, bins)
    for a_, b_ in zip(a, b):
        if a_ != 0:
            plt.text(b_+0.55, a_, '%d' % a_, ha='center', va='bottom', family='Times New Roman', color=color3)
    for a_, b_ in zip(a1, b1):
        if a_ != 0:
            plt.text(b_+1.45, a_, '%d' % a_, ha='center', va='bottom', family='Times New Roman', color=color2)
    b = b[:-1]
    b1 = b1[:-1]
    plt.bar(b + 0.55, a, width=width, color=color3, label='Coarse Network')
    plt.bar(b1 + 1.45, a1, width=width, color=color2, label='RGB Guidance')



    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid(axis='y', linestyle=':')

    # u, sig = 14.969576188369581, 11.149000139428104
    # x = np.linspace(8, 36, 280)
    # uniform = 857 * np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig)
    # plt.plot(x, uniform)

    plt.legend(prop=font, edgecolor='black')
    plt.yticks(fontproperties='Times New Roman')  # size=15
    plt.xticks(fontproperties='Times New Roman')
    plt.xlabel(r'$d_w$', font)
    plt.ylabel(r'Number of feature channels', font)
    # plt.title('plot')
    plt.savefig('1.pdf')
    plt.show()


def main():
    depth_ft = np.loadtxt('statistic/depth_ft.txt')
    depth_ft_channel = np.loadtxt('statistic/depth_ft_channel.txt')
    depth_ft_channel_ = np.loadtxt('statistic/depth_ft_channel_raw.txt')
    # plot_Xchannels(depth_ft_channel)
    # plot_Xsamples(depth_ft)
    plot_Xhistogram_multi(depth_ft_channel, depth_ft_channel_)


if __name__ == '__main__':
    main()
