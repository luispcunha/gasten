import json
import numpy as np
import matplotlib.pyplot as plt


def my_plotter(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out


if __name__ == '__main__':
    with open('out/stats.json', 'r') as file:
        data = json.load(file)

    plt.plot(data['G_losses'], label='G loss')  # Plot some data on the (implicit) axes.
    plt.plot(data['D_losses'], label='D loss')  # etc.
    plt.xlabel('mini batch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(data['G_losses_epoch'], label='G loss')  # Plot some data on the (implicit) axes.
    plt.plot(data['D_losses_epoch'], label='D loss')  # etc.
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(data['D_x_epoch'], label='D(x)')  # Plot some data on the (implicit) axes.
    plt.plot(data['D_G_z1_epoch'], label='D(G(z)) 1')  # etc.
    plt.plot(data['D_G_z2_epoch'], label='D(G(z)) 2')  # etc.
    plt.xlabel('epoch')
    plt.ylabel('output')
    plt.legend()
    plt.show()

    plt.plot(data['D_x'], label='D(x)')  # Plot some data on the (implicit) axes.
    plt.plot(data['D_G_z1'], label='D(G(z)) 1')  # etc.
    plt.plot(data['D_G_z2'], label='D(G(z)) 2')  # etc.
    plt.xlabel('mini batch')
    plt.ylabel('output')
    plt.legend()
    plt.show()

    plt.plot(data['D_acc_real'], label='D acc real')  # Plot some data on the (implicit) axes.
    plt.plot(data['D_acc_fake_1'], label='D acc fake 1')  # etc.
    plt.plot(data['D_acc_fake_2'], label='D acc fake 2')  # etc.
    plt.xlabel('mini batch')
    plt.ylabel('output')
    plt.legend()
    plt.show()
