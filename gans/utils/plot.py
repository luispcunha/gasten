import os
import numpy as np
import matplotlib.pyplot as plt


def plot_train_summary(data, out_path):
    plt.plot(data['G_losses_epoch'], label='G loss')
    plt.plot(data['D_losses_epoch'], label='D loss')

    if 'term_1_epoch' in data:
        plt.plot(np.array(data['term_1_epoch']), label="term_1")

    if 'term_2_epoch' in data:
        plt.plot(np.array(data['term_2_epoch']), label="term_2")

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss.png'))
    plt.clf()

    plt.plot(data['D_x_epoch'], label='D(x)')
    plt.plot(data['D_G_z1_epoch'], label='D(G(z)) 1')
    plt.plot(data['D_G_z2_epoch'], label='D(G(z)) 2')
    plt.xlabel('epoch')
    plt.ylabel('output')
    plt.title('d outputs')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'd_outputs.png'))
    plt.clf()

    plt.plot(data['D_acc_real_epoch'], label='D acc real')
    plt.plot(data['D_acc_fake_1_epoch'], label='D acc fake 1')
    plt.plot(data['D_acc_fake_2_epoch'], label='D acc fake 2')
    plt.xlabel('epoch')
    plt.ylabel('output')
    plt.title('d accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'd_accuracy.png'))
    plt.clf()

    plt.plot(data['fid'], label='FID')
    plt.xlabel('epoch')
    plt.ylabel('fid')
    plt.title('fid')
    plt.legend()
    plt.savefig(os.path.join(out_path, '{}.fid.png'))
    plt.clf()

    if 'focd' in data:
        plt.plot(data['focd'], label='F*D')
        plt.xlabel('epoch')
        plt.ylabel('f*d')
        plt.title('f*d')
        plt.legend()
        plt.savefig(os.path.join(out_path, '{}.f*d.png'))
        plt.clf()

    if 'conf_dist' in data:
        plt.plot(data['conf_dist'], label='conf_dist')
        plt.xlabel('epoch')
        plt.ylabel('conf_dist')
        plt.title('conf_dist')
        plt.legend()
        plt.savefig(os.path.join(out_path, '{}.conf_dist.png'))
        plt.clf()

