import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_context(context_dict, savepath = None):
    '''
    :param context_dict: contains data of context means (key1), along with the associated label (key2)
    :param savepath: optional path for saving figures
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    means_context = context_dict['data']
    labels = context_dict['labels']

    target_labels = ["exponential", "gaussian", "uniform", "laplace"]
    colours = ["r", "g", "b", "k"]

    for batch_i in range(len(means_context)):
        for context_i in range(means_context[batch_i].shape[0]):
            ax.scatter(means_context[batch_i][context_i][0],
                       means_context[batch_i][context_i][1],
                       means_context[batch_i][context_i][2],
                       label=target_labels[labels[batch_i][context_i]],
                       color=colours[labels[batch_i][context_i]])

    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

