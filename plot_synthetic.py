import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_context(context_dict, save_path = None):
    '''
    :param context_dict: contains data of context means (key1), along with the associated label (key2)
    :param save_path: optional path for saving figures
    '''
    # data has size (batch_size, context_dim). labels have size (batch_size, 1)?
    means_context = np.array(context_dict['data'])
    labels = np.array(context_dict['labels'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    target_labels = ["exponential", "gaussian", "uniform", "laplace"]
    colours = ["r", "g", "b", "k"]

    for batch_i in range(means_context.shape[0]):
        for context_i in range(means_context.shape[1]):
            ax.scatter(means_context[batch_i][context_i][:, 0],
                       means_context[batch_i][context_i][:, 1],
                       means_context[batch_i][context_i][:, 2],
                       label=target_labels[labels[batch_i][context_i]],
                       color=colours[labels[batch_i][context_i]])

    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
