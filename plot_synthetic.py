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

    contexts = np.array(context_dict['data'])       # shape: (67,)
    labels = np.array(context_dict['labels'])       # shape: (67,)

    # context[0] has size (batch_size, context_dim), labels[0] have size (batch_size, 1)
    print(contexts[0].shape)    # shape: (30,3)
    print(labels[0].shape)      # shape: (30,)

    distributions = [
            'exponential',
            'gaussian',
            'uniform',
            'laplace'
        ]

    
    n = len(contexts)
    labels = labels[:n]
    ix = [np.where(labels == label)
          for i, label in enumerate(distributions)]
    colors = [
        'indianred',
        'forestgreen',
        'gold',
        'cornflowerblue',
        'darkviolet'
    ]

    for batch_i in range(contexts.shape[0]):
        for label, i in enumerate(ix):
            ax.scatter(contexts[batch_i][i][:, 0], contexts[batch_i][i][:, 1], contexts[batch_i][i][:, 2],
                       label=distributions[label].title(),
                       color=colors[label])

        
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    plt.legend(loc='upper left')
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)


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
    '''
