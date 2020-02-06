import os
import pickle
import torch
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from collections import defaultdict
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D

def get_logger(opts):
    return Logger(opts)


class Logger:
    """Class for logging experiments."""
    def __init__(self, opts):
        self.tensorboard = opts.tensorboard

        now = datetime.now()
        now_str = now.strftime("%d:%m:%Y_%H:%M:%S")
        self.experiment_name = f'{opts.experiment}_{now_str}'
        self.log_dir = opts.log_dir
        self.save_dir = opts.save_dir

        os.makedirs(f'{self.save_dir}/{self.experiment_name}', exist_ok=True)

        if opts.tensorboard:
            self.writer = SummaryWriter(f'{self.log_dir}/{self.experiment_name}')
        else:
            os.makedirs(f'{self.log_dir}/{self.experiment_name}', exist_ok = True)
            self.losses_dict = defaultdict(list)

        self.iterations = {'train': 0, 'test': 0}

        self.embedding_step = 0

    def log_data(self, input_dict, loss_dict, split='train'):
        # First log losses
        for key in loss_dict:
            if self.tensorboard:
                self.writer.add_scalar(key + f'_{split}', loss_dict[key].cpu().item(),
                                       self.iterations[split])
            else:
                self.losses_dict[key + f'_{split}'].append(loss_dict[key].cpu().item())

        self.iterations[split] += 1

    def log_embedding(self, contexts, labels, means, variances):
        #         contexts (500*4, 3)
        #         labels (500*4, )
        #         means (500*4, )
        #         variances (500*4, )

        # Plot by distribution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = np.array([
            'indianred',
            'forestgreen',
            'gold',
            'cornflowerblue'
        ])

        idx_dict = {0: 'exponential', 1: 'gaussian', 2: 'uniform', 3: 'laplace'}

        for i, color in enumerate(colors):
            ## Take indices of the labels for each distribution
            idxs = (labels == i)
            ## Plot the context means
            ax.scatter(contexts[idxs, 0], contexts[idxs, 1], contexts[idxs, 2],
                color=color, label=idx_dict[i])

        ax.legend(loc='upper left')

        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                        right=False, left=False, labelleft=False)
        plt.tight_layout()

        plt.savefig(f'{self.log_dir}/{self.experiment_name}/{self.embedding_step}_distribution.png')
      
        plt.close()

        # Plot by means and variances 
        for name, statistic in zip(['mean', 'variance'], [means, variances]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ## Plot all context means, and color them according to mean / variance.
            cax = ax.scatter(contexts[:, 0], contexts[:, 1], contexts[:, 2],
                             c=statistic)
            fig.colorbar(cax)

            plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                            right='off', left='off', labelleft='off')
            plt.tight_layout()

            plt.savefig(f'{self.log_dir}/{self.experiment_name}/{self.embedding_step}_{name}.png')

            plt.close()

        self.embedding_step += 1

    def save_model(self, model, model_name):
        torch.save(model.state_dict(),
                   f'{self.save_dir}/{self.experiment_name}/{model_name}')

        if not self.tensorboard:
            with open(f'{self.log_dir}/{self.experiment_name}/losses', 'wb') as f:
                pickle.dump(self.losses_dict, f)
