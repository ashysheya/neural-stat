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
from torchvision.utils import make_grid
from PIL import Image

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
        self.batch_size = opts.batch_size

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
            idxs = (labels == i)
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
            with open(f'{self.save_dir}/{self.experiment_name}/losses', 'wb') as f:
                pickle.dump(self.losses_dict, f)


    def log_image(self, output_data, split):
        data_gen = output_data['proba_x'].cpu()
        nrows = output_data['train_data'].size()[1]
        data_real = output_data['train_data'].cpu().view_as(data_gen)

        data_gen = make_grid(data_gen, nrow=nrows)
        data_real = make_grid(data_real, nrow=nrows)

        if self.tensorboard:
            self.writer.add_image(f'{split}_gen', data_gen, self.embedding_step)
            self.writer.add_image(f'{split}_real', data_real, self.embedding_step)

        else:
            im = Image.fromarray(np.uint8(data_gen.transpose((1, 2, 0))*255))
            im.save(f'{self.log_dir}/{self.experiment_name}/{self.embedding_step}_gen_{split}.png')
            im = Image.fromarray(np.uint8(data_real.transpose((1, 2, 0))*255))
            im.save(f'{self.log_dir}/{self.experiment_name}/{self.embedding_step}_real_{split}.png')
        
        if split == 'test':    
            self.embedding_step += 1


    # https://github.com/conormdurkan/neural-statistician/blob/master/spatial/spatialplot.py
    def grid(self, inputs, samples, summaries=None, ncols=10, mode = 'test'):

        inputs = inputs.data.cpu().numpy()
        samples = samples.view(-1, 50, 2).data.cpu().numpy()
        if summaries is not None:
            summaries = summaries.data.cpu().numpy()
        fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols, 2))

        def plot_single(ax, points, s, color):
            ax.scatter(points[:, 0], points[:, 1], s=s,  color=color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, 27])
            ax.set_ylim([0, 27])
            ax.set_aspect('equal', adjustable='box')


        if inputs.shape[0] > 5:
            for i in range(ncols):
                # fill one column of subplots per loop iteration
                plot_single(axs[0, i], inputs[i], s=5, color='C0')
                plot_single(axs[1, i], samples[i], s=5, color='C1')

                if summaries is not None:
                	### TO DO: 6 point summary for inputs on the first axs!
                    plot_single(axs[1, i], summaries[i], s=10, color='C2')

            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.tight_layout()

        fig.savefig(f'{self.log_dir}/{self.experiment_name}/{self.embedding_step}_gen_{mode}.png')
        self.embedding_step += 1
