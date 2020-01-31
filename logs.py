import os
import pickle
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from collections import defaultdict

def get_logger(opts):
    return Logger(opts)


class Logger:
    """Class for logging experiments."""
    def __init__(self, opts):
        self.tensorboard = opts.tensorboard

        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y_%H:%M:%S")
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

    def log_data(self, input_dict, loss_dict, split='train'):
        # First log losses
        for key in loss_dict:
            if self.tensorboard:
                self.writer.add_scalar(key + f'_{split}', loss_dict[key].cpu().item(),
                                       self.iterations[split])
            else:
                self.losses_dict[key + f'_{split}'].append(loss_dict[key].cpu().item())

        self.iterations[split] += 1

        if split == 'test' and not self.tensorboard:
            with open(f'{self.log_dir}/{self.experiment_name}', 'wb') as f:
                pickle.dump(self.losses_dict, f)

    @staticmethod
    def save_model(model, model_name):
        torch.save(model.state_dict(),
                   f'{self.save_dir}/{self.experiment_name}/{model_name}')
