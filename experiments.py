import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os

from tgn import *
from modules import *
from train import *

class BasicTGNExperiment():
    # This class handles the experiment setup and model training and evaluation (at a high level)
    def __init__(self,
                exp_name, 
                data,
                B = 200, # batch size
                Ss = 30, # state/memory size
                init_learning_rate = 1e-2,
                min_learning_rate = 1e-5,
                decay_rate = 0.9995,
                tb_dir = 'runs/',
                write_every_loss = False
            ):
        super(BasicTGNExperiment).__init__()

        self.writer = SummaryWriter(os.path.join(tb_dir, exp_name))
        self.write_every_loss = write_every_loss

        self.Ss = Ss
        self.B = B
        self.data = data
        self.V = data[:,0] # node IDs vector
        self.N = max(data[:,0]) + 1 # number of nodes
        self.E = data.shape[0] # number of edges
        
        # Set up model (a basic Temporal Graph Network)
        self.tgn = TemporalGraphNetwork(self.N, self.Ss, [
            IdentityMessageEncoder,
            MeanMessageAggregator,
            MLPStateReducer,
            IdentityEmbedding,
            DecoderNet,
            RawMessageComposer
        ])

        # Set up a basic optimizer
        def lr_decay(global_step,
                init_learning_rate = init_learning_rate,
                min_learning_rate = min_learning_rate,
                decay_rate = decay_rate):
            lr = ((init_learning_rate - min_learning_rate) *
                pow(decay_rate, global_step) +
                min_learning_rate)
            return lr
        lr0 = lr_decay(0)
        self.optimizer = torch.optim.Adam(list(self.tgn.modules[2].parameters()) + list(self.tgn.modules[4].parameters()), lr=lr0) # mem and dec functions are learnable
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: lr_decay(step)/lr0)
        
        # Use BCE for loss
        self.loss_fn = nn.BCELoss()

        # train in a self-supervised manner
        self.tgn_trainer = SelfSupervisedTGNTrainer(self.tgn, self.optimizer, self.scheduler, self.loss_fn, B=B)

    def train_self_supervised(self):
        losses = []
        for batch_num, loss in self.tgn_trainer.train(self.data):
            losses.append(float(loss))
            if self.write_every_loss:
                self.writer.add_scalar('Loss/self-super', float(loss), batch_num)
        
        # compute some statistics
        AVGN = 10
        self._avgs = np.convolve(losses, np.ones(AVGN) / AVGN, mode='valid')
        self._losses = losses
        self._best_avg_loss = min(self._avgs)
        self._final_avg_loss = self._avgs[-1]

        print(self._best_avg_loss, self._final_avg_loss)