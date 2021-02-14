
from helpers import *

class SelfSupervisedTGNTrainer():
    def __init__(self,
            model,
            optimizer,
            scheduler,
            loss_fn,
            B=200):
        super(SelfSupervisedTGNTrainer).__init__()
        self.B = B
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

    def train(self, data):
        # training loop from https://arxiv.org/pdf/2006.10637
        # TODO: Make sure I'm computing+updating neighbors' embeddings as well (I think I may be actually)
        # TODO: double check whether I need to keep a temp copy of the state so I don't use too-recent state information
        # TODO: Add other possible ways to train (e.g., witholding a set of validation nodes with which to evaluation the learned model)
        tgn = self.model
        n_iter = -1
        for batch_start_idx in range(0, data.shape[0], self.B):
            self.optimizer.zero_grad()
            batch_data = data[batch_start_idx:batch_start_idx+self.B,:] # get batch data
            curr_time = min(batch_data[:,2])
            n_iter += 1

            ### FORMULATE PREDICTION TASK ###
            batch_edges = batch_data[:,0:2]
            neg_edges = torch.tensor(negative_sample(batch_edges, tgn.N, self.B)) # Bneg(<=B) x 2
            train_edges = torch.cat([batch_edges, neg_edges], axis=0) # Btrain (>B) x 2
            targets = torch.cat([torch.ones((self.B,)), torch.zeros((train_edges.shape[0] - self.B,))]).unsqueeze(-1) # predict 1 for batch edges, zero for negatively-sampled edges

            ### PREDICT, COMPUTE LOSS, AND UPDATE USING THESE EMBEDDINGS ###
            edge_probs = tgn.forward(train_edges, curr_time)
            loss = self.loss_fn(edge_probs, targets) # (6)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            ### UPDATE MSG STORE FOR NEXT ITERATION ###
            tgn.update_state(batch_data)

            ### YIELD SOME INFO TO CALLER
            yield n_iter, loss
