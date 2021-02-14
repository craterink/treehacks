import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os

from fun import TGNHelpers
from models import *

class Experiment():
    def __init__(self,
        exp_name, 
        data,
        B = 200, # batch size
        Ss = 30, # state/memory size
        init_learning_rate = 1e-2,
        min_learning_rate = 1e-5,
        decay_rate = 0.9995,
        tb_dir = 'runs/'
    ):
        self.writer = SummaryWriter(os.path.join(tb_dir, exp_name))
        ## Constants ##
        self.data = data
        self.V = data[:,0] # node IDs vector
        self.N = max(data[:,0]) + 1 # number of nodes
        self.E = data.shape[0] # number of edges
        self.B = B
        self.Ss = Ss
        self.Zs = self.Ss # embedding size
        self.Rs = 2*self.Ss + 2 # raw message s|ize (Si, Sj, t, i)
        self.Ms = 2*self.Ss + 1 # message size (Si, Sj,dt)

        ## Weights and state ##
        self.dec = DecoderNet(2*self.Zs, 2*self.Zs, self.Zs, 1) # decodes embedding pairs => prob of link appearing
        self.state_reducer = StateReducer(self.Ss+self.Ms, self.Ss+self.Ms, self.Ss) # takes two prev_states + dt => new_state
        self.S = torch.rand((self.N,self.Ss)) # initialize state information
        self.Z = torch.rand((self.N,self.Zs)) # initialize embedding information
        self.node_neighbors = torch.zeros((self.N,self.N), dtype=torch.int8) # node_neighbors[i,j] is True/1 iff node i has messaged node j before
        self.most_recently_seen = torch.zeros((self.N,), dtype=torch.int64) # most_recently_seen[i] is the timestamp corresponding to the most recent interaction involving node i
        self.tgn = TGNHelpers(self.N, self.B, self.state_reducer, self.node_neighbors, self.most_recently_seen)
        
        ## Optimizer ##
        self.LossFn = nn.BCELoss()
        def lr_decay(global_step,
            init_learning_rate = init_learning_rate,
            min_learning_rate = min_learning_rate,
            decay_rate = decay_rate):
            lr = ((init_learning_rate - min_learning_rate) *
                pow(decay_rate, global_step) +
                min_learning_rate)
            return lr
        lr0 = lr_decay(0)
        self.optimizer = torch.optim.Adam(list(self.tgn.red.parameters()) + list(self.dec.parameters()), lr=lr0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: lr_decay(step)/lr0)

    def train_self_supervised(self):
        # training loop from https://arxiv.org/pdf/2006.10637
        self._raw_msg_store = torch.zeros((self.N,self.Rs)) # N x Rs in our simple case
        tgn = self.tgn
        losses = []
        # rand_losses = []
        n_iter = -1
        for batch_start_idx in range(0, self.E, self.B):
            self.optimizer.zero_grad()
            batch_data = self.data[batch_start_idx:batch_start_idx+self.B,:] # get batch data 
            batch_srcs, batch_dsts = batch_data[:,0], batch_data[:,1]
            curr_time = min(batch_data[:,2])
            n_iter += 1
            # print('batch_data', batch_data.shape)

            ### AGGREGATE MESSAGES FROM RAW STORE (1 -> 6 in the diagram) ###
            # print('raw_msgs', raw_msg_store.shape)
            nodes = self._raw_msg_store[:,-1].long() # should just be a list of all the node ids in order (like range(N))
            msgs = tgn.msg(self._raw_msg_store, curr_time) # (1)
            # print('msgs', msgs.shape)
            agg = tgn.agg_mean_dummy(msgs, nodes) # (2) # N x Ms
            # print('agg', agg.shape)

            ### UPDATE STATE AND EMBEDDINGS USING AGG MSGS ###
            # TODO: Make sure I'm computing+updating neighbors' embeddings as well (I think I may be actually)
            # TODO: double check whether I need to keep a temp copy of the state so I don't use too-recent state information
            # print('S_prev',S.shape)
            S_t = tgn.update_mem_matmul(self.S, agg) # (3) 
            # print('S_t', S.shape)
            Z = torch.autograd.Variable(tgn.embed_identity(S_t, tgn.get_khop_neighbors(k=1))) # (4)
            # print('Z', Z.shape)

            ### FORMULATE PREDICTION TASK
            batch_edges = batch_data[:,0:2]
            neg_edges = torch.tensor(tgn.negative_sample(batch_edges)) # Bneg(<=B) x 2
            train_edges = torch.cat([batch_edges, neg_edges], axis=0) # Btrain (>B) x 2
            Zpairs = torch.cat([Z[train_edges[:,0],:],Z[train_edges[:,1],:]], axis=1)
            predictions = self.dec(Zpairs) # (5)
            targets = torch.cat([torch.ones((self.B,)), torch.zeros((predictions.shape[0] - self.B,))]).unsqueeze(-1) # predict 1 for batch edges, zero for negatively-sampled edges

            ### PREDICT, COMPUTE LOSS, AND UPDATE USING THESE EMBEDDINGS ###
            loss = self.LossFn(predictions, targets) # (6)
            # rand_loss = LossFn(torch.tensor(np.random.uniform(size=targets.shape)).float(),targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses.append(float(loss))
            #self.writer.add_scalar('Loss/self-super', float(loss), n_iter)
            # rand_losses.append(float(rand_loss))
            
            ### UPDATE MSG STORE FOR NEXT ITERATION ###
            self.S = S_t
            rmsgs_src = tgn.raw_msgs(batch_data, self.S, source=True) # B x Ms
            rmsgs_dst = tgn.raw_msgs(batch_data, self.S, source=False) # B x Ms
            all_rmsgs = torch.cat([rmsgs_src, rmsgs_dst], axis=0) # 2B x Ms
            self._raw_msg_store[all_rmsgs[:,-1].long(),:] = all_rmsgs # update raw_msgs at obj_node, keeping only most recent msg
            
            ### UPDATE NEIGHBORS GRAPH ###
            tgn.update_neighbors(batch_data) # update neighbors information

        AVGN = 10
        self._avgs = np.convolve(losses, np.ones(AVGN) / AVGN, mode='valid')
        self._losses = losses
        self._best_avg_loss = min(self._avgs)
        self._final_avg_loss = self._avgs[-1]

    # (TODO: MAYBE LATER: KEEP MORE RAW MSGS)
    # keep only the raw msgs that correspond to the most recent before this batch
    # recent_raw_lookup = most_recently_seen[raw_msg_store[:,-1].long()]
    # raw_msg_store = raw_msg_store[raw_msg_store[:,-2] == recent_raw_lookup,:]
    #  # append the raw msgs from this batch
    # raw_msg_store = torch.cat([raw_msg_store, all_rmsgs], axis=0)
    # update_most_recently_seen(batch_data) # update recency information