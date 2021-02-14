import torch
import torch.nn as nn

class TemporalGraphNetwork(nn.Module):
    def __init__(self, 
            N,
            Ss,
            layers):
        # N is number of nodes, layers is [msg, agg, mem, emb, dec, loss_fn, raw]
        super(TemporalGraphNetwork).__init__()
        self.N = N
        
        ## Shape constants
        self.Ss = Ss
        self.Zs = self.Ss # embedding size
        self.Rs = 2*self.Ss + 2 # raw message s|ize (Si, Sj, t, i)
        self.Ms = 2*self.Ss + 1 # message size (Si, Sj,dt)

        ## TGN internal state
        self.S = torch.rand((self.N,self.Ss)) # initialize state information
        self.Z = torch.rand((self.N,self.Zs)) # initialize embedding information
        self.raw_msg_store = torch.zeros((self.N,self.Rs)) # N x Rs in our simple case
        self.node_neighbors = torch.zeros((self.N,self.N), dtype=torch.int8) # node_neighbors[i,j] is True/1 iff node i has messaged node j before

        ## Set up graph as in paper
        self.modules = [layer_module(self) for layer_module in layers]
        self.msg = self.modules[0].forward # 1
        self.agg = self.modules[1].forward # 2
        self.mem = self.modules[2].forward # 3
        self.emb = self.modules[3].forward # 4
        self.dec = self.modules[4].forward # 5
        # this class does not implement loss function
        self.raw = self.modules[5].forward # 7 ***

    def _get_khop_neighbors(self, k=1):
        # returns (NxN) neighbors matrix. matrix[i,j] is set if node j is in node i's k-hop neighborhood
        # for now, k = 1 is the only option implemented for now
        assert k == 1
        return self.node_neighbors

    def _update_neighbors(self, edges):
        self.node_neighbors[edges[:,0], edges[:,1]] = 1

    def forward(self, batch_data, curr_time, node_features=None, edge_features=None):
        # Forward pass
        edges = batch_data[:,:2]
        raw_nodes = self.raw_msg_store[:,-1].long()

        self.next_state = self.mem(self.S, self.agg(self.msg(self.raw_msg_store, curr_time), raw_nodes))
        Z = torch.autograd.Variable(self.emb(self.next_state, self._get_khop_neighbors(k=1)))
        Z_pairs = torch.cat([Z[edges[:,0],:],Z[edges[:,1],:]], axis=1)
        edge_probs = self.dec(Z_pairs)
        return edge_probs

    def update_state(self, batch_data):
        # update node states
        self.S = self.next_state
        
        # update raw message store
        rmsgs_src = self.raw(batch_data, self.S, source=True) # B x Ms
        rmsgs_dst = self.raw(batch_data, self.S, source=False) # B x Ms
        all_rmsgs = torch.cat([rmsgs_src, rmsgs_dst], axis=0) # 2B x Ms
        self.raw_msg_store[all_rmsgs[:,-1].long(),:] = all_rmsgs # update raw_msgs at obj_node, keeping only most recent msg
        
        # update neighbors information
        self._update_neighbors(batch_data)