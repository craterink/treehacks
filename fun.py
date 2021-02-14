import torch
import numpy as np

class TGNHelpers():
    def __init__(self, N, B, red, node_neighbors, most_recently_seen=None):
        self.N = N
        self.B = B
        self.red = red
        self.node_neighbors = node_neighbors
        self.most_recently_seen = most_recently_seen

    # non-learnable functions
    def raw_msgs(self, batch, S, source=True):
        # batch is (B,3) - rows are (source id, dest id, timestamp)
        # Returns: (B,Ms) raw_msg matrix, each row representing (si, sj, dt) or (sj, si, dt) depending on SOURCE
        t = batch[:,2]
        obj_nodes, nb_nodes = (batch[:,0], batch[:,1]) if source else (batch[:,1], batch[:,0])
        rmsgs = torch.cat([S[obj_nodes, :], S[nb_nodes, :], t.unsqueeze(-1), obj_nodes.unsqueeze(-1)], axis=1)
        return rmsgs

    def get_khop_neighbors(self, k=1):
        # returns (NxN) neighbors matrix. matrix[i,j] is set if node j is in node i's k-hop neighborhood
        # for now, k = 1 is the only option implemented for now
        assert k == 1
        return self.node_neighbors

    def update_neighbors(self, batch_data):
        self.node_neighbors[batch_data[:,0], batch_data[:,1]] = 1

    def update_most_recently_seen(self, batch_data):
        self.most_recently_seen[batch_data[:,0]] = batch_data[:,2] # takes advantage of batches sorted by timestamp -- latest timestamp overwrites earlier ones

    def negative_sample(self, pos_edges):
        # negatively sample Bneg <= B edges (of N total nodes)
        # where each edge is not in pos_edges
        # Returns: (Bneg(<=B)x2)
        pos_inds = pos_edges[:,0] + self.N*pos_edges[:,1]
        cands = np.random.choice(self.N*self.N, size=(self.B,))
        ss = np.searchsorted(pos_inds, cands)
        pos_mask = torch.tensor(cands) == np.take(pos_inds, ss, mode='clip')
        neg_inds = torch.tensor(cands[~pos_mask])
        neg_edges = np.concatenate([(neg_inds % int(self.N)).unsqueeze(-1), (neg_inds // self.N).unsqueeze(-1)], axis=1)
        return neg_edges

    # def update_last_interactions(batch_data):
    #     node_last_update[batch_data[:,0], batch_data[:,1]] = 1

    # learnable functions, for now quite simple and mostly non-learnable
    def msg(self, rm, curr_time):
        # takes (RxMs) raw messages (si,sj,dt) as input and spits out messages (RxMs)
        # for now, let's just return identity (later I can try to append an indicator for whether the node is the message source)
        # but keep in mind that I should transform t => dt by taking curr_time - ti for every i
        # and ALSO remove node idx entry
        # Returns: msg, (Msx1) where Ms = 2*S + 1
        out = torch.clone(rm)
        out[:,-2] = curr_time - out[:,-2]
        return out[:,:-1]

    def agg_mean_dummy(self, M, nodes):
        # M is (RxMs), where R is # of messages in raw msg store
        # nodes is (Rx1), representing node idx for each message
        # This message aggregator simply averages the incoming messages.
        # Returns: (N,Ms) matrix corresponding to average message (row value) for node I (row index)
        selector_matrix = torch.zeros((self.N,M.shape[0]))
        selector_matrix[nodes, torch.arange(M.shape[0])] = 1 # set i,j to 1 if node i corresponds to message j
        agg = torch.matmul(selector_matrix, M) / torch.sum(selector_matrix, axis=1).unsqueeze(-1)
        agg[torch.isnan(agg)] = 0 # messages we haven't seen yet stay at zero
        return agg

    def update_mem_matmul(self, prev_state, agg_msg):
        # prev_state is (NxSs)
        # agg_msg is (NxMs)
        # Returns: state reducer MLP takes (Nx(Ss+Ms)) => (Nx(Ss))
        inp_vector = torch.cat([prev_state, agg_msg], axis=1)
        output = self.red(inp_vector)
        return output

    def embed_identity(self, node_state, neighbors):
        # node_state is ((NxSs), neighbors is (NxN) k-hop neighbors)
        # for identity embedding, just return the node's state
        return node_state