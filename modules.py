import torch.nn as nn
import torch.nn.functional as F
import torch

# 1 - msg
class IdentityMessageEncoder(nn.Module):
    def __init__(self, tgn):
        super(IdentityMessageEncoder, self).__init__()
    
    def forward(self, rm, curr_time):
        # takes (RxMs) raw messages (si,sj,dt) as input and spits out messages (RxMs)
        # for now, let's just return identity (later I can try to append an indicator for whether the node is the message source)
        # but keep in mind that I should transform t => dt by taking curr_time - ti for every i
        # and ALSO remove node idx entry
        # Returns: msg, (Msx1) where Ms = 2*S + 1
        ts_idx, node_id_idx = -2, -1
        out = torch.clone(rm)
        out[:,ts_idx] = curr_time - out[:,ts_idx]
        return out[:,:node_id_idx]

# 2 - agg 
class MeanMessageAggregator(nn.Module):
    def __init__(self, tgn):
        super(MeanMessageAggregator, self).__init__()
        self.N = tgn.N
    
    def forward(self, M, nodes):
        # M is (RxMs), where R is # of messages in raw msg store
        # nodes is (Rx1), representing node idx for each message
        # This message aggregator simply averages the incoming messages.
        # Returns: (N,Ms) matrix corresponding to average message (row value) for node I (row index)
        selector_matrix = torch.zeros((self.N,M.shape[0]))
        selector_matrix[nodes, torch.arange(M.shape[0])] = 1 # set i,j to 1 if node i corresponds to message j
        agg = torch.matmul(selector_matrix, M) / torch.sum(selector_matrix, axis=1).unsqueeze(-1)
        agg[torch.isnan(agg)] = 0 # messages we haven't seen yet stay at zero
        return agg

# 3 - mem
class MLPStateReducer(nn.Module):
    def __init__(self, tgn):
        super(MLPStateReducer, self).__init__()
        I, H1, O = tgn.Ss+tgn.Ms, tgn.Ss+tgn.Ms, tgn.Ss
        self.fc1 = nn.Linear(I, H1)
        self.fc2 = nn.Linear(H1, O)
    
    def forward(self, prev_state, agg_msg):
        x = torch.cat([prev_state, agg_msg], axis=1)
        x = F.relu(self.fc1(x))
        new_state = torch.sigmoid(self.fc2(x)) # squash state to between 0 and 1
        return new_state

# 4 - emb
class IdentityEmbedding(nn.Module):
    def __init__(self, tgn):
        super(IdentityEmbedding, self).__init__()
    
    def forward(self, node_state, neighbors=None):
        # node_state is ((NxSs), neighbors is (NxN) k-hop neighbors)
        # for identity embedding, just return the node's state
        return node_state

# 5 - dec
class DecoderNet(nn.Module):
    def __init__(self, tgn):
        super(DecoderNet, self).__init__()
        I, H1, H2, O = 2*tgn.Zs, 2*tgn.Zs, tgn.Zs, 1
        self.fc1 = nn.Linear(I, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, O) # predicts probability (after being passed through sigmoid)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob

# 6 - loss is defined in TGN

# 7 - raw
class RawMessageComposer(nn.Module):
    def __init__(self, tgn):
        super(RawMessageComposer, self).__init__()
    
    def forward(self, batch, S, source=True):
        # batch is (B,3) - rows are (source id, dest id, timestamp)
        # Returns: (B,Ms) raw_msg matrix, each row representing (si, sj, dt) or (sj, si, dt) depending on SOURCE
        t = batch[:,2]
        obj_nodes, nb_nodes = (batch[:,0], batch[:,1]) if source else (batch[:,1], batch[:,0])
        rmsgs = torch.cat([S[obj_nodes, :], S[nb_nodes, :], t.unsqueeze(-1), obj_nodes.unsqueeze(-1)], axis=1)
        return rmsgs


# (TODO: MAYBE LATER: KEEP MORE RAW MSGS)
# keep only the raw msgs that correspond to the most recent before this batch
# recent_raw_lookup = most_recently_seen[raw_msg_store[:,-1].long()]
# raw_msg_store = raw_msg_store[raw_msg_store[:,-2] == recent_raw_lookup,:]
#  # append the raw msgs from this batch
# raw_msg_store = torch.cat([raw_msg_store, all_rmsgs], axis=0)
# update_most_recently_seen(batch_data) # update recency information