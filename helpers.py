import torch
import numpy as np

def negative_sample(pos_edges, N, B):
    # negatively sample Bneg <= B edges (of N total nodes)
    # where each edge is not in pos_edges
    # Returns: (Bneg(<=B)x2)
    pos_inds = pos_edges[:,0] + N*pos_edges[:,1]
    cands = np.random.choice(N*N, size=(B,))
    ss = np.searchsorted(pos_inds, cands)
    pos_mask = torch.tensor(cands) == np.take(pos_inds, ss, mode='clip')
    neg_inds = torch.tensor(cands[~pos_mask])
    neg_edges = np.concatenate([(neg_inds % int(N)).unsqueeze(-1), (neg_inds // N).unsqueeze(-1)], axis=1)
    return neg_edges