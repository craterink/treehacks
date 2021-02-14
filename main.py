import numpy as np  
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from collections import defaultdict
import itertools
import sys
import argparse

from experiments import BasicTGNExperiment

def perform_experiment(data, B, Ss, lr0):
        # try experiment
        print('Training B={}, Ss={}, lr0={}'.format(B, Ss, lr0))
        exp = BasicTGNExperiment(
            'exp_B={}, Ss={}, lr0={}'.format(B, Ss, lr0),
            data,
            B=B,
            Ss=Ss,
            init_learning_rate=lr0
        )
        exp.train_self_supervised()
        return {
            'B' : B,
            'Ss' : Ss,
            'lr0' : lr0,
            'best_loss' : exp._best_avg_loss,
            'final_loss' : exp._final_avg_loss,
            'avg_loss' : exp._avgs
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-graph", "-i", type=str, help="Path to input graph data. E.g., 'data/CollegeMsg.txt'.")
    parser.add_argument("--output", "-o", type=str, help="Desired output file path. E.g., 'result.pkl'.", default='results.pkl')
    parser.add_argument("--parallel", "-p", action='store_true', help="Set this flag to spawn the experiments in parallel processes.", default=False)
    args = parser.parse_args()

    data = np.genfromtxt(args.input_graph)
    data = data.astype(int) # data is all integral
    data[:,2] -= np.min(data[:,2]) # zero-index timestamps
    data[:,0] -= 1 # zero-index nodes
    data[:,2] = data[:,2] // 60 # batch data by minute
    data = data[np.argsort(data[:,2]), :] # sort by timestamp
    data = torch.tensor(data) # move to tensor

    params = {
        'B' : [50, 100],#, 100, 200, 400],
        'Ss' : [15, 30],#[8, 15, 30, 60],
        'lr0' :  [5e-2],
    }

    if args.parallel:
        import multiprocessing as mp
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(perform_experiment, [(data, B, Ss, lr0) for B, Ss, lr0 in itertools.product(params['B'], params['Ss'], params['lr0'])])
    else:
        results = []
        for B, Ss, lr0 in itertools.product(params['B'], params['Ss'], params['lr0']):
            results.append(perform_experiment(data, B, Ss, lr0))

    import pickle
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)