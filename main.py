import numpy as np  
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from collections import defaultdict
import itertools

from experiments import Experiment

def parallel_train(data, B, Ss, lr0):
        # try experiment
        print('Training B={}, Ss={}, lr0={}'.format(B, Ss, lr0))
        exp = Experiment(
            'exp_B={}, Ss={}, lr0={}'.format(B, Ss, lr0),
            data,
            B=B,
            Ss=Ss,
            init_learning_rate=lr0
        )
        exp.train_self_supervised()

        # record results
        return {
            'B' : B,
            'Ss' : Ss,
            'lr0' : lr0,
            'best_loss' : exp._best_avg_loss,
            'final_loss' : exp._final_avg_loss,
            'avg_loss' : exp._avgs
        }

if __name__ == "__main__":
    data = np.genfromtxt('data/CollegeMsg.txt')
    data = data.astype(int) # data is all integral
    data[:,2] -= np.min(data[:,2]) # zero-index timestamps
    data[:,0] -= 1 # zero-index nodes
    data[:,2] = data[:,2] // 60 # batch data by minute
    data = data[np.argsort(data[:,2]), :] # sort by timestamp
    data = torch.tensor(data) # move to tensor

    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    params = {
        'B' : [25, 50, 100, 200, 400],
        'Ss' : [8, 15, 30, 60],
        'lr0' :  [5e-2, 1e-1],
    }

    # results = {}
    # for B, Ss, lr0 in itertools.product(params['B'], params['Ss'], params['lr0']):
    #     results[(B, Ss, lr0)] = parallel_train(B, Ss, lr0)

    results = pool.starmap(parallel_train, [(data, B, Ss, lr0) for B, Ss, lr0 in itertools.product(params['B'], params['Ss'], params['lr0'])])

    import pickle
    with open('result.pickle', 'wb') as f:
        pickle.dump(results, f)