import torch.nn as nn
import torch.nn.functional as F

class DecoderNet(nn.Module):
    def __init__(self, I, H1, H2, O):
        super(DecoderNet, self).__init__()
        self.fc1 = nn.Linear(I, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, O) # predicts probability (after being passed through sigmoid)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = F.sigmoid(self.fc3(x))
        return prob

class StateReducer(nn.Module):
    def __init__(self, I, H1, O):
        super(StateReducer, self).__init__()
        self.fc1 = nn.Linear(I, H1)
        self.fc2 = nn.Linear(H1, O)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        new_state = F.sigmoid(self.fc2(x)) # squash state to between 0 and 1
        return new_state