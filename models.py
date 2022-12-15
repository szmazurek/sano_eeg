import torch

from torch.nn import functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN,  GConvGRU, A3TGCN
from torch_geometric_temporal.nn.attention import STConv

class A3TRecurrent(torch.nn.Module):
    def __init__(self,node_features, timestep,sfreq):
        super(A3TRecurrent, self).__init__()
        out_features = 64
        self.recurrent_1 =  A3TGCN(node_features,out_features,timestep*sfreq)
        self.fc1 = torch.nn.Linear(1152, 1)
        self.flatten = torch.nn.Flatten(start_dim=0)
    def forward(self, x, edge_index, edge_weight):
        x = F.normalize(x,dim=1)
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.flatten(h)  
        h = self.fc1(h)
        return h

class ModelStconv(torch.nn.Module):
    def __init__(self, timestep,sfreq, n_nodes=18):
        super(RecurrentGCN, self).__init__()
        out_features = 64
        self.recurrent_1 =  STConv(
            num_nodes=n_nodes,
            in_channels = 1,
            hidden_channels=64,
            out_channels=out_features,
            kernel_size=3,
            K=1)
        self.fc1 = torch.nn.Linear(1469952, 1)
        self.flatten = torch.nn.Flatten(start_dim=0)
    def forward(self, x, edge_index, edge_weight):
        x = F.normalize(x,dim=1)
        x = x[None,:]
        x = x.reshape([x.shape[0],x.shape[3],x.shape[1],x.shape[2]])
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.flatten(h)
        h = self.fc1(h)
        return h

class RecurrentGCN(torch.nn.Module):
    def __init__(self, timestep,sfreq, n_nodes=18):
        super(RecurrentGCN, self).__init__()
        out_features = 64
        self.recurrent_1 = DCRNN(timestep*sfreq,out_features,1)
        self.fc1 = torch.nn.Linear(out_features*n_nodes, 1)
        self.flatten = torch.nn.Flatten(start_dim=0)
    def forward(self, x, edge_index, edge_weight):
        x = torch.squeeze(x)
        x = F.normalize(x,dim=1)
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.flatten(h)
        h = self.fc1(h)
        return h