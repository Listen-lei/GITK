import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):
        super(KAN_linear, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # fouriercoeffs shape: (2, outdim, inputdim, gridsize)
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) / (np.sqrt(inputdim) * np.sqrt(self.gridsize))
        )
        if self.addbias:
            # bias shape in original was (1, outdim)
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        # x can be (..., inputdim)
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)  # (B, inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)  # (B,1,inputdim,1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # reshape to (1, B, inputdim, gridsize)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))

        y = torch.einsum("dbik,djik->bj", torch.cat([c, s], dim=0), self.fouriercoeffs)
        if self.addbias:
            y = y + self.bias  # broadcast (B, outdim) + (1, outdim)
        y = y.view(outshape)
        return y


class NaiveFourierKANConv(MessagePassing):
    """
    PyG MessagePassing version of the DGL NaiveFourierKANLayer.
    Aggregation = sum (equivalent to DGL fn.sum).
    Message computation is Fourier-KAN transform on source node features.
    """
    def __init__(self, in_channels, out_channels, gridsize, addbias=True):
        super(NaiveFourierKANConv, self).__init__(aggr='add')  # sum aggregation
        self.gridsize = gridsize
        self.addbias = addbias
        self.in_channels = in_channels
        self.out_channels = out_channels

        # fouriercoeffs shape: (2, out_channels, in_channels, gridsize)
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, out_channels, in_channels, gridsize) / (np.sqrt(in_channels) * np.sqrt(gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        """
        x: (N, in_channels)
        edge_index: (2, E)
        returns: (N, out_channels)
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        x_j: source node features for each edge, shape (E, in_channels)
        compute Fourier features and einsum with fouriercoeffs -> (E, out_channels)
        """
        src_feat = x_j  # (E, in_channels)
        # create k: (1,1,1,gridsize)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=src_feat.device), (1, 1, 1, self.gridsize))
        src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)  # (E,1,in_channels,1)
        cos_kx = torch.cos(k * src_rshp)
        sin_kx = torch.sin(k * src_rshp)

        # reshape to (1, E, in_channels, gridsize)
        cos_kx = torch.reshape(cos_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
        sin_kx = torch.reshape(sin_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))

        # concat to (2, E, in_channels, gridsize)
        stacked = torch.cat([cos_kx, sin_kx], dim=0)

        # einsum with fouriercoeffs (2, out_channels, in_channels, gridsize)
        # yields (E, out_channels)
        m = torch.einsum("dbik,djik->bj", stacked, self.fouriercoeffs)

        return m  # will be aggregated (summed) by propagate

    def update(self, aggr_out):
        """
        aggr_out: (N, out_channels)
        add bias if required
        """
        if self.addbias:
            aggr_out = aggr_out + self.bias
        return aggr_out


class KA_GNN_two(nn.Module):
    """
    PyG version of KA_GNN_two (keeps computation logic identical).
    forward signature: (x, edge_index, batch)
    """
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # initial KAN linear (maps in_feat -> hidden_feat)
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        # stack num_layers-1 NaiveFourierKANConv
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANConv(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        # final linear readout as in original
        self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)

        # pooling choices
        self.pooling = pooling.lower() if isinstance(pooling, str) else pooling

        # Readout sequence: [linear_1, Sigmoid]
        layers_kan = [
            self.linear_1,
            nn.Sigmoid()
        ]
        self.Readout = nn.Sequential(*layers_kan)

    def forward(self, x, edge_index, batch):
        """
        x: (N, in_feat)
        edge_index: (2, E)
        batch: (N,) batch vector for pooling
        """
        # initial KAN linear mapping
        h = self.kan_line(x)  # (N, hidden_feat)

        # for each layer: compute message m and do leaky_relu(m + h)
        for layer in self.layers:
            m = layer(h, edge_index)  # (N, hidden_feat) via sum aggregation
            h = F.leaky_relu(m + h)

        # pooling
        if self.pooling == 'avg' or self.pooling == 'mean':
            y = global_mean_pool(h, batch)
        elif self.pooling == 'max':
            y = global_max_pool(h, batch)
        elif self.pooling == 'sum' or self.pooling == 'add':
            y = global_add_pool(h, batch)
        else:
            raise ValueError(f"No pooling found: {self.pooling}")

        out = self.Readout(y)
        return out

    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()


class KA_GNN(nn.Module):
    """
    PyG version of KA_GNN (keeps computation logic identical).
    forward signature: (x, edge_index, batch)
    """
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling

        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANConv(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        # as in original: a couple of KAN_line layers for readout
        self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        self.linear = KAN_linear(hidden_feat, out, grid_feat, addbias=use_bias)

        # Readout sequence: linear_1 -> leaky_relu -> linear_2 -> Sigmoid
        layers_kan = [
            self.linear_1,
            self.leaky_relu,
            self.linear_2,
            nn.Sigmoid()
        ]
        self.Readout = nn.Sequential(*layers_kan)

    def forward(self, x, edge_index, batch):
        """
        x: (N, in_feat)
        edge_index: (2, E)
        batch: (N,) batch vector for pooling
        """
        h = self.kan_line(x)  # (N, hidden_feat)

        # apply each KAN conv layer; preserve original control flow semantics
        for i, layer in enumerate(self.layers):
            # original DGL code had a conditional but in effect used layer(g, h) for each
            # here we use layer(h, edge_index) consistently (sum aggregation)
            h = layer(h, edge_index)

        # pooling
        if isinstance(self.pooling, str) and self.pooling.lower() in ['avg', 'mean']:
            y = global_mean_pool(h, batch)
        elif isinstance(self.pooling, str) and self.pooling.lower() == 'max':
            y = global_max_pool(h, batch)
        elif isinstance(self.pooling, str) and self.pooling.lower() in ['sum', 'add']:
            y = global_add_pool(h, batch)
        else:
            y = h
            # print('no pooling')

        out = self.Readout(y)
        return out

    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()