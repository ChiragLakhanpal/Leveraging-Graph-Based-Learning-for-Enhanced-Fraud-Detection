import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        """
        Defines a single layer of a Heterogeneous Relational Graph Convolutional Network (HeteroRGCN).

        :param in_size: The size of the input features.
        :param out_size: The size of the output features.
        :param etypes: A list of edge types in the heterogeneous graph.

        """
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name: nn.Linear(in_size, out_size) for name in etypes})
    def forward(self, G, feat_dict):
        """
        Performs the forward pass of the HeteroRGCN layer.

        :param G: The input heterogeneous graph.
        :param feat_dict: A dictionary containing input node features for each node type.
        :returns: A dictionary containing the updated node features for each node type.

        """
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}

class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers, target_node):
        """
            Defines a Heterogeneous Relational Graph Convolutional Network (HeteroRGCN).

            :param ntype_dict: A dictionary containing the number of nodes for each node type.
            :param etypes: A list of edge types in the heterogeneous graph.
            :param in_size: The size of the input features.
            :param hidden_size: The size of the hidden features.
            :param out_size: The size of the output features.
            :param n_layers: The number of HeteroRGCN layers.
            :param target_node: The target node type for which predictions are made.

        """
        super(HeteroRGCN, self).__init__()
        embed_dict = {ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                      for ntype, num_nodes in ntype_dict.items() if ntype != target_node}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(in_size, hidden_size, etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))
        self.lin = nn.Linear(hidden_size,out_size)

    def forward(self, g, features, target_node):
        """
        Performs the forward pass of the HeteroRGCN model.

        :params g : The input heterogeneous graph.
        :params features : The input features for the target node type.
        :param target_node: The target node type for which predictions are made.
        :returns torch.Tensor: The output predictions for the target node type.

        """
        x_dict = {ntype: emb for ntype, emb in self.embed.items()}
        x_dict[target_node] = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}
            x_dict = layer(g, x_dict)
        return self.lin(x_dict[target_node])

