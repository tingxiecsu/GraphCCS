# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:38:29 2022

@author: ZNDX002
"""

import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax,AttentiveFPReadout
import torch.nn.functional as F
from torch import nn 

class GCNPlain(nn.Sequential):
	def __init__(self,in_feats, hidden_feats=None, activation=None, predictor_dim=None):
		super(GCNPlain, self).__init__()
		self.input_dim_drug = 256

		self.gnn = GCN(in_feats=in_feats,
                        hidden_feats=hidden_feats,
                        activation=activation
                        )
		gnn_out_feats = self.gnn.hidden_feats[-1]
		self.readout = WeightedSumAndMax(gnn_out_feats)
		self.transform = nn.Linear((self.gnn.hidden_feats[-1] * 2), predictor_dim)

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = [1024,1024,512]
		layer_size = len(self.hidden_dims)
		dims = [self.input_dim_drug] + self.hidden_dims

		self.predictor = nn.ModuleList([nn.Linear((dims[i]), dims[i+1]) for i in range(layer_size)])
		self.properte = nn.Linear(self.hidden_dims[-1],1)

	def forward(self, v_D):
		# each encoding
		feats = v_D.ndata.pop('h') 
		node_feats = self.gnn(v_D, feats)
		graph_feats = self.readout(v_D, node_feats)
		v_f = self.transform(graph_feats)
		# concatenate and classify
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1): 
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		v_f = self.properte(v_f)
		return v_f

class LayerScale(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 > depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.scale = nn.Parameter(torch.full((dim,), init_eps))

    def forward(self, x):
        return x* self.scale

class EmbeddingLayerConcat(nn.Module):
    def __init__(self, node_in_dim, node_emb_dim, edge_in_dim=None, edge_emb_dim=None):
        super(EmbeddingLayerConcat, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_emb_dim= node_emb_dim
        self.edge_in_dim = edge_emb_dim
        self.edge_emb_dim=edge_emb_dim

        self.atom_encoder = nn.Linear(node_in_dim, node_emb_dim)
        if edge_emb_dim is not None:
            self.bond_encoder = nn.Linear(edge_in_dim, edge_emb_dim)

    def forward(self, g):
        node_feats, edge_feats= g.ndata["h"], g.edata["e"]
        node_feats = self.atom_encoder(node_feats)

        if self.edge_emb_dim is None:
            return node_feats
        else:
            edge_feats = self.bond_encoder(edge_feats)
            return  node_feats, edge_feats

class GCNLayerWithEdge(nn.Module):
    def __init__(self, in_feats, out_feats, depth,activation=None, dropout=0.,residual=True):
        super(GCNLayerWithEdge, self).__init__()

        self.activation = activation
        self.mlp = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.ls2 = LayerScale(in_feats, depth)
        self.residual=residual

    def reset_parameters(self):
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats
            g.apply_edges(fn.u_add_e('h', 'e', 'm'))

            g.edata['a'] = edge_softmax(g, g.edata['m'])
            g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                         fn.sum('x', 'm'))

            new_feats = g.ndata['m']
            new_feats = self.mlp(new_feats)
            new_feats = self.activation(new_feats)
            new_feats = self.dropout(new_feats)

            new_feats = self.ls2(new_feats) + node_feats

            return new_feats

class GraphCCS(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_feats=None, activation=F.relu,
                  dropout=0.1, gru_out_layer=2,residual=True):
        super(GraphCCS, self).__init__()

        if hidden_feats is None:
            hidden_feats = [200]*5

        in_feats = hidden_feats[0]
        n_layers = len(hidden_feats)

        activation = [activation for _ in range(n_layers)]
        dropout = [dropout for _ in range(n_layers)]

        lengths = [len(hidden_feats), len(activation),len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats ' \
                                       'activation, and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0], edge_in_dim, hidden_feats[0])
        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            depth=i
            self.gnn_layers.append(GCNLayerWithEdge(in_feats, hidden_feats[i], depth,activation[i],  dropout[i],residual=True))
            in_feats = hidden_feats[i]

        self.readout = AttentiveFPReadout(
            hidden_feats[-1], num_timesteps=gru_out_layer, dropout=dropout[-1]
        )
        #self.readout = WeightedSumAndMax(hidden_feats[-1])
        self.out = nn.Sequential(
            nn.Linear(hidden_feats[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        node_feat, edge_feat = self.embed_layer(g)
        for gnn in self.gnn_layers:
            node_feat = gnn(g, node_feat, edge_feat)
        out = self.readout(g, node_feat)
        out = self.out(out)
        return out
