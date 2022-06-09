# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:38:29 2022

@author: ZNDX002
"""


from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
import torch.nn.functional as F
from torch import nn 

class Graphccs(nn.Sequential):
	def __init__(self,in_feats, hidden_feats=None, activation=None, predictor_dim=None):
		super(Graphccs, self).__init__()
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
